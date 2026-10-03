"""In-process transport: an OpenAI-compatible chat endpoint driving AssemblyEnv.

Tool calls are text, not the API's `tools` field: the model writes one or more
<tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks. Servers differ
in whether they run a tool parser at all (the shared vLLM servers here do not),
and a text protocol makes every local model face byte-identical instructions.
Parsing is `ppbench.v2.toolcalls` (harness v2.3, 2026-09-14): tag blocks are read
exactly as v1's recovery read them; only when a reply yields no call are arithmetic
inside arguments evaluated, untagged JSON calls of real tools accepted, and the end
of the reasoning searched. A reply with no executable call gets a message saying
what failed instead of a generic nudge.

Images: the condition image and pool sheets go in the first user message for
vision models; render results are attached to the tool-result message. Only the
latest render stays in context; older tool results are cut to a stub after
KEEP_RECENT, the same policy as v1, and the number of stubs is recorded.

    .venv_eval/bin/python -m ppbench.v2.loop --model qwen3-vl-8b --base-url http://127.0.0.1:8081/v1 \
        --task swivel_office_chair --tier A --condition name_only+image --out results/v2/.../runs/<run>
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

from openai import OpenAI

from ppbench.agents.backends import strip_tool_blocks
from ppbench.v2.env import TOOL_CREATE, TOOLS, AssemblyEnv, system_prompt, tool_docs
from ppbench.v2.toolcalls import feedback as parse_feedback, note as parse_note, parse as parse_calls
from ppbench.v2.session import save_round_checkpoint, user_prompt
from ppbench.v2.task import RESULTS, Task

KEEP_RECENT = 10
STUB = 200
MAX_STEPS_PER_ROUND = 70
NUDGE_LIMIT = 3
HARNESS = "v2.3"          # v2.2 runs used ppbench.agents.backends.recover_tool_calls and a generic nudge
TOOL_NAMES = {t[0] for t in TOOLS} | {TOOL_CREATE[0]}
FORMAT_NOTES_MAX = 3

PROTOCOL = """
HOW TO CALL TOOLS
Write each tool call as a block on its own:
<tool_call>{"name": "place_part", "arguments": {"part_id": "P001", "instance_id": "wheel_1", "position": [0, 0, 0.03]}}</tool_call>
You may write several blocks in one reply; they run in order and you then see every result.
Arguments are strict JSON (double quotes, no comments, no trailing commas)."""


def _img(path, max_side=1024):
    from io import BytesIO
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def _compact(messages):
    n = 0
    idx = [i for i, m in enumerate(messages) if m.get("_tool")]
    for i in idx[:-KEEP_RECENT]:
        m = messages[i]
        if isinstance(m["content"], list):
            txt = " ".join(c.get("text", "") for c in m["content"] if c.get("type") == "text")
            m["content"] = txt
        if len(m["content"]) > STUB and not m["content"].startswith("[elided]"):
            m["content"] = "[elided] " + m["content"][:STUB] + " ..."
            n += 1
    # At most two images in context, for every model alike: the condition image, and the newest
    # other picture (the catalog sheet at first, then the latest tool result that returned one).
    img_msgs = [i for i, m in enumerate(messages) if m.get("_tool") and isinstance(m["content"], list)]
    for i in img_msgs[:-1]:
        messages[i]["content"] = " ".join(c.get("text", "") for c in messages[i]["content"] if c.get("type") == "text")
    if img_msgs:
        first = next(m for m in messages if m["role"] == "user")
        if isinstance(first["content"], list):
            first["content"] = [c for c in first["content"] if c.get("_kind") != "sheet"]
    return n


def _wire(messages):
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [{k: v for k, v in item.items() if not k.startswith("_")} for item in c]
        out.append({"role": m["role"], "content": c})
    return out


def run(model, base_url, task_id, tier, condition, out, rounds=2, vision=True, max_tokens=4096, temperature=0.6,
        seed=0, extra_body=None, api_key="local", system_name=None, timeout=900, snapshot=None):
    task = Task(task_id, snapshot=Path(snapshot) if snapshot else RESULTS / task_id / "task_snapshot.json")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = AssemblyEnv(task, tier, condition, rounds, workdir=out)
    cond = task.condition(condition)
    from ppbench.v2.render import sheet_all
    sheets = [] if str(tier).upper() == "C" else [s for s in [sheet_all(task_id)] if s]
    sys_text = system_prompt(tier) + "\n\nTOOLS\n" + tool_docs(tier) + "\n" + PROTOCOL
    first = [{"type": "text", "text": user_prompt(task, cond, tier, sheets, vision=vision, image_paths=False)}]
    if vision:
        if cond.get("image"):
            first.append(_img(cond["image"]["path"]))
        first += [{**_img(s, 1600), "_kind": "sheet"} for s in sheets]
    messages = [{"role": "system", "content": sys_text},
                {"role": "user", "content": first if vision else first[0]["text"]}]
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    log = {"model": model, "base_url": base_url, "vision": vision, "temperature": temperature, "seed": seed,
           "max_tokens": max_tokens, "extra_body": extra_body, "harness": HARNESS, "steps": [], "stubs": 0,
           "usage": {"prompt": 0, "completion": 0}, "format_notes": 0}
    steps_this_round, nudges, rnd = 0, 0, 1
    t0 = time.time()
    while not env.s["done"]:
        if env.s["round"] != rnd:
            rnd, steps_this_round, nudges = env.s["round"], 0, 0
        if steps_this_round >= MAX_STEPS_PER_ROUND:
            log["stop"] = f"step limit in round {rnd}"
            break
        steps_this_round += 1
        log["stubs"] += _compact(messages)
        wire = _wire(messages)
        try:
            r = client.chat.completions.create(model=model, messages=wire, max_tokens=max_tokens, temperature=temperature,
                                               seed=seed, extra_body=extra_body or {})
        except Exception as e:
            msg = str(e)
            log["steps"].append({"error": msg[:500]})
            if "maximum context" in msg or "context length" in msg or "too long" in msg:
                log["stop"] = "context overflow"
                break
            time.sleep(10)
            if sum(1 for s in log["steps"][-5:] if "error" in s) >= 5:
                log["stop"] = "repeated API errors"
                break
            continue
        text = r.choices[0].message.content or ""
        if r.usage:
            log["usage"]["prompt"] += r.usage.prompt_tokens or 0
            log["usage"]["completion"] += r.usage.completion_tokens or 0
        msg = r.choices[0].message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
        parsed = parse_calls(text, TOOL_NAMES, reasoning)
        calls = parsed["calls"]
        messages.append({"role": "assistant", "content": text})
        step = {"round": rnd, "text": strip_tool_blocks(text)[:1500], "calls": [], "finish": r.choices[0].finish_reason,
                "parse": {"source": parsed["source"], "repairs": parsed["repairs"], "errors": parsed["errors"][:3]}}
        if not calls:
            nudges += 1
            step["nudge"] = nudges
            log["steps"].append(step)
            if nudges > NUDGE_LIMIT:
                log["stop"] = "no tool calls after nudges"
                break
            messages.append({"role": "user", "content": parse_feedback(parsed, r.choices[0].finish_reason), "_tool": True})
            continue
        nudges = 0
        parts, images = [], []
        for c in calls:
            res = env.call(c["name"], c["arguments"])
            if c["name"] == "submit" and res.get("ok"):
                save_round_checkpoint(env, system_name or model, "openai-compatible text tool calls", res)
            step["calls"].append({"name": c["name"], "ok": res.get("ok"), "error": res.get("error")})
            imgs = list(res.get("images") or []) + list((res.get("review") or {}).get("images") or [])
            if imgs and vision:
                images += imgs
            parts.append(f"<tool_result name=\"{c['name']}\">{json.dumps(res)[:6000]}</tool_result>")
            if env.s["done"] or (c["name"] == "submit" and res.get("ok")):
                break
        log["steps"].append(step)
        fnote = parse_note(parsed)
        if fnote and log["format_notes"] < FORMAT_NOTES_MAX:
            parts.append(fnote)
            log["format_notes"] += 1
        content = "\n".join(parts)
        if images:
            # the two-image rule above counts pictures, not messages: one tool result may return several (e.g. a submit
            # review plus a render), which the servers reject ("At most 2 image(s)"); keep only the newest (2026-09-15)
            images = images[-1:]
            messages.append({"role": "user", "content": [{"type": "text", "text": content}] + [_img(p, 768) for p in images], "_tool": True})
        else:
            messages.append({"role": "user", "content": content, "_tool": True})
        (out / "loop_log.json").write_text(json.dumps(log, indent=0))
    log["wall_s"] = round(time.time() - t0, 1)
    d = env.design(system_name or model, {"transport": "openai-compatible text tool calls", "loop": {k: v for k, v in log.items() if k != "steps"}})
    d.save(out / "design")
    (out / "trace.json").write_text(json.dumps(env.s["trace"], indent=0))
    (out / "state.json").write_text(json.dumps(env.to_state()))
    (out / "loop_log.json").write_text(json.dumps(log, indent=0))
    (out / "transcript.json").write_text(json.dumps(
        [{k: (v if not isinstance(v, list) else [c if c.get("type") == "text" else {"type": "image"} for c in v]) for k, v in m.items()}
         for m in messages], indent=0))
    return d, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system-name")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--task", default="swivel_office_chair")
    ap.add_argument("--tier", choices=["A", "B", "C"], required=True)
    ap.add_argument("--condition", default="name_only+image")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--snapshot", help="task snapshot file (default results/v2/<task>/task_snapshot.json)")
    ns = ap.parse_args()
    d, log = run(ns.model, ns.base_url, ns.task, ns.tier, ns.condition, ns.out, ns.rounds, not ns.no_vision,
                 ns.max_tokens, ns.temperature, ns.seed, json.loads(ns.extra_body), system_name=ns.system_name,
                 snapshot=ns.snapshot)
    print(json.dumps({"parts": len(d.parts), "joints": len(d.joints), "stop": log.get("stop"), "wall_s": log["wall_s"],
                      "usage": log["usage"]}))


if __name__ == "__main__":
    main()

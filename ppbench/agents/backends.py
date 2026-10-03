"""Model backends.

One interface, two deployments, and deliberately only one protocol: the OpenAI
chat-completions shape with tool calling. OpenRouter speaks it, and so does a
local vLLM server, so a small model served on a spare GPU and Claude Fable 5.1
behind OpenRouter run through byte-identical harness code. That is what makes
the local model a real smoke test for the remote ones rather than a separate
code path that happens to work.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

# Hermes-style tool calls, as Qwen and several other open models emit them.
# Needed on the *client* side as well as the server side: vLLM's parser drops a
# call whose block it cannot parse cleanly, and the raw text then arrives as
# ordinary content. Scoring that as "the model made no tool call" would blame
# the model for a transport failure, which on the parametric tier is the
# difference between "cannot design" and "was not heard".
_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)
_DECODER = json.JSONDecoder()


def recover_tool_calls(text: str):
    """Tool calls embedded in assistant text. Returns [] when there are none.

    The block's contents are read with `raw_decode`, which parses the first
    complete JSON value and reports where it ended, rather than requiring the
    whole block to be valid JSON. That matters because models miscount closing
    braces: the call that first exposed this arrived as a well-formed object
    followed by one surplus `}`, and a strict parse rejected the entire thing.
    Trailing junk after a complete object is a tokenising slip, so it is
    dropped and the repair is counted. A block that is *truncated* before the
    object closes is not repaired -- completing it would be inventing the
    model's intent rather than recovering it.
    """
    out = []
    for i, m in enumerate(_TOOL_BLOCK.finditer(text or "")):
        blob = (m.group(1) or "").strip()
        start = blob.find("{")
        if start < 0:
            continue
        try:
            obj, _end = _DECODER.raw_decode(blob[start:])
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", obj.get("parameters", {})) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({"id": f"recovered_{i}", "name": name, "arguments": args})
    return out


def strip_tool_blocks(text: str) -> str:
    return _TOOL_BLOCK.sub("", text or "").strip()


@dataclass
class Reply:
    text: str = ""
    tool_calls: list = field(default_factory=list)   # [{id, name, arguments(dict)}]
    raw: dict | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None


class Backend:
    name = "abstract"

    def chat(self, messages, tools, temperature=0.6, max_tokens=2048) -> Reply:
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 name: str | None = None, extra_headers: dict | None = None,
                 supports_tools: bool = True, supports_vision: bool | None = None,
                 timeout: float = 300.0):
        from openai import OpenAI
        self.model = model
        self.name = name or model
        self.supports_tools = supports_tools
        # Whether review images are delivered. Hosted Claude, GPT and Gemini
        # models accept image content; a local text-only model does not, and
        # sending it images would fail the request.
        self.supports_vision = (supports_vision if supports_vision is not None else
                                any(k in model.lower() for k in ("claude", "gpt", "gemini", "vl", "vision")))
        self._extra = extra_headers or {}
        self._client = OpenAI(base_url=base_url,
                              api_key=api_key or "not-needed",
                              timeout=timeout, max_retries=2)

    def chat(self, messages, tools, temperature=0.6, max_tokens=2048) -> Reply:
        kw = dict(model=self.model, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
        if tools and self.supports_tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        if self._extra:
            kw["extra_headers"] = self._extra
        try:
            r = self._client.chat.completions.create(**kw)
        except Exception as exc:                              # noqa: BLE001
            return Reply(error=f"{type(exc).__name__}: {exc}")

        ch = r.choices[0].message
        calls = []
        for tc in (ch.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__unparsed__": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        usage = {}
        if getattr(r, "usage", None):
            usage = {"prompt_tokens": r.usage.prompt_tokens,
                     "completion_tokens": r.usage.completion_tokens}
        text = ch.content or ""
        recovered = 0
        if not calls:
            calls = recover_tool_calls(text)
            if calls:
                recovered = len(calls)
                text = strip_tool_blocks(text)
        return Reply(text=text, tool_calls=calls, usage=usage,
                     raw={"recovered_tool_calls": recovered} if recovered else None)


def openrouter(model: str, api_key: str | None = None, **kw) -> OpenAICompatBackend:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "no OpenRouter key: set OPENROUTER_API_KEY or pass api_key=")
    return OpenAICompatBackend(
        model=model, base_url="https://openrouter.ai/api/v1", api_key=key,
        name=model,
        extra_headers={"HTTP-Referer": "https://github.com/ppbench",
                       "X-Title": "LMBuild"}, **kw)


def local_vllm(model: str, port: int = 8077, **kw) -> OpenAICompatBackend:
    return OpenAICompatBackend(model=model, base_url=f"http://127.0.0.1:{port}/v1",
                              api_key="local", name=f"local/{model}", **kw)


def backend_for(spec: str, **kw) -> Backend:
    """Resolve a model spec to a backend.

      local:<model>[@port]   an OpenAI-compatible server on this machine (vLLM)
      hf:<path>[@device]     a local checkpoint in-process, no server
      anything else          OpenRouter, needs OPENROUTER_API_KEY
    """
    if spec.startswith("local:"):
        model, _, port = spec[len("local:"):].partition("@")
        return local_vllm(model, int(port or 8077), **kw)
    if spec.startswith("hf:"):
        path, _, dev = spec[len("hf:"):].partition("@")
        return TransformersBackend(path, device=dev or "cuda:0", **kw)
    return openrouter(spec, **kw)


def wait_for_local(port: int = 8077, timeout: float = 900.0) -> tuple[bool, str]:
    """Poll a local server until its model list answers."""
    import urllib.error
    import urllib.request
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                d = json.load(r)
                ids = [m["id"] for m in d.get("data", [])]
                return True, ",".join(ids)
        except Exception as exc:                              # noqa: BLE001
            last = f"{type(exc).__name__}"
            time.sleep(5)
    return False, f"timeout after {timeout:.0f}s ({last})"


class TransformersBackend(Backend):
    """A local model in-process, with no server.

    Insurance, and useful on its own: vLLM on this machine needed a short
    TMPDIR, eager mode, and a spare GPU before it would start, and a harness
    that cannot run without it is a harness that cannot run.

    Tool calls are parsed out of the text. Qwen's chat template emits Hermes
    style `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, and the
    template renders the tool list itself when `tools=` is passed to
    `apply_chat_template`, so the prompt a local model sees matches what a
    server-side parser would have produced.
    """

    name = "transformers"

    def __init__(self, model_path: str, device: str = "cuda:0",
                 dtype: str = "bfloat16", name: str | None = None,
                 max_input_tokens: int = 14000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name or f"transformers/{model_path.rstrip('/').split('/')[-1]}"
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        self.device = device
        self.max_input_tokens = max_input_tokens
        self._torch = torch

    def _openai_to_template(self, messages):
        """The chat template wants tool results as role=tool with a name."""
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = [{"type": "function", "function": {
                    "name": c["function"]["name"],
                    "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                    for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": m.get("content") or "",
                            "tool_calls": calls})
            else:
                out.append({k: v for k, v in m.items() if k != "tool_call_id"})
        return out

    def chat(self, messages, tools, temperature=0.6, max_tokens=2048) -> Reply:
        try:
            text = self.tok.apply_chat_template(
                self._openai_to_template(messages),
                tools=[t["function"] for t in tools] if tools else None,
                add_generation_prompt=True, tokenize=False)
            ids = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=self.max_input_tokens).to(self.device)
            with self._torch.inference_mode():
                out = self.model.generate(
                    **ids, max_new_tokens=max_tokens,
                    do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                    top_p=0.95, pad_token_id=self.tok.eos_token_id)
            gen = self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        except Exception as exc:                              # noqa: BLE001
            return Reply(error=f"{type(exc).__name__}: {exc}")

        calls = recover_tool_calls(gen)
        clean = strip_tool_blocks(gen)
        return Reply(text=clean, tool_calls=calls,
                     usage={"prompt_tokens": int(ids["input_ids"].shape[1]),
                            "completion_tokens": int(out.shape[1] - ids["input_ids"].shape[1])})

"""Tool-call recovery for the v2 text transport, strict where it was strict and lenient about wrappers.

The protocol asks for <tool_call>{"name": ..., "arguments": {...}}</tool_call>. In the office-chair demo runs
(2026-09-14) 42 open-model episodes ended on "no tool calls after nudges" although every one of them had written
calls: bare or fenced JSON without the tag (InternVL3.5, MiniCPM-V), arithmetic inside arguments such as
`0.1165 + 0.05455` (Qwen family, MiniCPM-V), or gpt-oss harmony markup leaking into the text. The old nudge
("No tool call was found") did not say what was wrong, so models resent the same call until the episode stopped.

Recovery order; a later stage runs only when the earlier ones found no call, so a reply the old parser read is
read identically:
  1. <tool_call> blocks, decoded as before (first complete JSON value; trailing junk dropped). A block that does
     not decode gets one repair: arithmetic between numbers outside strings is evaluated. Anything else
     (unbalanced braces, truncation, comments) is reported, not guessed.
  1b. <name>tool</name><arguments>{...}</arguments> pairs (ERNIE-4.5-VL, 2026-09-20): both fields are explicit,
     so each name is paired with the arguments block that follows it; the name must be a real tool and the
     arguments must decode to an object.
  2. JSON objects anywhere in the visible reply (bare, fenced, after a marker) whose tool name is a real tool;
     OpenAI-style {"function": {...}} and {"tool": ..., "args": ...} shapes are read too; then name(key=value)
     calls of real tools with literal arguments.
  3. Only if the visible reply is empty or has no call: the end of the model's reasoning (<think> blocks or the
     server's reasoning field), with the stage-2 rule.

`feedback` turns a failed parse into a specific message (what failed, where, and the expected form); `note`
is appended to tool results when a call was accepted through a repair or a non-tag form.

    .venv_eval/bin/python -m ppbench.v2.toolcalls      # self-test on snippets from the demo transcripts
"""
from __future__ import annotations

import ast
import json
import re

DEC = json.JSONDecoder()
TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)
XML_NAME = re.compile(r"<(name|tool_name|tool|function)>\s*\"?([A-Za-z_][A-Za-z_0-9.]*)\"?\s*(?:</\1>)?", re.S)
XML_TAIL = re.compile(r"^\s*,\s*\"arguments\"\s*:\s*(\{.*)$", re.S)
XML_ARGS = re.compile(r"<arguments>\s*(.*?)\s*(?:</arguments>|\Z)", re.S)
XML_PAIR = re.compile(r"<name>\s*(.*?)\s*</name>\s*<value>\s*(.*?)\s*</value>", re.S)
XML_ATTR = re.compile(r"<argument\s+name=\"([^\"]+)\"\s*>\s*(.*?)\s*(?:</argument>|\Z)", re.S)
XML_ATTR1 = re.compile(r"<argument\s+name=\"([^\"]+)\"\s+value=\"([^\"]*)\"\s*/?>", re.S)
XML_CDATA = re.compile(r"^<!\[CDATA\[(.*?)\]\]>", re.S)
XML_EMPTY = re.compile(r"<[^>]*/>|</?[A-Za-z_][^>]*>|\s+")
THINK = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S)
STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.S)
NUMRUN = re.compile(r"[0-9.eE+\-*/()\s]+")
REASONING_TAIL = 4000
ARG_KEYS = ("arguments", "parameters", "args", "input", "action_input")
NAME_KEYS = ("name", "tool", "tool_name", "action", "function")


# ---------------------------------------------------------------- arithmetic repair

def _eval_arith(expr: str):
    """Value of a pure arithmetic expression over numeric literals, or None."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    has_binop = False
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            has_binop = True
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                return None
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                return None
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return None
        elif not isinstance(node, (ast.Expression, ast.operator, ast.unaryop)):
            return None
    if not has_binop:
        return None
    try:
        v = eval(compile(tree, "<arith>", "eval"), {"__builtins__": {}}, {})
    except (ZeroDivisionError, OverflowError):
        return None
    return float(round(v, 9))


def repair_arithmetic(blob: str) -> tuple[str, int]:
    """Replace arithmetic between numbers outside JSON strings with its value. Returns (text, n_replaced)."""
    out, n, last = [], 0, 0

    def fix(segment):
        nonlocal n
        parts, pos = [], 0
        for m in NUMRUN.finditer(segment):
            run = m.group(0)
            core = run.strip()
            if not re.search(r"\d", core) or not re.search(r"\d[\s)]*[-+*/][\s(]*[-+]?[\d.(]", core):
                continue
            v = _eval_arith(core)
            if v is None:
                continue
            lead = run[:len(run) - len(run.lstrip())]
            trail = run[len(run.rstrip()):]
            parts.append(segment[pos:m.start()] + lead + repr(v) + trail)
            pos = m.end()
            n += 1
        parts.append(segment[pos:])
        return "".join(parts)

    for m in STRING.finditer(blob):
        out.append(fix(blob[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(fix(blob[last:]))
    return "".join(out), n


# ---------------------------------------------------------------- truncation repair

def close_truncated(blob: str):
    """Close a JSON value that was cut off mid-structure, or None when nothing is open.

    Qwen3-VL-32B/8B end a long `create_part` one closer short: the reply stops after
    `…"rotation": [0, 0, 45]}], "output": "n4"}}` with the call object's own `}` missing, and every nudge
    returns the same bytes, so the episode dies on "no tool calls after nudges" with the design in hand
    (2026-09-16: 271 of the stored Tier B blobs failed raw_decode, 239 of them only for this).
    The walk tracks strings and escapes, drops a half-written tail after the last finished element, and
    shuts the still-open containers in reverse order."""
    def walk(text):
        stack, in_str, esc, last = [], False, False, None
        for i, c in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack:
                    stack.pop()
                last = i + 1
            elif c == ",":
                last = i
        return stack, in_str, last

    stack, in_str, last = walk(blob)
    if not stack or in_str:
        return None
    # Only add the missing closers. A blob whose tail is a half-written value is refused: closing
    # `"position": [0, 0, 0.3` into `[0, 0]` would invent a coordinate the model never wrote, which is
    # worse than reporting the parse failure and letting the nudge ask for the call again.
    if last is None or blob[last:].strip():
        return None
    return blob[:last].rstrip() + "".join("}" if c == "{" else "]" for c in reversed(stack))


def _decode_repaired(blob: str, stats):
    """raw_decode after the arithmetic repair, then after closing a truncated value. (obj, error) — obj is
    None when neither worked, and `stats["repairs"]` counts what each stage changed."""
    try:
        return DEC.raw_decode(blob)[0], None
    except ValueError as first:
        fixed, k = repair_arithmetic(blob)
        if k:
            try:
                obj = DEC.raw_decode(fixed)[0]
                stats["repairs"] += k
                return obj, None
            except ValueError:
                pass
        else:
            fixed = blob
        for candidate in (close_truncated(fixed), close_truncated(blob)):
            if candidate is None:
                continue
            try:
                obj = DEC.raw_decode(candidate)[0]
                stats["repairs"] += 1
                return obj, None
            except ValueError:
                continue
        return None, first


# ---------------------------------------------------------------- shapes

def _as_call(obj, tool_names=None):
    """(name, arguments) if obj is a tool call of a known shape (and, when tool_names is given, a real tool)."""
    if not isinstance(obj, dict):
        return None
    name, args = None, None
    fn = obj.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        name, args = fn.get("name"), fn.get("arguments", fn.get("parameters"))
    else:
        for k in NAME_KEYS:
            if isinstance(obj.get(k), str) and obj.get(k):
                name = obj[k]
                break
        for k in ARG_KEYS:
            if k in obj:
                args = obj[k]
                break
    if not name or (tool_names is not None and name not in tool_names):
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if args is None:
        extra = {k: v for k, v in obj.items() if k not in NAME_KEYS and k not in ("type", "id")}
        args = extra if tool_names is not None else {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _balanced_end(text: str, start: int):
    """Index just past the object that opens at text[start] ('{'), skipping strings; None if it never closes."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            m = STRING.match(text, i)
            if not m:
                return None
            i = m.end()
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _scan(text: str, tool_names, stats):
    """Stage 2: JSON objects of real tools anywhere in text, then name(key=value) calls."""
    calls, i = [], 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        end = _balanced_end(text, j)
        if end is None:
            break
        seg, obj = text[j:end], None
        try:
            obj = json.loads(seg)
        except json.JSONDecodeError:
            fixed, k = repair_arithmetic(seg)
            if k:
                try:
                    obj = json.loads(fixed)
                    stats["repairs"] += k
                except json.JSONDecodeError:
                    obj = None
        c = _as_call(obj, tool_names)
        if c:
            calls.append(c)
            i = end
        else:
            i = j + 1
    if calls:
        return calls, "json"
    names = "|".join(sorted(map(re.escape, tool_names), key=len, reverse=True))
    for m in re.finditer(rf"(?<![\w.])({names})\((.*?)\)\s*$", text, re.M):
        try:
            node = ast.parse(f"f({m.group(2)})", mode="eval").body
            if node.args:
                continue
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except (SyntaxError, ValueError):
            continue
        calls.append((m.group(1), args))
    return calls, "python" if calls else None




def _call_end(text: str, pos: int, stop: int) -> int:
    """Where the call that starts at pos ends: its </tool_call>, else stop (the next call's name).

    Only the close tag ends a call. An opening <tool_call> does not: ERNIE wraps a call in a second <tool_call>
    and sometimes opens it between the name and the arguments, and a call that follows is already bounded by its
    own name tag.
    """
    i = text.find("</tool_call>", pos, stop)
    return i if i >= 0 else stop


def _inside_arguments(text: str, pos: int) -> bool:
    """True when pos sits inside an <arguments> block: a key name, not a tool name.

    ERNIE opens a fresh <arguments> for every extra argument and often never closes them, so the block ends at
    the call as well - without that, one unclosed block swallows every call that follows it.
    """
    o = text.rfind("<arguments>", 0, pos)
    if o < 0:
        return False
    return max(text.rfind("</arguments>", o, pos), text.rfind("</tool_call>", o, pos),
               text.rfind("<tool_call>", o, pos)) < 0


def _xml_value(raw: str):
    """One <value> or <argument> body: JSON when it decodes, else the text (CDATA unwrapped)."""
    raw = raw.strip()
    m = XML_CDATA.match(raw)
    if m:
        raw = m.group(1).strip()
    if raw == "":
        return ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _clean_args(obj):
    """An argument dict, or None when a key carries markup - the sign that a block ran past its own call."""
    if not isinstance(obj, dict):
        return None
    if any(not isinstance(k, str) or "<" in k or ">" in k for k in obj):
        return None
    return obj


def _xml_args(region: str, stats):
    """The arguments of one XML-shaped call, or None when the region is not one set of arguments.

    Four forms are read, all explicit: a JSON object (optionally wrapped in CDATA); <name>k</name><value>v</value>
    pairs, including ERNIE's habit of opening a fresh <arguments> per pair; <argument name="k">v</argument> and
    its self-closing value="..." form; and a block with no value in it at all, which is a tool that takes none.
    A list of argument sets, prose, or an object that will not decode is reported rather than guessed at.
    """
    blob = region.strip()
    while blob.startswith("<arguments>"):      # ERNIE doubles the wrapper: <arguments><arguments>{...}
        blob = blob[len("<arguments>"):].strip()
    while blob.endswith("</arguments>"):
        blob = blob[:-len("</arguments>")].strip()
    m = XML_CDATA.match(blob)
    if m:
        blob = m.group(1).strip()
    if blob.startswith("{"):
        obj, _err = _decode_repaired(blob, stats)
        return _clean_args(obj)
    pairs = XML_PAIR.findall(blob)
    if pairs:
        return _clean_args({k.strip().strip('"'): _xml_value(v) for k, v in pairs})
    attrs = XML_ATTR.findall(blob) + XML_ATTR1.findall(blob)
    if attrs:
        return _clean_args({k: _xml_value(v) for k, v in attrs})
    if not XML_EMPTY.sub("", blob):
        return {}
    return None


def _scan_xml(text: str, tool_names, stats):
    """Stage 1b: <name>tool</name><arguments>{...}</arguments>, the shape ERNIE-4.5-VL writes.

    ERNIE answers the protocol with the two fields as tags instead of as keys of one JSON object, often twenty
    pairs in one reply, and it varies the spelling: <name>, <tool_name> or <tool> for the name; the arguments as
    a JSON object, as <name>/<value> pairs, as <argument name="k">v</argument>, wrapped in CDATA, or empty.
    Stage 1 lands on the first `{` of such a block, which is the *arguments* object, and reports 'no "name"';
    the nudge then says the name is missing, which the model cannot act on because it believes it wrote one.
    Every field here is explicit, so nothing is invented: each name is paired with the arguments that follow it
    and before the next call, the name must be a real tool, and the arguments must resolve to one object whose
    keys carry no markup. Anything else is reported, so the model is told what failed.
    """
    calls, errors = [], []
    names = [m for m in XML_NAME.finditer(text) if not _inside_arguments(text, m.start())]
    for i, m in enumerate(names):
        tag, name = m.group(1), m.group(2)
        stop = names[i + 1].start() if i + 1 < len(names) else len(text)
        stop = _call_end(text, m.end(), stop)
        region = text[m.end():stop]
        a = XML_ARGS.search(region)
        if a:
            obj = _xml_args(region[a.start(1):], stats)
        elif XML_ATTR.search(region) or XML_ATTR1.search(region):
            obj = _xml_args(region, stats)
        elif XML_TAIL.match(region):
            # <name>"list_parts", "arguments": {...}: the model dropped the '{"name": ' and wrote the rest
            obj, _err = _decode_repaired(XML_TAIL.match(region).group(1), stats)
            obj = _clean_args(obj)
        elif text.startswith("</tool_call>", stop) and not region.strip():
            # <tool_call><name>get_scene</name></tool_call>: the block is closed with nothing in it, which is a
            # tool that takes no arguments. A block that simply ran out of text is not read this way.
            obj = {}
        else:
            obj = None
        if obj is None:
            errors.append(f"the <arguments> of <{tag}>{name}</{tag}> is not one JSON object")
        elif name in tool_names:
            calls.append((name, obj))
        else:
            errors.append(f"<{tag}>{name}</{tag}> is not a tool")
    return calls, errors


# ---------------------------------------------------------------- entry points

def parse(text: str, tool_names, reasoning: str | None = None) -> dict:
    """{"calls": [{"id", "name", "arguments"}], "source": tag|json|python|reasoning-json|reasoning-python|None,
    "repairs": n, "errors": [str], "empty": bool}"""
    text = text or ""
    tool_names = set(tool_names)
    stats = {"repairs": 0}
    think = "\n".join(THINK.findall(text))
    visible = THINK.sub("", text)
    errors, calls = [], []
    for i, m in enumerate(TAG.finditer(visible)):
        blob = (m.group(1) or "").strip()
        start = blob.find("{")
        if start < 0:
            errors.append("a <tool_call> block holds no JSON object")
            continue
        obj, err = _decode_repaired(blob[start:], stats)
        if obj is None:
            pos = getattr(err, "pos", 0)
            snippet = blob[start:][max(0, pos - 40):pos + 20].replace("\n", " ")
            errors.append(f"{getattr(err, 'msg', str(err))} at character {pos} (near …{snippet}…)")
            continue
        c = _as_call(obj)          # stage 1 keeps the old rule: any name; the environment reports unknown tools
        if c:
            calls.append(c)
        else:
            errors.append("a <tool_call> block has no \"name\"")
    source = "tag" if calls else None
    if not calls:
        xml_calls, xml_errors = _scan_xml(visible, tool_names, stats)
        if xml_calls:
            calls, source = xml_calls, "xml"
        else:
            errors += xml_errors
    if not calls:
        calls, source = _scan(visible, tool_names, stats)
    if not calls:
        tail = ((think or "") + "\n" + (reasoning or ""))[-REASONING_TAIL:]
        if tail.strip():
            for m in TAG.finditer(tail):
                blob = (m.group(1) or "").strip()
                st = blob.find("{")
                if st >= 0:
                    try:
                        c = _as_call(DEC.raw_decode(blob[st:])[0], tool_names)
                        if c:
                            calls.append(c)
                    except ValueError:
                        pass
            if calls:
                source = "reasoning-tag"
            else:
                calls, src = _scan(tail, tool_names, stats)
                source = f"reasoning-{src}" if calls else None
    return {"calls": [{"id": f"recovered_{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)],
            "source": source, "repairs": stats["repairs"], "errors": errors if not calls else errors[:0] or errors,
            "empty": not visible.strip()}


EXAMPLE = '<tool_call>{"name": "get_scene", "arguments": {}}</tool_call>'


def feedback(res: dict, finish: str | None = None) -> str:
    """The user message sent back when a reply produced no executable call."""
    if res["errors"]:
        msg = ("Your tool call could not be parsed, so nothing ran: " + "; ".join(res["errors"][:2]) + ". "
               "Arguments must be plain JSON: numbers as literal values (write 0.17105, not 0.1165 + 0.05455), "
               "double-quoted strings, no comments, balanced braces. Resend the corrected call as "
               "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>.")
    elif res["empty"]:
        msg = ("Your reply had no visible content, so nothing ran. Reply with a tool call, for example " + EXAMPLE + ".")
    else:
        msg = ("No tool call was found in your reply, so nothing ran. There is no human in this loop to answer questions: "
               "continue the task by calling tools, for example " + EXAMPLE + ". When the design is finished, call submit.")
    if finish == "length":
        msg += " Your reply hit the output length limit; send fewer or shorter calls per reply."
    return msg


def note(res: dict) -> str | None:
    """A one-line format note for tool results when a call was accepted through a repair or a non-tag form."""
    how = []
    if res["source"] and res["source"] != "tag":
        how.append({"json": "as JSON without a <tool_call> tag", "python": "as name(...)",
                    "reasoning-json": "inside your reasoning", "reasoning-python": "inside your reasoning",
                    "reasoning-tag": "inside your reasoning"}.get(res["source"], res["source"]))
    if res["repairs"]:
        how.append("with arithmetic inside the arguments, which was evaluated")
    if not how:
        return None
    return ("(format note: your call was written " + " and ".join(how) + "; it was accepted this time. "
            "Preferred form: <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> with plain numbers.)")


# ---------------------------------------------------------------- self-test

def _selftest():
    tools = {"place_part", "add_joint", "submit", "list_parts", "inspect_part", "get_scene", "set_assembly_sequence",
             "create_part", "list_materials", "set_part_info", "move_part", "remove_joint", "check", "render"}
    cases = [
        # unchanged: a well-formed tag call, and one surplus brace
        ('<tool_call>{"name": "list_parts", "arguments": {}}</tool_call>', "tag", [("list_parts", {})], 0),
        ('<tool_call>{"name": "submit", "arguments": {}}}</tool_call>', "tag", [("submit", {})], 0),
        # Qwen3-VL-8B / Qwen3-4B: arithmetic inside a tag call
        ('<tool_call>{"name": "place_part", "arguments": {"part_id": "P035", "instance_id": "seat", "position": [0, 0, 0.1161 / 2], '
         '"rotation": [0, 0, 0], "role": "seat", "material": "cotton_fabric"}}\n</tool_call>', "tag",
         [("place_part", {"part_id": "P035", "instance_id": "seat", "position": [0, 0, 0.05805], "rotation": [0, 0, 0],
                          "role": "seat", "material": "cotton_fabric"})], 1),
        ('<tool_call>{"name": "place_part", "arguments": {"part_id": "P035", "position": [0, 0, 0.1165 + 0.05455]}}</tool_call>', "tag",
         [("place_part", {"part_id": "P035", "position": [0, 0, 0.17105]})], 1),
        # strings with operators are not touched
        ('<tool_call>{"name": "inspect_part", "arguments": {"part_id": "P1 + P2", "note": 1 - 0.5}}</tool_call>', "tag",
         [("inspect_part", {"part_id": "P1 + P2", "note": 0.5})], 1),
        # InternVL3.5-38B: bare JSON calls between prose, several per reply
        ('I created a joint.\n\n{"name": "add_joint", "arguments": {"joint_id": "w3", "type": "revolute", "parent": "c3", "child": "w3", '
         '"axis": [0, 1, 0], "origin": [-0.2, -0.2, 0.1512], "limits": [-3.14, 3.14]}}\n\nNow the next.\n\n'
         '{"name": "add_joint", "arguments": {"joint_id": "w4", "type": "revolute", "parent": "c4", "child": "w4"}}', "json",
         [("add_joint", {"joint_id": "w3", "type": "revolute", "parent": "c3", "child": "w3", "axis": [0, 1, 0],
                         "origin": [-0.2, -0.2, 0.1512], "limits": [-3.14, 3.14]}),
          ("add_joint", {"joint_id": "w4", "type": "revolute", "parent": "c4", "child": "w4"})], 0),
        # MiniCPM-V: fenced JSON after a think block
        ('<think>\nlist first\n</think>\n```json\n{\n  "name": "list_parts",\n  "arguments": {}\n}\n```', "json", [("list_parts", {})], 0),
        # InternVL3.5-8B: name prefix glued to the object
        ('place_part{"name": "place_part", "arguments": {"part_id": "P22", "instance_id": "s", "position": [0, 0, 0.075]}}', "json",
         [("place_part", {"part_id": "P22", "instance_id": "s", "position": [0, 0, 0.075]})], 0),
        # gpt-oss: harmony marker instead of a tag
        ('<tool_call<|message|>{"name":"inspect_part","arguments":{"part_id":"P024"}}', "json", [("inspect_part", {"part_id": "P024"})], 0),
        # Qwen3-4B: python-style submit()
        ("The office chair is now fully built.\n\nsubmit()", "python", [("submit", {})], 0),
        # objects that are not calls of real tools are ignored
        ('The steps are {"part": "base", "direction": [0, 0, 1]} and {"name": "seat"}.', None, [], 0),
        # truncated tag call: reported, not completed
        ('<tool_call>{"name": "place_part", "arguments": {"part_id": "P035", "position": [0, 0,', None, [], 0),
        # ERNIE-4.5-VL: name and arguments as XML tags, several pairs per reply, doubled and nested wrappers
        ('<tool_call>\n<tool_call>\n<name>"list_materials"</name>\n<arguments>\n{}\n</arguments>\n</tool_call>\n'
         '<tool_call>\n<name>"list_parts"</name>\n<arguments>\n{"query": "wheel"}\n</arguments>\n</tool_call>\n</tool_call>',
         "xml", [("list_materials", {}), ("list_parts", {"query": "wheel"})], 0),
        ('<tool_calls>\n<tool_call>\n<name>"list_parts"</name>\n<call>\n<arguments>{"query": "tower"}</arguments>\n</call>\n'
         '</tool_call>\n<tool_call>\n<name>"place_part"</name>\n<call>\n<arguments>{"part_id": "P1", "instance_id": "b", '
         '"position": [0, 0, 0]}</arguments>\n</call>\n</tool_call>', "xml",
         [("list_parts", {"query": "tower"}), ("place_part", {"part_id": "P1", "instance_id": "b", "position": [0, 0, 0]})], 0),
        # a well-formed tag call in the same reply still wins, and the XML pairs beside it are not read twice
        ('<tool_call>{"name": "get_scene", "arguments": {}}</tool_call>\n<name>"list_parts"</name><arguments>{}</arguments>',
         "tag", [("get_scene", {})], 0),
        # ERNIE again: the name tag is also spelled <tool_name> or <tool>, and the arguments come as
        # <name>/<value> pairs, as <argument> tags, wrapped in CDATA, or empty
        ('<tool_call><tool_name>place_part</tool_name><arguments>{"part_id": "P007"}</arguments></tool_call>',
         "xml", [("place_part", {"part_id": "P007"})], 0),
        ('<tool_call><tool>list_parts</tool><arguments><![CDATA[{"query": "wheel"}]]></arguments></tool_call>',
         "xml", [("list_parts", {"query": "wheel"})], 0),
        ('<tool_call><name>list_parts</name><arguments><name>query</name><value>"wheel"</value></arguments></tool_call>',
         "xml", [("list_parts", {"query": "wheel"})], 0),
        ('<tool_call><name>"place_part"<argument name="part_id"> "base"</argument>'
         '<argument name="position" value="[0, 0, 0.1]"/></tool_call>', "xml",
         [("place_part", {"part_id": "base", "position": [0, 0, 0.1]})], 0),
        ('<tool_call><name>"list_parts", "arguments": {"query": ""}}</tool_call>', "xml", [("list_parts", {"query": ""})], 0),
        ('<tool_call><name>get_scene</name></tool_call>', "xml", [("get_scene", {})], 0),
        # a fresh <arguments> per argument, never closed: the call still ends at its own </tool_call>, so the
        # block cannot swallow the calls that follow it (bc_fire_engine, 2026-09-20)
        ('<tool_call><name>"move_part"</name><arguments><name>instance_id</name><value>"wheel_1"</value>'
         '<arguments><name>translate_by</name><value>[0, 0, 0.01]</value></arguments></tool_call>\n'
         '<tool_call><name>"remove_joint"</name><arguments><name>joint_id</name><value>"j1"</value></tool_call>',
         "xml", [("move_part", {"instance_id": "wheel_1", "translate_by": [0, 0, 0.01]}),
                 ("remove_joint", {"joint_id": "j1"})], 0),
        # doubled <arguments> wrapper, and a second <tool_call> opened between the name and the arguments
        ('<tool_call><name>"list_parts"</name><arguments><arguments>{"query": ""}</arguments></arguments></tool_call>',
         "xml", [("list_parts", {"query": ""})], 0),
        ('<tool_call><name>list_parts</name><tool_call><arguments>{"query": "wheel"}</arguments></tool_call></tool_call>',
         "xml", [("list_parts", {"query": "wheel"})], 0),
        # only key names, no values: nothing to recover, so it is refused
        ('<tool_call><name>list_parts</name><argument_names>query</argument_names></tool_call>', None, [], 0),
        # a block that simply ran out of text is not read as a zero-argument call
        ('<tool_call><name>place_part</name>', None, [], 0),
        # not guessed: arguments that are a list of argument sets, and a name that is not a tool
        ('<name>"set_part_info"</name>\n<arguments>\n[{"instance_id": "a", "role": "seat"}, {"instance_id": "b"}]\n</arguments>',
         None, [], 0),
        ('<name>"wheel_custom"</name>\n<arguments>\n{"radius": 0.05}\n</arguments>', None, [], 0),
    ]
    bad = 0
    for text, src, want, reps in cases:
        r = parse(text, tools)
        got = [(c["name"], c["arguments"]) for c in r["calls"]]
        ok = r["source"] == src and got == want and r["repairs"] == reps
        bad += not ok
        print(("ok  " if ok else "FAIL"), repr(text[:70]), "->", r["source"], got if not ok else "", r["repairs"], r["errors"][:1] if not ok else "")
    r = parse("", tools, reasoning='I will inspect it. {"name": "inspect_part", "arguments": {"part_id": "P024"}}')
    ok = r["source"] == "reasoning-json" and r["calls"][0]["name"] == "inspect_part"
    bad += not ok
    print("ok  " if ok else "FAIL", "empty reply, call at the end of reasoning ->", r["source"])
    r = parse('<tool_call>{"name": "place_part", "arguments": {"position": [0, 0, 1}}</tool_call>', tools)
    fb = feedback(r)
    ok = not r["calls"] and "could not be parsed" in fb
    bad += not ok
    print("ok  " if ok else "FAIL", "unbalanced call feedback:", fb[:110])
    print("SELFTEST", "PASSED" if not bad else f"FAILED ({bad})")
    return bad == 0


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)

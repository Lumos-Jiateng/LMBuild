"""Free-text part names -> canonical roles.

A design names its parts in its own words ("left castor 3", "gas_spring"), the
claims use theirs ("gas lift cylinder", "casters") and the reference uses its
dataset's labels ("seat support", "caster connector"). Every metric that
compares roles goes through one hand-written table per object type, small
enough to read in full. Matching is deterministic:

1. normalise: lower case, split on punctuation and camel case, drop digits and
   positional words (left, front, upper, ...), singularise plurals;
2. find every synonym phrase that occurs as a contiguous token run;
3. keep the longest; break ties by the phrase that ends last (the head noun of
   an English compound is its last word: "seat back" is a back, "back leg" a leg).

Names that match nothing are returned as None and reported as unrecognised,
never guessed. The table is a curation decision and is versioned with the code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

POSITIONAL = {
    # "top" and "bottom" are NOT here: they are head nouns as often as positions ("table top", "tabletop")
    "left", "right", "front", "rear", "upper", "lower", "center", "centre", "middle",
    "inner", "outer", "main", "primary", "secondary", "part", "piece", "component", "assembly", "unit",
    "a", "b", "c", "d", "e", "l", "r", "fl", "fr", "rl", "rr", "the", "of", "for", "and", "with",
}

# canonical role -> synonym phrases. Order inside a list does not matter.
TABLES = {
    "swivel_office_chair": {
        # no bare "cushion": "backrest cushion" and "armrest cushion" must not read as seats
        "seat": ["seat", "seat cushion", "seat pan", "seat pad", "seat shell", "sitting surface"],
        "backrest": ["backrest", "back rest", "back", "back support", "chair back", "back cushion", "back panel",
                     "lumbar support", "back frame"],
        "armrest": ["armrest", "arm rest", "armrest pad", "arm", "arm pad", "armrest support", "arm support",
                    "armrest post", "arm post", "armrest bracket"],
        "gas lift cylinder": ["gas lift cylinder", "gas lift", "gas cylinder", "gas column", "gas spring",
                              "pneumatic cylinder", "cylinder", "column", "seat support", "lift column",
                              "piston", "central column", "pole", "seat post", "lift", "height adjustment cylinder",
                              "height adjuster", "piston rod"],
        "base": ["base", "star base", "five star base", "chair base", "wheelbase", "wheel base", "leg base",
                 "spider base", "base leg", "leg", "star leg", "foot", "feet", "pedestal", "spoke"],
        "caster": ["caster", "castor", "caster fork", "caster connector", "caster housing", "caster bracket",
                   "caster body", "fork", "caster stem", "caster mount", "wheel housing", "wheel fork",
                   "wheel bracket", "caster assembly"],
        "caster wheel": ["wheel", "caster wheel", "castor wheel", "roller", "tire", "tyre"],
        "swivel mechanism": ["swivel", "swivel mechanism", "swivel plate", "tilt mechanism", "seat plate",
                             "seat mechanism", "chair control", "mechanism", "swivel joint", "bearing",
                             "seat bracket", "mounting plate", "swivel bearing", "tilt control", "control box",
                             "swivel base"],
        "lever": ["lever", "height lever", "adjustment lever", "height adjustment lever", "lever rod",
                  "lever handle", "paddle", "release lever", "control lever"],
    },
}

# For presence checks a claim about the whole assembly is satisfied by either
# canonical role that realises it ("casters" are the fork-and-wheel unit).
ACCEPT = {
    "swivel_office_chair": {"caster": {"caster", "caster wheel"}},
}


def _sing(t: str) -> str:
    if len(t) > 3 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokens(raw: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(raw or ""))
    s = re.sub(r"[^A-Za-z]+", " ", s).lower()
    return [_sing(t) for t in s.split() if t and t not in POSITIONAL]


def auto_table(task_id: str, task=None) -> dict:
    """The table for a task nobody has hand-written one for: every phrase the task itself uses for a part
    (claimed parts P, subsystem parts F, kinematics K) plus the reference object's own labels, grouped by
    head noun. "table leg", "leg" and "front leg" are one role; "leg" and "column" are not, because only a
    person can know that they are. Weaker than a curated table, and marked as such (`Lexicon.auto`)."""
    from ppbench.v2.task import RESULTS, Task
    if task is None:
        snap = RESULTS / task_id / "task_snapshot_core.json"
        task = Task(task_id, snapshot=snap) if snap.exists() else Task(task_id)
    phrases = [p["part"] for p in task.required_parts]
    phrases += [n for f in task.subsystems for n in f.get("parts", [])]
    for k in task.kinematics:
        phrases += [k.get("moving_part"), k.get("relative_to")]
    try:
        phrases += [p["role_raw"] for p in task.reference()["parts"]]
    except Exception:      # no reference object (the LDraw tasks): claims alone
        pass
    groups = {}
    for ph in phrases:
        tk = tokens(ph)
        if tk:
            groups.setdefault(tk[-1], set()).add(" ".join(tk))
    table = {}
    for head, phs in groups.items():
        name = min(phs, key=lambda x: (len(x.split()), len(x), x))   # the shortest phrase names the role
        joined = {x.replace(" ", "") for x in phs if " " in x}       # "table top" and "tabletop" are one word
        table[name] = sorted(phs | {head} | joined)
    return table


BRIDGES = Path(__file__).parent / "lexicons" / "bridges.json"
_bridges = None


def bridges() -> dict:
    """Authored synonym groups that join a claim's word to the reference's and to the words a
    design is likely to use ("carcass" is a cabinet frame, "handrim" is a push-rim). One file for
    every task, applied on top of the automatic table, so no metric has to guess."""
    global _bridges
    if _bridges is None:
        _bridges = json.loads(BRIDGES.read_text()) if BRIDGES.exists() else {}
    return _bridges


class Lexicon:
    def __init__(self, task_id: str, task=None):
        self.task_id = task_id
        self.auto = task_id not in TABLES
        self.table = dict(TABLES[task_id] if not self.auto else auto_table(task_id, task))
        self.accept = dict(ACCEPT.get(task_id, {}))
        br = bridges().get(task_id) or {}
        self.bridged = bool(br)
        for canon, syns in (br.get("table") or {}).items():
            want = {s.lower() for s in syns} | {canon.lower()}
            # a phrase this group claims must not stay in another role, or the match is a coin flip
            for other, olist in list(self.table.items()):
                if other == canon:
                    continue
                keep = [x for x in olist if x.lower() not in want]
                if keep:
                    self.table[other] = keep
                else:
                    del self.table[other]
            self.table[canon] = sorted(set(self.table.get(canon, [])) | want)
        for canon, acc in (br.get("accept") or {}).items():
            self.accept[canon] = set(self.accept.get(canon, {canon})) | set(acc)
        self._phr = []
        for canon, syns in self.table.items():
            for s in set(syns) | {canon}:
                tk = tokens(s)
                if tk:
                    self._phr.append((tuple(tk), canon))

    def canon(self, raw) -> str | None:
        """Parenthetical text is an annotation ("chair base (5-star swivel base)") and is ignored;
        a composite name ("gas lift column and seat plate") takes the role of its first conjunct."""
        s = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", str(raw or ""))
        head = re.split(r"\s+(?:and|with|plus)\s+|\s*[&+/,]\s*", s, maxsplit=1)[0]
        if head.strip() and head.strip() != s.strip():
            c = self._canon(head)
            if c is not None:
                return c
        return self._canon(s)

    def _canon(self, raw) -> str | None:
        tk = tokens(raw)
        if not tk:
            return None
        c = self._match(tk)
        if c is None and len(tk) > 1:
            # datasets write compounds both ways ("table top" / "tabletop"): retry with one adjacent pair merged
            for i in range(len(tk) - 1):
                c = self._match(tk[:i] + [tk[i] + tk[i + 1]] + tk[i + 2:])
                if c is not None:
                    break
        return c

    def _match(self, tk) -> str | None:
        best = None   # (len, end, canon)
        for phr, canon in self._phr:
            n = len(phr)
            for i in range(len(tk) - n + 1):
                if tuple(tk[i:i + n]) == phr:
                    key = (n, i + n)
                    if best is None or key > best[:2]:
                        best = (n, i + n, canon)
        return best[2] if best else None

    def satisfies(self, claim_phrase: str, role_canon: str | None, broad: bool = True) -> bool:
        """Does a part with canonical role `role_canon` realise the claim's part phrase?

        broad=True (presence checks): a generic phrase ("casters") is realised by any
        role in its ACCEPT set. The expansion never applies to a specific phrase
        ("caster fork"), and never when broad=False (kinematics and connections,
        where "casters -> wheels" must not match a wheel to itself)."""
        c = self.canon(claim_phrase)
        if c is None or role_canon is None:
            return False
        if broad and tokens(claim_phrase) == tokens(c):
            return role_canon in self.accept.get(c, {c})
        return role_canon == c

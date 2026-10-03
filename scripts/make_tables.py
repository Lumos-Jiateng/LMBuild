"""Paper tables from results/scores/all_scores.csv (written by scripts/export_scores.py). No main-repo paths, no GPU.

    python scripts/make_tables.py [--csv results/scores/all_scores.csv] [--latex results/paper] > results/tables.md

Every cell is the mean over tasks (x100) of one metric for one system in one setting.

    main200   THE PAPER'S MAIN TABLE: all 200 core tasks, Tier B, name_only+image, round 1 (rows flagged main200=1 by
              export_scores.py). A (system, task) with no record scores 0; a metric that is not applicable to a design
              (record present, no score) is left out of that mean. Also: original 50 vs new 150, the published setting
              (original 50, 3 rounds, final design), stratified 50-task resamples and ranking-validity statistics.
    main      original 50 tasks, Tier B, name_only+image, one 3-round trajectory, final design (published setting)
    tier      the same trajectory in Tier A (catalogue only) and Tier C (creation only)
    rounds    checkpoints after round 1 / 2 / 3 of the main trajectory
    prompts   Tier B, one round: name_only+image vs attributes+image vs functional+image
For the ablation sections a dimension with no score is left out of that mean; an empty or failed design is already 0.

--latex DIR writes the LaTeX tables of the paper (main table, resample results / correlations, metric audit) into DIR.
The resampling uses seed 20261002 and the catalog-sheet membership of benchmark/tasks/<t>/pool_sheet/pool_sheet_all.png.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import statistics
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ["S.1", "S.2", "S.3", "A.1", "A.2", "A.3", "D.1", "D.2", "D.3", "R.1", "R.2", "R.3"]
CLOSED = ["gpt-6-astra", "gpt-5.6-sol", "claude-opus-5", "claude-fable-5.1", "claude-sonnet-5", "claude-haiku-4.5"]
OPEN = ["qwen3.5-27b", "qwen3.5-35b-a3b", "gpt-oss-120b", "gemma-4-31b-it", "qwen3-vl-32b-instruct",
        "qwen3-vl-30b-a3b-instruct", "qwen3-vl-8b-instruct", "qwen3-4b-instruct-2507", "internvl3.5-38b",
        "internvl3.5-8b", "minicpm-v-4.5", "ernie-4.5-vl-28b-a3b", "ministral-3-8b-instruct-2512"]
GEN = [("partcrafter_np15", "image"), ("particulate-partcrafter", "image"), ("partcrafter_np8", "image"),
       ("particulate-partcrafter-np8", "image"), ("partpacker", "image"), ("particulate-partpacker", "image"),
       ("physx-anything", "image"), ("cubepart_core", "name_only"), ("particulate-cube3d", "name_only"),
       ("brickgpt", "name_only"), ("legoace", "name_only")]
EXT = [s for s, _ in GEN]
SYS = CLOSED + OPEN + EXT
# paper-table order and labels
ORDER_T = ["S.1", "S.2", "S.3", "A.1", "A.2", "A.3", "D.1", "D.2", "D.3", "R.1", "R.2", "R.3"]
LAB = {"gpt-6-astra": "GPT-6 Astra", "gpt-5.6-sol": "GPT-5.6 Sol", "claude-opus-5": "Claude Opus 5", "claude-fable-5.1": "Claude Fable 5.1",
       "claude-sonnet-5": "Claude Sonnet 5", "claude-haiku-4.5": "Claude Haiku 4.5", "qwen3.5-27b": "Qwen3.5-27B", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
       "gpt-oss-120b": "GPT-OSS-120B", "gemma-4-31b-it": "Gemma-4-31B-IT", "ministral-3-8b-instruct-2512": "Ministral-3-8B",
       "internvl3.5-38b": "InternVL3.5-38B", "qwen3-vl-30b-a3b-instruct": "Qwen3-VL-30B-A3B", "minicpm-v-4.5": "MiniCPM-V-4.5",
       "qwen3-4b-instruct-2507": "Qwen3-4B-Instruct", "qwen3-vl-32b-instruct": "Qwen3-VL-32B", "ernie-4.5-vl-28b-a3b": "ERNIE-4.5-VL-28B-A3B",
       "internvl3.5-8b": "InternVL3.5-8B", "qwen3-vl-8b-instruct": "Qwen3-VL-8B", "brickgpt": "BrickGPT", "legoace": "LegoACE",
       "partcrafter_np15": "PartCrafter-15", "partcrafter_np8": "PartCrafter-8", "partpacker": "PartPacker", "cubepart_core": "Cube3D+CubePart",
       "particulate-partcrafter": "PartCrafter15$^{\\dagger}$(P)", "particulate-partcrafter-np8": "PartCrafter8$^{\\dagger}$(P)",
       "particulate-partpacker": "PartPacker$^{\\dagger}$(P)", "particulate-cube3d": "Cube3D$^{\\dagger}$(P)", "physx-anything": "PhysX-Anything"}
T_ORDER = CLOSED + ["qwen3.5-27b", "qwen3.5-35b-a3b", "gpt-oss-120b", "gemma-4-31b-it", "ministral-3-8b-instruct-2512", "internvl3.5-38b",
                    "qwen3-vl-30b-a3b-instruct", "minicpm-v-4.5", "qwen3-4b-instruct-2507", "qwen3-vl-32b-instruct", "ernie-4.5-vl-28b-a3b",
                    "internvl3.5-8b", "qwen3-vl-8b-instruct"] + \
          ["brickgpt", "legoace", "partcrafter_np15", "partcrafter_np8", "partpacker", "cubepart_core", "particulate-partcrafter",
           "particulate-partcrafter-np8", "particulate-partpacker", "particulate-cube3d", "physx-anything"]
NA4, NAP = {"A.3", "R.1", "R.2", "R.3"}, {"R.1", "R.2", "R.3"}
NA = {"brickgpt": NA4, "legoace": NA4, "partcrafter_np15": NA4, "partcrafter_np8": NA4, "partpacker": NA4,
      "cubepart_core": {"A.3", "R.1", "R.2"}, "particulate-partcrafter": NAP, "particulate-partcrafter-np8": NAP,
      "particulate-partpacker": NAP, "particulate-cube3d": NAP, "physx-anything": {"R.1"}}   # shown as N/A in the paper
TITLES = ["State-of-the-art Closed-Source APIs", "Open-Source (Multi-modal) Large Language Models", "State-of-the-art Domain-Specific Models"]
MAIN_HEAD = '{\n\\renewcommand{\\arraystretch}{1.18}\n\n\\begin{table*}[t]\n\\centering\n\\caption{Evaluation results across 30 systems under Baseline Interaction Protocol (\\S~\\ref{sec:2.1}) on core set, assessed using our hierarchical evaluation metrics (\\S~\\ref{sec:2.3}). We use $^{\\dagger}$ (P) to denote systems integrated with Particulate to obtain joint information (Appendix~\\ref{app:environment}). Background colors encode performance on a common 0--100 scale, transitioning from blue (lower performance) through gray to orange (higher performance). Systems unable to produce the required output are assigned as N/A.}\n\\label{tab:main_results}\n\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l|ccc|ccc|ccc|ccc}\n\\toprule\n\n\\multirow{2}{*}{\\textbf{Models / Metrics}}\n& \\multicolumn{3}{c|}{\\textbf{Soundness}}\n& \\multicolumn{3}{c|}{\\textbf{Affordance}}\n& \\multicolumn{3}{c|}{\\textbf{Design}}\n& \\multicolumn{3}{c}{\\textbf{Realization}} \\\\\n\n\\cmidrule(lr){2-4}\n\\cmidrule(lr){5-7}\n\\cmidrule(lr){8-10}\n\\cmidrule(lr){11-13}\n\n& \\textbf{S.1} & \\textbf{S.2} & \\textbf{S.3}\n& \\textbf{A.1} & \\textbf{A.2} & \\textbf{A.3}\n& \\textbf{D.1} & \\textbf{D.2} & \\textbf{D.3}\n& \\textbf{R.1} & \\textbf{R.2} & \\textbf{R.3} \\\\\n\n\\midrule\n\n'
MAIN_FOOT = '\n\\bottomrule\n\\end{tabular}\n}\n\\end{table*}\n}\n'
RS_HEAD = '{\n\\renewcommand{\\arraystretch}{1.12}\n\\begin{table*}[t]\n\\centering\n\\caption{Level scores (mean of the three metrics in each level; N/A metrics excluded) on the full 200-object core set and on two disjoint stratified 50-object samples. Sample~A holds 13 objects with a catalog contact sheet and 37 without; sample~B holds 12 and 38, matching the 1:4 ratio of the full set. No new runs are involved: all three columns are computed from the same episodes.}\n\\label{tab:resample_results}\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l|cccc|cccc|cccc}\n\\toprule\n\\multirow{2}{*}{\\textbf{Models}} & \\multicolumn{4}{c|}{\\textbf{Full core (200)}} & \\multicolumn{4}{c|}{\\textbf{Sample A (50)}} & \\multicolumn{4}{c}{\\textbf{Sample B (50)}} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n & \\textbf{S} & \\textbf{A} & \\textbf{D} & \\textbf{R} & \\textbf{S} & \\textbf{A} & \\textbf{D} & \\textbf{R} & \\textbf{S} & \\textbf{A} & \\textbf{D} & \\textbf{R} \\\\\n\\midrule'
RC_HEAD = "\\begin{table}[t]\n\\centering\n\\caption{Agreement of system-level scores across task sets: Pearson $r$ (Spearman $\\rho$ in parentheses) over the 30 systems' means (19 agents for Realization, where generators are N/A). A vs.\\ B compares two disjoint 50-object samples; the last two columns repeat the stratified draw 1{,}000 times (mean, 95\\% interval in brackets). Sample-vs-full correlations are an upper bound because each sample is part of the full set.}\n\\label{tab:resample_corr}\n\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{l|ccc|cc}\n\\toprule\n\\multirow{2}{*}{\\textbf{Metric}} & \\multirow{2}{*}{\\textbf{A vs.\\ 200}} & \\multirow{2}{*}{\\textbf{B vs.\\ 200}} & \\multirow{2}{*}{\\textbf{A vs.\\ B}} & \\multicolumn{2}{c}{\\textbf{1{,}000 resamples}} \\\\\n\\cmidrule(lr){5-6}\n & & & & \\textbf{50 vs.\\ 200} & \\textbf{disjoint pairs} \\\\\n\\midrule"
AU_HEAD = "\\begin{table}[t]\n\\centering\n\\caption{Metric revisions after the audit (200-object means; before $\\rightarrow$ after). Every rule is applied to all 30 systems. $^{*}$Before the revision PhysX-Anything's R.2 was averaged over the 109 objects with at least one recognised part; after it, over all 200.}\n\\label{tab:metric_audit}\n\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{lllcc}\n\\toprule\n\\textbf{Metric} & \\textbf{Revision} & \\textbf{System} & \\textbf{Before} & \\textbf{After} \\\\\n\\midrule"


def load(path):
    cells = collections.defaultdict(list)
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        cells[(r["system"], r["tier"], r["condition"], r["trajectory_rounds"], r["checkpoint"])].append(r)
    return cells, rows


def row(label, rs, metrics=M):
    out = []
    for m in metrics:
        xs = [float(r[m]) for r in rs if r[m] != ""]
        out.append(f"{100 * statistics.mean(xs):.1f}" if xs else "–")
    return f"| {label} | " + " | ".join(out) + f" | {len({r['task'] for r in rs})} |"


def header(title, metrics=M):
    return [f"\n### {title}\n", "| system | " + " | ".join(metrics) + " | N |", "|---|" + "---:|" * (len(metrics) + 1)]


# ---------------------------------------------------------------- the 200-task main table (R1)
def _val(r, m):
    """A record's score: float, or None when the record exists but the metric does not apply; a missing record is 0."""
    if r is None:
        return 0.0
    if r[m] != "":
        return float(r[m])
    return 0.0 if r[m + "_status"] == "" else None


def grid(rows, tasks):
    """X[metric][system][task] for the main200 records (R1) and P[metric][agent][task] for the published r3 setting."""
    cand = collections.defaultdict(list)
    for r in rows:
        if r.get("main200"):
            cand[(r["system"], r["task"])].append(r)
    for v in cand.values():
        v.sort(key=lambda r: int(r["main200"]))

    def pick(s, t, m):                 # best-ranked candidate with a record in this metric's set
        return next((r for r in cand.get((s, t), []) if r[m + "_status"] != ""), None)
    r3 = {(r["system"], r["task"]): r for r in rows if r["tier"] == "B" and r["condition"] == "name_only+image"
          and r["trajectory_rounds"] == "3" and r["checkpoint"] == "final"}
    X = {m: {s: {t: _val(pick(s, t, m), m) for t in tasks} for s in SYS} for m in M}
    P = {m: {s: {t: _val(r3.get((s, t)), m) for t in tasks[:50]} for s in CLOSED + OPEN} for m in M}
    return X, P


def mean_over(X, m, s, ts):
    v = [X[m][s][t] for t in ts if X[m][s][t] is not None]
    return 100 * st.mean(v) if v else None


def table_mean(X, m, s, ts):
    """The published tables' cell value: the mean x100 rounded to two decimals, then shown to one (as in the paper)."""
    v = mean_over(X, m, s, ts)
    return None if v is None else round(v, 2)


def main200_md(X, P, tasks):
    O, N = tasks[:50], tasks[50:]
    out = header("Main table: 200 core tasks, Tier B, name_only+image, round 1 (paper Table 2)")
    for grp, names in (("Closed-source APIs", CLOSED), ("Open-source models", OPEN), ("Domain-specific generators", EXT)):
        out.append(f"| *{grp}* |" + " |" * (len(M) + 1))
        for s in names:
            out.append(f"| {s} | " + " | ".join("N/A" if m in NA.get(s, ()) else f"{table_mean(X, m, s, tasks):.1f}" for m in M) + f" | {len(tasks)} |")
    for title, ts in (("original 50 tasks (round 1)", O), ("new 150 tasks (round 1)", N)):
        out += header(f"Main setting on the {title}")
        for s in SYS:
            out.append(f"| {s} | " + " | ".join("N/A" if m in NA.get(s, ()) else f"{table_mean(X, m, s, ts):.1f}" for m in M) + f" | {len(ts)} |")
    out += header("Published setting: original 50 tasks, 3 rounds, final design")
    for s in CLOSED + OPEN:
        out.append(f"| {s} | " + " | ".join(f"{table_mean(P, m, s, O):.1f}" for m in M) + " | 50 |")
    return out


def main_tex(X, tasks):
    def trow(s):
        c = [("\\scoreNA" if m in NA.get(s, ()) else "\\score{%.1f}" % table_mean(X, m, s, tasks)) for m in ORDER_T]
        return f"{LAB[s]}\n& {' & '.join(c[0:3])}\n& {' & '.join(c[3:6])}\n& {' & '.join(c[6:9])}\n& {' & '.join(c[9:12])} \\\\\n"
    body = ""
    for i, (t, g) in enumerate(zip(TITLES, (T_ORDER[:6], T_ORDER[6:19], T_ORDER[19:]))):
        if i:
            body += "\n\\midrule\n\n"
        body += "\\rowcolor{gray!20}\n\\multicolumn{13}{c}{\\textbf{%s}} \\\\\n\n" % t + "\n".join(trow(s) for s in g)
    return MAIN_HEAD + body + MAIN_FOOT


# ---------------------------------------------------------------- stratified resampling and validity
def _pearson(x, y):
    from scipy.stats import pearsonr, spearmanr
    return pearsonr(x, y)[0], spearmanr(x, y)[0]


def resample(X, tasks, with_sheet):
    WITH = [t for t in tasks if t in with_sheet]
    WITHOUT = [t for t in tasks if t not in WITH]
    D = {"S.1": "1.1", "S.2": "1.2", "S.3": "1.3", "A.1": "2.1", "A.2": "2.2", "A.3": "2.3",
         "D.1": "3.1", "D.2": "3.2", "D.3": "3.3", "R.1": "P.1", "R.2": "P.2", "R.3": "P.3"}

    def means(m, ts):
        return {s: mean_over(X, m, s, ts) for s in SYS}

    def corr(a, b, m):
        ss = [s for s in SYS if a[s] is not None and b[s] is not None and not (m.startswith("R") and s in EXT)]
        x, y = [a[s] for s in ss], [b[s] for s in ss]
        if len(set(x)) < 2 or len(set(y)) < 2:
            return None, None
        return _pearson(x, y)

    rng = random.Random(20261002)
    w = rng.sample(WITH, 25); wo = rng.sample(WITHOUT, 75)
    A = w[:13] + wo[:37]; B = w[13:25] + wo[37:75]
    res = {"n_with_sheet": len(WITH), "subsets": {"A": A, "B": B}, "full": {}, "A": {}, "B": {}, "corr": {}, "resample": {}}
    for m in M:   # metric order S, A, D, R: the draw order of the paper's run
        F, MA, MB = means(m, tasks), means(m, A), means(m, B)
        res["full"][m], res["A"][m], res["B"][m] = F, MA, MB
        res["corr"][m] = {"A_vs_full": corr(MA, F, m), "B_vs_full": corr(MB, F, m), "A_vs_B": corr(MA, MB, m)}
        sub, dis = [], []
        for _ in range(1000):
            w1 = rng.sample(WITH, 25); o1 = rng.sample(WITHOUT, 75)
            k = rng.choice((12, 13))
            S1 = w1[:k] + o1[:50 - k]; S2 = w1[k:] + o1[50 - k:]
            m1, m2 = means(m, S1), means(m, S2)
            c1, c2 = corr(m1, F, m), corr(m1, m2, m)
            if c1[0] is not None: sub.append(c1[0])
            if c2[0] is not None: dis.append(c2[0])
        q = lambda v: {"mean": st.mean(v), "p2.5": sorted(v)[int(0.025 * len(v))], "p97.5": sorted(v)[int(0.975 * len(v)) - 1]}
        res["resample"][m] = {"subset_vs_full_pearson": q(sub), "disjoint_pair_pearson": q(dis)}
    return res


def validity(X, P, tasks):
    from scipy.stats import kendalltau, spearmanr
    O, N = tasks[:50], tasks[50:]
    out = {}
    for m in ["S.1", "S.2", "S.3", "D.1", "D.2", "D.3", "A.1", "A.2", "A.3", "R.1", "R.2", "R.3"]:   # the paper run's order
        ok = [s for s in SYS if mean_over(X, m, s, O) is not None and mean_over(X, m, s, N) is not None]
        a, b, c = [mean_over(X, m, s, O) for s in ok], [mean_over(X, m, s, N) for s in ok], [mean_over(X, m, s, tasks) for s in ok]
        ok3 = [s for s in ok if s in CLOSED + OPEN and mean_over(P, m, s, O) is not None]
        a3, c3 = [mean_over(P, m, s, O) for s in ok3], [mean_over(X, m, s, tasks) for s in ok3]
        rng = random.Random(0); boots = []
        for _ in range(1000):
            smp = [rng.choice(tasks) for _ in tasks]
            boots.append(spearmanr(c, [100 * st.mean([X[m][s][t] if X[m][s][t] is not None else 0 for t in smp]) for s in ok])[0])
        boots.sort()
        out[m] = {"rho_orig50_new150": spearmanr(a, b)[0], "tau_orig50_new150": kendalltau(a, b)[0],
                  "rho_published50r3_all200": spearmanr(a3, c3)[0], "bootstrap_rho": st.mean(boots), "ci95": [boots[25], boots[974]]}
    return out


def resample_tex(R):
    LV = [["S.1", "S.2", "S.3"], ["A.1", "A.2", "A.3"], ["D.1", "D.2", "D.3"], ["R.1", "R.2", "R.3"]]
    def lv(src, s, ms):
        v = [R[src][m][s] for m in ms if m not in NA.get(s, ())]
        return None if not v else st.mean(v)
    L = [RS_HEAD]
    for i, (t, g) in enumerate(zip(TITLES, (T_ORDER[:6], T_ORDER[6:19], T_ORDER[19:]))):
        L.append("\\rowcolor{gray!20}\n\\multicolumn{13}{c}{\\textbf{%s}} \\\\" % t)
        for s in g:
            cells = []
            for src in ("full", "A", "B"):
                for ms in LV:
                    x = lv(src, s, ms); cells.append("\\scoreNA" if x is None else "\\score{%.1f}" % x)
            L.append(LAB[s] + " & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule" if i < 2 else "\\bottomrule")
    L.append("\\end{tabular}\n}\n\\end{table*}\n}")
    return "\n".join(L) + "\n"


def corr_tex(R):
    C = [RC_HEAD]
    for m in M:
        c, z = R["corr"][m], R["resample"][m]
        f = lambda p: "%.3f (%.2f)" % tuple(p)
        q = lambda x: "%.3f [%.2f, %.2f]" % (x["mean"], x["p2.5"], x["p97.5"])
        C.append(f"{m} & {f(c['A_vs_full'])} & {f(c['B_vs_full'])} & {f(c['A_vs_B'])} & {q(z['subset_vs_full_pearson'])} & {q(z['disjoint_pair_pearson'])} \\\\")
        if m in ("S.3", "A.3", "D.3"):
            C.append("\\midrule")
    C.append("\\bottomrule\n\\end{tabular}\n}\n\\end{table}")
    return "\n".join(C) + "\n"


def audit_tex(X, tasks, before):
    OPEN13 = ["qwen3.5-27b", "qwen3.5-35b-a3b", "gpt-oss-120b", "gemma-4-31b-it", "qwen3-vl-32b-instruct", "qwen3-vl-30b-a3b-instruct",
              "qwen3-vl-8b-instruct", "qwen3-4b-instruct-2507", "internvl3.5-38b", "internvl3.5-8b", "minicpm-v-4.5",
              "ernie-4.5-vl-28b-a3b", "ministral-3-8b-instruct-2512"]
    D = {"S.1": "1.1", "S.2": "1.2", "S.3": "1.3", "A.2": "2.2", "D.2": "3.2", "R.2": "P.2", "R.3": "P.3"}
    b = lambda s, m: before[s][D[m]]
    n = lambda s, m: table_mean(X, m, s, tasks)
    g = lambda f, ss, m: st.mean(f(s, m) for s in ss)
    rows = [("S.2 Collision", "pitch and thresholds scale with the design", "LegoACE", b("legoace", "S.2"), n("legoace", "S.2")),
            ("", "", "BrickGPT", b("brickgpt", "S.2"), n("brickgpt", "S.2")),
            ("", "", "Closed APIs (mean)", g(b, CLOSED, "S.2"), g(n, CLOSED, "S.2")), ("", "", "Open models (mean)", g(b, OPEN13, "S.2"), g(n, OPEN13, "S.2")),
            ("S.1 Connectivity", "(same instrument)", "Closed APIs (mean)", g(b, CLOSED, "S.1"), g(n, CLOSED, "S.1")),
            ("S.3 Stability", "ground band scales with the design", "LegoACE", b("legoace", "S.3"), n("legoace", "S.3")),
            ("A.2 Parts", "a name must be confirmed by geometry", "Cube3D+CubePart", b("cubepart_core", "A.2"), n("cubepart_core", "A.2")),
            ("", "", "Closed APIs (mean)", g(b, CLOSED, "A.2"), g(n, CLOSED, "A.2")), ("", "", "Open models (mean)", g(b, OPEN13, "A.2"), g(n, OPEN13, "A.2")),
            ("R.3 Operability", "part presence uses verified roles", "Cube3D+CubePart", b("cubepart_core", "R.3"), n("cubepart_core", "R.3")),
            ("", "", "Closed APIs (mean)", g(b, CLOSED, "R.3"), g(n, CLOSED, "R.3")),
            ("R.2 Material", "role by name or geometry; no part skipped", "PhysX-Anything$^{*}$", b("physx-anything", "R.2"), n("physx-anything", "R.2")),
            ("", "", "Closed APIs (mean)", g(b, CLOSED, "R.2"), g(n, CLOSED, "R.2")),
            ("D.1--D.3 Design", "both judges on every design", "Closed APIs (mean D.2)", g(b, CLOSED, "D.2"), g(n, CLOSED, "D.2")),
            ("", "", "Open models (mean D.2)", g(b, OPEN13, "D.2"), g(n, OPEN13, "D.2")),
            ("", "", "Cube3D$^{\\dagger}$(P) (D.2)", b("particulate-cube3d", "D.2"), n("particulate-cube3d", "D.2"))]
    L = [AU_HEAD]
    for i, (a, bb, c, x, y) in enumerate(rows):
        if a and i:
            L.append("\\midrule")
        L.append(f"{a} & {bb} & {c} & {x:.1f} & {y:.1f} \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n}\n\\end{table}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "scores" / "all_scores.csv"))
    ap.add_argument("--core", default=os.environ.get("PPB_CORE", str(ROOT / "benchmark" / "core_v2.json")))
    ap.add_argument("--latex", default=None, help="write the paper's LaTeX tables into this directory")
    ap.add_argument("--json", default=None, help="also dump the main-table means, resamples and validity as JSON")
    a = ap.parse_args()
    C, rows = load(a.csv)
    tasks = [t["task_id"] for t in json.loads(Path(a.core).read_text())]
    out = ["# LMBuild results (generated by scripts/make_tables.py)"]

    if any(r.get("main200") for r in rows):
        X, P = grid(rows, tasks)
        out += main200_md(X, P, tasks)
        with_sheet = {t for t in tasks if (ROOT / "benchmark" / "tasks" / t / "pool_sheet" / "pool_sheet_all.png").exists()}
        R = resample(X, tasks, with_sheet)
        V = validity(X, P, tasks)
        out += [f"\n### Stratified 50-task resamples ({R['n_with_sheet']} tasks with a catalog sheet; A = 13+37, B = 12+38)\n",
                "| metric | A vs 200 r (rho) | B vs 200 r (rho) | A vs B r (rho) | 1000x 50 vs 200 r [95%] | 1000x disjoint r [95%] |", "|---|---|---|---|---|---|"]
        for m in M:
            c, z = R["corr"][m], R["resample"][m]
            f = lambda p: "%.3f (%.2f)" % tuple(p)
            q = lambda x: "%.3f [%.2f, %.2f]" % (x["mean"], x["p2.5"], x["p97.5"])
            out.append(f"| {m} | {f(c['A_vs_full'])} | {f(c['B_vs_full'])} | {f(c['A_vs_B'])} | {q(z['subset_vs_full_pearson'])} | {q(z['disjoint_pair_pearson'])} |")
        out += ["\n### Ranking validity (30 systems; published = original 50 at 3 rounds, 19 agents)\n",
                "| metric | rho orig50 vs new150 | tau | rho published vs 200 | bootstrap rho [95%] |", "|---|---|---|---|---|"]
        for m, v in V.items():
            out.append(f"| {m} | {v['rho_orig50_new150']:.3f} | {v['tau_orig50_new150']:.3f} | {v['rho_published50r3_all200']:.3f} | "
                       f"{v['bootstrap_rho']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
        if a.latex:
            d = Path(a.latex); d.mkdir(parents=True, exist_ok=True)
            (d / "table_main_results_core200.tex").write_text(main_tex(X, tasks))
            (d / "table_resample_results.tex").write_text(resample_tex(R))
            (d / "table_resample_corr.tex").write_text(corr_tex(R))
            bf = ROOT / "results" / "scores" / "metrics_before_v38.json"
            if bf.exists():
                (d / "table_metric_audit.tex").write_text(audit_tex(X, tasks, json.loads(bf.read_text())["values"]))
        if a.json:
            Path(a.json).write_text(json.dumps({"main": {s: {m: mean_over(X, m, s, tasks) for m in M} for s in SYS},
                                                "orig50": {s: {m: mean_over(X, m, s, tasks[:50]) for m in M} for s in SYS},
                                                "new150": {s: {m: mean_over(X, m, s, tasks[50:]) for m in M} for s in SYS},
                                                "published50r3": {s: {m: mean_over(P, m, s, tasks[:50]) for m in M} for s in CLOSED + OPEN},
                                                "resample": R, "validity": V}, indent=1))

    out += header("Published setting, original 50 tasks: Tier B, name_only+image, 3 rounds (final design)")
    for grp, names in (("Closed-source APIs", CLOSED), ("Open-source models", OPEN)):
        out.append(f"| *{grp}* |" + " |" * (len(M) + 1))
        out += [row(s, C[(s, "B", "name_only+image", "3", "final")]) for s in names if C[(s, "B", "name_only+image", "3", "final")]]
    out.append("| *Domain-specific generators* |" + " |" * (len(M) + 1))
    out += [row(f"{s} ({c})", [r for r in C[(s, "ext", c, "", "final")] if r["task"] in tasks[:50]])
            for s, c in GEN if C[(s, "ext", c, "", "final")]]

    for tier, what in (("A", "catalogue only"), ("C", "creation only")):
        out += header(f"Tier {tier} ({what}), name_only+image, 3 rounds (final design)")
        out += [row(s, C[(s, tier, "name_only+image", "3", "final")]) for s in CLOSED + OPEN
                if C[(s, tier, "name_only+image", "3", "final")]]

    out += header("Rounds: Tier B, name_only+image, checkpoints of the 3-round trajectory")
    for s in CLOSED + OPEN:
        for k in ("1", "2", "3"):
            rs = C[(s, "B", "name_only+image", "3", k)]
            if rs:
                out.append(row(f"{s} · R{k}", rs))

    out += header("Prompt ablation: Tier B, 1 round (final design)")
    for s in CLOSED + OPEN:
        for cond in ("name_only+image", "attributes+image", "functional+image"):
            rs = [r for r in C[(s, "B", cond, "1", "final")] if r["task"] in tasks[:50]]
            if rs:
                out.append(row(f"{s} · {cond}", rs))
    print("\n".join(out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write LaTeX table fragments and a plain-text summary from the result files.

Reads `<results>/*__s*.json` (written by run_experiments.py) and writes

    <out>/tables/*.tex          table bodies (tabular environments)
    <out>/tables/numbers.tex    macros for numbers that are quoted in the text
    <out>/RESULTS_SUMMARY.txt   every number in plain text, with the statistics

A spread is the sample standard deviation over seeds and is only printed if a
configuration has more than one seed. Tables that describe a single run
(actions, firing regularity, neuron roles, accuracy matrices) use the run with
the lowest seed.

Example
    python scripts/make_tables.py --results output/results --out output
"""

import argparse
import glob
import json
import os
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

# Make the package importable when the script is run from a clone that has not
# been installed with pip.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgcl_snn import cohens_d, mean_sd, welch_t  # noqa: E402

ACTIONS = ["UPDATE", "CONSOLIDATE", "PROTECT", "IGNORE"]
LADDER = ["LIF", "ALIF", "ALIF+SI", "ALIF+SI+DM"]
TOP, MID = "\\toprule", "\\midrule"

# Pairs (a, b) for which Cohen's d and Welch's t of a against b are reported.
COMPARISONS = [
    ("block1_taskIL|ALIF", "block1_taskIL|LIF"),
    ("block1_taskIL|ALIF+SI", "block1_taskIL|LIF"),
    ("block1_taskIL|ALIF+SI+DM", "block1_taskIL|ALIF+SI"),
    ("block1_taskIL|ALIF+SI+DM", "block2_control|ALIF+SI+Random"),
]


# ----------------------------------------------------------------------------
# Loading and formatting
# ----------------------------------------------------------------------------
def load_runs(results_dir):
    """Result files grouped by '<block>|<configuration>', sorted by file name."""
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*__s*.json"))):
        with open(path) as f:
            r = json.load(f)
        block = os.path.basename(path).split("__")[0]
        runs[f"{block}|{r['config']}"].append(r)
    return runs


def values(rs, *path):
    """One value per run, addressed by a path of keys."""
    out = []
    for r in rs:
        x = r
        for k in path:
            x = x[k]
        out.append(x)
    return out


def pm(vals, digits=4):
    """'$mean$' for one value, '$mean \\pm sd$' for several."""
    m, s = mean_sd(vals)
    if len(vals) < 2:
        return f"${m:.{digits}f}$"
    return f"${m:.{digits}f} \\pm {s:.{digits}f}$"


def pct(x):
    return f"{x * 100:.1f}\\,\\%"


def find(runs, block=None, exact=None, contains=None):
    """First key that matches the block and the configuration name."""
    for key in runs:
        b, c = key.split("|", 1)
        if block and b != block:
            continue
        if exact and c != exact:
            continue
        if contains and contains not in c:
            continue
        return key
    return None


def ignored_share(rs):
    return float(np.mean([r["action_global"].get("IGNORE", 0.0) for r in rs]))


class Writer:
    def __init__(self, out_dir):
        self.dir = os.path.join(out_dir, "tables")
        os.makedirs(self.dir, exist_ok=True)

    def table(self, name, cols, header, rows):
        body = f"\\begin{{tabular}}{{{cols}}}\n{header}\n{rows}\n\\bottomrule\n\\end{{tabular}}\n"
        self.text(name, body)

    def text(self, name, content):
        with open(os.path.join(self.dir, name), "w") as f:
            f.write(content)
        print("  ", name)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def ladder_table(runs, w, block, fname, suffix=""):
    """Accuracy, forgetting and backward transfer of the ablation ladder."""
    rows = []
    for name in LADDER:
        key = find(runs, block=block, exact=name + suffix)
        if not key:
            continue
        rs = runs[key]
        learned = float(np.mean(values(rs, "compute", "backward_sample_fraction")))
        rows.append(f"{name} & {pm(values(rs, 'avg_acc'))} & "
                    f"{pm(values(rs, 'avg_forgetting'))} & {pm(values(rs, 'bwt'))} & "
                    f"{pct(learned)} & {len(rs)} \\\\")
    if rows:
        header = (TOP + "\n\\textbf{Configuration} & \\textbf{Avg.\\ accuracy} & "
                  "\\textbf{Avg.\\ forgetting} & \\textbf{BWT} & \\textbf{Learned} & "
                  "\\textbf{Seeds} \\\\\n" + MID)
        w.table(fname, "@{}lccccc@{}", header, "\n".join(rows))


def control_table(runs, w):
    """No skipping, decision-guided skipping and random skipping at the same rate."""
    entries = (
        ("ALIF+SI (always update)", dict(block="block1_taskIL", exact="ALIF+SI")),
        ("ALIF+SI+DM (decision-guided)", dict(block="block1_taskIL", exact="ALIF+SI+DM")),
        ("ALIF+SI+Random (matched)", dict(block="block2_control", contains="Random")),
    )
    rows = []
    for label, query in entries:
        key = find(runs, **query)
        if not key:
            continue
        rs = runs[key]
        learned = float(np.mean(values(rs, "compute", "backward_sample_fraction")))
        rows.append(f"{label} & {pm(values(rs, 'avg_acc'))} & "
                    f"{pm(values(rs, 'avg_forgetting'))} & {pct(ignored_share(rs))} & "
                    f"{pct(learned)} & "
                    f"{pm(values(rs, 'compute', 'synops_total_G'), 1)} & "
                    f"{pm(values(rs, 'compute', 'synops_total_event_G'), 1)} \\\\")
    if rows:
        header = (TOP + "\n\\textbf{Configuration} & \\textbf{Avg.\\ accuracy} & "
                  "\\textbf{Avg.\\ forgetting} & \\textbf{Ignored} & \\textbf{Learned} & "
                  "\\textbf{Dense cost} & \\textbf{Event-driven cost} \\\\\n" + MID)
        w.table("control.tex", "@{}lcccccc@{}", header, "\n".join(rows))


def sweep_table(runs, w, block="block3_sweep"):
    """Decision module and random skipping for several target shares."""
    dm, rd = {}, {}
    for key, rs in runs.items():
        b, c = key.split("|", 1)
        if b != block:
            continue
        m = re.search(r"(\d+)", c)
        if m:
            (dm if c.startswith("DM") else rd)[int(m.group(1))] = rs
    if not dm:
        return
    rows = []
    for rate in sorted(set(dm) | set(rd)):
        cells = [f"{rate}\\,\\%"]
        for src in (dm, rd):
            if rate in src:
                rs = src[rate]
                cost = float(np.mean(values(rs, "compute", "synops_total_event_G")))
                cells += [pm(values(rs, "avg_acc")), pm(values(rs, "avg_forgetting")),
                          pct(ignored_share(rs)), f"{cost:.1f}"]
            else:
                cells += ["---"] * 4
        rows.append(" & ".join(cells) + " \\\\")
    header = (TOP + "\n& \\multicolumn{4}{c}{\\textbf{Decision module}} & "
              "\\multicolumn{4}{c}{\\textbf{Random}} \\\\\n"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
              "\\textbf{Target} & Accuracy & Forgetting & Ignored & Cost & "
              "Accuracy & Forgetting & Ignored & Cost \\\\\n" + MID)
    w.table("sweep.tex", "@{}lcccccccc@{}", header, "\n".join(rows))


def compute_table(runs, w):
    """Training cost in giga synaptic operations under both cost models."""
    rows = []
    for name in LADDER + ["ALIF+SI+Random"]:
        if name == "ALIF+SI+Random":
            key = find(runs, block="block2_control", contains="Random")
        else:
            key = find(runs, block="block1_taskIL", exact=name)
        if not key:
            continue
        rs = runs[key]
        learned = float(np.mean(values(rs, "compute", "backward_sample_fraction")))
        rows.append(f"{name} & {pm(values(rs, 'compute', 'synops_total_G'), 1)} & "
                    f"{pm(values(rs, 'compute', 'synops_total_event_G'), 1)} & "
                    f"{pct(learned)} \\\\")
    if rows:
        header = (TOP + "\n\\textbf{Configuration} & \\textbf{Dense (GSynOps)} & "
                  "\\textbf{Event-driven (GSynOps)} & \\textbf{Learned} \\\\\n" + MID)
        w.table("compute.tex", "@{}lccc@{}", header, "\n".join(rows))


def single_run_tables(runs, w):
    """Action shares, firing regularity and neuron roles of one decision-module run."""
    key = find(runs, block="block1_taskIL", exact="ALIF+SI+DM")
    if not key:
        return
    r = runs[key][0]

    if r.get("action_by_task"):
        rows = [" & ".join([f"Task {i + 1}"] + [f"{d.get(a, 0.0) * 100:.1f}" for a in ACTIONS])
                + " \\\\" for i, d in enumerate(r["action_by_task"])]
        g = r["action_global"]
        rows.append(MID)
        rows.append(" & ".join(["overall"] + [f"{g.get(a, 0.0) * 100:.1f}" for a in ACTIONS])
                    + " \\\\")
        header = (TOP + "\n& \\textbf{UPDATE} & \\textbf{CONSOLIDATE} & \\textbf{PROTECT} & "
                  "\\textbf{IGNORE} \\\\\n" + MID)
        w.table("actions.tex", "@{}lcccc@{}", header, "\n".join(rows))

    if r.get("isi_trace"):
        rows = [f"Task {x['task_id'] + 1} & {x['imp_l1_mean']:.4f} & {x['imp_l1_std']:.4f} & "
                f"{x['cv_l1_mean']:.4f} & {pct(x['silent_frac_l1'])} & "
                f"{x['imp_l2_mean']:.4f} \\\\" for x in r["isi_trace"]]
        header = (TOP + "\n& \\textbf{Imp.\\ L1} & \\textbf{s.d.} & \\textbf{Mean CV} & "
                  "\\textbf{Silent} & \\textbf{Imp.\\ L2} \\\\\n" + MID)
        w.table("isi.tex", "@{}lccccc@{}", header, "\n".join(rows))

    if r.get("memory_trace"):
        rows = [f"Task {x['task_id'] + 1} & {x.get('l1_FAST_PLASTIC', 0)} & "
                f"{x.get('l1_ATTENTION_GATE', 0)} & {x.get('l1_STABLE_CONSOL', 0)} & "
                f"{x.get('l1_churn', 0)} & {x.get('l1_n_consolidated', 0)} & "
                f"{x.get('l2_n_consolidated', 0)} & {x.get('new_transitions', 0)} \\\\"
                for x in r["memory_trace"]]
        header = (TOP + "\n& \\textbf{Fast plastic} & \\textbf{Gating} & \\textbf{Stable} & "
                  "\\textbf{Changed} & \\textbf{Consolidated L1} & \\textbf{Consolidated L2} & "
                  "\\textbf{New} \\\\\n" + MID)
        w.table("roles.tex", "@{}lccccccc@{}", header, "\n".join(rows))


def matrix_tables(runs, w):
    """Accuracy matrix of the first run of every configuration."""
    for key, rs in runs.items():
        cfg = re.sub(r"[^A-Za-z0-9]", "", key.split("|", 1)[1])
        A = np.array(rs[0]["acc_matrix"])
        N = A.shape[0]
        rows = [" & ".join([f"after T{i + 1}"]
                           + [(f"{A[i, j]:.4f}" if j <= i else "---") for j in range(N)])
                + " \\\\" for i in range(N)]
        header = (TOP + "\n& " + " & ".join(f"\\textbf{{T{j + 1}}}" for j in range(N))
                  + " \\\\\n" + MID)
        w.table(f"matrix_{cfg}.tex", "@{}l" + "c" * N + "@{}", header, "\n".join(rows))


def number_macros(runs, w):
    """LaTeX macros for the numbers that are quoted in running text."""
    lines = ["% generated by scripts/make_tables.py"]

    def cmd(name, value):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for name, mac in (("LIF", "Lif"), ("ALIF", "Alif"), ("ALIF+SI", "Si"), ("ALIF+SI+DM", "Dm")):
        key = find(runs, block="block1_taskIL", exact=name)
        if not key:
            continue
        a = mean_sd(values(runs[key], "avg_acc"))
        f = mean_sd(values(runs[key], "avg_forgetting"))
        cmd(f"acc{mac}", f"{a[0]:.4f}")
        cmd(f"accSd{mac}", f"{a[1]:.4f}")
        cmd(f"fgt{mac}", f"{f[0]:.4f}")
        cmd(f"fgtSd{mac}", f"{f[1]:.4f}")
    key = find(runs, block="block2_control", contains="Random")
    if key:
        a = mean_sd(values(runs[key], "avg_acc"))
        f = mean_sd(values(runs[key], "avg_forgetting"))
        cmd("accRand", f"{a[0]:.4f}")
        cmd("accSdRand", f"{a[1]:.4f}")
        cmd("fgtRand", f"{f[0]:.4f}")
        cmd("fgtSdRand", f"{f[1]:.4f}")
    key = find(runs, block="block1_taskIL", exact="ALIF+SI+DM")
    if key:
        rs = runs[key]
        g = rs[0]["action_global"]
        for a_ in ACTIONS:
            cmd(f"share{a_.capitalize()}", f"{g.get(a_, 0.0) * 100:.1f}")
        cmd("bwdFracDm", f"{np.mean(values(rs, 'compute', 'backward_sample_fraction')) * 100:.1f}")
        cmd("synopsDenseDm", f"{np.mean(values(rs, 'compute', 'synops_total_G')):.1f}")
        cmd("synopsEventDm", f"{np.mean(values(rs, 'compute', 'synops_total_event_G')):.1f}")
        cmd("nSeeds", str(len(rs)))
    w.text("numbers.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Plain-text summary
# ----------------------------------------------------------------------------
def write_summary(runs, out_dir):
    path = os.path.join(out_dir, "RESULTS_SUMMARY.txt")
    with open(path, "w") as fh:
        fh.write("RESULTS SUMMARY\n" + "=" * 78 + "\n")
        fh.write("mean +- sample standard deviation over seeds\n\n")
        for key in sorted(runs):
            rs = runs[key]
            a = mean_sd(values(rs, "avg_acc"))
            f = mean_sd(values(rs, "avg_forgetting"))
            b = mean_sd(values(rs, "bwt"))
            d = mean_sd(values(rs, "compute", "synops_total_G"))
            e = mean_sd(values(rs, "compute", "synops_total_event_G"))
            learned = float(np.mean(values(rs, "compute", "backward_sample_fraction")))
            fh.write(f"{key}   (n={len(rs)} seeds: {values(rs, 'seed')})\n")
            fh.write(f"   avg accuracy   {a[0]:.4f} +- {a[1]:.4f}\n")
            fh.write(f"   avg forgetting {f[0]:.4f} +- {f[1]:.4f}\n")
            fh.write(f"   BWT            {b[0]:+.4f} +- {b[1]:.4f}\n")
            if len(rs) > 1:
                acc = [round(x, 4) for x in values(rs, "avg_acc")]
                fgt = [round(x, 4) for x in values(rs, "avg_forgetting")]
                fh.write(f"   per seed       accuracy   {acc}\n")
                fh.write(f"                  forgetting {fgt}\n")
            fh.write(f"   final acc      {[round(x, 4) for x in rs[0]['final_acc']]}"
                     f"   (seed {rs[0]['seed']})\n")
            g = rs[0].get("action_global", {})
            if g:
                shares = {k: round(v * 100, 1) for k, v in g.items()}
                fh.write(f"   actions        {shares}   (seed {rs[0]['seed']})\n")
            fh.write(f"   synops (G)     dense={d[0]:.1f} +- {d[1]:.1f}  "
                     f"event={e[0]:.1f} +- {e[1]:.1f}  learned={learned:.1%}\n\n")

        fh.write("COMPARISONS  (Cohen's d with pooled sd, Welch's t; a against b)\n")
        fh.write("-" * 78 + "\n")
        for ka, kb in COMPARISONS:
            if ka not in runs or kb not in runs:
                continue
            if len(runs[ka]) < 2 or len(runs[kb]) < 2:
                fh.write(f"{ka}  vs  {kb}: needs at least two seeds each\n")
                continue
            fh.write(f"{ka}  vs  {kb}\n")
            for metric in ("avg_forgetting", "avg_acc"):
                va, vb = values(runs[ka], metric), values(runs[kb], metric)
                fh.write(f"   {metric:<15} d = {cohens_d(va, vb):+.2f}   "
                         f"t = {welch_t(va, vb):+.2f}\n")
    print(f"\nWrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="output/results", help="directory with the JSON files")
    p.add_argument("--out", default="output", help="output directory")
    args = p.parse_args()

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"No result files in {args.results}. Run run_experiments.py first.")
    print(f"loaded {sum(len(v) for v in runs.values())} runs / {len(runs)} configurations")

    w = Writer(args.out)
    ladder_table(runs, w, "block1_taskIL", "ladder_task_il.tex")
    ladder_table(runs, w, "block4_classIL", "ladder_class_il.tex", suffix="/classIL")
    control_table(runs, w)
    sweep_table(runs, w)
    compute_table(runs, w)
    single_run_tables(runs, w)
    matrix_tables(runs, w)
    number_macros(runs, w)
    write_summary(runs, args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Draw the result figures from the result files.

Reads `<results>/*__s*.json` (written by run_experiments.py) and writes every
figure as PNG and PDF to `<out>/figures/`. A figure is skipped if the runs it
needs are missing. Error bars are sample standard deviations over seeds;
figures that describe a single run use the run with the lowest seed.

Example
    python scripts/make_figures.py --results output/results --out output
"""

import argparse
import glob
import json
import os
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Make the package importable when the script is run from a clone that has not
# been installed with pip.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgcl_snn import mean_sd  # noqa: E402

plt.rcParams.update({"figure.constrained_layout.use": True})

FIGURES = "output/figures"      # set in main()
DPI = 300

C_UPDATE, C_CONSOL, C_PROTECT, C_IGNORE = "#2196F3", "#9C27B0", "#FF9800", "#9E9E9E"
ORDER_IDX = {n: i for i, n in enumerate(
    ["LIF", "ALIF", "ALIF+SI", "ALIF+SI+DM",
     "LIF/classIL", "ALIF/classIL", "ALIF+SI/classIL", "ALIF+SI+DM/classIL"])}
ACOL = {"UPDATE": C_UPDATE, "CONSOLIDATE": C_CONSOL, "PROTECT": C_PROTECT, "IGNORE": C_IGNORE}


def load_runs(results_dir) -> Dict[str, List[Dict]]:
    """Result files grouped by '<block>|<configuration>', sorted by file name."""
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(results_dir, "*__s*.json"))):
        with open(p) as f:
            r = json.load(f)
        block = os.path.basename(p).split("__")[0]
        runs[f"{block}|{r['config']}"].append(r)
    return runs


def ms(runs, key):
    """Mean and sample standard deviation of one metric over the runs."""
    return mean_sd([r[key] for r in runs])


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


# ──────────────────────────────────────────────────────────────────────────
def fig1_ablation(runs, block="block1_taskIL", tag="fig1_ablation_ladder", suffix=""):
    order = ["LIF", "ALIF", "ALIF+SI", "ALIF+SI+DM"]
    keys = [f"{block}|{o}{suffix}" for o in order]
    keys = [k for k in keys if k in runs]
    if not keys:
        return
    names = [k.split("|")[1] for k in keys]
    acc = [ms(runs[k], "avg_acc") for k in keys]
    fgt = [ms(runs[k], "avg_forgetting") for k in keys]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    n_seeds = len(runs[keys[0]])
    ye_a = [a[1] for a in acc] if n_seeds > 1 else None
    ye_f = [v[1] for v in fgt] if n_seeds > 1 else None
    ax[0].bar(x, [a[0] for a in acc], yerr=ye_a,
              capsize=5, color="#2196F3", alpha=.85)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=15)
    ax[0].set_ylabel("Average accuracy")
    ax[0].set_title("Average accuracy after all tasks")
    ax[0].set_ylim(0, 1.05)
    ax[0].grid(axis="y", alpha=.3)
    for i, (m, s) in enumerate(acc):
        ax[0].text(i, m + s + .02, f"{m:.3f}", ha="center", fontsize=9)

    ax[1].bar(x, [v[0] for v in fgt], yerr=ye_f,
              capsize=5, color="#E53935", alpha=.85)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=15)
    ax[1].set_ylabel("Average forgetting")
    ax[1].set_title("Average forgetting (lower is better)")
    ax[1].grid(axis="y", alpha=.3)
    for i, (m, s) in enumerate(fgt):
        ax[1].text(i, m + s + .005, f"{m:.4f}", ha="center", fontsize=9)

    sub = (f"mean $\\pm$ sd over {n_seeds} seeds" if n_seeds > 1
           else f"single run (seed {runs[keys[0]][0]['seed']})")
    fig.suptitle(f"Ablation ladder — {sub}", fontsize=13)
    save(fig, tag)


def fig2_matrices(runs, block="block1_taskIL"):
    keys = [k for k in runs if k.startswith(block + "|")]
    if not keys:
        return
    keys = sorted(keys, key=lambda k: ORDER_IDX.get(k.split("|")[1], 99))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.9))
    axes = np.atleast_1d(axes)
    for ci, (ax, k) in enumerate(zip(axes, keys)):
        A = np.array(runs[k][0]["acc_matrix"])
        M = np.where(np.tril(np.ones_like(A)) > 0, A, np.nan)
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
        N = A.shape[0]
        for i in range(N):
            for j in range(i + 1):
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(k.split("|")[1], fontsize=9.5, pad=8)
        ax.set_xlabel("evaluated on task", fontsize=8.5)
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels([f"T{i+1}" for i in range(N)], fontsize=8)
        if ci == 0:
            ax.set_ylabel("trained through task", fontsize=8.5)
            ax.set_yticklabels([f"T{i+1}" for i in range(N)], fontsize=8)
        else:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=.72, pad=.02)
    cb.set_label("accuracy", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle("Accuracy matrices $A_{i,j}$ (seed 0)", fontsize=11.5)
    save(fig, "fig2_accuracy_matrices")


def fig3_decision_vs_random(runs):
    dm = [k for k in runs if "ALIF+SI+DM" in k and "classIL" not in k]
    rd = [k for k in runs if "Random" in k]
    si = [k for k in runs if k.endswith("|ALIF+SI")]
    if not (dm and rd):
        return
    keys = si[:1] + dm[:1] + rd[:1]
    names = ["ALIF+SI\n(always update)", "ALIF+SI+DM\n(decision-guided)",
             "ALIF+SI+Random\n(matched rate)"]
    acc = [ms(runs[k], "avg_acc") for k in keys]
    fgt = [ms(runs[k], "avg_forgetting") for k in keys]
    cols = ["#78909C", "#2196F3", "#FF9800"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(keys))
    for a, vals, ttl, yl in ((ax[0], acc, "Average accuracy", "accuracy"),
                             (ax[1], fgt, "Average forgetting", "forgetting")):
        a.bar(x, [v[0] for v in vals], yerr=[v[1] for v in vals],
              capsize=6, color=cols, alpha=.88)
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=9)
        a.set_title(ttl)
        a.set_ylabel(yl)
        a.grid(axis="y", alpha=.3)
        for i, (m, s) in enumerate(vals):
            a.text(i, m + s + .01 * max(1, m), f"{m:.4f}", ha="center", fontsize=9)
    fig.suptitle("No skipping, decision-guided skipping and random skipping at the same rate",
                 fontsize=12)
    save(fig, "fig3_decision_vs_random")


def fig4_sweep(runs, block="block3_sweep"):
    dm, rd = {}, {}
    for k, v in runs.items():
        if not k.startswith(block + "|"):
            continue
        name = k.split("|")[1]
        rate = int("".join(ch for ch in name if ch.isdigit()))
        (dm if name.startswith("DM") else rd)[rate] = v
    if not dm:
        return
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for src, lbl, col in ((dm, "Decision module", "#2196F3"),
                          (rd, "Random skip", "#FF9800")):
        if not src:
            continue
        rs = sorted(src)
        for a, key, yl in ((ax[0], "avg_acc", "average accuracy"),
                           (ax[1], "avg_forgetting", "average forgetting")):
            m = [ms(src[r], key)[0] for r in rs]
            s = [ms(src[r], key)[1] for r in rs]
            a.errorbar(rs, m, yerr=s, marker="o", capsize=4, label=lbl, color=col, lw=2)
            a.set_xlabel("target IGNORE rate (%)")
            a.set_ylabel(yl)
            a.grid(alpha=.3)
            a.legend()
        c = [np.mean([x["compute"]["synops_total_event_G"] for x in src[r]]) for r in rs]
        ax[2].plot(rs, c, "o-", label=lbl, color=col, lw=2)
    ax[0].set_title("Accuracy vs skip rate")
    ax[1].set_title("Forgetting vs skip rate")
    ax[2].set_title("Event-driven SynOps vs skip rate")
    ax[2].set_xlabel("target IGNORE rate (%)")
    ax[2].set_ylabel("GSynOps")
    ax[2].grid(alpha=.3)
    ax[2].legend()
    fig.suptitle("Accuracy, forgetting and cost against the target share of ignored samples",
                 fontsize=13)
    save(fig, "fig4_skip_rate_tradeoff")


def fig5_actions(runs):
    key = next((k for k in runs if "ALIF+SI+DM" in k), None)
    if key is None:
        return
    r = runs[key][0]
    per_task = r.get("action_by_task", [])
    if not per_task:
        return
    names = ["UPDATE", "CONSOLIDATE", "PROTECT", "IGNORE"]
    N = len(per_task)
    x = np.arange(N)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    bottom = np.zeros(N)
    for n in names:
        v = np.array([d.get(n, 0.0) for d in per_task])
        ax[0].bar(x, v, .6, bottom=bottom, label=n, color=ACOL[n], alpha=.88)
        bottom += v
        ax[1].plot(x + 1, v, "o-", label=n, color=ACOL[n], lw=2)
    for a in ax:
        a.grid(axis="y", alpha=.3)
        a.legend(fontsize=9)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"Task {i+1}" for i in range(N)])
    ax[0].set_ylabel("share of samples")
    ax[0].set_title("Action distribution within each task")
    ax[1].set_xlabel("task")
    ax[1].set_ylabel("share of samples")
    ax[1].set_title("Action trends across the task sequence")
    fig.suptitle("Share of each action, counted per task", fontsize=12)
    save(fig, "fig5_action_distribution")


def fig6_compute(runs, block="block1_taskIL"):
    keys = [k for k in runs if k.startswith(block + "|")]
    keys += [k for k in runs if "Random" in k]
    if not keys:
        return
    names = [k.split("|")[1] for k in keys]
    dense = [np.mean([r["compute"]["synops_total_G"] for r in runs[k]]) for k in keys]
    event = [np.mean([r["compute"]["synops_total_event_G"] for r in runs[k]]) for k in keys]
    x = np.arange(len(keys))
    w = .38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, dense, w, label="Dense (GPU) cost", color="#78909C")
    ax.bar(x + w/2, event, w, label="Event-driven (neuromorphic) cost", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("GSynOps")
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    ax.set_title("Training cost under the dense and the event-driven cost model", fontsize=11)
    save(fig, "fig6_compute")


def fig7_signals(runs):
    key = next((k for k in runs if "ALIF+SI+DM" in k), None)
    if key is None:
        return
    log = runs[key][0].get("epoch_logs", [])
    if not log:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    tasks = sorted({e["task_id"] for e in log})
    for t in tasks:
        e = [x for x in log if x["task_id"] == t]
        ax[0].plot([x["epoch"] for x in e], [x["entropy"] for x in e],
                   "o-", label=f"Task {t+1}")
        ax[1].plot([x["epoch"] for x in e], [x["active_frac"] for x in e],
                   "o-", label=f"Task {t+1}")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("mean predictive entropy")
    ax[0].set_title("Mean predictive entropy per epoch")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("fraction of samples with non-zero weight")
    ax[1].set_title("Share of samples contributing gradient")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    save(fig, "fig7_signals")


def fig8_isi(runs):
    key = next((k for k in runs if "ALIF+SI+DM" in k), None)
    if key is None:
        return
    tr = runs[key][0].get("isi_trace", [])
    if not tr:
        return
    t = [x["task_id"] + 1 for x in tr]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].errorbar(t, [x["imp_l1_mean"] for x in tr], yerr=[x["imp_l1_std"] for x in tr],
                   marker="o", capsize=4, label="Layer 1", lw=2)
    ax[0].plot(t, [x["imp_l2_mean"] for x in tr], "s-", label="Layer 2", lw=2)
    ax[0].set_xlabel("after task")
    ax[0].set_ylabel("ISI-CV importance")
    ax[0].set_title("ISI-CV importance (mean ± sd across neurons)")
    ax[1].plot(t, [x["silent_frac_l1"] for x in tr], "o-", color="#E53935", lw=2)
    ax[1].set_xlabel("after task")
    ax[1].set_ylabel("fraction of neurons")
    ax[1].set_title("Neurons with too few spikes for a valid ISI-CV\n"
                    "(assigned importance 0)")
    ax[0].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=.3)
    save(fig, "fig8_isi_importance")


def fig9_omega(runs):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for k, rs in runs.items():
        tr = rs[0].get("omega_trace", [])
        if not tr:
            continue
        ax.plot(range(1, len(tr) + 1), tr, "o-", label=k.split("|")[1], lw=2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("after task")
    ax.set_ylabel("mean Omega")
    ax.set_title("Synaptic Intelligence importance accumulation")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    save(fig, "fig9_omega_growth")




# ── Figure 10 — neuron roles (analysis only) ──────────────────────────────
C_FAST, C_GATE, C_STABLE = "#2196F3", "#9C27B0", "#E67E22"
ROLE_LABELS = {"FAST_PLASTIC": "Fast plastic", "ATTENTION_GATE": "Gating",
               "STABLE_CONSOL": "Stable"}


def fig10_memory_states(runs):
    key = next((k for k in runs if "ALIF+SI+DM" in k and "classIL" not in k), None)
    if key is None:
        return
    trace = runs[key][0].get("memory_trace", [])
    final = runs[key][0].get("memory_states_final")
    if not trace:
        return

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = ["FAST_PLASTIC", "ATTENTION_GATE", "STABLE_CONSOL"]
    cols = [C_FAST, C_GATE, C_STABLE]
    T = [r["task_id"] + 1 for r in trace]

    # (a) evolution of the three roles in layer 1
    bottom = np.zeros(len(T))
    for n, c in zip(names, cols):
        v = np.array([r.get(f"l1_{n}", 0) for r in trace], dtype=float)
        ax[0].bar(T, v, .62, bottom=bottom, label=ROLE_LABELS[n], color=c, alpha=.9)
        bottom += v
    ax[0].set_xlabel("after task")
    ax[0].set_ylabel("neurons (layer 1)")
    ax[0].set_title("Neuron roles after each task", fontsize=10)
    ax[0].set_xticks(T)
    ax[0].legend(fontsize=7.5, frameon=False)

    # (b) newly consolidated neurons
    newt = [r["new_transitions"] for r in trace]
    cum = np.cumsum(newt)
    ax[1].bar(T, newt, .55, color=C_STABLE, alpha=.85, label="new this task")
    ax[1].plot(T, cum, "o-", color="#2c3e50", lw=1.8, label="cumulative")
    ax[1].set_xlabel("after task")
    ax[1].set_ylabel("neurons consolidated")
    ax[1].set_title("Newly consolidated neurons (both layers)", fontsize=10)
    ax[1].set_xticks(T)
    ax[1].legend(fontsize=7.5, frameon=False)

    # (c) roles after the last task, layer 1
    if final and final.get("l1", {}).get("states"):
        st = np.array(final["l1"]["states"])
        side = int(np.ceil(np.sqrt(st.size)))
        pad = np.full(side * side, -1)
        pad[: st.size] = st
        cmap = matplotlib.colors.ListedColormap(["#eceff1", C_FAST, C_GATE, C_STABLE])
        ax[2].imshow(pad.reshape(side, side), cmap=cmap, vmin=-1.5, vmax=2.5)
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title("Roles after the last task (layer 1)", fontsize=10)
        handles = [plt.Line2D([], [], marker="s", ls="", ms=8, color=c,
                              label=ROLE_LABELS[n])
                   for n, c in zip(names, cols)]
        ax[2].legend(handles=handles, fontsize=7, frameon=False,
                     loc="upper center", bbox_to_anchor=(.5, -.04), ncol=3)
    for a in (ax[0], ax[1]):
        a.grid(axis="y", alpha=.25)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    save(fig, "fig10_memory_states")


# ── Figure 11 — combined importance: SI vs ISI-CV ─────────────────────────
def fig11_importance_components(runs):
    key = next((k for k in runs if "ALIF+SI+DM" in k and "classIL" not in k), None)
    if key is None:
        return
    trace = runs[key][0].get("memory_trace", [])
    isi = runs[key][0].get("isi_trace", [])
    if not trace or not isi:
        return
    T = [r["task_id"] + 1 for r in trace]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))

    ax[0].errorbar(T, [r["l1_imp_mean"] for r in trace],
                   yerr=[r["l1_imp_std"] for r in trace],
                   marker="o", capsize=4, lw=2, color="#2c6ca0",
                   label="combined $\\mathcal{I}$, layer 1")
    ax[0].plot(T, [x["imp_l1_mean"] for x in isi], "s--", lw=1.6,
               color="#2e8b57", label="ISI-CV component")
    ax[0].set_xlabel("after task")
    ax[0].set_ylabel("importance")
    ax[0].set_title("Combined importance and its firing-regularity component", fontsize=10)
    ax[0].set_xticks(T)
    ax[0].legend(fontsize=7.5, frameon=False)

    ax[1].plot(T, [r["l1_n_consolidated"] for r in trace], "o-", lw=2,
               color=C_STABLE, label="layer 1")
    ax[1].plot(T, [r["l2_n_consolidated"] for r in trace], "s-", lw=2,
               color="#8e44ad", label="layer 2")
    ax[1].set_xlabel("after task")
    ax[1].set_ylabel("neurons in consolidated set")
    ax[1].set_title("Size of the consolidated set", fontsize=10)
    ax[1].set_xticks(T)
    ax[1].legend(fontsize=7.5, frameon=False)
    for a in ax:
        a.grid(alpha=.25)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    save(fig, "fig11_importance_components")



# ══════════════════════════════════════════════════════════════════════════
#  Diagnostic figures, from the evaluation pass over the trained model
# ══════════════════════════════════════════════════════════════════════════

def _diag(runs, want="ALIF+SI+DM"):
    for k in runs:
        if want in k and "classIL" not in k:
            d = runs[k][0].get("diagnostics")
            if d:
                return d
    for k in runs:
        d = runs[k][0].get("diagnostics")
        if d:
            return d
    return None


# ── Figure 12 — dataset and spike encoding ────────────────────────────────
def fig12_dataset_encoding(runs):
    d = _diag(runs)
    if not d or not d.get("samples"):
        return
    S = d["samples"]
    n = len(S)
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [1.9], hspace=.35, wspace=.25)

    for i, s in enumerate(S):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.array(s["image"]), cmap="gray_r")
        ax.set_title(f"Task {s['task_id']+1}\n{s['digits'][0]} vs {s['digits'][1]}", fontsize=9)
        ax.axis("off")
        ax2 = fig.add_subplot(gs[1, i])
        R = np.array(s["input_raster"])
        ax2.imshow(R.T, cmap="Greys", aspect="auto", interpolation="nearest")
        ax2.set_xlabel("timestep", fontsize=7.5)
        if i == 0:
            ax2.set_ylabel("input channel", fontsize=7.5)
        ax2.tick_params(labelsize=6.5)

    ax3 = fig.add_subplot(gs[:, n])
    R = np.array(S[0]["input_raster"])
    rate = R.mean(0)
    ax3.hist(rate, bins=28, color="#2c6ca0", alpha=.85)
    ax3.set_xlabel("spike probability per channel", fontsize=8)
    ax3.set_ylabel("count", fontsize=8)
    ax3.set_title("Poisson encoding\nrate distribution", fontsize=9)
    ax3.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)
    fig.suptitle("Split-MNIST tasks and their Poisson spike encoding", fontsize=11)
    save(fig, "fig12_dataset_encoding")


# ── Figure 13 — spike activity in the trained network ─────────────────────
def fig13_spike_activity(runs):
    d = _diag(runs)
    if not d or not d.get("samples"):
        return
    s = d["samples"][-1]
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    for a, key, ttl in ((ax[0, 0], "raster_l1", "Hidden layer 1"),
                        (ax[0, 1], "raster_l2", "Hidden layer 2")):
        R = np.array(s[key])
        tt, nn = np.where(R > 0)
        a.scatter(tt, nn, s=5, color="#c0392b", marker="|")
        a.set_xlabel("timestep", fontsize=8.5)
        a.set_ylabel("neuron", fontsize=8.5)
        a.set_title(f"{ttl} — spike raster ({R.mean():.1%} active)", fontsize=9.5)
        a.set_xlim(-.5, R.shape[0] - .5)

    pt = d.get("per_task", [])
    if pt:
        for p in pt:
            ax[1, 0].hist(p["rate_l1"], bins=32, alpha=.45, label=f"T{p['task_id']+1}")
        ax[1, 0].set_xlabel("per-sample firing rate, layer 1", fontsize=8.5)
        ax[1, 0].set_ylabel("count", fontsize=8.5)
        ax[1, 0].set_title("Firing-rate distribution by task", fontsize=9.5)
        ax[1, 0].legend(fontsize=7, frameon=False, ncol=2)

    mh = d.get("membrane_hist_l1")
    if mh:
        ax[1, 1].hist(mh, bins=60, color="#2c6ca0", alpha=.85)
        ax[1, 1].axvline(0.3, color="#c0392b", ls="--", lw=1.4, label="threshold $v_{th0}$")
        ax[1, 1].set_xlabel("membrane potential, layer 1", fontsize=8.5)
        ax[1, 1].set_ylabel("count", fontsize=8.5)
        ax[1, 1].set_title("Membrane potential distribution", fontsize=9.5)
        ax[1, 1].legend(fontsize=7.5, frameon=False)
    for a in ax.ravel():
        a.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    save(fig, "fig13_spike_activity")


# ── Figure 14 — ALIF threshold dynamics, trained model ────────────────────
def fig14_threshold_dynamics(runs):
    d = _diag(runs)
    if not d or not d.get("threshold_trace"):
        return
    vth = np.array(d["threshold_trace"])
    v = np.array(d["membrane_trace"])
    r = np.array(d["rate_trace"])
    T = np.arange(len(vth))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(T, vth, "o-", color="#c0392b", lw=2, label=r"adaptive threshold $v_{th}(t)$")
    ax[0].plot(T, v, "s-", color="#2c6ca0", lw=1.6, label=r"mean membrane $v(t)$")
    ax[0].axhline(vth[0], color="#7f8c8d", ls=":", lw=1.2, label="initial threshold")
    ax[0].set_xlabel("timestep", fontsize=8.5)
    ax[0].set_ylabel("potential", fontsize=8.5)
    ax[0].set_title("Threshold adapts as the network fires", fontsize=9.5)
    ax[0].legend(fontsize=7.5, frameon=False)
    ax[1].bar(T, r, .65, color="#2e8b57", alpha=.85)
    ax[1].set_xlabel("timestep", fontsize=8.5)
    ax[1].set_ylabel("fraction firing", fontsize=8.5)
    ax[1].set_title("Population firing rate over the window", fontsize=9.5)
    for a in ax:
        a.grid(alpha=.25)
        a.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    save(fig, "fig14_threshold_dynamics")


# ── Figure 15 — uncertainty as a correctness signal ───────────────────────
def fig15_uncertainty(runs):
    d = _diag(runs)
    if not d or not d.get("per_task"):
        return
    pt = d["per_task"]
    allok = np.concatenate([p["entropy_correct"] for p in pt if p["entropy_correct"]] or [[0]])
    allbad = np.concatenate([p["entropy_wrong"] for p in pt if p["entropy_wrong"]] or [[0]])
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))

    ax[0].hist(allok, bins=44, alpha=.75, color="#2e8b57",
               label=f"correct (n={allok.size})", density=True)
    if allbad.size > 1:
        ax[0].hist(allbad, bins=44, alpha=.75, color="#c0392b",
                   label=f"wrong (n={allbad.size})", density=True)
    ax[0].set_xlabel("predictive entropy $H_i$", fontsize=8.5)
    ax[0].set_ylabel("density", fontsize=8.5)
    ax[0].set_title(f"Correct vs incorrect\nmeans {allok.mean():.4f} / "
                    f"{allbad.mean() if allbad.size else 0:.4f}", fontsize=9.5)
    ax[0].legend(fontsize=7.5, frameon=False)

    T = [p["task_id"] + 1 for p in pt]
    ax[1].plot(T, [p["entropy_correct_mean"] for p in pt], "o-", color="#2e8b57", lw=2,
               label="correct")
    ax[1].plot(T, [p["entropy_wrong_mean"] for p in pt], "s-", color="#c0392b", lw=2,
               label="wrong")
    ax[1].set_xlabel("task", fontsize=8.5)
    ax[1].set_ylabel("mean entropy", fontsize=8.5)
    ax[1].set_title("Separation holds across all tasks", fontsize=9.5)
    ax[1].set_xticks(T)
    ax[1].legend(fontsize=7.5, frameon=False)

    data = [p["entropy_correct"] for p in pt]
    bp = ax[2].boxplot(data, tick_labels=[f"T{t}" for t in T], patch_artist=True, widths=.6)
    for b in bp["boxes"]:
        b.set_facecolor("#dce9f5")
        b.set_edgecolor("#2c6ca0")
    ax[2].set_ylabel("entropy (correct predictions)", fontsize=8.5)
    ax[2].set_title("Confidence per task", fontsize=9.5)
    for a in ax:
        a.grid(alpha=.25)
        a.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    save(fig, "fig15_uncertainty_diagnostics")


# ── Figure 16 — attention signal ──────────────────────────────────────────
def fig16_attention(runs):
    d = _diag(runs)
    if not d or not d.get("per_task"):
        return
    pt = d["per_task"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for p in pt:
        ax[0].hist(p["attention"], bins=40, alpha=.4, label=f"T{p['task_id']+1}")
    allatt = np.concatenate([p["attention"] for p in pt])
    tau = np.quantile(allatt, .25)
    ax[0].axvline(tau, color="#22313f", ls="--", lw=1.5, label=r"$\tau_a$ (p25)")
    ax[0].set_xlabel("attention $a_i$", fontsize=8.5)
    ax[0].set_ylabel("count", fontsize=8.5)
    ax[0].set_title("Attention distribution by task", fontsize=9.5)
    ax[0].legend(fontsize=6.5, frameon=False, ncol=2)

    T = [p["task_id"] + 1 for p in pt]
    ax[1].bar(T, [float(np.mean(np.array(p["attention"]) <= tau)) for p in pt],
              .6, color="#95a5a6", alpha=.9)
    ax[1].axhline(.25, color="#c0392b", ls=":", lw=1.4, label="target 25%")
    ax[1].set_xlabel("task", fontsize=8.5)
    ax[1].set_ylabel("fraction below $\\tau_a$", fontsize=8.5)
    ax[1].set_title("Test samples below the pooled 25 % quantile", fontsize=9.5)
    ax[1].set_xticks(T)
    ax[1].legend(fontsize=7.5, frameon=False)

    # The result files store attention and firing rates in sample order, but
    # entropy only split into correct and incorrect predictions. The entropy of
    # each sample is therefore recovered from a_i = H_i * phi(rate_i).
    H, A = [], []
    for p in pt:
        att = np.asarray(p["attention"], dtype=float)
        rate = 0.5 * (np.asarray(p["rate_l1"], dtype=float)
                      + np.asarray(p["rate_l2"], dtype=float))
        phi = np.clip((rate - 0.05) / 0.90, 0.0, 1.0)
        valid = phi > 0
        H.append(att[valid] / phi[valid])
        A.append(att[valid])
    H, A = np.concatenate(H), np.concatenate(A)
    ax[2].scatter(H, A, s=3, alpha=.25, color="#2c6ca0")
    if H.size > 2:
        rho = float(np.corrcoef(H, A)[0, 1])
        ax[2].set_title(f"Attention vs entropy  (r = {rho:.3f})", fontsize=9.5)
    ax[2].set_xlabel("entropy $H_i$", fontsize=8.5)
    ax[2].set_ylabel("attention $a_i$", fontsize=8.5)
    for a in ax:
        a.grid(alpha=.25)
        a.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    save(fig, "fig16_ras_attention")


# ── Figure 17 — Synaptic Intelligence importance maps ─────────────────────
def fig17_omega_maps(runs):
    d = _diag(runs)
    if not d:
        return
    m1, m2 = d.get("omega_map_l1"), d.get("omega_map_l2")
    n1 = d.get("omega_neuron_l1")
    if m1 is None and n1 is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    if m1 is not None:
        im = ax[0].imshow(np.array(m1), cmap="viridis", aspect="auto")
        ax[0].set_title(r"$\Omega$ map — layer 1 (subsampled)", fontsize=9.5)
        ax[0].set_xlabel("input", fontsize=8.5)
        ax[0].set_ylabel("neuron", fontsize=8.5)
        fig.colorbar(im, ax=ax[0], shrink=.85)
    if m2 is not None:
        im = ax[1].imshow(np.array(m2), cmap="viridis", aspect="auto")
        ax[1].set_title(r"$\Omega$ map — layer 2", fontsize=9.5)
        ax[1].set_xlabel("input neuron", fontsize=8.5)
        fig.colorbar(im, ax=ax[1], shrink=.85)
    if n1 is not None:
        v = np.sort(np.array(n1))[::-1]
        ax[2].plot(v, lw=2, color="#8e44ad")
        ax[2].fill_between(range(len(v)), v, alpha=.3, color="#8e44ad")
        ax[2].set_xlabel("neuron (sorted)", fontsize=8.5)
        ax[2].set_ylabel(r"normalised $\Omega$", fontsize=8.5)
        ax[2].set_title("Normalised importance per neuron, layer 1", fontsize=9.5)
        ax[2].grid(alpha=.25)
        for sp in ("top", "right"):
            ax[2].spines[sp].set_visible(False)
    for a in ax:
        a.tick_params(labelsize=7.5)
    save(fig, "fig17_omega_maps")


# ── Figure 18 — ISI-CV importance map ─────────────────────────────────────
def fig18_isi_map(runs):
    d = _diag(runs)
    if not d or not d.get("isi_importance_l1"):
        return
    i1 = np.array(d["isi_importance_l1"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.7))
    side = int(np.ceil(np.sqrt(i1.size)))
    pad = np.full(side * side, np.nan)
    pad[:i1.size] = i1
    im = ax[0].imshow(pad.reshape(side, side), cmap="magma", vmin=0, vmax=1)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("ISI-CV importance map — layer 1", fontsize=9.5)
    fig.colorbar(im, ax=ax[0], shrink=.85)

    ax[1].hist(i1[i1 > 0], bins=34, color="#e67e22", alpha=.85, label="valid")
    n_sil = int((i1 == 0).sum())
    ax[1].axvline(i1[i1 > 0].mean() if (i1 > 0).any() else 0, color="#22313f",
                  ls="--", lw=1.4, label="mean")
    ax[1].set_xlabel(r"$1/(1+\mathrm{CV})$", fontsize=8.5)
    ax[1].set_ylabel("neurons", fontsize=8.5)
    ax[1].set_title(f"Importance distribution\n{n_sil} silent neurons scored 0", fontsize=9.5)
    ax[1].legend(fontsize=7.5, frameon=False)

    om = d.get("omega_neuron_l1")
    if om:
        o = np.array(om)[:i1.size]
        ax[2].scatter(o, i1[:o.size], s=12, alpha=.55, color="#2c6ca0")
        if o.size > 2 and o.std() > 0 and i1[:o.size].std() > 0:
            rho = float(np.corrcoef(o, i1[:o.size])[0, 1])
            ax[2].set_title(f"Gradient vs spiking importance  (r = {rho:.3f})", fontsize=9.5)
        ax[2].set_xlabel(r"SI $\Omega$ (normalised)", fontsize=8.5)
        ax[2].set_ylabel("ISI-CV importance", fontsize=8.5)
        ax[2].grid(alpha=.25)
        for sp in ("top", "right"):
            ax[2].spines[sp].set_visible(False)
    for a in ax:
        a.tick_params(labelsize=7.5)
    save(fig, "fig18_isi_cv_map")


# ── Figure 19 — retention vs cost (Pareto view) ───────────────────────────
def fig19_pareto(runs):
    pts = []
    for k, rs in runs.items():
        if "sweep" in k:
            continue
        name = k.split("|")[1]
        if "classIL" in name:
            continue
        pts.append((name,
                    float(np.mean([r["avg_forgetting"] for r in rs])),
                    float(np.mean([r["compute"]["synops_total_event_G"] for r in rs])),
                    float(np.mean([r["avg_acc"] for r in rs]))))
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    cols = plt.cm.tab10(np.linspace(0, 1, len(pts)))
    for (n, f, c, a), col in zip(pts, cols):
        ax[0].scatter(c, f, s=140, color=col, edgecolor="white", lw=1.2, zorder=5)
        ax[0].annotate(n, (c, f), xytext=(7, 7), textcoords="offset points", fontsize=7.5)
        ax[1].scatter(c, a, s=140, color=col, edgecolor="white", lw=1.2, zorder=5)
        ax[1].annotate(n, (c, a), xytext=(7, 7), textcoords="offset points", fontsize=7.5)
    ax[0].set_xlabel("event-driven cost (GSynOps)", fontsize=8.5)
    ax[0].set_ylabel("average forgetting", fontsize=8.5)
    ax[0].set_title("Retention against cost\n(bottom-left is better)", fontsize=9.5)
    ax[1].set_xlabel("event-driven cost (GSynOps)", fontsize=8.5)
    ax[1].set_ylabel("average accuracy", fontsize=8.5)
    ax[1].set_title("Accuracy against cost\n(top-left is better)", fontsize=9.5)
    for a in ax:
        a.grid(alpha=.25)
        a.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    save(fig, "fig19_retention_vs_cost")


# ── Figure 20 — per-task forgetting breakdown ─────────────────────────────
def fig20_per_task_forgetting(runs):
    keys = [k for k in runs if k.startswith("block1_taskIL|")]
    keys = sorted(keys, key=lambda k: ORDER_IDX.get(k.split("|")[1], 99))
    if not keys:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    W = .8 / max(len(keys), 1)
    for i, k in enumerate(keys):
        pf = np.mean([r["per_task_forgetting"] for r in runs[k]], axis=0)
        x = np.arange(len(pf)) + i * W
        ax[0].bar(x, pf, W, label=k.split("|")[1], alpha=.9)
        fa = np.mean([r["final_acc"] for r in runs[k]], axis=0)
        ax[1].plot(range(1, len(fa) + 1), fa, "o-", lw=2, label=k.split("|")[1])
    ax[0].set_xticks(np.arange(len(pf)) + .4 - W / 2)
    ax[0].set_xticklabels([f"T{i+1}" for i in range(len(pf))])
    ax[0].set_ylabel("forgetting", fontsize=8.5)
    ax[0].set_title("Forgetting per task", fontsize=9.5)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].set_xlabel("task", fontsize=8.5)
    ax[1].set_ylabel("final accuracy", fontsize=8.5)
    ax[1].set_title("Final accuracy per task", fontsize=9.5)
    ax[1].legend(fontsize=7, frameon=False)
    for a in ax:
        a.grid(alpha=.25)
        a.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    save(fig, "fig20_per_task_forgetting")


# ── Figure 21 — overview of one decision-module run ───────────────────────
def fig21_dashboard(runs):
    d = _diag(runs)
    key = next((k for k in runs if "ALIF+SI+DM" in k and "classIL" not in k), None)
    if d is None or key is None:
        return
    r = runs[key][0]
    fig = plt.figure(figsize=(15, 9.0))
    gs = fig.add_gridspec(3, 3, hspace=.38, wspace=.26)
    pt = d.get("per_task", [])
    T = [p["task_id"] + 1 for p in pt]

    a = fig.add_subplot(gs[0, 0])
    A = np.array(r["acc_matrix"])
    N = A.shape[0]
    a.imshow(np.where(np.tril(np.ones_like(A)) > 0, A, np.nan), cmap="RdYlGn", vmin=.5, vmax=1)
    for i in range(N):
        for j in range(i + 1):
            a.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=6)
    a.set_title("Retention matrix", fontsize=9.5)
    a.set_xticks(range(N))
    a.set_yticks(range(N))
    a.set_xticklabels([f"T{i+1}" for i in range(N)], fontsize=7)
    a.set_yticklabels([f"T{i+1}" for i in range(N)], fontsize=7)

    a = fig.add_subplot(gs[0, 1])
    a.plot(T, [p["entropy_correct_mean"] for p in pt], "o-", color="#2e8b57", lw=2, label="correct")
    a.plot(T, [p["entropy_wrong_mean"] for p in pt], "s-", color="#c0392b", lw=2, label="wrong")
    a.set_title("Uncertainty signal", fontsize=9.5)
    a.legend(fontsize=6.5, frameon=False)
    a.set_xticks(T)

    a = fig.add_subplot(gs[0, 2])
    for p in pt:
        a.hist(p["attention"], bins=26, alpha=.4)
    a.set_title("Attention", fontsize=9.5)

    a = fig.add_subplot(gs[1, 0])
    ot = r.get("omega_trace", [])
    if ot:
        a.plot(range(1, len(ot) + 1), ot, "o-", color="#8e44ad", lw=2)
    a.set_title(r"SI importance $\bar\Omega$", fontsize=9.5)

    a = fig.add_subplot(gs[1, 1])
    it = r.get("isi_trace", [])
    if it:
        a.errorbar([x["task_id"] + 1 for x in it], [x["imp_l1_mean"] for x in it],
                   yerr=[x["imp_l1_std"] for x in it], marker="o", capsize=3,
                   color="#e67e22", lw=2)
    a.set_title("ISI-CV importance", fontsize=9.5)

    a = fig.add_subplot(gs[1, 2])
    ab = r.get("action_by_task", [])
    if ab:
        bottom = np.zeros(len(ab))
        for nme, c in zip(["UPDATE", "CONSOLIDATE", "PROTECT", "IGNORE"],
                          [C_UPDATE, C_CONSOL, C_PROTECT, C_IGNORE]):
            v = np.array([x.get(nme, 0.0) for x in ab])
            a.bar(range(1, len(ab) + 1), v, .6, bottom=bottom, color=c, label=nme, alpha=.9)
            bottom += v
        a.legend(fontsize=5.5, frameon=False, ncol=2)
    a.set_title("Plasticity actions", fontsize=9.5)

    a = fig.add_subplot(gs[2, 0])
    mt = r.get("memory_trace", [])
    if mt:
        bottom = np.zeros(len(mt))
        for nme, c in zip(["FAST_PLASTIC", "ATTENTION_GATE", "STABLE_CONSOL"],
                          ["#2196F3", "#9C27B0", "#E67E22"]):
            v = np.array([x.get(f"l1_{nme}", 0) for x in mt], dtype=float)
            a.bar(range(1, len(mt) + 1), v, .6, bottom=bottom, color=c, alpha=.9)
            bottom += v
    a.set_title("Neuron roles", fontsize=9.5)

    a = fig.add_subplot(gs[2, 1])
    if mt:
        a.plot(range(1, len(mt) + 1), np.cumsum([x["new_transitions"] for x in mt]),
               "o-", color="#22313f", lw=2)
    a.set_title("Consolidated neurons (cumulative)", fontsize=9.5)

    a = fig.add_subplot(gs[2, 2])
    if pt:
        a.plot(T, [float(np.mean(p["rate_l1"])) for p in pt], "o-", lw=2, label="layer 1")
        a.plot(T, [float(np.mean(p["rate_l2"])) for p in pt], "s-", lw=2, label="layer 2")
        a.legend(fontsize=6.5, frameon=False)
    a.set_title("Firing rate", fontsize=9.5)

    for a in fig.get_axes():
        a.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        # tasks are integers; suppress fractional ticks such as 1.5, 2.5
        xt = a.get_xticks()
        if len(xt) and float(np.max(xt)) <= len(T) + 1 and a.get_images() == []:
            a.set_xticks(range(1, len(T) + 1))
            a.set_xticklabels([str(i) for i in range(1, len(T) + 1)])
        a.grid(alpha=.2)
    fig.suptitle("Overview of one decision-module run across the task sequence",
                 fontsize=11.5)
    save(fig, "fig21_dashboard")


FIGURE_FUNCTIONS = [
    fig1_ablation,
    lambda runs: fig1_ablation(runs, "block4_classIL", "fig1b_ablation_classIL", "/classIL"),
    fig2_matrices,
    fig3_decision_vs_random,
    fig4_sweep,
    fig5_actions,
    fig6_compute,
    fig7_signals,
    fig8_isi,
    fig9_omega,
    fig10_memory_states,
    fig11_importance_components,
    fig12_dataset_encoding,
    fig13_spike_activity,
    fig14_threshold_dynamics,
    fig15_uncertainty,
    fig16_attention,
    fig17_omega_maps,
    fig18_isi_map,
    fig19_pareto,
    fig20_per_task_forgetting,
    fig21_dashboard,
]


def main():
    global FIGURES
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="output/results", help="directory with the JSON files")
    p.add_argument("--out", default="output", help="output directory")
    args = p.parse_args()
    FIGURES = os.path.join(args.out, "figures")

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"No result files in {args.results}. Run run_experiments.py first.")
    print(f"Loaded {sum(len(v) for v in runs.values())} runs "
          f"across {len(runs)} configurations.\n")
    for draw in FIGURE_FUNCTIONS:
        draw(runs)
    print(f"\nFigures written to {FIGURES}/")


if __name__ == "__main__":
    main()

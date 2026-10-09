"""Continual-learning metrics and statistics over seeds."""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def cl_metrics(A: np.ndarray) -> Dict:
    """Continual-learning metrics from the accuracy matrix.

    A[i, j] is the accuracy on task j after training on tasks 0..i (lower
    triangle).

        avg_acc        = mean_j A[N-1, j]
        forgetting_j   = max_{j <= i < N-1} A[i, j] - A[N-1, j]
        bwt            = mean_{j < N-1} (A[N-1, j] - A[j, j])
    """
    N = A.shape[0]
    final = A[N - 1, :]
    avg_acc = float(final.mean())
    forg = []
    for j in range(N - 1):
        prev = A[j:N - 1, j]
        forg.append(float(prev.max() - final[j]) if prev.size else 0.0)
    bwt = float(np.mean([A[N - 1, j] - A[j, j] for j in range(N - 1)])) if N > 1 else 0.0
    return {
        "avg_acc": avg_acc,
        "avg_forgetting": float(np.mean(forg)) if forg else 0.0,
        "per_task_forgetting": forg,
        "bwt": bwt,
        "final_acc": final.tolist(),
        "acc_matrix": A.tolist(),
    }


def sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation (n - 1 in the denominator); 0.0 for one value."""
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1)) if v.size > 1 else 0.0


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample standard deviation."""
    return float(np.mean(values)), sample_std(values)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d of two independent samples, with the pooled standard deviation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    pooled = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def welch_t(a: Sequence[float], b: Sequence[float]) -> float:
    """Welch's t statistic of two independent samples (unequal variances)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    se2 = a.var(ddof=1) / a.size + b.var(ddof=1) / b.size
    if se2 == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(se2))


def aggregate(results: List[Dict]) -> Dict:
    """Mean and sample standard deviation over the seeds of one configuration."""
    acc_m, acc_s = mean_sd([r["avg_acc"] for r in results])
    f_m, f_s = mean_sd([r["avg_forgetting"] for r in results])
    b_m, b_s = mean_sd([r["bwt"] for r in results])
    syn_m, syn_s = mean_sd([r["compute"]["synops_total_G"] for r in results])
    ev_m, ev_s = mean_sd([r["compute"]["synops_total_event_G"] for r in results])
    bwf = [r["compute"]["backward_sample_fraction"] for r in results]
    return {
        "config": results[0]["config"],
        "n_seeds": len(results),
        "avg_acc_mean": acc_m, "avg_acc_std": acc_s,
        "avg_forgetting_mean": f_m, "avg_forgetting_std": f_s,
        "bwt_mean": b_m, "bwt_std": b_s,
        "synops_G_mean": syn_m, "synops_G_std": syn_s,
        "synops_event_G_mean": ev_m, "synops_event_G_std": ev_s,
        "backward_sample_fraction_mean": float(np.mean(bwf)),
        "seeds": [r["seed"] for r in results],
    }

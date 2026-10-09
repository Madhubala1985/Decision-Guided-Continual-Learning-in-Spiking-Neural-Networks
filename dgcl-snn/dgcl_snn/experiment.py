"""Configuration and execution of one continual-learning run."""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from .accounting import ComputeAccountant
from .data import build_split_mnist
from .decision import DecisionConfig, PerSampleDecisionModule, RandomSkipController
from .diagnostics import collect_diagnostics
from .isi import ISICVEstimator
from .metrics import cl_metrics
from .network import SpikingMLP
from .roles import FunctionalMemoryLayer
from .si import SynapticIntelligence
from .training import evaluate, train_task
from .utils import get_device, set_seed


@dataclass
class ExpConfig:
    """Configuration of one run.

    `adaptive`, `use_si`, `use_decision` and `random_skip_rate` select the
    configuration of the ablation ladder or the random-skip control;
    `multi_head` selects the task- or class-incremental setting.
    """
    name: str = "unnamed"
    adaptive: bool = True             # ALIF vs LIF
    use_si: bool = True
    use_decision: bool = True
    random_skip_rate: Optional[float] = None   # set: use the random-skip control
    multi_head: bool = True           # task-IL vs class-IL
    beta: float = 0.6
    t_steps: int = 20
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    lambda_si: float = 0.5
    xi: float = 1e-3
    protect_lambda_boost: float = 3.0
    grad_clip: float = 1.0
    h1: int = 256
    h2: int = 256
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    isi_min_isi: int = 2
    track_isi: bool = True
    imp_alpha: float = 0.7          # weight on SI importance
    imp_beta: float = 0.3           # weight on ISI-CV importance
    mem_p_low: float = 0.33         # quantile: upper boundary of FAST_PLASTIC
    mem_p_high: float = 0.67        # quantile: lower boundary of STABLE_CONSOL
    consol_top_k: float = 0.10      # fraction tagged as consolidated per task
    collect_diagnostics: bool = False   # extra eval pass for the diagnostic figures
    synthetic: bool = False
    n_synth: int = 512
    verbose: bool = True


def run_experiment(cfg: ExpConfig, seed: int, tasks=None, device=None) -> Dict:
    """Train on the task sequence with one seed and return the result dict.

    After each task the model is evaluated on all tasks seen so far, and the
    firing-regularity importance and the neuron roles are recorded (analysis
    only). The result contains the accuracy matrix, the metrics, the action
    shares, the cost summary and the logs.
    """
    device = device or get_device()
    set_seed(seed)

    if tasks is None:
        tasks = build_split_mnist(
            batch_size=cfg.batch_size,
            multi_head=cfg.multi_head,
            synthetic=cfg.synthetic,
            n_synth=cfg.n_synth,
            device=device,
        )
    n_tasks = len(tasks)
    n_classes = 2 if cfg.multi_head else 2 * n_tasks

    model = SpikingMLP(
        h1=cfg.h1, h2=cfg.h2, n_tasks=n_tasks,
        multi_head=cfg.multi_head, adaptive=cfg.adaptive, beta=cfg.beta,
    ).to(device)

    si = (SynapticIntelligence(model, lambda_si=cfg.lambda_si, xi=cfg.xi)
          if cfg.use_si else None)

    if cfg.random_skip_rate is not None:
        controller = RandomSkipController(cfg.random_skip_rate)
    elif cfg.use_decision:
        controller = PerSampleDecisionModule(cfg.decision)
    else:
        controller = None

    acct = ComputeAccountant(cfg.h1, cfg.h2, n_classes)
    isi_est = ISICVEstimator(min_isi=cfg.isi_min_isi)
    fml = FunctionalMemoryLayer(cfg.h1, cfg.h2,
                                p_low=cfg.mem_p_low, p_high=cfg.mem_p_high,
                                consol_top_k=cfg.consol_top_k)

    A = np.zeros((n_tasks, n_tasks))
    all_logs, omega_trace, isi_trace, action_by_task = [], [], [], []
    t0 = time.time()

    for t in tasks:
        tid = t["task_id"]
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        if si is not None:
            si.begin_task()

        logs = train_task(model, si, controller, opt, t["train_loader"],
                          tid, device, cfg, acct, epoch_offset=tid * 10_000)
        all_logs += logs

        if si is not None:
            si.end_task()
            omega_trace.append(si.mean_omega())

        if cfg.track_isi:
            r = isi_est.from_loader(model, t["test_loader"], tid, device,
                                    cfg.t_steps, n_batches=5)
            isi_trace.append({
                "task_id": tid,
                "imp_l1_mean": float(r["imp_l1"].mean()),
                "imp_l1_std": float(r["imp_l1"].std()),
                "cv_l1_mean": float(r["cv_l1"].mean()),
                "silent_frac_l1": float((r["imp_l1"] == 0).float().mean()),
                "imp_l2_mean": float(r["imp_l2"].mean()),
                "cv_l2_mean": float(r["cv_l2"].mean()),
            })

            # ---- neuron roles (analysis only, not used by training) ------
            om = si.neuron_omega() if si is not None else {}
            z1 = torch.zeros_like(r["imp_l1"])
            z2 = torch.zeros_like(r["imp_l2"])
            imp1 = FunctionalMemoryLayer.combined_importance(
                om.get("l1", z1), r["imp_l1"], cfg.imp_alpha, cfg.imp_beta)
            imp2 = FunctionalMemoryLayer.combined_importance(
                om.get("l2", z2), r["imp_l2"], cfg.imp_alpha, cfg.imp_beta)
            fml.snapshot(tid, imp1, imp2)

        if controller is not None:
            action_by_task.append(controller.task_distribution(tid))

        for j in range(tid + 1):
            acc, _ = evaluate(model, tasks[j]["test_loader"], j, device,
                              cfg.t_steps, cfg.multi_head)
            A[tid, j] = acc

        if cfg.verbose:
            d = action_by_task[-1] if action_by_task else {}
            ds = "  ".join(f"{k}={v:.1%}" for k, v in d.items() if v > 0)
            print(f"  [{cfg.name} s{seed}] task {tid+1}/{n_tasks} "
                  f"acc={A[tid, tid]:.4f}  {ds}")

    acct.wall_time = time.time() - t0

    diagnostics = None
    if cfg.collect_diagnostics:
        try:
            diagnostics = collect_diagnostics(model, tasks, device, cfg,
                                              si=si, isi_est=isi_est)
        except Exception as e:                       # never lose a run to a plot
            print(f"    [diagnostics skipped: {type(e).__name__}: {e}]")

    m = cl_metrics(A)
    return {
        "config": cfg.name,
        "seed": seed,
        "multi_head": cfg.multi_head,
        **m,
        "omega_trace": omega_trace,
        "isi_trace": isi_trace,
        "memory_trace": fml.summary(),
        "memory_states_final": (fml.history[-1] if fml.history else None),
        "stm_ltm_transitions": fml.transitions,
        "action_by_task": action_by_task,
        "action_global": (controller.global_distribution() if controller
                          else {"UPDATE": 1.0}),
        "compute": acct.summary(),
        "epoch_logs": all_logs,
        "diagnostics": diagnostics,
    }

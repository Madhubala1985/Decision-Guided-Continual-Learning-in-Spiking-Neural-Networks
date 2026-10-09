"""End-to-end runs on a small synthetic dataset. The numbers have no meaning;
the tests check that the pipeline runs and that its bookkeeping is consistent."""
import numpy as np
import pytest
import torch

from dgcl_snn import (
    ComputeAccountant,
    DecisionConfig,
    ExpConfig,
    RandomSkipController,
    SpikingMLP,
    SynapticIntelligence,
    build_split_mnist,
    run_experiment,
    train_task,
)

FAST = dict(epochs=1, synthetic=True, n_synth=256, verbose=False)


def test_decision_module_run(cpu):
    cfg = ExpConfig(name="dm", decision=DecisionConfig(min_warmup_samples=64), **FAST)
    r = run_experiment(cfg, seed=0, device=cpu)
    A = np.array(r["acc_matrix"])
    assert A.shape == (5, 5) and np.all(np.triu(A, 1) == 0)
    assert len(r["action_by_task"]) == 5
    for shares in r["action_by_task"]:
        assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert r["action_global"]["IGNORE"] > 0
    c = r["compute"]
    assert 0 < c["backward_sample_fraction"] < 1
    assert c["synops_total_event_G"] < c["synops_total_G"]
    assert len(r["omega_trace"]) == 5 and len(r["isi_trace"]) == 5


def test_runs_are_reproducible(cpu):
    cfg = ExpConfig(name="dm", decision=DecisionConfig(min_warmup_samples=64), **FAST)
    a = run_experiment(cfg, seed=3, device=cpu)
    b = run_experiment(cfg, seed=3, device=cpu)
    assert a["acc_matrix"] == b["acc_matrix"]
    assert a["action_by_task"] == b["action_by_task"]
    assert a["compute"]["synops_total"] == b["compute"]["synops_total"]


@pytest.mark.parametrize("kw", [
    dict(adaptive=False, use_si=False, use_decision=False),      # LIF
    dict(adaptive=True, use_si=False, use_decision=False),       # ALIF
    dict(adaptive=True, use_si=True, use_decision=False),        # ALIF+SI
])
def test_without_a_controller_every_sample_is_learned(cpu, kw):
    r = run_experiment(ExpConfig(name="x", **kw, **FAST), seed=0, device=cpu)
    c = r["compute"]
    assert c["backward_sample_fraction"] == 1.0
    assert c["synops_total_event_G"] == c["synops_total_G"]
    assert r["action_global"] == {"UPDATE": 1.0}


def test_random_control_run(cpu):
    cfg = ExpConfig(name="rand", use_decision=False, random_skip_rate=0.5, **FAST)
    r = run_experiment(cfg, seed=0, device=cpu)
    assert abs(r["action_global"]["IGNORE"] - 0.5) < 0.1
    assert r["action_global"]["PROTECT"] == 0.0 and r["action_global"]["CONSOLIDATE"] == 0.0
    assert abs(r["compute"]["backward_sample_fraction"] - 0.5) < 0.1


def test_class_incremental_run(cpu):
    cfg = ExpConfig(name="cil", multi_head=False, use_decision=False, **FAST)
    r = run_experiment(cfg, seed=0, device=cpu)
    assert np.array(r["acc_matrix"]).shape == (5, 5) and r["multi_head"] is False


def test_fully_ignored_batches_are_skipped(cpu):
    """If every sample of a batch is ignored, nothing is updated and nothing is charged."""
    tasks = build_split_mnist(synthetic=True, n_synth=128, device=cpu)
    model = SpikingMLP()
    si = SynapticIntelligence(model)
    si.begin_task()
    before = [p.detach().clone() for p in model.parameters()]
    acct = ComputeAccountant(256, 256, 2)
    cfg = ExpConfig(epochs=1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    logs = train_task(model, si, RandomSkipController(1.0), opt, tasks[0]["train_loader"],
                      0, cpu, cfg, acct)
    s = acct.summary()
    assert s["n_skipped_batches"] == s["n_batches"] > 0
    assert s["synops_backward"] == 0.0 and s["synops_forward"] > 0.0
    assert logs[0]["ignored_batches"] == s["n_batches"]
    for p, q in zip(model.parameters(), before):
        assert torch.equal(p, q)

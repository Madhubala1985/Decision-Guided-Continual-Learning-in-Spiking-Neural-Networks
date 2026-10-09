import math

import torch

from dgcl_snn import Action, DecisionConfig, PerSampleDecisionModule, RandomSkipController

U, C, P, I = (int(Action.UPDATE), int(Action.CONSOLIDATE), int(Action.PROTECT),
              int(Action.IGNORE))


def count(actions, a):
    return int((actions == a).sum())


def test_entropy():
    H = PerSampleDecisionModule.entropy(torch.tensor([[0.0, 0.0], [20.0, -20.0]]))
    assert abs(H[0].item() - math.log(2)) < 1e-5
    assert H[1].item() < 1e-5


def test_attention_scales_entropy_by_the_normalised_rate():
    H = torch.ones(3)
    rate = torch.tensor([0.05, 0.5, 0.95])
    att = PerSampleDecisionModule.attention(H, rate, rate)
    assert torch.allclose(att, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)


def test_every_sample_is_updated_during_warm_up():
    dm = PerSampleDecisionModule(DecisionConfig(min_warmup_samples=512))
    H = torch.linspace(0, 1, 64)
    actions = dm.decide(H, H, mean_omega=1.0, task_id=0, step=0)
    assert count(actions, U) == 64


def test_rule_with_the_gate_closed():
    dm = PerSampleDecisionModule(DecisionConfig(min_warmup_samples=64))
    H = torch.linspace(0, 1, 64)
    actions = dm.decide(H, H, mean_omega=0.0, task_id=0, step=0)
    assert count(actions, I) == 16                  # attention below the 25 % quantile
    assert count(actions, U) == 16                  # entropy above the 75 % quantile
    assert count(actions, P) == 0                   # the gate is closed
    assert count(actions, C) == 32
    assert (actions[:16] == I).all() and (actions[48:] == U).all()


def test_protect_needs_the_open_gate():
    dm = PerSampleDecisionModule(DecisionConfig(min_warmup_samples=64))
    H = torch.linspace(0, 1, 64)
    for step in range(3):
        dm.decide(H, H, mean_omega=0.0, task_id=0, step=step)
    # the importance rises above the 70 % quantile of its recent values
    actions = dm.decide(H, H, mean_omega=1.0, task_id=1, step=3)
    # The window now holds the 64 values four times. Its 40 % quantile is the
    # 26th value, so 25 samples lie below it; 16 of them are ignored.
    assert count(actions, I) == 16                  # ignore has priority over protect
    assert count(actions, P) == 9
    assert count(actions, U) == 16
    assert count(actions, C) == 23
    assert (actions[16:25] == P).all()
    # the same importance again: it no longer exceeds its own 70 % quantile
    dm.decide(H, H, mean_omega=1.0, task_id=1, step=4)
    actions = dm.decide(H, H, mean_omega=1.0, task_id=1, step=5)
    assert count(actions, P) == 0


def test_ignore_is_reachable_when_the_threshold_is_zero():
    dm = PerSampleDecisionModule(DecisionConfig(min_warmup_samples=8))
    H = torch.linspace(0.1, 1, 8)
    actions = dm.decide(H, torch.zeros(8), mean_omega=0.0, task_id=0, step=0)
    assert count(actions, I) == 8


def test_loss_weights():
    dm = PerSampleDecisionModule(DecisionConfig())
    w = dm.sample_weights(torch.tensor([U, C, P, I]), "cpu")
    assert torch.allclose(w, torch.tensor([1.0, 0.3, 0.1, 0.0]))


def test_action_shares_are_counted_per_task():
    dm = PerSampleDecisionModule(DecisionConfig(min_warmup_samples=64))
    H = torch.linspace(0, 1, 64)
    dm.decide(H, H, 0.0, task_id=0, step=0)
    dm.decide(H, torch.zeros(64), 0.0, task_id=1, step=1)
    d0, d1 = dm.task_distribution(0), dm.task_distribution(1)
    assert abs(sum(d0.values()) - 1.0) < 1e-9 and abs(sum(d1.values()) - 1.0) < 1e-9
    assert d0["IGNORE"] == 0.25
    assert d1["IGNORE"] > d0["IGNORE"]


def test_random_control_uses_two_actions_at_the_given_rate():
    ctrl = RandomSkipController(0.3)
    H = torch.zeros(20_000)
    actions = ctrl.decide(H, H, 0.0, task_id=0, step=0)
    assert count(actions, C) == 0 and count(actions, P) == 0
    assert abs(count(actions, I) / 20_000 - 0.3) < 0.02
    w = ctrl.sample_weights(actions, "cpu")
    assert set(w.unique().tolist()) == {0.0, 1.0}
    assert abs(ctrl.task_distribution(0)["IGNORE"] - 0.3) < 0.02


def test_random_control_extremes():
    H = torch.zeros(100)
    assert count(RandomSkipController(0.0).decide(H, H, 0.0, 0, 0), U) == 100
    assert count(RandomSkipController(1.0).decide(H, H, 0.0, 0, 0), I) == 100

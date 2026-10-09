import torch

from dgcl_snn import ComputeAccountant


def forward_only():
    acct = ComputeAccountant(h1=7, h2=6, n_classes=2)
    in_spikes = torch.ones(2, 3, 4)                           # T=2, B=3, D=4: 24 spikes
    info = {"count_l1": torch.full((3, 7), 10 / 21), "count_l2": torch.full((3, 6), 5 / 18)}
    fwd = acct.charge_forward(in_spikes, info)
    return acct, fwd


def test_forward_cost():
    acct, fwd = forward_only()
    # 24 * 7 + 10 * 6 + 5 * 2
    assert abs(float(fwd) - 238.0) < 1e-3
    assert acct.n_forward_samples == 3 and acct.n_batches == 1


def test_dense_and_event_driven_backward_cost():
    acct, fwd = forward_only()
    acct.charge_backward(n_samples=1, batch_size=4, fwd_synops_of_batch=fwd)
    s = acct.summary()
    assert abs(s["synops_backward"] - 2 * 238.0) < 1e-3            # full cost
    assert abs(s["synops_backward_event"] - 2 * 238.0 * 0.25) < 1e-3
    assert abs(s["synops_total"] - 3 * 238.0) < 1e-3
    assert s["synops_total_event_G"] < s["synops_total_G"]


def test_skipped_batch_has_no_backward_cost():
    acct, _ = forward_only()
    acct.note_skipped_batch()
    s = acct.summary()
    assert s["synops_backward"] == 0.0 and s["synops_backward_event"] == 0.0
    assert s["n_skipped_batches"] == 1
    assert s["backward_sample_fraction"] == 0.0

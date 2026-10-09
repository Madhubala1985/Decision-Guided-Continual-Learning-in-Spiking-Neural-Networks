import torch

from dgcl_snn import RunningQuantile


def test_warm_up():
    q = RunningQuantile(capacity=100, warmup=10)
    q.update(torch.arange(9.0))
    assert not q.ready
    q.update(torch.tensor([9.0]))
    assert q.ready


def test_empty_buffer_returns_zero():
    assert float(RunningQuantile().q(0.5)) == 0.0


def test_quantile_of_the_stored_values():
    q = RunningQuantile(capacity=100, warmup=1)
    v = torch.rand(37)
    q.update(v)
    for p in (0.25, 0.4, 0.75):
        assert torch.isclose(q.q(p), torch.quantile(v, p))


def test_window_keeps_the_most_recent_values():
    q = RunningQuantile(capacity=8, warmup=1)
    q.update(torch.arange(0.0, 5.0))                # 0..4
    q.update(torch.arange(5.0, 10.0))               # 5..9, wraps around
    assert q.n_filled == 8
    assert sorted(q.buf.tolist()) == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert torch.isclose(q.q(0.5), torch.quantile(torch.arange(2.0, 10.0), 0.5))


def test_update_larger_than_the_window():
    q = RunningQuantile(capacity=8, warmup=1)
    q.update(torch.arange(0.0, 20.0))
    assert sorted(q.buf.tolist()) == [float(i) for i in range(12, 20)]

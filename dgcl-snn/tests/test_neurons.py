import torch

from dgcl_snn import ALIFCell, LIFCell, SuperSpike, poisson_encode, spike_fn


def test_spike_function_is_a_step():
    u = torch.tensor([-0.1, 0.0, 0.2])
    assert spike_fn(u).tolist() == [0.0, 1.0, 1.0]


def test_surrogate_gradient():
    u = torch.tensor([0.0, 0.01, -0.01], requires_grad=True)
    spike_fn(u).sum().backward()
    # 1 / (1 + alpha * |u|)^2 with alpha = 100
    assert SuperSpike.alpha == 100.0
    assert torch.allclose(u.grad, torch.tensor([1.0, 0.25, 0.25]))


def test_lif_threshold_is_constant():
    cell = LIFCell(4)
    s = cell.init_state(2, "cpu")
    for _ in range(10):
        z, s, v_th = cell(torch.ones(2, 4), s)
        assert torch.allclose(v_th, torch.full((2, 4), 0.3))
    assert z.sum() > 0                     # the neurons do fire


def test_alif_threshold_rises_after_a_spike():
    cell = ALIFCell(1)
    s = cell.init_state(1, "cpu")
    z, s, v_th = cell(torch.ones(1, 1), s)          # v = 1 > 0.3: spike
    assert z.item() == 1.0 and abs(v_th.item() - 0.3) < 1e-7
    z, s, v_th = cell(torch.zeros(1, 1), s)
    assert v_th.item() > 0.3


def test_spike_resets_the_membrane_by_the_threshold():
    cell = LIFCell(1)
    s = cell.init_state(1, "cpu")
    _, s, _ = cell(torch.ones(1, 1), s)             # v = 1, spike
    _, s2, _ = cell(torch.zeros(1, 1), s)
    assert abs(s2.v.item() - (cell.decay_m * 1.0 - 0.3)) < 1e-6


def test_poisson_encoding():
    x = torch.cat([torch.zeros(1, 1, 28, 28), torch.ones(1, 1, 28, 28),
                   torch.full((1, 1, 28, 28), 0.5)])
    sp = poisson_encode(x, 50)
    assert sp.shape == (50, 3, 784)
    assert set(sp.unique().tolist()) <= {0.0, 1.0}
    assert sp[:, 0].sum() == 0                      # intensity 0: no spikes
    assert sp[:, 1].mean() == 1                     # intensity 1: a spike at every step
    assert abs(sp[:, 2].mean().item() - 0.5) < 0.02

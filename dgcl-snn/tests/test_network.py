import torch

from dgcl_snn import SpikingMLP, poisson_encode


def test_parameter_count():
    model = SpikingMLP()                            # 784-256-256, five two-way heads
    assert sum(p.numel() for p in model.parameters()) == 269_322


def test_forward_returns_per_sample_statistics():
    model = SpikingMLP()
    sp = poisson_encode(torch.rand(6, 1, 28, 28), 20)
    logits, info = model(sp, task_id=3, collect_raster=True)
    assert logits.shape == (6, 2)
    assert info["rate_l1"].shape == (6,) and info["rate_l2"].shape == (6,)
    assert info["count_l1"].shape == (6, 256)
    assert info["spikes_l1"].shape == (20, 6, 256)
    assert torch.equal(info["spikes_l1"].sum(0), info["count_l1"])
    assert torch.allclose(info["rate_l1"], info["count_l1"].mean(1) / 20)
    rate = info["rate_l1"].detach()
    assert 0.0 <= float(rate.min()) and float(rate.max()) <= 1.0


def test_single_head_has_ten_outputs():
    model = SpikingMLP(multi_head=False)
    logits, _ = model(poisson_encode(torch.rand(3, 1, 28, 28), 5))
    assert logits.shape == (3, 10)


def test_a_sample_does_not_depend_on_the_rest_of_the_batch():
    model = SpikingMLP()
    sp = poisson_encode(torch.rand(4, 1, 28, 28), 20)
    logits, info = model(sp)
    logits0, info0 = model(sp[:, :1])
    assert torch.allclose(logits[:1], logits0, atol=1e-6)
    assert torch.allclose(info["rate_l1"][:1], info0["rate_l1"])


def test_lif_and_alif_networks_share_the_code_path():
    lif = SpikingMLP(adaptive=False)
    assert lif.cell1.beta == 0.0 and lif.cell2.beta == 0.0
    alif = SpikingMLP(adaptive=True, beta=0.6)
    assert alif.cell1.beta == 0.6

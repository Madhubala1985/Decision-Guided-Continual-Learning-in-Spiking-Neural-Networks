import torch
import torch.nn as nn

from dgcl_snn import SynapticIntelligence


def one_weight_model():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return model


def test_path_integral_and_importance():
    model = one_weight_model()
    si = SynapticIntelligence(model, lambda_si=0.5, xi=1e-3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    si.begin_task()

    loss = model(torch.ones(1, 1)).pow(2).sum()     # w^2, gradient 2
    opt.zero_grad()
    loss.backward()
    opt.step()                                      # w = 1 - 0.1 * 2 = 0.8
    si.accumulate_step()                            # omega = -(2) * (-0.2) = 0.4
    assert abs(si.omega["weight"].item() - 0.4) < 1e-6

    si.end_task()                                   # Omega = 0.4 / (0.2^2 + 0.001)
    assert abs(si.Omega["weight"].item() - 0.4 / 0.041) < 1e-4
    assert abs(si.mean_omega() - 0.4 / 0.041) < 1e-4


def test_penalty_is_zero_at_the_reference_and_grows_with_distance():
    model = one_weight_model()
    si = SynapticIntelligence(model, lambda_si=0.5)
    si.Omega["weight"].fill_(2.0)
    assert si.penalty().item() == 0.0
    with torch.no_grad():
        model.weight.add_(0.5)
    assert abs(si.penalty().item() - 0.5 * 2.0 * 0.25) < 1e-6


def test_begin_task_resets_the_path_integral_only():
    model = one_weight_model()
    si = SynapticIntelligence(model)
    si.omega["weight"].fill_(1.0)
    si.Omega["weight"].fill_(3.0)
    si.begin_task()
    assert si.omega["weight"].item() == 0.0
    assert si.Omega["weight"].item() == 3.0


def test_negative_path_integral_gives_no_importance():
    model = one_weight_model()
    si = SynapticIntelligence(model)
    si.omega["weight"].fill_(-1.0)
    with torch.no_grad():
        model.weight.add_(0.1)
    si.end_task()
    assert si.Omega["weight"].item() == 0.0


def test_mean_omega_is_refreshed_after_a_task():
    model = one_weight_model()
    si = SynapticIntelligence(model)
    assert si.mean_omega() == 0.0
    si.omega["weight"].fill_(1.0)
    with torch.no_grad():
        model.weight.add_(0.1)
    si.end_task()
    assert si.mean_omega() > 0.0

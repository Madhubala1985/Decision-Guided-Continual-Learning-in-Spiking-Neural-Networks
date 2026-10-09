"""Poisson rate coding of the input."""

import torch


def poisson_encode(x: torch.Tensor, t_steps: int) -> torch.Tensor:
    """Poisson rate coding.

    Each pixel with intensity p in [0, 1] emits an independent Bernoulli(p)
    spike at every timestep.

    Args:
        x:       (B, 1, 28, 28) or (B, D), values in [0, 1]
        t_steps: number of timesteps T
    Returns:
        (T, B, D) float tensor of 0/1 spikes
    """
    flat = x.view(x.size(0), -1)
    rates = flat.unsqueeze(0).expand(t_steps, -1, -1)
    return torch.bernoulli(rates)

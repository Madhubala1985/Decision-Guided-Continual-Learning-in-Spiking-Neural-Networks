"""Spiking neuron models and the surrogate gradient."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class SuperSpike(torch.autograd.Function):
    """Heaviside step in the forward pass, SuperSpike surrogate in the backward pass.

        forward :  z = Theta(u)
        backward:  dz/du = 1 / (1 + alpha * |u|)^2

    Zenke and Ganguli (2018).
    """

    alpha = 100.0

    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + SuperSpike.alpha * u.abs()) ** 2
        return grad_output * sg


spike_fn = SuperSpike.apply


@dataclass
class ALIFState:
    """State of a layer of ALIF neurons at one timestep."""
    v: torch.Tensor   # membrane potential
    b: torch.Tensor   # adaptation trace
    z: torch.Tensor   # last spike


class ALIFCell(nn.Module):
    """Adaptive leaky integrate-and-fire neurons with spike-frequency adaptation.

    Discrete-time update (Bellec et al., 2020):

        b[t]    = rho * b[t-1] + (1 - rho) * z[t-1]
        v_th[t] = v_th0 + beta * b[t]
        v[t]    = decay_m * v[t-1] + I[t] - z[t-1] * v_th[t]
        z[t]    = Theta(v[t] - v_th[t])

    with decay_m = exp(-dt / tau_m) and rho = exp(-dt / tau_b).

    The input current I[t] is added without the factor (1 - decay_m). With
    tau_m = 20 ms that factor would make the membrane drive about twenty times
    smaller, and the network would not fire.
    """

    def __init__(
        self,
        n: int,
        tau_m: float = 20.0,
        tau_b: float = 80.0,
        v_th0: float = 0.3,
        beta: float = 0.6,
        dt: float = 1.0,
    ):
        super().__init__()
        self.n = n
        self.tau_m, self.tau_b = tau_m, tau_b
        self.v_th0, self.beta, self.dt = v_th0, beta, dt
        self.decay_m = float(np.exp(-dt / tau_m))
        self.decay_b = float(np.exp(-dt / tau_b))

    def init_state(self, batch: int, device) -> ALIFState:
        z = torch.zeros(batch, self.n, device=device)
        return ALIFState(v=torch.zeros_like(z), b=torch.zeros_like(z), z=z)

    def forward(self, I: torch.Tensor,
                s: ALIFState) -> Tuple[torch.Tensor, ALIFState, torch.Tensor]:
        """One timestep. Returns (spikes, new state, threshold)."""
        b = self.decay_b * s.b + (1.0 - self.decay_b) * s.z
        v_th = self.v_th0 + self.beta * b
        v = self.decay_m * s.v + I - s.z * v_th
        z = spike_fn(v - v_th)
        return z, ALIFState(v=v, b=b, z=z), v_th

    def extra_repr(self) -> str:
        return (f"n={self.n} tau_m={self.tau_m} tau_b={self.tau_b} "
                f"v_th0={self.v_th0} beta={self.beta}")


class LIFCell(ALIFCell):
    """Leaky integrate-and-fire neurons: an ALIF cell with beta = 0."""

    def __init__(self, n: int, tau_m: float = 20.0, v_th0: float = 0.3, dt: float = 1.0):
        super().__init__(n, tau_m=tau_m, tau_b=1e9, v_th0=v_th0, beta=0.0, dt=dt)

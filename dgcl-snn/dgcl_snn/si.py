"""Synaptic Intelligence: importance-based regularisation."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class SynapticIntelligence:
    """Synaptic Intelligence (Zenke, Poole and Ganguli, 2017).

        omega_k -= grad_k * delta_theta_k                 after every step
        Omega_k += max(omega_k, 0) / (delta_k^2 + xi)     at the end of a task
        penalty  = lambda * sum_k Omega_k * (theta_k - theta_ref_k)^2

    `grad_k` is the gradient that was used for the update, that is the gradient
    of the total loss (weighted task loss plus penalty) after clipping. Zenke
    et al. use the gradient of the task loss alone.
    """

    def __init__(self, model: nn.Module, lambda_si: float = 0.5, xi: float = 1e-3):
        self.model = model
        self.lambda_si = lambda_si
        self.xi = xi
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.omega = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.Omega = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.theta_ref = {n: p.detach().clone() for n, p in self.params.items()}
        self._theta_prev = {n: p.detach().clone() for n, p in self.params.items()}
        self._omega_cache: Optional[float] = None

    def begin_task(self) -> None:
        """Reset the path integral omega. Omega and the reference weights are kept."""
        for n, p in self.params.items():
            self.omega[n].zero_()
            self._theta_prev[n] = p.detach().clone()

    @torch.no_grad()
    def accumulate_step(self) -> None:
        """Add one step to the path integral.

        Must be called directly after `optimizer.step()`, while `.grad` still
        holds the gradient of that step.
        """
        for n, p in self.params.items():
            if p.grad is None:
                self._theta_prev[n] = p.detach().clone()
                continue
            delta = p.detach() - self._theta_prev[n]
            self.omega[n] -= p.grad.detach() * delta
            self._theta_prev[n] = p.detach().clone()

    @torch.no_grad()
    def end_task(self) -> None:
        """Convert the path integral into importance and move the reference weights."""
        for n, p in self.params.items():
            delta = p.detach() - self.theta_ref[n]
            self.Omega[n] += torch.clamp(self.omega[n], min=0.0) / (delta.pow(2) + self.xi)
            self.theta_ref[n] = p.detach().clone()
        self._omega_cache = None      # invalidate: Omega has changed

    def penalty(self) -> torch.Tensor:
        """Quadratic penalty on the distance to the reference weights."""
        loss = torch.zeros((), device=next(self.model.parameters()).device)
        for n, p in self.params.items():
            loss = loss + (self.Omega[n] * (p - self.theta_ref[n]).pow(2)).sum()
        return self.lambda_si * loss

    @torch.no_grad()
    def mean_omega(self) -> float:
        """Mean of Omega over all parameter tensors.

        Omega changes only in `end_task()`, so the value is cached and reading it
        in every training step costs nothing.
        """
        if self._omega_cache is None:
            vals = [O.mean() for O in self.Omega.values()]
            self._omega_cache = float(torch.stack(vals).mean()) if vals else 0.0
        return self._omega_cache

    @torch.no_grad()
    def layer_omega(self) -> Dict[str, float]:
        """Mean Omega of every parameter tensor."""
        return {n: float(O.mean().item()) for n, O in self.Omega.items()}

    @torch.no_grad()
    def neuron_omega(self) -> Dict[str, torch.Tensor]:
        """Importance per hidden neuron, derived from the per-parameter Omega.

        `fc1.weight` has shape (h1, n_in) and `fc2.weight` (h2, h1). The mean over
        the input dimension gives one value per postsynaptic neuron. The values are
        min-max normalised to [0, 1] so that they can be combined with the
        firing-regularity importance, which is on that scale already.
        """
        out = {}
        for name, key in (("fc1.weight", "l1"), ("fc2.weight", "l2")):
            if name not in self.Omega:
                continue
            v = self.Omega[name].mean(dim=1).detach().cpu()
            lo, hi = v.min(), v.max()
            out[key] = (v - lo) / (hi - lo + 1e-12)
        return out

    @torch.no_grad()
    def normalised_omega_map(self) -> Dict[str, torch.Tensor]:
        """Omega rescaled to [0, 1] per tensor, for plotting."""
        out = {}
        for n, O in self.Omega.items():
            lo, hi = O.min(), O.max()
            out[n] = (O - lo) / (hi - lo + 1e-12)
        return out

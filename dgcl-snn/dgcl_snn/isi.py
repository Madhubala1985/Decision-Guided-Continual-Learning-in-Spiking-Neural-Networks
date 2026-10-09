"""Firing-regularity importance from inter-spike intervals (analysis only)."""

from typing import Tuple

import torch

from .encoding import poisson_encode


class ISICVEstimator:
    """Firing-regularity importance from inter-spike intervals (ISI).

        CV         = std(ISI) / mean(ISI)
        importance = 1 / (1 + CV)     if the neuron has at least `min_isi` intervals
        importance = 0                otherwise

    A neuron with too few intervals has no defined CV. It is marked invalid and
    gets importance 0. Assigning CV = 0 instead would give silent neurons the
    maximum importance.

    The statistics are computed for all (sample, neuron) pairs at once, with one
    pass over the timesteps.

    This quantity is computed after each task for analysis. It does not enter
    the training loss or the decision rule.
    """

    def __init__(self, min_isi: int = 2, eps: float = 1e-8):
        self.min_isi = min_isi     # need >= min_isi intervals, i.e. min_isi+1 spikes
        self.eps = eps

    @torch.no_grad()
    def cv_from_raster(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Coefficient of variation per (sample, neuron).

        Args:
            spikes: (T, B, H) binary spike trains
        Returns:
            cv:    (B, H), 0 where invalid
            valid: (B, H) bool, True where the neuron has >= `min_isi` intervals
        """
        T, B, H = spikes.shape
        dev = spikes.device
        last = torch.full((B, H), -1.0, device=dev)
        s_isi = torch.zeros(B, H, device=dev)
        s_isi2 = torch.zeros(B, H, device=dev)
        n_isi = torch.zeros(B, H, device=dev)

        for t in range(T):
            z = spikes[t]
            has_prev = (last >= 0).float()
            isi = (t - last) * has_prev
            contrib = z * has_prev
            s_isi += isi * contrib
            s_isi2 += isi.pow(2) * contrib
            n_isi += contrib
            last = torch.where(z > 0, torch.full_like(last, float(t)), last)

        valid = n_isi >= self.min_isi
        n_safe = n_isi.clamp(min=1.0)
        mean = s_isi / n_safe
        var = (s_isi2 / n_safe - mean.pow(2)).clamp(min=0.0)
        cv = var.sqrt() / (mean + self.eps)
        cv = torch.where(valid, cv, torch.zeros_like(cv))
        return cv, valid

    @torch.no_grad()
    def neuron_importance(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average over the samples of a batch for which a neuron is valid.

        Returns:
            importance: (H,) in [0, 1]; 0 for neurons that are valid in no sample
            mean_cv:    (H,) mean CV over the valid samples. It is returned as
                        well because it cannot be recovered from the averaged
                        importance.
        """
        cv, valid = self.cv_from_raster(spikes)
        vf = valid.float()
        n_valid = vf.sum(dim=0)
        mean_cv = (cv * vf).sum(dim=0) / n_valid.clamp(min=1.0)
        importance = torch.where(
            n_valid > 0,
            1.0 / (1.0 + mean_cv),
            torch.zeros_like(mean_cv),
        )
        return importance.cpu(), mean_cv.cpu()

    @torch.no_grad()
    def from_loader(self, model, loader, task_id, device, t_steps, n_batches=10):
        """Importance and CV of both hidden layers, averaged over batches."""
        imp1, imp2, cv1, cv2 = [], [], [], []
        was_training = model.training
        model.eval()
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device)
            sp = poisson_encode(x, t_steps)
            _, info = model(sp, task_id=task_id, collect_raster=True)
            a, ca = self.neuron_importance(info["spikes_l1"])
            b, cb = self.neuron_importance(info["spikes_l2"])
            imp1.append(a)
            imp2.append(b)
            cv1.append(ca)
            cv2.append(cb)
        if was_training:
            model.train()
        return {
            "imp_l1": torch.stack(imp1).mean(0),
            "imp_l2": torch.stack(imp2).mean(0),
            "cv_l1": torch.stack(cv1).mean(0),
            "cv_l2": torch.stack(cv2).mean(0),
        }

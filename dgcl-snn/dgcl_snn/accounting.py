"""Training cost in synaptic operations under two cost models."""

from typing import Dict

import torch


class ComputeAccountant:
    """Counts the synaptic operations (SynOps) of training.

    A synaptic operation is the delivery of one presynaptic spike along one
    synapse. The forward cost of a batch is

        (input spikes) * h1 + (layer-1 spikes) * h2 + (layer-2 spikes) * n_classes

    The spike counts are measured. The backward cost is modelled as
    `backward_cost_factor` times the forward cost of the batch, under two
    conventions (see `charge_backward`). The backward figures are therefore
    model estimates, not measurements.
    """

    def __init__(self, h1: int, h2: int, n_classes: int, backward_cost_factor: float = 2.0):
        self.h1, self.h2, self.n_classes = h1, h2, n_classes
        self.bcf = backward_cost_factor
        self.reset()

    def reset(self) -> None:
        self.synops_fwd = 0.0
        self.synops_bwd = 0.0
        self.synops_bwd_event = 0.0
        self._fwd_acc = torch.zeros((), dtype=torch.float64)
        self._bwd_acc = torch.zeros((), dtype=torch.float64)
        self._bwd_ev_acc = torch.zeros((), dtype=torch.float64)
        self.n_forward_samples = 0
        self.n_backward_samples = 0
        self.n_batches = 0
        self.n_skipped_batches = 0
        self.wall_time = 0.0

    @torch.no_grad()
    def charge_forward(self, in_spikes: torch.Tensor, info: Dict) -> torch.Tensor:
        """Charge the forward pass of one batch and return its SynOps.

        Args:
            in_spikes: (T, B, D) input spikes
            info:      output of the network, with the per-sample spike counts

        The totals are kept as tensors and converted to Python numbers once, in
        `summary()`, so that no device synchronisation happens per batch.
        """
        n_in = in_spikes.sum()
        n_l1 = info["count_l1"].sum()
        n_l2 = info["count_l2"].sum()
        batch = n_in * self.h1 + n_l1 * self.h2 + n_l2 * self.n_classes
        self._fwd_acc = self._fwd_acc + batch
        self.n_forward_samples += int(in_spikes.shape[1])
        self.n_batches += 1
        return batch

    def charge_backward(self, n_samples: int, batch_size: int,
                        fwd_synops_of_batch: float) -> None:
        """Charge the backward pass of one batch under both cost models.

        Dense (`synops_backward`)
            The full backward cost is charged as soon as one sample has a non-zero
            weight. Tensors are dense, so a masked sample is still computed.
            Ignoring single samples saves nothing here; only a batch in which every
            sample is ignored is free.

        Event-driven (`synops_backward_event`)
            The backward cost is scaled by the fraction of samples with a non-zero
            weight. This models hardware on which an update is triggered per
            sample.
        """
        active_frac = n_samples / max(batch_size, 1)
        self._bwd_acc = self._bwd_acc + self.bcf * fwd_synops_of_batch
        self._bwd_ev_acc = self._bwd_ev_acc + self.bcf * fwd_synops_of_batch * active_frac
        self.n_backward_samples += int(n_samples)

    def note_skipped_batch(self) -> None:
        """Count a batch in which every sample was ignored (no backward pass)."""
        self.n_skipped_batches += 1

    def summary(self) -> Dict:
        # single synchronisation point for the whole run
        """Totals as Python numbers."""
        self.synops_fwd = float(self._fwd_acc)
        self.synops_bwd = float(self._bwd_acc)
        self.synops_bwd_event = float(self._bwd_ev_acc)
        total = self.synops_fwd + self.synops_bwd
        total_ev = self.synops_fwd + self.synops_bwd_event
        return {
            "synops_forward": self.synops_fwd,
            "synops_backward": self.synops_bwd,
            "synops_backward_event": self.synops_bwd_event,
            "synops_total": total,
            "synops_total_G": total / 1e9,
            "synops_total_event_G": total_ev / 1e9,
            "n_forward_samples": self.n_forward_samples,
            "n_backward_samples": self.n_backward_samples,
            "backward_sample_fraction": (
                self.n_backward_samples / max(self.n_forward_samples, 1)
            ),
            "n_batches": self.n_batches,
            "n_skipped_batches": self.n_skipped_batches,
            "wall_time_s": self.wall_time,
        }

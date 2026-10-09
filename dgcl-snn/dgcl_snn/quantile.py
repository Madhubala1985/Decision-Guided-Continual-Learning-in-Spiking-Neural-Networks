"""Sliding-window quantile estimator for the decision thresholds."""

from typing import Optional

import torch


class RunningQuantile:
    """Quantiles over a sliding window of the most recent values.

    The decision thresholds are percentiles of the observed signals instead of
    constants, because the range of a signal is not known in advance and changes
    during training. The window is limited to the most recent `capacity` values
    because the signal distribution changes from task to task.

    The ring buffer is kept on the device of the signals, so updating it and
    reading a quantile needs no transfer to the host.
    """

    def __init__(self, capacity: int = 4096, warmup: int = 256,
                 device: Optional[torch.device] = None):
        self.capacity = capacity
        self.warmup = warmup
        self.device = device or torch.device("cpu")
        self.buf = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.pos = 0
        self.n_filled = 0
        self.n_seen = 0

    def to(self, device) -> "RunningQuantile":
        self.device = device
        self.buf = self.buf.to(device)
        return self

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """Append values to the ring buffer, overwriting the oldest ones."""
        v = values.detach().reshape(-1).to(self.buf.dtype)
        if v.device != self.buf.device:
            v = v.to(self.buf.device)
        k = v.numel()
        if k == 0:
            return
        if k >= self.capacity:                       # buffer fully replaced
            self.buf.copy_(v[-self.capacity:])
            self.pos = 0
            self.n_filled = self.capacity
        else:
            end = self.pos + k
            if end <= self.capacity:
                self.buf[self.pos:end] = v
            else:                                    # wrap around
                first = self.capacity - self.pos
                self.buf[self.pos:] = v[:first]
                self.buf[:k - first] = v[first:]
            self.pos = end % self.capacity
            self.n_filled = min(self.n_filled + k, self.capacity)
        self.n_seen += k

    @property
    def ready(self) -> bool:
        return self.n_filled >= self.warmup

    @torch.no_grad()
    def q(self, p: float) -> torch.Tensor:
        """Quantile p of the stored values, as a 0-dim tensor on the buffer's device."""
        if self.n_filled == 0:
            return torch.zeros((), device=self.buf.device)
        return torch.quantile(self.buf[: self.n_filled], p)

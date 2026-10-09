"""The spiking network: two hidden layers and one or several linear heads."""

import torch
import torch.nn as nn

from .neurons import ALIFCell, LIFCell


class SpikingMLP(nn.Module):
    """Spiking multilayer perceptron: n_in -> h1 -> h2 -> linear head(s).

    The forward pass returns the logits together with spike statistics for
    every sample of the batch. The decision module needs them per sample; a
    batch mean would give every sample of a batch the same signal.

    multi_head=True   task-incremental: one head with `classes_per_task`
                      outputs per task, selected by `task_id`
    multi_head=False  class-incremental: a single head over all classes
    """

    def __init__(
        self,
        n_in: int = 784,
        h1: int = 256,
        h2: int = 256,
        n_tasks: int = 5,
        classes_per_task: int = 2,
        multi_head: bool = True,
        adaptive: bool = True,
        beta: float = 0.6,
    ):
        super().__init__()
        self.multi_head = multi_head
        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task

        self.fc1 = nn.Linear(n_in, h1)
        self.fc2 = nn.Linear(h1, h2)

        cell = (lambda n: ALIFCell(n, beta=beta)) if adaptive else (lambda n: LIFCell(n))
        self.cell1, self.cell2 = cell(h1), cell(h2)

        if multi_head:
            self.heads = nn.ModuleList(
                [nn.Linear(h2, classes_per_task) for _ in range(n_tasks)]
            )
        else:
            self.heads = nn.ModuleList([nn.Linear(h2, n_tasks * classes_per_task)])

    def forward(self, spikes: torch.Tensor, task_id: int = 0, collect_raster: bool = False):
        """Simulate T timesteps.

        Args:
            spikes:         (T, B, n_in) input spikes
            task_id:        index of the head (ignored for a single head)
            collect_raster: also return the hidden spike trains, shape (T, B, H)
        Returns:
            logits: (B, C), the head output averaged over the timesteps
            info:   per-sample firing rates `rate_l1`, `rate_l2` of shape (B,),
                    spike counts `count_l1`, `count_l2` of shape (B, H), and, if
                    requested, the rasters `spikes_l1`, `spikes_l2`
        """
        T, B, _ = spikes.shape
        dev = spikes.device
        s1 = self.cell1.init_state(B, dev)
        s2 = self.cell2.init_state(B, dev)

        out_acc = 0.0
        # per-sample, per-neuron spike counts -> (B, H)
        cnt1 = torch.zeros(B, self.cell1.n, device=dev)
        cnt2 = torch.zeros(B, self.cell2.n, device=dev)
        vth1_sum = 0.0
        r1_list, r2_list = [], []

        for t in range(T):
            z1, s1, vth1 = self.cell1(self.fc1(spikes[t]), s1)
            z2, s2, _ = self.cell2(self.fc2(z1), s2)
            head = self.heads[task_id] if self.multi_head else self.heads[0]
            out_acc = out_acc + head(z2)
            cnt1 = cnt1 + z1
            cnt2 = cnt2 + z2
            vth1_sum = vth1_sum + vth1.mean()
            if collect_raster:
                r1_list.append(z1.detach())
                r2_list.append(z2.detach())

        logits = out_acc / T

        info = {
            # per-sample firing rates, shape (B,)
            "rate_l1": cnt1.mean(dim=1) / T,
            "rate_l2": cnt2.mean(dim=1) / T,
            # per-sample, per-neuron counts, shape (B, H)
            "count_l1": cnt1,
            "count_l2": cnt2,
            "mean_vth_l1": vth1_sum / T,
            "T": T,
        }
        if collect_raster:
            info["spikes_l1"] = torch.stack(r1_list)   # (T, B, H1)
            info["spikes_l2"] = torch.stack(r2_list)
        return logits, info

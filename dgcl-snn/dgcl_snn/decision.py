"""The per-sample decision module and the random-skip control."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import torch
import torch.nn.functional as F

from .quantile import RunningQuantile


class Action(IntEnum):
    """Plasticity action assigned to a training sample."""
    UPDATE = 0
    CONSOLIDATE = 1
    PROTECT = 2
    IGNORE = 3


ACTION_NAMES = {int(a): a.name for a in Action}


@dataclass
class DecisionConfig:
    """Percentiles and switches of the decision rule."""
    p_attention_low: float = 0.25    # attention below this quantile: IGNORE
    p_entropy_high: float = 0.75     # entropy above this quantile: UPDATE
    p_entropy_low: float = 0.40      # entropy below this quantile: PROTECT, if the gate is open
    omega_protect_p: float = 0.70    # quantile of the mean-Omega window that opens the gate
    min_warmup_samples: int = 512
    enable_protect: bool = True
    enable_consolidate: bool = True


class PerSampleDecisionModule:
    """Assigns one of four plasticity actions to every sample of a batch.

    Per-sample inputs, both taken from the training forward pass:

        H_i    predictive entropy of the output distribution
        a_i    attention: H_i times the normalised mean firing rate

    Global input:

        gate   open if the mean SI importance exceeds the `omega_protect_p`
               quantile of its own recent values

    Rule, in priority order (a later line applies only if no earlier one does):

        IGNORE       a_i <  tau_a   (a_i <= tau_a if tau_a is zero)
        UPDATE       H_i >  tau_H_high
        PROTECT      H_i <  tau_H_low  and the gate is open
        CONSOLIDATE  otherwise

    The thresholds are quantiles of a sliding window of the respective signal.
    Until `min_warmup_samples` values have been seen, every sample is UPDATE.

    An action is a weight on the sample's loss (`WEIGHTS`), so one backward
    pass over the weighted mean implements all four actions.
    """

    WEIGHTS = {
        Action.UPDATE: 1.0,
        Action.CONSOLIDATE: 0.3,
        Action.PROTECT: 0.1,
        Action.IGNORE: 0.0,
    }

    def __init__(self, cfg: DecisionConfig):
        self.cfg = cfg
        self.q_att = RunningQuantile(warmup=cfg.min_warmup_samples)
        self.q_H = RunningQuantile(warmup=cfg.min_warmup_samples)
        self.q_omega = RunningQuantile(warmup=1)
        self.log: List[Dict] = []
        self._task_counts: Dict[int, Dict[int, int]] = {}
        self._gpu_counts: Dict[int, torch.Tensor] = {}
        self._n_steps = 0

    # ---- signals -------------------------------------------------------
    @staticmethod
    def entropy(logits: torch.Tensor) -> torch.Tensor:
        """Predictive entropy of each sample, in nats."""
        p = F.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-8)).sum(dim=-1)

    @staticmethod
    def attention(H: torch.Tensor, rate_l1: torch.Tensor, rate_l2: torch.Tensor) -> torch.Tensor:
        """Attention signal of each sample.

        The entropy is multiplied by the mean firing rate of the two hidden layers,
        normalised from [0.05, 0.95] to [0, 1]. The rates are per sample; with
        layer means the factor would be the same for all samples of a batch.
        """
        rate = 0.5 * (rate_l1 + rate_l2)
        norm = ((rate - 0.05) / 0.90).clamp(0.0, 1.0)
        return H * norm

    # ---- decision ------------------------------------------------------
    @torch.no_grad()
    def decide(
        self,
        H: torch.Tensor,
        att: torch.Tensor,
        mean_omega: float,
        task_id: int,
        step: int,
    ) -> torch.Tensor:
        """Assign an action to every sample.

        Args:
            H, att:     (B,) entropy and attention, detached
            mean_omega: mean SI importance (a Python float)
            task_id:    used to count the actions per task
            step:       not used by the rule
        Returns:
            (B,) long tensor of `Action` values, on the device of the signals
        """
        dev = H.device
        if self.q_H.device != dev:
            self.q_H.to(dev)
            self.q_att.to(dev)
            self.q_omega.to(dev)

        self.q_H.update(H)
        self.q_att.update(att)
        self.q_omega.update(torch.as_tensor([mean_omega], device=dev))

        B = H.shape[0]
        actions = torch.full((B,), int(Action.CONSOLIDATE), dtype=torch.long, device=dev)

        if not (self.q_att.ready and self.q_H.ready):
            # Warm-up: every sample is UPDATE until the quantiles can be
            # estimated. Otherwise the first batches of the first task would
            # be judged against thresholds from a handful of values.
            actions.fill_(int(Action.UPDATE))
            self._record(actions, task_id, step, H, att)
            return actions

        tau_att = self.q_att.q(self.cfg.p_attention_low)
        tau_H_hi = self.q_H.q(self.cfg.p_entropy_high)
        tau_H_lo = self.q_H.q(self.cfg.p_entropy_low)
        omega_gate = (self.cfg.enable_protect
                      and mean_omega > float(self.q_omega.q(self.cfg.omega_protect_p)))

        if not self.cfg.enable_consolidate:
            actions.fill_(int(Action.UPDATE))

        m_update = H > tau_H_hi
        actions[m_update] = int(Action.UPDATE)

        if omega_gate:
            m_protect = (H < tau_H_lo) & (~m_update)
            actions[m_protect] = int(Action.PROTECT)

        # Attention is exactly zero whenever the firing rate is below the lower
        # rate bound. If the threshold itself is zero, a strict '<' would never
        # be true and IGNORE could not occur, so '<=' is used in that case.
        m_ignore = (att <= tau_att) if float(tau_att) <= 0.0 else (att < tau_att)
        actions[m_ignore] = int(Action.IGNORE)

        self._record(actions, task_id, step, H, att)
        return actions

    def _record(self, actions, task_id, step, H, att) -> None:
        """Add the actions of one batch to the per-task counts (kept on the device)."""
        dev = actions.device
        c = self._gpu_counts.get(task_id)
        if c is None or c.device != dev:
            c = torch.zeros(len(Action), dtype=torch.long, device=dev)
            self._gpu_counts[task_id] = c
        c += torch.bincount(actions, minlength=len(Action))
        self._n_steps += 1

    def _flush(self, task_id: int) -> None:
        """Copy the counts of one task to the host. Called once per task."""
        c = self._gpu_counts.get(task_id)
        if c is None:
            return
        self._task_counts[task_id] = {int(a): int(v) for a, v in
                                      zip(Action, c.tolist())}

    # ---- weights -------------------------------------------------------
    def sample_weights(self, actions: torch.Tensor, device) -> torch.Tensor:
        """Loss weight of every sample."""
        w = torch.zeros(actions.shape[0], device=device)
        for a, val in self.WEIGHTS.items():
            w[actions == int(a)] = val
        return w

    def task_distribution(self, task_id: int) -> Dict[str, float]:
        """Share of each action within one task."""
        self._flush(task_id)
        c = self._task_counts.get(task_id, {})
        tot = max(sum(c.values()), 1)
        return {ACTION_NAMES[k]: v / tot for k, v in c.items()}

    def global_distribution(self) -> Dict[str, float]:
        """Share of each action over all tasks."""
        for t in list(self._gpu_counts):
            self._flush(t)
        agg = {int(a): 0 for a in Action}
        for c in self._task_counts.values():
            for k, v in c.items():
                agg[k] += v
        tot = max(sum(agg.values()), 1)
        return {ACTION_NAMES[k]: v / tot for k, v in agg.items()}


class RandomSkipController:
    """Control: ignores a fixed share of the samples, chosen uniformly at random.

    A kept sample is UPDATE (weight 1), an ignored one IGNORE (weight 0). The
    controller never assigns CONSOLIDATE or PROTECT, so it also never raises
    the regularisation strength. It is matched to the decision module in the
    share of ignored samples only.
    """

    def __init__(self, skip_rate: float):
        self.skip_rate = skip_rate
        self.log: List[Dict] = []
        self._task_counts: Dict[int, Dict[int, int]] = {}
        self._gpu_counts: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def decide(self, H, att, mean_omega, task_id, step) -> torch.Tensor:
        """Same interface as the decision module; the signals are not used."""
        dev = H.device
        B = H.shape[0]
        keep = torch.rand(B, device=dev) >= self.skip_rate
        actions = torch.where(
            keep,
            torch.full((B,), int(Action.UPDATE), dtype=torch.long, device=dev),
            torch.full((B,), int(Action.IGNORE), dtype=torch.long, device=dev),
        )
        c = self._gpu_counts.get(task_id)
        if c is None or c.device != dev:
            c = torch.zeros(len(Action), dtype=torch.long, device=dev)
            self._gpu_counts[task_id] = c
        c += torch.bincount(actions, minlength=len(Action))
        return actions

    def _flush(self, task_id: int) -> None:
        c = self._gpu_counts.get(task_id)
        if c is not None:
            self._task_counts[task_id] = {int(a): int(v) for a, v in
                                          zip(Action, c.tolist())}

    def sample_weights(self, actions: torch.Tensor, device) -> torch.Tensor:
        """Loss weight of every sample."""
        w = torch.zeros(actions.shape[0], device=device)
        w[actions == int(Action.UPDATE)] = 1.0
        return w

    def task_distribution(self, task_id: int) -> Dict[str, float]:
        """Share of each action within one task."""
        self._flush(task_id)
        c = self._task_counts.get(task_id, {})
        tot = max(sum(c.values()), 1)
        return {ACTION_NAMES[k]: v / tot for k, v in c.items()}

    def global_distribution(self) -> Dict[str, float]:
        """Share of each action over all tasks."""
        for t in list(self._gpu_counts):
            self._flush(t)
        agg = {int(a): 0 for a in Action}
        for c in self._task_counts.values():
            for k, v in c.items():
                agg[k] += v
        tot = max(sum(agg.values()), 1)
        return {ACTION_NAMES[k]: v / tot for k, v in agg.items()}

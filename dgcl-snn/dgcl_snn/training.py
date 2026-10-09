"""Training of one task and evaluation."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .accounting import ComputeAccountant
from .decision import Action, PerSampleDecisionModule
from .encoding import poisson_encode
from .si import SynapticIntelligence

if TYPE_CHECKING:                       # import for type hints only
    from .experiment import ExpConfig


@torch.no_grad()
def evaluate(model, loader, task_id, device, t_steps, multi_head=True) -> Tuple[float, float]:
    """Accuracy and mean predictive entropy on one task."""
    model.eval()
    correct = total = 0
    H_sum = 0.0
    for x, y in loader:
        if x.device != device:
            x, y = x.to(device), y.to(device)
        sp = poisson_encode(x, t_steps)
        logits, _ = model(sp, task_id=task_id if multi_head else 0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += y.numel()
        H_sum += float(PerSampleDecisionModule.entropy(logits).sum().item())
    model.train()
    return correct / max(total, 1), H_sum / max(total, 1)


def train_task(
    model,
    si: Optional[SynapticIntelligence],
    controller,                       # decision module, random-skip control or None
    optimizer,
    loader,
    task_id: int,
    device,
    cfg: "ExpConfig",
    accountant: ComputeAccountant,
    epoch_offset: int = 0,
) -> List[Dict]:
    """Train one task for `cfg.epochs` epochs.

    The signals of the decision module are computed from the forward pass that
    also produces the loss, so no additional forward pass is needed. If every
    sample of a batch is ignored, the backward pass and the optimiser step are
    skipped.

    `controller` is a `PerSampleDecisionModule`, a `RandomSkipController` or
    None (every sample has weight 1). Returns one log entry per epoch.
    """
    model.train()
    logs = []
    step = epoch_offset
    dev = device

    for epoch in range(cfg.epochs):
        # Accumulators stay on the device. Reading any of them with .item()
        # inside the loop would synchronise the GPU on every batch, which for a
        # model this small costs more than the arithmetic itself.
        acc_loss = torch.zeros((), device=dev)
        acc_si = torch.zeros((), device=dev)
        acc_acc = torch.zeros((), device=dev)
        acc_H = torch.zeros((), device=dev)
        acc_active = torch.zeros((), device=dev)
        n_batch = 0
        n_ignored_batches = 0

        for x, y in loader:
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)
            B = y.numel()

            sp = poisson_encode(x, cfg.t_steps)

            # ---- single forward pass -----------------------------------
            logits, info = model(sp, task_id=task_id if cfg.multi_head else 0)
            batch_fwd_synops = accountant.charge_forward(sp, info)

            # ---- signals, per sample, from the same forward pass --------
            H = PerSampleDecisionModule.entropy(logits)
            att = PerSampleDecisionModule.attention(H, info["rate_l1"], info["rate_l2"])

            # ---- decision ----------------------------------------------
            if controller is None:
                w = torch.ones(B, device=dev)
                n_active_t = torch.full((), float(B), device=dev)
                frac_protect = 0.0
                all_ignored = False
            else:
                mo = si.mean_omega() if si is not None else 0.0
                actions = controller.decide(H.detach(), att.detach(), mo, task_id, step)
                w = controller.sample_weights(actions, dev)
                n_active_t = (w > 0).sum()
                # one cheap scalar read: needed to know whether to skip the
                # backward pass at all
                n_active = int(n_active_t)
                all_ignored = (n_active == 0)
                frac_protect = float((actions == int(Action.PROTECT)).float().mean()) \
                    if not all_ignored else 0.0

            if all_ignored:
                accountant.note_skipped_batch()
                step += 1
                n_batch += 1
                n_ignored_batches += 1
                continue

            # ---- weighted loss -----------------------------------------
            per_sample = F.cross_entropy(logits, y, reduction="none")
            task_loss = (per_sample * w).sum() / w.sum().clamp(min=1e-8)

            loss = task_loss
            si_val = torch.zeros((), device=dev)
            if si is not None:
                lam_scale = 1.0 + (cfg.protect_lambda_boost - 1.0) * frac_protect
                base = si.lambda_si
                si.lambda_si = base * lam_scale
                pen = si.penalty()
                si.lambda_si = base
                loss = loss + pen
                si_val = pen.detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if si is not None:
                si.accumulate_step()

            accountant.charge_backward(int(n_active_t), B, batch_fwd_synops)

            with torch.no_grad():
                acc_loss += task_loss.detach()
                acc_si += si_val
                acc_acc += (logits.argmax(1) == y).float().mean()
                acc_H += H.mean().detach()
                acc_active += n_active_t / B
            step += 1
            n_batch += 1

        # one synchronisation per epoch rather than one per batch
        nb_ = max(n_batch, 1)
        logs.append({
            "task_id": task_id,
            "epoch": epoch + 1,
            "loss": float(acc_loss) / nb_,
            "si_penalty": float(acc_si) / nb_,
            "train_acc": float(acc_acc) / nb_,
            "entropy": float(acc_H) / nb_,
            "active_frac": float(acc_active) / nb_,
            "ignored_batches": n_ignored_batches,
        })
    return logs

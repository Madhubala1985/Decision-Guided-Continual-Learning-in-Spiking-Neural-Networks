"""Neuron roles derived from the combined importance (analysis only)."""

from enum import IntEnum
from typing import Dict, List, Tuple

import torch


class MemoryState(IntEnum):
    """Role of a hidden neuron, assigned from its combined importance."""
    FAST_PLASTIC = 0     # low combined importance
    ATTENTION_GATE = 1   # intermediate
    STABLE_CONSOL = 2    # high combined importance


MEMORY_NAMES = {int(m): m.name for m in MemoryState}


class FunctionalMemoryLayer:
    """Neuron roles derived from the combined importance. Analysis only.

    After each task every hidden neuron is assigned one of three roles:

        FAST_PLASTIC    combined importance at or below the lower boundary
        ATTENTION_GATE  between the boundaries
        STABLE_CONSOL   at or above the upper boundary

    In addition, the `consol_top_k` fraction of neurons with the highest
    combined importance is tagged as consolidated. A tag is never removed.

    The roles and tags are recorded and reported. They are not used by the
    training loop, the loss or the decision rule.

    The boundaries are percentiles of the importance distribution after the
    first task and are then held fixed. Boundaries recomputed after every task
    would keep the three groups at the same size by construction.
    """

    def __init__(self, n_l1: int, n_l2: int,
                 p_low: float = 0.33, p_high: float = 0.67,
                 consol_top_k: float = 0.10):
        self.p_low, self.p_high = p_low, p_high
        self.consol_top_k = consol_top_k
        self.consolidated_mask = {"l1": torch.zeros(n_l1, dtype=torch.bool),
                                  "l2": torch.zeros(n_l2, dtype=torch.bool)}
        self.history: List[Dict] = []          # one snapshot per task
        self.transitions: List[Dict] = []      # newly tagged neurons per task
        self.thresholds: Dict[str, Tuple[float, float]] = {}   # fixed after task 1
        self._prev_states: Dict[str, torch.Tensor] = {}        # to count role changes

    # ---- combined importance --------------------------------------------
    @staticmethod
    def combined_importance(omega_n: torch.Tensor, isi_imp: torch.Tensor,
                            alpha: float = 0.7, beta: float = 0.3) -> torch.Tensor:
        """Weighted sum of SI importance and firing-regularity importance."""
        n = min(omega_n.numel(), isi_imp.numel())
        return alpha * omega_n[:n] + beta * isi_imp[:n]

    # ---- classification --------------------------------------------------
    def classify(self, layer: str, imp: torch.Tensor) -> torch.Tensor:
        """Assign a role to every neuron of a layer.

        The boundaries are set from the first importance vector seen for the layer
        (after the first task) and reused afterwards.
        """
        if imp.numel() == 0:
            return torch.zeros(0, dtype=torch.long)
        if layer not in self.thresholds:
            self.thresholds[layer] = (float(torch.quantile(imp, self.p_low)),
                                      float(torch.quantile(imp, self.p_high)))
        lo, hi = self.thresholds[layer]
        states = torch.full((imp.numel(),), int(MemoryState.ATTENTION_GATE),
                            dtype=torch.long)
        states[imp <= lo] = int(MemoryState.FAST_PLASTIC)
        states[imp >= hi] = int(MemoryState.STABLE_CONSOL)
        return states

    # ---- consolidation ---------------------------------------------------
    def tag_consolidated(self, layer: str, imp: torch.Tensor) -> int:
        """Tag the top-k fraction by importance as consolidated.

        Returns the number of neurons that were tagged for the first time.
        """
        mask = self.consolidated_mask[layer]
        n = min(imp.numel(), mask.numel())
        if n == 0:
            return 0
        k = max(1, int(self.consol_top_k * n))
        top = torch.topk(imp[:n], k).indices
        new = int((~mask[top]).sum().item())
        mask[top] = True
        return new

    # ---- snapshot --------------------------------------------------------
    def snapshot(self, task_id: int, imp_l1: torch.Tensor,
                 imp_l2: torch.Tensor) -> Dict:
        """Record roles, role changes and tags of both layers after a task."""
        rec = {"task_id": int(task_id)}
        total_new = 0
        for layer, imp in (("l1", imp_l1), ("l2", imp_l2)):
            st = self.classify(layer, imp)
            counts = {MEMORY_NAMES[int(m)]: int((st == int(m)).sum().item())
                      for m in MemoryState}
            prev = self._prev_states.get(layer)
            churn = int((st != prev).sum().item()) if prev is not None else 0
            self._prev_states[layer] = st.clone()
            total_new += self.tag_consolidated(layer, imp)
            rec[layer] = {
                "counts": counts,
                "churn": churn,
                "n_consolidated": int(self.consolidated_mask[layer].sum().item()),
                "imp_mean": float(imp.mean()) if imp.numel() else 0.0,
                "imp_std": float(imp.std()) if imp.numel() > 1 else 0.0,
                "states": st.tolist(),
            }
        rec["new_transitions"] = total_new
        self.transitions.append({"task_id": int(task_id), "new_ltm": total_new})
        self.history.append(rec)
        return rec

    def summary(self) -> List[Dict]:
        """History without the per-neuron role vectors."""
        out = []
        for r in self.history:
            out.append({
                "task_id": r["task_id"],
                "new_transitions": r["new_transitions"],
                **{f"{lay}_{k}": v
                   for lay in ("l1", "l2")
                   for k, v in {**r[lay]["counts"],
                                "churn": r[lay]["churn"],
                                "n_consolidated": r[lay]["n_consolidated"],
                                "imp_mean": round(r[lay]["imp_mean"], 4),
                                "imp_std": round(r[lay]["imp_std"], 4)}.items()},
            })
        return out

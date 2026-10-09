import torch

from dgcl_snn import FunctionalMemoryLayer, MemoryState


def test_combined_importance():
    imp = FunctionalMemoryLayer.combined_importance(torch.tensor([1.0, 0.0]),
                                                    torch.tensor([0.0, 1.0]), 0.7, 0.3)
    assert torch.allclose(imp, torch.tensor([0.7, 0.3]))


def test_boundaries_are_fixed_after_the_first_task():
    fml = FunctionalMemoryLayer(9, 9)
    first = fml.classify("l1", torch.linspace(0, 1, 9))
    assert (first == int(MemoryState.FAST_PLASTIC)).sum() == 3
    assert (first == int(MemoryState.ATTENTION_GATE)).sum() == 3
    assert (first == int(MemoryState.STABLE_CONSOL)).sum() == 3
    lo, hi = fml.thresholds["l1"]
    later = fml.classify("l1", torch.ones(9))        # all importance values are now high
    assert fml.thresholds["l1"] == (lo, hi)
    assert (later == int(MemoryState.STABLE_CONSOL)).sum() == 9


def test_consolidated_tags_are_permanent():
    fml = FunctionalMemoryLayer(20, 20, consol_top_k=0.10)
    imp = torch.arange(20.0)
    assert fml.tag_consolidated("l1", imp) == 2      # neurons 18 and 19
    assert fml.tag_consolidated("l1", imp) == 0      # the same neurons again
    assert fml.tag_consolidated("l1", -imp) == 2     # neurons 0 and 1
    assert int(fml.consolidated_mask["l1"].sum()) == 4


def test_snapshot_counts_role_changes():
    fml = FunctionalMemoryLayer(9, 9)
    imp = torch.linspace(0, 1, 9)
    rec = fml.snapshot(0, imp, imp)
    assert rec["l1"]["churn"] == 0 and rec["new_transitions"] == 2
    rec = fml.snapshot(1, imp.flip(0), imp)
    assert rec["l1"]["churn"] == 6                   # the outer thirds swap roles
    assert rec["l2"]["churn"] == 0

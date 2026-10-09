import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    """Every test starts from the same random state."""
    torch.manual_seed(0)


@pytest.fixture
def cpu():
    return torch.device("cpu")

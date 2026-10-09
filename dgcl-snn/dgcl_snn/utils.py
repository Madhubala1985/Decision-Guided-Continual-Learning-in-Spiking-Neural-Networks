"""Seeding and device selection."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random number generators.

    Called once at the start of every run. Data shuffling, parameter
    initialisation, Poisson encoding and the random-skip control all draw from
    these generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """CUDA if it is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_device(verbose: bool = True) -> torch.device:
    """Return a device on which this PyTorch build actually runs.

    `torch.cuda.is_available()` can be true for a GPU whose architecture the
    installed PyTorch build does not support; every CUDA call then fails.
    The function runs a small computation on the GPU and falls back to the
    CPU if that fails.
    """
    if not torch.cuda.is_available():
        if verbose:
            print("No GPU available, using the CPU.")
        return torch.device("cpu")
    name = torch.cuda.get_device_name(0)
    try:
        a = torch.randn(64, 64, device="cuda")
        _ = (a @ a).sum().item()
        _ = torch.bernoulli(torch.rand(8, 16, device="cuda")).sum().item()
    except Exception as e:                      # unsupported GPU architecture
        if verbose:
            print(f"GPU '{name}' cannot run this PyTorch build "
                  f"({type(e).__name__}: {e}); using the CPU.")
        return torch.device("cpu")
    if verbose:
        print(f"Using GPU '{name}' with PyTorch {torch.__version__}.")
    return torch.device("cuda")

"""Split-MNIST as a sequence of five two-class tasks."""

from typing import Optional

import torch

from .utils import get_device

TASK_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


class GPUBatches:
    """Mini-batches from a tensor dataset that is held on the device.

    Split-MNIST fits into device memory. The whole task is moved to the device
    once and every batch is an index into that tensor, which avoids one
    host-to-device copy per batch.

    With `shuffle=True` the order is a new random permutation in every epoch,
    drawn from the global PyTorch generator, so runs are reproducible under
    `set_seed()`.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 64,
                 shuffle: bool = False, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.x = x.to(self.device, non_blocking=True)
        self.y = y.to(self.device, non_blocking=True)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (self.y.numel() + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.y.numel()
        idx = (torch.randperm(n, device=self.device) if self.shuffle
               else torch.arange(n, device=self.device))
        for i in range(0, n, self.batch_size):
            j = idx[i:i + self.batch_size]
            yield self.x[j], self.y[j]


def _load_mnist_tensors(data_dir="./data"):
    """Return (train_x, train_y, test_x, test_y) with pixel values in [0, 1].

    Looks for MNIST in `data_dir` and in the usual Kaggle input directories,
    and downloads it with torchvision if it is not found.
    """
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()

    candidates = [data_dir,
                  "/kaggle/input/mnist-dataset",
                  "/kaggle/input/mnist",
                  "/kaggle/working/data"]
    last_err = None
    for root in candidates:
        for dl in (False, True):
            try:
                tr = datasets.MNIST(root, train=True,  download=dl, transform=tf)
                te = datasets.MNIST(root, train=False, download=dl, transform=tf)
                trx = tr.data.float().div_(255.0).unsqueeze(1)
                tex = te.data.float().div_(255.0).unsqueeze(1)
                print(f"    MNIST loaded from '{root}' "
                      f"(train {tuple(trx.shape)}, test {tuple(tex.shape)})")
                return trx, tr.targets.clone(), tex, te.targets.clone()
            except Exception as e:
                last_err = e
    raise RuntimeError(
        "Could not load MNIST. Check the internet connection, or place the "
        f"dataset in '{data_dir}'. Last error: {last_err}")


def build_split_mnist(
    data_dir: str = "./data",
    batch_size: int = 64,
    multi_head: bool = True,
    synthetic: bool = False,
    n_synth: int = 512,
    device: Optional[torch.device] = None,
):
    """Split-MNIST: five two-class tasks, 0/1, 2/3, 4/5, 6/7, 8/9.

    multi_head=True   labels are remapped to {0, 1}        (task-incremental)
    multi_head=False  labels keep the digit 0..9           (class-incremental)

    `synthetic=True` builds a small random dataset of the same shape. It needs
    no download and is used for tests and smoke runs only.

    Returns a list with one dict per task: `task_id`, `digits`,
    `train_loader`, `test_loader`, `n_train`, `n_test`.
    """
    if synthetic:
        tasks = []
        g = torch.Generator().manual_seed(0)
        for tid, (a, b) in enumerate(TASK_PAIRS):
            xs, ys = [], []
            for k, digit in enumerate((a, b)):
                proto = torch.rand(1, 1, 28, 28, generator=g) * 0.6
                x = (proto + 0.25 * torch.rand(n_synth // 2, 1, 28, 28,
                                               generator=g)).clamp(0, 1)
                y = torch.full((n_synth // 2,), k if multi_head else digit,
                               dtype=torch.long)
                xs.append(x)
                ys.append(y)
            X, Y = torch.cat(xs), torch.cat(ys)
            n_tr = int(0.8 * len(X))
            tasks.append({
                "task_id": tid, "digits": (a, b),
                "train_loader": GPUBatches(X[:n_tr], Y[:n_tr], batch_size=batch_size,
                                           shuffle=True, device=device),
                "test_loader": GPUBatches(X[n_tr:], Y[n_tr:], batch_size=batch_size,
                                          shuffle=False, device=device),
                "n_train": n_tr, "n_test": len(X) - n_tr,
            })
        return tasks

    trx, try_, tex, tey = _load_mnist_tensors(data_dir)

    tasks = []
    for tid, (a, b) in enumerate(TASK_PAIRS):
        out = {}
        for split, X, Y in (("train", trx, try_), ("test", tex, tey)):
            m = (Y == a) | (Y == b)
            xs = X[m]
            ys = (Y[m] == b).long() if multi_head else Y[m].clone()
            out[split] = (xs, ys)
        tasks.append({
            "task_id": tid, "digits": (a, b),
            "train_loader": GPUBatches(*out["train"], batch_size=batch_size,
                                       shuffle=True, device=device),
            "test_loader": GPUBatches(*out["test"], batch_size=batch_size,
                                      shuffle=False, device=device),
            "n_train": len(out["train"][1]), "n_test": len(out["test"][1]),
        })
        print(f"    task {tid+1}  digits {a} vs {b}   "
              f"train {len(out['train'][1]):>6}   test {len(out['test'][1]):>5}")
    return tasks

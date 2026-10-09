"""Per-sample plasticity control for continual learning in spiking neural networks.

The package contains the model, the signals, the decision module, the
random-skip control, Synaptic Intelligence, the cost model, the metrics and
the training loop. The scripts in `scripts/` run the experiments and produce
the tables and figures.
"""

from .accounting import ComputeAccountant
from .data import TASK_PAIRS, GPUBatches, build_split_mnist
from .decision import (
    ACTION_NAMES,
    Action,
    DecisionConfig,
    PerSampleDecisionModule,
    RandomSkipController,
)
from .diagnostics import collect_diagnostics
from .encoding import poisson_encode
from .experiment import ExpConfig, run_experiment
from .isi import ISICVEstimator
from .metrics import aggregate, cl_metrics, cohens_d, mean_sd, sample_std, welch_t
from .network import SpikingMLP
from .neurons import ALIFCell, ALIFState, LIFCell, SuperSpike, spike_fn
from .quantile import RunningQuantile
from .roles import MEMORY_NAMES, FunctionalMemoryLayer, MemoryState
from .si import SynapticIntelligence
from .training import evaluate, train_task
from .utils import get_device, select_device, set_seed

__version__ = "1.0.0"

__all__ = [
    "ACTION_NAMES",
    "ALIFCell",
    "ALIFState",
    "Action",
    "ComputeAccountant",
    "DecisionConfig",
    "ExpConfig",
    "FunctionalMemoryLayer",
    "GPUBatches",
    "ISICVEstimator",
    "LIFCell",
    "MEMORY_NAMES",
    "MemoryState",
    "PerSampleDecisionModule",
    "RandomSkipController",
    "RunningQuantile",
    "SpikingMLP",
    "SuperSpike",
    "SynapticIntelligence",
    "TASK_PAIRS",
    "aggregate",
    "build_split_mnist",
    "cl_metrics",
    "cohens_d",
    "collect_diagnostics",
    "evaluate",
    "get_device",
    "mean_sd",
    "poisson_encode",
    "run_experiment",
    "sample_std",
    "select_device",
    "set_seed",
    "spike_fn",
    "train_task",
    "welch_t",
]

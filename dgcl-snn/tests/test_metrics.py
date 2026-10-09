import math

import numpy as np

from dgcl_snn import aggregate, cl_metrics, cohens_d, mean_sd, sample_std, welch_t


def test_metrics_from_the_accuracy_matrix():
    A = np.array([[0.90, 0.00, 0.00],
                  [0.80, 0.95, 0.00],
                  [0.70, 0.90, 0.99]])
    m = cl_metrics(A)
    assert abs(m["avg_acc"] - (0.70 + 0.90 + 0.99) / 3) < 1e-12
    assert np.allclose(m["per_task_forgetting"], [0.20, 0.05])
    assert abs(m["avg_forgetting"] - 0.125) < 1e-12
    assert abs(m["bwt"] + 0.125) < 1e-12


def test_forgetting_can_be_negative():
    A = np.array([[0.90, 0.0], [0.95, 0.8]])
    assert abs(cl_metrics(A)["avg_forgetting"] + 0.05) < 1e-12


def test_sample_standard_deviation():
    assert sample_std([1.0, 2.0, 3.0]) == 1.0       # n - 1 in the denominator
    assert sample_std([5.0]) == 0.0
    assert mean_sd([1.0, 2.0, 3.0]) == (2.0, 1.0)


def test_effect_size_and_welch_t():
    a = [0.0010, 0.0046, 0.0128]
    b = [0.0006, 0.0025, 0.0014]
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    diff = np.mean(a) - np.mean(b)
    assert abs(cohens_d(a, b) - diff / math.sqrt((sa**2 + sb**2) / 2)) < 1e-9
    assert abs(welch_t(a, b) - diff / math.sqrt(sa**2 / 3 + sb**2 / 3)) < 1e-9
    assert abs(cohens_d(a, b) - 1.07) < 0.01
    assert abs(welch_t(a, b) - 1.31) < 0.01
    assert math.isnan(cohens_d([1.0], [2.0]))


def test_aggregate_over_seeds():
    def run(seed, acc, fgt):
        return {"config": "X", "seed": seed, "avg_acc": acc, "avg_forgetting": fgt, "bwt": -fgt,
                "compute": {"synops_total_G": 10.0 + seed, "synops_total_event_G": 8.0,
                            "backward_sample_fraction": 0.7}}
    agg = aggregate([run(0, 0.90, 0.10), run(1, 0.92, 0.08), run(2, 0.94, 0.06)])
    assert agg["n_seeds"] == 3 and agg["seeds"] == [0, 1, 2]
    assert abs(agg["avg_acc_mean"] - 0.92) < 1e-12
    assert abs(agg["avg_acc_std"] - 0.02) < 1e-12
    assert abs(agg["synops_G_std"] - 1.0) < 1e-12
    assert agg["synops_event_G_std"] == 0.0

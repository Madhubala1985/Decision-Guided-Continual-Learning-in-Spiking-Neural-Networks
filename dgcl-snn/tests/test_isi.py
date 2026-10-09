import torch

from dgcl_snn import ISICVEstimator


def raster(times, T=20):
    """(T, 1, 1) spike train with spikes at the given timesteps."""
    s = torch.zeros(T, 1, 1)
    s[list(times)] = 1.0
    return s


def test_regular_train_has_cv_zero_and_importance_one():
    est = ISICVEstimator(min_isi=2)
    cv, valid = est.cv_from_raster(raster(range(0, 20, 2)))
    assert valid.item() and abs(cv.item()) < 1e-6
    imp, _ = est.neuron_importance(raster(range(0, 20, 2)))
    assert abs(imp.item() - 1.0) < 1e-6


def test_known_coefficient_of_variation():
    # intervals 2, 10, 1, 6: mean 4.75, standard deviation 3.562, CV 0.75
    est = ISICVEstimator(min_isi=2)
    cv, valid = est.cv_from_raster(raster([0, 2, 12, 13, 19]))
    assert valid.item()
    assert abs(cv.item() - 0.7499) < 1e-3
    imp, mean_cv = est.neuron_importance(raster([0, 2, 12, 13, 19]))
    assert abs(imp.item() - 1.0 / 1.7499) < 1e-3
    assert abs(mean_cv.item() - 0.7499) < 1e-3


def test_silent_neuron_gets_importance_zero():
    est = ISICVEstimator(min_isi=2)
    cv, valid = est.cv_from_raster(raster([]))
    assert not valid.item()
    imp, _ = est.neuron_importance(raster([]))
    assert imp.item() == 0.0                        # and not 1.0


def test_one_interval_is_not_enough():
    est = ISICVEstimator(min_isi=2)
    _, valid = est.cv_from_raster(raster([3, 9]))   # two spikes, one interval
    assert not valid.item()
    imp, _ = est.neuron_importance(raster([3, 9]))
    assert imp.item() == 0.0


def test_importance_is_averaged_over_valid_samples_only():
    est = ISICVEstimator(min_isi=2)
    spikes = torch.cat([raster(range(0, 20, 2)), raster([])], dim=1)   # (T, 2, 1)
    imp, _ = est.neuron_importance(spikes)
    assert abs(imp.item() - 1.0) < 1e-6             # the silent sample is not counted

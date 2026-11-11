import os

import numpy as np
import pytest

from hpob_handler import HPOBHandler
from methods.random_search import RandomSearch
from sweep.config import REPO_ROOT
from sweep.recording import RecordingMethod

DATA_DIR = os.path.join(REPO_ROOT, "sweep_data", "hpob-data", "")
SURROGATES_DIR = os.path.join(REPO_ROOT, "sweep_data", "surrogates", "")


@pytest.fixture(scope="module")
def handler():
    return HPOBHandler(root_dir=DATA_DIR, mode="v3-test", surrogates_dir=SURROGATES_DIR)


def test_slice_holds_search_space_5971(handler):
    assert handler.get_search_spaces() == ["5971"]
    assert handler.get_search_space_dim("5971") == 16
    assert sorted(handler.get_datasets("5971")) == sorted(
        ["10093", "3954", "43", "34536", "9970", "6566"]
    )


def test_every_dataset_has_a_surrogate(handler):
    for dataset_id in handler.get_datasets("5971"):
        name = "surrogate-5971-" + dataset_id
        assert os.path.isfile(os.path.join(SURROGATES_DIR, name + ".json"))
        assert handler.surrogates_stats[name]["y_min"] < handler.surrogates_stats[name]["y_max"]


def test_continuous_evaluation_returns_one_incumbent_per_trial_plus_the_start(handler):
    curve = handler.evaluate_continuous(
        RandomSearch(), search_space_id="5971", dataset_id="10093", seed="test0", n_trials=5
    )

    assert len(curve) == 6
    assert all(0.0 <= value <= 1.0 for value in curve)
    assert all(later >= earlier for earlier, later in zip(curve, curve[1:]))


def test_recording_method_keeps_each_suggested_configuration(handler):
    method = RecordingMethod(RandomSearch())

    handler.evaluate_continuous(
        method, search_space_id="5971", dataset_id="3954", seed="test1", n_trials=4
    )

    suggestions = np.array(method.suggestions)
    assert suggestions.shape == (4, 16)
    assert ((suggestions >= 0.0) & (suggestions <= 1.0)).all()


def test_recording_method_passes_discrete_requests_through():
    method = RecordingMethod(RandomSearch())
    X_obs = np.zeros((2, 3))
    X_pen = np.ones((5, 3))

    index = method.observe_and_suggest(X_obs, np.zeros((2, 1)), X_pen)

    assert 0 <= index < 5
    assert method.suggestions == []

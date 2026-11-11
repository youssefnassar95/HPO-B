import json
import os

import pytest

from sweep.config import SweepConfig, _resolve
from sweep.coordinator import run_sweep
from sweep.storage import count_runs


def _config(**overrides):
    values = {
        "experiment_id": "smoke",
        "search_space_id": "5971",
        "dataset_ids": ["10093"],
        "seeds": ["test0", "test1"],
        "methods": {"random": {}},
        "n_trials": 3,
        "base_seed": 0,
        "data_dir": _resolve("sweep_data/hpob-data/"),
        "surrogates_dir": _resolve("sweep_data/surrogates/"),
    }
    values.update(overrides)
    return SweepConfig.from_dict(values)


def test_single_worker_sweep_writes_merged_results_and_a_summary(tmp_path):
    out_dir = str(tmp_path / "run")

    merged = run_sweep(_config(), out_dir, n_workers=1)

    assert count_runs(merged) == 2
    assert all(len(curve) == 4 for curve in merged["random"]["5971"]["10093"].values())
    with open(os.path.join(out_dir, "merged.json")) as f:
        assert json.load(f) == merged
    with open(os.path.join(out_dir, "summary.txt")) as f:
        assert "10093" in f.read()


def test_finished_output_directory_is_not_overwritten_without_resume(tmp_path):
    out_dir = str(tmp_path / "run")
    run_sweep(_config(), out_dir, n_workers=1)

    with pytest.raises(RuntimeError):
        run_sweep(_config(), out_dir, n_workers=1)


def test_resume_only_runs_what_is_missing(tmp_path):
    out_dir = str(tmp_path / "run")
    run_sweep(_config(seeds=["test0"]), out_dir, n_workers=1)

    merged = run_sweep(_config(seeds=["test0", "test1", "test2"]), out_dir, n_workers=1, resume=True)

    assert sorted(merged["random"]["5971"]["10093"]) == ["test0", "test1", "test2"]
    with open(os.path.join(out_dir, "parts", "worker-0.jsonl")) as f:
        assert len(f.read().splitlines()) == 3

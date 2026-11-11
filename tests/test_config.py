import json
import os

import pytest

from sweep.config import REPO_ROOT, Job, SweepConfig, load_config

SHIPPED_CONFIG = os.path.join(REPO_ROOT, "configs", "sweep_5971.json")


def _values(**overrides):
    values = {
        "experiment_id": "unit",
        "search_space_id": "5971",
        "dataset_ids": ["10093", "3954"],
        "seeds": ["test0", "test1", "test2"],
        "methods": {"random": {}},
        "n_trials": 3,
        "base_seed": 7,
        "data_dir": "sweep_data/hpob-data/",
        "surrogates_dir": "sweep_data/surrogates/",
    }
    values.update(overrides)
    return values


def test_shipped_config_lists_every_run():
    config = load_config(SHIPPED_CONFIG)
    jobs = config.jobs()

    assert len(jobs) == len(config.methods) * len(config.dataset_ids) * len(config.seeds)
    assert len({job.job_id for job in jobs}) == len(jobs)


def test_jobs_are_grouped_by_method_then_dataset_then_seed():
    config = SweepConfig.from_dict(_values(methods={"random": {}, "gen-vae": {}}))

    assert [job.job_id for job in config.jobs()[:4]] == [
        "random/5971/10093/test0",
        "random/5971/10093/test1",
        "random/5971/10093/test2",
        "random/5971/3954/test0",
    ]
    assert config.jobs()[-1] == Job("gen-vae", "5971", "3954", "test2")


def test_data_paths_are_resolved_from_the_repository_root(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_values()))

    config = load_config(str(path))

    assert config.data_dir == os.path.join(REPO_ROOT, "sweep_data", "hpob-data", "")
    assert config.surrogates_dir.endswith(os.sep)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        SweepConfig.from_dict(_values(methods={"simulated-annealing": {}}))


def test_config_survives_a_round_trip_through_a_dictionary():
    config = SweepConfig.from_dict(_values())

    assert SweepConfig.from_dict(config.to_dict()) == config

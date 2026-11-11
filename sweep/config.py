"""Sweep configuration: which runs to do and where the benchmark data lives."""

import json
import os
from dataclasses import asdict, dataclass

from sweep.registry import BUILDERS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Job:
    """One run: a method on one dataset of a search space, from one initialization."""

    method: str
    search_space_id: str
    dataset_id: str
    seed: str

    @property
    def job_id(self):
        return "/".join([self.method, self.search_space_id, self.dataset_id, self.seed])


@dataclass
class SweepConfig:
    experiment_id: str
    search_space_id: str
    dataset_ids: list
    seeds: list
    methods: dict
    n_trials: int
    base_seed: int
    data_dir: str
    surrogates_dir: str

    def __post_init__(self):
        if not self.dataset_ids or not self.seeds or not self.methods:
            raise ValueError("dataset_ids, seeds and methods must not be empty")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        unknown = sorted(set(self.methods) - set(BUILDERS))
        if unknown:
            raise ValueError("unknown methods: {}".format(", ".join(unknown)))

    def jobs(self):
        """All runs of the sweep, grouped by method, then dataset, then seed."""
        return [
            Job(method, self.search_space_id, dataset_id, seed)
            for method in self.methods
            for dataset_id in self.dataset_ids
            for seed in self.seeds
        ]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


def _resolve(path):
    """Relative data paths are taken from the repository root."""
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    # HPOBHandler builds surrogate file names by plain concatenation.
    return os.path.join(path, "")


def load_config(path):
    with open(path) as f:
        values = json.load(f)
    values["data_dir"] = _resolve(values["data_dir"])
    values["surrogates_dir"] = _resolve(values["surrogates_dir"])
    return SweepConfig.from_dict(values)

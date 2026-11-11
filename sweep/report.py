"""Plain-text summary of a merged sweep."""

import os

import numpy as np

SUMMARY_FILE = "summary.txt"

_HEADER = "{:<10} {:<8} {:>5} {:>11} {:>11} {:>11} {:>11}".format(
    "method", "dataset", "runs", "mean_final", "std_final", "min_final", "max_final"
)
_ROW = "{:<10} {:<8} {:>5d} {:>11.6f} {:>11.6f} {:>11.6f} {:>11.6f}"


def summarise(config, merged):
    """One row per method and dataset with statistics of the final incumbent over seeds."""
    lines = [
        "experiment: {}".format(config.experiment_id),
        "search space: {}   trials per run: {}   base seed: {}".format(
            config.search_space_id, config.n_trials, config.base_seed
        ),
        "",
        _HEADER,
    ]
    for method in config.methods:
        datasets = merged.get(method, {}).get(config.search_space_id, {})
        for dataset_id in config.dataset_ids:
            finals = [curve[-1] for curve in datasets.get(dataset_id, {}).values()]
            if not finals:
                continue
            lines.append(
                _ROW.format(
                    method,
                    dataset_id,
                    len(finals),
                    float(np.mean(finals)),
                    float(np.std(finals)),
                    float(np.min(finals)),
                    float(np.max(finals)),
                )
            )
    return "\n".join(lines) + "\n"


def write_summary(out_dir, text):
    path = os.path.join(out_dir, SUMMARY_FILE)
    with open(path, "w") as f:
        f.write(text)
    return path

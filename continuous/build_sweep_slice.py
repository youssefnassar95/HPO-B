"""
Builds the small benchmark slice that the sweep runner uses (see sweep/).

It copies one search space out of the HPO-B meta-test split and trains one
XGBoost surrogate per dataset of that space, with the same model family as
train-surrogates.py but a fixed number of boosting rounds. The result is small
enough to live in the repository, so a sweep does not need the full benchmark
download.

Usage (from the repository root, with the full benchmark in hpob-data/):

    python continuous/build_sweep_slice.py --space 5971 --src hpob-data/ --dst sweep_data/
"""

import argparse
import json
import os

import numpy as np
import xgboost as xgb

PARAMS = {
    "max_depth": 6,
    "eta": 0.1,
    "objective": "reg:squarederror",
    "nthread": 1,
    "seed": 0,
}
NUM_ROUNDS = 100


def build(space, src, dst):
    with open(os.path.join(src, "meta-test-dataset.json")) as f:
        meta_test = json.load(f)
    with open(os.path.join(src, "bo-initializations.json")) as f:
        initializations = json.load(f)

    data_dir = os.path.join(dst, "hpob-data")
    surrogates_dir = os.path.join(dst, "surrogates")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(surrogates_dir, exist_ok=True)

    with open(os.path.join(data_dir, "meta-test-dataset.json"), "w") as f:
        json.dump({space: meta_test[space]}, f, separators=(",", ":"))
    with open(os.path.join(data_dir, "bo-initializations.json"), "w") as f:
        json.dump({space: initializations[space]}, f, separators=(",", ":"))

    stats = {}
    for dataset_id, task in meta_test[space].items():
        X = np.array(task["X"])
        y = np.array(task["y"]).reshape(-1)
        dtrain = xgb.DMatrix(X, label=y)
        booster = xgb.train(PARAMS, dtrain, NUM_ROUNDS)
        predictions = booster.predict(dtrain)

        name = "surrogate-" + space + "-" + dataset_id
        booster.save_model(os.path.join(surrogates_dir, name + ".json"))
        stats[name] = {
            "y_min": float(predictions.min()),
            "y_max": float(predictions.max()),
        }
        mse = float(np.mean((predictions - y) ** 2))
        print("{}: {} rows, {} dims, train MSE {:.6f}".format(name, X.shape[0], X.shape[1], mse))

    with open(os.path.join(surrogates_dir, "summary-stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="5971", help="Search space id to extract")
    parser.add_argument("--src", default="hpob-data/", help="Directory with the full benchmark")
    parser.add_argument("--dst", default="sweep_data/", help="Where to write the slice")
    args = parser.parse_args()
    build(args.space, args.src, args.dst)

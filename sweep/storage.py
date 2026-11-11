"""
On-disk layout of a sweep.

    <out_dir>/parts/worker-<k>.jsonl   one line per finished run, written by worker k
    <out_dir>/merged.json              {method: {search_space: {dataset: {seed: incumbents}}}}

Every worker appends to its own part file, so workers never write to the same
file. The merged file has, per method, the same structure as the files in
results/.
"""

import glob
import json
import os

PARTS_DIR = "parts"
MERGED_FILE = "merged.json"


def part_path(out_dir, worker_index):
    return os.path.join(out_dir, PARTS_DIR, "worker-{}.jsonl".format(worker_index))


def list_parts(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, PARTS_DIR, "worker-*.jsonl")))


def append_record(out_dir, worker_index, record):
    """Add one finished run to the part file of a worker."""
    path = part_path(out_dir, worker_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_part(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def completed_job_ids(out_dir):
    """Ids of the runs that already have a record in the output directory."""
    done = set()
    for path in list_parts(out_dir):
        for record in read_part(path):
            done.add(record["job_id"])
    return done


def nest(records):
    """Arrange run records as {method: {search_space: {dataset: {seed: incumbents}}}}."""
    nested = {}
    for record in records:
        datasets = nested.setdefault(record["method"], {}).setdefault(record["search_space_id"], {})
        datasets.setdefault(record["dataset_id"], {})[record["seed"]] = record["incumbent"]
    return nested


def merge_parts(out_dir):
    """Combine the part files of all workers into one result dictionary."""
    merged = {}
    for path in list_parts(out_dir):
        part = nest(read_part(path))
        for method, spaces in part.items():
            for space, datasets in spaces.items():
                merged.setdefault(method, {}).setdefault(space, {}).update(datasets)
    return merged


def count_runs(merged):
    return sum(
        len(seeds)
        for spaces in merged.values()
        for datasets in spaces.values()
        for seeds in datasets.values()
    )


def write_merged(out_dir, merged):
    path = os.path.join(out_dir, MERGED_FILE)
    with open(path, "w") as f:
        json.dump(merged, f, indent=1, sort_keys=True)
    return path

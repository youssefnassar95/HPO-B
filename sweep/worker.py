"""Worker side of a sweep: runs one job at a time inside a pool process."""

import logging
import os
import time

import torch

from hpob_handler import HPOBHandler
from sweep.config import SweepConfig
from sweep.recording import RecordingMethod
from sweep.registry import get_method
from sweep.seeding import set_seed
from sweep.storage import append_record

_state = {}


def _open_log(out_dir, index):
    logger = logging.getLogger("sweep.worker.{}".format(index))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(os.path.join(out_dir, "worker-{}.txt".format(index)))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def init_worker(config_values, out_dir, index_counter):
    """Runs once in every pool process before it takes jobs."""
    with index_counter.get_lock():
        index = index_counter.value
        index_counter.value += 1

    config = SweepConfig.from_dict(config_values)

    # The pool already uses every core, so keep each process on one thread.
    torch.set_num_threads(1)
    # Fixed seed, so that a sweep gives the same numbers every time it is run.
    set_seed(config.base_seed)

    handler = HPOBHandler(
        root_dir=config.data_dir,
        mode="v3-test",
        surrogates_dir=config.surrogates_dir,
    )

    log = _open_log(out_dir, index)
    log.info("worker %d ready (pid %d, seed %d)", index, os.getpid(), config.base_seed)

    _state.update(index=index, config=config, out_dir=out_dir, handler=handler, log=log)


def run_job(job):
    """Evaluate one job, store its record and return a short status for the coordinator."""
    config = _state["config"]
    handler = _state["handler"]
    log = _state["log"]
    index = _state["index"]

    log.info("start %s", job.job_id)
    started = time.time()

    dim = handler.get_search_space_dim(job.search_space_id)
    method = RecordingMethod(get_method(job.method, dim, config.methods[job.method]))

    incumbent = handler.evaluate_continuous(
        method,
        search_space_id=job.search_space_id,
        dataset_id=job.dataset_id,
        seed=job.seed,
        n_trials=config.n_trials,
    )
    incumbent = [float(value) for value in incumbent]
    duration = time.time() - started

    append_record(
        _state["out_dir"],
        index,
        {
            "job_id": job.job_id,
            "method": job.method,
            "search_space_id": job.search_space_id,
            "dataset_id": job.dataset_id,
            "seed": job.seed,
            "worker": index,
            "pid": os.getpid(),
            "duration_sec": round(duration, 3),
            "incumbent": incumbent,
            "suggestions": method.suggestions,
        },
    )

    log.info(
        "done %s final=%.6f suggestions=%d (%.1fs)",
        job.job_id,
        incumbent[-1],
        len(method.suggestions),
        duration,
    )
    return {
        "job_id": job.job_id,
        "worker": index,
        "final": incumbent[-1],
        "duration_sec": duration,
    }

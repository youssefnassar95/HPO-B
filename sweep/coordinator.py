"""Coordinator side of a sweep: plans the jobs, feeds the pool and merges the results."""

import logging
import multiprocessing
import os
import sys

from sweep.report import summarise, write_summary
from sweep.storage import completed_job_ids, count_runs, merge_parts, write_merged
from sweep.worker import init_worker, run_job


def _open_log(out_dir):
    logger = logging.getLogger("sweep.coordinator")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(os.path.join(out_dir, "coordinator.txt")),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_sweep(config, out_dir, n_workers=1, resume=False):
    """
    Run every job of `config` that has no record in `out_dir` yet, then merge.

    Returns the merged results, {method: {search_space: {dataset: {seed: incumbents}}}}.
    """
    os.makedirs(out_dir, exist_ok=True)

    done = completed_job_ids(out_dir)
    if done and not resume:
        raise RuntimeError(
            "{} already holds {} finished runs; pass resume=True (--resume) to continue it".format(
                out_dir, len(done)
            )
        )

    log = _open_log(out_dir)
    jobs = config.jobs()
    pending = [job for job in jobs if job.job_id not in done]
    log.info(
        "experiment=%s jobs=%d already_done=%d pending=%d workers=%d base_seed=%d",
        config.experiment_id,
        len(jobs),
        len(jobs) - len(pending),
        len(pending),
        n_workers,
        config.base_seed,
    )

    if pending:
        # Fresh interpreters for the workers: forking a process that has already
        # used XGBoost or torch threads can leave the children deadlocked.
        context = multiprocessing.get_context("spawn")
        index_counter = context.Value("i", 0)
        with context.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(config.to_dict(), out_dir, index_counter),
        ) as pool:
            results = pool.imap_unordered(run_job, pending, chunksize=1)
            for finished, result in enumerate(results, start=1):
                log.info(
                    "[%d/%d] %s worker=%d final=%.6f (%.1fs)",
                    finished,
                    len(pending),
                    result["job_id"],
                    result["worker"],
                    result["final"],
                    result["duration_sec"],
                )
        log.info("all %d pending jobs finished", len(pending))

    merged = merge_parts(out_dir)
    merged_path = write_merged(out_dir, merged)
    log.info("merged %d runs into %s", count_runs(merged), merged_path)

    summary_path = write_summary(out_dir, summarise(config, merged))
    log.info("summary written to %s", summary_path)
    return merged

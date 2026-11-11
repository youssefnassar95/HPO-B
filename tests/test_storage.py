import json

from sweep.storage import (
    append_record,
    completed_job_ids,
    count_runs,
    list_parts,
    merge_parts,
    nest,
    read_part,
    write_merged,
)


def _record(dataset_id, seed, incumbent, method="random"):
    return {
        "job_id": "{}/5971/{}/{}".format(method, dataset_id, seed),
        "method": method,
        "search_space_id": "5971",
        "dataset_id": dataset_id,
        "seed": seed,
        "worker": 0,
        "pid": 1,
        "duration_sec": 0.1,
        "incumbent": incumbent,
        "suggestions": [[0.5, 0.5]],
    }


def test_records_are_read_back_in_the_order_they_were_appended(tmp_path):
    out_dir = str(tmp_path)
    first = _record("10093", "test0", [0.1, 0.2])
    second = _record("10093", "test1", [0.3, 0.4])

    append_record(out_dir, 0, first)
    append_record(out_dir, 0, second)

    (path,) = list_parts(out_dir)
    assert read_part(path) == [first, second]


def test_completed_job_ids_cover_every_part_file(tmp_path):
    out_dir = str(tmp_path)
    append_record(out_dir, 0, _record("10093", "test0", [0.1]))
    append_record(out_dir, 1, _record("3954", "test0", [0.2]))

    assert completed_job_ids(out_dir) == {"random/5971/10093/test0", "random/5971/3954/test0"}


def test_an_empty_output_directory_has_no_completed_jobs(tmp_path):
    assert completed_job_ids(str(tmp_path)) == set()


def test_nest_groups_runs_by_method_space_dataset_and_seed():
    nested = nest(
        [
            _record("10093", "test0", [0.1]),
            _record("10093", "test1", [0.2]),
            _record("3954", "test0", [0.3], method="gen-vae"),
        ]
    )

    assert nested == {
        "random": {"5971": {"10093": {"test0": [0.1], "test1": [0.2]}}},
        "gen-vae": {"5971": {"3954": {"test0": [0.3]}}},
    }


def test_merge_collects_the_runs_of_a_worker(tmp_path):
    out_dir = str(tmp_path)
    append_record(out_dir, 0, _record("10093", "test0", [0.1, 0.2]))
    append_record(out_dir, 0, _record("10093", "test1", [0.3, 0.4]))
    append_record(out_dir, 0, _record("3954", "test0", [0.5, 0.6]))

    merged = merge_parts(out_dir)

    assert count_runs(merged) == 3
    assert merged["random"]["5971"]["10093"]["test1"] == [0.3, 0.4]


def test_merged_file_is_valid_json(tmp_path):
    out_dir = str(tmp_path)
    merged = {"random": {"5971": {"10093": {"test0": [0.1, 0.2]}}}}

    path = write_merged(out_dir, merged)

    with open(path) as f:
        assert json.load(f) == merged

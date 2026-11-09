"""Request folders and names, version recording, files under the output root, clock times and time left."""
from __future__ import annotations

import base64
import time

import pytest

from llmt_runtime import requests as R


def test_new_request_and_names(rt):
    d = R.new_request(rt, "bench")
    assert d.parent == rt.out_root() and d.name.startswith("bench_") and d.is_dir()
    assert R.check_request_name(rt, None) == ("new", None)
    assert R.check_request_name(rt, "run_abc-1.2")[0] == "new"
    for bad in ("abc", "run_", "run_a b", "run_" + "x" * 81, 5):
        kind, err = R.check_request_name(rt, bad)
        assert kind == "error" and "request_name" in err["error"]
    d = R.new_request(rt, request_name="run_abc")
    kind, err = R.check_request_name(rt, "run_abc")                  # being submitted: no request.json yet
    assert kind == "error" and "call again" in err["error"]
    with pytest.raises(FileExistsError):                              # a second call racing the first
        R.new_request(rt, request_name="run_abc")
    (d / "request.json").write_text("{}")
    assert R.check_request_name(rt, "run_abc") == ("repeat", d)       # status, never a second submission


def test_request_meta_records_versions(rt):
    m = R.request_meta(rt, names=["a"])
    assert m == {"names": ["a"], "translator": "2026.11.09.7", "runtime": rt.runtime_version}


def test_ran_with(rt):
    assert R.ran_with(rt, {"translator": "2026.11.09.7"}, {}) == {"translator": "2026.11.09.7"}
    assert R.ran_with(rt, {"translator": "2026.11.09.2"}, {}) == {"translator": "2026.11.09.2",
                                                                 "translator_now": "2026.11.09.7"}
    assert R.ran_with(rt, {"translator": "x"}, {"a": {"translator": "2026.11.09.3"}})["translator"] == "2026.11.09.3"
    two = R.ran_with(rt, {}, {"a": {"translator": "1"}, "b": {"translator": "2"}, "c": "bad"})
    assert two == {"translator": ["1", "2"], "translator_now": "2026.11.09.7"}
    old = R.ran_with(rt, {}, {})
    assert old["translator"] == "2026.11.09.7" and "not recorded" in old["translator_note"]


def test_read_file_only_under_root(rt, tmp_path):
    d = R.new_request(rt)
    (d / "x.txt").write_bytes(b"hi")
    assert R.read_file(rt, str(d / "x.txt")) == {"name": "x.txt", "bytes": 2,
                                                  "base64": base64.b64encode(b"hi").decode()}
    (tmp_path / "secret").write_text("no")
    assert "only files under" in R.read_file(rt, str(tmp_path / "secret"))["error"]
    assert "only files under" in R.read_file(rt, str(d / ".." / ".." / "secret"))["error"]
    assert "no file" in R.read_file(rt, str(d / "missing"))["error"]
    assert "more than max_mb" in R.read_file(rt, str(d / "x.txt"), max_mb=1e-9)["error"]
    (d / "p.png").write_bytes(b"png")
    assert R.image_path(rt, str(d / "p.png")) == (d / "p.png").resolve()
    for bad in (d / "x.txt", tmp_path / "secret"):
        with pytest.raises(ValueError):
            R.image_path(rt, str(bad))


def test_clock_and_slurm_start():
    now = time.mktime(time.strptime("2026-11-09T12:00:00", "%Y-%m-%dT%H:%M:%S"))
    assert R.clock(now + 3600, now) == "13:00"
    assert R.clock(now + 86400, now) == "12:00 (tomorrow)"
    assert R.clock(now - 86400, now) == "12:00 (yesterday)"
    assert R.clock(now + 3 * 86400, now) == "2026-11-12 12:00"
    assert R.slurm_start("2026-11-09T13:00:00") == now + 3600
    assert R.slurm_start("N/A") is None and R.slurm_start(None) is None


def test_timing_slurm(tmp_path):
    soon = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 1800))
    prog = {"a": {"state": "running", "eta_min": 10, "eta_min_max": 20},
            "b": {"state": "queued", "eta_min": 5, "eta_min_max": 8, "expected_start": soon}}
    t, nxt = R.timing_slurm(tmp_path, prog, {"a": "1001", "b": "1002"})
    assert 14 <= t["eta_min"] <= 16                                    # b starts when a frees its CPUs
    assert 37 <= t["eta_min_max"] <= 39                                # or at SLURM's (latest) start
    assert t["includes_queue_wait"] and "ids 1001, 1002" in nxt and f"study_status(request='{tmp_path}')" in nxt
    t, nxt = R.timing_slurm(tmp_path, {"b": {"state": "queued", "eta_min": 5, "eta_min_max": 8}}, {"b": "1"},
                            tool="bench_status")
    assert t["eta_min"] is None and "SLURM gives no start time" in nxt and "bench_status(bench=" in nxt


def test_timing_local(tmp_path):
    prog = {"a": {"stage": "solve", "eta_min": 1, "eta_min_max": 2},
            "b": {"stage": "queued", "eta_min": 1, "eta_min_max": 2}}
    t, nxt = R.timing_local(tmp_path, prog, workers=1)
    assert t["eta_min"] == 2 and t["eta_min_max"] == 4 and "wait_s=" in nxt
    prog["a"]["eta_min"] = 30
    t, nxt = R.timing_local(tmp_path, prog, workers=2)
    assert t["eta_min"] == 31 and "do not poll" in nxt


def test_until_within_deadline():
    from llmt_runtime.slurm import deadline
    assert R.until(10) - time.time() > 9
    with deadline(5):
        assert R.until(100) - time.time() < 2.1

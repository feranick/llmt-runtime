"""run.after (a study waits until other runs have ended), cancel at='job_end', and the times' offsets (2026.11.18.1).

The case behind it (cfdt, 2026-11-18): run B, a geometry variant, should start on spark2 only when run A, another
request already running there, has finished all of its jobs. A multi-job run submits each next job while (or after)
the current one runs, so a wait on the current job alone would let B start after A's first job.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from conftest import make_rt

from llmt_runtime import Hooks
from llmt_runtime import jobs as J


def request(rt, names, studies=None) -> Path:
    d = rt.requests.new_request()
    meta = rt.requests.request_meta(names=list(names), slurm=list(names), studies=studies or {n: {} for n in names})
    (d / "request.json").write_text(json.dumps(meta))
    return d


def rec(state_dir: Path, jid: str) -> dict:
    return json.loads((state_dir / "jobs" / f"{jid}.json").read_text())


def finish(rt, req: Path, name: str, ok: bool = True, **keys) -> None:
    (req / name).mkdir(parents=True, exist_ok=True)
    J.write_atomic(J.result_path(rt, req, name), json.dumps({"name": name, "ok": ok, **keys}))


def test_validation(rt, monkeypatch):
    monkeypatch.setenv("TT_EXECUTOR", "slurm")
    a = request(rt, ["a"])
    assert J.after_problems(rt, None, "x", {"run": {"after": f"{a.name}/a"}}) == []
    assert J.after_problems(rt, None, "x", {"run": {"after": [f"{a.name}/a"], "after_mode": "ok"}}) == []
    assert "cannot wait for itself" in J.after_problems(rt, None, "x", {"run": {"after": "x"}})[0]
    assert "no study nope" in J.after_problems(rt, None, "x", {"run": {"after": f"{a.name}/nope"}})[0]
    assert "no study 'y' in this request" in J.after_problems(rt, None, "x", {"run": {"after": "y"}},
                                                               {"x": {}, "z": {}})[0]
    assert "after_mode" in J.after_problems(rt, None, "x", {"run": {"after": f"{a.name}/a", "after_mode": "all"}})[0]
    assert "needs run.after" in J.after_problems(rt, None, "x", {"run": {"after_mode": "ok"}})[0]
    assert "'<request>/<study>'" in J.after_problems(rt, None, "x", {"run": {"after": 3}})[0]
    # a cycle within a request
    studies = {"p": {"run": {"after": "q"}}, "q": {"run": {"after": "p"}}}
    assert "cycle (p -> q -> p)" in J.after_problems(rt, None, "p", studies["p"], studies)[0]
    # a cycle through another request: a's study waits for this request's x
    b = request(rt, ["x"])
    (a / "a").mkdir()
    (a / "a" / "job.json").write_text(json.dumps({"job_id": "1", "run_after": [{"folder": str(b / "x")}]}))
    assert "cycle" in J.after_problems(rt, b, "x", {"run": {"after": f"{a.name}/a"}})[0]


def test_waits_for_the_running_run_and_moves_with_its_next_job(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_MODE", "sleep:20")
    a = request(rt, ["a"])
    ja = J.submit(rt, a, "a", {})["job_id"]                      # A runs
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    b = request(rt, ["b"])
    jb = J.submit(rt, b, "b", {"run": {"after": f"{a.name}/a"}})
    assert jb["job_id"] and jb["run_after"][0]["folder"] == str(a / "a") and jb["after_mode"] == "any"
    assert f"--dependency=afterany:{ja}" in jb["command"] and "--hold" not in jb["command"]
    text = (b / "b" / "job.sh").read_text()
    assert f"-m llmt_runtime.jobs gate {b / 'b'} || _llmt_gate=$?; [ \"$_llmt_gate\" -eq {J.GATE_SKIP} ] && exit 0" in text
    assert json.loads((a / "a" / J.WAITERS_FILE).read_text()) == [str(b / "b")]
    # A's job submits A's next job (a multi-job run): B now waits for that one (option a: scontrol update)
    monkeypatch.setenv("SLURM_JOB_ID", ja)
    ja2 = J.submit(rt, a, "a", {}, after=[ja])["job_id"]
    monkeypatch.delenv("SLURM_JOB_ID")
    assert [f"JobId={jb['job_id']}", f"Dependency=afterany:{ja2}"] in rec(slurm, jb["job_id"])["updates"]
    assert J.read_job(b / "b")["after_jobs"] == [ja2]               # A's newest job (job.json is now ja2's)
    # cfdt's chain 'together' keeps job.json describing the running job k, the waiting k+1 under next_job
    jk = dict(J.read_job(a / "a"), job_id=ja, next_job={"job_id": ja2})
    J.write_atomic(a / "a" / "job.json", json.dumps(jk))
    # while it waits: state 'waiting', how far A is, and B's expected start from A's expected finish
    rt.hooks.running_progress = lambda folder, study: {"time": 1.23, "end_time": 3.0, "eta_min": 90}
    st = J.job_states(rt, b, ["b"])["b"]
    assert st["state"] == "waiting" and st["reason"].startswith(f"after {a.name}/a (")
    assert "t = 1.23 of 3 s" in st["reason"] and "expected finish" in st["reason"] and st["expected_start"][19] in "+-"
    assert "ahead" not in st
    assert J.list_jobs(rt, active=True)["requests"]                 # waiting counts as active
    # A's jobs as the gate and cancel see them: the running one and the waiting next one
    assert J.target_ids(a / "a") == [ja, ja2]


def test_job_sh_gate_end_to_end(rt, slurm, monkeypatch):
    """The fake SLURM ignores dependencies, so B's job starts while A still runs: the gate in job.sh must send it back
    to wait (as a real dependency that missed a move would), then let it run once A has its result."""
    monkeypatch.setenv("TT_MODE", "sleep:2")
    a = request(rt, ["a"])
    J.submit(rt, a, "a", {})
    monkeypatch.setenv("TT_MODE", "ok")
    b = request(rt, ["b"])
    J.submit(rt, b, "b", {"run": {"after": f"{a.name}/a"}})
    import time
    t_end = time.time() + 30
    while time.time() < t_end and not J.result_path(rt, b, "b").is_file():
        time.sleep(0.2)
    assert J.result_path(rt, a, "a").is_file() and J.result_path(rt, b, "b").is_file()
    assert os.path.getmtime(J.result_path(rt, b, "b")) >= os.path.getmtime(J.result_path(rt, a, "a"))
    assert J.read_job(b / "b").get("waited_jobs")                   # it waited again at least once
    assert "previous_jobs" not in J.read_job(b / "b")                # (cfdt's segments, not these)
    assert "has not finished: waiting again as job" in (b / "b" / "job.log").read_text()


def test_same_request_study_not_yet_submitted_holds(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    r = request(rt, ["x", "y"], {"x": {"run": {"after": "y"}}, "y": {}})
    jx = J.submit(rt, r, "x", {"run": {"after": "y"}})
    assert "--hold" in jx["command"] and jx["held"] is True and not jx["after_jobs"]
    jy = J.submit(rt, r, "y", {})                                   # y submitted: x waits for it, released
    ups = rec(slurm, jx["job_id"])["updates"]
    assert [f"JobId={jx['job_id']}", f"Dependency=afterany:{jy['job_id']}"] in ups
    assert J.read_job(r / "x")["held"] is False and J.read_job(r / "x")["after_jobs"] == [jy["job_id"]]
    calls = (slurm / "calls" / "scontrol").read_text()
    assert int(calls) >= 2                                          # update and release


def test_gate(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    a = request(rt, ["a"])
    ja = J.submit(rt, a, "a", {})["job_id"]
    b = request(rt, ["b", "c"])
    jb = J.submit(rt, b, "b", {"run": {"after": f"{a.name}/a"}})
    jc = J.submit(rt, b, "c", {"run": {"after": f"{a.name}/a", "after_mode": "ok"}})
    # A still queued when B's job starts (a missed move): B waits again behind A's job, and does not run
    assert J.gate(b / "b", retries=1, pause=0) == J.GATE_SKIP
    jb2 = J.read_job(b / "b")
    assert jb2["job_id"] != jb["job_id"] and jb2["waited_jobs"] == [jb["job_id"]]
    assert str(b / "b") in json.loads((a / "a" / J.WAITERS_FILE).read_text())   # registered again
    assert f"--dependency=afterany:{ja}" in jb2["command"]
    # A ends (scancel): with 'any', B runs; with 'ok', C is cancelled with the reason
    J.scancel([ja])
    assert J.gate(b / "b", retries=1, pause=0) == 0
    assert J.gate(b / "c", retries=1, pause=0) == J.GATE_SKIP
    res = json.loads(J.result_path(rt, b, "c").read_text())
    assert res["state"] == "cancelled" and "after_mode is 'ok'" in res["why"] and f"{a.name}/a" in res["why"]
    st = J.job_states(rt, b, ["c"])["c"]
    assert st["state"] == "cancelled" and "after_mode is 'ok'" in st["why"]
    # A done ok: both may run
    finish(rt, a, "a", ok=True)
    assert J.gate(b / "b", retries=1, pause=0) == 0
    assert J.read_job(b / "c")["job_id"] == jc["job_id"]


def test_ok_mode_refused_at_submission_when_the_run_already_failed(rt, slurm):
    a = request(rt, ["a"])
    finish(rt, a, "a", ok=False, state="failed", why="diverged")
    b = request(rt, ["b"])
    info = J.submit(rt, b, "b", {"run": {"after": f"{a.name}/a", "after_mode": "ok"}})
    assert info["ok"] is False and info["state"] == "cancelled" and "diverged" in info["why"]
    J.failed_submission(rt, b, "b", info)
    assert J.job_states(rt, b, ["b"])["b"]["state"] == "cancelled"
    # with 'any' it is submitted, waiting for nothing
    c = request(rt, ["c"])
    info = J.submit(rt, c, "c", {"run": {"after": f"{a.name}/a"}})
    assert info["job_id"] and not info["after_jobs"] and "--dependency" not in " ".join(info["command"])


def test_submission_refused_when_a_ref_is_missing(rt, slurm):
    b = request(rt, ["b"])
    info = J.submit(rt, b, "b", {"run": {"after": "run_nope/x"}})
    assert info["ok"] is False and "no study x" in info["why"]


def test_stop_after_job(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    a = request(rt, ["a", "b"])
    ja = J.submit(rt, a, "a", {})["job_id"]
    # a pre-submitted next job (cfdt's chain 'together': job.json keeps job k, next_job is k+1)
    nxt = J.submit(rt, a, "b", {})                                  # any queued job stands in for k+1
    j = J.read_job(a / "a")
    J.write_atomic(a / "a" / "job.json", json.dumps(dict(j, next_job={"job_id": nxt["job_id"]})))
    out = J.cancel_request(rt, a, json.loads((a / "request.json").read_text()), ["a"], at="job_end")
    assert out["stop_after_job"] == ["a"] and out["dropped_next"] == [nxt["job_id"]]
    assert J.stop_requested(a / "a") and rec(slurm, nxt["job_id"])["state"] == "CANCELLED"
    assert rec(slurm, ja).get("state") == "PENDING"                 # the current job is left to finish
    # the next job is not chained from inside the run's own job
    monkeypatch.setenv("SLURM_JOB_ID", ja)
    info = J.submit(rt, a, "a", {}, after=[ja])
    assert info["ok"] is False and info["stopped"] and "job_end" in info["why"]
    monkeypatch.delenv("SLURM_JOB_ID")
    assert "error" in J.cancel_request(rt, a, {"names": ["a"]}, at="later")
    # the command line the router uses
    assert J.main(["stop-after-job", str(a), "b"]) == 0 and J.stop_requested(a / "b")


def test_cancel_takes_a_waiting_next_job_too(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    a = request(rt, ["a", "b"])
    ja = J.submit(rt, a, "a", {})["job_id"]
    nxt = J.submit(rt, a, "b", {})["job_id"]
    j = J.read_job(a / "a")
    J.write_atomic(a / "a" / "job.json", json.dumps(dict(j, next_job={"job_id": nxt})))
    out = J.cancel_request(rt, a, json.loads((a / "request.json").read_text()), ["a"])
    assert sorted(out["cancelled_jobs"]) == sorted([ja, nxt])


# ----------------------------------------------------------------------------- 2026.11.18.2


def test_gate_line_is_safe_under_set_e(tmp_path):
    """job.sh with `set -e` (P_JOB_SETUP or a translator's lines): exit 75 ends the job with 0 (COMPLETED, not
    FAILED), and a gate that fails otherwise still lets the study run."""
    import subprocess
    import sys
    for code, want in ((75, ("0", "")), (0, ("0", "ran")), (1, ("0", "ran"))):
        fake = tmp_path / f"py{code}"
        fake.write_text(f"#!/bin/sh\nexit {code}\n")
        fake.chmod(0o755)
        line = J.gate_line(tmp_path).replace(__import__("shlex").quote(sys.executable), str(fake))
        p = subprocess.run(["bash", "-c", f"set -e\n{line}\necho ran"], capture_output=True, text=True)
        assert (str(p.returncode), p.stdout.strip()) == want, (code, p.returncode, p.stdout, p.stderr)


def test_job_end_moves_waiters_back_before_cancelling_the_next_job(rt, slurm, monkeypatch):
    """afterany counts a cancel: a study waiting for the dropped next job must wait for the current job before the
    next one is cancelled, or it could start while the current job still runs."""
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    a = request(rt, ["a", "n"])
    ja = J.submit(rt, a, "a", {})["job_id"]
    nxt = J.submit(rt, a, "n", {})["job_id"]
    J.write_atomic(a / "a" / "job.json", json.dumps(dict(J.read_job(a / "a"), next_job={"job_id": nxt})))
    b = request(rt, ["b"])
    jb = J.submit(rt, b, "b", {"run": {"after": f"{a.name}/a"}})
    assert sorted(jb["after_jobs"]) == sorted([ja, nxt])
    order = []
    real_scancel, real_notify = J.scancel, J.notify_waiters
    monkeypatch.setattr(J, "scancel", lambda ids: order.append(("scancel", list(ids))) or real_scancel(ids))
    monkeypatch.setattr(J, "notify_waiters", lambda *x, **k: order.append(("notify",)) or real_notify(*x, **k))
    out = J.cancel_request(rt, a, json.loads((a / "request.json").read_text()), ["a"], at="job_end")
    assert out["dropped_next"] == [nxt] and order[0] == ("notify",) and order[1] == ("scancel", [nxt])
    assert J.read_job(a / "a")["next_job"]["dropped"]
    assert J.read_job(b / "b")["after_jobs"] == [ja]
    assert [f"JobId={jb['job_id']}", f"Dependency=afterany:{ja}"] in rec(slurm, jb["job_id"])["updates"]


def test_a_refused_dependency_move_is_not_recorded(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    a = request(rt, ["a"])
    ja = J.submit(rt, a, "a", {})["job_id"]
    b = request(rt, ["b"])
    jb = J.submit(rt, b, "b", {"run": {"after": f"{a.name}/a"}})
    monkeypatch.setenv("FAKE_SCONTROL_DENY", "1")
    ja2 = J.submit(rt, a, "a", {}, after=[ja])["job_id"]
    jb2 = J.read_job(b / "b")
    assert jb2["after_jobs"] == [ja] and "denied" in jb2["after_move_error"]
    monkeypatch.delenv("FAKE_SCONTROL_DENY")
    J.notify_waiters(rt, a, "a", ja2)                               # the next move succeeds and clears it
    jb3 = J.read_job(b / "b")
    assert jb3["after_jobs"] == [ja2] and "after_move_error" not in jb3
    assert jb["job_id"] == jb3["job_id"]


def test_selftest_slurm_on_the_fake_queue(tmp_path, slurm, monkeypatch, capsys):
    """`python -m llmt_runtime.selftest slurm` with the fake SLURM honouring dependencies and holds: every step
    passes, and the folder is removed."""
    from llmt_runtime import selftest
    monkeypatch.setenv("FAKE_DEPENDENCIES", "1")
    monkeypatch.setenv("FAKE_MIN_JOB_AGE", "30")
    monkeypatch.setenv("LLMT_SELFTEST_OUT", str(tmp_path / "st"))
    monkeypatch.setenv("LLMT_SELFTEST_JOB_SETUP", "set -e")
    monkeypatch.setenv("PYTHONPATH", str(Path(J.__file__).parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    rc = selftest.main(["slurm", "--seconds", "3", "--timeout-min", "1"])
    out = capsys.readouterr().out
    assert rc == 0, out
    for step in ("slurm", "validate", "submit", "wait", "chain", "move", "hold", "release", "job_end", "gate",
                 "finish"):
        assert any(ln.startswith("PASS") and ln.split()[1] == step for ln in out.splitlines()), (step, out)
    assert not list((tmp_path / "st").glob("run_*"))


def test_selftest_reports_missing_slurm(tmp_path, monkeypatch, capsys):
    from llmt_runtime import selftest
    monkeypatch.setenv("PATH", str(tmp_path))
    assert selftest.main(["slurm"]) == 1
    assert "FAIL  slurm" in capsys.readouterr().out
    assert selftest.main([]) == 2


def test_run_after_needs_a_slurm_job(rt, monkeypatch):
    """Only a SLURM job can wait: a study with run.after is always a SLURM job with P_EXECUTOR=slurm, whatever its
    estimate, and is refused where the executor is local or with queue: false (feat's report, 2026-11-18: short
    studies went to the local pool and started at once)."""
    from llmt_runtime import slurm as S
    a = request(rt, ["a"])
    st = {"run": {"after": f"{a.name}/a"}}
    monkeypatch.setenv("TT_EXECUTOR", "local")
    assert "run.after needs SLURM: TT_EXECUTOR is local" in J.after_problems(rt, None, "x", st)[0]
    assert S.use_slurm(rt, st, 0.01) is False
    monkeypatch.setenv("TT_EXECUTOR", "slurm")
    assert J.after_problems(rt, None, "x", st) == []
    assert S.use_slurm(rt, st, 0.01) is True                       # under P_SLURM_MIN_MINUTES, still a job
    assert S.use_slurm(rt, {}, 0.01) is False                      # (others as before)
    assert S.use_slurm(rt, {"run": {"after": f"{a.name}/a", "queue": True}}, 0.01) is True
    bad = J.after_problems(rt, None, "x", {"run": {"after": f"{a.name}/a", "queue": False}})
    assert len(bad) == 1 and "drop queue: false" in bad[0]
    assert "drop queue: false" in J.after_problems(rt, None, "x", {"run": {"after": f"{a.name}/a"}, "queue": False})[0]
    assert J.after_problems(rt, None, "x", {"run": {"queue": False}}) == []    # no run.after: no opinion

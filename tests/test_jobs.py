"""job.sh and job.json, submission, states (queued, running, done collected by a new process, every failure state from
sacct and from job.log), cancel, extend, the queue."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import make_rt

from llmt_runtime import Hooks
from llmt_runtime import jobs as J
from llmt_runtime import slurm as S


def request(rt, names=("a",), slurm=True) -> Path:
    d = rt.requests.new_request()
    (d / "request.json").write_text(json.dumps(rt.requests.request_meta(names=list(names),
                                                                         slurm=list(names) if slurm else [])))
    return d


def wait_for(cond, timeout=10.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def set_job(state_dir: Path, jid: str, **keys):
    f = state_dir / "jobs" / f"{jid}.json"
    rec = json.loads(f.read_text())
    rec.update(keys)
    f.write_text(json.dumps(rec))


# ---- job.sh

def test_job_script_threads(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_PROCS", "6")
    req = request(rt)
    info = J.submit(rt, req, "a", {"run": {"mem": "4G"}})
    text = (req / "a" / "job.sh").read_text().splitlines()
    assert text[0] == "#!/bin/bash"
    assert f"#SBATCH --job-name=tt:{req.name}:a" in text
    assert "#SBATCH --ntasks=1" in text and "#SBATCH --cpus-per-task=6" in text
    assert "#SBATCH --time=20" in text and "#SBATCH --mem=4G" in text and "#SBATCH --export=ALL" in text
    assert text[-1].startswith("exec ") and text[-1].endswith(f"-m ttpkg.cli run-study {req} a")
    assert not any("--nodelist" in ln for ln in text)
    assert info["procs"] == 6 and info["mem"] == "4G" and info["mem_from"] == "run.mem"
    for k in ("executor", "job_id", "submitted", "time_min", "slurm_time", "slurm_time_info", "time_from",
              "timeout_s", "nodelist", "run", "command", "translator", "runtime"):
        assert k in info
    assert json.loads((req / "a" / "job.json").read_text()) == info


def test_job_script_mpi(tmp_path, slurm, monkeypatch):
    rt = make_rt(tmp_path, kind="mpi", one_node=True)
    monkeypatch.setenv("TT_PROCS", "64")                              # capped at the node's 20 cores
    req = request(rt)
    J.submit(rt, req, "a", {})
    text = (req / "a" / "job.sh").read_text()
    assert "--ntasks=20\n" in text and "--nodes=1" in text and "--ntasks-per-core=1" in text
    monkeypatch.setenv("TT_SLURM_ARGS", "--hint=multithread")         # the user's placement wins
    J.submit(rt, req, "a", {})
    assert "--ntasks-per-core" not in (req / "a" / "job.sh").read_text()


def test_hooks_shape_the_job(tmp_path, slurm):
    hooks = Hooks(procs=lambda st: 3, mem_auto=lambda st: "5G", time_need_min=lambda st: 77,
                  resource_lines=lambda st, p: [f"--ntasks={p}", "--gpus=1"], job_extra=lambda st: {"cells": 9})
    rt = make_rt(tmp_path, hooks=hooks)
    req = request(rt)
    info = J.submit(rt, req, "a", {})
    text = (req / "a" / "job.sh").read_text()
    assert "--ntasks=3\n#SBATCH --gpus=1" in text and "--time=77" in text and "--mem=5G" in text
    assert info["cells"] == 9 and info["mem_from"] == "auto" and "export PYTHONPATH" in text


def test_nodelist_pinned_and_dependency(rt, slurm, tmp_path, monkeypatch):
    monkeypatch.setenv("TT_SLURM_NODE", "fake1")
    req = request(rt)
    info = J.submit(rt, req, "a", {}, after=["900", "901"], estimate_min=1.5)
    assert "#SBATCH --nodelist=fake1" in (req / "a" / "job.sh").read_text()
    assert "--dependency=afterany:900:901" in info["command"] and info["after"] == ["900", "901"]
    assert info["estimate_min"] == 1.5


# ---- submission failures

def test_sbatch_refuses(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_FAIL", "1")
    req = request(rt)
    info = J.submit(rt, req, "a", {})
    assert info["ok"] is False and "Unable to contact slurm controller" in info["why"]
    res = J.failed_submission(rt, req, "a", info)
    assert json.loads((req / "a" / "a.json").read_text()) == res and res["state"] == "failed"


def test_no_sbatch(rt, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    info = J.submit(rt, request(rt), "a", {})
    assert info == {"ok": False, "why": "could not submit: sbatch not found (TT_EXECUTOR=slurm needs SLURM)"}


def test_sbatch_hangs_then_breaker(rt, slurm, monkeypatch):
    monkeypatch.setattr(J, "SBATCH_TIMEOUT", 1.0)
    monkeypatch.setenv("FAKE_SBATCH_SLEEP", "5")
    req = request(rt, ("a", "b"))
    t0 = time.time()
    assert "did not answer within 1 s" in J.submit(rt, req, "a", {})["why"]
    assert "did not answer" in J.submit(rt, req, "b", {})["why"]          # at once: the breaker
    assert time.time() - t0 < 4


def test_submit_within_call_deadline(rt, slurm):
    with S.deadline(1.5):
        time.sleep(0.6)
        info = J.submit(rt, request(rt), "a", {})
    assert info["ok"] is False and "too slow" in info["why"]


# ---- states

def test_running_then_done_collected_by_a_new_process(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_MODE", "sleep:1.5")
    req = request(rt)
    info = J.submit(rt, req, "a", {"run": {"mem": "3G"}})
    st = J.job_states(rt, req, ["a"])["a"]
    assert st["state"] == "running" and st["job_id"] == info["job_id"] and st["node"] == "fake1"
    assert wait_for(lambda: (req / "a" / "a.json").is_file())
    code = (f"import json,sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
            f"from conftest import make_rt; from pathlib import Path; "
            f"rt = make_rt(Path({str(rt.out_root().parent)!r})); "
            f"print(json.dumps(rt.jobs.job_states(Path({str(req)!r}), ['a'])))")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env)
    assert json.loads(out.stdout)["a"]["state"] == "done", out.stderr
    res = json.loads((req / "a" / "a.json").read_text())
    assert res["env"]["SLURM_MEM_PER_NODE"] == "3072" and res["env"]["SLURM_CPUS_PER_TASK"] == "1"


def test_queued_with_reason_start_and_ahead(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    monkeypatch.setenv("FAKE_START", "2026-11-10T03:00:00")
    req = request(rt, ("a", "b", "c"))
    monkeypatch.setenv("FAKE_PRIORITY", "100")
    J.submit(rt, req, "a", {})
    monkeypatch.setenv("FAKE_PRIORITY", "500")                        # submitted later, higher priority
    J.submit(rt, req, "b", {})
    monkeypatch.setenv("FAKE_PRIORITY", "100")
    J.submit(rt, req, "c", {})
    st = J.job_states(rt, req, ["a", "b", "c"])
    assert st["a"]["state"] == "queued" and st["a"]["reason"] == "Priority"
    assert st["a"]["expected_start"] == "2026-11-10T03:00:00"
    assert (st["b"]["ahead"], st["a"]["ahead"], st["c"]["ahead"]) == (0, 1, 2)


def test_squeue_down_is_unknown_not_gone(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_MODE", "sleep:5")
    req = request(rt)
    J.submit(rt, req, "a", {})
    monkeypatch.setenv("FAKE_SQUEUE_FAIL", "1")
    assert J.job_states(rt, req, ["a"])["a"]["state"] == "unknown"


def test_submitting_then_never_submitted(rt, slurm, monkeypatch):
    req = request(rt, ("a",))
    assert J.job_states(rt, req, ["a"])["a"] == {"state": "queued", "job_id": None, "reason": "submitting"}
    assert J.job_states(rt, req, ["a"], submitting_s=0)["a"]["state"] == "failed"


@pytest.mark.parametrize("acct,state,words", [
    ("TIMEOUT", "timed_out", "raise TT_TIMEOUT"), ("DEADLINE", "timed_out", "time limit"),
    ("OUT_OF_MEMORY", "out_of_memory", "SLURM killed the job for memory (8G with 1 thread)"),
    ("CANCELLED by 1000", "cancelled", "cancelled by 1000"), ("FAILED", "failed", "SLURM: FAILED"),
    ("NODE_FAIL", "failed", "SLURM: NODE_FAIL"), ("COMPLETED", "failed", "ended without a result")])
def test_failure_states_from_sacct(rt, slurm, monkeypatch, acct, state, words):
    monkeypatch.setenv("TT_MODE", "crash")
    req = request(rt)
    jid = J.submit(rt, req, "a", {})["job_id"]
    assert wait_for(lambda: J.job_states(rt, req, ["a"])["a"]["state"] != "running")
    set_job(slurm, jid, sacct=acct)
    st = J.job_states(rt, req, ["a"])["a"]
    assert st["state"] == state and words in st["why"], st


@pytest.mark.parametrize("line,state", [
    ("slurmstepd: error: *** JOB 1 ON n1 CANCELLED AT 2026-11-09T10:00:00 DUE TO TIME LIMIT ***", "timed_out"),
    ("slurmstepd: error: *** JOB 1 ON n1 CANCELLED AT 2026-11-09T10:00:00 ***", "cancelled"),
    ("slurmstepd: error: Detected 2 oom_kill events in StepId=5807.batch. Some of the step tasks have been OOM Killed.",
     "out_of_memory"),
    ("slurmstepd: error: Detected 1 oom-kill event(s) in step 12.batch cgroup.", "out_of_memory"),
    ("Traceback (most recent call last):\nValueError: boom", "failed")])
def test_failure_states_from_job_log_without_accounting(rt, slurm, monkeypatch, line, state):
    monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    monkeypatch.setenv("TT_MODE", "crash")
    req = request(rt)
    jid = J.submit(rt, req, "a", {})["job_id"]
    assert wait_for(lambda: J.job_states(rt, req, ["a"])["a"]["state"] != "running")
    (req / "a" / "job.log").write_text(line + "\n")
    st = J.job_states(rt, req, ["a"])["a"]
    assert st["state"] == state and st["job_id"] == jid, st
    if state == "failed":
        assert "ValueError: boom" in st["why"]


def test_oom_wording_wins_over_failed_accounting(rt, slurm, monkeypatch):
    """SLURM may account a job killed in its batch step as FAILED: its log still says oom_kill (cfdt and omct read
    the log only without accounting, and matched only 'oom-kill')."""
    monkeypatch.setenv("TT_MODE", "oom")
    req = request(rt)
    jid = J.submit(rt, req, "a", {})["job_id"]
    assert wait_for(lambda: J.job_states(rt, req, ["a"])["a"]["state"] != "running")
    set_job(slurm, jid, sacct="FAILED")
    assert J.job_states(rt, req, ["a"])["a"]["state"] == "out_of_memory"


def test_oom_advice_hook_and_estimate_hint(tmp_path, slurm, monkeypatch):
    rt = make_rt(tmp_path, hooks=Hooks(oom_advice=lambda folder: f"give it 64G ({folder.name})",
                                       timeout_hint="or coarsen the mesh"))
    req = request(rt)
    assert J.ended_state(rt, "OUT_OF_MEMORY", req / "a" / "job.log", job={}) == ("out_of_memory", "give it 64G (a)")
    st, why = J.ended_state(rt, "TIMEOUT", req / "a" / "job.log", job={"time_from": "estimate"})
    assert st == "timed_out" and "or coarsen the mesh" in why and "TT_SLURM_TIME=timeout" in why


def test_job_gone_to_another_node(rt, slurm, tmp_path, monkeypatch):
    f = tmp_path / "nodes.txt"
    f.write_text("NodeName=spark-5da5 CPUTot=20 RealMemory=1000 NodeHostName=spark-5da5\n"
                 "NodeName=spark-0808 CPUTot=20 RealMemory=1000 NodeHostName=spark-0808\n")
    monkeypatch.setenv("FAKE_SCONTROL_NODES", str(f))
    monkeypatch.setattr(S.socket, "gethostname", lambda: "spark-5da5")
    req = request(rt)
    st, why = J.ended_state(rt, None, req / "a" / "job.log", job={}, node="spark-0808")
    assert st == "failed" and "went to node spark-0808" in why and "TT_SLURM_NODE=local" in why
    st, why = J.ended_state(rt, None, req / "a" / "job.log", job={})
    assert st == "cancelled" and "without starting" in why          # no accounting, no job.log: it never started
    st, why = J.ended_state(rt, "NODE_FAIL", req / "a" / "job.log", job={})
    assert st == "failed" and "without job.log (SLURM: NODE_FAIL)" in why


def test_classify_end_on_text(rt):
    """cfd-translator's final_state(acct, log text): the same rules on the text of job.log's end."""
    assert J.classify_end(rt, None, "", has_log=False)[0] == "cancelled"
    assert J.classify_end(rt, None, "Traceback ...\nValueError: boom")[0] == "failed"
    assert J.classify_end(rt, None, "error: Detected 1 oom_kill event in StepId=3.batch")[0] == "out_of_memory"
    st, why = J.classify_end(rt, "OUT_OF_MEMORY", "")                  # no folder: the generic advice
    assert st == "out_of_memory" and "run.mem" in why
    assert J.classify_end(rt, None, "*** JOB 1 ON n CANCELLED AT 2026 DUE TO TIME LIMIT ***")[0] == "timed_out"
    assert J.classify_end(rt, "CANCELLED by 1000", "") == ("cancelled", "cancelled by 1000 (scancel or cancel())")


def test_cancel_request(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_MODE", "sleep:30")
    req = request(rt, ("a", "b"))
    meta = json.loads((req / "request.json").read_text())
    jid = J.submit(rt, req, "a", {})["job_id"]
    J.submit(rt, req, "b", {})
    (req / "b" / "b.json").write_text("{}")                            # finished
    assert "error" in J.cancel_request(rt, req, meta, ["zz"])
    out = J.cancel_request(rt, req, meta)
    assert out["cancelled_jobs"] == [jid] and out["already_finished"] == ["b"] and "error" not in out
    assert (req / "a" / "cancel").is_file()
    assert wait_for(lambda: J.job_states(rt, req, ["a"])["a"]["state"] == "cancelled")


def test_cancel_flag_turns_failed_into_cancelled(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    req = request(rt)
    jid = J.submit(rt, req, "a", {})["job_id"]
    set_job(slurm, jid, vanished=True)                                 # left the queue before running, no log
    (req / "a" / "cancel").write_text("1")
    assert J.job_states(rt, req, ["a"])["a"]["state"] == "cancelled"


# ---- extend

def extend_setup(rt, slurm, monkeypatch, studies=None):
    monkeypatch.setenv("TT_MODE", "sleep:30")
    req = request(rt, ("a",))
    J.submit(rt, req, "a", {})
    return req, json.loads((req / "request.json").read_text())


def test_extend_not_needed_when_the_job_has_time(rt, slurm, monkeypatch):
    req, meta = extend_setup(rt, slurm, monkeypatch)
    monkeypatch.setenv("FAKE_TIME_LEFT", "UNLIMITED")
    out = J.extend_request(rt, req, meta, {}, 2)
    assert out["studies"]["a"]["slurm_time_limit"] == "not needed: the job has no SLURM time limit"
    assert (req / "a" / "extend").read_text() == "7200.0"
    monkeypatch.setenv("FAKE_TIME_LEFT", "5:00:00")
    assert J.extend_request(rt, req, meta, {}, 1)["studies"]["a"]["slurm_time_limit"].startswith("not needed")


def test_extend_raises_or_is_refused(rt, slurm, monkeypatch):
    req, meta = extend_setup(rt, slurm, monkeypatch)
    rt.hooks.solver_left_s = lambda d, st: 1800.0
    monkeypatch.setenv("FAKE_TIME_LEFT", "0:40:00")                    # 40 min < 30 min + 1 h + 10 min
    assert J.extend_request(rt, req, meta, {}, 1)["studies"]["a"]["slurm_time_limit"] == "raised"
    jid = J.read_job(req / "a")["job_id"]
    rec = json.loads((slurm / "jobs" / f"{jid}.json").read_text())
    assert rec["updates"] == [[f"JobId={jid}", "TimeLimit=+60"]]
    monkeypatch.setenv("FAKE_SCONTROL_DENY", "1")
    r = J.extend_request(rt, req, meta, {}, 1)["studies"]["a"]["slurm_time_limit"]
    assert r.startswith("not raised (Access/permission denied") and "TT_SLURM_TIME=max" in r


def test_extend_caps_and_errors(rt, slurm, monkeypatch):
    req, meta = extend_setup(rt, slurm, monkeypatch)
    monkeypatch.setenv("FAKE_TIME_LEFT", "UNLIMITED")
    monkeypatch.setenv("TT_MAX_TIMEOUT", "4200")                       # 600 s of timeout + 1 h of room
    out = J.extend_request(rt, req, meta, {}, 5)["studies"]["a"]
    assert out["added_s"] == 3600 and "TT_MAX_TIMEOUT" in out["capped"]
    assert J.take_extension(req / "a") == 3600 and J.extended_s(req / "a") == 3600
    assert J.take_extension(req / "a") == 0.0                          # consumed
    assert "already" in J.extend_request(rt, req, meta, {}, 1)["studies"]["a"]["error"]
    assert "error" in J.extend_request(rt, req, meta, {}, -1)
    assert "error" in J.extend_request(rt, req, meta, {}, 1, ["zz"])
    (req / "a" / "a.json").write_text("{}")
    assert J.extend_request(rt, req, meta, {}, 1)["studies"]["a"] == {"state": "finished: nothing to extend"}


# ---- the queue, jobs()

def test_queue_summary_and_list_jobs(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_MODE", "sleep:30")
    monkeypatch.setenv("TT_PROCS", "4")
    req = request(rt, ("a", "b"))
    J.submit(rt, req, "a", {})
    rt.hooks.running_progress = lambda d, st: {"stage": "solve", "eta_min": 3.0}
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    J.submit(rt, req, "b", {})
    q = J.queue_summary(rt)
    assert q["queue"] == {"running": 1, "queued": 1, "cpus_in_use": 4}
    assert q["machine"] == {"cpus": 8, "cpus_allocated": 2, "cpus_idle": 6}
    out = J.list_jobs(rt, active=True)
    assert out["translator"] == rt.version and out["executor"] == "slurm"
    st = out["requests"][0]["studies"]
    assert st["a"]["state"] == "running" and st["a"]["stage"] == "solve" and st["b"]["state"] == "queued"
    J.scancel([J.read_job(req / n)["job_id"] for n in ("a", "b")])
    assert J.list_jobs(rt, active=True)["requests"] == []
    assert len(J.list_jobs(rt)["requests"]) == 1


def test_write_atomic(tmp_path):
    f = tmp_path / "x.json"
    J.write_atomic(f, "{}")
    assert f.read_text() == "{}" and not list(tmp_path.glob(".x.json*"))


def test_timeout_without_accounting_math(rt):
    assert math.isinf(S.slurm_minutes("UNLIMITED"))


# ---- fixes from the review (2026.11.09.1)

@pytest.mark.parametrize("mode,final,state", [("crash", None, "failed"), ("ok_no_result", None, "failed"),
                                              ("crash", "TIMEOUT", "timed_out"), ("crash", "OUT_OF_MEMORY",
                                                                                   "out_of_memory")])
def test_ended_jobs_still_in_squeue_are_not_running(rt, slurm, monkeypatch, mode, final, state):
    """squeue keeps an ended job for MinJobAge (300 s) in its final state: it must not read as running (feat read any
    state but PENDING as running)."""
    monkeypatch.setenv("FAKE_MIN_JOB_AGE", "300")
    monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    monkeypatch.setenv("TT_MODE", "crash" if mode == "crash" else "ok")
    req = request(rt)
    jid = J.submit(rt, req, "a", {})["job_id"]
    assert wait_for(lambda: (slurm / "jobs" / f"{jid}.rc").is_file())
    if mode == "ok_no_result":
        (req / "a" / "a.json").unlink()
    if final:
        set_job(slurm, jid, final=final)
    assert J.squeue([jid])[jid]["state"] in ("FAILED", "COMPLETED", final)
    st = J.job_states(rt, req, ["a"])["a"]
    assert st["state"] == state, st
    if state == "failed":
        assert "SLURM: " in st["why"]


def test_cancelled_reads_cancelled_at_once(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_MIN_JOB_AGE", "300")
    monkeypatch.setenv("TT_MODE", "sleep:30")
    req = request(rt)
    meta = json.loads((req / "request.json").read_text())
    J.submit(rt, req, "a", {})
    J.cancel_request(rt, req, meta)
    assert J.job_states(rt, req, ["a"])["a"]["state"] == "cancelled"


def test_slow_sbatch_job_adopted(rt, slurm, monkeypatch):
    monkeypatch.setattr(J, "SBATCH_TIMEOUT", 1.0)
    monkeypatch.setenv("FAKE_SBATCH_SLOW_AFTER", "4")
    monkeypatch.setenv("TT_MODE", "sleep:30")
    req = request(rt)
    info = J.submit(rt, req, "a", {})
    assert info.get("job_id") == "1001" and "adopted" in info
    assert J.read_job(req / "a")["job_id"] == "1001"


def test_queue_summary_counts_each_node_once(rt, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SINFO_PARTITIONS", "debug*|1:00:00;long|infinite")   # the node in two partitions
    assert J.queue_summary(rt)["machine"] == {"cpus": 8, "cpus_allocated": 2, "cpus_idle": 6}


def test_short_cpus_per_task_option_respected(tmp_path, slurm, monkeypatch):
    rt = make_rt(tmp_path, kind="mpi")
    monkeypatch.setenv("TT_SLURM_ARGS", "-c 2")
    assert "--ntasks-per-core=1" not in J.resource_lines(rt, {}, 4)


def test_lost_node_only_on_a_cluster(rt, slurm):
    req = request(rt)
    st, why = J.ended_state(rt, None, req / "a" / "job.log", job={}, node="localhost")   # one node: not 'lost'
    assert "went to node" not in why and st == "cancelled"


def test_job_timeout_s(rt, slurm, tmp_path):
    req = request(rt)
    assert J.job_timeout_s(req / "a", 900) == 900
    (req / "a").mkdir(exist_ok=True)
    (req / "a" / "job.json").write_text(json.dumps({"timeout_s": 300}))
    assert J.job_timeout_s(req / "a", 900) == 300

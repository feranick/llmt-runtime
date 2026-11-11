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


PERLMUTTER = "\n".join(
    f"NodeName=nid{i:03d} CoresPerSocket=64 CPUAlloc=0 CPUTot=256 NodeAddr=nid{i:03d} NodeHostName=nid{i:03d} "
    f"RealMemory=512000 Sockets=2 State=IDLE ThreadsPerCore=2" for i in range(1, 7))


def multinode(tmp_path, monkeypatch, nodes="4", **kw):
    f = tmp_path / "nodes.txt"
    f.write_text(PERLMUTTER)                                          # six 128-core nodes, 500 GB each
    monkeypatch.setenv("FAKE_SCONTROL_NODES", str(f))
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    monkeypatch.setenv("TT_SLURM_NODE", "any")
    if nodes:
        monkeypatch.setenv("TT_SLURM_NODES", nodes)
    else:
        monkeypatch.delenv("TT_SLURM_NODES", raising=False)
    S.forget()
    return make_rt(tmp_path, kind="mpi", **kw)


def sbatch_lines(req, name="a"):
    return [ln[len("#SBATCH "):] for ln in (req / name / "job.sh").read_text().splitlines() if ln.startswith("#SBATCH")]


def test_multinode_512_ranks_on_128_core_nodes(tmp_path, slurm, monkeypatch):
    """cfdt on Perlmutter (S50 sliding mesh at 512 ranks): 4 nodes x 128, --mem per node, caps cores x nodes."""
    rt = multinode(tmp_path, monkeypatch, hooks=Hooks(mem_auto=lambda st: "400G"))
    monkeypatch.setenv("TT_PROCS", "512")
    req = request(rt)
    info = J.submit(rt, req, "a", {})
    ln = sbatch_lines(req)
    assert {"--ntasks=512", "--nodes=4", "--ntasks-per-node=128", "--ntasks-per-core=1", "--mem=100G"} <= set(ln)
    assert "--nodes=1" not in ln
    assert info["procs"] == 512 and info["nodes"] == 4 and info["ntasks_per_node"] == 128
    assert info["mem"] == "100G" and info["mem_total"] == "400G" and "400G over 4 nodes" in info["mem_from"]
    c = S.caps(rt)
    assert c["procs"] == 512 and c["nodes"] == 4 and "128 x 4 nodes" in c["procs_from"] and c["mem_per"] == "node"
    assert S.status_key(rt)["max_nodes"] == 4
    assert S.run_errors(rt, {"run": {"procs": 513}}) and not S.run_errors(rt, {"run": {"procs": 512}})
    monkeypatch.setenv("TT_PROCS", "300")                             # balanced: 3 x 100
    J.submit(rt, req, "a", {})
    assert {"--ntasks=300", "--nodes=3", "--ntasks-per-node=100", "--mem=136534M"}   # 400G / 3, rounded up <= set(sbatch_lines(req))
    monkeypatch.setenv("TT_PROCS", "1000")                            # capped at 4 x 128
    J.submit(rt, req, "a", {})
    assert {"--ntasks=512", "--nodes=4"} <= set(sbatch_lines(req))
    monkeypatch.setenv("TT_PROCS", "100")                             # fits one node: kept on one
    info = J.submit(rt, req, "a", {})
    assert {"--ntasks=100", "--nodes=1", "--mem=400G"} <= set(sbatch_lines(req)) and "nodes" not in info


def test_multinode_mem_capped_per_node_and_given_mem_per_node(tmp_path, slurm, monkeypatch):
    rt = multinode(tmp_path, monkeypatch, hooks=Hooks(mem_auto=lambda st: "4000G"))
    monkeypatch.setenv("TT_PROCS", "256")
    req = request(rt)
    info = J.submit(rt, req, "a", {})
    assert info["mem"] == "500G" and info["mem_capped_from"] == "2000G"     # 4000G / 2, capped at a node's
    info = J.submit(rt, req, "a", {"run": {"mem": "64G"}})
    assert info["mem"] == "64G" and info["mem_from"] == "run.mem"     # run.mem: per node, as given


def test_one_node_and_default_stay_on_one_node(tmp_path, slurm, monkeypatch):
    rt = multinode(tmp_path, monkeypatch, nodes="8", one_node=True)   # the translator's one_node wins
    monkeypatch.setenv("TT_PROCS", "512")
    req = request(rt)
    J.submit(rt, req, "a", {})
    assert {"--ntasks=128", "--nodes=1"} <= set(sbatch_lines(req)) and "procs" not in S.caps(rt)
    assert S.max_nodes(rt) == 1 and S.status_key(rt)["max_nodes"] == 1
    rt = multinode(tmp_path, monkeypatch, nodes=None)                 # P_SLURM_NODES unset: one node, lines as before
    J.submit(rt, req, "a", {})
    ln = sbatch_lines(req)
    assert "--ntasks=128" in ln and not any(x.startswith("--nodes") for x in ln) and "procs" not in S.caps(rt)
    rt = multinode(tmp_path, monkeypatch, nodes="99")                 # at most the nodes there are
    assert S.max_nodes(rt) == 6 and S.max_job_procs(rt) == 768
    monkeypatch.setenv("TT_SLURM_NODE", "nid001")                     # pinned to one node: one node
    S.forget()
    assert S.max_nodes(rt) == 1
    rt = make_rt(tmp_path, kind="threads")                             # threads: one node whatever the setting
    assert S.max_nodes(rt) == 1


def test_nodes_from_constraint_and_partition(tmp_path, slurm, monkeypatch):
    """A cluster with two kinds of nodes (Perlmutter: 64-core GPU nodes, 128-core CPU nodes): P_SLURM_ARGS'
    --constraint / --partition choose which ones size the job, not the smallest of the cluster."""
    f = tmp_path / "nodes.txt"
    gpu = [f"NodeName=gpu{i} CoresPerSocket=64 CPUTot=128 Sockets=1 ThreadsPerCore=2 RealMemory=256000 "
           f"AvailableFeatures=gpu,a100 Partitions=gpu_ss11 State=IDLE" for i in range(1, 3)]
    cpu = [f"NodeName=nid{i} CoresPerSocket=64 CPUTot=256 Sockets=2 ThreadsPerCore=2 RealMemory=512000 "
           f"AvailableFeatures=cpu Partitions=regular_milan_ss11 State=IDLE" for i in range(1, 5)]
    f.write_text("\n".join(gpu + cpu))
    monkeypatch.setenv("FAKE_SCONTROL_NODES", str(f))
    monkeypatch.setenv("TT_SLURM_NODE", "any")
    monkeypatch.setenv("TT_SLURM_NODES", "8")
    rt = make_rt(tmp_path, kind="mpi")
    S.forget()
    assert S.node_cores(rt) == 64 and S.max_nodes(rt) == 6               # no filter: the smallest, every node
    monkeypatch.setenv("TT_SLURM_ARGS", "-C cpu -q regular")
    assert S.node_cores(rt) == 128 and S.max_nodes(rt) == 4 and S.max_job_procs(rt) == 512
    assert S.job_layout(rt, 512) == (4, 128)
    monkeypatch.setenv("TT_SLURM_ARGS", "--partition=gpu_ss11")
    assert S.node_cores(rt) == 64 and S.max_nodes(rt) == 2
    monkeypatch.setenv("TT_SLURM_ARGS", "-C 'cpu|gpu'")                  # an expression: SLURM decides, no filter
    assert S.node_cores(rt) == 64 and S.max_nodes(rt) == 6
    monkeypatch.setenv("TT_SLURM_ARGS", "-C knl")                         # matches none: every node
    assert S.max_nodes(rt) == 6


def test_job_setup_lines(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_JOB_SETUP", "module load openfoam/2406\\nsource $FOAM_BASH")
    req = request(rt)
    J.submit(rt, req, "a", {})
    text = (req / "a" / "job.sh").read_text().splitlines()
    i = text.index("module load openfoam/2406")
    assert text[i + 1] == "source $FOAM_BASH" and text[i + 2].startswith("export PYTHONPATH")
    assert text[-1].startswith("exec ") and text[i - 1] == "#SBATCH --export=ALL"


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


def test_one_year_start_is_no_estimate(rt, slurm, monkeypatch):
    """SLURM's placeholder start one year ahead (behind a job with no time limit: carbonio 2026-11-10, cfdt jobs
    behind an UNLIMITED feat job showed 2027-11-09) is no estimate: expected_start_note, no expected_start."""
    from llmt_runtime import requests as R
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    year = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 365 * 86400))
    soon = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 299 * 86400))
    req = request(rt, ("a", "b"))
    J.submit(rt, req, "a", {})
    J.submit(rt, req, "b", {})
    monkeypatch.setenv("FAKE_START", year)
    st = J.job_states(rt, req, ["a", "b"])["a"]
    assert st["state"] == "queued" and "expected_start" not in st and st["ahead"] == 0
    assert st["expected_start_note"] == S.NO_START_NOTE and "no time limit" in st["expected_start_note"]
    q = J.squeue([st["job_id"]])[st["job_id"]]
    assert q["start"] == "N/A" and q["start_note"] == S.NO_START_NOTE
    assert R.slurm_start(year) is None and S.start_epoch(year) is not None
    _, nxt = R.timing_slurm(req, {"a": {"state": "queued", "eta_min": 5, "eta_min_max": 8, **st}}, {"a": st["job_id"]})
    assert "no time limit ends" in nxt
    monkeypatch.setenv("FAKE_START", soon)                            # 299 days: still SLURM's own estimate
    st = J.job_states(rt, req, ["a"])["a"]
    assert st["expected_start"] == soon and "expected_start_note" not in st and R.slurm_start(soon) is not None


def test_ahead_counts_per_node(rt, slurm, monkeypatch):
    """Pinned jobs count only the jobs that can run on their node (spark1's job 506 counted spark2's 469-471)."""
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    req = request(rt, ("s2a", "s2b", "any", "s1", "both"))
    monkeypatch.setenv("TT_SLURM_NODE", "spark2")
    J.submit(rt, req, "s2a", {})
    J.submit(rt, req, "s2b", {})
    monkeypatch.delenv("TT_SLURM_NODE")
    J.submit(rt, req, "any", {})                                     # may go anywhere: shares every node
    monkeypatch.setenv("TT_SLURM_NODE", "spark1")
    J.submit(rt, req, "s1", {})
    monkeypatch.setenv("TT_SLURM_NODE", "spark[1-2]")
    J.submit(rt, req, "both", {})
    st = J.job_states(rt, req, ["s2a", "s2b", "any", "s1", "both"])
    assert [st[n]["ahead"] for n in ("s2a", "s2b", "any", "s1", "both")] == [0, 1, 2, 1, 4]
    ids = {n: int(st[n]["job_id"]) for n in st}
    assert J.pending_jobs() == [(1000.0, ids[n]) for n in ("s2a", "s2b", "any", "s1", "both")]   # old shape
    assert J.pending_detail()[3] == (1000.0, ids["s1"], frozenset({"spark1"}))
    assert J.ahead_of(st["s1"]["job_id"], J.pending_jobs()) == 3        # 2-tuples: everything shares, as before
    assert J.ahead_of(st["s1"]["job_id"], share=lambda a, b: True) == 3
    assert J.ahead_of(st["both"]["job_id"], share=lambda a, b: bool(a & b)) == 3   # a stricter rule: "any" not counted


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


def test_slow_sbatch_adopted_within_a_call(rt, slurm, monkeypatch):
    """Inside an MCP call (a deadline) sbatch leaves time to find the job it may have submitted (2026.11.09.4: it had
    all the time left, and the look-up never ran)."""
    monkeypatch.setenv("FAKE_SBATCH_SLOW_AFTER", "30")
    monkeypatch.setenv("TT_MODE", "sleep:30")
    req = request(rt)
    t0 = time.time()
    with S.deadline(12):
        info = J.submit(rt, req, "a", {})
    assert info.get("job_id") == "1001" and "adopted" in info, info
    assert time.time() - t0 < 12


def test_queue_summary_leaves_out_ended_jobs(rt, slurm, monkeypatch):
    """squeue lists an ended job for MinJobAge in its final state: neither running nor queued (2026.11.09.5: it was
    counted as queued)."""
    monkeypatch.setenv("FAKE_MIN_JOB_AGE", "300")
    monkeypatch.setenv("TT_MODE", "crash")
    req = request(rt)
    jid = J.submit(rt, req, "a", {})["job_id"]
    assert wait_for(lambda: (slurm / "jobs" / f"{jid}.rc").is_file())
    assert J.squeue([jid])[jid]["state"] == "FAILED"
    assert J.queue_summary(rt)["queue"] == {"running": 0, "queued": 0, "cpus_in_use": 0}


def test_slurm_commands_do_not_read_the_servers_stdin(tmp_path, monkeypatch):
    """An MCP server speaks on its stdio: a command it runs gets /dev/null as stdin (2026.11.09.5)."""
    b = tmp_path / "bin"
    b.mkdir()
    (b / "readstdin").write_text("#!/bin/sh\nread x && echo \"got:$x\" || echo eof\n")
    (b / "readstdin").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:/usr/bin:/bin")
    p = S.run(["readstdin"], timeout=5)
    assert p is not None and p.stdout.strip() == "eof"


def test_failed_with_an_empty_log(rt, tmp_path):
    log = tmp_path / "s" / "job.log"
    log.parent.mkdir()
    log.write_text("\n")
    st, why = J.ended_state(rt, "NODE_FAIL", log, job={})
    assert st == "failed" and why.endswith("(SLURM: NODE_FAIL); job.log is empty")


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

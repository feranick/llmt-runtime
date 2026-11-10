"""jobs.py — one SLURM job per study: job.sh, submission, states read from the files and the queue, cancel, extend.
Version: 2026.11.10.1

    job_script / submit         <request>/<name>/job.sh (a complete script: `sbatch job.sh` reruns the study) and
                                job.json (the id, what was asked and why, the study's run settings)
    job_states(request, names)  {name: {state, job_id, ...}}: one squeue call for all the ids; sacct, else job.log,
                                for jobs that left the queue without a result (ended_state)
    queue_summary, list_jobs    this translator's jobs in the queue and the machine's CPUs; the jobs() tool
    cancel_request, scancel     cancel flags and scancel (the translator stops its local runs)
    extend_request, take_extension   more time for running studies: an 'extend' file the running solve consumes,
                                and SLURM's limit only when the job's time left does not cover the new deadline

The state of a study never comes from this process, so it works after restarts and from other conversations. SLURM
that does not answer is asked again next time, never taken for 'job gone'.
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import slurm as S

FINAL = ("timed_out", "out_of_memory", "cancelled", "failed")
QUEUED = ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD")
ACTIVE = ("RUNNING", "COMPLETING", "STOPPED", "SIGNALING", "STAGE_OUT", "RESIZING")
# any other state squeue shows (COMPLETED, FAILED, TIMEOUT, OUT_OF_MEMORY, CANCELLED, NODE_FAIL, ...) is a job that
# has ended: squeue keeps it listed for MinJobAge (300 s by default) and it must not read as running meanwhile
# SLURM's out-of-memory wordings in job.log: 'Detected 2 oom_kill events in StepId=5807.batch', 'oom-kill event(s)',
# 'Out Of Memory' (cfdt and omct matched only 'oom-kill', so such jobs came back as 'failed')
OOM_RE = re.compile(r"oom[-_ ]kill|out of memory", re.I)
EXTEND_FILE = "extend"
SUBMITTING_S = 600.0      # a study with no job.json yet is 'submitting' this long after its request was written
SBATCH_TIMEOUT = 60.0
FIND_JOB_S = 5.0          # kept for finding a job by name after sbatch timed out
_SBATCH_DOWN: dict = {"until": 0.0, "why": ""}
_PENDING: dict = {"t": 0.0, "v": None}


# ----------------------------------------------------------------------------- files

def write_atomic(path: Path, text: str) -> None:
    """Writes a file whole or not at all (a temporary file, then rename), so a reader never sees half of it."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def result_path(rt, request: Path, name: str) -> Path:
    """A finished study's result file: hooks.result_file, else <request>/<name>/<name>.json."""
    if rt.hooks.result_file:
        return Path(rt.hooks.result_file(Path(request), name))
    return Path(request) / name / f"{name}.json"


def read_job(folder: Path) -> dict:
    try:
        return json.loads((Path(folder) / "job.json").read_text())
    except (OSError, ValueError):
        return {}


def tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------- the job

def job_name(rt, request: Path, name: str) -> str:
    """'<cli>:<request folder name>:<study name>': how llmt jobs and jobs() find the translators' jobs."""
    return f"{rt.cli}:{Path(request).name}:{name}"


def resource_lines(rt, study: dict, procs: int) -> list[str]:
    """The resource options (without '#SBATCH '): hooks.resource_lines, else by kind. threads: --ntasks=1
    --cpus-per-task=N. mpi: --ntasks=N (--nodes=1 with rt.one_node) and --ntasks-per-core=1 (OpenMPI 5 refuses more
    ranks than physical cores in the job's CPU set; mochi, cfdt), unless P_SLURM_ARGS already says how to place them."""
    if rt.hooks.resource_lines:
        return list(rt.hooks.resource_lines(study, procs))
    if rt.kind == "threads":
        return ["--ntasks=1", f"--cpus-per-task={procs}"]
    out = [f"--ntasks={procs}"] + (["--nodes=1"] if rt.one_node else [])
    if not S.has_opt(S.slurm_args(rt), "--hint", "--threads-per-core", "--ntasks-per-core", "--cpus-per-task",
                     shorts=("-c",)):
        out.append("--ntasks-per-core=1")
    return out


def run_study_command(rt, request: Path, name: str) -> str:
    return (f"exec {shlex.quote(sys.executable)} -m {rt.package}.cli run-study {shlex.quote(str(request))} "
            f"{shlex.quote(name)}")


def job_script(rt, request: Path, name: str, study: dict, procs: int, time_limit: str, mem: str | None,
               nodelist: str | None) -> str:
    """job.sh: the job's name, folder, log, resources, time, memory, node and environment, then the run-study line.
    --mem is in the script (cfdt and omct passed it to sbatch only, so `sbatch job.sh` lost it)."""
    request = Path(request)
    d = request / name
    pre = list(rt.hooks.script_lines(request, name, study)) if rt.hooks.script_lines else []
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name(rt, request, name)}",
        f"#SBATCH --chdir={d}",
        f"#SBATCH --output={d / 'job.log'}",
        *[f"#SBATCH {x}" for x in resource_lines(rt, study, procs)],
        f"#SBATCH --time={time_limit}",
        *([f"#SBATCH --mem={mem}"] if mem else []),
        *([f"#SBATCH --nodelist={nodelist}"] if nodelist else []),
        "#SBATCH --export=ALL",
        *pre,
        run_study_command(rt, request, name),
        ""])


def submit(rt, request: Path, name: str, study: dict, estimate_min: float | None = None,
           after: list[str] | None = None, extra: dict | None = None) -> dict:
    """Writes job.sh and submits it (sbatch --parsable [P_SLURM_ARGS] job.sh, within the call's deadline); returns
    job.json's content, or {'ok': False, 'why': 'could not submit: ...'} (never a silent fallback to local; the
    translator writes that as the study's result). after: job ids this one waits for (--dependency=afterany).
    A submission that timed out stops further ones for 30 s (one slow call, not one per study)."""
    request = Path(request)
    d = request / name
    d.mkdir(parents=True, exist_ok=True)
    if not shutil.which("sbatch"):
        return {"ok": False, "why": f"could not submit: sbatch not found ({rt.var('EXECUTOR')}=slurm needs SLURM)"}
    if time.time() < _SBATCH_DOWN["until"]:
        return {"ok": False, "why": f"could not submit: {_SBATCH_DOWN['why']}"}
    procs = S.job_procs(rt, study)
    t, tinfo = S.job_time(rt, study, estimate_min)
    mplan = S.mem_plan(rt, study)
    add, nodelist, _ = S.node_pin(rt)
    script = d / "job.sh"
    script.write_text(job_script(rt, request, name, study, procs, t, mplan["mem"], add))
    script.chmod(0o755)
    dep = [f"--dependency=afterany:{':'.join(str(a) for a in after)}"] if after else []
    cmd = ["sbatch", "--parsable", *dep, *S.sbatch_args(rt, study), str(script)]
    left = S.remaining()
    if left is not None and left < 2:
        return {"ok": False, "why": "could not submit: SLURM was too slow to answer within this call; run the study "
                                    "again"}
    # within a call, keep a few seconds for the look-up of a job a slow controller took after sbatch gave up (before
    # 2026.11.09.4 sbatch had all the time left, so the look-up never ran inside an MCP call)
    limit = SBATCH_TIMEOUT if left is None else min(SBATCH_TIMEOUT, left - FIND_JOB_S - 1 if left > FIND_JOB_S + 3
                                                    else left - 1)
    adopted = None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        why = f"sbatch did not answer within {limit:g} s (is slurmctld running?)"
        _SBATCH_DOWN.update(until=time.time() + 30, why=why)
        # a slow controller may have taken the job after sbatch gave up: find it by its (unique) name, or it would
        # run with no job.json, beyond cancel(), and overwrite the 'could not submit' result
        adopted = find_job(job_name(rt, request, name))
        if adopted is None:
            return {"ok": False, "why": f"could not submit: {why}"}
    except OSError as exc:
        return {"ok": False, "why": f"could not submit: {exc}"}
    if adopted is not None:
        job_id = adopted
    else:
        out = p.stdout.strip()
        if p.returncode != 0 or not out:
            return {"ok": False, "why": f"could not submit: {(p.stderr or p.stdout).strip()[:500]}"}
        job_id = out.splitlines()[-1].split(";")[0].split()[0]
    info = {"executor": "slurm", "job_id": job_id, "submitted": round(time.time(), 1), "procs": procs,
            "time_min": S.time_need_min(rt, study), "slurm_time": t, "slurm_time_info": tinfo,
            "time_from": tinfo.get("time_from"), "timeout_s": tinfo.get("timeout_s", S.timeout_s(rt, study)),
            "mem": mplan["mem"], "mem_from": mplan["from"], "nodelist": nodelist, "run": S.run_of(study),
            "command": cmd, "translator": rt.version, "runtime": rt.runtime_version}
    if mplan.get("capped_from"):
        info["mem_capped_from"] = mplan["capped_from"]
    if estimate_min is not None:
        info["estimate_min"] = round(float(estimate_min), 2)
    if after:
        info["after"] = list(after)
    if adopted is not None:
        info["adopted"] = "sbatch timed out, but SLURM had taken the job: found by its name"
    try:
        if rt.hooks.job_extra:
            info.update(rt.hooks.job_extra(study) or {})
    except Exception as exc:  # noqa: BLE001 (extras are advice; the job is submitted)
        info["job_extra_error"] = f"{type(exc).__name__}: {exc}"[:200]
    info.update(extra or {})
    write_atomic(d / "job.json", json.dumps(info))
    return info


def find_job(name: str) -> str | None:
    """The id of the one job in the queue with this name (squeue --name), else None."""
    p = S.run(["squeue", "-h", f"--name={name}", "-o", "%i"], timeout=FIND_JOB_S)
    if p is None or p.returncode != 0:
        return None
    ids = p.stdout.split()
    return ids[-1] if len(ids) >= 1 else None


def job_timeout_s(folder: Path, default: float) -> float:
    """For run-study: each run's own timeout as the job was sized (job.json timeout_s: P_SLURM_TIME=estimate makes it
    shorter than P_TIMEOUT), else `default`."""
    try:
        return float(read_job(folder).get("timeout_s") or default)
    except (TypeError, ValueError):
        return float(default)


def failed_submission(rt, request: Path, name: str, info: dict, **keys) -> dict:
    """Writes a failed submission as the study's result ({ok: false, state: failed, why, execution}) and returns it."""
    res = {"name": name, "ok": False, "state": "failed", "why": info.get("why", "could not submit"),
           "execution": {"executor": "slurm", "slurm_job": None}, "translator": rt.version, **keys}
    f = result_path(rt, request, name)
    f.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(f, json.dumps(res))
    return res


# ----------------------------------------------------------------------------- the queue

def squeue(ids: list[str]) -> dict | None:
    """{id: {state, reason, start, elapsed_min, node, start_note?}} of the given jobs still in the queue (one call);
    None when SLURM does not answer (then nothing can be said: the jobs are not 'gone'). SLURM's one-year placeholder
    start (slurm.no_planned_start) reads 'N/A', with start_note saying why."""
    if not ids:
        return {}
    p = S.run(["squeue", "-h", f"--jobs={','.join(str(i) for i in ids)}", "-o", "%i|%T|%r|%S|%M|%N"])
    if p is None:
        return None
    if p.returncode != 0:
        # squeue errors on ids that have all left the queue ("Invalid job id specified"): none listed; anything
        # else (the controller is down) is no answer
        return {} if "invalid job id" in (p.stderr + p.stdout).lower() else None
    out = {}
    for line in p.stdout.splitlines():
        f = line.strip().split("|")
        if len(f) >= 5:
            out[f[0]] = {"state": f[1], "reason": f[2], "start": f[3], "elapsed_min": S.elapsed_minutes(f[4]),
                         "node": (f[5] if len(f) > 5 else "") or None}
            if S.no_planned_start(f[3]):
                out[f[0]].update(start="N/A", start_note=S.NO_START_NOTE)
    return out


def pending_detail() -> list[tuple[float, int, frozenset]] | None:
    """(priority, id, requested nodes) of every pending job in the queue (one squeue call, kept 3 s); the nodes are
    the job's requested node list (squeue %n, expanded), empty when SLURM may choose; None when SLURM does not
    answer."""
    if time.time() - _PENDING["t"] < 3 and _PENDING["v"] is not None:
        return _PENDING["v"]
    p = S.run(["squeue", "-h", "-t", "PENDING", "-o", "%i|%Q|%n"])
    if p is None or p.returncode != 0:
        return None
    out = []
    for ln in p.stdout.split():
        f = ln.split("|")
        jid = f[0].split("_")[0]
        if not jid.isdigit():
            continue
        try:
            prio = float(f[1]) if len(f) > 1 and f[1] else 0.0
        except ValueError:
            prio = 0.0
        req = f[2].strip() if len(f) > 2 else ""
        nodes = frozenset(S.expand_nodelist(req)) if req and req not in ("(null)", "N/A", "None") else frozenset()
        out.append((prio, int(jid), nodes))
    _PENDING.update(t=time.time(), v=out)
    return out


def pending_jobs() -> list[tuple[float, int]] | None:
    """(priority, id) of every pending job in the queue (pending_detail without the nodes); None when SLURM does not
    answer."""
    d = pending_detail()
    return None if d is None else [(p, i) for p, i, _ in d]


def share_node(a: frozenset, b: frozenset) -> bool:
    """Two pending jobs can run on the same node: either may go anywhere (no node list), or their lists meet."""
    return not a or not b or bool(set(a) & set(b))


def ahead_of(job_id: str, pending: list | None = None, share=None) -> int | None:
    """Pending jobs SLURM will consider before this one on the node(s) it can run on: higher priority, or equal and
    submitted earlier, and able to share a node with it (share(nodes_a, nodes_b), default share_node). All of the
    queue's jobs, not only the translators'. pending: pending_detail()'s rows (default), or pending_jobs()' (priority,
    id) pairs, which count as able to go anywhere. Before 2026.11.09.7 jobs pinned to another node of a cluster counted
    too (spark1's job 506 counted spark2's 469-471)."""
    if pending is None:
        pending = pending_detail()
    if pending is None:
        return None
    try:
        me = int(str(job_id).split("_")[0])
    except ValueError:
        return None
    share = share or share_node
    rows = [r if len(r) >= 3 else (r[0], r[1], frozenset()) for r in pending]
    mine = next((r for r in rows if r[1] == me), None)
    if mine is None:
        return sum(1 for _, i, _ in rows if i < me)
    p0, _, n0 = mine
    return sum(1 for p, i, n in rows if i != me and (p > p0 or (p == p0 and i < me)) and share(n0, n))


def sacct_rows(ids: list[str]) -> dict:
    """{id: {state, node}} from SLURM's accounting; {} when there is none (common on single-node installs)."""
    if not ids:
        return {}
    p = S.run(["sacct", "-n", "-P", "-X", f"--jobs={','.join(str(i) for i in ids)}", "-o", "JobID,State,NodeList"])
    if p is None or p.returncode != 0:
        return {}
    out = {}
    for line in p.stdout.splitlines():
        f = line.strip().split("|")
        if len(f) >= 2 and f[0]:
            node = f[2].strip() if len(f) > 2 else ""
            out[f[0].split(".")[0]] = {"state": f[1].strip(),   # e.g. "CANCELLED by 1000": keep who
                                       "node": None if node in ("", "None assigned", "(null)") else node}
    return out


def sacct(ids: list[str]) -> dict:
    """{id: STATE} from SLURM's accounting; {} when there is none."""
    return {k: v["state"] for k, v in sacct_rows(ids).items()}


def scancel(ids: list[str]) -> str | None:
    """Cancels the jobs; returns an error message or None."""
    if not ids:
        return None
    p = S.run(["scancel", *[str(i) for i in ids]])
    if p is None:
        return "scancel not found or did not answer"
    return None if p.returncode == 0 else ((p.stderr or p.stdout).strip()[:300] or f"exit {p.returncode}")


# ----------------------------------------------------------------------------- how a job ended

def default_oom_advice(rt, folder: Path | None) -> str:
    job = read_job(folder) if folder is not None else {}
    n, mem = int(job.get("procs") or 1), job.get("mem")
    unit = "thread" if rt.kind == "threads" else "rank"
    head = "SLURM killed the job for memory" + (f" ({mem} with {n} {unit}{'s' if n != 1 else ''})" if mem else "")
    return head + ": give the study more memory (run.mem, within study_format's caps) or make it smaller"


def ended_state(rt, acct: str | None, log: Path, job: dict | None = None, node: str | None = None) -> tuple[str, str]:
    """(state, why) of a job that left the queue without writing a result: sacct's state when there is accounting,
    else what SLURM wrote in job.log ('DUE TO TIME LIMIT', 'CANCELLED', out of memory: OOM_RE, whatever sacct says).
    node: where the job ran (sacct, squeue), to tell a node that cannot see the run folder. (classify_end on the
    log's tail.)"""
    log = Path(log)
    folder = log.parent
    return classify_end(rt, acct, tail(log), has_log=log.is_file(), folder=folder,
                        job=job if job is not None else read_job(folder), node=node)


def classify_end(rt, acct: str | None, log_tail: str, has_log: bool = True, folder: Path | None = None,
                 job: dict | None = None, node: str | None = None) -> tuple[str, str]:
    """ended_state on the text of job.log's end (has_log False: there is no job.log). folder: the study folder, for
    the out-of-memory advice (hooks.oom_advice) and messages; job: its job.json. A job with neither accounting nor
    job.log never started (SLURM writes job.log when a job starts; a scancel while queued leaves nothing): cancelled
    (cfd-translator's rule; feat called it failed)."""
    a = (acct or "").upper()
    t = log_tail or ""
    job = job or {}
    if a.startswith(("TIMEOUT", "DEADLINE")) or (not a and "DUE TO TIME LIMIT" in t):
        why = (f"SLURM's time limit stopped it before {rt.cli}'s own timeout: raise {rt.var('TIMEOUT')} or the "
               f"study's run.timeout_s (it also sets the job's limit)" + (f" {rt.hooks.timeout_hint}"
                                                                           if rt.hooks.timeout_hint else ""))
        if job.get("time_from") == "estimate":
            why += (f"; this limit came from the study's estimate ({rt.var('SLURM_TIME')}=estimate): give the study "
                    f"run.timeout_s, or set {rt.var('SLURM_TIME')}=timeout")
        return "timed_out", why
    if a.startswith("OUT_OF_MEMORY") or OOM_RE.search(t):
        try:
            adv = (rt.hooks.oom_advice(folder) if rt.hooks.oom_advice and folder is not None
                   else default_oom_advice(rt, folder))
        except Exception:  # noqa: BLE001 (the state matters more than the advice)
            adv = default_oom_advice(rt, folder)
        return "out_of_memory", adv
    if a.startswith("CANCELLED") or (not a and "CANCELLED" in t):
        who = acct.split(" by ", 1)[1].strip() if acct and " by " in acct else None
        return "cancelled", "cancelled" + (f" by {who}" if who else "") + " (scancel or cancel())"
    nodes = S.cluster_nodes() if (not has_log and node) else None
    if nodes and len(nodes) > 1:
        me = S.this_node(nodes)
        if me is None or me.get("NodeName") != node:
            where = f"{folder.parent}" if folder is not None else "the run folder"
            return "failed", (f"the job went to node {node} and left no job.log: that node cannot see {where} "
                              f"(its filesystem is not shared with this one); keep jobs on this node with "
                              f"{rt.var('SLURM_NODE')}=local")
    if not has_log and not a:
        return "cancelled", ("left the queue without starting (no job.log): cancelled while queued (scancel), or "
                             "SLURM dropped it; submit it again if it was not cancelled on purpose")
    if not has_log:
        return "failed", f"the job left the queue without a result and without job.log (SLURM: {acct})"
    end = f"; job.log:\n{t}" if t.strip() else "; job.log is empty"
    return "failed", "the job ended without a result" + (f" (SLURM: {acct})" if acct else "") + end


def job_states(rt, request: Path, names: list[str], submitting_s: float = SUBMITTING_S) -> dict:
    """{name: {state, job_id, ...}} for the request's SLURM studies, from the files and the queue: done (a result),
    queued (reason, expected_start or expected_start_note, ahead), running (job_elapsed_min, node), unknown (SLURM did not answer), a final
    state (FINAL, with why), or queued with reason 'submitting' while a submission may still be under way."""
    request = Path(request)
    jobs, out = {}, {}
    age = time.time() - ((request / "request.json").stat().st_mtime if (request / "request.json").is_file() else 0)
    for n in names:
        if result_path(rt, request, n).is_file():
            out[n] = {"state": "done", "job_id": read_job(request / n).get("job_id")}
            continue
        j = read_job(request / n)
        if j.get("job_id"):
            jobs[n] = j
        elif age < submitting_s:
            out[n] = {"state": "queued", "job_id": None, "reason": "submitting"}
        else:
            out[n] = {"state": "failed", "job_id": None, "why": "no job.json: the study was never submitted"}
    if not jobs:
        return out
    q = squeue([j["job_id"] for j in jobs.values()])
    if q is None:   # SLURM did not answer: the jobs are as they were, not gone
        for n, j in jobs.items():
            out[n] = ({"state": "done", "job_id": j["job_id"]} if result_path(rt, request, n).is_file() else
                      {"state": "unknown", "job_id": j["job_id"], "why": "SLURM did not answer (squeue); ask again later"})
        return out
    ended = {i: v["state"] for i, v in q.items() if v["state"] not in QUEUED and v["state"] not in ACTIVE}
    q = {i: v for i, v in q.items() if i not in ended}      # ended jobs squeue still lists: gone
    gone = [j["job_id"] for n, j in jobs.items() if j["job_id"] not in q and not result_path(rt, request, n).is_file()]
    acct = sacct_rows(gone) if gone else {}
    pend = pending_detail() if any(v["state"] in QUEUED for v in q.values()) else []
    for n, j in jobs.items():
        jid = j["job_id"]
        if result_path(rt, request, n).is_file():
            out[n] = {"state": "done", "job_id": jid}
        elif jid in q:
            st = q[jid]
            if st["state"] in QUEUED:
                rec = {"state": "queued", "job_id": jid, "reason": st["reason"]}
                if st["start"] and st["start"] not in ("N/A", "Unknown", "None"):
                    rec["expected_start"] = st["start"]
                elif st.get("start_note"):
                    rec["expected_start_note"] = st["start_note"]
                rec["ahead"] = ahead_of(jid, pend) if pend is not None else None
            else:
                rec = {"state": "running", "job_id": jid, "job_elapsed_min": st["elapsed_min"]}
                if st.get("node"):
                    rec["node"] = st["node"]
            out[n] = rec
        else:
            row = acct.get(jid, {})
            a = row.get("state") or ended.get(jid)        # no accounting: the final state squeue still shows
            if a and a.upper().startswith(("RUNNING", "COMPLETING")):   # sacct is ahead of squeue for a moment
                out[n] = {"state": "running", "job_id": jid, "job_elapsed_min": None}
                continue
            if a and a.upper().startswith(QUEUED):
                out[n] = {"state": "queued", "job_id": jid, "reason": a}
                continue
            state, why = ended_state(rt, a, request / n / "job.log", job=j, node=row.get("node") or j.get("node"))
            if state == "failed" and (request / n / rt.cancel_file).exists():
                state, why = "cancelled", "cancelled with cancel(request) (it left the queue before running)"
            out[n] = {"state": state, "job_id": jid, "why": why}
    return out


def queue_summary(rt) -> dict:
    """{queue: {running, queued, cpus_in_use}} for this translator's jobs (names '<cli>:...', every user's), and
    {machine: {cpus, cpus_allocated, cpus_idle}} of the nodes its jobs go to (sinfo %C)."""
    out = {"queue": {"running": 0, "queued": 0, "cpus_in_use": 0}, "machine": {}}
    p = S.run(["squeue", "-h", "-o", "%i|%j|%T|%C"])
    if p is not None and p.returncode == 0:
        for line in p.stdout.splitlines():
            f = line.strip().split("|")
            if len(f) >= 4 and f[1].startswith(f"{rt.cli}:"):
                if f[2] in ACTIVE:
                    out["queue"]["running"] += 1
                    out["queue"]["cpus_in_use"] += int(f[3]) if f[3].isdigit() else 0
                elif f[2] in QUEUED:
                    out["queue"]["queued"] += 1
                # (any other state: an ended job squeue still lists for MinJobAge; neither running nor queued)
    nl = S.node_pin(rt)[1]
    # one row per node and partition: counted once per node (summing sinfo %C rows counted shared nodes per partition)
    p = S.run(["sinfo", "-h", "-N", "-o", "%N|%C", *(["-n", nl] if nl else [])])
    if p is not None and p.returncode == 0 and p.stdout.strip():
        per_node: dict = {}
        for row in p.stdout.split():
            name, _, c = row.partition("|")
            try:
                per_node[name] = tuple(int(x) for x in c.split("/"))
            except ValueError:
                continue
        if per_node:
            out["machine"] = {"cpus": sum(v[3] for v in per_node.values()),
                              "cpus_allocated": sum(v[0] for v in per_node.values()),
                              "cpus_idle": sum(v[1] for v in per_node.values())}
    return out


# ----------------------------------------------------------------------------- the tools' common parts

def request_dirs(rt) -> list[Path]:
    """Request folders (run_*, bench_*) under the output root, newest first."""
    root = rt.out_root()
    if not root.is_dir():
        return []
    return sorted((d for g in ("run_*", "bench_*") for d in root.glob(g) if (d / "request.json").is_file()),
                  key=lambda d: d.stat().st_mtime, reverse=True)


def list_jobs(rt, limit: int = 20, active: bool = False) -> dict:
    """The jobs() tool: the newest requests with SLURM jobs, their studies' states (done ones with ok/why, running
    ones with hooks.running_progress), the translator's queue and the machine's CPUs. active: only requests with
    queued or running studies. Within the call's deadline: a request not checked in time says so."""
    from .requests import clock
    out = []
    for d in request_dirs(rt):
        if len(out) >= max(1, int(limit)):
            break
        try:
            meta = json.loads((d / "request.json").read_text())
        except (OSError, ValueError):
            continue
        names = [n for n in meta.get("slurm", []) or meta.get("names", []) if (d / n / "job.json").is_file()]
        if not names:
            continue
        left = S.remaining()
        if left is not None and left < 5:
            out.append({"request": str(d), "started": clock(meta.get("started", d.stat().st_mtime)),
                        "note": "not checked: SLURM was slow and the call ran out of time; ask again"})
            continue
        st = job_states(rt, d, names)
        if active and not any(v["state"] in ("queued", "running", "unknown") for v in st.values()):
            continue
        studies = {}
        for n, v in st.items():
            rec = dict(v)
            if v["state"] == "done":
                try:
                    r = json.loads(result_path(rt, d, n).read_text())
                    rec["ok"] = r.get("ok")
                    if r.get("ok") is False:
                        rec["why"] = r.get("why")
                except (OSError, ValueError):
                    pass
            elif v["state"] == "running" and rt.hooks.running_progress:
                try:
                    rec.update(rt.hooks.running_progress(d / n, (meta.get("studies") or {}).get(n, {})) or {})
                except Exception:  # noqa: BLE001 (progress is extra)
                    pass
            studies[n] = rec
        out.append({"request": str(d), "started": clock(meta.get("started", d.stat().st_mtime)), "studies": studies})
    return {"translator": rt.version, "executor": S.kind(rt), "requests": out, **queue_summary(rt)}


def cancel_request(rt, request: Path, meta: dict, names: list[str] | None = None) -> dict:
    """The common part of cancel(): a cancel flag in each unfinished study's folder (the translator's runner checks
    it, in any process) and scancel for its SLURM jobs. Returns {want, todo, slurm, cancelled_jobs, already_finished,
    error?} or {error} for unknown names; the translator then stops its own local runs and reports the states."""
    request = Path(request)
    known = list(meta.get("names", []))
    want = list(names) if names else known
    unknown = [n for n in want if n not in known]
    if unknown:
        return {"error": f"no studies {unknown} in this request ({known})"}
    todo = [n for n in want if not result_path(rt, request, n).is_file()]
    for n in todo:
        (request / n).mkdir(parents=True, exist_ok=True)
        (request / n / rt.cancel_file).write_text(str(time.time()))
    slurm = [n for n in todo if n in meta.get("slurm", []) or (request / n / "job.json").is_file()]
    ids = [j["job_id"] for j in (read_job(request / n) for n in slurm) if j.get("job_id")]
    out = {"want": want, "todo": todo, "slurm": slurm, "cancelled_jobs": ids,
           "already_finished": [n for n in want if n not in todo]}
    err = scancel(ids)
    if err:
        out["error"] = err
    return out


def take_extension(folder: Path | None) -> float:
    """The running solve's side of extend: seconds asked for in <study folder>/extend, consumed (renamed to
    extend.applied.<time>), the total so far kept in extended.json. Call it every few seconds while the solver runs
    and move its deadline by what it returns."""
    if folder is None:
        return 0.0
    f = Path(folder) / EXTEND_FILE
    try:
        secs = float(f.read_text().strip())
        f.rename(f.with_name(f"{EXTEND_FILE}.applied.{int(time.time())}"))
    except (OSError, ValueError):
        return 0.0
    total = extended_s(folder) + secs
    try:
        write_atomic(Path(folder) / "extended.json", json.dumps({"total_s": total, "at": time.time()}))
    except OSError:
        pass
    return secs


def extended_s(folder: Path) -> float:
    """Seconds added to a study's time limit so far (extended.json)."""
    try:
        return float(json.loads((Path(folder) / "extended.json").read_text()).get("total_s", 0.0))
    except (OSError, ValueError, AttributeError):
        return 0.0


def extend_request(rt, request: Path, meta: dict, studies: dict, hours: float, names: list[str] | None = None) -> dict:
    """The extend(request, hours) tool: for each unfinished study, an 'extend' file the running solve consumes
    (take_extension), within P_MAX_TIMEOUT (the study's timeout plus what was added); for a SLURM job, its limit is
    raised only when the job's time left does not cover the new deadline (hooks.solver_left_s + the hours + 10 min)
    (SLURM may refuse a user's raise: the answer says so). Offer it only where the solver consumes the file."""
    request = Path(request)
    if not (isinstance(hours, (int, float)) and hours > 0):
        return {"error": "hours: a number > 0"}
    known = list(meta.get("names", []))
    want = list(names) if names else known
    unknown = [n for n in want if n not in known]
    if unknown:
        return {"error": f"no studies {unknown} in this request ({known})"}
    cap = S.caps(rt).get("timeout_s")
    out: dict = {"request": str(request), "translator": rt.version, "studies": {}}
    for n in want:
        d = request / n
        rec: dict = {}
        if result_path(rt, request, n).is_file():
            out["studies"][n] = {"state": "finished: nothing to extend"}
            continue
        secs = float(hours) * 3600
        if cap:
            room = float(cap) - S.timeout_s(rt, studies.get(n)) - extended_s(d)
            if room <= 0:
                out["studies"][n] = {"error": f"at {rt.var('MAX_TIMEOUT')} ({cap:g} s) already"}
                continue
            if secs > room:
                secs, rec["capped"] = room, f"to {rt.var('MAX_TIMEOUT')} ({cap:g} s)"
        d.mkdir(parents=True, exist_ok=True)
        (d / EXTEND_FILE).write_text(str(secs))
        rec["added_s"] = round(secs, 1)
        jid = read_job(d).get("job_id")
        if n in meta.get("slurm", []) and not jid:
            rec["slurm_time_limit"] = "no job.json: not raised"
        if jid:
            rec["slurm_job"] = jid
            left = S.job_time_left_min(jid)
            try:
                solver_left = float(rt.hooks.solver_left_s(d, studies.get(n) or {})) if rt.hooks.solver_left_s else 0.0
            except Exception:  # noqa: BLE001
                solver_left = 0.0
            need = solver_left + secs + 600
            if left is not None and (math.isinf(left) or left * 60 >= need):
                rec["slurm_time_limit"] = ("not needed: the job has no SLURM time limit" if math.isinf(left) else
                                           f"not needed: the job has {left / 60:.1f} h left before SLURM's limit")
            else:
                mins = int(math.ceil(secs / 60))
                err = S.extend_job(jid, mins)
                end = f" (in about {left / 60:.1f} h)" if left is not None else ""
                rec["slurm_time_limit"] = "raised" if err is None else (
                    f"not raised ({err}): SLURM lets only an admin raise a running job's limit, so the job still ends "
                    f"at its old limit{end}; an admin can run scontrol update JobId={jid} TimeLimit=+{mins}. With "
                    f"{rt.var('SLURM_TIME')}=max jobs take the partition's own limit, so extend needs no admin")
        rec["note"] = "the running solve takes it within seconds (extended.json records it)"
        out["studies"][n] = rec
    return out

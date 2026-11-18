"""jobs.py — one SLURM job per study: job.sh, submission, states read from the files and the queue, cancel, extend.
Version: 2026.11.18.4 (selftest: a runs 2 x --seconds; 2026.11.18.3: run.after always a SLURM job, refused with a local executor or queue: false; 2026.11.18.2: the gate safe under set -e and re-registering its study, waited_jobs, dependency moves checked,
job_end moves waiters back; 2026.11.18.1: run.after, cancel at='job_end', times with an offset)

    job_script / submit         <request>/<name>/job.sh (a complete script: `sbatch job.sh` reruns the study) and
                                job.json (the id, what was asked and why, the study's run settings)
    job_states(request, names)  {name: {state, job_id, ...}}: one squeue call for all the ids; sacct, else job.log,
                                for jobs that left the queue without a result (ended_state)
    queue_summary, list_jobs    this translator's jobs in the queue and the machine's CPUs; the jobs() tool
    cancel_request, scancel     cancel flags and scancel (the translator stops its local runs)
    extend_request, take_extension   more time for running studies: an 'extend' file the running solve consumes,
                                and SLURM's limit only when the job's time left does not cover the new deadline
    run.after                   a study waits until other runs (of this machine) have ended: after_problems for
                                validation; submit waits for their newest jobs, notify_waiters moves the wait when they
                                submit a next job, gate (job.sh's first step) checks before the study runs
    stop_after_job, stop_requested   cancel(at='job_end'): the current job finishes, no next one is chained

    python -m llmt_runtime.jobs gate <study folder>                  (job.sh of a run.after study)
    python -m llmt_runtime.jobs stop-after-job <request> [names...]  (the router's cancel at='job_end')

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
from .requests import with_offset

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
    --cpus-per-task=N. mpi: --ntasks=N and --ntasks-per-core=1 (OpenMPI 5 refuses more ranks than physical cores in
    the job's CPU set; mochi, cfdt), unless P_SLURM_ARGS already says how to place them; --nodes=1 whenever the job
    is on one node (max_nodes 1, or the ranks fit on one node's cores), unless P_SLURM_ARGS has its own --nodes/-N:
    without it, SLURM may spread an MPI job's ranks over nodes (P_SLURM_NODE=any on a multi-node cluster; before
    2026.11.11.2 only rt.one_node wrote it). Ranks beyond a node's cores, with P_SLURM_NODES above 1: --nodes=K and a
    balanced --ntasks-per-node (slurm.job_layout; 512 ranks on 128-core nodes: 4 x 128)."""
    if rt.hooks.resource_lines:
        return list(rt.hooks.resource_lines(study, procs))
    if rt.kind == "threads":
        return ["--ntasks=1", f"--cpus-per-task={procs}"]
    nodes, per = S.job_layout(rt, procs)
    if nodes > 1:
        out = [f"--ntasks={procs}", f"--nodes={nodes}", f"--ntasks-per-node={per}"]
    else:
        own = S.has_opt(S.slurm_args(rt), "--nodes", shorts=("-N",))
        out = [f"--ntasks={procs}"] + ([] if own else ["--nodes=1"])
    if not S.has_opt(S.slurm_args(rt), "--hint", "--threads-per-core", "--ntasks-per-core", "--cpus-per-task",
                     shorts=("-c",)):
        out.append("--ntasks-per-core=1")
    return out


def run_study_command(rt, request: Path, name: str) -> str:
    return (f"exec {shlex.quote(sys.executable)} -m {rt.package}.cli run-study {shlex.quote(str(request))} "
            f"{shlex.quote(name)}")


GATE_SKIP = 75        # gate's exit code for 'do not run now' (EX_TEMPFAIL); any other code lets the study run


def gate_line(folder: Path) -> str:
    """job.sh's check before a run.after study runs (gate): it ends the job (exit 0) only when the gate says so, i.e.
    a listed run has not really finished (the job waits again behind it) or, with after_mode 'ok', ended otherwise (the
    study is cancelled). Any other outcome, a broken environment included, runs the study: late is worse than never.
    Safe under `set -e` in P_JOB_SETUP or a translator's script lines (2026.11.18.2): the gate's code is caught by
    `|| _llmt_gate=$?`, so neither 75 nor a failing gate ends job.sh with an error, and a job sent back to wait ends
    COMPLETED (exit 0), not FAILED."""
    return (f"_llmt_gate=0; {shlex.quote(sys.executable)} -m llmt_runtime.jobs gate {shlex.quote(str(folder))} "
            f"|| _llmt_gate=$?; [ \"$_llmt_gate\" -eq {GATE_SKIP} ] && exit 0")


def job_script(rt, request: Path, name: str, study: dict, procs: int, time_limit: str, mem: str | None,
               nodelist: str | None, gate: bool = False) -> str:
    """job.sh: the job's name, folder, log, resources, time, memory, node and environment, P_JOB_SETUP's lines, the
    translator's (hooks.script_lines), the run.after gate (gate), then the run-study line. --mem is in the script (cfdt
    and omct passed it to sbatch only, so `sbatch job.sh` lost it)."""
    request = Path(request)
    d = request / name
    pre = S.job_setup_lines(rt) + (list(rt.hooks.script_lines(request, name, study)) if rt.hooks.script_lines else [])
    if gate:
        pre.append(gate_line(d))
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
    me = os.environ.get("SLURM_JOB_ID")
    if after and me and str(me) in [str(a) for a in after] and (d / STOP_FILE).is_file():
        return {"ok": False, "stopped": True,             # a continuation, asked from inside the run's own job
                "why": "stopped with its current job (cancel at='job_end'): no next job; a new request can continue it"}
    if not shutil.which("sbatch"):
        return {"ok": False, "why": f"could not submit: sbatch not found ({rt.var('EXECUTOR')}=slurm needs SLURM)"}
    if time.time() < _SBATCH_DOWN["until"]:
        return {"ok": False, "why": f"could not submit: {_SBATCH_DOWN['why']}"}
    plan = after_plan(rt, request, name, study)           # run.after: wait for other runs to finish
    if plan.get("refuse"):
        return plan["refuse"]
    procs = S.job_procs(rt, study)
    nodes, per_node = S.job_layout(rt, procs) if rt.kind == "mpi" and not rt.hooks.resource_lines else (1, None)
    t, tinfo = S.job_time(rt, study, estimate_min)
    mplan = S.mem_plan(rt, study, nodes)
    add, nodelist, _ = S.node_pin(rt)
    script = d / "job.sh"
    script.write_text(job_script(rt, request, name, study, procs, t, mplan["mem"], add, gate=bool(plan.get("targets"))))
    script.chmod(0o755)
    deps = [str(a) for a in after or []] + [i for i in plan.get("ids", []) if i not in [str(a) for a in after or []]]
    dep = [f"--dependency=afterany:{':'.join(deps)}"] if deps else []
    hold = ["--hold"] if plan.get("hold") else []
    cmd = ["sbatch", "--parsable", *dep, *hold, *S.sbatch_args(rt, study), str(script)]
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
        info["mem_capped_by"] = mplan.get("capped_by")
    if nodes > 1:
        info.update(nodes=nodes, ntasks_per_node=per_node, mem_per="node")
        if mplan.get("total"):
            info["mem_total"] = mplan["total"]
    if estimate_min is not None:
        info["estimate_min"] = round(float(estimate_min), 2)
    if after:
        info["after"] = list(after)
    if plan.get("targets"):
        info.update(run_after=plan["targets"], after_mode=plan["mode"], after_jobs=plan.get("ids", []),
                    held=bool(plan.get("hold")), result=str(result_path(rt, request, name)))
    if adopted is not None:
        info["adopted"] = "sbatch timed out, but SLURM had taken the job: found by its name"
    try:
        if rt.hooks.job_extra:
            info.update(rt.hooks.job_extra(study) or {})
    except Exception as exc:  # noqa: BLE001 (extras are advice; the job is submitted)
        info["job_extra_error"] = f"{type(exc).__name__}: {exc}"[:200]
    info.update(extra or {})
    write_atomic(d / "job.json", json.dumps(info))
    notify_waiters(rt, request, name, str(job_id))
    return info


# ----------------------------------------------------------------------------- run.after: one run after another

STOP_FILE = "stop_after_job"     # cancel(at='job_end'): the running job finishes, no next job is chained (cfdt's name)
WAITERS_FILE = "waiters.json"    # in a run's study folder: the studies waiting for it (run.after)
AFTER_MODES = ("any", "ok")


def after_list(study: dict) -> tuple[list, str]:
    """(refs, mode) of a study's run.after / run.after_mode ('any' by default)."""
    r = S.run_of(study)
    a = r.get("after")
    refs = [a] if isinstance(a, str) else list(a) if isinstance(a, (list, tuple)) else []
    return refs, str(r.get("after_mode") or "any")


def ref_target(rt, request: Path, ref: str) -> tuple[Path, Path, str]:
    """(request folder, study folder, study name) that a run.after ref names: '<study>' of the same request,
    '<request>/<study>' under the output root, or an absolute study folder."""
    ref = str(ref).strip().rstrip("/")
    if "/" not in ref:
        return Path(request), Path(request) / ref, ref
    p = Path(ref).expanduser()
    if not p.is_absolute():
        p = rt.out_root() / p
    return p.parent, p, p.name


def _request_meta(req: Path) -> dict:
    try:
        return json.loads((Path(req) / "request.json").read_text())
    except (OSError, ValueError):
        return {}


def _refs_of(rt, folder: Path) -> list[Path]:
    """The study folders a (submitted or listed) study waits for: its job.json's run_after, else its request.json's
    study run.after."""
    j = read_job(folder)
    if j.get("run_after"):
        return [Path(t["folder"]) for t in j["run_after"]]
    st = (_request_meta(folder.parent).get("studies") or {}).get(folder.name)
    refs, _ = after_list(st if isinstance(st, dict) else {})
    return [ref_target(rt, folder.parent, r)[1] for r in refs]


def after_problems(rt, request: Path | None, name: str, study: dict, studies: dict | None = None) -> list[str]:
    """What is wrong with a study's run.after / run.after_mode ([] when nothing): translators call it in validation
    next to their run-key check (request: the request folder when it exists, else None; studies: the request's
    {name: study}, for refs within it). A ref is '<request>/<study>' of this machine's output root, an absolute study
    folder, or '<study>' of the same request; it must exist, must not be the study itself, and must not close a
    cycle."""
    r = S.run_of(study)
    if r.get("after") is None:
        return ["run.after_mode needs run.after"] if r.get("after_mode") is not None else []
    refs, mode = after_list(study)
    if not refs or not all(isinstance(x, str) and x.strip().strip("/") for x in refs):
        return ["run.after: '<request>/<study>', '<study>' (same request), or a list of them"]
    errs = []
    # only a SLURM job can wait: a study of the local pool starts at once (2026.11.18.3; feat, lmpt and omct send
    # short studies there)
    if S.kind(rt) != "slurm":
        errs.append(f"run.after needs SLURM: {rt.var('EXECUTOR')} is local on this machine, so the study would start "
                    f"at once (choose a machine with SLURM: run.host)")
    elif S.queue_of(study) is False:
        errs.append("run.after needs a SLURM job, and queue: false runs the study in the local pool at once: drop "
                    "queue: false")
    if mode not in AFTER_MODES:
        errs.append("run.after_mode: 'any' (default: start when they end, however they end) or 'ok' (start only if "
                    "they all end ok; otherwise this study is cancelled)")
    studies = studies or {}
    for ref in refs:
        if "/" not in ref.strip().rstrip("/"):
            if ref == name:
                errs.append(f"run.after {ref!r}: a study cannot wait for itself")
            elif studies and ref not in studies:
                errs.append(f"run.after {ref!r}: no study {ref!r} in this request (another request's: "
                            f"'<request>/<study>')")
            continue
        req, folder, sname = ref_target(rt, request or rt.out_root(), ref)
        if request is not None and folder.resolve() == (Path(request) / name).resolve():
            errs.append(f"run.after {ref!r}: a study cannot wait for itself")
            continue
        listed = sname in (_request_meta(req).get("names") or [])
        if not (listed or (folder / "job.json").is_file() or result_path(rt, req, sname).is_file()):
            errs.append(f"run.after {ref!r}: no study {sname} in {req} on this machine (a run of another machine "
                        f"cannot be waited for: run this study there)")
    if errs:
        return errs
    # cycles: within this request (its studies' own refs), then through other requests' (submitted) studies
    graph = {n: [x for x in after_list(s)[0] if "/" not in x.strip().rstrip("/")]
             for n, s in studies.items() if isinstance(s, dict)}
    graph[name] = [x for x in refs if "/" not in x.strip().rstrip("/")]
    seen, path = set(), []

    def loop(n):
        if n in path:
            return path[path.index(n):] + [n]
        if n in seen:
            return None
        seen.add(n)
        path.append(n)
        for m in graph.get(n, []):
            c = loop(m)
            if c:
                return c
        path.pop()
        return None
    c = loop(name)
    if c:
        return ["run.after: a cycle (" + " -> ".join(c) + ")"]
    if request is not None:
        me = (Path(request) / name).resolve()
        todo = [ref_target(rt, request, x)[1] for x in refs if "/" in x.strip().rstrip("/")]
        visited = set()
        while todo:
            f = todo.pop()
            k = str(Path(f).resolve())
            if k in visited:
                continue
            visited.add(k)
            if Path(f).resolve() == me:
                return [f"run.after: a cycle (it comes back to {name} through {f.parent.name}/{f.name})"]
            todo += _refs_of(rt, Path(f))
    return []


def target_ids(folder: Path) -> list[str]:
    """A run's job ids that may still be queued or running: its job.json's job_id and a waiting next job
    (next_job.job_id: a multi-job run that submits job k+1 while k runs, cfdt's chain 'together')."""
    j = read_job(folder)
    ids = [str(j["job_id"])] if j.get("job_id") else []
    nx = (j.get("next_job") or {}).get("job_id")
    if nx and not (j.get("next_job") or {}).get("dropped"):
        ids.append(str(nx))
    return ids


def target_state(t: dict, q: dict | None) -> tuple[str, bool | None, str]:
    """('final' | 'active' | 'unsubmitted' | 'unknown', ok, what) of a run.after target ({folder, result}); q: squeue
    of its ids (None: SLURM did not answer)."""
    res = Path(t["result"])
    if res.is_file():
        try:
            r = json.loads(res.read_text())
        except (OSError, ValueError):
            r = {}
        st = r.get("state") or ("done" if r.get("ok") else "done with ok false")
        return "final", bool(r.get("ok")), st + (f": {str(r.get('why'))[:200]}" if r.get("why") and not r.get("ok")
                                                  else "")
    ids = target_ids(Path(t["folder"]))
    if not ids:
        return "unsubmitted", None, "not submitted yet"
    if q is None:
        return "unknown", None, "SLURM did not answer"
    if any(i in q and (q[i]["state"] in QUEUED or q[i]["state"] in ACTIVE) for i in ids):
        return "active", None, "running or queued"
    return "final", False, "ended without a result"


def after_plan(rt, request: Path, name: str, study: dict) -> dict:
    """What submit needs for run.after: {targets, mode, ids (the listed runs' jobs it waits for), hold (a listed study of
    this request is not submitted yet: held until it is)} or {refuse: submit's answer}. Registers the study in each
    target's waiters.json, so the target's next job moves the wait (notify_waiters)."""
    refs, mode = after_list(study)
    if not refs:
        return {}
    probs = after_problems(rt, request, name, study, _request_meta(request).get("studies") or {})
    if probs:
        return {"refuse": {"ok": False, "why": "could not submit: " + "; ".join(probs)}}
    targets = []
    for ref in refs:
        req, folder, sname = ref_target(rt, request, ref)
        targets.append({"ref": ref, "request": str(req), "name": sname, "folder": str(folder),
                        "result": str(result_path(rt, req, sname))})
    me = str(Path(request) / name)
    for t in targets:                      # before looking at their jobs: a job they submit meanwhile moves ours
        _add_waiter(Path(t["folder"]), me)
    all_ids = [i for t in targets for i in target_ids(Path(t["folder"]))]
    q = squeue(all_ids) if all_ids else {}
    ids, hold = [], False
    for t in targets:
        st, ok, what = target_state(t, q)
        if st == "final" and mode == "ok" and not ok:
            return {"refuse": {"ok": False, "state": "cancelled", "cancelled_by": "run.after",
                               "why": f"cancelled: run.after {t['ref']} ended ({what}), and after_mode is 'ok'"}}
        if st == "active":
            ids += [i for i in target_ids(Path(t["folder"])) if i in q]
        elif st in ("unsubmitted", "unknown"):
            hold = True
    return {"targets": targets, "mode": mode, "ids": list(dict.fromkeys(ids)), "hold": hold}


def _add_waiter(folder: Path, waiter: str) -> None:
    f = Path(folder) / WAITERS_FILE
    try:
        cur = json.loads(f.read_text()) if f.is_file() else []
    except (OSError, ValueError):
        cur = []
    if waiter not in cur:
        Path(folder).mkdir(parents=True, exist_ok=True)
        write_atomic(f, json.dumps(cur + [waiter]))


def notify_waiters(rt, request: Path, name: str, new_job: str | None = None) -> list[str]:
    """A run submitted a job (or ended its submission): the studies waiting for it (waiters.json) now wait for its newest
    job (scontrol update Dependency=afterany:<ids>), and a held one whose listed studies all have jobs, or have ended,
    is released. Option (a) of cfdt's request (2026-11-18): a multi-job run submits each next job while (or after) the
    current one runs, so a wait on the current job alone would let the waiting study start after the first job.
    Returns the updated waiters' job ids. Never raises (the job is submitted)."""
    f = Path(request) / name / WAITERS_FILE
    try:
        waiters = json.loads(f.read_text()) if f.is_file() else []
    except (OSError, ValueError):
        return []
    done, keep = [], []
    for w in waiters:
        try:
            wf = Path(w)
            j = read_job(wf)
            jid = j.get("job_id")
            if not jid or not j.get("run_after") or Path(j.get("result", "")).is_file():
                continue
            q = squeue([str(jid)])
            if q is None:
                keep.append(w)
                continue
            if str(jid) not in q or q[str(jid)]["state"] not in QUEUED:
                continue                       # started, ended or cancelled: nothing to move (the gate checks)
            keep.append(w)
            tids = {i for t in j["run_after"] for i in target_ids(Path(t["folder"]))}
            own = [str(a) for a in j.get("after") or []]
            qq = squeue(sorted(tids | set(own))) or {}
            ids = [i for i in sorted(tids | set(own), key=lambda x: (len(x), x))
                   if i in qq and (qq[i]["state"] in QUEUED or qq[i]["state"] in ACTIVE)]
            if new_job and str(new_job) not in ids and str(new_job) in tids:
                ids.append(str(new_job))
            waiting_unsubmitted = any(target_state(t, qq)[0] == "unsubmitted" for t in j["run_after"])
            moved = True
            if ids and ids != [str(x) for x in j.get("after_jobs") or []]:
                # SLURM 23+: a pending job's dependency may be replaced, held or not (a job id SLURM has already
                # purged is refused: 'Job dependency problem'; the ids are those still in the queue)
                p = S.run(["scontrol", "update", f"JobId={jid}", f"Dependency=afterany:{':'.join(ids)}"])
                moved = p is not None and p.returncode == 0
                if not moved:
                    j["after_move_error"] = ((p.stderr or p.stdout).strip()[-200:] if p is not None
                                             else "scontrol did not answer") + " (the gate in job.sh checks again)"
            if j.get("held") and not waiting_unsubmitted:
                p = S.run(["scontrol", "release", str(jid)])
                if p is not None and p.returncode == 0:
                    j["held"] = False
            if moved:
                j["after_jobs"] = ids
                j.pop("after_move_error", None)
            write_atomic(wf / "job.json", json.dumps(j))
            done.append(str(jid))
        except Exception:  # noqa: BLE001 (a waiter's bookkeeping never fails the run's own submission)
            keep.append(w)
    try:
        write_atomic(f, json.dumps(keep))
    except OSError:
        pass
    return done


def gate(folder: Path, retries: int = 5, pause: float = 30.0) -> int:
    """`python -m llmt_runtime.jobs gate <study folder>`: job.sh's first step for a run.after study (no translator code
    needed: job.json holds the targets and the result's path). 0: every listed run has ended (and, with after_mode
    'ok', ended ok): run the study. GATE_SKIP (75): do not run it now, because a listed run is still active (the job is submitted again,
    waiting for it: a dependency that missed a move) or, with 'ok', one ended otherwise (the study's result says
    'cancelled', with why)."""
    folder = Path(folder)
    j = read_job(folder)
    targets, mode = j.get("run_after") or [], j.get("after_mode") or "any"
    if not targets:
        return 0
    states = []
    for attempt in range(max(1, retries)):
        ids = [i for t in targets for i in target_ids(Path(t["folder"]))]
        q = squeue(ids) if ids else {}
        states = [(t, *target_state(t, q)) for t in targets]
        if not any(s[1] == "unknown" for s in states):
            break
        time.sleep(pause)
    wait = [t for t, st, ok, what in states if st in ("active", "unsubmitted")]
    if wait:
        for t in targets:                  # notify_waiters forgot this study when its job left the queue: again
            _add_waiter(Path(t["folder"]), str(folder))
        ids = [i for t in wait for i in target_ids(Path(t["folder"]))]
        q = squeue(ids) or {}
        ids = [i for i in ids if i in q]
        cmd = [x for x in j.get("command") or ["sbatch", "--parsable", str(folder / "job.sh")]
               if not x.startswith("--dependency") and x != "--hold"]
        cmd[2:2] = [f"--dependency=afterany:{':'.join(ids)}"] if ids else ["--hold"]
        p = S.run(cmd, timeout=SBATCH_TIMEOUT)
        new = p.stdout.strip().splitlines()[-1].split(";")[0].strip() if p is not None and p.returncode == 0 and \
            p.stdout.strip() else None
        print(f"run.after: {', '.join(t['ref'] for t in wait)} has not finished: "
              + (f"waiting again as job {new}" if new else f"could not wait again ({(p.stderr if p else '')[-200:]})"),
              flush=True)
        if new:
            # its own key: previous_jobs is a multi-job run's earlier segments (cfdt), not jobs sent back to wait
            j.update(waited_jobs=list(j.get("waited_jobs") or []) + [str(j.get("job_id"))], job_id=new,
                     after_jobs=ids, held=not ids, command=cmd, submitted=round(time.time(), 1))
            write_atomic(folder / "job.json", json.dumps(j))
            notify_waiters(_Plain, folder.parent, folder.name, new)   # studies waiting for this one follow it
            return GATE_SKIP
        if mode == "any":
            return 0
        res = {"name": folder.name, "ok": False, "state": "failed",
               "why": f"run.after {', '.join(t['ref'] for t in wait)} has not finished, and this job could not wait "
                      f"again ({(p.stderr if p else 'sbatch did not answer')[-200:].strip()}); with after_mode 'ok' "
                      f"it did not run: submit it again",
               "translator": j.get("translator"), "execution": {"executor": "slurm", "slurm_job": j.get("job_id")}}
        out = Path(j.get("result") or folder / f"{folder.name}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(out, json.dumps(res))
        return GATE_SKIP
    bad = [(t, what) for t, st, ok, what in states if mode == "ok" and not ok]
    if bad:
        t, what = bad[0]
        res = {"name": folder.name, "ok": False, "state": "cancelled", "cancelled_by": "run.after",
               "why": f"cancelled: run.after {t['ref']} ended ({what}), and after_mode is 'ok'",
               "translator": j.get("translator"), "execution": {"executor": "slurm", "slurm_job": j.get("job_id")}}
        out = Path(j.get("result") or folder / f"{folder.name}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(out, json.dumps(res))
        print(res["why"], flush=True)
        return GATE_SKIP
    return 0


def waiting_detail(rt, j: dict) -> dict | None:
    """For a queued run.after study: {state: 'waiting', reason, expected_start?} while a listed run has not ended; None
    once all have. The reason says how far the listed runs are (hooks.running_progress: time t of end_time, eta)."""
    from .requests import clock, iso
    targets = j.get("run_after") or []
    if not targets:
        return None
    ids = [i for t in targets for i in target_ids(Path(t["folder"]))]
    q = squeue(ids) if ids else {}
    parts, ends = [], []
    for t in targets:
        st, ok, what = target_state(t, q)
        if st == "final":
            continue
        bits = [what if st != "active" else
                ("running" if any(q and i in q and q[i]["state"] in ACTIVE for i in target_ids(Path(t["folder"])))
                 else "queued")]
        pr = {}
        if rt.hooks.running_progress and bits[0] == "running":
            try:
                study = (_request_meta(Path(t["request"])).get("studies") or {}).get(t["name"], {})
                pr = rt.hooks.running_progress(Path(t["folder"]), study) or {}
            except Exception:  # noqa: BLE001
                pr = {}
        tm, end = pr.get("time"), pr.get("end_time")
        if isinstance(tm, (int, float)) and isinstance(end, (int, float)):
            bits.append(f"t = {tm:.3g} of {end:.3g} s")
        eta = pr.get("eta_min") if pr.get("eta_min") is not None else pr.get("eta_min_max")
        if isinstance(eta, (int, float)):
            ends.append(time.time() + 60 * float(eta))
            bits.append(f"expected finish {clock(ends[-1])}")
        else:
            ends.append(None)
        parts.append(f"{t['ref']} ({', '.join(bits)})")
    if not parts:
        return None
    rec = {"state": "waiting", "reason": f"after {'; '.join(parts)}"}
    if ends and all(e is not None for e in ends):
        rec["expected_start"] = iso(max(ends))
    return rec


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
    """Writes a failed submission as the study's result ({ok: false, state: failed, why, execution}) and returns it.
    A study run.after cancelled at submission (after_mode 'ok', a listed run ended otherwise) keeps state 'cancelled'.
    Studies waiting for this one (run.after) are told: it has ended."""
    res = {"name": name, "ok": False, "state": info.get("state", "failed"), "why": info.get("why", "could not submit"),
           "execution": {"executor": "slurm", "slurm_job": None}, "translator": rt.version, **keys}
    if info.get("cancelled_by"):
        res["cancelled_by"] = info["cancelled_by"]
    f = result_path(rt, request, name)
    f.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(f, json.dumps(res))
    notify_waiters(rt, request, name)
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
            try:                       # a run.after study cancelled because a listed run did not end ok
                r = json.loads(result_path(rt, request, n).read_text())
                if r.get("cancelled_by") == "run.after":
                    out[n].update(state="cancelled", why=r.get("why"))
            except (OSError, ValueError):
                pass
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
                    rec["expected_start"] = with_offset(st["start"])     # SLURM's time, with this machine's offset
                elif st.get("start_note"):
                    rec["expected_start_note"] = st["start_note"]
                rec["ahead"] = ahead_of(jid, pend) if pend is not None else None
                w = waiting_detail(rt, j) if j.get("run_after") else None
                if w:                           # run.after: waiting for other runs, with how far they are
                    rec.update(w)
                    rec.pop("ahead", None)
                    if "expected_start" not in w:
                        rec.pop("expected_start", None)
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
        if active and not any(v["state"] in ("queued", "waiting", "running", "unknown") for v in st.values()):
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


def stop_requested(folder: Path) -> bool:
    """cancel(at='job_end') asked this run to end with its current job (no next job)."""
    return (Path(folder) / STOP_FILE).is_file()


def stop_after_job(rt, request: Path, meta: dict, names: list[str] | None = None) -> dict:
    """cancel(at='job_end'): a stop_after_job flag in each unfinished SLURM study's folder. The running job finishes
    (its results at the time reached) and submit() chains no next job from inside it; a next job already waiting for
    it (next_job, a 'together' chain) is cancelled. A study queued but not started runs its one job, then stops. The run
    stays resumable (a new request continues it). Returns {request, stop_after_job, dropped_next?, not_flagged?}."""
    request = Path(request)
    known = list(meta.get("names", []))
    want = list(names) if names else known
    unknown = [n for n in want if n not in known]
    if unknown:
        return {"error": f"no studies {unknown} in this request ({known})"}
    flagged, skipped, dropped = [], {}, []
    for n in want:
        d = request / n
        if result_path(rt, request, n).is_file():
            skipped[n] = "finished"
            continue
        if not (d / "job.json").is_file():
            skipped[n] = "not a SLURM job: cancel without 'at' stops it now"
            continue
        (d / STOP_FILE).write_text(str(time.time()))
        flagged.append(n)
        cur = read_job(d)
        nx = cur.get("next_job") or {}
        if nx.get("job_id") and not nx.get("dropped"):
            # first marked dropped (as cfdt's drop_next does) and the studies waiting for this run moved back to its
            # current job, then cancelled: afterany counts a cancel, so a study still waiting for the next job when it
            # is cancelled could start at once, while this run's current job still runs
            cur["next_job"] = dict(nx, dropped=round(time.time(), 1))
            write_atomic(d / "job.json", json.dumps(cur))
            notify_waiters(rt, request, n)
            if scancel([str(nx["job_id"])]) is None:
                dropped.append(str(nx["job_id"]))
            else:                                   # it stays: the waiters follow it again
                write_atomic(d / "job.json", json.dumps(dict(cur, next_job=nx)))
                notify_waiters(rt, request, n)
    out = {"request": str(request), "stop_after_job": flagged,
           "note": "each flagged run ends with its current SLURM job (results at the time it reached) and chains no "
                   "next job; a new request can continue it"}
    if dropped:
        out["dropped_next"] = dropped
    if skipped:
        out["not_flagged"] = skipped
    return out


def cancel_request(rt, request: Path, meta: dict, names: list[str] | None = None, at: str | None = None) -> dict:
    """The common part of cancel(): a cancel flag in each unfinished study's folder (the translator's runner checks
    it, in any process) and scancel for its SLURM jobs, a waiting next job (next_job) included. Returns {want, todo,
    slurm, cancelled_jobs, already_finished, error?} or {error} for unknown names; the translator then stops its own
    local runs and reports the states. at='job_end': stop_after_job instead (the current jobs finish)."""
    if at not in (None, "now", "job_end"):
        return {"error": f"at: 'now' (default) or 'job_end', not {at!r}"}
    if at == "job_end":
        return stop_after_job(rt, request, meta, names)
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
    ids = [i for n in slurm for i in target_ids(request / n)]      # the job, and a next job waiting for it
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


# ----------------------------------------------------------------------------- command line

class _Plain:
    """A stand-in runtime for the command line (no translator loaded): default result files."""
    class hooks:  # noqa: N801
        result_file = None


def main(argv: list[str]) -> int:
    if argv[:1] == ["gate"] and len(argv) == 2:
        return gate(Path(argv[1]))
    if argv[:1] == ["stop-after-job"] and len(argv) >= 2:
        req = Path(argv[1]).expanduser()
        meta = _request_meta(req)
        if not meta:
            print(json.dumps({"error": f"no request {req}"}))
            return 1
        out = stop_after_job(_Plain, req, meta, argv[2:] or None)
        print(json.dumps(out))
        return 1 if out.get("error") else 0
    print("usage: python -m llmt_runtime.jobs gate <study folder> | stop-after-job <request> [names...]",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

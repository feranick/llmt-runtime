"""llmt_runtime.selftest — checks this machine's SLURM for what the runtime's run.after and cancel(at='job_end') use,
with no translator:

    python -m llmt_runtime.selftest slurm [--seconds 30] [--keep] [--like CFDT] [--timeout-min 15]
    llmt selftest slurm [...]          (the same, with the release's Python)

Four studies of a request (a, b after a, c after d with after_mode 'ok', d), each a job that sleeps --seconds, go
through the runtime's own code (jobs.submit, notify_waiters, the gate in job.sh, cancel_request(at='job_end')) on the
real queue. Each step prints PASS, FAIL or SKIP (could not be checked here, and why):

    slurm      sbatch, squeue, scontrol, scancel are there; SLURM's version
    validate   after_problems accepts 'b after a' and refuses a study that waits for itself
    submit     a's job (a1)
    wait       b waits for a1 (sbatch --dependency=afterany:a1)
    chain      a's next job a2 waits for a1 (a 'together' chain: job.json keeps a1, a2 under next_job)
    move       b now waits for a2: scontrol update JobId=b Dependency=afterany:a2 on a pending job
    hold       c waits for d, not submitted yet: sbatch --hold
    release    d's submission releases c (scontrol release) and c waits for d
    job_end    cancel(at='job_end') on a: a2 cancelled and marked dropped, b moved back to a1, stop_after_job
               written, and a continuation from inside a1 refused
    gate       b's dependency cleared so it starts while a1 runs: the gate in job.sh sends it back to wait (exit 75
               caught, the job ends COMPLETED with exit 0), job.json names the new job, jobs() does not say failed
    finish     all four end ok, in order (b after a, c after d), a2 never ran, no study reported failed

It takes about 3 x --seconds plus the queue's wait; the folder (~/.llmt/selftest/run_<time>, or LLMT_SELFTEST_OUT)
is removed when everything passed, unless --keep. --like CFDT takes CFDT_SLURM_ARGS and CFDT_SLURM_NODE (a
partition, an account, the node rule) for its jobs; LLMT_SELFTEST_SLURM_ARGS sets them directly.
Exit code: 0 when nothing failed, 1 otherwise, 2 for usage.
Version: 2026.11.18.2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .. import __version__
from .. import jobs as J
from ..config import Runtime

PREFIX = "LLMT_SELFTEST"
POLL = 2.0


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def step(self, name: str, ok: bool | None, detail: str = "") -> bool:
        tag = "PASS" if ok else "SKIP" if ok is None else "FAIL"
        self.rows.append((name, tag, detail))
        print(f"{tag:4}  {name:9} {detail}", flush=True)
        return bool(ok) or ok is None

    @property
    def failed(self) -> list[str]:
        return [n for n, t, _ in self.rows if t == "FAIL"]


def run(cmd: list[str], timeout: float = 30) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None


def show_job(jid) -> dict:
    """scontrol -o show job: {JobState, Reason, Dependency, Priority, ExitCode, ...}; {} when SLURM no longer has it."""
    if not jid:
        return {}
    p = run(["scontrol", "-o", "show", "job", str(jid)])
    if p is None or p.returncode != 0 or not p.stdout.strip():
        return {}
    return dict(re.findall(r"(\w+)=(\S*)", p.stdout.strip().splitlines()[0]))


def names_in(dep: str | None, jid) -> bool:
    return bool(jid) and bool(re.search(rf"(?<![0-9]){re.escape(str(jid))}(?![0-9])", dep or ""))


def active(jid) -> bool:
    q = J.squeue([str(jid)]) or {}
    return str(jid) in q and (q[str(jid)]["state"] in J.QUEUED or q[str(jid)]["state"] in J.ACTIVE)


def slurm_version() -> tuple[str, int | None]:
    p = run(["sbatch", "--version"])
    text = (p.stdout or p.stderr).strip() if p else ""
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    return text or "unknown", (int(m.group(1)) if m else None)


def make_runtime(root: Path, like: str | None) -> Runtime:
    os.environ[f"{PREFIX}_OUT"] = str(root)
    os.environ[f"{PREFIX}_EXECUTOR"] = "slurm"        # run.after needs it (after_problems)
    if like:
        for k in ("SLURM_ARGS", "SLURM_NODE"):
            v = os.environ.get(f"{like.upper()}_{k}")
            if v and not os.environ.get(f"{PREFIX}_{k}"):
                os.environ[f"{PREFIX}_{k}"] = v
    return Runtime(prefix=PREFIX, cli="llmt-selftest", package="llmt_runtime.selftest", version=__version__,
                   kind="threads", out_default=str(root), timeout_default=300, slurm_time_default="timeout",
                   mem_default="200M")


def wait_until(cond, limit_s: float) -> bool:
    end = time.time() + limit_s
    while time.time() < end:
        if cond():
            return True
        time.sleep(POLL)
    return bool(cond())


def slurm(seconds: float = 30, keep: bool = False, like: str | None = None, timeout_min: float = 15) -> int:
    rep = Report()
    print(f"llmt-runtime {__version__} self-test of SLURM ({seconds:g} s jobs)", flush=True)
    missing = [c for c in ("sbatch", "squeue", "scontrol", "scancel") if not shutil.which(c)]
    if missing:
        rep.step("slurm", False, f"not on PATH: {', '.join(missing)}")
        return 1
    text, major = slurm_version()
    rep.step("slurm", True, text + ("" if major is None or major >= 23 else
                                    " (older than 23: the steps below say what works)"))

    root = Path(os.environ.get(f"{PREFIX}_OUT") or "~/.llmt/selftest").expanduser()
    req = root / time.strftime("run_%Y%m%d_%H%M%S")
    req.mkdir(parents=True, exist_ok=False)
    rt = make_runtime(root, like)
    base = {"timeout_s": max(120, int(4 * seconds)), "mem": "200M"}
    studies = {"a": {"seconds": seconds, "run": dict(base)},
               "b": {"seconds": seconds, "run": dict(base, after="a")},
               "c": {"seconds": seconds, "run": dict(base, after="d", after_mode="ok")},
               "d": {"seconds": seconds, "run": dict(base)}}
    meta = {"names": list(studies), "studies": studies, "slurm": list(studies), "selftest": __version__}
    (req / "request.json").write_text(json.dumps(meta, indent=1))
    print(f"      folder    {req}", flush=True)
    ids: set[str] = set()

    def track(info: dict) -> str | None:
        jid = str(info.get("job_id")) if info.get("job_id") else None
        if jid:
            ids.add(jid)
        return jid

    try:
        probs = J.after_problems(rt, req, "b", studies["b"], studies) + \
            J.after_problems(rt, req, "c", studies["c"], studies)
        self_ref = J.after_problems(rt, req, "a", {"run": {"after": "a"}}, studies)
        rep.step("validate", not probs and bool(self_ref),
                 "; ".join(probs) if probs else (f"refuses: {self_ref[0]}" if self_ref else
                                                 "a study waiting for itself was accepted"))

        ia = J.submit(rt, req, "a", studies["a"])
        a1 = track(ia)
        if not rep.step("submit", bool(a1), f"a1 = job {a1}" if a1 else str(ia.get("why"))):
            return finish(rep, req, ids, keep)

        ib = J.submit(rt, req, "b", studies["b"])
        b1 = track(ib)
        dep = show_job(b1).get("Dependency")
        if not b1:
            rep.step("wait", False, str(ib.get("why")))
            return finish(rep, req, ids, keep)
        if not active(a1):
            rep.step("wait", None, f"a1 ended before b was submitted (--seconds {seconds:g} is too short)")
        else:
            rep.step("wait", names_in(dep, a1) and ib.get("after_jobs") == [a1],
                     f"b1 = job {b1}, Dependency={dep}, job.json after_jobs={ib.get('after_jobs')}")

        ia2 = J.submit(rt, req, "a", studies["a"], after=[a1])
        a2 = track(ia2)
        if a2:                                   # a 'together' chain: job.json keeps a1, with a2 waiting under next_job
            J.write_atomic(req / "a" / "job.json", json.dumps(dict(ia, next_job=dict(ia2))))
        dep_a2 = show_job(a2).get("Dependency")
        rep.step("chain", bool(a2) and (names_in(dep_a2, a1) or not active(a1)),
                 f"a2 = job {a2}, Dependency={dep_a2}" if a2 else str(ia2.get("why")))

        jb = J.read_job(req / "b")
        dep = show_job(b1).get("Dependency")
        rep.step("move", bool(a2) and names_in(dep, a2) and jb.get("after_jobs") == [a2],
                 f"b1 Dependency={dep}, job.json after_jobs={jb.get('after_jobs')}"
                 + (f", error: {jb['after_move_error']}" if jb.get("after_move_error") else ""))

        ic = J.submit(rt, req, "c", studies["c"])
        c1 = track(ic)
        sc = show_job(c1)
        rep.step("hold", bool(c1) and ic.get("held") is True and
                 (sc.get("Reason") == "JobHeldUser" or sc.get("Priority") == "0"),
                 f"c1 = job {c1}, Reason={sc.get('Reason')}, Priority={sc.get('Priority')}" if c1 else str(ic.get("why")))

        idd = J.submit(rt, req, "d", studies["d"])
        d1 = track(idd)
        sc, jc = show_job(c1), J.read_job(req / "c")
        rep.step("release", bool(d1) and jc.get("held") is False and sc.get("Reason") != "JobHeldUser"
                 and sc.get("Priority") != "0" and (names_in(sc.get("Dependency"), d1) or not active(d1)),
                 f"d1 = job {d1}; c1 Reason={sc.get('Reason')}, Priority={sc.get('Priority')}, "
                 f"Dependency={sc.get('Dependency')}")

        r = J.cancel_request(rt, req, meta, ["a"], at="job_end")
        old = os.environ.get("SLURM_JOB_ID")
        os.environ["SLURM_JOB_ID"] = str(a1)     # as from inside a1: its continuation is refused
        try:
            cont = J.submit(rt, req, "a", studies["a"], after=[a1])
        finally:
            if old is None:
                os.environ.pop("SLURM_JOB_ID", None)
            else:
                os.environ["SLURM_JOB_ID"] = old
        track(cont)
        time.sleep(1)
        ja, jb = J.read_job(req / "a"), J.read_job(req / "b")
        dep = show_job(b1).get("Dependency")
        a1_on = active(a1)
        checks = {"a2 cancelled": r.get("dropped_next") == [a2] and not active(a2),
                  "marked dropped": bool((ja.get("next_job") or {}).get("dropped")),
                  "stop_after_job written": (req / "a" / J.STOP_FILE).is_file(),
                  "continuation refused": cont.get("stopped") is True,
                  "b back on a1": (names_in(dep, a1) and jb.get("after_jobs") == [a1]) if a1_on else True}
        rep.step("job_end", all(checks.values()),
                 ", ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in checks.items())
                 + (f"; b1 Dependency={dep}" if a1_on else "; a1 had ended (b's move back not checked)"))

        gate_step(rep, rt, req, a1, b1, seconds)

        limit = timeout_min * 60
        done = wait_until(lambda: J.squeue(sorted(ids)) is not None and          # (also: the fake SLURM moves on)
                          all(J.result_path(rt, req, n).is_file() for n in studies), limit)
        res = {n: json.loads(J.result_path(rt, req, n).read_text()) for n in studies
               if J.result_path(rt, req, n).is_file()}
        states = J.job_states(rt, req, list(studies))
        failed = [n for n, s in states.items() if isinstance(s, dict) and s.get("state") == "failed"]
        checks = {"all ended": done,
                  "all ok": all(r.get("ok") for r in res.values()) and len(res) == len(studies),
                  "b after a": done and res["b"]["started"] >= res["a"]["ended"] - 1,
                  "c after d": done and res["c"]["started"] >= res["d"]["ended"] - 1,
                  "a2 never ran": done and str(res["a"].get("slurm_job")) == str(a1),
                  "none reported failed": not failed}
        rep.step("finish", all(checks.values()),
                 ", ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in checks.items())
                 + ("" if done else f" (not all ended within {timeout_min:g} min: {sorted(set(studies) - set(res))} "
                                    f"left)") + (f"; failed: {failed}" if failed else ""))
    except KeyboardInterrupt:
        rep.step("stopped", False, "interrupted")
    except Exception as exc:  # noqa: BLE001 (a self-test reports, it does not crash)
        rep.step("error", False, f"{type(exc).__name__}: {exc}")
    return finish(rep, req, ids, keep)


def gate_step(rep: Report, rt, req: Path, a1: str, b1: str, seconds: float) -> None:
    """b's dependency cleared: it starts while a1 runs, and the gate must send it back to wait."""
    if not active(a1):
        rep.step("gate", None, "a1 had ended: nothing to send back to wait (use a larger --seconds)")
        return
    p = run(["scontrol", "update", f"JobId={b1}", "Dependency="])
    if p is None or p.returncode != 0:
        rep.step("gate", False, "could not clear b1's dependency: " + ((p.stderr or p.stdout).strip() if p else
                                                                      "scontrol did not answer"))
        return
    moved = wait_until(lambda: str(J.read_job(req / "b").get("job_id")) != str(b1) or not active(a1),
                       seconds + 60)
    jb = J.read_job(req / "b")
    b2 = str(jb.get("job_id"))
    if not moved or b2 == str(b1):
        rep.step("gate", None if not active(a1) else False,
                 "b1 did not start while a1 ran (the node was busy)" if not active(a1) else
                 f"b1 started but was not sent back to wait (job.log: {J.tail(req / 'b' / 'job.log', 3)!r})")
        return
    wait_until(lambda: not active(b1), 30)
    s1, s2 = show_job(b1), show_job(b2)
    a_state = J.sacct([str(b1)]).get(str(b1))          # None without accounting (the sparks)
    ended_ok = (s1.get("JobState") == "COMPLETED" and s1.get("ExitCode", "0:0") == "0:0") or \
        (a_state or "").startswith("COMPLETED") or (not s1 and "waiting again as job" in J.tail(req / "b" / "job.log", 5))
    st = J.job_states(rt, req, ["b"]).get("b") or {}
    checks = {"sent back": True, "b1 ended COMPLETED (exit 0)": ended_ok,
              "waited_jobs": jb.get("waited_jobs") == [str(b1)],
              "b2 waits for a1": names_in(s2.get("Dependency"), a1) or not active(a1),
              "jobs() not failed": st.get("state") != "failed"}
    rep.step("gate", all(checks.values()),
             f"b2 = job {b2}; " + ", ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in checks.items())
             + f" (b1 JobState={s1.get('JobState') or a_state or 'purged'}, jobs() state {st.get('state')})")


def finish(rep: Report, req: Path, ids: set, keep: bool) -> int:
    left = []
    for n in ("a", "b", "c", "d"):
        j = J.read_job(req / n)
        left += J.target_ids(req / n) + [str(x) for x in j.get("waited_jobs") or []]
    left = [i for i in dict.fromkeys(list(ids) + left) if active(i)]
    if left:
        J.scancel(left)
        print(f"      cleanup   cancelled jobs {', '.join(left)}", flush=True)
    bad = rep.failed
    if bad or keep:
        print(f"      kept      {req}", flush=True)
    else:
        shutil.rmtree(req, ignore_errors=True)
    print(("FAILED: " + ", ".join(bad)) if bad else "all steps passed (SKIP: not checkable here, see why)", flush=True)
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m llmt_runtime.selftest",
                                 description="Checks this machine's SLURM for run.after and cancel(at='job_end').")
    sub = ap.add_subparsers(dest="what")
    s = sub.add_parser("slurm", help="four short jobs: wait, chain, move, hold, release, job_end, gate")
    s.add_argument("--seconds", type=float, default=30, help="each job's length (default 30)")
    s.add_argument("--keep", action="store_true", help="keep the folder even when everything passed")
    s.add_argument("--like", help="take this translator's SLURM_ARGS and SLURM_NODE (e.g. CFDT)")
    s.add_argument("--timeout-min", type=float, default=15, help="how long to wait for the jobs to end (default 15)")
    a = ap.parse_args(argv)
    if a.what != "slurm":
        ap.print_help()
        return 2
    return slurm(a.seconds, a.keep, a.like, a.timeout_min)

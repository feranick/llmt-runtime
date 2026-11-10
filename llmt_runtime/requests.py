"""requests.py — request folders, request names chosen by the caller, the version a request ran with, files under the
output root, and the time left as clock times.
Version: 2026.11.10.1

    new_request(prefix, request_name)  <out>/<prefix>_<YYYYmmdd_HHMMSS>_<hex6>, or <out>/<request_name>
    check_request_name(name)    ('new' | 'repeat' | 'error', detail): a repeated name returns that request's status,
                                never submits twice (the router retries a call that timed out)
    request_meta(**keys)        request.json's content with "translator" (and "runtime") recorded at submission
    ran_with(meta, results)     {"translator": the version the studies ran with, "translator_now"?: this server's}
    read_file(path), under_root(path)   only files under the output root
    clock(t), slurm_start(s)    'HH:MM (tomorrow)'; SLURM's expected start as epoch seconds
    timing_slurm(...), timing_local(...)   the request's time_left and 'next'
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import re
import time
import uuid
from pathlib import Path

from . import slurm as S

REQUEST_NAME = re.compile(r"^run_[A-Za-z0-9_.-]{1,80}$")
POLL_LIMIT_MIN = 5.0         # longer than this: tell the user when to come back instead of polling


# ----------------------------------------------------------------------------- folders and names

def new_request(rt, prefix: str = "run", request_name: str | None = None) -> Path:
    """A new request folder under the output root: <prefix>_<time>_<hex6>, or the caller's request_name (which must
    not exist: FileExistsError when another call took it first)."""
    root = rt.out_root()
    if request_name:
        d = root / request_name
        d.mkdir(parents=True, exist_ok=False)
        return d
    d = root / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_request_name(rt, name) -> tuple[str, dict | Path | None]:
    """('new', None) | ('repeat', folder: answer with status(folder) and a note, submit nothing) | ('error', {error,
    request?}). A name is 'run_' followed by letters, digits, _ . - (up to 80)."""
    if name is None:
        return "new", None
    if not isinstance(name, str) or not REQUEST_NAME.match(name):
        return "error", {"error": "request_name: 'run_' followed by letters, digits, _ . - (up to 80)"}
    d = rt.out_root() / name
    if (d / "request.json").is_file():
        return "repeat", d
    if d.exists():
        return "error", {"error": f"request_name {name!r}: the folder exists but holds no request (still being "
                                  "submitted by an earlier call? call again in a few seconds)", "request": str(d)}
    return "new", None


REPEAT_NOTE = "this request_name was submitted before: its status, nothing submitted again"


def request_meta(rt, **keys) -> dict:
    """request.json's content: the translator's keys, plus the versions it is submitted with (status reports what a
    request ran with, not what is installed when it is asked: a 2026.11.09.2 bench read '2026.11.09.3' after an
    upgrade)."""
    return {**keys, "translator": rt.version, "runtime": rt.runtime_version}


def ran_with(rt, meta: dict, results: dict) -> dict:
    """{"translator": the version a request's studies ran with (their results' own, else the one recorded at
    submission; a list when they differ), "translator_now": this server's when it is another}; translator_note when
    the request recorded none (submitted before version recording)."""
    vs = sorted({r["translator"] for r in results.values() if isinstance(r, dict) and r.get("translator")})
    ran = (vs[0] if len(vs) == 1 else vs) if vs else meta.get("translator")
    out = {"translator": ran if ran else rt.version}
    if not ran:
        out["translator_note"] = "the version it ran with was not recorded: this server's"
    elif ran != rt.version:
        out["translator_now"] = rt.version
    return out


# ----------------------------------------------------------------------------- files under the output root

def under_root(rt, path: str) -> Path:
    """The resolved path when it is under the output root (ValueError otherwise)."""
    p = Path(path).expanduser().resolve()
    root = rt.out_root().resolve()
    if root not in p.parents:
        raise ValueError(f"only files under {root} can be read")
    return p


def read_file(rt, path: str, max_mb: float = 20) -> dict:
    """{name, bytes, base64} of a file under the output root; {error} otherwise."""
    try:
        p = under_root(rt, path)
    except ValueError as exc:
        return {"error": str(exc)}
    if not p.is_file():
        return {"error": f"no file {p}"}
    if p.stat().st_size > max_mb * 1e6:
        return {"error": f"{p.stat().st_size / 1e6:.1f} MB is more than max_mb"}
    return {"name": p.name, "bytes": p.stat().st_size, "base64": base64.b64encode(p.read_bytes()).decode()}


def image_path(rt, path: str, suffixes: tuple = (".png",)) -> Path:
    """An image written by a run (only under the output root); ValueError otherwise."""
    try:
        p = under_root(rt, path)
    except ValueError:
        p = Path(path).expanduser().resolve()
        raise ValueError(f"no image {p} under {rt.out_root().resolve()}") from None
    if p.suffix.lower() not in suffixes or not p.is_file():
        raise ValueError(f"no image {p} under {rt.out_root().resolve()}")
    return p


# ----------------------------------------------------------------------------- times

def clock(t: float, now: float | None = None) -> str:
    """A local clock time, with its day when it is not today: 'HH:MM', 'HH:MM (tomorrow)', 'HH:MM (yesterday)', or
    'YYYY-MM-DD HH:MM'."""
    when = dt.datetime.fromtimestamp(t)
    days = (when.date() - dt.datetime.fromtimestamp(time.time() if now is None else now).date()).days
    hm = when.strftime("%H:%M")
    return {0: hm, 1: f"{hm} (tomorrow)", -1: f"{hm} (yesterday)"}.get(days, when.strftime("%Y-%m-%d %H:%M"))


def slurm_start(s: str | None) -> float | None:
    """SLURM's expected start (local ISO time, e.g. 2026-10-18T22:22:27) as epoch seconds; None for N/A, Unknown and
    SLURM's one-year placeholder (slurm.no_planned_start)."""
    return None if S.no_planned_start(s) else S.start_epoch(s)


def timing_slurm(folder: Path, progress: dict, ids: dict, tool: str = "study_status") -> tuple[dict, str]:
    """Request-level time left when studies are SLURM jobs (progress: {name: {state, eta_min, eta_min_max,
    expected_start?}}). A running study ends after its own time left; a queued one starts at SLURM's expected start
    (planned from the time limits of the jobs ahead: a latest start), or when the request's first running study frees
    its CPUs if that is sooner, and then needs its run time."""
    now = time.time()
    exp_end, max_end = [], []
    run_ends = sorted(now + 60 * p["eta_min"] for p in progress.values()
                      if p.get("state") != "queued" and p.get("eta_min") is not None)
    for p in progress.values():
        if p.get("state") != "queued":
            s_exp = s_max = now
        else:
            s_max = slurm_start(p.get("expected_start"))
            s_exp = min(s_max, run_ends[0]) if s_max is not None and run_ends else s_max
        exp_end.append(None if s_exp is None or p.get("eta_min") is None else s_exp + 60 * p["eta_min"])
        max_end.append(None if s_max is None or p.get("eta_min_max") is None else s_max + 60 * p["eta_min_max"])
    t_exp = max(exp_end) if exp_end and all(e is not None for e in exp_end) else None
    t_max = max(max_end) if max_end and all(e is not None for e in max_end) else None
    t = {"eta_min": None if t_exp is None else round((t_exp - now) / 60, 1),
         "eta_min_max": None if t_max is None else round((t_max - now) / 60, 1),
         "expected_finish": None if t_exp is None else clock(t_exp),
         "latest_finish": None if t_max is None else clock(t_max), "includes_queue_wait": True}
    if t_exp is not None and t_max is not None:
        when = f"around {t['expected_finish']} (at the latest {t['latest_finish']})"
    elif t_max is not None:
        when = f"by {t['latest_finish']} at the latest (no estimate of the run time yet)"
    elif any(p.get("expected_start_note") == S.NO_START_NOTE for p in progress.values()):
        when = ("when the job ahead of them with no time limit ends (SLURM gives no start time: it cannot plan behind "
                "a job without a time limit)")
    else:
        when = "when the jobs ahead of them in the queue are done (SLURM gives no start time yet)"
    arg = "bench" if tool == "bench_status" else "request"
    nxt = (f"The studies are SLURM jobs on this machine (ids {', '.join(str(v) for v in ids.values())}). They keep "
           f"running without this conversation. Tell the user they should finish {when} and stop; collect them "
           f"later with {tool}({arg}='{folder}') or find them with jobs().")
    return t, nxt


def timing_local(folder: Path, progress: dict, workers: int, tool: str = "study_status",
                 poll_limit_min: float = POLL_LIMIT_MIN) -> tuple[dict, str]:
    """Request-level time left in the local worker pool (progress: {name: {stage, eta_min, eta_min_max}}, stage
    'queued' for studies waiting for a worker) and the advice: poll soon, or tell the user when to come back."""
    workers = max(1, int(workers))
    queued = [p for p in progress.values() if p.get("stage") == "queued"]
    running = [p for p in progress.values() if p.get("stage") != "queued"]
    exp = [p["eta_min"] for p in running if p.get("eta_min") is not None]
    exp_q = [p["eta_min"] for p in queued if p.get("eta_min") is not None]
    rounds = -(-len(queued) // workers)
    known = len(exp) == len(running) and len(exp_q) == len(queued)
    eta = max(exp, default=0) + rounds * max(exp_q, default=0) if known else None
    eta_max = (max((p.get("eta_min_max") or 0 for p in running), default=0)
               + rounds * max((p.get("eta_min_max") or 0 for p in queued), default=0))
    now = time.time()
    t = {"eta_min": None if eta is None else round(eta, 1), "eta_min_max": round(eta_max, 1),
         "expected_finish": None if eta is None else clock(now + 60 * eta), "latest_finish": clock(now + 60 * eta_max)}
    basis = eta if eta is not None else eta_max
    arg = "bench" if tool == "bench_status" else "request"
    if basis > poll_limit_min:
        when = t["expected_finish"] or f"at the latest {t['latest_finish']}"
        nxt = (f"this run needs about {basis:.0f} more min: tell the user it should finish around {when} "
               f"(at the latest {t['latest_finish']}) and stop; do not poll. Collect the results later with "
               f"{tool}({arg}='{folder}') when the user asks, or from a task scheduled after that time.")
    else:
        nxt = f"call {tool}({arg}='{folder}', wait_s={min(50, int(60 * basis) + 10)}) to collect the rest"
    return t, nxt


def until(wait: float, spare: float = 3.0) -> float:
    """When a wait of `wait` s ends: within the call's own deadline (slurm.deadline), `spare` s left for the answer."""
    from .slurm import remaining
    t = time.time() + max(0.0, float(wait))
    left = remaining()
    return t if left is None else min(t, time.time() + left - spare)


def write_json(path: Path, data) -> None:
    """A JSON file written atomically."""
    from .jobs import write_atomic
    write_atomic(Path(path), json.dumps(data))


def env_out(rt) -> dict:
    """{P_OUT: the output root} for a worker process that may outlive a change of P_OUT."""
    return {rt.var("OUT"): str(rt.out_root())}

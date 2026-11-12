"""slurm.py — the SLURM settings, the machine, caps, time limits and a call's time budget.
Version: 2026.11.12.1

Settings (P = the translator's prefix; llm-translators docs/EXECUTION.md):
    P_EXECUTOR local | slurm (local)        P_SLURM_MIN_MINUTES (rt.min_minutes): shorter studies run locally
    P_SLURM_ARGS: extra sbatch arguments, split as a shell does; a --mem there wins over P_SLURM_MEM, a --nodelist/-w
        over P_SLURM_NODE, a --partition/-p picks the partition whose limit 'max' takes
    P_SLURM_NODE local (default: --nodelist=<this node> when the cluster has several nodes) | any | a node list
    P_SLURM_NODES the most nodes one MPI job may use (default 1; 1 for threaded translators and rt.one_node; at most
        the nodes jobs may go to, so a multi-node machine sets P_SLURM_NODE=any or a node list). A job whose ranks
        exceed a node's cores spans ceil(ranks / cores) nodes with the ranks balanced (job_layout); caps.procs is
        then cores x nodes, and mem auto is split over the nodes (--mem is per node in SLURM). The nodes counted are
        those of P_SLURM_ARGS' --partition and plain --constraint features (job_nodes), so a cluster with several
        kinds of node (Perlmutter: '-C cpu') is sized by the right ones.
    P_JOB_SETUP: shell lines written into job.sh before the command (module loads, the OpenFOAM bashrc), so jobs do
        not depend on the submitting shell's environment; several lines separated by newlines or a literal '\n'.
    P_SLURM_MEM auto (default: the translator's estimate, rt.mem_default when unknown; never above the node's memory)
        | a size | none
    P_SLURM_TIME max | timeout | estimate | minutes (default rt.slurm_time_default):
        timeout   the study's runs' timeouts + their margins + 10 min (hooks.time_need_min)
        estimate  max(P_SLURM_TIME_MIN (20), ceil(P_SLURM_TIME_FACTOR (3) x the study's estimate) + 10), at most
                  'timeout'; each run's own timeout then fits in it (job_time()'s info.timeout_s, recorded in job.json:
                  (limit - 10 min) / solves, or hooks.timeout_for_limit; run-study reads jobs.job_timeout_s). A study
                  with run.timeout_s, or without an estimate, takes 'timeout'.
        max       the partition's own limit (UNLIMITED where it has none): SLURM lets only an admin raise a running
                  job's limit (mochi job 3177, 'Access/permission denied'), so extend() then needs no admin. The price:
                  SLURM cannot plan a start behind an UNLIMITED job holding a node's memory, and the router then sees
                  that machine as busy indefinitely.
        minutes   that many, at least 'timeout'
    P_PROCS (1), P_TIMEOUT (rt.timeout_default s); caps P_MAX_TIMEOUT (s), P_MAX_MEM, P_MAX_PROCS (unset: none)
Per study: "run": {"timeout_s", "mem", "procs", "queue"}.

The machine: cluster_nodes() (scontrol -o show node: kept 60 s; a failure is asked again, but not within 5 s, so
a dead controller costs one timeout per call), this_node(), node_pin() / node_status(), node_cores(), node_mem_mb()
(what a job can get per node: RealMemory less MemSpecLimit, within the partition's MaxMemPerNode and MaxMemPerCPU x
CPUs; node_mem_limit says which), caps() with it.

SLURM's start times: start_epoch(s); no_planned_start(s) for its one-year placeholder (NO_START_DAYS, NO_START_NOTE).

A call's time budget: deadline(seconds) bounds every SLURM command of the thread (run()) to what is left of the call,
so each MCP tool answers within its wait_s; remaining() says how much.
"""
from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
from contextlib import contextmanager

# ----------------------------------------------------------------------------- a call's time budget

_CALL = threading.local()


@contextmanager
def deadline(seconds: float):
    """Bounds the SLURM commands (and anything that asks remaining()) of this thread to `seconds` from now."""
    prev = getattr(_CALL, "until", None)
    until = time.time() + max(0.0, float(seconds))
    _CALL.until = until if prev is None else min(prev, until)
    try:
        yield
    finally:
        _CALL.until = prev


def remaining() -> float | None:
    """Seconds left in this thread's call; None outside a call (CLI, SLURM jobs, workers: no limit)."""
    until = getattr(_CALL, "until", None)
    return None if until is None else until - time.time()


def run(cmd: list[str], timeout: float = 15) -> subprocess.CompletedProcess | None:
    """A SLURM (or system) command's result; None when it is missing, fails to run or does not answer in time
    (within the call's remaining time, in an MCP call)."""
    if not shutil.which(cmd[0]):
        return None
    left = remaining()
    if left is not None:
        if left < 1:
            return None
        timeout = min(timeout, left - 0.5)
    try:
        # stdin from /dev/null: an MCP server speaks on its own stdio, which no command it runs may read
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None


# ----------------------------------------------------------------------------- settings

def kind(rt) -> str:
    k = rt.env("EXECUTOR", "local").strip().lower()
    return k if k in ("local", "slurm") else "local"


def _float(rt, name: str, default: float) -> float:
    try:
        return float(rt.env(name, "") or default)
    except ValueError:
        return float(default)


def min_minutes(rt) -> float:
    return _float(rt, "SLURM_MIN_MINUTES", rt.min_minutes)


def default_procs(rt) -> int:
    try:
        return max(1, int(float(rt.env("PROCS", "1") or 1)))
    except ValueError:
        return 1


def run_of(study: dict | None) -> dict:
    r = (study or {}).get("run")
    return r if isinstance(r, dict) else {}


def timeout_s(rt, study: dict | None = None) -> float:
    """Seconds per run: the study's run.timeout_s, else P_TIMEOUT (rt.timeout_default)."""
    t = run_of(study).get("timeout_s")
    if t:
        return float(t)
    return _float(rt, "TIMEOUT", rt.timeout_default)


def procs_of(rt, study: dict | None) -> int:
    """Ranks or threads asked for: run.procs, then the study's own (hooks.study_procs, default study['procs']), then
    P_PROCS (before any size cap)."""
    st = study or {}
    own = rt.hooks.study_procs(st) if rt.hooks.study_procs else st.get("procs")
    try:
        return max(1, int(run_of(st).get("procs") or own or default_procs(rt)))
    except (TypeError, ValueError):
        return default_procs(rt)


def job_procs(rt, study: dict) -> int:
    """What the job asks for: hooks.procs (the translator's size caps), else procs_of capped at the node's cores (x
    max_nodes when jobs may span nodes)."""
    if rt.hooks.procs:
        return max(1, int(rt.hooks.procs(study)))
    return max(1, min(procs_of(rt, study), max_job_procs(rt)))


def queue_of(study: dict | None):
    """run.queue, then the study's queue; None: decided by the time estimate."""
    q = run_of(study).get("queue")
    return q if q is not None else (study or {}).get("queue")


def use_slurm(rt, study: dict, estimate_min: float | None) -> bool:
    """Whether a study becomes a SLURM job: only with P_EXECUTOR=slurm; then "queue" decides if given, else the
    estimate (none: SLURM)."""
    if kind(rt) != "slurm":
        return False
    q = queue_of(study)
    if q is not None:
        return bool(q)
    return estimate_min is None or estimate_min >= min_minutes(rt)


# ----------------------------------------------------------------------------- sbatch arguments

def slurm_args(rt) -> list[str]:
    """P_SLURM_ARGS split as a shell does ("--mem=8G --nodelist=spark-5da5"); unbalanced quotes: the plain words."""
    v = rt.env("SLURM_ARGS", "")
    try:
        return shlex.split(v)
    except ValueError:
        return v.split()


def opt_value(args: list[str], longs: tuple, shorts: tuple = ()) -> str | None:
    """The value of an sbatch option among the arguments ('--nodelist=a', '--nodelist a', '-w a', '-wa', '-w=a');
    '' when given without a value; None when absent. Only exact long names match ('--mem' is not '--mem-bind')."""
    for i, a in enumerate(args):
        for lo in longs:
            if a == lo:
                return args[i + 1] if i + 1 < len(args) else ""
            if a.startswith(lo + "="):
                return a[len(lo) + 1:]
        for sh in shorts:
            if a == sh:
                return args[i + 1] if i + 1 < len(args) else ""
            if a.startswith(sh) and not a.startswith("--"):
                return a[len(sh):].lstrip("=")
    return None


def has_opt(args: list[str], *longs: str, shorts: tuple = ()) -> bool:
    return opt_value(args, tuple(longs), tuple(shorts)) is not None


def args_mem(rt) -> str | None:
    """A --mem / --mem-per-cpu in P_SLURM_ARGS (it wins over P_SLURM_MEM)."""
    return opt_value(slurm_args(rt), ("--mem", "--mem-per-cpu"))


def args_nodelist(rt) -> str | None:
    """A --nodelist / -w in P_SLURM_ARGS (it wins over P_SLURM_NODE)."""
    return opt_value(slurm_args(rt), ("--nodelist",), ("-w",))


def sbatch_args(rt, study: dict | None) -> list[str]:
    """P_SLURM_ARGS for this study's sbatch: with run.mem, its --mem/--mem-per-cpu are dropped (the rest kept)."""
    args = slurm_args(rt)
    if not run_of(study).get("mem"):
        return args
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--mem", "--mem-per-cpu"):
            skip = True
            continue
        if a.startswith(("--mem=", "--mem-per-cpu=")):
            continue
        out.append(a)
    return out


# ----------------------------------------------------------------------------- sizes and times

def mem_mb(v) -> float | None:
    """A SLURM memory size (500M, 8G, 1T, 8GB, a plain number in MB) in MB; None if malformed."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", str(v), re.I)
    if not m:
        return None
    return float(m.group(1)) * {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2).upper()]


def mem_text(mb: float) -> str:
    """MB as SLURM writes a size, exactly: whole GB as '62G', else MB ('64000M', '500M'). (Rounded down to GB, a node's
    64000 MB read '62G' and refused a run.mem of 64000M.)"""
    mb = int(mb)
    return f"{mb // 1024}G" if mb >= 1024 and mb % 1024 == 0 else f"{mb}M"


def mem_arg(v) -> str | None:
    """A size as sbatch's --mem takes it: '8G' and '500' as given; what SLURM would refuse but mem_mb reads ('8GB',
    '1.5G', ' 4 G ') in whole MB; None when it is not a size."""
    s = str(v).strip()
    if re.fullmatch(r"\d+[KMGT]?", s, re.I):
        return s
    mb = mem_mb(s)
    return None if mb is None else f"{int(math.ceil(mb))}M"


def slurm_minutes(t: str | None) -> float | None:
    """A SLURM time ('infinite', 'UNLIMITED', '7-00:00:00', '2:00:00', '30', '1-12', '5:30') in minutes; inf for no
    limit; None when it cannot be read (N/A, INVALID, '')."""
    if not t:
        return None
    t = t.strip().lower()
    if t in ("infinite", "unlimited"):
        return math.inf
    try:
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
            f = [int(x) for x in t.split(":")] + [0, 0]
            h, m, sec = f[0], f[1], f[2]
        else:
            f = [int(x) for x in t.split(":")]
            if len(f) > 3:
                return None
            h, m, sec = {1: (0, f[0], 0), 2: (0, f[0], f[-1]), 3: tuple(f)}[len(f)]
        return days * 1440 + h * 60 + m + sec / 60
    except (ValueError, KeyError):
        return None


NO_START_DAYS = 300
NO_START_NOTE = ("SLURM gives no start time: a job ahead of it (or running on the node) has no time limit; it starts "
                 "when that one ends")


def start_epoch(s: str | None) -> float | None:
    """SLURM's start time (local ISO time, e.g. 2026-10-18T22:22:27) as epoch seconds; None for N/A, Unknown, None."""
    if not s:
        return None
    try:
        return time.mktime(time.strptime(s.strip(), "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def no_planned_start(s: str | None, now: float | None = None) -> bool:
    """A start NO_START_DAYS (300) days or more ahead: SLURM's placeholder when it cannot plan the job, typically
    behind a job with no time limit (carbonio 2026-11-10: three cfdt jobs behind a feat job with --time UNLIMITED
    showed 2027-11-09)."""
    t = start_epoch(s)
    return t is not None and t - (time.time() if now is None else now) >= NO_START_DAYS * 86400


def elapsed_minutes(t: str | None) -> float | None:
    """squeue's elapsed time ([D-]HH:MM:SS, MM:SS) in minutes, to 0.1; None when unreadable."""
    if not t:
        return None
    try:
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
        parts = [int(x) for x in t.strip().split(":")]
        if len(parts) > 3:
            return None
        while len(parts) < 3:
            parts.insert(0, 0)
        h, m, s = parts
        return round(days * 1440 + h * 60 + m + s / 60, 1)
    except ValueError:
        return None


# ----------------------------------------------------------------------------- the machine

NODES_TTL = 60.0          # an answer of scontrol is reused this long
NODES_RETRY_S = 5.0       # a failure is not asked again sooner (one timeout per call when the controller is down)
_NODES: dict = {"t": 0.0, "nodes": None, "failed": 0.0}
_PART: dict = {"t": 0.0, "key": None, "v": None}


_PARTS: dict = {"t": 0.0, "v": None, "failed": 0.0}


def forget() -> None:
    """Drops what was learnt about the cluster (tests; after a configuration change)."""
    _NODES.update(t=0.0, nodes=None, failed=0.0)
    _PART.update(t=0.0, key=None, v=None)
    _PARTS.update(t=0.0, v=None, failed=0.0)


def cluster_nodes() -> list[dict] | None:
    """The cluster's nodes from `scontrol -o show node`, one {key: value} per node; None when SLURM does not answer.
    An answer is reused for NODES_TTL s; a failure is never remembered beyond NODES_RETRY_S (a controller that was
    down when the server started must not count as 'no SLURM' for the server's lifetime)."""
    now = time.time()
    if _NODES["nodes"] is not None and now - _NODES["t"] < NODES_TTL:
        return _NODES["nodes"]
    if now - _NODES["failed"] < NODES_RETRY_S:
        return None
    p = run(["scontrol", "-o", "show", "node"], timeout=10)
    nodes = [dict(re.findall(r"(\w+)=(\S*)", ln)) for ln in p.stdout.splitlines() if "NodeName=" in ln] \
        if p is not None and p.returncode == 0 else []
    if not nodes:
        _NODES["failed"] = now
        return None
    _NODES.update(t=now, nodes=nodes, failed=0.0)
    return nodes


def node_cpus(n: dict) -> int:
    try:
        return int(n.get("CPUTot") or 0)
    except ValueError:
        return 0


def node_core_count(n: dict) -> int:
    """Physical cores of a node (jobs ask for one rank or thread per core): Boards x Sockets x CoresPerSocket when
    SLURM gives them (at most its CPUs: a node configured with CPUs= the core count), else CPUs / threads per core."""
    cpus = node_cpus(n)
    try:
        cores = int(n.get("Boards") or 1) * int(n.get("Sockets") or 0) * int(n.get("CoresPerSocket") or 0)
    except ValueError:
        cores = 0
    if cores > 0:
        return max(1, min(cores, cpus) if cpus else cores)
    try:
        return max(1, cpus // max(1, int(n.get("ThreadsPerCore") or 1)))
    except ValueError:
        return max(1, cpus)


def partitions() -> list[dict] | None:
    """The cluster's partitions from `scontrol -o show partition`, one {key: value} each; kept and retried as
    cluster_nodes; None when SLURM does not answer."""
    now = time.time()
    if _PARTS["v"] is not None and now - _PARTS["t"] < NODES_TTL:
        return _PARTS["v"]
    if now - _PARTS["failed"] < NODES_RETRY_S:
        return None
    p = run(["scontrol", "-o", "show", "partition"], timeout=10)
    if p is None or p.returncode != 0:
        _PARTS["failed"] = now
        return None
    v = [dict(re.findall(r"(\w+)=(\S*)", ln)) for ln in p.stdout.splitlines() if "PartitionName=" in ln]
    _PARTS.update(t=now, v=v, failed=0.0)
    return v


def job_partition(rt) -> dict | None:
    """The partition this translator's jobs go to: P_SLURM_ARGS' --partition/-p, else the default one (Default=YES);
    None when SLURM does not say."""
    parts = partitions() or []
    want = args_node_filter(rt)[0]
    for p in parts:
        if (want and p.get("PartitionName") == want) or (not want and p.get("Default") == "YES"):
            return p
    return None


def _mb_field(d: dict, key: str) -> float | None:
    try:
        v = float(d.get(key) or 0)
    except ValueError:              # UNLIMITED, N/A
        return None
    return v if v > 0 else None


def node_job_mb(n: dict, part: dict | None = None) -> tuple[float | None, str]:
    """(MB, why) a job can get on one node: RealMemory less MemSpecLimit (the memory SLURM keeps for the system), and
    no more than the partition's MaxMemPerNode or MaxMemPerCPU x the node's CPUs (CPUEfctv, else CPUTot) when set.
    Perlmutter (RealMemory 515100M) refused --mem=480G: 'Memory required by task is not available'."""
    real = node_real_mb(n)
    if real is None:
        return None, ""
    mb, why = real, "a node's memory (SLURM RealMemory)"
    spec = _mb_field(n, "MemSpecLimit")
    if spec and spec < real:
        mb, why = real - spec, f"a node's memory less MemSpecLimit (SLURM RealMemory {int(real)}M - {int(spec)}M)"
    if part:
        name = part.get("PartitionName", "?")
        mpn = _mb_field(part, "MaxMemPerNode")
        if mpn and mpn < mb:
            mb, why = mpn, f"partition {name}'s MaxMemPerNode ({int(mpn)}M)"
        mpc = _mb_field(part, "MaxMemPerCPU")
        cpus = _mb_field(n, "CPUEfctv") or node_cpus(n)
        if mpc and cpus and mpc * cpus < mb:
            mb, why = mpc * cpus, f"partition {name}'s MaxMemPerCPU ({int(mpc)}M) x {int(cpus)} CPUs"
    return mb, why


def node_real_mb(n: dict) -> float | None:
    """A node's RealMemory (MB); None when SLURM does not know it (unset nodes report 1)."""
    try:
        v = float(n.get("RealMemory") or 0)
    except ValueError:
        return None
    return v if v > 1 else None


def addresses() -> set[str]:
    """This machine's addresses: `hostname -I` (Linux; the Sparks' Tailscale addresses among them), and the host
    name's own."""
    out = set()
    p = run(["hostname", "-I"], timeout=5)
    if p is not None and p.returncode == 0:
        out.update(p.stdout.split())
    try:
        out.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return out


def this_node(nodes: list[dict] | None) -> dict | None:
    """The cluster node that is this machine: NodeName or NodeHostName equal to the host name (or its short form),
    else NodeAddr among this machine's addresses; None if none matches."""
    if not nodes:
        return None
    host = socket.gethostname()
    names = {host, host.split(".")[0]}
    for n in nodes:
        if n.get("NodeName") in names or n.get("NodeHostName") in names \
                or (n.get("NodeHostName") or "").split(".")[0] in names:
            return n
    addrs = addresses()
    for n in nodes:
        if n.get("NodeAddr") in addrs:
            return n
    return None


def expand_nodelist(s: str | None) -> list[str]:
    """A SLURM node list as names: 'spark-5da5,node[01-03,7]' -> ['spark-5da5', 'node01', 'node02', 'node03',
    'node7']."""
    if not s:
        return []
    out, depth, cur = [], 0, ""
    for ch in s:                       # split on commas outside brackets
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    names: list[str] = []
    for item in (x.strip() for x in out if x.strip()):
        names += _expand_one(item)
    return names


def _expand_one(item: str) -> list[str]:
    """One host expression, every bracket expanded ('rack[1-2]-n[01-02]': four names)."""
    m = re.fullmatch(r"([^\[]*)\[([^\]]+)\](.*)", item)
    if not m:
        return [item]
    pre, body, post = m.groups()
    out = []
    for part in body.split(","):
        vals = [part]
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                vals = [str(i).zfill(len(a)) for i in range(int(a), int(b) + 1)]
            except ValueError:
                pass
        for v in vals:
            out += _expand_one(f"{pre}{v}{post}")
    return out


def node_setting(rt) -> str:
    return rt.env("SLURM_NODE", "").strip() or "local"


def node_pin(rt) -> tuple[str | None, str | None, dict]:
    """(--nodelist this translator adds to a job or None, the job's effective nodelist or None when SLURM may choose,
    details: cluster_nodes, this_node, warning)."""
    nodes = cluster_nodes()
    me = this_node(nodes)
    info = {"cluster_nodes": len(nodes) if nodes else None, "this_node": me.get("NodeName") if me else None}
    given = args_nodelist(rt)
    if given:
        return None, given, info
    setting = node_setting(rt)
    if setting.lower() == "any":
        return None, None, info
    if setting.lower() != "local":
        return setting, setting, info
    if not nodes or len(nodes) < 2:
        return None, None, info
    if me is None:
        info["warning"] = (f"{rt.var('SLURM_NODE')}=local, but this machine is none of the cluster's nodes (by name "
                           f"or address): jobs may run on another node; set {rt.var('SLURM_NODE')} to this node's name")
        return None, None, info
    return me["NodeName"], me["NodeName"], info


def node_status(rt) -> dict:
    """tool_status().executor.node: {setting, nodelist (as jobs ask: null when SLURM may choose; the router's sbatch
    --test-only uses the same), cluster_nodes, this_node, warning?; from_args when P_SLURM_ARGS sets the nodelist}."""
    _, nodelist, info = node_pin(rt)
    out = {"setting": node_setting(rt), "nodelist": nodelist, **info}
    if args_nodelist(rt):
        out["from_args"] = True
    return out


def args_node_filter(rt) -> tuple[str | None, list[str]]:
    """(partition, features every node must have) from P_SLURM_ARGS: --partition/-p, and a --constraint/-C of plain
    features joined by '&' (or one); a constraint with '|', '[' or counts filters nothing (SLURM decides)."""
    a = slurm_args(rt)
    part = opt_value(a, ("--partition",), ("-p",))
    con = opt_value(a, ("--constraint",), ("-C",)) or ""
    feats = [] if not con or re.search(r"[|\[\]*()]", con) else [f.strip() for f in con.split("&") if f.strip()]
    return (part.split(",")[0] if part and "," not in part else None), feats


def job_nodes(rt) -> list[dict]:
    """The nodes a job of this translator may run on: those of its nodelist, else every node ([] without SLURM); only
    those of P_SLURM_ARGS' partition and constraint features when SLURM lists them (Perlmutter: '-C cpu' keeps the
    128-core CPU nodes, not the 64-core GPU ones), unless that leaves none."""
    nodes = cluster_nodes() or []
    part, feats = args_node_filter(rt)
    if part or feats:
        keep = [n for n in nodes
                if (not part or "Partitions" not in n or part in (n.get("Partitions") or "").split(","))
                and (not feats or "AvailableFeatures" not in n
                     or set(feats) <= set((n.get("AvailableFeatures") or "").split(",")))]
        nodes = keep or nodes
    nl = node_pin(rt)[1]
    if nl:
        want = set(expand_nodelist(nl))
        pinned = [n for n in nodes if n.get("NodeName") in want]
        if pinned:
            return pinned
    return nodes


def node_cores(rt) -> int:
    """Cores a job can use: the smallest node of a pinned nodelist, else this node (or the only one), else the
    smallest of the nodes jobs may go to (job_nodes: P_SLURM_ARGS' partition and constraint); without SLURM, this
    machine's CPUs. Caps use these, never the cluster's total."""
    nodes = cluster_nodes()
    if nodes:
        if node_pin(rt)[1]:
            return min(node_core_count(n) for n in job_nodes(rt))
        me = this_node(nodes) or (nodes[0] if len(nodes) == 1 else None)
        if me:
            return node_core_count(me)
        return min(node_core_count(n) for n in (job_nodes(rt) or nodes))
    return max(1, os.cpu_count() or 1)


def node_mem_limit(rt) -> tuple[float | None, str]:
    """(MB, why) of the most memory a job can ask for per node: the largest node_job_mb among the nodes it may go to
    (RealMemory less MemSpecLimit, within the job's partition's MaxMemPerNode / MaxMemPerCPU x CPUs); (None, '')
    without SLURM or when SLURM does not know it."""
    if kind(rt) != "slurm":
        return None, ""
    part = job_partition(rt) if job_nodes(rt) else None
    vals = [v for v in (node_job_mb(n, part) for n in job_nodes(rt)) if v[0]]
    return max(vals, key=lambda v: v[0]) if vals else (None, "")


def node_mem_mb(rt) -> float | None:
    """The most memory one job can ask for per node (MB): node_mem_limit's (before 2026.11.12.1 the largest RealMemory,
    which Perlmutter refused); None without SLURM or when SLURM does not know it."""
    return node_mem_limit(rt)[0]


def max_nodes(rt) -> int:
    """The most nodes one job may use: P_SLURM_NODES (default 1); always 1 for threaded translators and rt.one_node;
    at most the nodes jobs may go to (a pinned nodelist's, else the cluster's, when SLURM lists them)."""
    if rt.kind != "mpi" or rt.one_node:
        return 1
    try:
        n = max(1, int(float(rt.env("SLURM_NODES", "1") or 1)))
    except ValueError:
        return 1
    if n > 1:
        avail = len(job_nodes(rt))
        if avail:
            n = min(n, avail)
    return n


def max_job_procs(rt) -> int:
    """The most ranks or threads one job can have: a node's cores (node_cores) x max_nodes."""
    return node_cores(rt) * max_nodes(rt)


def job_layout(rt, procs: int) -> tuple[int, int | None]:
    """(nodes, ranks per node or None) of a job of `procs` ranks: (1, None) while they fit on a node's cores or the
    job may use one node; else ceil(procs / cores) nodes (at most max_nodes) with the ranks balanced over them
    (512 ranks on 128-core nodes: 4 x 128; 300: 3 x 100)."""
    mx = max_nodes(rt)
    if mx <= 1:
        return 1, None
    cores = node_cores(rt)
    procs = max(1, int(procs))
    if procs <= cores:
        return 1, None
    n = min(mx, int(math.ceil(procs / cores)))
    return n, int(math.ceil(procs / n))


def job_setup_lines(rt) -> list[str]:
    """P_JOB_SETUP as job.sh lines (newlines or a literal '\\n' separate them; blank ones dropped)."""
    v = rt.env("JOB_SETUP", "") or ""
    return [ln.strip() for ln in v.replace("\\n", "\n").splitlines() if ln.strip()]


def caps(rt) -> dict:
    """Ceilings on what a study may ask: P_MAX_TIMEOUT (s), P_MAX_MEM, P_MAX_PROCS, and with SLURM the memory of the
    node jobs go to when it is lower or no P_MAX_MEM is set; mem_mb (MB) and mem_from say how much and why (the
    router leaves out machines below a study's run.mem)."""
    out: dict = {}
    for key, name in (("timeout_s", "MAX_TIMEOUT"), ("mem", "MAX_MEM"), ("procs", "MAX_PROCS")):
        v = rt.env(name, "").strip()
        if not v:
            continue
        try:
            out[key] = v if key == "mem" else (float(v) if key == "timeout_s" else int(float(v)))
        except ValueError:
            pass
    if out.get("mem"):
        out["mem_from"] = rt.var("MAX_MEM")
    nm, why = node_mem_limit(rt)
    if nm and (not out.get("mem") or (mem_mb(out["mem"]) or 0) > nm):
        out["mem"] = mem_text(nm)
        out["mem_from"] = why
    if out.get("mem") and mem_mb(out["mem"]):
        out["mem_mb"] = int(mem_mb(out["mem"]))
    nn = max_nodes(rt) if kind(rt) == "slurm" else 1
    if nn > 1:                  # jobs may span nodes: the ranks one job can have (one node: as before, no cap here)
        nc = node_cores(rt)
        if not out.get("procs") or out["procs"] > nc * nn:
            out["procs"] = nc * nn
            out["procs_from"] = (f"a node's cores x {rt.var('SLURM_NODES')} ({nc} x {nn} nodes; SLURM, one rank per "
                                 "core)")
        out["nodes"] = nn
        if out.get("mem"):
            out["mem_per"] = "node"
    return out


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_errors(rt, study: dict) -> list[str]:
    """A study's run settings against the caps and their types: one message per bad key (validation shows them)."""
    r, c, errs = run_of(study), caps(rt), []
    if r.get("timeout_s") is not None:
        t = _num(r["timeout_s"])
        if t is None or t <= 0:
            errs.append(f"run.timeout_s {r['timeout_s']!r}: seconds, a number > 0")
        elif c.get("timeout_s") and t > c["timeout_s"]:
            errs.append(f"run.timeout_s {r['timeout_s']} is above {rt.var('MAX_TIMEOUT')} ({c['timeout_s']:g} s)")
    if r.get("mem"):
        mb = mem_mb(r["mem"])
        if mb is None:
            errs.append(f"run.mem {r['mem']!r}: a size such as '8G' or '500M'")
        elif c.get("mem_mb") and mb > c["mem_mb"]:
            errs.append(f"run.mem {r['mem']} is above {c['mem']} ({c.get('mem_from')})")
    if r.get("procs") is not None:
        p = _num(r["procs"])
        if p is None or p < 1 or p != int(p):
            errs.append(f"run.procs {r['procs']!r}: a whole number >= 1")
        elif c.get("procs") and p > c["procs"]:
            errs.append(f"run.procs {r['procs']} is above {rt.var('MAX_PROCS')} ({c['procs']})")
    return errs


# ----------------------------------------------------------------------------- memory of a job

def mem_plan(rt, study: dict | None, nodes: int | None = None) -> dict:
    """{mem: the job's --mem or None (do not ask), from: where it comes from, capped_from?, total?}: run.mem; else a
    --mem in P_SLURM_ARGS (None here: sbatch gets it from the arguments); else P_SLURM_MEM none | a size | auto (the
    translator's estimate, hooks.mem_auto, else rt.mem_default, never above the node's memory). --mem is per node in
    SLURM: run.mem and a size are taken as per node; auto's total is split over the job's nodes (nodes: given, else
    job_layout's when jobs may span nodes), total says the whole."""
    r = run_of(study).get("mem")
    if r:
        return {"mem": mem_arg(r) or str(r).strip(), "from": "run.mem"}
    v = rt.env("SLURM_MEM", "auto").strip() or "auto"
    if args_mem(rt) is not None:
        return {"mem": None, "from": f"{rt.var('SLURM_ARGS')} {args_mem(rt)}"}
    if v.lower() == "none":
        return {"mem": None, "from": f"{rt.var('SLURM_MEM')}=none (SLURM's default)"}
    if v.lower() != "auto":
        return {"mem": mem_arg(v) or v, "from": rt.var("SLURM_MEM")}
    auto = rt.hooks.mem_auto(study or {}) if rt.hooks.mem_auto else None
    out = {"mem": mem_arg(auto or rt.mem_default) or (auto or rt.mem_default),
           "from": "auto" if auto else f"auto ({rt.mem_default}: size unknown)"}
    if nodes is None:
        nodes = job_layout(rt, job_procs(rt, study or {}))[0] if max_nodes(rt) > 1 else 1
    total = mem_mb(out["mem"])
    if nodes > 1 and total:
        out["total"] = out["mem"]
        out["mem"] = mem_text(int(math.ceil(total / nodes)))
        out["from"] += f", {out['total']} over {nodes} nodes"
    nm, why = node_mem_limit(rt)
    if nm and (mem_mb(out["mem"]) or 0) > nm:
        out["capped_from"], out["mem"], out["capped_by"] = out["mem"], mem_text(nm), why
    return out


def mem_for(rt, study: dict | None) -> str | None:
    """The --mem value of a study's job; None: do not ask (also when P_SLURM_ARGS has its own --mem)."""
    return mem_plan(rt, study)["mem"]


def given_mem_mb(rt, study: dict | None) -> tuple[float | None, str]:
    """(MB, where from) of the memory a study's job gets when that is fixed: run.mem, a --mem in P_SLURM_ARGS, a size
    in P_SLURM_MEM; with auto, what the machine allows (P_MAX_MEM, the node's memory); (None, '') when nothing limits
    it. (A --mem-per-cpu grows with the CPUs: not a fixed amount.)"""
    r = run_of(study).get("mem")
    if r:
        mb = mem_mb(str(r).strip())
        return mb, f"run.mem {str(r).strip()}"
    a = opt_value(slurm_args(rt), ("--mem",))
    if a:
        mb = mem_mb(a)
        if mb:
            return mb, f"{rt.var('SLURM_ARGS')} --mem={a}"
    v = rt.env("SLURM_MEM", "auto").strip()
    if v and v.lower() not in ("auto", "none"):
        mb = mem_mb(v)
        if mb:
            return mb, f"{rt.var('SLURM_MEM')}={v}"
    c = caps(rt)
    if mem_mb(c.get("mem", "")):
        return mem_mb(c["mem"]), f"{c.get('mem_from', rt.var('MAX_MEM'))} {c['mem']}"
    return None, ""


# ----------------------------------------------------------------------------- time limits

def time_need_min(rt, study: dict | None = None) -> int:
    """Minutes the study's runs may take, with the 10 minutes that let the translator's own timeout fire before
    SLURM's: hooks.time_need_min, else ceil(solves x timeout_s / 60) + 10."""
    if rt.hooks.time_need_min:
        return int(rt.hooks.time_need_min(study or {}))
    n = rt.hooks.solves(study or {}) if rt.hooks.solves else 1
    return int(math.ceil(n * timeout_s(rt, study) / 60)) + 10


def partition_max_min(rt) -> float | None:
    """The time limit (minutes; inf: none) of the partition this translator's jobs go to (P_SLURM_ARGS --partition,
    else the default one), from sinfo; None when SLURM does not say. Kept 5 minutes."""
    want = opt_value(slurm_args(rt), ("--partition",), ("-p",))
    if _PART["key"] == want and time.time() - _PART["t"] < 300 and _PART["v"] is not None:
        return _PART["v"]
    p = run(["sinfo", "-h", "-o", "%P|%l"], timeout=10)
    if p is None or p.returncode != 0:
        return None
    rows = [ln.split("|", 1) for ln in p.stdout.splitlines() if "|" in ln]
    pick = None
    for name, lim in rows:
        if (want and name.rstrip("*") == want) or (not want and name.endswith("*")):
            pick = lim
            break
    if pick is None and rows and not want:
        pick = rows[0][1]
    v = slurm_minutes(pick)
    if v is not None:
        _PART.update(t=time.time(), key=want, v=v)
    return v


TIME_SETTINGS = ("max", "timeout", "estimate")


def time_setting(rt) -> str:
    """P_SLURM_TIME, through rt.time_aliases (a translator moving to the common meanings can map its old words for a
    while: feat's former 'estimate' is 'timeout')."""
    v = rt.env("SLURM_TIME", rt.slurm_time_default).strip().lower() or rt.slurm_time_default
    return rt.time_aliases.get(v, v)


def job_time(rt, study: dict | None = None, estimate_min: float | None = None) -> tuple[str, dict]:
    """(the job's --time, info: setting, needed_min, time_from, and for 'estimate' the study's own timeout_s).
    P_SLURM_TIME: max | timeout | estimate | minutes (module docstring). The translator's own timeout still ends the
    solver and writes a clean result; SLURM's limit is the backstop."""
    need = time_need_min(rt, study)
    setting = time_setting(rt)
    info: dict = {"needed_min": need, "setting": setting}
    if setting == "timeout":
        info["time_from"] = "timeout"
        return str(need), info
    if setting == "estimate":
        if run_of(study).get("timeout_s"):
            info["time_from"] = "study"
            return str(need), info
        if estimate_min is None:
            info.update(time_from="timeout", note="no estimate for this study: the timeouts")
            return str(need), info
        f = _float(rt, "SLURM_TIME_FACTOR", 3.0)
        floor = _float(rt, "SLURM_TIME_MIN", 20.0)
        t = int(min(need, max(floor, math.ceil(f * float(estimate_min)) + 10)))
        info.update(time_from="estimate", estimate_min=round(float(estimate_min), 2),
                    timeout_s=timeout_for_limit(rt, study, t))
        return str(t), info
    if setting != "max":
        try:
            info["time_from"] = "minutes"
            return str(max(need, int(float(setting)))), info
        except ValueError:
            info.update(time_from="timeout", warning=f"{rt.var('SLURM_TIME')}={setting!r} is not max, timeout, "
                                                     "estimate or minutes: the timeouts")
            return str(need), info
    mx = partition_max_min(rt)
    info["time_from"] = "max"
    if mx is None:
        info.update(time_from="timeout", note="the partition's limit is unknown (sinfo): the timeouts")
        return str(need), info
    if math.isinf(mx):
        info["partition_max"] = "infinite"
        return "UNLIMITED", info
    info["partition_max_min"] = int(mx)
    if mx < need:
        info["warning"] = (f"the partition's limit ({int(mx)} min) is below what the runs may take ({need} min): "
                           f"SLURM may stop the job before {rt.cli}'s own timeout")
    return str(int(mx)), info


def timeout_for_limit(rt, study: dict | None, minutes: int) -> float:
    """Seconds each of a study's runs may take when its job has `minutes` (P_SLURM_TIME=estimate):
    hooks.timeout_for_limit, else (minutes - 10) x 60 / solves, so every run's own timeout fires before SLURM's."""
    if rt.hooks.timeout_for_limit:
        return float(rt.hooks.timeout_for_limit(study or {}, minutes))
    n = max(1, int(rt.hooks.solves(study or {}) if rt.hooks.solves else 1))
    return max(60.0, (minutes - 10) * 60.0 / n)


def job_time_left_min(job_id: str) -> float | None:
    """Minutes a running job has left before SLURM's limit (squeue %L); inf without a limit; None when unknown."""
    p = run(["squeue", "-h", "-j", str(job_id), "-o", "%L"], timeout=10)
    if p is None or p.returncode != 0 or not p.stdout.strip():
        return None
    return slurm_minutes(p.stdout.strip().splitlines()[0])


def extend_job(job_id: str, minutes: int) -> str | None:
    """Raises a SLURM job's time limit by `minutes` (scontrol update TimeLimit=+N); None, or why not (SLURM lets only
    an operator or admin raise a running job's limit)."""
    p = run(["scontrol", "update", f"JobId={job_id}", f"TimeLimit=+{int(minutes)}"])
    if p is None:
        return "scontrol not found or did not answer"
    return None if p.returncode == 0 else ((p.stderr or p.stdout).strip()[:300] or f"exit {p.returncode}")


# ----------------------------------------------------------------------------- tool_status().executor

def status_key(rt) -> dict:
    """The 'executor' key of tool_status (llmt status and the router read it): the same keys in local and SLURM mode.
    'time' (the contract's name) and 'slurm_time' (feat's) both give P_SLURM_TIME; 'runtime' is llmt-runtime's
    version."""
    st = time_setting(rt)
    try:
        need = time_need_min(rt)
    except Exception:  # noqa: BLE001 (a hook that needs a real study: none for the defaults)
        need = int(math.ceil(timeout_s(rt) / 60)) + 10
    return {"kind": kind(rt), "sbatch": shutil.which("sbatch"), "min_minutes": min_minutes(rt),
            "procs": default_procs(rt), "timeout_s": timeout_s(rt), "time_limit_min": need,
            "time": st, "slurm_time": st, "mem": rt.env("SLURM_MEM", "auto") or "auto", "caps": caps(rt),
            "node": node_status(rt), "node_cores": node_cores(rt), "max_nodes": max_nodes(rt),
            "runtime": rt.runtime_version}

"""memory.py — the memory a run may use, and whose memory ran out when a solver was killed.
Version: 2026.11.09.1

    memory_limit_mb()          SLURM_MEM_PER_NODE, else the cgroup's memory.max (v2) / memory.limit_in_bytes (v1)
    oom_kills(), oom_limit_hits()  cgroup counters: kills by the OOM killer; times the cgroup reached its limit
    meminfo_mb(), NodeSampler  the node's total and available memory, sampled while a solver runs
    kernel_oom_constraint(pid, since)  the kernel's own word (journalctl -k), when this user may read it
    oom_scope(...)             'job' (its cgroup's limit: more memory or fewer threads helps), 'node' (the node ran out:
                               other processes; more memory does not help), None (not killed for memory)
    OomWatch                   the above around one solver run: start, sample() while waiting, classify(code, rss)
    threads_for_memory(...)    threads cut to 1 when the asked ones would not fit the job (the need from the translator)
    killed_text, node_oom_text, oom_message, exit_text   the wording

From fea-translator runner.py 2026.11.05.3 - 2026.11.08.1 (carbonio, November 2026): job 5797 (6 threads, 32 GB,
killed at 33.4 GB) looked like a crash; job 5807 was killed as a whole batch step (the threads are now checked again on
the real size before the solver starts); two runs killed at 20.1 and 13.9 GB in 34 GB jobs were the node's, not the
job's. What a translator does after a kill (feat restarts a threaded CalculiX run on one thread) stays in the translator.

Lesson kept: do NOT raise the solver's oom_score_adj. feat 2026.11.07.1-.2 did so that feat would survive the job's
limit; it also made CalculiX the node's first victim when other processes ran the node out of memory. The solver is by
far the job's largest process, so the job's own limit picks it anyway.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

# where the kernel's files are (tests point these elsewhere)
PROC_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_MEMINFO = Path("/proc/meminfo")
PROC_STATUS = Path("/proc/self/status")


def cgroup_dirs() -> list[Path]:
    """This process's memory cgroups, its own first and then each parent (cgroup v2, or v1's memory hierarchy); []
    where there are none (macOS). A SLURM job's solver runs in the same cgroup as the translator."""
    try:
        lines = PROC_CGROUP.read_text().splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) != 3:
            continue
        hid, ctrl, rel = parts
        if hid == "0" and ctrl == "":
            root = CGROUP_ROOT
        elif "memory" in ctrl.split(","):
            root = CGROUP_ROOT / "memory"
        else:
            continue
        d = root / rel.strip().lstrip("/")
        while True:
            out.append(d)
            if d == root or d.parent == d:
                break
            d = d.parent
    return out


def oom_kills() -> dict:
    """{cgroup file: processes killed by the out-of-memory killer so far} (v2 memory.events, v1 memory.oom_control)."""
    out = {}
    for d in cgroup_dirs():
        for f in ("memory.events", "memory.oom_control"):
            try:
                for ln in (d / f).read_text().splitlines():
                    k, _, v = ln.partition(" ")
                    if k == "oom_kill":
                        out[str(d / f)] = int(v)
            except (OSError, ValueError):
                pass
    return out


def oom_limit_hits() -> dict:
    """{cgroup: times its memory reached its limit (v2 memory.events 'oom')}: a kill without one of these is the
    node's out-of-memory killer, not the job's limit ({} on cgroup v1, which has no such count)."""
    out = {}
    for d in cgroup_dirs():
        try:
            for ln in (d / "memory.events").read_text().splitlines():
                k, _, v = ln.partition(" ")
                if k == "oom":
                    out[str(d)] = int(v)
        except (OSError, ValueError):
            pass
    return out


def memory_limit_mb() -> float | None:
    """The memory this process (and its solver) may use: the SLURM job's (SLURM_MEM_PER_NODE, MB), else its cgroup's
    memory.max (v2) or memory.limit_in_bytes (v1); None: no limit known."""
    v = os.environ.get("SLURM_MEM_PER_NODE")
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    for d in cgroup_dirs():
        for f in ("memory.max", "memory.limit_in_bytes"):
            try:
                t = (d / f).read_text().strip()
            except OSError:
                continue
            if t.isdigit() and int(t) < 2 ** 60:
                return int(t) / 2 ** 20
    return None


def meminfo_mb() -> tuple[float, float] | None:
    """(total, available) memory of this node in MB (/proc/meminfo); None elsewhere."""
    try:
        vals = {}
        for ln in PROC_MEMINFO.read_text().splitlines():
            k, _, v = ln.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                vals[k] = int(v.split()[0]) / 1024
        return vals["MemTotal"], vals["MemAvailable"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def own_rss_mb() -> float:
    """This process's resident memory (MB): it shares the job's memory with the solver. 0 where unknown."""
    try:
        for ln in PROC_STATUS.read_text().splitlines():
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


class NodeSampler:
    """The node's available memory, sampled while a solver runs (the last `keep` samples)."""

    def __init__(self, keep: int = 200):
        self.total_mb: float | None = None
        self.samples: list[tuple[float, float]] = []
        self.keep = keep

    def sample(self) -> None:
        m = meminfo_mb()
        if m:
            self.total_mb = m[0]
            self.samples.append((time.time(), m[1]))
            del self.samples[:-self.keep]

    def min_free_mb(self, seconds: float = 15.0) -> float | None:
        """The node's least available memory in the last `seconds` of samples (MB); None without samples."""
        if not self.samples:
            return None
        t_end = self.samples[-1][0]
        return min(a for t, a in self.samples if t >= t_end - seconds)


def kernel_oom_constraint(pid: int, since: float) -> str | None:
    """The kernel's word on why it killed `pid`: 'CONSTRAINT_MEMCG' (a cgroup's limit: the job's) or
    'CONSTRAINT_NONE' (the node ran out), from its log (journalctl -k), when this user may read it; else None."""
    if not shutil.which("journalctl"):
        return None
    try:
        r = subprocess.run(["journalctl", "-k", "-q", "-o", "cat", "--no-pager", "--since", f"@{int(since) - 5}"],
                           capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for ln in r.stdout.splitlines():
        m = re.search(r"oom-kill:constraint=(CONSTRAINT_\w+).*\bpid=(\d+)", ln)
        if m and int(m.group(2)) == pid:
            return m.group(1)
    return None


def _grew(a: dict, b: dict) -> bool:
    return any(b.get(k, 0) > n for k, n in a.items()) or any(k not in a and n > 0 for k, n in b.items())


def oom_scope(kills0: dict, kills1: dict, hits0: dict, hits1: dict, code: int, rss_mb: float, limit_mb: float | None,
              node_free_mb: float | None, node_total_mb: float | None, kernel: str | None = None) -> str | None:
    """Whose memory ran out when a solver ended with `code`: 'job', 'node' or None (not killed for memory). From the
    kernel's log when readable; else the cgroup counters (a kill with no 'limit reached' count is the node's); else the
    sizes (SIGKILL near the job's limit: the job; with the node nearly out: the node)."""
    if code == 0:
        return None
    if kernel:
        return "job" if kernel == "CONSTRAINT_MEMCG" else "node"
    node_low = bool(node_free_mb is not None and node_total_mb and node_free_mb < max(0.05 * node_total_mb, 1024))
    near_limit = bool(limit_mb and rss_mb >= 0.8 * limit_mb)
    if _grew(kills0, kills1):
        if hits1:
            return "job" if _grew(hits0, hits1) else "node"
        return "node" if (node_low and not near_limit) else "job"
    if code == -9:
        if limit_mb and rss_mb >= 0.9 * limit_mb:
            return "job"
        if node_low:
            return "node"
    return None


def out_of_memory(code: int, before: dict, after: dict, rss_mb: float, limit_mb: float | None) -> bool:
    """Was a run that ended with `code` killed for memory? Its cgroup counted an OOM kill meanwhile, or (no counters)
    SIGKILL with its peak resident memory at 90 % of the limit or more."""
    if code == 0:
        return False
    if _grew(before, after):
        return True
    return code == -9 and bool(limit_mb) and rss_mb >= 0.9 * limit_mb


def node_jobs_text(total_mb: float | None = None) -> str:
    """'; running on <node> now: <jobs with the memory each asked for>', with a note when they add up to more than the
    node has (SLURM then does not count memory when it places jobs); '' outside SLURM."""
    node = os.environ.get("SLURMD_NODENAME")
    if not node or not shutil.which("squeue"):
        return ""
    try:
        r = subprocess.run(["squeue", "-h", "-w", node, "-t", "R", "-o", "%i|%m|%j"], capture_output=True, text=True,
                           timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    from .slurm import mem_mb
    jobs, total = [], 0.0
    for ln in r.stdout.splitlines():
        f = ln.split("|")
        if len(f) >= 3:
            total += mem_mb(f[1]) or 0
            jobs.append(f"{f[0]} ({f[2][:40]}, {f[1]})")
    if not jobs:
        return ""
    out = f"; running on {node} now: " + ", ".join(jobs[:8])
    if total_mb and total > total_mb:
        out += (f" — together {total / 1024:.0f} GB asked on a node with {total_mb / 1024:.0f} GB: SLURM is not "
                "counting memory when it places jobs here")
    return out


# ---- the wording

def exit_text(code: int) -> str:
    """'exit code 3', or the signal that ended the process ('signal 11 (SIGSEGV)')."""
    if code < 0:
        try:
            return f"signal {-code} ({signal.Signals(-code).name})"
        except ValueError:
            return f"signal {-code}"
    return f"exit code {code}"


def procs_text(n: int, unit: str = "thread") -> str:
    return f"{n} {unit}{'s' if n != 1 else ''}"


def killed_text(rec: dict, unit: str = "thread") -> str:
    """'killed at 33.4 GB of 32.0 GB with 6 threads' (rec: peak_rss_mb, limit_mb, threads)."""
    of = f" of {rec['limit_mb'] / 1024:.1f} GB" if rec.get("limit_mb") else ""
    w = f" with {procs_text(int(rec['threads']), unit)}" if rec.get("threads") else ""
    return f"killed at {(rec.get('peak_rss_mb') or 0) / 1024:.1f} GB{of}{w}"


def node_oom_text(rec: dict) -> str:
    """'the node ran out of memory (other processes): killed at 13.9 GB, below the job's 34.0 GB; the node's free
    memory fell to 0.3 GB of 62.6 GB' (+ node_jobs)."""
    lim = f", below the job's {rec['limit_mb'] / 1024:.1f} GB" if rec.get("limit_mb") else ""
    free = (f"; the node's free memory fell to {rec['node_min_free_mb'] / 1024:.1f} GB of "
            f"{rec['node_total_mb'] / 1024:.1f} GB"
            if rec.get("node_min_free_mb") is not None and rec.get("node_total_mb") else "")
    return (f"the node ran out of memory (other processes): killed at {(rec.get('peak_rss_mb') or 0) / 1024:.1f} GB"
            f"{lim}{free}{rec.get('node_jobs', '')}")


def oom_message(rec: dict, solver: str = "the solver", smaller: str = "make the study smaller",
                unit: str = "thread") -> str:
    """The error for a run killed for memory (rec from OomWatch.classify). The node's case never advises more
    memory: it is not the job's limit that was reached."""
    t = procs_text(int(rec.get("threads") or 1), unit)
    if rec.get("oom_scope") == "node":
        return (f"{solver} was killed with {t} because {node_oom_text(rec)}. Run it when the node is free (jobs(), "
                "machines()); more run.mem will not help")
    lim = f" (the job's limit is {rec['limit_mb'] / 1024:.1f} GB)" if rec.get("limit_mb") else ""
    return (f"{solver} ran out of memory with {t}: killed at {(rec.get('peak_rss_mb') or 0) / 1024:.1f} GB{lim}. "
            f"Ask for more memory (run.mem) or {smaller}")


class OomWatch:
    """One solver run: the cgroup counters and the node's memory before, samples while it runs, and the verdict.

        w = OomWatch()            # just before starting the solver
        ... w.sample() in the wait loop ...
        rec = w.classify(code, peak_rss_mb, pid, threads)   # {out_of_memory, oom_scope, peak_rss_mb, limit_mb, ...}
    """

    def __init__(self):
        self.kills0, self.hits0 = oom_kills(), oom_limit_hits()
        self.node = NodeSampler()
        self.node.sample()
        self.t0 = time.time()

    def sample(self) -> None:
        self.node.sample()

    def classify(self, code: int, rss_mb: float, pid: int | None = None, threads: int | None = None) -> dict:
        limit = memory_limit_mb()
        scope = None
        if code != 0:
            kernel = kernel_oom_constraint(pid, self.t0) if (code == -9 and pid) else None
            scope = oom_scope(self.kills0, oom_kills(), self.hits0, oom_limit_hits(), code, rss_mb, limit,
                              self.node.min_free_mb(), self.node.total_mb, kernel)
        rec = {"out_of_memory": scope is not None, "oom_scope": scope, "peak_rss_mb": round(rss_mb, 1),
               "limit_mb": None if limit is None else round(limit, 1), "run_s": round(time.time() - self.t0, 1)}
        if threads is not None:
            rec["threads"] = threads
        if scope == "node":
            low = self.node.min_free_mb()
            rec.update(node_total_mb=None if self.node.total_mb is None else round(self.node.total_mb, 1),
                       node_min_free_mb=None if low is None else round(low, 1),
                       node_jobs=node_jobs_text(self.node.total_mb))
        return rec


def threads_for_memory(need_gb: Callable[[int], float], threads: int, limit_mb: float | None = None,
                       what: str = "this run", solver: str = "the solver",
                       ask_gb: Callable[[float], int] | None = None) -> tuple[int, dict | None]:
    """(threads, note) for a run whose memory need with t threads is need_gb(t) (GB, the translator's model): 1 thread
    when the threads asked for would not fit the job's memory (limit_mb, default memory_limit_mb()) less this
    process's own; a note when even 1 may not fit. The size estimate before a run can be far off (carbonio job 5807:
    107,171 nodes estimated, 424,303 meshed; 6 threads in 30 GB were killed after 7 min); check on the real size."""
    if limit_mb is None:
        limit_mb = memory_limit_mb()
    if not limit_mb or threads < 1:
        return threads, None
    ask = ask_gb or (lambda gb: int(-(-(gb + 1) // 1)))
    avail = limit_mb / 1024 - own_rss_mb() / 1024 - 0.5
    need = {t: float(need_gb(t)) for t in sorted({threads, 1})}
    note = {"threads_asked": threads, "limit_gb": round(limit_mb / 1024, 1), "available_gb": round(avail, 1),
            "estimate_gb": {str(k): round(v, 1) for k, v in need.items()}}
    use = threads
    if threads > 1 and need[threads] > avail:
        use = 1
        note["message"] = (f"{what} need about {need[threads]:.0f} GB with {threads} threads and {need[1]:.0f} GB with "
                           f"1; the job has {limit_mb / 1024:.0f} GB: {solver} runs on 1 thread")
    if need[1] > avail and use == 1:
        note["message"] = (note.get("message", f"{what} need about {need[1]:.0f} GB with 1 thread; the job has "
                                               f"{limit_mb / 1024:.0f} GB")
                           + f"; it may run out of memory: ask for {ask(need[1])}G (run.mem)")
    note["threads"] = use
    return use, (note if "message" in note else None)

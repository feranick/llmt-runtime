"""workers.py — worker processes end with the server that started them, and take their solver runs with them.
Version: 2026.11.09.1

    watch_parent(ppid, name, term)  initializer of every worker pool and worker process: a daemon thread ends the worker
                                    (and the solver runs it started) when its parent is gone; term: SIGTERM does the same
    track(p) / untrack(p)           solver child processes started in their own process group: killed with the worker
    on_orphan(fn)                   another cleanup to run first (a translator that keeps its own list of runs)
    end_children()                  kills every tracked process group now
    on_signals()                    a run-study process: SIGTERM/SIGHUP/SIGINT end the tracked runs, then SystemExit
    stop_pool(pool), stop_executor(ex, terminate), stop_process(p), shutdown(*stoppers), run_bounded(fn, s)
                                    what a server calls before os._exit (which skips atexit); SIGTERM, then SIGKILL
                                    after a grace period, so a stop never hangs

From fea-translator workers.py 2026.11.03.1 (carbonio, 2026.11.02: 7 orphaned spawn workers and a resource_tracker
from three releases kept them 'in use' for llmt gc). A spawn worker holds both ends of its task queue, so it never sees
EOF when the server dies and would wait forever.

Not PR_SET_PDEATHSIG: on Linux it fires when the *thread* that created the child exits, and the pools are created
from the MCP server's worker threads, which come and go; it would kill healthy workers in mid-study. Every platform
polls os.getppid() instead (PARENT_POLL_S): when the parent dies the worker is re-parented (to init, launchd or a
subreaper) and its parent pid changes.

New here: SIGTERM (Pool.terminate(), llmt servers --stop) also ends the tracked runs. Before, a terminated worker died
at once and its solver, in its own process group, ran on (found in openmodelica-translator's review: an omc build has
no time limit of its own).
"""
from __future__ import annotations

import os
import signal
import threading
import time
from typing import Callable

PARENT_POLL_S = 1.0
_CHILDREN: set = set()          # subprocess.Popen objects started in their own process group (the solver)
_HOOKS: list = []               # other cleanups (a translator's own list of runs: cfdt foam.kill_all, lmpt lammps.kill_all)
_STARTED = {"pid": None}        # the process the watcher runs in (a fork would inherit the flag)


def track(p) -> None:
    """A solver process this worker started in its own process group (start_new_session=True)."""
    _CHILDREN.add(p)


def untrack(p) -> None:
    _CHILDREN.discard(p)


def on_orphan(fn: Callable[[], None]) -> None:
    """Runs fn before the tracked groups are killed, when this worker is orphaned or terminated."""
    if fn not in _HOOKS:
        _HOOKS.append(fn)


def end_children() -> None:
    """Runs the on_orphan cleanups, then kills every tracked process group (SIGKILL)."""
    for fn in list(_HOOKS):
        try:
            fn()
        except Exception:  # noqa: BLE001 (kill the rest anyway)
            pass
    for p in list(_CHILDREN):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


EXIT_GRACE_S = 5.0     # a cleanup that hangs (a hook waiting on a lock the interrupted code holds) is cut short


def _exit_after(seconds: float, code: int = 0) -> None:
    def go():
        time.sleep(seconds)
        os._exit(code)
    threading.Thread(target=go, daemon=True, name="llmt-exit-backstop").start()


def _orphaned() -> None:
    _exit_after(EXIT_GRACE_S)
    end_children()
    os._exit(0)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:          # exists, not ours
        return True
    return True


def watch_parent(ppid: int | None = None, name: str = "llmt", term: bool = True) -> None:
    """Starts (once per process) a daemon thread that ends this process, and the solver runs it started, when its
    parent is gone: the process that started it (its parent pid changes when it is re-parented) or, when given, the
    server `ppid` (with the forkserver start method a worker's parent is the fork server, not the server). term:
    SIGTERM ends them too (only from the main thread, where Python allows a handler; a worker's initializer runs there;
    a handler already installed runs after the tracked runs are killed). Python runs a handler between bytecodes: a
    worker inside a long native call takes SIGTERM when it returns, so stop_pool escalates to SIGKILL."""
    if _STARTED["pid"] == os.getpid():
        return
    _STARTED["pid"] = os.getpid()
    parent = os.getppid()
    server = int(ppid) if ppid else None
    if server is not None and not _alive(server):      # the server died before the worker started
        _orphaned()
    if term:
        try:
            prev = signal.getsignal(signal.SIGTERM)

            def on_term(signum, frame):
                _exit_after(EXIT_GRACE_S)
                end_children()
                if callable(prev):
                    prev(signum, frame)
                os._exit(0)
            signal.signal(signal.SIGTERM, on_term)
        except ValueError:        # not the main thread
            pass

    def loop():
        while True:
            time.sleep(PARENT_POLL_S)
            if os.getppid() != parent or (server is not None and not _alive(server)):
                _orphaned()
    threading.Thread(target=loop, daemon=True, name=f"{name}-parent-watch").start()


def on_signals(raise_exit: bool = True) -> None:
    """For a run-study process (a SLURM job, the CLI): SIGTERM, SIGHUP and SIGINT (scancel, a lost terminal) end the
    tracked runs first, then raise SystemExit(128 + signal) so that `finally` blocks run (raise_exit False: exit at
    once). Only from the main thread."""
    def handler(signum, frame):
        end_children()
        if raise_exit:
            raise SystemExit(128 + signum)
        os._exit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except ValueError:
            return


# ---- stopping a server's workers before os._exit

def _end(procs, grace: float) -> None:
    """SIGTERM to each process (their handler ends their tracked runs), then SIGKILL to those still alive after
    `grace` s (a worker inside a long native call does not run Python's handler until it returns)."""
    procs = [p for p in procs if p is not None]
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:  # noqa: BLE001
            pass
    t_end = time.time() + grace
    for p in procs:
        try:
            p.join(max(0.0, t_end - time.time()))
        except Exception:  # noqa: BLE001
            pass
    for p in procs:        # by pid: is_alive() can say True for a process another thread (the pool's) already reaped
        try:
            if p.pid and p.exitcode is None:
                os.kill(p.pid, signal.SIGKILL)
                p.join(2)
        except Exception:  # noqa: BLE001 (gone already)
            pass


def stop_pool(pool, grace: float = 3.0) -> None:
    """A multiprocessing.Pool: its workers are terminated (and killed after `grace` s), then the pool itself, so that
    Pool.terminate() never waits on a worker that does not stop."""
    if pool is None:
        return
    _end(list(getattr(pool, "_pool", None) or []), grace)
    pool.terminate()


def stop_executor(ex, terminate: bool = True, grace: float = 3.0) -> None:
    """A concurrent.futures.ProcessPoolExecutor: shutdown without waiting; terminate: also ends its worker processes
    (shutdown(wait=False) leaves a busy or idle spawn worker running), killed after `grace` s."""
    if ex is None:
        return
    procs = list((getattr(ex, "_processes", None) or {}).values())
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass
    if terminate:
        _end(procs, grace)


def stop_process(p, wait: float = 2.0) -> None:
    """A multiprocessing.Process: terminated, killed after `wait` s."""
    if p is not None:
        _end([p], wait)


def shutdown(*stoppers: Callable[[], None]) -> None:
    """Runs every stopper, each even when another fails (the server calls this before os._exit)."""
    for stop in stoppers:
        try:
            stop()
        except Exception:  # noqa: BLE001 (stop the others anyway)
            pass


def run_bounded(fn: Callable[[], None], seconds: float) -> bool:
    """Runs fn in a daemon thread and waits at most `seconds`; True if it finished (a server's exit must not hang on
    a stopper)."""
    t = threading.Thread(target=fn, daemon=True, name="llmt-bounded")
    t.start()
    t.join(seconds)
    return not t.is_alive()

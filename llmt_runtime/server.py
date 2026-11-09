"""server.py — what every translator's MCP server shares: stale installs, the tool wrapper, the server's log.
Version: 2026.11.09.1

    guard = Server(RT, api.__file__, busy=api.pool_busy, shutdown=api.shutdown_workers)
    @server.tool()
    @guard.tool(ToolError)          # the SDK's ToolError (1.x and 2.x differ): errors come back with their text
    def run_studies(...): ...
    def main(): guard.serve(); server.run()

A stale server (llm-translators docs/SHARED.md) keeps the code it started with after a newer one is installed (shared
install: <prefix>/current points to another release; a checkout: the translator's or llmt-runtime's __init__.py on
disk carries another version). A call that finds it idle is refused with STALE_SERVER (the router repeats it on a new
server) and the server exits; an idle stale server exits by itself within about two periods (P_STALE_CHECK_S,
default rt.stale_period: 10 s; cfdt and omct had 20). Busy servers answer (old code, with a server_stale note) and
exit once idle. Only a server started by serve() refuses or exits (tests and run-study in a release that is not yet
current run normally). P_EXIT_WHEN_STALE=1 always, 0 never; default: shared installs only.

Idle means: no call in flight, the translator's busy() is false, and no background work started with
guard.start_background() (lmpt submits the rest of a request in a thread after the call returns: a stale exit used to
kill it). The exit stops the workers (shutdown) for at most SHUTDOWN_S, then exits whatever happens.

Every tool runs in a worker thread (a blocking call must not stop the server's other requests and pings) under a
deadline of budget_s (slurm.deadline: SLURM commands and submissions keep to it), so it answers within its wait_s.
"""
from __future__ import annotations

import atexit
import faulthandler
import functools
import os
import re
import signal
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from . import __version__ as RUNTIME_VERSION
from .slurm import deadline
from .workers import run_bounded

STALE_MARK = "STALE_SERVER"
SHUTDOWN_S = 10.0        # the most a server's exit waits for its workers to stop
VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.M)
BUDGET_S = 48.0          # a call's deadline: the router (and the Claude app) count on an answer within 50 s
LOG_KEEP = 20


def _disk_version(pkg_dir: Path) -> str | None:
    try:
        m = VERSION_RE.search((pkg_dir / "__init__.py").read_text())
    except OSError:
        return None
    return m.group(1) if m else None


def stale_reason(pkg_dir: Path, running: str, runtime_dir: Path | None = None,
                 runtime_running: str | None = None) -> tuple[str | None, bool]:
    """(why this process runs old code or None, shared install?). Shared install (a release under <prefix>/releases
    with RELEASE.json): <prefix>/current now points to another release. Otherwise: the package's __init__.py on disk
    carries another version (a checkout or editable install updated in place), or llmt-runtime's does."""
    pkg_dir = Path(pkg_dir)
    rel = pkg_dir.parent.parent
    if (rel / "RELEASE.json").is_file() and rel.parent.name == "releases":
        cur = rel.parent.parent / "current"
        try:
            if cur.exists() and cur.resolve() != rel.resolve():
                return f"release {cur.resolve().name} is installed; this server runs {rel.name} ({running})", True
        except OSError:
            pass
        return None, True
    v = _disk_version(pkg_dir)
    if v and v != running:
        return f"version {v} is installed; this server runs {running}", False
    if runtime_dir is not None and runtime_running:
        rv = _disk_version(Path(runtime_dir))
        if rv and rv != runtime_running:
            return f"llmt-runtime {rv} is installed; this server runs llmt-runtime {runtime_running}", False
    return None, False


class Server:
    """One MCP server's common state: stale checks, calls in flight, the exit, the log."""

    def __init__(self, rt, package_file: str, busy: Callable[[], bool] | None = None,
                 shutdown: Callable[[], None] | None = None, period: float | None = None):
        self.rt = rt
        self.pkg_dir = Path(package_file).resolve().parent
        self.busy = busy or (lambda: False)
        self.shutdown = shutdown or (lambda: None)
        try:
            self.period = float(rt.env("STALE_CHECK_S", "") or (period if period is not None else rt.stale_period))
        except ValueError:
            self.period = float(rt.stale_period)
        self.state = {"inflight": 0, "background": 0, "exiting": False, "checked": 0.0, "why": None,
                      "shared": False, "serving": False}
        self.lock = threading.Lock()
        self.log_file = None
        self.log_path: str | None = None

    # ---- stale
    def stale(self) -> str | None:
        """stale_reason for this process, checked at most every 2 s."""
        now = time.time()
        if now - self.state["checked"] > 2.0:
            self.state["why"], self.state["shared"] = stale_reason(
                self.pkg_dir, self.rt.version, Path(__file__).resolve().parent, RUNTIME_VERSION)
            self.state["checked"] = now
        return self.state["why"]

    def may_exit(self) -> bool:
        """Only a server started by serve(). Then P_EXIT_WHEN_STALE: 1 always, 0 never; default: a shared install."""
        if not self.state["serving"]:
            return False
        v = self.rt.env("EXIT_WHEN_STALE", "").strip()
        return v == "1" or (v != "0" and self.state["shared"])

    def idle(self) -> bool:
        try:
            busy = bool(self.busy())
        except Exception:  # noqa: BLE001 (unknown: count as busy)
            busy = True
        return not self.state["inflight"] and not self.state["background"] and not busy

    def exit_soon(self, why: str, delay: float) -> None:
        """Exits in `delay` s (an answer in flight reaches the caller first), after stopping every worker (os._exit
        skips atexit). Caller holds the lock."""
        if self.state["exiting"]:
            return
        self.state["exiting"] = True
        self.log(f"exit: {why}")
        sys.stderr.write(f"{self.rt.cli}: {STALE_MARK}: exiting to pick up the new code: {why}\n")
        sys.stderr.flush()

        def go():
            time.sleep(delay)
            try:
                run_bounded(self.shutdown, SHUTDOWN_S)      # a stopper that hangs must not keep a stale server alive
            finally:
                os._exit(0)
        threading.Thread(target=go, daemon=True, name=f"{self.rt.cli}-exit").start()

    def enter_call(self) -> str | None:
        """Counts a call in. Returns a STALE_SERVER message (the call is refused) when this server is stale, may
        exit and has nothing else running; otherwise None."""
        with self.lock:
            why = self.stale()
            if why and self.may_exit() and (self.state["exiting"] or self.idle()):
                self.exit_soon(why, 1.0)
                return (f"{STALE_MARK}: {why}. This server is exiting so that the next one runs the installed "
                        "version: call again.")
            self.state["inflight"] += 1
            return None

    def leave_call(self) -> None:
        with self.lock:
            self.state["inflight"] -= 1

    def note(self) -> str | None:
        """The server_stale note added to answers of a stale server that keeps serving."""
        if not self.state["why"]:
            return None
        return (f"{self.state['why']}: this answer comes from the old code"
                + ("; the server restarts when idle" if self.may_exit() else "; restart the MCP server"))

    @contextmanager
    def background(self):
        """Work that goes on after a call returned, entered while the call is still in flight (the stale exit waits
        for it). For a thread, use start_background: entering this inside the new thread leaves a gap after the call
        returns."""
        with self.lock:
            self.state["background"] += 1
        try:
            yield
        finally:
            with self.lock:
                self.state["background"] -= 1

    def start_background(self, fn: Callable, *args, name: str | None = None, **kwargs) -> threading.Thread:
        """Runs fn(*args, **kwargs) in a thread that outlives the call (lmpt submits the rest of a request so): counted
        before the thread starts, in the calling thread, so the stale exit cannot slip in between."""
        with self.lock:
            self.state["background"] += 1

        def run():
            try:
                fn(*args, **kwargs)
            finally:
                with self.lock:
                    self.state["background"] -= 1
        t = threading.Thread(target=run, daemon=False, name=name or f"{self.rt.cli}-background")
        t.start()
        return t

    def watch(self) -> None:
        """Exits this server when it is stale, may exit and is idle."""
        def loop():
            while True:
                time.sleep(self.period)
                with self.lock:
                    if self.stale() and self.may_exit() and self.idle():
                        self.exit_soon(self.state["why"], 0.0)
        threading.Thread(target=loop, daemon=True, name=f"{self.rt.cli}-watch-install").start()

    def serve(self, log: bool = True, signals: bool = True) -> None:
        """Call in main() before running the server: this process may now refuse stale calls and exit; the install
        watch starts; with log, the server's log (start_log); with signals, SIGTERM/SIGHUP/SIGINT stop the workers
        first (llmt servers --stop: re-raising skips atexit)."""
        self.state["serving"] = True
        if log:
            self.start_log()
        if signals:
            self.install_signal_handlers()
        self.watch()

    # ---- the log: <out>/.server/<time>_<host>_<pid>.log, the newest LOG_KEEP kept
    def start_log(self) -> None:
        d = self.rt.out_root() / ".server"
        try:
            d.mkdir(parents=True, exist_ok=True)
            old = sorted(d.glob("*.log"))
            for f in old[:max(0, len(old) - (LOG_KEEP - 1))]:
                try:
                    f.unlink()
                except OSError:
                    pass
            p = d / f"{time.strftime('%Y%m%d_%H%M%S')}_{socket.gethostname().split('.')[0]}_{os.getpid()}.log"
            self.log_file = open(p, "a", buffering=1)
            self.log_path = str(p)
        except OSError:
            return
        faulthandler.enable(file=self.log_file)     # a fatal signal (segfault, abort) dumps every thread here
        self.log(f"start {self.rt.cli} {self.rt.version} (llmt-runtime {RUNTIME_VERSION}) argv {sys.argv[1:]} "
                 f"python {sys.version.split()[0]}")
        atexit.register(lambda: self.log("exit"))
        old_hook = sys.excepthook

        def hook(t, v, tb):
            self.log("uncaught " + "".join(traceback.format_exception(t, v, tb)))
            old_hook(t, v, tb)
        sys.excepthook = hook

    def log(self, msg: str) -> None:
        if self.log_file is None:
            return
        try:
            self.log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        except (OSError, ValueError):
            pass

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
            except (ValueError, OSError):
                continue

            def handler(signum, frame, prev=prev):
                self.log(f"signal {signal.Signals(signum).name} received")
                run_bounded(self.shutdown, SHUTDOWN_S)
                signal.signal(signum, prev if callable(prev) or prev in (signal.SIG_DFL, signal.SIG_IGN)
                              else signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            try:
                signal.signal(sig, handler)
            except ValueError:      # not the main thread
                return

    # ---- the tool wrapper
    def tool(self, tool_error: type, budget_s: float = BUDGET_S):
        """Decorator for every tool: refused with STALE_SERVER when stale and idle; otherwise run in a worker thread
        under a deadline of budget_s; errors come back as tool_error with their text (SDK 2.x hides the text of other
        exceptions; with the log, where the traceback is); calls logged; a stale server's answers carry server_stale."""
        def deco(fn):
            def bounded(*args, **kwargs):
                with deadline(budget_s):
                    return fn(*args, **kwargs)

            @functools.wraps(fn)
            async def wrap(*args, **kwargs):
                import anyio.to_thread   # (the MCP SDK's; the runtime itself is standard library only)
                t0 = time.time()
                self.log(f"call {fn.__name__} {_summary(kwargs)}")
                refused = self.enter_call()
                if refused:
                    self.log(f"refused {fn.__name__}: {refused}")
                    raise tool_error(refused)
                try:
                    r = await anyio.to_thread.run_sync(functools.partial(bounded, *args, **kwargs))
                except Exception as exc:  # noqa: BLE001
                    self.log(f"FAIL {fn.__name__} after {time.time() - t0:.1f} s: {type(exc).__name__}: {exc}\n"
                             + traceback.format_exc())
                    if isinstance(exc, tool_error):
                        raise
                    where = f" (traceback: {self.log_path})" if self.log_path else ""
                    raise tool_error(f"{type(exc).__name__}: {exc}{where}") from exc
                except BaseException as exc:  # cancellation, exit: logged, re-raised
                    self.log(f"FAIL {fn.__name__} after {time.time() - t0:.1f} s: {type(exc).__name__}")
                    raise
                finally:
                    self.leave_call()
                self.log(f"done {fn.__name__} {time.time() - t0:.1f} s")
                n = self.note()
                if n and isinstance(r, dict):
                    r = {**r, "server_stale": n}
                return r
            return wrap
        return deco


def _summary(kwargs: dict, limit: int = 300) -> str:
    """A call's arguments for the log, shortened."""
    try:
        s = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    except Exception:  # noqa: BLE001
        s = "?"
    return s if len(s) <= limit else s[:limit] + "..."

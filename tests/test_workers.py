"""Worker processes end with their server and take their solver runs with them; stopping a server's workers."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from llmt_runtime import workers as W

ROOT = Path(__file__).resolve().parents[1]

WORKER = textwrap.dedent("""
    import os, subprocess, sys, time
    sys.path.insert(0, {root!r})
    from llmt_runtime import workers as W
    W.PARENT_POLL_S = 0.2
    W.watch_parent(int(sys.argv[1]), name="tt")
    p = subprocess.Popen(["sleep", "60"], start_new_session=True)
    W.track(p)
    open(sys.argv[2], "w").write(f"{{os.getpid()}} {{p.pid}}")
    time.sleep(60)
""")

SERVER = textwrap.dedent("""
    import subprocess, sys
    subprocess.Popen([sys.executable, sys.argv[1], str(__import__("os").getpid()), sys.argv[2]])
    import time; time.sleep(float(sys.argv[3]))
""")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
        return bool(r.stdout.strip()) and not r.stdout.strip().startswith("Z")


def wait(cond, t=8.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def start(tmp_path, server_life: float):
    w, s, pids = tmp_path / "worker.py", tmp_path / "server.py", tmp_path / "pids"
    w.write_text(WORKER.format(root=str(ROOT)))
    s.write_text(SERVER)
    srv = subprocess.Popen([sys.executable, str(s), str(w), str(pids), str(server_life)])
    assert wait(lambda: pids.is_file() and pids.read_text().strip())
    worker, child = (int(x) for x in pids.read_text().split())
    return srv, worker, child


def test_server_killed_takes_worker_and_solver(tmp_path):
    srv, worker, child = start(tmp_path, 60)
    assert alive(worker) and alive(child)
    srv.kill()                                                       # kill -9: no cleanup in the server
    srv.wait()
    assert wait(lambda: not alive(worker) and not alive(child)), "orphaned worker or solver left running"


def test_sigterm_ends_the_tracked_solver(tmp_path):
    srv, worker, child = start(tmp_path, 60)
    os.kill(worker, signal.SIGTERM)                                  # Pool.terminate(), llmt servers --stop
    assert wait(lambda: not alive(child)), "a terminated worker left its solver running"
    srv.kill()
    srv.wait()


def test_watch_once_per_process_and_track():
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    W.track(p)
    assert p in W._CHILDREN
    W.untrack(p)
    assert p not in W._CHILDREN
    p.kill()
    p.wait()


def test_on_orphan_hooks_and_end_children():
    seen = []
    W.on_orphan(lambda: seen.append(1))
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    W.track(p)
    try:
        W.end_children()
        assert p.wait(5) == -9 and seen == [1]
    finally:
        W.untrack(p)
        W._HOOKS.clear()


def test_shutdown_runs_every_stopper():
    done = []

    def bad():
        raise RuntimeError("x")
    W.shutdown(lambda: done.append("a"), bad, lambda: done.append("b"))
    assert done == ["a", "b"]


def test_stop_executor_terminates_spawn_workers():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ex = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    ex.submit(time.sleep, 30)
    assert wait(lambda: bool(getattr(ex, "_processes", None)))
    procs = list(ex._processes.values())
    W.stop_executor(ex)
    assert wait(lambda: all(not alive(p.pid) for p in procs), 3)
    W.stop_executor(None)
    W.stop_pool(None)
    W.stop_process(None)


def test_stop_process_escalates_to_kill(tmp_path):
    import multiprocessing as mp
    p = mp.get_context("spawn").Process(target=time.sleep, args=(60,))
    p.start()
    os.kill(p.pid, signal.SIGSTOP)                               # stopped: SIGTERM stays pending (a native call)
    t0 = time.time()
    W.stop_process(p, wait=0.5)
    assert wait(lambda: not alive(p.pid), 3) and time.time() - t0 < 5


def test_stop_pool_escalates(tmp_path):
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(1)
    pool.apply_async(time.sleep, (60,))
    time.sleep(0.5)
    procs = list(pool._pool)
    for p in procs:
        os.kill(p.pid, signal.SIGSTOP)
    t0 = time.time()
    W.stop_pool(pool, grace=0.5)
    assert wait(lambda: all(not alive(p.pid) for p in procs), 3) and time.time() - t0 < 8


def test_server_pid_watched_even_when_not_the_parent(tmp_path):
    """forkserver: the worker's parent is the fork server, not the server: the server's pid is watched too."""
    server = subprocess.Popen(["sleep", "60"])
    w = tmp_path / "w.py"
    w.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from llmt_runtime import workers as W
        W.PARENT_POLL_S = 0.2
        W.watch_parent({server.pid})
        print("started", flush=True)
        time.sleep(60)
    """))
    worker = subprocess.Popen([sys.executable, str(w)], stdout=subprocess.PIPE, text=True)
    assert worker.stdout.readline().strip() == "started"         # its own parent (this test) lives: it runs
    server.kill()
    server.wait()
    assert worker.wait(8) == 0

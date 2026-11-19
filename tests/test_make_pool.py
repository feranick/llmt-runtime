"""workers.make_pool / make_executor / make_process: a sandbox that refuses semaphores or processes gives a clear error
(the Gemini Mac app's Local MCP with a customized sandbox, 2026-11-18: PermissionError [Errno 1] from SemLock)."""
from __future__ import annotations

import errno
import multiprocessing as mp
import time

import pytest

from llmt_runtime import workers as W


def test_pools_work_where_permitted():
    # builtins as the work: a spawn child cannot import this test module
    pool = W.make_pool(1)
    try:
        assert pool.apply_async(pow, (3, 2)).get(timeout=60) == 9
    finally:
        W.stop_pool(pool)
    ex = W.make_executor(1)
    try:
        assert ex.submit(pow, 4, 2).result(timeout=60) == 16
    finally:
        W.stop_executor(ex)
    p = W.make_process(time.sleep, (0,))
    p.join(60)
    assert p.exitcode == 0


@pytest.mark.parametrize("code", [errno.EPERM, errno.EACCES, errno.ENOSYS])
def test_refused_by_the_sandbox(monkeypatch, code):
    def refuse(*a, **k):
        raise OSError(code, "Operation not permitted")
    ctx = mp.get_context("spawn")
    monkeypatch.setattr(type(ctx), "Pool", refuse)
    monkeypatch.setattr(type(ctx), "Process", refuse)
    monkeypatch.setattr(type(ctx), "Lock", refuse)            # ProcessPoolExecutor's queues
    for make in (lambda: W.make_pool(2, maxtasksperchild=10), lambda: W.make_executor(2),
                 lambda: W.make_process(time.sleep, (0,))):
        with pytest.raises(W.WorkersNotPermitted) as info:
            make()
        text = str(info.value)
        assert text.startswith(W.NOT_PERMITTED) and "give this app Full Access, or run through the router" in text
        assert "Operation not permitted" in text and info.value.cause.errno == code


def test_other_errors_pass_through(monkeypatch):
    def full(*a, **k):
        raise OSError(errno.EMFILE, "Too many open files")
    monkeypatch.setattr(type(mp.get_context("spawn")), "Pool", full)
    with pytest.raises(OSError) as info:
        W.make_pool(2)
    assert not isinstance(info.value, W.WorkersNotPermitted) and info.value.errno == errno.EMFILE
    assert W.not_permitted(PermissionError(errno.EPERM, "x")) is not None
    assert W.not_permitted(ValueError("x")) is None


def test_stop_pool_does_not_hang_on_idle_or_busy_workers():
    """stop_pool used to kill the workers first; an idle worker killed while it held the inqueue's read lock left
    Pool.terminate() waiting forever (found while testing make_pool, 2026.11.18)."""
    for busy in (False, True):
        pool = W.make_pool(2, initializer=W.watch_parent)
        assert pool.apply_async(pow, (2, 3)).get(timeout=60) == 8
        if busy:
            pool.apply_async(time.sleep, (60,))
            time.sleep(0.5)
        procs = list(pool._pool)
        t0 = time.time()
        W.stop_pool(pool, grace=2.0)
        assert time.time() - t0 < 10
        for p in procs:
            p.join(5)
            assert not p.is_alive()

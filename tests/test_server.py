"""Stale installs (a shared release, a checkout, the runtime itself), refusing and exiting only when idle, background
work, the tool wrapper and the server's log."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from llmt_runtime import server as SV


def release(tmp_path, name="2026.11.09.1"):
    prefix = tmp_path / "opt"
    rel = prefix / "releases" / name
    pkg = rel / "fea-translator" / "fea_translator"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "2026.11.09.5"\n')
    (rel / "RELEASE.json").write_text("{}")
    return prefix, rel, pkg


def test_stale_reason_shared_release(tmp_path):
    prefix, rel, pkg = release(tmp_path)
    (prefix / "current").symlink_to(rel)
    assert SV.stale_reason(pkg, "2026.11.09.5") == (None, True)
    _, rel2, _ = release(tmp_path, "2026.11.09.2")
    (prefix / "current").unlink()
    (prefix / "current").symlink_to(rel2)
    why, shared = SV.stale_reason(pkg, "2026.11.09.5")
    assert shared and why == "release 2026.11.09.2 is installed; this server runs 2026.11.09.1 (2026.11.09.5)"


def test_stale_reason_checkout_and_runtime(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "2026.11.09.6"\n')
    assert SV.stale_reason(pkg, "2026.11.09.5") == ("version 2026.11.09.6 is installed; this server runs 2026.11.09.5",
                                                     False)
    rtd = tmp_path / "rt"
    rtd.mkdir()
    (rtd / "__init__.py").write_text('__version__ = "2026.11.10.1"\n')
    why, _ = SV.stale_reason(pkg, "2026.11.09.6", rtd, "2026.11.09.1")
    assert why == "llmt-runtime 2026.11.10.1 is installed; this server runs llmt-runtime 2026.11.09.1"
    assert SV.stale_reason(pkg, "2026.11.09.6", rtd, "2026.11.10.1") == (None, False)


@pytest.fixture
def stale_server(rt, tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "2099.01.01.1"\n')
    exits = []
    monkeypatch.setattr(SV.os, "_exit", lambda code: exits.append(code))
    busy = {"v": False}
    stopped = []
    s = SV.Server(rt, str(pkg / "__init__.py"), busy=lambda: busy["v"], shutdown=lambda: stopped.append(1))
    return s, exits, busy, stopped


def test_not_serving_never_refuses(stale_server):
    s, exits, *_ = stale_server
    assert s.enter_call() is None                                  # tests, run-study: never refused
    s.leave_call()
    assert s.note().startswith("version 2099.01.01.1 is installed") and s.note().endswith("restart the MCP server")


def test_refuse_and_exit_when_idle(stale_server, monkeypatch):
    s, exits, busy, stopped = stale_server
    monkeypatch.setenv("TT_EXIT_WHEN_STALE", "1")
    s.state["serving"] = True
    busy["v"] = True                                               # studies in the pool: answer with the old code
    assert s.enter_call() is None
    s.leave_call()
    busy["v"] = False
    with s.background():                                           # a submission still under way
        assert s.enter_call() is None
        s.leave_call()
    msg = s.enter_call()
    assert msg.startswith("STALE_SERVER: version 2099.01.01.1") and msg.endswith("call again.")
    time.sleep(1.3)
    assert exits == [0] and stopped == [1]                          # workers stopped before os._exit


def test_exit_when_stale_settings(stale_server, monkeypatch):
    s, *_ = stale_server
    s.state["serving"] = True
    s.stale()
    assert s.may_exit() is False                                   # a checkout: default never
    s.state["shared"] = True
    assert s.may_exit() is True
    monkeypatch.setenv("TT_EXIT_WHEN_STALE", "0")
    assert s.may_exit() is False


def test_watch_exits_idle_server(stale_server, monkeypatch):
    s, exits, _, stopped = stale_server
    monkeypatch.setenv("TT_EXIT_WHEN_STALE", "1")
    monkeypatch.setenv("TT_STALE_CHECK_S", "0.2")
    s2 = SV.Server(s.rt, str(s.pkg_dir / "__init__.py"), shutdown=lambda: stopped.append(2))
    assert s2.period == 0.2
    s2.state["serving"] = True
    s2.watch()
    time.sleep(0.8)
    assert 0 in exits and 2 in stopped


class ToolError(Exception):
    pass


def test_tool_wrapper(stale_server, rt, tmp_path):
    pytest.importorskip("anyio")
    s, *_ = stale_server
    s.start_log()
    from llmt_runtime.slurm import remaining

    @s.tool(ToolError, budget_s=20)
    def ok(x=1):
        return {"x": x, "left": remaining()}

    @s.tool(ToolError)
    def boom():
        raise ValueError("bad input")

    r = asyncio.run(ok(x=2))
    assert r["x"] == 2 and 0 < r["left"] <= 20                     # under the call's deadline
    assert r["server_stale"].startswith("version 2099.01.01.1")    # the old code says so
    with pytest.raises(ToolError) as e:
        asyncio.run(boom())
    assert str(e.value).startswith("ValueError: bad input (traceback: ")
    log = Path(s.log_path).read_text()
    assert "call ok x=2" in log and "done ok" in log and "FAIL boom" in log and "Traceback" in log
    assert s.state["inflight"] == 0
    assert Path(s.log_path).parent == rt.out_root() / ".server"


def test_log_keeps_the_newest(rt):
    d = rt.out_root() / ".server"
    d.mkdir(parents=True)
    for i in range(25):
        (d / f"2026010{i:02d}_x_{i}.log").write_text("")
    s = SV.Server(rt, SV.__file__)
    s.start_log()
    assert len(list(d.glob("*.log"))) == SV.LOG_KEEP
    json.dumps(s.state)


def test_start_background_counts_before_the_thread_runs(stale_server, monkeypatch):
    import threading
    s, exits, *_ = stale_server
    monkeypatch.setenv("TT_EXIT_WHEN_STALE", "1")
    s.state["serving"] = True
    go = threading.Event()
    t = s.start_background(go.wait, 5)
    assert s.state["background"] == 1 and not s.idle()
    assert s.enter_call() is None                                # not idle: answered, no exit
    s.leave_call()
    go.set()
    t.join(5)
    assert s.state["background"] == 0 and s.idle()


def test_exit_does_not_hang_on_a_stopper(rt, tmp_path, monkeypatch):
    pkg = tmp_path / "p2"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "2099.01.01.1"\n')
    exits = []
    monkeypatch.setattr(SV.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(SV, "SHUTDOWN_S", 0.3)
    s = SV.Server(rt, str(pkg / "__init__.py"), shutdown=lambda: time.sleep(30))
    with s.lock:
        s.exit_soon("test", 0.0)
    time.sleep(1.0)
    assert exits == [0]

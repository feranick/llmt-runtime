"""Shared fixtures: a test translator (prefix TT) whose run-study is tests/ttpkg/cli.py, the fake SLURM, a clean
environment and fresh caches."""
from __future__ import annotations

import json
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest

from llmt_runtime import Hooks, Runtime
from llmt_runtime import jobs as J
from llmt_runtime import slurm as S
from llmt_runtime.testing import fake_slurm

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("TT_", "UU_", "SLURM_", "FAKE_")):
            monkeypatch.delenv(k, raising=False)
    S.forget()
    J._SBATCH_DOWN.update(until=0.0, why="")
    J._PENDING.update(t=0.0, v=None)
    yield
    S.forget()


def make_rt(tmp_path: Path, **kw) -> Runtime:
    hooks = kw.pop("hooks", None) or Hooks()
    if hooks.script_lines is None:   # the test translator's package is in tests/
        hooks.script_lines = lambda request, name, study: [
            f"export PYTHONPATH={shlex.quote(str(HERE))}${{PYTHONPATH:+:$PYTHONPATH}}"]
    args = dict(prefix="TT", cli="tt", package="ttpkg", version="2026.11.09.7", kind="threads",
                out_default=str(tmp_path / "out"), timeout_default=600)
    args.update(kw)
    return Runtime(hooks=hooks, **args)


@pytest.fixture
def rt(tmp_path):
    return make_rt(tmp_path)


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """The fake SLURM on PATH, TT_EXECUTOR=slurm; returns its state folder."""
    state = fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setenv("TT_EXECUTOR", "slurm")
    yield state
    for f in (state / "jobs").glob("*.json"):          # no job outlives its test
        try:
            pid = json.loads(f.read_text()).get("pid")
            if pid:
                os.killpg(pid, signal.SIGKILL)
        except (OSError, ValueError):
            pass


def python() -> str:
    return sys.executable

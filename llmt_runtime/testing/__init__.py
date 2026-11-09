"""llmt_runtime.testing — the fake SLURM for the translators' tests (and llmt-runtime's own).
Version: 2026.11.09.1

    bin_dir = install_fake_slurm(tmp_path)        # sbatch, squeue, scancel, sacct, sinfo, scontrol in tmp_path/bin
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(tmp_path / "slurm"))
or, in one step: fake_slurm(tmp_path, monkeypatch). Knobs: fakeslurm.py's docstring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

FAKESLURM = Path(__file__).with_name("fakeslurm.py")
COMMANDS = ("sbatch", "squeue", "scancel", "sacct", "sinfo", "scontrol")


def install_fake_slurm(where: Path, python: str | None = None) -> Path:
    """Writes the six SLURM commands into <where>/bin as small scripts that run fakeslurm.py with this Python;
    returns the folder (put it first on PATH)."""
    b = Path(where) / "bin"
    b.mkdir(parents=True, exist_ok=True)
    py = python or sys.executable
    for c in COMMANDS:
        f = b / c
        f.write_text(f'#!/bin/sh\nexec "{py}" "{FAKESLURM}" {c} "$@"\n')
        f.chmod(0o755)
    return b


def fake_slurm(tmp_path: Path, monkeypatch) -> Path:
    """install_fake_slurm, on PATH, with its state in tmp_path/slurm; returns the state folder."""
    b = install_fake_slurm(tmp_path)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ.get('PATH', '')}")
    state = Path(tmp_path) / "slurm"
    state.mkdir(exist_ok=True)
    monkeypatch.setenv("FAKE_SLURM_DIR", str(state))
    return state


def calls(state: Path, command: str) -> int:
    """How often a fake command ran."""
    f = Path(state) / "calls" / command
    return int(f.read_text()) if f.is_file() else 0

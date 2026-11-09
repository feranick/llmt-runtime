"""config.py — a translator's runtime: its settings prefix, CLI, package, version, SLURM kind and hooks.
Version: 2026.11.09.1

Configured per translator, never by globals: two translators in one process (tests, the router) each have their own
Runtime. Environment variables keep their names: Runtime(prefix="FEAT") reads FEAT_SLURM_ARGS, FEAT_MAX_MEM, ...
"""
from __future__ import annotations

import functools
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

Study = dict


@dataclass
class Hooks:
    """What differs between translators. Every hook is optional; None takes the default given here.

    study_procs(study) -> int | None    the study's own ranks/threads (default: study["procs"]; lmpt, cfdt: solve.procs)
    procs(study) -> int                 what the job asks for, after the translator's size caps (default: procs_of,
                                        capped at the node's cores)
    mem_auto(study) -> str | None       P_SLURM_MEM=auto: the job's --mem for this study (default: rt.mem_default)
    time_need_min(study) -> int         minutes the study's runs may take, with the 10-minute margin that lets the
                                        translator's own timeout fire first (default: ceil(timeout_s / 60) + 10)
    solves(study) -> int                runs with the full timeout_s each, for the default time_need_min (default 1)
    timeout_for_limit(study, minutes) -> float  P_SLURM_TIME=estimate: each run's timeout (s) in a job of `minutes`
                                        (default (minutes - 10) x 60 / solves)
    resource_lines(study, procs) -> [str]  the #SBATCH resource options without '#SBATCH ' (default: by rt.kind)
    script_lines(request, name, study) -> [str]  shell lines before the exec line (e.g. an export)
    job_extra(study) -> dict            extra keys for job.json (size estimates, a memory plan)
    oom_advice(study_folder) -> str     why and what to ask for when SLURM killed a job for memory
    timeout_hint: str                   appended to the timed-out advice ("or coarsen the mesh")
    result_file(request, name) -> Path  a finished study's result (default <request>/<name>/<name>.json)
    running_progress(study_folder, study) -> dict   progress keys of a running study for jobs()
    solver_left_s(study_folder, study) -> float     seconds before the running solve's own deadline (extend)
    """
    study_procs: Optional[Callable[[Study], Any]] = None
    procs: Optional[Callable[[Study], int]] = None
    mem_auto: Optional[Callable[[Study], Optional[str]]] = None
    time_need_min: Optional[Callable[[Study], int]] = None
    solves: Optional[Callable[[Study], int]] = None
    timeout_for_limit: Optional[Callable[[Study, int], float]] = None
    resource_lines: Optional[Callable[[Study, int], list]] = None
    script_lines: Optional[Callable[[Path, str, Study], list]] = None
    job_extra: Optional[Callable[[Study], dict]] = None
    oom_advice: Optional[Callable[[Path], str]] = None
    timeout_hint: str = ""
    result_file: Optional[Callable[[Path, str], Path]] = None
    running_progress: Optional[Callable[[Path, Study], dict]] = None
    solver_left_s: Optional[Callable[[Path, Study], float]] = None


class _Bound:
    """A module whose functions take the Runtime first, with it bound: rt.slurm.caps() is slurm.caps(rt)."""

    def __init__(self, module, rt):
        self._m, self._rt = module, rt

    def __getattr__(self, name):
        v = getattr(self._m, name)
        if inspect.isfunction(v):
            params = list(inspect.signature(v).parameters)
            if params and params[0] == "rt":
                return functools.partial(v, self._rt)
        return v


@dataclass
class Runtime:
    """One translator's runtime settings.

    prefix              settings prefix without '_' (FEAT, CFDT, LMPT, OMCT)
    cli                 the CLI name: job names start with '<cli>:'; messages say it
    package             the Python package: jobs run `python -m <package>.cli run-study REQUEST NAME`
    version             the translator's __version__ (recorded in requests, reported by status)
    kind                'threads' (--ntasks=1 --cpus-per-task=N) or 'mpi' (--ntasks=N --ntasks-per-core=1)
    out_default         P_OUT's default (~/feat_runs)
    min_minutes         P_SLURM_MIN_MINUTES's default
    timeout_default     P_TIMEOUT's default (s)
    slurm_time_default  P_SLURM_TIME's default: max | timeout | estimate | minutes
    mem_default         the job's --mem when P_SLURM_MEM=auto and the study's size is unknown
    one_node            MPI: also #SBATCH --nodes=1
    cancel_file         the flag file cancel() writes in a study folder
    stale_period        s between two looks of the MCP server at the install (P_STALE_CHECK_S overrides)
    time_aliases        old P_SLURM_TIME words mapped to the common ones during a migration ({"estimate": "timeout"})
    """
    prefix: str
    cli: str
    package: str
    version: str
    kind: str = "threads"
    out_default: Optional[str] = None
    min_minutes: float = 1.0
    timeout_default: float = 900.0
    slurm_time_default: str = "timeout"
    mem_default: str = "8G"
    one_node: bool = False
    cancel_file: str = "cancel"
    stale_period: float = 10.0
    time_aliases: dict = field(default_factory=dict)
    hooks: Hooks = field(default_factory=Hooks)

    def __post_init__(self):
        self.prefix = self.prefix.rstrip("_").upper()
        if self.kind not in ("threads", "mpi"):
            raise ValueError("kind: 'threads' or 'mpi'")
        if self.out_default is None:
            self.out_default = f"~/{self.cli}_runs"

    # ---- settings
    def var(self, name: str) -> str:
        """The environment variable's full name: var('SLURM_ARGS') -> 'FEAT_SLURM_ARGS'."""
        return f"{self.prefix}_{name}"

    def env(self, name: str, default: str = "") -> str:
        """The setting's value (read at each call: tests and llmt change the environment)."""
        return os.environ.get(self.var(name), default)

    def out_root(self) -> Path:
        return Path(self.env("OUT", self.out_default) or self.out_default).expanduser()

    @property
    def runtime_version(self) -> str:
        from . import __version__
        return __version__

    # ---- the modules, with this runtime bound
    @property
    def slurm(self):
        from . import slurm
        return _Bound(slurm, self)

    @property
    def jobs(self):
        from . import jobs
        return _Bound(jobs, self)

    @property
    def requests(self):
        from . import requests
        return _Bound(requests, self)

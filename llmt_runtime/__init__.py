"""llmt_runtime — the code the llm-translators share: worker processes, memory and out of memory, the SLURM executor,
request folders and the MCP server's stale-install handling (llm-translators docs/EXECUTION.md made code).
Version: 2026.11.09.4

    from llmt_runtime import Runtime, Hooks
    RT = Runtime(prefix="FEAT", cli="feat", package="fea_translator", version=__version__, kind="threads",
                 out_default="~/feat_runs", timeout_default=900, slurm_time_default="max", hooks=Hooks(...))
    RT.slurm.caps(); RT.jobs.submit(request, name, study); RT.requests.ran_with(meta, results)

Standard library only. Every function that depends on a translator takes its Runtime first (rt.<module>.<fn>(...)
binds it); the rest (workers, memory, the call deadline) are plain functions.
"""
from __future__ import annotations

__version__ = "2026.11.09.4"

from .config import Hooks, Runtime  # noqa: E402

__all__ = ["Hooks", "Runtime", "__version__"]

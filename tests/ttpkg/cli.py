"""run-study for the tests. TT_MODE: ok (default: writes <name>/<name>.json), sleep:<s> (then ok), oom (SLURM's
out-of-memory line, exit 1), crash (a traceback, exit 1)."""
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    _, cmd, request, name = sys.argv
    mode = os.environ.get("TT_MODE", "ok")
    if mode.startswith("sleep:"):
        time.sleep(float(mode.split(":", 1)[1]))
        mode = "ok"
    if mode == "oom":
        print(f"slurmstepd: error: Detected 2 oom_kill events in StepId={os.environ.get('SLURM_JOB_ID')}.batch. "
              "Some of the step tasks have been OOM Killed.")
        return 1
    if mode == "crash":
        print("Traceback (most recent call last):\nValueError: boom")
        return 1
    d = Path(request) / name
    tmp = d / f".{name}.json.tmp"
    tmp.write_text(json.dumps({"name": name, "ok": True, "translator": "2026.11.09.7",
                               "env": {k: os.environ.get(k) for k in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS",
                                                                      "SLURM_MEM_PER_NODE")}}))
    tmp.replace(d / f"{name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

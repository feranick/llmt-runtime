"""llmt_runtime.selftest.cli — the self-test's 'translator': `python -m llmt_runtime.selftest.cli run-study REQUEST
NAME` sleeps the study's seconds, then writes <request>/<name>/<name>.json (ok, its SLURM job, start and end times).
Version: 2026.11.18.2
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def run_study(request: Path, name: str) -> int:
    meta = json.loads((request / "request.json").read_text())
    study = (meta.get("studies") or {}).get(name) or {}
    t0 = time.time()
    print(f"selftest {name}: job {os.environ.get('SLURM_JOB_ID')} runs {study.get('seconds', 30)} s", flush=True)
    time.sleep(float(study.get("seconds", 30)))
    res = {"name": name, "ok": True, "state": "done", "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "started": round(t0, 2), "ended": round(time.time(), 2)}
    out = request / name / f"{name}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(out)
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["run-study"] and len(argv) == 3:
        return run_study(Path(argv[1]), argv[2])
    print("usage: python -m llmt_runtime.selftest.cli run-study REQUEST NAME", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

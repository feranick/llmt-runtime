#!/usr/bin/env python3
"""fakeslurm.py — fake SLURM commands for tests: sbatch, squeue, scancel, sacct, sinfo, scontrol in one script, called
by the name it is invoked as (or the name as its first argument). State lives in $FAKE_SLURM_DIR (jobs/<id>.json).
Version: 2026.11.09.1

Merged from fea-translator (tests/fake_slurm), cfd-translator, lammps-translator and openmodelica-translator.

sbatch starts job.sh in the background (honouring --chdir and --output from its #SBATCH lines and the command line),
with SLURM_JOB_ID, SLURM_NTASKS, SLURM_CPUS_PER_TASK, SLURMD_NODENAME and (with --mem) SLURM_MEM_PER_NODE in MB; it
prints the id (--parsable) and records the pid. squeue reports RUNNING while the pid lives (PENDING when held), and
'Invalid job id specified' when none of the -j ids is in the queue; scancel kills the job and writes SLURM's CANCELLED
line to its log; sacct answers from recorded states.

Knobs (environment):
  FAKE_SBATCH_FAIL=1|message    sbatch refuses (the message, or 'Unable to contact slurm controller')
  FAKE_SBATCH_PENDING=1 (or FAKE_PENDING=1)   jobs stay PENDING (reason Priority, FAKE_START as expected start)
  FAKE_SBATCH_DELAY=s           the job sleeps first        FAKE_SBATCH_SLEEP=s   sbatch itself takes s to answer
  FAKE_SBATCH_SLOW_AFTER=s      sbatch takes the job, then takes s to answer (a slow controller)
  FAKE_MIN_JOB_AGE=s            an ended job stays in squeue in its final state (COMPLETED, FAILED by its exit code,
                                CANCELLED, or a record's "final") for s seconds, as SLURM's MinJobAge (default 0)
  FAKE_PRIORITY=n               the job's priority (%Q, default 1000)
  FAKE_RUN_NODE=name            the node jobs run on (default: the first node)
  FAKE_SQUEUE_FAIL=1            squeue cannot reach the controller
  FAKE_SACCT_FAIL=1             sacct fails (no accounting)
  FAKE_SCONTROL_NODES=file      `scontrol -o show node` prints its lines (default: one node fake1, 40 CPUs, 2 threads
                                per core, 64000 MB)
  FAKE_SCONTROL_FAIL=1          scontrol cannot reach the controller   FAKE_SCONTROL_DENY=1   scontrol update refused
  FAKE_SINFO_FAIL=1             sinfo fails
  FAKE_SINFO_PARTITIONS='long*|infinite;short|2:00:00'   sinfo -o '%P|%l' (unset: sinfo fails for that format)
  FAKE_SINFO_CPUS='2/6/0/8'     sinfo %C (sinfo -N: per node, one row per partition of FAKE_SINFO_PARTITIONS)
  FAKE_TIME_LEFT='1:00:00'      squeue %L
A test changes a job's record (jobs/<id>.json) for the rest: "sacct": "OUT_OF_MEMORY" (what sacct says), "final":
"TIMEOUT" (its final state in squeue and sacct), "vanished": true (gone from squeue), "state": "PENDING".
Calls are counted in $FAKE_SLURM_DIR/calls/<command>.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path(os.environ.get("FAKE_SLURM_DIR", "/tmp/fake_slurm"))
J = D / "jobs"
NAMES = ("sbatch", "squeue", "scancel", "sacct", "sinfo", "scontrol")
DEFAULT_NODE = ("NodeName=fake1 Arch=x86_64 CoresPerSocket=20 CPUAlloc=0 CPUTot=40 NodeAddr=fake1 NodeHostName=fake1 "
                "OS=Linux RealMemory=64000 Sockets=1 State=IDLE ThreadsPerCore=2")


def _count(cmd: str) -> None:
    c = D / "calls"
    c.mkdir(parents=True, exist_ok=True)
    f = c / cmd
    f.write_text(str(int(f.read_text()) + 1 if f.is_file() else 1))


def _jobs() -> dict:
    J.mkdir(parents=True, exist_ok=True)
    return {f.stem: json.loads(f.read_text()) for f in sorted(J.glob("*.json"))}


def _save(jid: str, rec: dict) -> None:
    (J / f"{jid}.json").write_text(json.dumps(rec))


def _alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie is not alive
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        try:
            r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
            return not r.stdout.strip().startswith("Z")
        except (OSError, subprocess.TimeoutExpired):
            return True


def _rc(jid: str) -> int | None:
    try:
        return int((J / f"{jid}.rc").read_text().strip())
    except (OSError, ValueError):
        return None


def _final(jid: str, rec: dict) -> str:
    """The state of a job that has ended: what a test set ('final'), CANCELLED, else by the exit code."""
    if rec.get("final"):
        return rec["final"]
    if rec.get("state") == "CANCELLED":
        return "CANCELLED"
    return "COMPLETED" if _rc(jid) in (0, None) else "FAILED"


def _ended_at(jid: str, rec: dict) -> float:
    if rec.get("ended"):
        return rec["ended"]
    try:
        return (J / f"{jid}.rc").stat().st_mtime
    except OSError:
        return 0.0


def _state(rec: dict, jid: str = "") -> str | None:
    """RUNNING while the job lives, PENDING when held; an ended job stays listed in its final state for
    FAKE_MIN_JOB_AGE s (real SLURM: MinJobAge, 300 s by default; here 0 unless set); None: not in the queue."""
    if rec.get("vanished"):
        return None
    if rec.get("state") == "PENDING":
        return "PENDING"
    if rec.get("state") != "CANCELLED" and _alive(rec.get("pid", 0)):
        return "RUNNING"
    age = float(os.environ.get("FAKE_MIN_JOB_AGE", "0") or 0)
    if age and time.time() - _ended_at(jid, rec) < age:
        return _final(jid, rec)
    return None


def _nodes() -> list[dict]:
    f = os.environ.get("FAKE_SCONTROL_NODES")
    text = Path(f).read_text() if f else DEFAULT_NODE
    return [dict(re.findall(r"(\w+)=(\S*)", ln)) for ln in text.splitlines() if "NodeName=" in ln]


def _mem_mb(v: str) -> int | None:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)B?", v.strip().upper())
    if not m:
        return None
    return int(float(m.group(1)) * {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2)])


def _opts(args: list[str], text: str) -> dict:
    opt = dict(re.findall(r"^#SBATCH --([\w-]+)=(.*)$", text, re.M))
    i = 0
    while i < len(args):   # command-line options override the script's, as in sbatch
        a = args[i]
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opt[k] = v
        elif a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("-") \
                and not args[i + 1].endswith(".sh"):
            opt[a[2:]] = args[i + 1]
            i += 1
        elif a == "-w" and i + 1 < len(args):
            opt["nodelist"] = args[i + 1]
            i += 1
        i += 1
    return opt


def sbatch(args):
    s = os.environ.get("FAKE_SBATCH_SLEEP")
    if s:
        time.sleep(float(s))
    fail = os.environ.get("FAKE_SBATCH_FAIL")
    if fail:
        msg = fail if fail != "1" else "Batch job submission failed: Unable to contact slurm controller"
        print(f"sbatch: error: {msg}", file=sys.stderr)
        return 1
    script = Path([a for a in args if not a.startswith("-")][-1])
    text = script.read_text()
    opt = _opts(args, text)
    J.mkdir(parents=True, exist_ok=True)
    counter = D / "counter"
    jid = int(counter.read_text()) + 1 if counter.is_file() else 1001
    counter.write_text(str(jid))
    ntasks = int(opt.get("ntasks", 1))
    cpus = int(opt.get("cpus-per-task", 1))
    nodes = _nodes()
    node = os.environ.get("FAKE_RUN_NODE") or (opt.get("nodelist") or "").split(",")[0] or \
        (nodes[0].get("NodeName") if nodes else "fake1")
    rec = {"name": opt.get("job-name", script.name), "cpus": ntasks * cpus, "t0": time.time(),
           "log": opt.get("output", str(script.parent / f"slurm-{jid}.out")), "pid": 0, "args": args,
           "node": node, "prio": int(os.environ.get("FAKE_PRIORITY", "1000")), "mem": opt.get("mem"),
           "user": os.environ.get("USER", "user"), "time": opt.get("time")}
    if os.environ.get("FAKE_SBATCH_PENDING") or os.environ.get("FAKE_PENDING"):
        rec["state"] = "PENDING"
    else:
        delay = os.environ.get("FAKE_SBATCH_DELAY")
        rc = J / f"{jid}.rc"
        cmd = ["bash", "-c", (f"sleep {delay}; " if delay else "") + f"bash '{script}'; echo $? > '{rc}'"]
        Path(rec["log"]).parent.mkdir(parents=True, exist_ok=True)
        log = open(rec["log"], "a")
        env = dict(os.environ, SLURM_JOB_ID=str(jid), SLURM_CPUS_PER_TASK=str(cpus), SLURM_NTASKS=str(ntasks),
                   SLURMD_NODENAME=node)
        env.pop("SLURM_MEM_PER_NODE", None)
        if rec["mem"] and _mem_mb(rec["mem"]):   # SLURM gives the job's --mem in MB
            env["SLURM_MEM_PER_NODE"] = str(_mem_mb(rec["mem"]))
        p = subprocess.Popen(cmd, cwd=opt.get("chdir", str(script.parent)), stdout=log, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        rec["pid"] = p.pid
    _save(str(jid), rec)
    slow = os.environ.get("FAKE_SBATCH_SLOW_AFTER")      # the controller took the job, but answers late
    if slow:
        time.sleep(float(slow))
    print(f"{jid}" if "--parsable" in args else f"Submitted batch job {jid}")
    return 0


def _fmt(fmt: str, jid: str, rec: dict, state: str) -> str:
    el = int(time.time() - rec["t0"])
    vals = {"i": jid, "T": state, "r": "Priority" if state == "PENDING" else "None",
            "S": os.environ.get("FAKE_START", "N/A") if state == "PENDING" else time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(rec["t0"])),
            "M": f"{el // 60}:{el % 60:02d}", "j": rec["name"], "C": str(rec["cpus"]), "Q": str(rec.get("prio", 1000)),
            "L": os.environ.get("FAKE_TIME_LEFT", "1:00:00"), "N": "" if state == "PENDING" else rec.get("node", ""),
            "n": "", "u": rec.get("user", ""), "m": rec.get("mem") or "0"}
    return re.sub(r"%(\w)", lambda m: vals.get(m.group(1), ""), fmt)


def _arg(args, *flags):
    """An option's value in any of SLURM's forms: '-o X', '-oX', '--format X', '--format=X'."""
    for i, a in enumerate(args):
        for f in flags:
            if a == f and i + 1 < len(args):
                return args[i + 1]
            if f.startswith("--") and a.startswith(f + "="):
                return a[len(f) + 1:]
            if not f.startswith("--") and a.startswith(f) and not a.startswith("--") and len(a) > len(f):
                return a[len(f):]
    return None


def squeue(args):
    if os.environ.get("FAKE_SQUEUE_FAIL"):
        print("slurm_load_jobs error: Unable to contact slurm controller (connect failure)", file=sys.stderr)
        return 1
    fmt = _arg(args, "-o", "--format") or "%i %T"
    ids = set(_arg(args, "-j", "--jobs").split(",")) if _arg(args, "-j", "--jobs") else None
    states = _arg(args, "-t", "--states")
    states = {{"PD": "PENDING", "R": "RUNNING"}.get(s, s) for s in states.split(",")} if states else None
    user, node, name = _arg(args, "-u", "--user"), _arg(args, "-w", "--nodelist"), _arg(args, "-n", "--name")
    lines = []
    for jid, rec in _jobs().items():
        st = _state(rec, jid)
        if st is None or (ids is not None and jid not in ids) or (states is not None and st not in states) \
                or (user and rec.get("user") != user) or (node and rec.get("node") != node) \
                or (name and rec.get("name") != name):
            continue
        lines.append(_fmt(fmt, jid, rec, st))
    if ids is not None and not lines:
        print("slurm_load_jobs error: Invalid job id specified", file=sys.stderr)
        return 1
    if lines:
        print("\n".join(lines))
    return 0


def scancel(args):
    for jid in (a for a in args if not a.startswith("-")):
        f = J / f"{jid}.json"
        if not f.is_file():
            continue
        rec = json.loads(f.read_text())
        if rec.get("pid") and _alive(rec["pid"]):
            try:
                os.killpg(rec["pid"], signal.SIGTERM)
            except OSError:
                pass
        rec["state"] = "CANCELLED"
        rec["ended"] = time.time()
        rec.setdefault("sacct", f"CANCELLED by {os.getuid()}")
        _save(jid, rec)
        if rec.get("pid"):
            with open(rec["log"], "a") as fh:
                fh.write(f"slurmstepd: error: *** JOB {jid} ON {rec.get('node')} CANCELLED AT "
                         f"{time.strftime('%Y-%m-%dT%H:%M:%S')} ***\n")
    return 0


def sacct(args):
    if os.environ.get("FAKE_SACCT_FAIL"):
        print("sacct: error: Slurm accounting storage is disabled", file=sys.stderr)
        return 1
    ids = set(_arg(args, "-j", "--jobs").split(",")) if _arg(args, "-j", "--jobs") else None
    fields = (_arg(args, "-o", "--format") or "JobID,State").split(",")
    for jid, rec in _jobs().items():
        if ids is not None and jid not in ids:
            continue
        if rec.get("sacct"):
            st = rec["sacct"]
        elif rec.get("state") == "PENDING":
            st = "PENDING"
        else:
            st = "RUNNING" if _alive(rec.get("pid", 0)) else _final(jid, rec)
        ran = bool(rec.get("pid"))
        vals = {"JobID": jid, "State": st, "NodeList": rec.get("node") if ran else "None assigned",
                "JobName": rec["name"]}
        print("|".join(vals.get(f, "") for f in fields))
    return 0


def sinfo(args):
    if os.environ.get("FAKE_SINFO_FAIL"):
        print("sinfo: error: Unable to contact slurm controller", file=sys.stderr)
        return 1
    fmt = _arg(args, "-o", "--format") or ""
    if "%P" in fmt:
        parts = os.environ.get("FAKE_SINFO_PARTITIONS")
        if not parts:
            return 1
        print("\n".join(parts.split(";")))
        return 0
    if "-N" in args or "%N" in fmt or "%m" in fmt:
        # one row per node and partition, as sinfo -N prints them
        nparts = len((os.environ.get("FAKE_SINFO_PARTITIONS") or "x").split(";"))
        want = set((_arg(args, "-n", "--nodes") or "").split(",")) - {""}
        for n in _nodes():
            if want and n.get("NodeName") not in want:
                continue
            vals = {"N": n.get("NodeName", ""), "n": n.get("NodeHostName", ""), "c": n.get("CPUTot", ""),
                    "m": n.get("RealMemory", ""), "Z": n.get("ThreadsPerCore", "1"),
                    "C": os.environ.get("FAKE_SINFO_CPUS", "2/6/0/8")}
            for _ in range(nparts):
                print(re.sub(r"%(\w)", lambda m: vals.get(m.group(1), ""), fmt))
        return 0
    print(os.environ.get("FAKE_SINFO_CPUS", "2/6/0/8"))
    return 0


def scontrol(args):
    if os.environ.get("FAKE_SCONTROL_FAIL"):
        print("slurm_load_node error: Unable to contact slurm controller (connect failure)", file=sys.stderr)
        return 1
    if args and args[0] == "update":
        if os.environ.get("FAKE_SCONTROL_DENY"):
            print(f"Access/permission denied for {args[1] if len(args) > 1 else 'job'}", file=sys.stderr)
            return 1
        jid = next((a.split("=", 1)[1] for a in args if a.startswith("JobId=")), None)
        if jid and (J / f"{jid}.json").is_file():
            rec = json.loads((J / f"{jid}.json").read_text())
            rec.setdefault("updates", []).append(args[1:])
            _save(jid, rec)
        return 0
    if "node" in args:
        f = os.environ.get("FAKE_SCONTROL_NODES")
        print(Path(f).read_text().rstrip("\n") if f else DEFAULT_NODE)
        return 0
    return 0


def main(argv: list[str]) -> int:
    cmd = Path(argv[0]).name
    rest = argv[1:]
    if cmd not in NAMES and rest and rest[0] in NAMES:
        cmd, rest = rest[0], rest[1:]
    _count(cmd)
    return {"sbatch": sbatch, "squeue": squeue, "scancel": scancel, "sacct": sacct, "sinfo": sinfo,
            "scontrol": scontrol}[cmd](rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

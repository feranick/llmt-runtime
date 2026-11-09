"""cgroup counters and limits (v2 and v1), node memory, job vs node out of memory, threads sized to memory, wording."""
from __future__ import annotations

import pytest

from llmt_runtime import memory as M


@pytest.fixture
def cg(tmp_path, monkeypatch):
    """A cgroup v2 tree: /sys/fs/cgroup/system.slice/job_5807 (the job) under system.slice."""
    root = tmp_path / "cgroup"
    job = root / "system.slice" / "job_5807"
    job.mkdir(parents=True)
    (tmp_path / "proc_cgroup").write_text("0::/system.slice/job_5807\n")
    monkeypatch.setattr(M, "PROC_CGROUP", tmp_path / "proc_cgroup")
    monkeypatch.setattr(M, "CGROUP_ROOT", root)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       65536000 kB\nMemFree: 1 kB\nMemAvailable:   32768000 kB\n")
    monkeypatch.setattr(M, "PROC_MEMINFO", meminfo)
    monkeypatch.setattr(M, "kernel_oom_constraint", lambda pid, since: None)
    return job


def events(d, oom=0, kill=0):
    (d / "memory.events").write_text(f"low 0\nhigh 0\nmax 3\noom {oom}\noom_kill {kill}\n")


def test_cgroup_dirs_v2_and_v1(cg, tmp_path, monkeypatch):
    assert M.cgroup_dirs() == [cg, cg.parent, cg.parent.parent]
    (tmp_path / "proc_cgroup").write_text("12:cpu,cpuacct:/x\n4:memory:/slurm/uid_1/job_9\n")
    root = tmp_path / "cgroup"
    assert M.cgroup_dirs()[0] == root / "memory" / "slurm" / "uid_1" / "job_9"
    monkeypatch.setattr(M, "PROC_CGROUP", tmp_path / "missing")
    assert M.cgroup_dirs() == []                                     # macOS


def test_counters_and_limit(cg, monkeypatch):
    events(cg, oom=1, kill=2)
    (cg / "memory.max").write_text("34359738368\n")
    (cg.parent / "memory.max").write_text("max\n")
    assert M.oom_kills() == {str(cg / "memory.events"): 2}
    assert M.oom_limit_hits() == {str(cg): 1}
    assert M.memory_limit_mb() == 32768
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4096")                 # SLURM's own number first
    assert M.memory_limit_mb() == 4096


def test_v1_limit_and_kills(tmp_path, monkeypatch):
    root = tmp_path / "cg"
    d = root / "memory" / "slurm" / "job_1"
    d.mkdir(parents=True)
    (d / "memory.limit_in_bytes").write_text(str(8 * 2 ** 30))
    (d / "memory.oom_control").write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 1\n")
    (tmp_path / "pc").write_text("4:memory:/slurm/job_1\n")
    monkeypatch.setattr(M, "PROC_CGROUP", tmp_path / "pc")
    monkeypatch.setattr(M, "CGROUP_ROOT", root)
    assert M.memory_limit_mb() == 8192
    assert list(M.oom_kills().values()) == [1]
    assert M.oom_limit_hits() == {}


def test_meminfo_and_sampler(cg):
    assert M.meminfo_mb() == (64000.0, 32000.0)
    s = M.NodeSampler()
    assert s.min_free_mb() is None
    s.sample()
    assert s.total_mb == 64000.0 and s.min_free_mb() == 32000.0


@pytest.mark.parametrize("args,scope", [
    # kills0, kills1, hits0, hits1, code, rss, limit, free, total, kernel
    (({}, {}, {}, {}, 0, 33000, 32768, 30000, 64000, None), None),
    (({}, {}, {}, {}, 1, 1000, 32768, 30000, 64000, None), None),                  # a crash, not memory
    (({}, {}, {}, {}, -9, 1000, 32768, 30000, 64000, "CONSTRAINT_MEMCG"), "job"),   # the kernel says
    (({}, {}, {}, {}, -9, 1000, 32768, 30000, 64000, "CONSTRAINT_NONE"), "node"),
    (({"e": 0}, {"e": 1}, {"c": 0}, {"c": 1}, -9, 33000, 32768, 30000, 64000, None), "job"),   # the limit was hit
    (({"e": 0}, {"e": 1}, {"c": 0}, {"c": 0}, -9, 14000, 34816, 300, 64000, None), "node"),   # killed, no limit hit
    (({"e": 0}, {"e": 1}, {}, {}, -9, 14000, 34816, 300, 64000, None), "node"),       # v1: sizes say the node
    (({"e": 0}, {"e": 1}, {}, {}, -9, 33000, 34816, 300, 64000, None), "job"),        # v1: near the job's limit
    (({}, {}, {}, {}, -9, 31000, 32768, 30000, 64000, None), "job"),                  # no counters: 90 % of the limit
    (({}, {}, {}, {}, -9, 10000, 32768, 500, 64000, None), "node"),
])
def test_oom_scope(args, scope):
    assert M.oom_scope(*args) == scope


def test_out_of_memory():
    assert M.out_of_memory(-9, {"a": 0}, {"a": 1}, 1, None)
    assert M.out_of_memory(-9, {}, {}, 30000, 32768)
    assert not M.out_of_memory(-11, {}, {}, 30000, 32768)
    assert not M.out_of_memory(0, {"a": 0}, {"a": 1}, 1, None)


def test_oom_watch_job_and_node(cg, monkeypatch):
    events(cg, 0, 0)
    (cg / "memory.max").write_text(str(32 * 2 ** 30))
    w = M.OomWatch()
    events(cg, 1, 1)
    rec = w.classify(-9, 33400, pid=123, threads=6)
    assert rec["oom_scope"] == "job" and rec["out_of_memory"] and rec["limit_mb"] == 32768
    assert M.killed_text(rec) == "killed at 32.6 GB of 32.0 GB with 6 threads"
    msg = M.oom_message(rec, "CalculiX", "use a coarser mesh")
    assert msg.startswith("CalculiX ran out of memory with 6 threads: killed at 32.6 GB (the job's limit is 32.0 GB)")
    assert msg.endswith("Ask for more memory (run.mem) or use a coarser mesh")

    w = M.OomWatch()
    (M.PROC_MEMINFO).write_text("MemTotal: 65536000 kB\nMemAvailable: 300000 kB\n")
    w.sample()
    events(cg, 1, 2)                                                # a kill, the limit not reached
    monkeypatch.setenv("SLURMD_NODENAME", "")                       # no squeue lookup
    rec = w.classify(-9, 14000, threads=2)
    assert rec["oom_scope"] == "node" and rec["node_total_mb"] == 64000 and rec["node_min_free_mb"] < 300
    msg = M.oom_message(rec, "CalculiX")
    assert "the node ran out of memory (other processes): killed at 13.7 GB, below the job's 32.0 GB" in msg
    assert "more run.mem will not help" in msg and "Ask for more memory" not in msg   # never more memory for the node


def test_oom_watch_clean_exit(cg):
    rec = M.OomWatch().classify(0, 100)
    assert rec["out_of_memory"] is False and rec["oom_scope"] is None


def test_threads_for_memory(monkeypatch):
    monkeypatch.setattr(M, "own_rss_mb", lambda: 1024.0)
    need = {1: 10.0, 6: 40.0}.__getitem__
    monkeypatch.setattr(M, "memory_limit_mb", lambda: None)
    assert M.threads_for_memory(need, 6) == (6, None)               # no limit known: as asked
    t, note = M.threads_for_memory(need, 6, 32768, what="the meshed 424,303 nodes", solver="CalculiX")
    assert t == 1 and note["threads"] == 1 and note["available_gb"] == 30.5
    assert note["message"] == ("the meshed 424,303 nodes need about 40 GB with 6 threads and 10 GB with 1; the job has "
                               "32 GB: CalculiX runs on 1 thread")
    t, note = M.threads_for_memory(lambda n: 50.0, 1, 32768, what="it")
    assert t == 1 and note["message"].endswith("it may run out of memory: ask for 51G (run.mem)")
    assert M.threads_for_memory(need, 6, 65536) == (6, None)


def test_exit_text_and_node_jobs(monkeypatch):
    assert M.exit_text(3) == "exit code 3"
    assert M.exit_text(-11) == "signal 11 (SIGSEGV)"
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    assert M.node_jobs_text(64000) == ""

"""Settings parsing, the machine (nodes, cores, memory), caps, time limits, the call budget."""
from __future__ import annotations

import math
import time

import pytest
from conftest import make_rt

from llmt_runtime import Hooks
from llmt_runtime import slurm as S
from llmt_runtime.testing import calls

TWO_NODES = ("NodeName=spark-5da5 CPUTot=20 ThreadsPerCore=1 RealMemory=120000 NodeAddr=10.0.0.1 NodeHostName=spark-5da5\n"
             "NodeName=spark-0808 CPUTot=20 ThreadsPerCore=1 RealMemory=60000 NodeAddr=10.0.0.2 NodeHostName=spark-0808\n")


def nodes_file(tmp_path, monkeypatch, text=TWO_NODES):
    f = tmp_path / "nodes.txt"
    f.write_text(text)
    monkeypatch.setenv("FAKE_SCONTROL_NODES", str(f))
    S.forget()


# ---- settings

def test_settings_defaults_and_bad_values(rt, monkeypatch):
    assert S.kind(rt) == "local"
    monkeypatch.setenv("TT_EXECUTOR", "SLURM")
    assert S.kind(rt) == "slurm"
    monkeypatch.setenv("TT_EXECUTOR", "cloud")
    assert S.kind(rt) == "local"
    assert S.min_minutes(rt) == 1.0
    monkeypatch.setenv("TT_SLURM_MIN_MINUTES", "x")
    assert S.min_minutes(rt) == 1.0
    monkeypatch.setenv("TT_PROCS", "4.0")
    assert S.default_procs(rt) == 4
    monkeypatch.setenv("TT_PROCS", "many")
    assert S.default_procs(rt) == 1
    assert S.timeout_s(rt) == 600
    assert S.timeout_s(rt, {"run": {"timeout_s": 30}}) == 30


def test_procs_and_queue(rt, monkeypatch):
    monkeypatch.setenv("TT_PROCS", "3")
    assert S.procs_of(rt, {}) == 3
    assert S.procs_of(rt, {"procs": 5}) == 5
    assert S.procs_of(rt, {"procs": 5, "run": {"procs": 7}}) == 7
    rt.hooks.study_procs = lambda st: (st.get("solve") or {}).get("procs")
    assert S.procs_of(rt, {"solve": {"procs": 9}}) == 9
    assert S.queue_of({"queue": True}) is True
    assert S.queue_of({"queue": True, "run": {"queue": False}}) is False
    assert S.use_slurm(rt, {}, 0.1) is False                # local executor
    monkeypatch.setenv("TT_EXECUTOR", "slurm")
    assert S.use_slurm(rt, {}, 0.1) is False                # below min_minutes
    assert S.use_slurm(rt, {}, None) is True                 # no estimate: SLURM
    assert S.use_slurm(rt, {"queue": True}, 0.1) is True


def test_slurm_args_shell_split(rt, monkeypatch):
    monkeypatch.setenv("TT_SLURM_ARGS", '--mem=8G --nodelist=spark-5da5 --comment="two words"')
    assert S.slurm_args(rt) == ["--mem=8G", "--nodelist=spark-5da5", "--comment=two words"]
    assert S.args_mem(rt) == "8G" and S.args_nodelist(rt) == "spark-5da5"
    monkeypatch.setenv("TT_SLURM_ARGS", '--comment="unbalanced --mem=4G')
    assert S.slurm_args(rt) == ['--comment="unbalanced', "--mem=4G"]


@pytest.mark.parametrize("args,want", [(["-w", "a"], "a"), (["-wa"], "a"), (["-w=a"], "a"), (["--nodelist", "a"], "a"),
                                       (["--nodelist=a,b"], "a,b"), (["--nodelist"], ""), (["--nice=5"], None)])
def test_opt_value_nodelist(args, want):
    assert S.opt_value(args, ("--nodelist",), ("-w",)) == want


def test_opt_value_exact_long_names():
    assert S.opt_value(["--mem-bind=local"], ("--mem",)) is None      # omct matched --mem-bind as --mem
    assert S.opt_value(["--mem-per-cpu", "2G"], ("--mem", "--mem-per-cpu")) == "2G"


def test_sbatch_args_drop_mem_only_with_run_mem(rt, monkeypatch):
    monkeypatch.setenv("TT_SLURM_ARGS", "--mem 8G --nice=10 --mem-per-cpu=1G -p long")
    assert S.sbatch_args(rt, {}) == ["--mem", "8G", "--nice=10", "--mem-per-cpu=1G", "-p", "long"]
    assert S.sbatch_args(rt, {"run": {"mem": "4G"}}) == ["--nice=10", "-p", "long"]


@pytest.mark.parametrize("v,mb", [("500M", 500), ("8G", 8192), ("8GB", 8192), ("1T", 1048576), ("2048", 2048),
                                  (" 4 G ", 4096), ("1.5G", 1536), ("lots", None), ("", None)])
def test_mem_mb(v, mb):
    assert S.mem_mb(v) == mb


@pytest.mark.parametrize("t,m", [("infinite", math.inf), ("UNLIMITED", math.inf), ("7-00:00:00", 10080),
                                 ("2:00:00", 120), ("30", 30), ("1-12", 2160), ("5:30", 5.5), ("N/A", None),
                                 ("", None), ("1:2:3:4", None)])
def test_slurm_minutes(t, m):
    assert S.slurm_minutes(t) == m


@pytest.mark.parametrize("t,m", [("1:30", 1.5), ("2:00:00", 120), ("1-00:00:00", 1440), ("INVALID", None), (None, None)])
def test_elapsed_minutes(t, m):
    assert S.elapsed_minutes(t) == m


def test_expand_nodelist():
    assert S.expand_nodelist("spark-5da5,node[01-03,7]x") == ["spark-5da5", "node01x", "node02x", "node03x", "node7x"]
    assert S.expand_nodelist("") == []


# ---- the machine

def test_cluster_nodes_cached_and_failure_retried(rt, slurm, monkeypatch):
    assert S.cluster_nodes()[0]["NodeName"] == "fake1"
    S.cluster_nodes()
    assert calls(slurm, "scontrol") == 1                          # kept NODES_TTL
    S.forget()
    monkeypatch.setenv("FAKE_SCONTROL_FAIL", "1")
    assert S.cluster_nodes() is None
    assert S.cluster_nodes() is None
    assert calls(slurm, "scontrol") == 2                          # a failure is not asked again within 5 s ...
    monkeypatch.delenv("FAKE_SCONTROL_FAIL")
    S._NODES["failed"] = time.time() - 6                           # ... but after
    assert S.cluster_nodes() is not None                           # never remembered as 'no SLURM'


def test_single_node(rt, slurm):
    assert S.node_pin(rt)[:2] == (None, None)                      # nothing to pin on one node
    assert S.node_cores(rt) == 20                                  # 40 CPUs / 2 threads per core
    assert S.node_mem_mb(rt) == 64000
    assert S.caps(rt) == {"mem": "64000M", "mem_from": "a node's memory (SLURM RealMemory)", "mem_mb": 64000}


def test_several_nodes_local_any_list_args(rt, slurm, tmp_path, monkeypatch):
    nodes_file(tmp_path, monkeypatch)
    monkeypatch.setattr(S.socket, "gethostname", lambda: "spark-0808.lan")
    add, nl, info = S.node_pin(rt)
    assert (add, nl, info["this_node"], info["cluster_nodes"]) == ("spark-0808", "spark-0808", "spark-0808", 2)
    assert S.node_mem_mb(rt) == 60000                               # the pinned node's, not the largest
    monkeypatch.setenv("TT_SLURM_NODE", "any")
    assert S.node_pin(rt)[:2] == (None, None)
    assert S.node_mem_mb(rt) == 120000
    monkeypatch.setenv("TT_SLURM_NODE", "spark-5da5")
    assert S.node_pin(rt)[:2] == ("spark-5da5", "spark-5da5")
    monkeypatch.setenv("TT_SLURM_ARGS", "-w spark-0808")
    assert S.node_pin(rt)[:2] == (None, "spark-0808")
    assert S.node_status(rt)["from_args"] is True


def test_local_not_found_warns(rt, slurm, tmp_path, monkeypatch):
    nodes_file(tmp_path, monkeypatch)
    monkeypatch.setattr(S.socket, "gethostname", lambda: "elsewhere")
    monkeypatch.setattr(S, "addresses", lambda: {"192.168.1.9"})
    add, nl, info = S.node_pin(rt)
    assert add is None and nl is None and "TT_SLURM_NODE=local" in info["warning"]


def test_this_node_by_address(tmp_path, monkeypatch):
    nodes = [{"NodeName": "a", "NodeAddr": "10.0.0.1"}, {"NodeName": "b", "NodeAddr": "10.0.0.2"}]
    monkeypatch.setattr(S.socket, "gethostname", lambda: "box")
    monkeypatch.setattr(S, "addresses", lambda: {"10.0.0.2"})
    assert S.this_node(nodes)["NodeName"] == "b"


def test_no_slurm(rt, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert S.cluster_nodes() is None
    assert S.node_cores(rt) >= 1
    assert S.caps(rt) == {}


def test_caps_max_mem_and_node(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_MAX_MEM", "16G")
    monkeypatch.setenv("TT_MAX_TIMEOUT", "3600")
    monkeypatch.setenv("TT_MAX_PROCS", "8.0")
    assert S.caps(rt) == {"timeout_s": 3600.0, "mem": "16G", "procs": 8, "mem_from": "TT_MAX_MEM", "mem_mb": 16384}
    monkeypatch.setenv("TT_MAX_MEM", "128G")                        # above the node: the node's
    c = S.caps(rt)
    assert c["mem"] == "64000M" and c["mem_from"].startswith("a node's memory")
    errs = S.run_errors(rt, {"run": {"mem": "100G", "timeout_s": 7200, "procs": 9}})
    assert len(errs) == 3 and "TT_MAX_TIMEOUT" in errs[0] and "TT_MAX_PROCS" in errs[2]
    assert S.run_errors(rt, {"run": {"mem": "lots"}}) == ["run.mem 'lots': a size such as '8G' or '500M'"]


def test_mem_plan(rt, slurm, monkeypatch):
    assert S.mem_plan(rt, {}) == {"mem": "8G", "from": "auto (8G: size unknown)"}
    rt.hooks.mem_auto = lambda st: f"{st['gb']}G"
    assert S.mem_plan(rt, {"gb": 12}) == {"mem": "12G", "from": "auto"}
    assert S.mem_plan(rt, {"gb": 100}) == {"mem": "64000M", "from": "auto", "capped_from": "100G",   # never above the node
                                           "capped_by": "a node's memory (SLURM RealMemory)"}
    assert S.mem_for(rt, {"run": {"mem": " 4G "}}) == "4G"
    monkeypatch.setenv("TT_SLURM_MEM", "none")
    assert S.mem_for(rt, {"gb": 3}) is None
    monkeypatch.setenv("TT_SLURM_MEM", "6G")
    assert S.mem_for(rt, {"gb": 3}) == "6G"
    monkeypatch.setenv("TT_SLURM_ARGS", "--mem=10G")
    assert S.mem_for(rt, {"gb": 3}) is None                         # sbatch gets it from the arguments
    assert S.given_mem_mb(rt, {}) == (10240, "TT_SLURM_ARGS --mem=10G")
    assert S.given_mem_mb(rt, {"run": {"mem": "2G"}}) == (2048, "run.mem 2G")
    monkeypatch.delenv("TT_SLURM_ARGS")
    monkeypatch.setenv("TT_SLURM_MEM", "auto")
    assert S.given_mem_mb(rt, {})[0] == 64000                       # what the machine allows


# ---- time limits

def test_time_need_default_and_hooks(rt):
    assert S.time_need_min(rt, {}) == 20                            # 600 s + 10 min
    rt.hooks.solves = lambda st: 3
    assert S.time_need_min(rt, {}) == 40
    rt.hooks.time_need_min = lambda st: 99
    assert S.time_need_min(rt, {}) == 99


def test_job_time_timeout_and_minutes(rt, monkeypatch):
    assert S.job_time(rt, {})[0] == "20"                            # rt.slurm_time_default: timeout
    monkeypatch.setenv("TT_SLURM_TIME", "300")
    assert S.job_time(rt, {}) == ("300", {"needed_min": 20, "setting": "300", "time_from": "minutes"})
    monkeypatch.setenv("TT_SLURM_TIME", "5")
    assert S.job_time(rt, {})[0] == "20"                            # at least the timeouts
    monkeypatch.setenv("TT_SLURM_TIME", "soon")
    t, info = S.job_time(rt, {})
    assert t == "20" and "is not max, timeout, estimate or minutes" in info["warning"]


def test_job_time_estimate(rt, monkeypatch):
    monkeypatch.setenv("TT_SLURM_TIME", "estimate")
    rt.hooks.time_need_min = lambda st: 190
    t, info = S.job_time(rt, {}, estimate_min=2.0)
    assert t == "20" and info["time_from"] == "estimate" and info["timeout_s"] == 600     # max(20, 3 x 2 + 10)
    t, info = S.job_time(rt, {}, estimate_min=30.0)
    assert t == "100" and info["timeout_s"] == 5400
    assert S.job_time(rt, {}, estimate_min=100.0)[0] == "190"                           # at most the timeouts
    assert S.job_time(rt, {"run": {"timeout_s": 60}}, estimate_min=2.0)[1]["time_from"] == "study"
    assert S.job_time(rt, {}, estimate_min=None)[1]["time_from"] == "timeout"
    monkeypatch.setenv("TT_SLURM_TIME_FACTOR", "2")
    monkeypatch.setenv("TT_SLURM_TIME_MIN", "5")
    assert S.job_time(rt, {}, estimate_min=1.0)[0] == "12"


def test_job_time_max(rt, slurm, monkeypatch):
    monkeypatch.setenv("TT_SLURM_TIME", "max")
    t, info = S.job_time(rt, {})                                    # sinfo does not say: the timeouts
    assert t == "20" and "unknown" in info["note"]
    monkeypatch.setenv("FAKE_SINFO_PARTITIONS", "long*|infinite;short|0:10:00")
    S.forget()
    assert S.job_time(rt, {})[0] == "UNLIMITED"
    monkeypatch.setenv("TT_SLURM_ARGS", "-p short")                  # a partition shorter than needed
    t, info = S.job_time(rt, {})
    assert t == "10" and "below what the runs may take" in info["warning"]
    monkeypatch.setenv("FAKE_SINFO_PARTITIONS", "debug*|7-00:00:00")
    monkeypatch.delenv("TT_SLURM_ARGS")
    S.forget()
    assert S.job_time(rt, {}) [0] == "10080"
    rt2 = make_rt(rt.out_root().parent, slurm_time_default="max")
    monkeypatch.delenv("TT_SLURM_TIME")
    assert S.job_time(rt2, {})[1]["setting"] == "max"


# ---- the call budget, status

def test_deadline_bounds_commands(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    (b / "slowcmd").write_text("#!/bin/sh\nsleep 5\n")
    (b / "slowcmd").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:/usr/bin:/bin")
    assert S.remaining() is None
    t0 = time.time()
    with S.deadline(2.0):
        assert 0 < S.remaining() <= 2.0
        with S.deadline(30):                                         # an inner deadline never extends the outer
            assert S.remaining() <= 2.0
        assert S.run(["slowcmd"], timeout=30) is None
    assert time.time() - t0 < 4 and S.remaining() is None
    with S.deadline(0.5):
        assert S.run(["slowcmd"]) is None                            # under a second left: not even started
    assert S.run(["no-such-command"]) is None


def test_status_key(rt, slurm):
    k = S.status_key(rt)
    assert set(k) == {"kind", "sbatch", "min_minutes", "procs", "timeout_s", "time_limit_min", "time", "slurm_time",
                      "mem", "caps", "node", "node_cores", "max_nodes", "runtime"}
    assert k["kind"] == "slurm" and k["runtime"] == rt.runtime_version and k["time"] == "timeout"


def test_two_translators_one_process(tmp_path, monkeypatch):
    a, b = make_rt(tmp_path, prefix="TT"), make_rt(tmp_path, prefix="UU", cli="uu", hooks=Hooks())
    monkeypatch.setenv("TT_SLURM_ARGS", "--mem=1G")
    monkeypatch.setenv("UU_SLURM_ARGS", "--mem=2G")
    assert (a.slurm.args_mem(), b.slurm.args_mem()) == ("1G", "2G")
    assert a.slurm.mem_mb("1G") == 1024                               # functions without rt stay plain


# ---- fixes from the review (2026.11.09.1)

def test_mem_arg_and_exact_text():
    assert (S.mem_arg("8G"), S.mem_arg("500"), S.mem_arg("8GB"), S.mem_arg("1.5G"), S.mem_arg(" 4 G "),
            S.mem_arg("lots")) == ("8G", "500", "8192M", "1536M", "4096M", None)
    assert (S.mem_text(65536), S.mem_text(64000), S.mem_text(500)) == ("64G", "64000M", "500M")


def test_mem_plan_writes_what_sbatch_takes(rt, slurm, monkeypatch):
    assert S.mem_for(rt, {"run": {"mem": "1.5G"}}) == "1536M"
    monkeypatch.setenv("TT_SLURM_MEM", "8GB")
    assert S.mem_for(rt, {}) == "8192M"


def test_core_count_from_sockets():
    assert S.node_core_count({"CPUTot": "40", "ThreadsPerCore": "2"}) == 20
    assert S.node_core_count({"CPUTot": "20", "ThreadsPerCore": "2", "Sockets": "1", "CoresPerSocket": "20"}) == 20
    assert S.node_core_count({"CPUTot": "64", "Boards": "1", "Sockets": "2", "CoresPerSocket": "16"}) == 32


def test_expand_nodelist_several_brackets():
    assert S.expand_nodelist("rack[1-2]-n[01-02]") == ["rack1-n01", "rack1-n02", "rack2-n01", "rack2-n02"]


def test_run_errors_types(rt):
    errs = S.run_errors(rt, {"run": {"procs": "many", "timeout_s": "soon"}})
    assert errs == ["run.timeout_s 'soon': seconds, a number > 0", "run.procs 'many': a whole number >= 1"]
    assert S.run_errors(rt, {"run": {"procs": 2.5}}) == ["run.procs 2.5: a whole number >= 1"]


def test_estimate_splits_the_limit_over_runs(rt, monkeypatch):
    monkeypatch.setenv("TT_SLURM_TIME", "estimate")
    rt.hooks.solves = lambda st: 2
    t, info = S.job_time(rt, {}, estimate_min=5.0)
    assert t == "25" and info["timeout_s"] == 450               # min(2 x 600 s + 10 min, 3 x 5 + 10): (25 - 10) min / 2
    rt.hooks.timeout_for_limit = lambda st, m: 123.0
    assert S.job_time(rt, {}, estimate_min=30.0)[1]["timeout_s"] == 123.0


def test_time_aliases(tmp_path, monkeypatch):
    rt = make_rt(tmp_path, time_aliases={"estimate": "timeout"}, slurm_time_default="max")
    monkeypatch.setenv("TT_SLURM_TIME", "estimate")             # feat's former 'estimate' during its migration
    assert S.job_time(rt, {}, estimate_min=1.0) == ("20", {"needed_min": 20, "setting": "timeout",
                                                            "time_from": "timeout"})


def test_status_key_survives_a_hook_that_needs_a_study(rt):
    rt.hooks.time_need_min = lambda st: st["analysis"] and 5
    assert S.status_key(rt)["time_limit_min"] == 20


PERLMUTTER_NODE = ("NodeName=nid001 CoresPerSocket=64 CPUTot=256 CPUEfctv=256 Sockets=2 ThreadsPerCore=2 "
                   "RealMemory=515100 MemSpecLimit={spec} State=IDLE Partitions=regular_milan_ss11 AvailableFeatures=cpu")


def test_mem_cap_is_what_a_job_can_get(tmp_path, rt, slurm, monkeypatch):
    """Perlmutter (RealMemory 515100M) refused --mem=480G ('Memory required by task is not available'): the cap is
    RealMemory less MemSpecLimit, within the partition's MaxMemPerNode and MaxMemPerCPU x CPUs; mem_from says which."""
    nodes, parts = tmp_path / "nodes.txt", tmp_path / "parts.txt"
    monkeypatch.setenv("FAKE_SCONTROL_NODES", str(nodes))
    monkeypatch.setenv("FAKE_SCONTROL_PARTITIONS", str(parts))

    def setup(spec, part_line):
        nodes.write_text(PERLMUTTER_NODE.format(spec=spec))
        parts.write_text(part_line)
        S.forget()

    setup(0, "PartitionName=regular_milan_ss11 Default=YES MaxMemPerNode=UNLIMITED")
    c = S.caps(rt)
    assert c["mem"] == "515100M" and c["mem_from"] == "a node's memory (SLURM RealMemory)"
    assert not S.run_errors(rt, {"run": {"mem": "480G"}})             # RealMemory alone let 480G through
    setup(30000, "PartitionName=regular_milan_ss11 Default=YES MaxMemPerNode=UNLIMITED")
    c = S.caps(rt)
    assert c["mem_mb"] == 515100 - 30000 and "MemSpecLimit" in c["mem_from"] and "515100M - 30000M" in c["mem_from"]
    assert S.run_errors(rt, {"run": {"mem": "480G"}})                 # 480G = 491520M > 485100M: refused here now
    setup(0, "PartitionName=regular_milan_ss11 Default=YES MaxMemPerNode=460000")
    c = S.caps(rt)
    assert c["mem"] == "460000M" and c["mem_from"] == "partition regular_milan_ss11's MaxMemPerNode (460000M)"
    assert S.run_errors(rt, {"run": {"mem": "480G"}})
    setup(0, "PartitionName=other Default=NO MaxMemPerNode=1000\n"
             "PartitionName=regular_milan_ss11 Default=YES MaxMemPerCPU=1900")
    c = S.caps(rt)
    assert c["mem_mb"] == 1900 * 256 and c["mem_from"] == "partition regular_milan_ss11's MaxMemPerCPU (1900M) x 256 CPUs"
    monkeypatch.setenv("TT_SLURM_ARGS", "-p other")                  # the job's partition, not the default one
    assert S.caps(rt)["mem_from"] == "partition other's MaxMemPerNode (1000M)"
    monkeypatch.delenv("TT_SLURM_ARGS")
    rt.hooks.mem_auto = lambda st: "600G"                              # mem_plan's cap is the same, and says why
    plan = S.mem_plan(rt, {})
    assert plan["mem"] == S.mem_text(1900 * 256) == "475G" and plan["capped_from"] == "600G" and "MaxMemPerCPU" in plan["capped_by"]
    monkeypatch.setenv("FAKE_SCONTROL_FAIL", "1")                     # no partitions known: the node's own limit
    S._PARTS.update(t=0.0, v=None, failed=0.0)
    assert S.job_partition(rt) is None

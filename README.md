# llmt-runtime

The code the llm-translators share. llm-translators' `docs/EXECUTION.md` describes a contract, and this package implements it, so a fix made here reaches every translator:

- worker processes
- memory and out-of-memory handling
- the SLURM executor
- request folders
- the MCP server's handling of stale installs

Standard library only, Python ≥ 3.10, typed.

| key | translator | prefix | kind |
|---|---|---|---|
| feat | fea-translator | `FEAT_` | threads (`--cpus-per-task`) |
| cfdt | cfd-translator | `CFDT_` | MPI (`--ntasks`) |
| lmpt | lammps-translator | `LMPT_` | MPI (`--ntasks`) |
| omct | openmodelica-translator | `OMCT_` | threads (sweep points) |

build123d-translator has no executor. It could use `server` only.

## Install

**Floors only.** Each translator declares `llmt-runtime>=<version>` with no upper bound, and the runtime stays backward compatible: it only adds names, keys and hooks. A change that breaks the API needs a new floor in every translator that uses it, all in the same release, and `llmt deploy` refuses to build a release until the floors agree.

- **Inside llm-translators:** llmt-runtime is a submodule, pinned like the translators. It is installed in the same pip command as the translators, first:

  ```
  pip install -e <runtime> -e <feat> -e <cfdt> ...
  ```

  pip then satisfies each floor with the local copy and never looks online. pip normalises versions (`2026.11.09.2` is read as `2026.11.9.2`), so `>=` works with the `YYYY.MM.DD.x` scheme.
- **Standalone:** from a translator's checkout, with llmt-runtime next to it:

  ```
  pip install -e ../llmt-runtime && pip install -e .[test]
  ```

  On a machine without the umbrella checkout, use a translator's optional `requirements-standalone.txt`, which installs the runtime from its git URL.
- **The package's own tests:** `pip install -e .[test] && pytest`.

## Use

```python
from llmt_runtime import Hooks, Runtime
from . import __version__

RT = Runtime(prefix="FEAT", cli="feat", package="fea_translator", version=__version__, kind="threads",
             out_default="~/feat_runs", min_minutes=1, timeout_default=900, slurm_time_default="max",
             hooks=Hooks(procs=executor.procs_for, mem_auto=executor.mem_auto, time_need_min=executor.time_limit_min,
                         job_extra=executor.job_extra, oom_advice=executor.oom_advice,
                         timeout_hint="or coarsen the mesh / drop converge", solver_left_s=api._feat_left_s))

RT.slurm.caps()                       # functions that take the Runtime first are bound: slurm.caps(RT)
RT.jobs.submit(request, name, study)
RT.jobs.job_states(request, names)
RT.requests.ran_with(meta, results)
```

Each Runtime reads its own environment variables (`FEAT_SLURM_ARGS`, `CFDT_MAX_MEM`, …) at each call. Two translators in one process (in tests, or in the router) don't share settings. While a translator migrates, it keeps thin wrappers under its old function names, so its callers and tests keep working.

### Modules

| module | what |
|---|---|
| `config` | `Runtime` (settings prefix, CLI, package, version, kind, defaults, aliases) and `Hooks` |
| `workers` | `watch_parent(ppid, name, term)`, `track`/`untrack`, `on_orphan`, `end_children`, `on_signals` (run-study), `stop_pool`/`stop_executor`/`stop_process` (SIGTERM, then SIGKILL after a grace period), `shutdown`, `run_bounded` |
| `memory` | `memory_limit_mb`, `oom_kills`, `oom_limit_hits`, `cgroup_dirs` (v2 and v1), `meminfo_mb`, `NodeSampler`, `kernel_oom_constraint`, `oom_scope` (job / node / None), `OomWatch` (around one solver run), `threads_for_memory` (the translator supplies the need), `node_jobs_text`, wording: `killed_text`, `node_oom_text`, `oom_message`, `exit_text` |
| `slurm` | settings (`kind`, `min_minutes`, `timeout_s`, `procs_of`, `job_procs`, `queue_of`, `use_slurm`, `slurm_args`, `opt_value`, `args_mem`, `args_nodelist`, `sbatch_args`), sizes and times (`mem_mb`, `mem_text`, `mem_arg`, `slurm_minutes`, `elapsed_minutes`), the machine (`cluster_nodes`, `this_node`, `expand_nodelist`, `node_pin`, `node_status`, `job_nodes`, `node_cores`, `node_mem_mb`), `caps`, `run_errors`, `mem_plan`/`mem_for`/`given_mem_mb`, time (`time_need_min`, `partition_max_min`, `job_time`, `timeout_for_limit`, `job_time_left_min`, `extend_job`), the call budget (`deadline`, `remaining`, `run`), `status_key` |
| `jobs` | `job_script`, `submit` (`job.sh` and `job.json`; `after=` dependencies; a slow sbatch whose job went through is adopted), `failed_submission`, `squeue`, `pending_jobs`/`ahead_of` (by priority), `sacct`/`sacct_rows`, `ended_state`, `job_states`, `queue_summary`, `list_jobs` (the `jobs()` tool), `cancel_request`, `extend_request` + `take_extension`/`extended_s`, `job_timeout_s`, `write_atomic`, `find_job` |
| `requests` | `new_request`, `check_request_name` (new / repeat / error) + `REPEAT_NOTE`, `request_meta` (records `translator` and `runtime`), `ran_with`, `read_file`, `under_root`, `image_path`, `clock`, `slurm_start`, `timing_slurm`, `timing_local`, `until`, `write_json` |
| `server` | `stale_reason`, `Server(rt, package_file, busy, shutdown)`: `.tool(ToolError)` (a worker thread, a 48 s deadline, errors as ToolError with their text, call logging, the `server_stale` note), `.serve()` (the install watch, the log, signals), `.enter_call`/`.leave_call`, `.start_background`, `.note` |
| `testing` | the merged fake SLURM (`fakeslurm.py`), `install_fake_slurm`, `fake_slurm(tmp_path, monkeypatch)`, `calls` |

### Hooks (what differs between translators)

| hook | default | used for |
|---|---|---|
| `study_procs(study)` | `study["procs"]` | the study's own ranks/threads (lmpt, cfdt: `solve.procs`) |
| `procs(study)` | `procs_of` capped at the node's cores | what the job asks for after size caps (feat nodes, cfdt cells, lmpt atoms, omct sweep) |
| `mem_auto(study)` | `rt.mem_default` (8G) | `P_SLURM_MEM=auto` (never above the node's memory) |
| `time_need_min(study)` | `ceil(solves × timeout_s / 60) + 10` | the job's limit with `timeout` |
| `solves(study)` | 1 | runs per study (feat joule, converge; omct sweep points) |
| `timeout_for_limit(study, min)` | `(min − 10) × 60 / solves` | each run's timeout under `estimate` |
| `resource_lines(study, procs)` | by kind | `#SBATCH` resource lines |
| `script_lines(request, name, study)` | none | shell lines before `exec` (omct: `export PYTHONPATH`) |
| `job_extra(study)` | none | `job.json` keys (feat: `nodes_estimate`, `memory`; lmpt: `natoms`; cfdt: `cells_estimate`) |
| `oom_advice(folder)` | generic | what to ask for after a SLURM out-of-memory kill (feat: from the meshed nodes) |
| `timeout_hint` | `""` | appended to the timed-out advice |
| `result_file(request, name)` | `<request>/<name>/<name>.json` | a finished study's result |
| `running_progress(folder, study)` | none | progress keys in `jobs()` |
| `solver_left_s(folder, study)` | 0 | how far the running solve's deadline is, for `extend` |

## Settings (P = prefix)

| variable | values (default) |
|---|---|
| `P_EXECUTOR` | `local` \| `slurm` (`local`) |
| `P_SLURM_MIN_MINUTES` | number (`rt.min_minutes`) |
| `P_SLURM_ARGS` | sbatch arguments, shell-split. A `--mem` here wins over `P_SLURM_MEM`, a `--nodelist`/`-w` over `P_SLURM_NODE`, and `--partition`/`-p` picks the partition whose limit `max` uses |
| `P_SLURM_NODE` | `local` \| `any` \| a node list (`local`) |
| `P_SLURM_MEM` | `auto` \| a size \| `none` (`auto`) |
| `P_SLURM_TIME` | `max` \| `timeout` \| `estimate` \| minutes (`rt.slurm_time_default`) |
| `P_SLURM_TIME_FACTOR`, `P_SLURM_TIME_MIN` | 3, 20 (for `estimate`) |
| `P_PROCS`, `P_TIMEOUT` | 1, `rt.timeout_default` |
| `P_MAX_TIMEOUT`, `P_MAX_MEM`, `P_MAX_PROCS` | caps on a study's `run` (unset: no cap) |
| `P_OUT` | the output root (`rt.out_default`) |
| `P_EXIT_WHEN_STALE` | `1` \| `0` (default: shared installs only) |
| `P_STALE_CHECK_S` | s (`rt.stale_period`, 10) |

### `P_SLURM_TIME`: one set of meanings for all translators

- **`timeout`:** the runs' timeouts, their margins and 10 minutes (`hooks.time_need_min`).
- **`estimate`:** `max(P_SLURM_TIME_MIN, ceil(P_SLURM_TIME_FACTOR × estimate) + 10)` minutes, at most `timeout`.
  - Each run's own timeout then fits inside that limit: `job.json` `timeout_s`, which run-study reads with `jobs.job_timeout_s`.
  - A study with `run.timeout_s`, or without an estimate, uses `timeout`.
- **`max`:** the partition's own limit, or `UNLIMITED` where the partition has none.
  - Why it exists: SLURM lets only an admin raise a running job's limit (mochi job 3177, `Access/permission denied`). With `max`, `extend` needs no admin.
  - **The price:** SLURM cannot plan a start behind an `UNLIMITED` job that holds a node's memory, so the router then sees that machine as busy indefinitely.
- **minutes:** that many, but at least `timeout`.

feat 2026.11.09.5 used `estimate` to mean what is now `timeout`. To keep existing `FEAT_SLURM_TIME=estimate` settings working during feat's migration, pass `Runtime(time_aliases={"estimate": "timeout"})`, then drop the alias once `config/env` files are updated.

## Inventory: where each piece was, and what was taken

**Item 1: worker processes**
- feat `workers.py` (11.03.1): `watch_parent`, `track`.
- cfdt `workers.py`: `foam.kill_all`, no `track`.
- lmpt `workers.py`: `lammps.kill_all`. It does have the parent watch; what it lacked was a pool stop when the server exits cleanly.
- omct `workers.py`: same as feat; all its pools watched.
- **Taken:** feat's version, plus `on_orphan` for cfdt's and lmpt's own run lists, plus SIGTERM ending the tracked runs (new), plus escalation to SIGKILL.

**Item 2: memory and out of memory**
- feat `runner.py` (11.05.3–11.08.1).
- cfdt: none. Its `foam.was_killed` gives a memory hint, but no `out_of_memory` state.
- lmpt: none. A SIGKILL is reported as "timed out" (a bug).
- omct: none.
- **Taken:** feat's version, generalised (`OomWatch`, `threads_for_memory` with the need supplied by the translator).

**Item 3: settings and the machine**
- feat `executor.py` (11.09.5).
- cfdt: `sinfo %m` node memory. It raises on unbalanced quotes, has no `slurm_minutes`, and no call deadline.
- lmpt: failed `scontrol` cached for 5 s, `node[1-3]` expansion, `P_SLURM_TIME=estimate`. No deadline.
- omct: node memory cap done. `--mem-bind` was taken for `--mem`; `int("4.0")` crashes. No deadline.
- **Taken:** feat's version, plus lmpt's 5 s failure cache, node-list expansion and `estimate`, plus cores from Sockets×CoresPerSocket, plus exact memory sizes.

**Item 4: jobs**
- feat (`job.sh` with `--mem`; `job_states` with one squeue).
- cfdt: `--dependency`, atomic writes, lost-node diagnosis, and a better queue; `--mem` only on sbatch's command line.
- lmpt: queue position by priority, background submission; one sacct per job.
- omct: sbatch breaker, `PYTHONPATH` line, lost-node diagnosis.
- **Taken:** feat's version, plus cfdt's `after=` and `write_atomic`, plus the lost-node diagnosis (cfdt, omct), plus queue position by priority (lmpt, cfdt), plus omct's sbatch breaker.

**Item 5: requests**
- feat (`ran_with` 11.09.4, clock times, `time_left`).
- cfdt: no `translator` in `request.json`; status reports the installed version.
- lmpt: version recorded but not reported; `run_studies` stays within 60 % of `wait_s`.
- omct: no version recorded; no `time_left`.
- **Taken:** feat's version, plus lmpt's `FileExistsError` race and its `submitting` state.

**Item 6: server**
- feat (10 s period, 48 s deadline).
- cfdt (20 s period, no deadline).
- lmpt `stale.py` (20 s period, `LMPT_STALE_CHECK_S`, no deadline).
- omct (20 s period, server log, signal handlers, no deadline).
- **Taken:** feat's version, plus omct's log and signal handlers, plus lmpt's `P_STALE_CHECK_S`, plus `start_background`, plus the runtime's own version counted in staleness.

**Fake SLURM**
- feat: `FAKE_*` variables.
- cfdt: `FAKE_NODES`, `%Q`/`%N`.
- lmpt: knobs as files, a call counter.
- omct: no `scontrol`.
- **Taken:** one script with all of them (see its docstring), plus `FAKE_MIN_JOB_AGE` and `FAKE_SBATCH_SLOW_AFTER`.

`tool_status().executor` keys:
- feat: `kind, sbatch, min_minutes, procs, timeout_s, time_limit_min, slurm_time, mem, caps, node, node_cores`.
- cfdt adds `args, nodes, cores_shared`, and drops keys in local mode.
- lmpt adds `slurm_args, workers, atoms_per_rank, run_keys, time`.
- omct adds `time, args, nodes, compile_procs, workers`.

`status_key()` gives feat's keys plus `time` (the contract's name) and `runtime`. Each translator adds its own keys on top.

### Fixed on the way (these change what a translator reports, by design)

**Out of memory**
- SLURM's out-of-memory wording (`Detected 2 oom_kill events in StepId=…`) is matched with `oom[-_ ]kill|out of memory` in every case, whatever sacct says. cfdt and omct matched only `oom-kill`, and only without accounting, so these jobs came back as `failed`.

**Status and states**
- Status reports the version a request ran with (`ran_with`). cfdt, lmpt and omct reported the installed version.
- An ended job that squeue still lists (for MinJobAge, 300 s) is read as ended, not running. feat read every state except PENDING as running, so it showed a job that had ended without a result as running for 5 minutes.
- `failed` without `job.log` says it never started. "Went to another node" is said only on clusters of more than one node.

**Memory sizes**
- `--mem` goes in `job.sh` (cfdt and omct passed it only to sbatch, so `sbatch job.sh` lost it).
- `--mem` values are written the way sbatch accepts them: `8GB` and `1.5G` become MB.
- A node's memory is given exactly (`64000M`, not `62G`), so a `run.mem` of 64000M is no longer refused.
- `auto` memory is never above the node's.

**Counting and placement**
- `queue_summary` counts each node once. Summing sinfo's per-partition rows counted shared nodes twice.
- `-c` in `P_SLURM_ARGS` is respected for MPI placement.

**Submission**
- A job that sbatch submitted but answered too late for is found by its name. Before, it ran with no `job.json` and could not be cancelled.

**Server and workers**
- Stale exit:
  - Every server checks every 10 s (cfdt and omct checked every 20 s).
  - Work started with `start_background` holds the exit off (lmpt's submit thread could be killed).
  - The exit stops its workers for at most 10 s, then exits whatever happens.
- Workers:
  - A worker given SIGTERM ends its tracked solver runs. Before, the runs lived on in their own process groups.
  - Stopping a pool escalates to SIGKILL after 3 s.
- `watch_parent` also watches the server's pid, so it works with the forkserver start method (the default on Linux from Python 3.14).

**Validation**
- `run_errors` reports a non-numeric `run.procs` or `run.timeout_s` instead of crashing.

## Moving the translators onto it (after review)

Order: feat, cfdt, lmpt, omct, one step each. After each step, the translator's full test suite passes, both repos have a version bump, the README rows are added, and the commit command is given.

1. **Phase 1:**
   - `workers` (each translator's `workers.py` becomes a re-export).
   - `memory` (feat `runner.py` keeps `run_ccx`'s restart policy).
   - `ended_state`, `request_meta`/`ran_with`, `check_request_name`.
2. **Phase 2:**
   - The rest of `slurm` and `jobs`, with the hooks listed above.
   - `extend` for all translators, once each solver consumes the `extend` file with `jobs.take_extension`.
3. **Phase 3:** `server.Server` replaces the four stale-handling copies.

Each translator's tests use `llmt_runtime.testing.fake_slurm`. Their own fakes are removed, and their `FAKE_PENDING` and file-based knobs map to the merged variables.

## Proposed changes to llm-translators (not made; for approval)

**`.gitmodules` and `translators.json`**
- Add `llmt-runtime` as a submodule.
- Add `"runtime": {"repo": "llmt-runtime", "package": "llmt_runtime"}` at the top level of `translators.json`. It is not a translator key, so the router ignores it.

**`llmt deploy`**
- Archive `llmt-runtime` with the translators.
- Install it in the same pip command, first: `pip install -e <rel>/llmt-runtime -e <rel>/fea-translator[...] ...`.
- Before building, read each translator's `pyproject.toml` floor (`llmt-runtime>=X`) and compare it with the pinned runtime's version, normalised as pip does (`2026.11.09.2` = `2026.11.9.2`). Refuse with `feat needs llmt-runtime>=X; the pin is Y`.
- Write `"runtime": "<version> (<commit>)"` in `RELEASE.json`.

**Other `llmt` commands**
- **`llmt update` and `llmt install`:** the same order and the same floor check.
- **`llmt pin`:** moves the runtime's pin like a translator's, and warns when a translator's floor is above the new pin.
- **`llmt status` and `machines()`:** show `executor.runtime` from each translator's `tool_status()`, and flag two different runtime versions in one release. That cannot happen with one venv, but it can with a stale server.

**Docs**
- **`EXECUTION.md`:**
  - `P_SLURM_TIME`: `max | timeout | estimate | minutes`, with the trade-off of `max`.
  - The out-of-memory wording.
  - Ended jobs in squeue.
  - `executor.runtime`.
  - "Implemented by llmt-runtime X" in place of the per-translator status lines.
- **`ROUTER.md`:** `caps.mem` is now exact (`64000M`); `caps.mem_mb` is unchanged.
- **`SHARED.md`:** the runtime in a release, and the floor check.
- **`create_new_translators.md`:** a new translator builds a `Runtime` and hooks instead of copying `executor.py`.

**Each translator**
- `pyproject.toml`: `dependencies += ["llmt-runtime>=2026.11.09.1"]`.
- README: the standalone install line, and the line "The executor comes from llmt-runtime (version ≥ …)".
- Optionally, a `requirements-standalone.txt` with `llmt-runtime @ git+https://github.com/feranick/llmt-runtime@2026.11.09.1`.

## Tests

`pytest` runs 151 tests in about 15 s, with the merged fake SLURM. They cover:

- **Settings and machine:** settings parsing; caps and node memory (one node, two nodes, `local`/`any`/list/args, this node by name or address).
- **Time limits:** partition limits, `UNLIMITED`, a partition shorter than needed, `estimate` (split across runs), minutes, aliases.
- **Jobs:**
  - submit, then queued, running and done, with the result collected by a new process.
  - Every failure state from sacct and from `job.log`, including the `oom_kill events` wording.
  - Ended jobs still in squeue; a lost node; a cancel flag.
- **Submission:** sbatch refusing, hanging, and taking a job but answering late.
- **`extend`:** not needed, raised, refused (`Access/permission denied`), capped.
- **Memory:** job vs node out of memory (cgroup v2 and v1, counters, the kernel's word, sizes); threads cut for memory.
- **Workers:** a server killed with `-9` takes its worker and the solver with it; SIGTERM; escalation to SIGKILL; a watched server pid.
- **Server and requests:** stale releases, checkouts and runtime versions; refusing only when idle; background work; an exit that cannot hang; the tool wrapper and the log; `request_name` repeats and races; version recording.

## Versions

| version | what changed and why |
|---|---|
| 2026.11.09.2 | `memory.threads_for_memory(..., own_mb=)`: the translator may give its own measure of its resident memory (feat measures itself, and its tests fix it at 0 to check the arithmetic); default as before, `own_rss_mb()`. Found moving feat onto the runtime (phase 1). Additive: floors at 2026.11.09.1 still hold. |
| 2026.11.09.1 | First version. The workers, memory, SLURM, jobs, requests and server code of fea-translator 2026.11.09.5, merged with the better pieces of cfd-translator 2026.11.08.2, lammps-translator 2026.11.7.1 and openmodelica-translator 2026.11.07.1 (see "Inventory"), with the fixes listed above. Findings behind them: carbonio jobs 5797 and 5807 (out of memory taken for a crash; a kill of the whole batch step), the node's kills at 20.1 and 13.9 GB in 34 GB jobs, mochi job 3177 (SLURM refused a raise of a running job's limit), a 2026.11.09.2 bench that read 2026.11.09.3. The merged fake SLURM comes from the four translators' fakes. |

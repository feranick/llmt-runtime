# Prompts for the translators' dev chats: what changed with llmt-runtime

Paste the common part, then the translator's own section, at the start of that translator's chat. This was written 2026-11-09, after phases 1–3. When the runtime changes how translators use it, update this file.

---

## Common part (every translator)

Since 2026-11-09 this translator runs on **llmt-runtime**, the package that holds the code the four translators used to copy:

- **Repository:** `~/Software/llmt-runtime`, package `llmt_runtime`, a submodule of llm-translators.
- **Versions at this writing:** runtime 2026.11.09.6; feat 2026.11.09.10, cfdt 2026.11.09.6, lmpt 2026.11.09.5, omct 2026.11.09.3.
- **Read first:** its `README.md` (API, hooks, settings, the versions table) and `PLANNING.md` (what is still open).

**What is the runtime's now** (do not copy or re-implement it here):

- **Workers** (`llmt_runtime.workers`): the parent watch, tracking solver runs, SIGTERM ending them, and stopping pools (SIGTERM, then SIGKILL).
- **Memory and out of memory** (`memory`): cgroup counters, the job's limit, the node's memory, job vs node (`OomWatch`, `oom_scope`), `threads_for_memory`, and the wording.
- **The SLURM layer** (`slurm`, `jobs`):
  - Settings (`P_EXECUTOR`, `P_SLURM_ARGS`, `P_SLURM_NODE`, `P_SLURM_MEM`, `P_SLURM_TIME` = `max | timeout | estimate | minutes`, caps).
  - The machine (nodes, pinning, cores, node memory), time limits, and the call deadline (`slurm.deadline`, `remaining`, `run`).
  - `job.sh` and `job.json`, `submit` (dependencies, adopting a late sbatch's job), states (`job_states`, `ended_state` / `classify_end`), `queue_summary`, `list_jobs`, `cancel_request`, `extend_request`, `take_extension`.
- **Requests** (`requests`): request folders, `request_name`, `request_meta` (records `translator` and `runtime`), `ran_with`, `read_file`, clock times, `timing_slurm` / `timing_local`, `until`.
- **The MCP server** (`server.Server`): stale installs, the tool wrapper (a worker thread, the 48 s deadline, ToolError with the message, the `server_stale` note), the server log in `$P_OUT/.server`, signal handling, `start_background`.
- **The fake SLURM for tests** (`llmt_runtime.testing.fake_slurm`, `FAKE_*` knobs).

**How this translator uses it:**

- `<package>/runtime.py` makes one `RT = Runtime(prefix=..., cli=..., package=..., kind=..., hooks=Hooks(...))`. Its sizing reaches the runtime only through the hooks (procs, mem_auto, solves / time_need_min, job_extra, script_lines, running_progress, oom_advice, timeout_hint, solver_left_s). The hooks are small lazy wrappers, which avoids circular imports.
- `executor.py` keeps its old public names as thin wrappers over `RT.slurm.*` / `RT.jobs.*`. Callers and tests use those names; keep them.
- `mcp_server.py` makes one `GUARD = Server(RT, api.__file__, busy=..., shutdown=...)` and wraps every tool with `@GUARD.tool(ToolError)`; `main()` calls `GUARD.serve()`. The old stale-handling names stay as wrappers over `GUARD`.

**Rules for changes from now on:**

1. **Shared behaviour is fixed once, in llmt-runtime.** That covers anything about SLURM, job states, memory or out of memory, workers, requests or the MCP server's common code: never patch a translator's wrapper around it. Keep runtime changes backward compatible (additions only). Bump the runtime's version (`__init__.py`, `pyproject.toml`, the changed modules' `Version:` lines, a README row), add a test in its `tests/`, and tag the release (`git tag <version> && git push --tags`).
2. **Floors only.** This translator declares `llmt-runtime>=X` in `pyproject.toml` (also in `requirements-standalone.txt`'s git tag and the README install line). Raise it only when it needs something newer; never set an upper bound.
3. **Order of pins.** First push the runtime, then the translator, then `./llmt pin runtime <key> && git add -A && git commit -m "Pin …" && git push` in llm-translators. `llmt deploy` refuses to build if a floor is above the pinned runtime.
4. **What stays here:** sizing (cells, atoms, nodes, sweep points), the solver and its progress and time left, decks, meshing, results, rendering, benchmark content, and this translator's advice texts (given through hooks).
5. **Tests use the runtime's fake SLURM:** `from llmt_runtime.testing import fake_slurm`, the `FAKE_*` knobs in `llmt_runtime/testing/fakeslurm.py`, and job records with `state`, `final`, `sacct`, `vanished`. Patch the runtime's names where the runtime's code runs (`llmt_runtime.slurm.cluster_nodes`, `.addresses`, `.forget()`), not this translator's wrappers. A test helper taking a folder plus `**kwargs` must not call the folder parameter `state`. A test of the MCP server's answers must pin `GUARD.stale` to None: in a release that is not yet current, the real check finds the server stale.
6. **Standalone install:** `pip install -e ../llmt-runtime && pip install -e .[test]` (or `pip install -r requirements-standalone.txt`). In llm-translators, llmt installs the pinned runtime first.
7. **Before users get a release** (on a server, in tmux):
   ```
   sudo ./llmt deploy --no-switch
   R=/opt/llm-translators/releases/$(ls /opt/llm-translators/releases | sort -V | tail -1)
   $R/llmt test runtime <key>                # all five when the runtime changed
   ```
   Then list the MCP server's tools, keeping its input open (`( printf '<initialize>\n<initialized>\n<tools/list>\n'; sleep 30 ) | $R/bin/mcp <key>`). Then `sudo ./llmt rollback $(basename $R)`.
8. **The user's rules still hold:**
   - Versions are `YYYY.MM.DD.x`, two-digit month and day, bumped in `__init__.py`, `pyproject.toml` and the `Version:` line of every changed file, with a README Versions row saying what and why (with the run or job that showed it).
   - General code, not special cases.
   - Concise answers, with paste-ready commands, one `git add -A && git commit -m "<version>: <what>" && git push` line per repo.
   - The user commits and pushes.
   - Long tests on servers run in tmux.
   - When a file becomes a stub, delete it with `git rm -f`.

**Behaviour every translator got from the runtime** (users may ask about these):

- **Job states:**
  - An ended job that squeue still lists (MinJobAge) is no longer "running".
  - SLURM's `Detected N oom_kill events` gives `out_of_memory` whatever sacct says.
  - A job with neither accounting nor `job.log` never started: `cancelled`.
  - An empty `job.log` says so.
- **Job resources:**
  - `--mem` is written inside `job.sh` and in the form sbatch accepts.
  - `caps.mem` is a node's memory exactly (`64000M`).
  - Each node's CPUs are counted once.
  - Queue position is by priority.
- **Submission:** a job that sbatch accepted but answered too late for is found by name and adopted.
- **MCP server:**
  - Every tool runs under the 48 s deadline.
  - Commands get `/dev/null` as stdin.
  - The stale check runs every 10 s (`P_STALE_CHECK_S`).
  - The stale exit stops workers for at most 10 s.
  - The server log is in `$P_OUT/.server/`.
- **Status:** `tool_status().executor.runtime`, and `study_status` reports the version a request ran with (`translator_now` when the server runs another).

---

## fea-translator (feat)

- **Hooks on `RT`:**
  - `procs=procs_for`, `mem_auto`, `time_need_min=time_limit_min`, `solves=solves_of`.
  - `job_extra`: `nodes_estimate`, `nodes_estimate_raw`, `equation_solver`, `memory`.
  - `oom_advice`, `timeout_hint`, `running_progress=api.job_progress`, `solver_left_s=api._feat_left_s`.
  - `slurm_time_default="max"`.
- **`time_aliases={"estimate": "timeout"}`:** feat's old `FEAT_SLURM_TIME=estimate` keeps its meaning (the runs' timeouts + meshing + 10 min). Drop the alias once no `config/env` sets it (PLANNING.md, section 1).
- **`extend` is feat's:** the running CalculiX side uses `llmt_runtime.jobs.take_extension`; the request side is `RT.jobs.extend_request`.
- **Memory:** `runner.py` keeps feat's own memory model (`executor.mem_gb`) and its policy of restarting a threaded run on one thread. The counters and wording come from `llmt_runtime.memory`, and the runner's names (`oom_kills`, `memory_limit_mb`, …) are aliases of them.
- **Tests:**
  - `tests/fake_slurm` is gone; tests use the runtime's fake.
  - `tests/test_runtime.py` covers the three phases.
- **Open:** nothing feat-specific beyond the alias.

## cfd-translator (cfdt)

- **Hooks on `RT`:**
  - `study_procs` = `solve.procs`.
  - `procs`: `procs_for_job` with the cells cap.
  - `mem_auto`: 2 GB + 4 GB per million cells.
  - `job_extra`: `cells_estimate`.
  - `running_progress`, `oom_advice`, `timeout_hint`.
  - `one_node=True`, `min_minutes=0`, `slurm_time_default="timeout"`.
- **Stays cfdt's:**
  - `eta_start` / `running_etas` (a queued job's start, from running jobs' own ETAs).
  - The per-node queue position (`pending_ahead`) and transient segments (dependencies through `RT.jobs.submit(..., after=[ids])`).
  - `queue_summary`'s extras (`cluster_queue`, `starts_now`, `pending_for_node`), `cores_shared` (cpushare), `foam.kill_all` as the workers' `on_orphan` cleanup, and failures written as the study's result.
- **New since the migration:**
  - `CFDT_SLURM_TIME` (default `timeout`, as before).
  - A benchmark case sent to SLURM is waited for, never also run locally.
  - An interrupted `cfdt bench` / `cfdt run` lists the queued jobs and exits 130.
  - An OpenFOAM run killed for memory has `state: out_of_memory`.
- **Behaviour to keep in mind:** an explicit `CFDT_SLURM_MEM` size is no longer cut to `CFDT_MAX_MEM` (PLANNING.md, section 1).
- **Open:**
  - Two tests fail only on macOS: tool errors (SDK version) and the Ahmed snappy benchmark.
  - The bench test depends on timing.
  - The extra `squeue` call for `pending_ahead`.
  - `extend` (PLANNING.md, section 2).

## lammps-translator (lmpt)

- **Hooks on `RT`:**
  - `study_procs` = the top-level `procs` (not `solve.procs`).
  - `procs`: `ranks_for` over atoms / `LMPT_ATOMS_PER_RANK` and node cores.
  - `mem_auto`: 1 GB + 256 MB per rank + 4 kB per atom.
  - `job_extra`: `natoms`, `estimate_min`, `procs_requested`.
  - `running_progress=api.progress`, `oom_advice`, `timeout_hint`.
  - `cancel_file="cancelled.json"`, `timeout_default=21600`, `slurm_time_default="timeout"`.
- **Stays lmpt's:**
  - `study_state` over one `queue()`, and `final_result`.
  - The budgeted submission in `api.start` (60 % of the call's remaining time; the rest in `api.background` → `GUARD.start_background` when serving, with a `submitting` state). Likewise bench's fea-translator chain thread.
  - `stale.py` is now a thin module over `GUARD`.
- **New since the migration:**
  - A LAMMPS run killed for memory is `out_of_memory` (job or node), no longer "timed out".
  - `LMPT_SLURM_TIME` gains `max` and minutes; with `estimate`, `run_one` reads the shorter timeout from `job.json` (`jobs.job_timeout_s`).
  - An interrupted `lmpt bench` / `lmpt run` lists the queued jobs and exits 130.
  - Versions now use two-digit months and days.
- **Open:**
  - The `Pool.__del__` noise at the end of the suite (a test leaves a pool for exit).
  - The tests assume `$USER` is set.
  - One real lmpt SLURM bench on an idle node.
  - `extend`.

## openmodelica-translator (omct)

- **Hooks on `RT`:**
  - `procs=job_cpus`: sweep points at once, the compile CPUs on a cache miss, capped at the node.
  - `mem_auto`: 1 GB + 256 MB per CPU.
  - `solves=sweep_rounds`: the first point alone, then the rest in groups of the job's CPUs, so a sweep's time limit covers all its rounds.
  - `script_lines`: `export PYTHONPATH=<package root>…`.
  - `job_extra`: `estimate_min`, `time_min`.
  - `running_progress`, `oom_advice`, `timeout_hint`, `slurm_time_default="timeout"`.
- **Stays omct's:**
  - Sizing (counting CPUs, not cores), estimates, `run_errors` / `caps`, and the sweep pools.
  - `run_study_job`, failures written as `<name>.json`, and `jobs()`' shape with a per-user queue count.
  - `slog` is now an alias of `GUARD.log`.
- **New since the migration:**
  - A sweep's SLURM limit covers all its rounds (12 points on 4 CPUs: 50 min, not 20).
  - `OMCT_SLURM_TIME` gains `max` and minutes; with `estimate`, each run's timeout fits the limit.
  - `--mem` is inside `job.sh`.
  - `--mem-bind` in the args is no longer taken for `--mem`.
  - Every tool has the 48 s deadline.
- **Open:**
  - The log's argument summary (the runtime logs `name=value`; omct logged study names and flags).
  - The `list_jobs` shape.
  - `extend`.

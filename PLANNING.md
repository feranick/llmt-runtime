# llmt-runtime: planning

Status on 2026-11-09. All three phases are done for the four translators:

| translator | version | what runs from llmt-runtime |
|---|---|---|
| fea-translator (feat) | 2026.11.09.10 | phases 1–3 |
| cfd-translator (cfdt) | 2026.11.09.6 | phases 1–3 |
| lammps-translator (lmpt) | 2026.11.09.6 | phases 1–3 |
| openmodelica-translator (omct) | 2026.11.09.3 | phases 1–3 |

The runtime is at 2026.11.09.7 (per-node `ahead_of`); llm-translators pinned 2026.11.09.6 until it is pinned again. Phase 1 covered workers, out of memory, how a job ended, version recording and request names. Phase 2 covered the SLURM layer. Phase 3 covered the MCP server's stale handling, tool wrapper, log and signals.

On carbonio (release 2026.11.09.27):

- **Suites:** runtime 160, feat 192, cfdt 252, lmpt 99 and omct 404 tests pass.
- **MCP servers:** all four list their tools when started from the release.
- **Real SLURM jobs:** feat, cfdt and omct ran and were collected.

Below is what is left. The items are grouped by repo, and each says where it came from and what to do. None blocks the translators' use today.

## 1. Before calling the migration finished

- [ ] **Deploy to the other servers** (mochi, spark1, spark2) with `sudo ./llmt deploy --test`. Then check `machines()` from the Mac: every machine should report `runtime: 2026.11.09.7` and its `runtime_commit`.
- [ ] **lmpt through a real SLURM queue.** Run `LMPT_EXECUTOR=slurm LMPT_SLURM_MIN_MINUTES=0 llmt bench lmpt madelung_nacl` on a server whose queue is empty. The phase-2 code has not yet submitted, run and collected a real lmpt job: on carbonio the node was busy and the bench was interrupted.
- [ ] **The machine tests the original plan asked for, not done yet:**
  - On each machine with `P_EXECUTOR=slurm`, one fast benchmark per translator (feat `axial_bar`, cfdt `couette_2d`, lmpt `madelung_nacl`, omct `rc_step`).
  - Two benchmarks at once from two server processes. They should queue, not oversubscribe.
  - A result collected after the MCP connection was closed and reopened.
- [ ] **Drop feat's `FEAT_SLURM_TIME` alias.** `RT.time_aliases={"estimate": "timeout"}` keeps old `config/env` files meaning what they meant. Once no machine sets `FEAT_SLURM_TIME=estimate`, remove the alias: then `estimate` means the runtime's estimate-based limit in feat too. Check with `grep -r FEAT_SLURM_TIME /etc/llm-translators/env ~/.config/llm-translators/env`.
- [ ] **Decide `CFDT_SLURM_MEM` against `CFDT_MAX_MEM`.** Since cfdt 2026.11.09.4, an explicit size in `CFDT_SLURM_MEM` is used as given. Before, it was cut down to `CFDT_MAX_MEM`. `auto` is still capped at the node's memory. Keep this (the runtime's rule for all four), or restore the cut for every translator through a runtime option.

## 2. llmt-runtime

Shared code that translators still copy, or work around:

- [ ] **`jobs.queued_note()` / `print_queued_note()`.** This is the message an interrupted CLI wait prints: the queued job ids, `scancel <ids>` and `<cli> cancel <request>`. cfdt (`cli.py`) and lmpt (`executor.py`) each have a copy. Move it to the runtime and use it in all four.
- [ ] **Bounding waits to a call's deadline.** The runtime has `requests.until(wait)`. feat, cfdt, lmpt (`_within_call`) and omct (`status()`) each use their own copy. Make all four use `requests.until`. The README should also say that the call deadline only bounds SLURM commands: a translator's own wait loops must stop in time themselves.
- [ ] **A module-level `server.start_background(fn, ...)`.** It would use the serving `Server`, or a plain thread when nothing is serving. Today lmpt's `api.BACKGROUND` hook does this, set by `mcp_server.serve()`. Any translator that keeps work running after a call returns needs it.
- [ ] **`Server`: a hook for the log's argument summary.** omct used to log a study's names and flags. The runtime logs `name=value`, cut to 300 characters.
- [ ] **`Server`: the runtime folder and the exit call as settable attributes.** Today they are `Path(__file__)` and `os._exit`, and cfdt's tests patch `llmt_runtime.server.__file__` and `.threading` to replace them.
- [ ] **`stale_reason`: look for `RELEASE.json` in every parent folder**, as lmpt's `find_release` did, not only two levels up. Today's layout (`releases/<name>/<repo>/<package>`) works; an install inside the release's venv would not.
- [ ] **`testing.reset()`.** It should clear `slurm.forget()`, `jobs._PENDING` and `jobs._SBATCH_DOWN`. The translators' `conftest.py` files clear these private names themselves.
- [ ] **`submit`'s `job.json` keys.** lmpt and omct pass their own values through `extra`:
  - `time_min`: the actual limit (the runtime writes the time the runs need).
  - `time_from: "study"` when `run.timeout_s` is set.
  - `estimate_min`, unrounded.

  Settle one meaning in the runtime and drop the overrides.
- [ ] **A hook for what `cancel_request`'s flag file contains.** lmpt writes JSON `{by, at}` to `cancelled.json`; the runtime writes a bare timestamp.
- [ ] **The shape of `list_jobs`.** lmpt and omct keep their own `jobs()` because the shapes differ: lmpt has `queue` and `machine` keys, and omct reports `_collect` per request. Give `list_jobs` hooks for the per-request row so all four can use it.
- [ ] **`queue_summary`: count only this user's jobs.** cfdt and omct count only `$USER`'s jobs; the runtime counts every user's jobs carrying the translator's prefix. Add an option.
- [ ] **`node_cores`: the smallest or the largest node.** The runtime uses the smallest of the job's nodes; cfdt keeps the largest. Without SLURM, fall back to physical cores rather than logical CPUs (lmpt). It should also not ask `scontrol` when `P_EXECUTOR=local`.
- [ ] **`caps` value types.** cfdt reports `caps` values as strings (`"procs": "20"`); the runtime reports numbers. Agree on numbers in the contract (`docs/EXECUTION.md`) and change cfdt's.
- [ ] **Where `--nodelist` goes in `job.sh`.** It now comes after `--time`; cfdt and omct wrote it before. This is cosmetic, since sbatch does not care.
- [ ] **README:** it lists lmpt's `study_procs` as `solve.procs`. lmpt studies use the top-level `procs` (fixed in lmpt 2026.11.09.3).

`extend` beyond feat:

- [ ] **`extend` in cfdt, lmpt and omct.** The runtime's `jobs.extend_request` and `jobs.take_extension` are generic, but only feat's solver reads the `extend` file. For each of the others:
  1. Read it in the solver's wait loop with `take_extension(folder)` and move the run's deadline.
  2. Add `hooks.solver_left_s`.
  3. Add an `extend` tool to the MCP server (as feat's), with a test.

## 3. The translators

- [ ] **feat:** nothing open beyond the alias in section 1.
- [ ] **cfdt:**
  - Two tests fail only on macOS, before and after the migration: `test_errors_reach_the_caller_as_tool_errors` (likely the MCP SDK version on the Mac) and `test_vehicle_aero_guide_and_ahmed_benchmarks` (snappy). Make them pass or skip where the dependency is missing.
  - The new bench test (`test_runtime.py`, the bench runs a case once) depends on timing: a 2 s sleep while the job is held. Watch it on loaded machines.
  - Drop `pending_ahead` (its extra `squeue` call per status): the runtime's `ahead_of` counts per node since 2026.11.09.7. Raise cfdt's floor to `>=2026.11.09.7`.
- [x] **lmpt (2026.11.09.6):** the `Pool.__del__` noise at the end of the suite (a session fixture stops `api._POOL`), `$USER` defaulted in `conftest.py`, and the subprocess tests keep `PYTHONPATH`.
- [ ] **lmpt and omct:** nothing to change for the per-node queue position: they get it from runtime 2026.11.09.7 through `job_states` once it is pinned.
- [ ] **omct:** nothing open beyond the shared items: the log summary hook and `list_jobs`.
- [ ] **build123d-translator (b3dt):** optional. It has no executor, but its MCP server could use `llmt_runtime.server.Server` for the same stale handling, tool wrapper and log.

## 4. llm-translators

- [ ] **`llmt mcp test`** reports FAILED for every translator on a machine with no MCP entries (carbonio, whose servers the Mac starts over ssh).
  - It should say "skipped: no entry".
  - A `--direct` mode would start `<release>/bin/mcp KEY` and list the tools, so a server can check a release before switching. Keep the input open until the answer comes: feat, cfdt and lmpt take a few seconds to import, and a closed input made them exit before answering.
  - Run it from `llmt deploy --test` before switching.
- [ ] **Tests on macOS** (`tests/test_llmt.py`), failing before this work:
  - The four Staging tests: macOS temporary paths `/var/…` resolve to `/private/var/…`, so the expected keys differ.
  - `test_starts_now`: the start time comes back None, probably because the fake `sbatch --test-only` relies on Linux-only behaviour.
- [ ] **`translators.json`: feat's `min_minutes` is 5, but feat's default is 1** (`FEAT_SLURM_MIN_MINUTES`, `RT.min_minutes`). `llmt tune` and the router read the JSON. Make them agree.
- [ ] **`llmt gc`:** each server has several unused releases from the test builds (`--no-switch` builds never switched to). Run `sudo ./llmt gc` once everything is deployed.
- [ ] **Docs:**
  - `ROUTER.md`: `machines()` now shows `runtime` and `runtime_commit`; its rows could show `max_job_mem` from the exact caps.
  - `EXECUTION.md`: the per-translator "Implemented: …" notes can go, now that one implementation exists (llmt-runtime's README keeps the history).

## 5. Rules to keep (learnt during the migration)

- **Floors only.** A translator declares `llmt-runtime>=X` and never an upper bound, and the runtime only adds things. A change that breaks the API needs a new floor in every translator that uses it, in the same release; `llmt deploy` refuses to build until they agree.
- **Pin the runtime before a translator that needs a newer one.** `llmt pin runtime <key>` pins the runtime first.
- **Test a release before users get it:**
  - `sudo ./llmt deploy --no-switch`.
  - Then `<release>/llmt test runtime <keys>` (all five when the runtime changed), and the MCP tool lists.
  - Then `sudo ./llmt rollback <release>` to switch.
  - A test that inspects the real stale check must pin it, because a release that is not yet current is stale.
- **`git rm -f`** for files the migration turned into stubs: a plain `git rm` refuses a modified file and, inside `&&`, silently skips the commit.
- **Test helpers** that take a folder plus `**kwargs` for a job record must not call the folder parameter `state`: a record's `state` key then collides with it.
- **Every command the runtime runs gets `/dev/null` as stdin.** The MCP server speaks on its own stdio.

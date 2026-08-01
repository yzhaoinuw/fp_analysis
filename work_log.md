# Work Log

Prepend new session notes to the top of this file. The live log holds at most the five most recent unique calendar dates; each archive file holds exactly five. See [Work Log Discipline](treaty_conventions.md#work-log-discipline) and [Work Log Rotation And Dating](treaty_conventions.md#work-log-rotation-and-dating).

## 2026-08-01

### Migrate the collaboration treaty to v0.6.0 (gpt-5)

- Migrated the tracked Copier adoption from template `v0.3.2` to `v0.6.0` on the dedicated `chore/treaty-v0.6.0` branch, preserving the active desktop runtime, event/export, packaging, updater, and local-data boundaries.
- Kept `AGENTS.md` as a lean project-specific map, added the upstream-managed `treaty_conventions.md`, and added an authored-versus-derived map to `project_overview.md`.
- Recorded the repository's real release, Git-ownership, verification, and no-pre-commit answers; resolved all three expected Copier conflicts without changing application code or generated/local artifacts.
- Rotated the previous five live dates into `work_log_archive/work_log_2026-07-23_to_2026-07-30.md` before adding this date.
- Opened and verified upstream issue `agent_collab_treaty#17` for a newly observed update gap: older managed adopters silently receive defaults for newly introduced questions, including an incorrect `uses_precommit: true` in this repository.
- The user accepted the validated migration for local fast-forward integration into `dev` and `main`; remote publication remains a separate action.
- Verification:
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe --version` - reported CLI and pinned template `v0.6.0`; the source checkout was verified at stable release commit `cdfb3c84822ac549dcae3d279c356c64490ea424`.
  - Clean-tree `treaty.exe update . --dry-run` - exited 0 outside the sandbox, left status clean, and preserved hashes for every existing managed file.
  - Applied `treaty.exe update .` - exited 1 and named the three expected customized-doc conflicts; all were resolved and no unmerged index entries remain.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 85 tests passed with 8 private-fixture skips.
  - Active analysis-module import smoke - printed `import ok`.
  - `$env:FP_ANALYSIS_SKIP_UPDATE='1'; C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe run_desktop_app.py --smoke` - printed `smoke ok: v0.6.0`.
  - `treaty.exe validate .`, `treaty.exe diff .`, and `git diff --cached --check` - passed; the final treaty diff has no removed headings, and `treaty_conventions.md` is untouched against v0.6.0.
  - GitHub issue inspection - verified `https://github.com/yzhaoinuw/agent_collab_treaty/issues/17`, including the `Codex (GPT-5)` signature.

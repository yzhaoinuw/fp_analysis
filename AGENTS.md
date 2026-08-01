# Guidelines and Tips for Agents

Read this file first when joining the repository. It is the project-specific quick-reference map; generic collaboration mechanics live in [`treaty_conventions.md`](treaty_conventions.md). Keep this file lean and use the documentation map instead of reading every Markdown file.

## Startup Rule

At the beginning of a new session, read this file first. Open only the other documents relevant to the task.

## Runtime Environment

Use the `fiber_photometry` conda environment. Non-interactive shells should call its known-good Python 3.10 interpreter directly:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe --version
```

Do not assume PowerShell supports Bash-style `&&`; use separate commands or PowerShell-native sequencing.

## Common Tasks

Run the active desktop app:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe run_desktop_app.py
```

Run the CI-equivalent focused suite:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging
```

The fixture-backed `F268` tests run only when `data/F268.mat` and `data/Transitions_F268.xlsx` are available; synthetic export tests do not require them.

Useful import smoke check:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"
```

Before committing, run the focused suite, relevant import smoke, `git diff --check`, and `treaty validate .`; stage only task-relevant files. No formatter or linter is configured, so do not invent one as a gate.

## When To Update Treaty Docs

At the end of substantive work, update `work_log.md` unless the user asks not to document the session. Update `next_steps.md` only when future work changes. Follow [Work Log Discipline](treaty_conventions.md#work-log-discipline) and [Work Log Rotation And Dating](treaty_conventions.md#work-log-rotation-and-dating).

## Branch Handoff Discipline

`main` is the release/integration branch; `dev` is the staging line. Feature work normally starts from the appropriate current staging branch. Before switching branches or worktrees, confirm the current branch is committed, verified, and pushed, merged, or deliberately parked.

This repository has multiple linked worktrees. Do not remove, repoint, or switch them casually. Useful checks:

```powershell
git status --short --branch
git log --oneline --left-right --cherry-pick main...HEAD
git merge-base --is-ancestor main HEAD
git worktree list
```

See [Branch Handoff](treaty_conventions.md#branch-handoff) for the generic procedure.

## Release / Tag Checklist

This repository ships GitHub Releases and Windows packages. Treat commit + push + tag or release requests as release work, and clear the documentation/version/verification gate in [Release Gate](treaty_conventions.md#release-gate) before tagging.

## Updating The Treaty

Only update the treaty when explicitly asked. Use `treaty diff`, then `treaty update --dry-run`, then apply on a clean dedicated branch and resolve every unmerged file. See [Updating The Treaty](treaty_conventions.md#updating-the-treaty).

## Documentation

- `treaty_conventions.md` — upstream-managed collaboration, logging, branch, release, and update procedures; prefer not to edit it.
- `project_overview.md` — architecture, active-versus-legacy and authored-versus-derived maps, data contracts, and reading order.
- `next_steps.md` — unfinished work; read "Currently Hot" when planning or continuing a thread.
- `work_log.md` and `work_log_archive/` — recent decisions, evidence, and verification breadcrumbs; default to the two newest dates.
- `README.md` — user-facing installation, operation, export behavior, and input requirements.
- `CHANGELOG.md` — product release notes.

`PROJECT_MEMORY.md`, the archived pre-treaty `AGENTS.md`, `change_log.txt`, and completed one-off planning documents were retired. Do not recreate parallel agent-memory documents.

## Commit Message Guidelines

Use a short title. When one commit contains several user-requested changes, add a short body with flat bullets describing high-level behavior. For features, emphasize user-visible behavior rather than tests or documentation bookkeeping unless those are the purpose.

## Git Ownership Note

If Git reports dubious ownership, add this worktree's absolute forward-slash path to `safe.directory`. In this Windows setup, Git commands that write shared metadata or contact the remote often require outside-sandbox execution; do not retry known lock, credential, or `cannot spawn sh` failures in the sandbox.

## Active Runtime Boundary

The packaged path is `run_desktop_app.py -> fp_analysis_app/app_dev.py -> components_dev.py` and `pages/`, then visualization, event-analysis, and export helpers. Inspect `app_dev.py` before the secondary browser/upload `app.py`; do not broadly synchronize or delete the secondary path without confirming intended behavior.

## Startup Source-Update Boundary

`run_desktop_app.py` invokes the shared `desktop_app_source_updater` before importing the app. Keep app-specific release URLs, state, environment variables, allowed paths, and exclusions in `startup_update_config.py`; reusable update logic stays in the pinned shared package. Source-only releases are allowed only after a compatible updater ships in a full package. See `project_overview.md` and `packaging/windows/` for the detailed package/asset contract.

## Project-Specific Reminders

- The app is desktop-first Dash hosted in pywebview at `127.0.0.1:8050`.
- Spreadsheet analysis is intentionally two-phase: `Show Results` caches export-ready data, then `Save Spreadsheets` writes only selected workbook types.
- Export grouping and fallback behavior lives in `analysis_export.py` and `export_settings.py`; numeric analysis and workbook sheet logic lives in `event_analysis.py`.
- MAT visualization normally uses `fp_signal_names` plus `fp_frequency`; `mat_utils.py` supports an NE-only `ne` plus `ne_frequency` fallback.
- Annotation imports accept ordinary event tables and sleep-bout tables with `sleep_scores`, `start`, `end`, and `duration`.
- Generated/local content under `cache/`, `data/`, `fp_analysis_app/assets/`, `build/`, `dist/`, and `release_artifacts/` is out of scope unless explicitly requested. Inspect before changing or deleting it.
- `environment.yml` and `setup.py` retain historical metadata; the `fiber_photometry` environment and `requirements.txt` are the practical runtime references.
- Keep only app-specific updater configuration and CLI defaults here; do not reintroduce a copied updater implementation.
- This checkout can be intentionally dirty. Never sweep unrelated tracked or untracked files into a focused change.

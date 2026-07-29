# Guidelines and Tips for Agents

Read this file first when joining the repository. It defines the supported runtime, active code path, verification commands, and documentation workflow. Do not automatically read every Markdown file; use the documentation map below.

## Runtime Environment

Use the `fiber_photometry` conda environment.

```powershell
conda activate fiber_photometry
```

Codex and other non-interactive shells should prefer the known-good interpreter directly because `python`, `py`, and `conda` are not guaranteed to be on `PATH`:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe --version
```

The current interpreter is Python 3.10. Do not assume PowerShell supports Bash-style `&&`; use separate commands or PowerShell-native sequencing.

## Common Tasks

Run the active desktop application:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe run_desktop_app.py
```

Run the CI-equivalent and full focused test suite:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging
```

The fixture-backed `F268` tests run when `data/F268.mat` and `data/Transitions_F268.xlsx` are available. Synthetic export tests run without those local fixtures.

Useful import smoke check:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"
```

Pre-flight checklist before committing:

- `git diff --check` is clean.
- The focused unittest suite passes.
- Active modules touched by the change still import.
- Only task-relevant files are staged; this checkout may intentionally contain unrelated untracked drafts and generated data.
- A new entry is prepended to `work_log.md` with the exact verification commands actually run.
- `next_steps.md` is updated when the session changes future work.
- The treaty validator passes:

```powershell
C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .
```

No formatter or linter is currently configured for this repository. Do not invent one as a pre-flight requirement.

## Active Runtime Boundary

The packaged desktop path is:

```text
run_desktop_app.py
  -> fp_analysis_app/app_dev.py
  -> fp_analysis_app/components_dev.py and fp_analysis_app/pages/
  -> visualization, event analysis, and spreadsheet-export helpers
```

For desktop behavior, inspect `app_dev.py` before `app.py`. The two modules are not interchangeable:

- `app_dev.py` is the active pywebview desktop runtime and current selective spreadsheet-export flow.
- `app.py` is a secondary browser/upload implementation that still contains manual annotation save behavior not fully migrated to `app_dev.py`.

Do not delete or broadly synchronize the secondary path without confirming the intended product behavior.

## Startup Source-Update Boundary

`run_desktop_app.py` calls the shared `desktop_app_source_updater` package before importing `fp_analysis_app`. App-specific release URLs, per-user check-state location, environment variables, allowed payload paths, and generated-data exclusions live in `startup_update_config.py`; reusable discovery, throttling/backoff, manifest, hash, download, config-merge, and rollback logic must stay in the shared package.

Normal startup uses GitHub's ordinary `/releases/latest` redirect and persists a 24-hour check interval under the user's local application-data folder. `--check-update` must call the updater with `force_check=True`; `FP_ANALYSIS_FORCE_UPDATE_CHECK=1` is the environment-variable equivalent.

`tools/build_update_asset.py` is only a thin app-specific argument wrapper around `desktop_app_source_updater.build_update_asset`. Build a later source-only asset with one `--from-ref` per supported packaged baseline:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe tools\build_update_asset.py --from-ref <tag-or-commit> --to-ref <tag-or-commit>
```

The shared dependency is pinned in `requirements.txt` for reproducible packaging. Any change that introduces or upgrades that dependency must ship in a normal full package first. Only releases based on a package that already contains the compatible updater may use source-only assets.

The repeatable full Windows build uses `fp_analysis_dist` and the portable spec under `packaging/windows/`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1
```

Use `-AllowDirty` only for a local packaging rehearsal. Release artifacts must come from a clean tracked worktree so the manifest and side-by-side `fp_analysis_app/` bytes resolve to one commit.

## Treaty Documentation Workflow

At the end of substantive work, update `work_log.md` unless the user explicitly asks not to document the session. Substantive work includes file edits, meaningful validation or debugging, technical decisions, reusable discoveries, branch/release changes, and unfinished follow-up.

When future work changes, update `next_steps.md` in the same pass. Keep its "Currently Hot" links accurate and remove completed threads rather than leaving a permanent backlog.

The live work log holds at most five unique calendar dates. When a sixth date is added, move the oldest five dates together into:

```text
work_log_archive/work_log_<earliest>_to_<latest>.md
```

If today's date already exists at the top of `work_log.md`, add another `###` session under that date instead of creating a duplicate date heading.

## Branch Handoff Discipline

`main` is the release/integration branch. `dev` is the staging line used by CI, and feature or experimental branches should normally start from the appropriate current staging branch.

Before switching branches or worktrees, confirm that the current branch's intended work is committed, verified, and either pushed, merged, or deliberately parked. This repository has multiple linked worktrees; do not remove, repoint, or switch them casually.

Useful checks:

```powershell
git status --short --branch
git log --oneline --left-right --cherry-pick main...HEAD
git merge-base --is-ancestor main HEAD
git worktree list
```

If Git reports dubious ownership:

```powershell
git config --global --add safe.directory C:/Users/yzhao/python_projects/fp_analysis
```

On this Windows checkout, mutating Git commands often fail inside the sandbox with
credential, safe-directory, or lock-file errors (`cannot spawn sh`,
`ORIG_HEAD.lock`, `.git/objects`, etc.). For Git commands that write repository
state or contact the remote, such as `git push`, `git tag`, `git merge`, branch
operations, and `git config --global --add safe.directory`, request outside-
sandbox execution immediately instead of retrying the same command in the
sandbox. Read-only commands such as `git status`, `git diff`, `git log`, and
`git worktree list` can run normally unless they fail.

## Documentation Map

- `project_overview.md`
  - Read for architecture, active-versus-legacy boundaries, data contracts, and the recommended file-reading order.
- `next_steps.md`
  - Read its "Currently Hot" section when planning or continuing unfinished work.
- `work_log.md` and `work_log_archive/`
  - Read the newest entries for implementation history, decisions, and verification breadcrumbs.
- `README.md`
  - Read or edit for user-facing installation, operation, export behavior, and input-file requirements.
- `CHANGELOG.md`
  - Read or edit for product release notes.

`PROJECT_MEMORY.md`, the archived pre-treaty `AGENTS.md`, the obsolete `change_log.txt`, and completed one-off planning documents were retired when this treaty was adopted. Do not recreate parallel agent-memory documents; put durable orientation in `project_overview.md`, active work in `next_steps.md`, and dated evidence in the work log.

## Commit Message Guidelines

Use a short title line. If a commit contains multiple user-requested changes, add a short body with flat bullets describing high-level behavior. For feature commits, emphasize user-visible behavior rather than tests, documentation bookkeeping, or implementation details unless those are the purpose of the commit.

## Project-Specific Reminders

- The app is desktop-first Dash hosted inside pywebview at `127.0.0.1:8050`.
- Spreadsheet analysis is intentionally two-phase in `app_dev.py`: `Show Results` computes and caches export-ready data, then `Save Spreadsheets` writes only selected workbook types.
- Export grouping and fallback behavior lives in `analysis_export.py` and `export_settings.py`; numeric analysis and workbook sheet logic lives in `event_analysis.py`.
- MAT visualization normally uses `fp_signal_names` plus `fp_frequency`. `mat_utils.py` supports an NE-only fallback using `ne` plus `ne_frequency`.
- Annotation imports accept ordinary event spreadsheets (`event name` columns containing event times) and sleep-bout tables with `sleep_scores`, `start`, `end`, and `duration`.
- Generated files under `cache/`, `data/`, `fp_analysis_app/assets/`, `build/`, and `dist/` may be local artifacts. Inspect before changing or deleting them.
- `packaging/windows/` owns the portable PyInstaller spec, full-build script, release smoke checks, and packaging rationale. Generated `release_artifacts/`, `build/`, and `dist/` content is local output.
- `environment.yml` and `setup.py` contain historical project naming and dependency metadata. The `fiber_photometry` environment and `requirements.txt` are the practical local runtime references unless a task explicitly modernizes packaging.
- The startup updater implementation comes from the pinned `desktop_app_source_updater` dependency. Keep only app-specific configuration and CLI defaults in this repo; do not reintroduce a copied updater implementation.
- This checkout can be intentionally dirty. Never sweep unrelated tracked or untracked files into a focused commit.

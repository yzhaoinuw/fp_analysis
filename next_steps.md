# Next Steps

Use this checklist alongside `work_log.md`. Keep only actionable engineering threads here; release history belongs in `CHANGELOG.md` and dated evidence belongs in the work log.

## Currently Hot

- [v0.5.0-beta stabilization](#v050-beta-stabilization-gpt-5) - reproduce and fix the stale analysis-result refresh reported after changing a second signal.
- [Desktop runtime convergence](#desktop-runtime-convergence-gpt-5) - decide whether the manual annotation save flow in `app.py` belongs in the active desktop runtime.
- [Packaging reproducibility](#packaging-reproducibility-gpt-5) - remove or document machine-specific packaging assumptions before the next distributable build.

## v0.5.0-beta Stabilization (gpt-5)

Status: in progress; beta behavior is implemented and tested, but a user-visible refresh issue remains documented.

The current beta separates `Show Results` from `Save Spreadsheets`, remembers export selections, resets controls for new MAT files, and performs workbook saving in a background callback.

Remaining work:

- Reproduce the README issue where adding or removing a second signal can finish analysis without visibly refreshing the result until the user switches tabs.
- Identify whether stale page content, callback output ordering, or cached export payload state is responsible.
- Resolve the annotation-file contract: the desktop dialog offers `.csv`, but `Event_Utils.read_events()` currently reads file paths with `pandas.read_excel()`.
- Add a focused regression test where practical.
- Run the unittest suite and a manual two-signal desktop check before changing release status.

## Desktop Runtime Convergence (gpt-5)

Status: paused pending a product decision.

`app_dev.py` is the active pywebview desktop runtime. `app.py` is a secondary browser/upload path and still contains manual annotation save/export behavior that is not equivalent in `app_dev.py`.

Remaining work:

- Decide whether manual annotation editing and save/export are still supported product requirements.
- If yes, migrate the required behavior into `app_dev.py` and `components_dev.py` with focused tests.
- If no, document the boundary and then retire the secondary path in a separate, reviewable cleanup.

## Packaging Reproducibility (gpt-5)

Status: paused until the next packaging task.

`app.spec` contains absolute paths to this workstation and references an `fp_analysis_dist` environment, while routine development uses `fiber_photometry`. `environment.yml` and `setup.py` also retain historical sleep-scoring metadata.

Remaining work:

- Choose the supported build environment and record an exact build command.
- Replace hardcoded paths in `app.spec` with paths derived from the build environment or project root.
- Confirm which assets and external modules must remain outside the executable for patch-style distribution.
- Build and smoke-test the packaged app on a clean output directory before updating installation instructions.

## Background / Paused

### Linked worktree cleanup

Several historical Codex and synchronization worktrees are still registered under `.worktrees/`. They are not an active documentation task. Inspect branch reachability and uncommitted state before removing any worktree.

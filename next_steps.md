# Next Steps

Use this checklist alongside `work_log.md`. Keep only actionable engineering threads here; release history belongs in `CHANGELOG.md` and dated evidence belongs in the work log.

## Currently Hot

- [v0.5.0-beta stabilization](#v050-beta-stabilization-gpt-5) - reproduce and fix the stale analysis-result refresh reported after changing a second signal.
- [Startup auto-update integration](#startup-auto-update-integration-gpt-5) - publish a clean full-package baseline, then smoke-test a later source-only release.
- [Desktop runtime convergence](#desktop-runtime-convergence-gpt-5) - decide whether the manual annotation save flow in `app.py` belongs in the active desktop runtime.

## v0.5.0-beta Stabilization (gpt-5)

Status: in progress; beta behavior is implemented and tested, but a user-visible refresh issue remains documented.

The current beta separates `Show Results` from `Save Spreadsheets`, remembers export selections, resets controls for new MAT files, and performs workbook saving in a background callback.

Remaining work:

- Reproduce the README issue where adding or removing a second signal can finish analysis without visibly refreshing the result until the user switches tabs.
- Identify whether stale page content, callback output ordering, or cached export payload state is responsible.
- Resolve the annotation-file contract: the desktop dialog offers `.csv`, but `Event_Utils.read_events()` currently reads file paths with `pandas.read_excel()`.
- Add a focused regression test where practical.
- Run the unittest suite and a manual two-signal desktop check before changing release status.

## Startup Auto-Update Integration (gpt-5)

Status: the app-local prototype has been migrated to the pinned `desktop_app_source_updater` package, and the lean full-package pipeline has passed a dirty-worktree rehearsal; clean baseline publication and a real later source update remain.

The launcher delegates source-update discovery, manifest validation, hash/baseline checks, config merging, and rollback to the shared package before importing the active app. `startup_update_config.py` holds only the `fp_analysis` release URL, environment-variable names, allowed `fp_analysis_app/` payload boundary, and generated-data exclusions. `tools/build_update_asset.py` is now a thin app-specific wrapper around the shared builder.

`packaging/windows/` now provides a portable PyInstaller spec, exact tracked-byte export of the side-by-side `fp_analysis_app/` tree, packaged and fresh-extraction smoke checks, an unblock-and-start helper, and ZIP/hash/manifest/environment outputs. The rehearsal built and smoke-tested an updater-enabled Windows ZIP; because it intentionally used `-AllowDirty`, it is evidence for the pipeline rather than a release candidate.

Because this integration adds a dependency and removes the copied updater, it must first ship in a normal full packaged release from a clean commit. A later release containing additional source-only changes can then provide `fp_analysis_app_update_<version>.zip` and test the real update path.

Remaining work:

- Commit the updater and packaging work, then rerun `packaging/windows/make_full_app_zip.ps1` without `-AllowDirty`.
- Review and publish that clean full package as the first compatible updater baseline.
- Add a later source-only change, build a multi-baseline asset with `tools/build_update_asset.py`, attach it to a test GitHub Release, and confirm the packaged app updates its external `fp_analysis_app/` folder before import.
- Decide how much update status should be visible in the GUI versus console output.

## Desktop Runtime Convergence (gpt-5)

Status: paused pending a product decision.

`app_dev.py` is the active pywebview desktop runtime. `app.py` is a secondary browser/upload path and still contains manual annotation save/export behavior that is not equivalent in `app_dev.py`.

Remaining work:

- Decide whether manual annotation editing and save/export are still supported product requirements.
- If yes, migrate the required behavior into `app_dev.py` and `components_dev.py` with focused tests.
- If no, document the boundary and then retire the secondary path in a separate, reviewable cleanup.

## Background / Paused

### Linked worktree cleanup

Several historical Codex and synchronization worktrees are still registered under `.worktrees/`. They are not an active documentation task. Inspect branch reachability and uncommitted state before removing any worktree.

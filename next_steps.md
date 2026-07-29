# Next Steps

Use this checklist alongside `work_log.md`. Keep only actionable engineering threads here; release history belongs in `CHANGELOG.md` and dated evidence belongs in the work log.

## Currently Hot

- [Post-v0.6.0 follow-up](#post-v060-follow-up-gpt-5) - reproduce and fix the stale analysis-result refresh in a future update.
- [Startup auto-update integration](#startup-auto-update-integration-gpt-5) - use the distributed v0.6.0 baseline to smoke-test the following source-only release.
- [Desktop runtime convergence](#desktop-runtime-convergence-gpt-5) - decide whether the manual annotation save flow in `app.py` belongs in the active desktop runtime.

## Post-v0.6.0 Follow-up (gpt-5)

Status: deferred to a future update; v0.6.0 ships the accepted analysis workflow, while one user-visible refresh issue remains documented.

The current release separates `Show Results` from `Save Spreadsheets`, remembers export selections, resets controls for new MAT files, and performs workbook saving in a background callback.

Remaining work:

- Reproduce the README issue where adding or removing a second signal can finish analysis without visibly refreshing the result until the user switches tabs.
- Identify whether stale page content, callback output ordering, or cached export payload state is responsible.
- Add a focused regression test where practical.
- Run the unittest suite and a manual two-signal desktop check before shipping the fix in a future update.

## Startup Auto-Update Integration (gpt-5)

Status: the clean `v0.6.0` full Windows package is the distributed updater 0.2.0 baseline, with API-free release discovery, durable per-user check throttling/backoff, and forced explicit checks. The focused suite, clean package build, packaged smoke, and fresh-extraction smoke pass.

The launcher delegates release discovery, durable throttling/backoff, manifest validation, hash/baseline checks, config merging, and rollback to the shared package before importing the active app. `startup_update_config.py` holds only the `fp_analysis` latest-release URL, per-user state path, environment-variable names, allowed `fp_analysis_app/` payload boundary, and generated-data exclusions. `tools/build_update_asset.py` remains a thin app-specific wrapper around the shared builder.

Updater 0.2.0 discovers the latest release through GitHub's ordinary redirect rather than the unauthenticated REST API, compares the tag before downloading an asset, persists a 24-hour check interval and rate-limit backoff under the user's local application-data folder, and lets `--check-update` bypass that interval explicitly.

Because the updater is bundled inside the executable, the distributed `v0.6.0` package is the installed baseline for the first updater 0.2.0 source-only trial. A later release based on that package can exercise updater 0.2.0 through `fp_analysis_app_update_<version>.zip`.

Remaining work:

- Keep one extracted `v0.6.0` package unchanged as the user test baseline.
- Choose the later normal source-update version when its app changes are ready; `v0.6.1` is available again after removal of the obsolete local tag.
- Add the later source-only change, build an asset from `v0.6.0` with `tools/build_update_asset.py`, attach it to the following non-prerelease GitHub Release, and confirm the baseline executable updates its external `fp_analysis_app/` folder before import. For a prerelease-only gate, point `FP_ANALYSIS_UPDATE_ZIP_URL` directly at the test asset instead.
- Run the baseline executable with `--check-update`, then run `--smoke` and the normal desktop launch from the updated folder.
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

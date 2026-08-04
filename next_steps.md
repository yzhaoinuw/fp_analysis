# Next Steps

Use this checklist alongside `work_log.md`. Keep only actionable engineering threads here; release history belongs in `CHANGELOG.md` and dated evidence belongs in the work log.

## Currently Hot

- [Post-v0.6.0 refresh fix](#post-v060-refresh-fix-gpt-5) - reproduce and fix the stale analysis-result refresh in a future update.
- [First source-update release trial](#first-source-update-release-trial-gpt-5) - prove the distributed v0.6.0 baseline updates correctly from the next compatible normal release.
- [Manual sleep-scoring feature decision](#manual-sleep-scoring-feature-decision-gpt-5) - decide whether the secondary runtime's manual sleep-state workflow remains a product requirement.

## Post-v0.6.0 Refresh Fix (gpt-5)

Status: unresolved and documented in the README; v0.6.0 otherwise ships the accepted analysis workflow.

Remaining work:

- Reproduce the README issue where adding or removing a second signal can finish analysis without visibly refreshing the result until the user switches tabs.
- Identify whether stale page content, callback output ordering, or cached export payload state is responsible.
- Add a focused regression test where practical.
- Run the unittest suite and a manual two-signal desktop check before shipping the fix in a future update.

## First Source-Update Release Trial (gpt-5)

Status: the unchanged v0.6.0 full package is published and preserved locally with updater 0.2.0. The updater integration and full-package gates are complete; only a real later-release trial remains.

Compatibility boundary: keep the prefix-style `fp_analysis_app_update_<version>.zip` filename for the distributed v0.6.0 baseline.

Remaining work:

- Choose the next compatible app-only change and normal release version.
- Build its update asset from `v0.6.0` with `tools/build_update_asset.py`, then attach it to the next non-prerelease GitHub Release.
- From the unchanged v0.6.0 package, run `--check-update`, `--smoke`, and a normal desktop launch; confirm the external `fp_analysis_app/` folder updates before import.

Zenodo archival rides on this same release:

- Zenodo is enabled for the repository, but it did not retroactively archive `v0.6.0`. Archival begins with the next newly published release, and no citation-only release is planned to force it earlier.
- Update `CITATION.cff` `version` and `date-released` in this release candidate, together with the app's other version and release surfaces, before publishing.
- After publication, add the assigned concept DOI to `CITATION.cff` and a DOI badge plus link to the README's Citation section.

## Manual Sleep-Scoring Feature Decision (gpt-5)

Status: paused pending a product decision.

The active `app_dev.py` runtime already supports event-time deletion, one-step undo, and saving annotated MAT files. The secondary `app.py` path additionally supports manual sample-level sleep-state scoring and exports an updated MAT file plus sleep-bout and sleep-statistics workbooks.

Remaining work:

- Decide whether manual sleep-state scoring and its sleep-bout/statistics exports are still product requirements.
- If yes, migrate that workflow into `app_dev.py` and `components_dev.py` with focused tests.
- If no, document the boundary and then retire the secondary path in a separate, reviewable cleanup.

## Background / Paused

### Future updater packaging and status UX

- When the next full-package baseline is planned, decide whether to add suffix-style update discovery to the shared updater and bundle it before renaming any source-update assets.
- After the first real source-update trial, decide whether console-only update status is sufficient or whether the GUI should show more.

### Linked worktree cleanup

Several historical Codex and synchronization worktrees are still registered under `.worktrees/`. They are not an active documentation task. Inspect branch reachability and uncommitted state before removing any worktree.

# Next Steps

Use this checklist alongside `work_log.md`. Keep only actionable engineering threads here; release history belongs in `CHANGELOG.md` and dated evidence belongs in the work log.

## Currently Hot

- [Analysis workflow feedback update](#analysis-workflow-feedback-update-gpt-5) - implement the recent user-feedback TODOs as gated, item-by-item analysis-page improvements.
- [v0.5.0-beta stabilization](#v050-beta-stabilization-gpt-5) - reproduce and fix the stale analysis-result refresh reported after changing a second signal.
- [Startup auto-update prototype](#startup-auto-update-prototype-gpt-5) - package-smoke the release-zip updater against a realistic distributable folder.
- [Desktop runtime convergence](#desktop-runtime-convergence-gpt-5) - decide whether the manual annotation save flow in `app.py` belongs in the active desktop runtime.
- [Packaging reproducibility](#packaging-reproducibility-gpt-5) - remove or document machine-specific packaging assumptions before the next distributable build.

## Analysis Workflow Feedback Update (gpt-5)

Status: item 5 event-annotation save/undo implemented; awaiting user acceptance gate.

Product goal: make the analysis flow clearer, make exclusions visible and respected by analysis, and fix peak metrics so plotted/exported values match detected positive and negative peaks.

Implementation boundaries:

- Work in the active desktop runtime: `fp_analysis_app/app_dev.py`, `fp_analysis_app/components_dev.py`, `fp_analysis_app/make_figure.py`, `fp_analysis_app/event_analysis.py`, `fp_analysis_app/analysis_export.py`, and focused tests in `tests/test_perievent_analysis.py`.
- Do not broadly sync behavior from secondary `fp_analysis_app/app.py` unless a specific item requires reusing a small helper.
- After each item, run the narrowest useful verification plus `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` when numeric analysis, exports, or event filtering changed.
- After each user gate, update this section's status/checklist and add a compact `work_log.md` note with the commands actually run.

Gated implementation checklist:

1. [x] Analysis-page signal selection cue.
   - Highlight the signal selection dropdown so users' attention is directed there before results can be shown.
   - Preserve the disabled `Show Results` behavior while no signal is selected, and keep it disabled if more than two signals are selected.
   - Acceptance gate: accepted after visual check; users can reach `/analysis`, notice the selector, select one or two signals, and then run analysis.

2. [x] README plot explanations.
   - Add a concise README section explaining each analysis plot and metric: raw perievent traces, mean trace, perievent heatmap, normalized perievent traces, AUC, positive peak value, negative peak value, first peak time, decay time, and cross-correlation when two signals are selected.
   - Explain that peak values are based on detected peaks and become `NaN` when no peak is found.
   - Acceptance gate: accepted after user review; wording is concise enough for the README.

3. [x] Event timestamp lines on the main full-recording visualization.
   - Show all imported event timestamps at once on the main Plotly full-recording graph after annotations are loaded.
   - Use palette-colored vertical line traces with one legend entry per event type.
   - Suppress expanded perievent-window coloring in the active event-annotation visualization path, while keeping it available behind a figure toggle.
   - Remove or hide event lines immediately when their event timestamps are excluded by a removed period.
   - Acceptance gate: accepted after visual check; loaded annotation event timestamps are visible on the main graph without expanded-window coloring.

4. [x] Annotation-rectangle event deletion before analysis.
   - In annotation mode, let the user draw a rectangle on the interactive full-recording plot.
   - When a rectangle span is selected and the user presses `Delete` or `Backspace`, remove event timestamps whose times fall inside the rectangle's x/time span.
   - Keep the selection/delete interaction clientside for responsive visual updates; use the event-time store as the bridge back to server-side analysis.
   - Do not add a dedicated removal button for this flow.
   - Store removals in session/runtime state first. Removed event timestamps should disappear from the vertical event lines and be excluded from downstream analysis/export payloads.
   - Leave photometry signal arrays untouched; only the cached event-time arrays are edited in this pass.
   - Keep only the most recently drawn selection rectangle if the user draws boxes in multiple subplots.
   - Acceptance gate: user can enter annotation mode, draw a rectangle spanning event timestamp lines, press `Delete` or `Backspace`, see those vertical lines disappear, and run analysis with contained events excluded.

5. Save and undo modified event annotations.
   - Move the `Analysis ->` link to the lower-right side of the main page action row.
   - Place `Save Annotations` at the lower-left side of the action row, with `Undo Annotation` immediately to its right.
   - Keep one-step undo for event timestamp edits using the runtime event-time history.
   - Save current event timestamp annotations into app-owned MAT fields via a native Save dialog, using a temp MAT file before copying to the chosen target.
   - Reopening a saved MAT should prefer the saved app-owned event timestamp fields over the original embedded `event` payload.
   - Future period-label persistence should reuse this action row once period edit/removal returns to scope.
   - Acceptance gate: user can delete event timestamps, undo the most recent edit, save a MAT copy, reload it, and see the saved event timestamp state.

6. Positive-integer baseline and analysis window inputs.
   - Replace baseline/analysis window dropdowns with numeric text inputs.
   - Validate positive integers only.
   - Reject values greater than or equal to one quarter of the full recording duration.
   - Keep `Show Results` disabled, or show a clear inline error, when inputs are invalid.
   - Acceptance gate: invalid values cannot launch analysis; valid integer values do.

7. Perievent heatmap row dividers.
   - Add visible dividers between event-occurrence rows in the perievent heatmap generated by `Perievent_Plots.plot_perievent_heatmaps`.
   - Keep labels readable and avoid crowding on high-count events.
   - Acceptance gate: heatmap rows are easier to distinguish without obscuring signal intensity.

8. Negative peak row in analysis plots.
   - Extend analysis results with negative peak detection by flipping reaction signals around `y=0` and reusing the same `find_peaks` settings.
   - Add a row or clearly separated plot area for negative peak values in the analysis plots.
   - Negative peak values should remain negative in plots and exports.
   - Acceptance gate: positive and negative peak summaries are both visible and correspond to detected peaks.

9. Rename and correct peak-value exports.
   - Replace the old `max_peak_magnitude` behavior with peak-value metrics based on detected peak indices.
   - Positive peak value is the signal value at the first detected positive peak; `NaN` if no positive peak is detected.
   - Negative peak value is the signal value at the first detected negative peak; `NaN` if no negative peak is detected.
   - Rename labels, export type names, workbook names, tests, and README wording away from "max peak magnitude" toward `positive_peak_value` and `negative_peak_value`.
   - Acceptance gate: spreadsheets use the new names, missing peaks export as `NaN`, and tests cover positive/negative/no-peak cases.

## v0.5.0-beta Stabilization (gpt-5)

Status: in progress; beta behavior is implemented and tested, but a user-visible refresh issue remains documented.

The current beta separates `Show Results` from `Save Spreadsheets`, remembers export selections, resets controls for new MAT files, and performs workbook saving in a background callback.

Remaining work:

- Reproduce the README issue where adding or removing a second signal can finish analysis without visibly refreshing the result until the user switches tabs.
- Identify whether stale page content, callback output ordering, or cached export payload state is responsible.
- Resolve the annotation-file contract: the desktop dialog offers `.csv`, but `Event_Utils.read_events()` currently reads file paths with `pandas.read_excel()`.
- Add a focused regression test where practical.
- Run the unittest suite and a manual two-signal desktop check before changing release status.

## Startup Auto-Update Prototype (gpt-5)

Status: release-zip prototype implemented on the `auto-update` branch; not yet package-smoked.

The launcher now has a code-only updater that checks for a custom GitHub Release asset named like `fp_analysis_app_update_<version>.zip` before importing the active app. The zip must include a root `manifest.json` with target version, changed runtime files, payload hashes, and version-specific previous-file hashes. One latest asset can support skipped-release jump-ahead updates by listing multiple `from_versions`. The updater applies only verified `fp_analysis_app/` and `startup_update.py` payloads, skips local source edits detected by hash mismatch, and blocks dependency, packaging, build, or local-data paths that require a packaged refresh.

Remaining work:

- Build or stage a realistic distributable folder and confirm the bundled launcher can update the external `fp_analysis_app/` source folder before app import.
- Attach a real multi-baseline `fp_analysis_app_update_<version>.zip` asset to a test GitHub Release and confirm latest-release asset discovery works outside local fixtures.
- Decide how much update status should be visible in the GUI versus console output.

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

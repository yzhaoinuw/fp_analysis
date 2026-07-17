# Work Log

Prepend new session notes to this file. The live log holds at most the five most recent unique calendar dates. When a sixth date is added, move the oldest five dates together into `work_log_archive/work_log_<earliest>_to_<latest>.md`.

If today's date already appears at the top, add another `###` session under it rather than creating a duplicate date heading. Each substantive session needs compact model metadata and a `- Verification:` subsection containing commands that were actually run.

## 2026-07-17

### Publish event annotation save and undo (gpt-5)

- Prepared the accepted event annotation save/undo UI and MAT persistence changes for commit and push on `dev`.
- Verified the active desktop runtime imports, callback registration remains collision-free, focused tests pass, and the task-relevant tracked files are the only files being staged.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-17`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 48 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.event_editing, fp_analysis_app.mat_utils, fp_analysis_app.make_figure, fp_analysis_app.app_dev; print('import ok')"` - printed `import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 7 unique 7`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

## 2026-07-08

### Polish annotation action row placement (gpt-5)

- Fixed the main action row so `Analysis ->` is right-aligned instead of appearing before `Save Annotations`.
- Kept `Save Annotations` and `Undo Annotation` grouped on the lower left, and removed the visible status messages emitted by event-span selection, event deletion, and undo.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-08`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev; print('app import ok')"` - printed `app import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 7 unique 7`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestAnnotationModeEventDeletion` - 9 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.components_dev import save_div; children=save_div.children; print(save_div.style); print([getattr(c, 'id', None) for c in children]); print(children[0].children[0].id, children[0].children[1].id, children[2].id, children[2].style)"` - confirmed Save and Undo are in the left group and `analysis-link` has `justifySelf: end`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Implement event annotation save and undo (gpt-5)

- Consulted the cookbook undo/crash-recovery and save/export recipes, then applied the minimal active-runtime pieces needed for event timestamp annotations.
- Moved the main action row so `Save Annotations` sits at the lower left, `Undo Annotation` sits immediately to its right, and `Analysis ->` sits at the lower right.
- Added a one-step event timestamp undo history, app-owned MAT persistence fields for edited event timestamps, and saved-MAT reload preference for those fields over the original embedded `event` payload.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-08`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 48 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.event_editing, fp_analysis_app.mat_utils, fp_analysis_app.make_figure, fp_analysis_app.app_dev; print('import ok')"` - printed `import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 7 unique 7`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Publish annotation-rectangle event deletion (gpt-5)

- Prepared the accepted clientside annotation-rectangle event deletion flow for commit and push on `dev`.
- Verified the active desktop runtime still imports, callback duplicate-output registration is clean, focused analysis tests pass, and treaty documentation remains valid before staging.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-08`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 45 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.event_editing, fp_analysis_app.mat_utils, fp_analysis_app.make_figure, fp_analysis_app.app_dev; print('import ok')"` - printed `import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 6 unique 6`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

## 2026-07-07

### Fix duplicate callback signature regression (gpt-5)

- Fixed the blank-app duplicate-output regression caused by the new clientside callbacks sharing the same Dash duplicate-output input hashes as the existing pan and resampler callbacks.
- Kept the reliability changes by adding distinct clientside trigger signatures and guarding the callbacks so selection handling only responds to `graph.relayoutData` and event annotation updates only respond to `keyboard.n_events`.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev; print('app import ok')"` - printed `app import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('\n'.join(keys)); print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 6 unique 6`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 45 tests passed.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Stabilize event deletion and analysis counts (gpt-5)

- Changed the event annotation delete callback to trigger from `keyboard.n_events`, matching the older keyboard callbacks and making repeated Delete/Backspace presses reliable even when the key payload is unchanged.
- Changed selection-span capture to listen to Plotly `relayoutData` selection updates, with the resampler callback ignoring selection-only relayouts so drawing annotation rectangles is less fragile.
- Rebuilt the analysis-page event counts and tabs whenever `event-time-store` changes so deleted event timestamps are reflected before running analysis.
- Clarified the no-results tab text so removed event types are distinguished from event types that are still present but excluded by the selected baseline/analysis window.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-07`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 45 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev; print('app import ok')"` - printed `app import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 6 unique 6`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; import pprint; pprint.pp([(k, v['inputs'], v['state']) for k,v in app.callback_map.items() if 'box-select-store.data' in k or 'event-time-sync-store.data' in k])"` - confirmed selection capture uses `graph.relayoutData`, event updates use `keyboard.n_events`, and event-store sync updates `analysis-page.children`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.

### Rename event annotation clientside callback (gpt-5)

- Renamed the deletion-focused clientside callback convention comment from `remove_events_in_selected_span` to `update_event_annotations` so it can grow into future add/update/delete annotation behavior.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-07`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev; print('app import ok')"` - printed `app import ok`.
  - `rg -n "remove_events_in_selected_span|update_event_annotations|store_annotation_selection_span" fp_analysis_app\app_dev.py work_log.md` - confirmed only the new callback comment name remains in active code.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

## 2026-07-06

### Move event deletion to clientside rectangle flow (gpt-5)

- Moved annotation-rectangle event deletion from a server keyboard callback into clientside callbacks modeled after the cookbook keypress-annotation pattern.
- Added an always-present `event-time-store` so clientside deletion can remove event timestamps from browser state and vertical event-line traces immediately, while server-side analysis reads the same current event set.
- Added a small store-sync callback that mirrors `event-time-store` into the existing cache and clears stale export payloads after event edits.
- Normalized Plotly selections clientside so only the latest rectangle remains even if the user draws boxes in multiple subplots.
- Added JSON-friendly event-time store round-trip coverage.
- Moved the new clientside callbacks into the clientside callback section and named them with convention comments: `store_annotation_selection_span` and `update_event_annotations`.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-06`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev; print('app import ok')"` - printed `app import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('\n'.join(keys)); print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 6 unique 6`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 6 unique 6` after callback relocation.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; import pprint; pprint.pp([(k, v['inputs']) for k,v in app.callback_map.items() if 'box-select-store' in k or 'event-time-store' in k or any(i['property'] in ('selectedData','event') for i in v['inputs'])])"` - confirmed the selectedData and keyboard.event clientside callbacks are registered.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestAnalysisPageSignalHighlight tests.test_perievent_analysis.TestAnnotationModeEventDeletion tests.test_perievent_analysis.TestFullRecordingEventTimestampLines` - 13 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 45 tests passed after callback relocation.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.event_editing, fp_analysis_app.mat_utils, fp_analysis_app.make_figure, fp_analysis_app.app_dev; print('import ok')"` - printed `import ok` after callback relocation.
  - `git diff --check` - clean apart from Git line-ending conversion warnings after callback relocation.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

## 2026-07-04

### Implement annotation-rectangle event deletion (gpt-5)

- Revised item 4 from period-label removal to event-timestamp deletion inside an annotation-mode rectangle span.
- Added runtime-only helpers for extracting a selected rectangle's x/time span, recognizing `Delete`/`Backspace`, and removing cached event timestamps inside the selected span.
- Added active desktop callbacks that store `graph.selectedData` rectangle spans and, on `Delete`/`Backspace`, redraw the main full-recording figure so affected vertical event lines disappear.
- Cleared cached spreadsheet export payloads after event deletion so stale analysis cannot be saved.
- Fixed the analysis page's empty-event rendering path so removing the last event does not break tab creation.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-04`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestAnalysisPageSignalHighlight tests.test_perievent_analysis.TestAnnotationModeEventDeletion tests.test_perievent_analysis.TestFullRecordingEventTimestampLines` - 12 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev; print('app import ok')"` - printed `app import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "from fp_analysis_app.app_dev import app; keys=[k for k in app.callback_map if 'graph.figure' in k]; print('count', len(keys), 'unique', len(set(keys)))"` - printed `count 5 unique 5`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 44 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.event_editing, fp_analysis_app.mat_utils, fp_analysis_app.make_figure, fp_analysis_app.app_dev; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Refine event timestamp visualization (gpt-5)

- Kept imported and embedded event annotations as exact timestamp lines colored by the event palette, with one legend entry per event type.
- Replaced the built-in event legend samples with a custom annotation legend that keeps each colored square and event-name label in one text item so they share a baseline, shifts the event legend left when needed to keep common three-item legends on one row, and uses a negative per-item indent to tighten the gap between adjacent event legend items.
- Added a `show_period_labels` toggle to `make_figure()` so expanded perievent-window coloring remains available but can be suppressed cleanly.
- Disabled expanded perievent-window coloring for active event-annotation visualization paths so event timestamp lines are not drowned in the previous window heatmap.
- Added focused tests for palette/legend event-line behavior and the expanded-window coloring toggle.
- Expanded the GitHub Actions perievent-test dependency install to include Dash, Dash Extensions, Plotly, Plotly Resampler, and XlsxWriter after recent UI and figure tests made those imports part of test collection.
- Marked item 3 accepted after final visual check.
- Archived the previous five live work-log dates into `work_log_archive/work_log_2026-06-18_to_2026-07-03.md` before adding this July 4 entry.
- Verification:
  - `Get-Date -Format yyyy-MM-dd` - printed `2026-07-04`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestFullRecordingEventTimestampLines` - 4 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 38 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils, fp_analysis_app.make_figure, fp_analysis_app.app_dev; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

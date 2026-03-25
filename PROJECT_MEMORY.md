# fp_analysis Project Memory

## Purpose

`fp_analysis` is a desktop-first Dash application for viewing fiber photometry `.mat` files, overlaying event annotations, and generating perievent analysis figures and spreadsheets.

The repository currently contains two closely related app entrypoints:

- `fp_analysis_app/app_dev.py`: the desktop-oriented app that is currently launched by `run_desktop_app.py`
- `fp_analysis_app/app.py`: a parallel/newer-looking multipage app variant that still contains older upload/save behavior not fully carried into `app_dev.py`

## Current Runtime Path

- Desktop startup goes through `run_desktop_app.py`
- `run_desktop_app.py` imports `fp_analysis_app.app_dev`
- `app_dev.py` uses `pywebview` native file dialogs for `.mat` and annotation spreadsheet selection

Practical takeaway: for changes that affect the packaged desktop app, start by checking `app_dev.py` before `app.py`.

## Main Project Areas

- `fp_analysis_app/app_dev.py`: active desktop app flow
- `fp_analysis_app/app.py`: alternate app flow with older Dash uploader / save-export behavior
- `fp_analysis_app/components_dev.py`: UI components used by `app_dev.py`
- `fp_analysis_app/components.py`: UI components used by `app.py`
- `fp_analysis_app/event_analysis.py`: perievent event parsing, plotting, analysis, and spreadsheet-writing helpers
- `fp_analysis_app/postprocessing.py`: sleep bout table creation and sleep statistics spreadsheet logic
- `fp_analysis_app/make_figure.py`: signal visualization construction
- `fp_analysis_app/pages/`: page registration for the multipage app structure
- `fp_analysis_app/assets/figures/`: generated analysis figures
- `fp_analysis_app/assets/spreadsheets/`: generated analysis spreadsheets
- `data/`: sample `.mat` inputs and example annotation spreadsheet(s)

## Spreadsheet Output Flows

There are two distinct spreadsheet-related flows in the repo.

### 1. Perievent Analysis Spreadsheet Export

Core logic lives in `fp_analysis_app/event_analysis.py`.

Key classes / functions:

- `Event_Utils`: reads event spreadsheets, builds event windows, and creates perievent label overlays
- `Analyses.get_perievent_analyses()`: computes normalized traces and summary metrics
- `Perievent_Plots.write_spreadsheet()`: writes stats-style sheets
- `Perievent_Plots.write_mean_perievent_sheet()`: writes time-series mean trace sheets

Current behavior differs by app entrypoint:

- `app_dev.py`
  - This is the spreadsheet path most relevant to the current desktop app
  - Export flow is now split into two phases:
    1. compute/store per-signal perievent results for each event
    2. build/write spreadsheet exports through an `export_specs` registry in `make_analysis_plots()`
  - Current export registry entries:
    - mean trace workbook per signal: `<signal>_bw<baseline>_aw<analysis>.xlsx`
    - AUC workbook per signal: `<signal>_auc_bw<baseline>_aw<analysis>.xlsx`
    - max peak magnitude workbook per signal: `<signal>_max_peak_magnitude_bw<baseline>_aw<analysis>.xlsx`
  - In both workbook types:
    - each event type gets its own sheet
    - each subject appends as a new column
  - Mean trace sheets use `time_s` as the compatibility key and require matching downsampled time axes
  - AUC and max peak magnitude sheets use `event_index` as the row key so subjects with different event counts can still append cleanly

- `app.py`
  - Creates one workbook per event
  - File pattern: `<mat_name>_<event>_bw<baseline>_aw<analysis>.xlsx`
  - Writes event summary metrics using `write_spreadsheet()`

### 2. Sleep Bout / Manual Annotation Export

This is separate from the perievent analysis spreadsheet output.

- `fp_analysis_app/app.py` contains `save_annotations()`
- When manual scoring is complete, it writes `<temp_mat>_table.xlsx`
- Sheets:
  - `Sleep_bouts`
  - `Sleep_stats`

Related helper functions live in `fp_analysis_app/postprocessing.py`:

- `get_sleep_segments()`
- `get_pred_label_stats()`

Note: this save/export path exists in `app.py` but is not currently implemented in the same way in `app_dev.py`.

## Important Repo Understanding

- `app_dev.py` appears to be the newer desktop-focused runtime path.
- `app.py` is not simply obsolete; it still contains functionality that `app_dev.py` does not, especially the manual annotation save/export flow.
- Because of that, the repository is in a split state rather than a clean old/new replacement.
- When making spreadsheet changes, verify which app path the change is meant to affect before editing.

## Known Samples / Artifacts

- Example annotation spreadsheet: `data/Transitions_F268.xlsx`
- Example generated spreadsheet: `fp_analysis_app/assets/spreadsheets/F268_bw30_aw60.xlsx`
- Example `.mat` inputs live under `data/`

These are useful for validating spreadsheet structure after changes.

## Tests

- Perievent export coverage lives in `tests/test_perievent_analysis.py`
- That test file now covers:
  - event filtering / extraction against the `F268` fixture
  - AUC reference values
  - mean trace workbook append behavior
  - occurrence-value workbook alignment for AUC and max peak magnitude

## Local Execution Note

- For this repo, use the `fiber_photometry` conda env interpreter when running tests from Codex:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe`
- A known-good verification command is:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis`
- The base Miniconda Python and the `sleep_scoring_dist` env were missing dependencies needed for this test path during the last run.

## Recommended Starting Points For Future Work

If the next task is about spreadsheet output:

- Check `run_desktop_app.py` to confirm which app path is active
- Inspect `fp_analysis_app/app_dev.py` first for desktop export behavior
- Inspect `fp_analysis_app/event_analysis.py` for the actual workbook/sheet writing logic
- Compare with `fp_analysis_app/app.py` if behavior seems inconsistent or partially migrated
- For new export types in the desktop app, prefer adding a new entry to the `export_specs` registry in `app_dev.py` instead of expanding the analysis loop inline

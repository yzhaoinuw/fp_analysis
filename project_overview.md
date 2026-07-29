# Project Overview

## What This Repo Is

`fp_analysis` is a desktop-first Dash application for viewing fiber-photometry MATLAB files, overlaying event annotations, computing perievent analyses, and exporting analysis workbooks for downstream tools such as GraphPad Prism.

Users launch a small Python/pywebview desktop wrapper. The UI runs locally in Dash, opens native file dialogs, visualizes continuous signals, imports event or sleep-bout spreadsheets, and produces figures plus selectively chosen spreadsheet outputs. The current application version is defined in `fp_analysis_app/__init__.py`.

## Active Runtime Path

### 1. Desktop entrypoint

[`run_desktop_app.py`](run_desktop_app.py)

- Starts the Dash server on the configured localhost port in a background thread.
- Opens the application in a pywebview window.
- Runs the shared source updater with app-specific settings from `startup_update_config.py` before importing application code.
- Uses API-free latest-release discovery plus a per-user state file to limit ordinary checks to once per 24 hours and preserve GitHub rate-limit backoff.
- Imports `fp_analysis_app.app_dev`, making that module the active desktop app.
- Adds the app version to the native window title.
- Supports headless `--smoke` package-import checks and forced `--check-update` installed-update checks that bypass the normal interval.

### 2. Active Dash application

[`fp_analysis_app/app_dev.py`](fp_analysis_app/app_dev.py)

- Owns file selection, filesystem cache state, page callbacks, visualization setup, analysis execution, and spreadsheet-save callbacks.
- Uses native pywebview dialogs for `.mat`, `.xlsx`, and `.csv` selection.
- Separates analysis generation (`Show Results`) from selected workbook writing (`Save Spreadsheets`).
- Creates local figure, video, spreadsheet, and cache directories as needed.

### 3. UI and pages

- [`fp_analysis_app/components_dev.py`](fp_analysis_app/components_dev.py) builds active controls, tabs, modals, and visualization containers.
- [`fp_analysis_app/pages/main_page.py`](fp_analysis_app/pages/main_page.py) registers the main visualization page.
- [`fp_analysis_app/pages/analysis_page.py`](fp_analysis_app/pages/analysis_page.py) registers the analysis page.

### 4. Visualization and data normalization

- [`fp_analysis_app/make_figure.py`](fp_analysis_app/make_figure.py) builds the interactive signal figure.
- [`fp_analysis_app/mat_utils.py`](fp_analysis_app/mat_utils.py) normalizes signal-name/frequency metadata and provides the NE-only fallback.

### 5. Event analysis and exports

- [`fp_analysis_app/event_analysis.py`](fp_analysis_app/event_analysis.py) reads events, extracts perievent windows, computes signal metrics and cross-correlation, creates plots, and updates workbook sheets.
- [`fp_analysis_app/analysis_export.py`](fp_analysis_app/analysis_export.py) defines selectable export types, workbook names, and primary/fallback save behavior.
- [`fp_analysis_app/export_settings.py`](fp_analysis_app/export_settings.py) creates setup-specific output folders and maintains `data_description.txt`.
- [`fp_analysis_app/sleep_event_import.py`](fp_analysis_app/sleep_event_import.py) converts sleep-bout tables into transition events.

## Analysis and Export Model

Each `.mat` file represents one subject with continuous synchronized signals. Each event type can occur multiple times. For a selected signal and event:

1. Event occurrences too close to the recording boundaries are filtered out.
2. A fixed baseline-plus-analysis window is extracted around every occurrence.
3. Per-occurrence signals are normalized and analyzed.
4. Figure results and export-ready DataFrames are cached.
5. The user selects which workbook types to write.

Current signal export types are:

- mean trace
- AUC
- max peak magnitude
- first peak time
- decay time

Two-signal analyses can additionally export:

- mean cross-correlation
- strongest cross-correlation lag

Exports are grouped by sorted signal names, baseline window, and analysis window. The app first writes beside the selected MAT file and falls back to `fp_analysis_app/assets/spreadsheets/` when necessary. Re-exporting the same subject/setup replaces that subject's columns; additional subjects append to the same workbook set.

Mean-trace and mean cross-correlation sheets use Prism-friendly `<subject>_mean`, `<subject>_sd`, and `<subject>_n` columns. Occurrence-level exports use `event_index`.

## Repo Structure Map

```text
fp_analysis/
|- AGENTS.md
|- project_overview.md
|- next_steps.md
|- work_log.md
|- work_log_archive/
|- README.md
|- CHANGELOG.md
|- run_desktop_app.py
|- startup_update_config.py
|- tools/
|  `- build_update_asset.py
|- packaging/
|  `- windows/
|     |- app.spec
|     |- make_full_app_zip.ps1
|     |- smoke_check_release.ps1
|     `- README.md
|- fp_analysis_app/
|  |- app_dev.py
|  |- components_dev.py
|  |- event_analysis.py
|  |- analysis_export.py
|  |- export_settings.py
|  |- make_figure.py
|  |- mat_utils.py
|  |- sleep_event_import.py
|  |- pages/
|  `- assets/
|- tests/
|  |- test_perievent_analysis.py
|  |- test_startup_update.py
|  `- test_packaging.py
|- .github/workflows/perievent-tests.yml
|- requirements.txt
|- environment.yml
|- data/
|- cache/
|- build/
`- dist/
```

## What Looks Active vs. Legacy

### Active / relevant now

- `run_desktop_app.py`
- `fp_analysis_app/app_dev.py`
- `fp_analysis_app/components_dev.py`
- `fp_analysis_app/pages/`
- `fp_analysis_app/event_analysis.py`
- `fp_analysis_app/analysis_export.py`
- `fp_analysis_app/export_settings.py`
- `fp_analysis_app/make_figure.py`
- `packaging/windows/`
- `fp_analysis_app/mat_utils.py`
- `fp_analysis_app/sleep_event_import.py`
- `tests/test_perievent_analysis.py`
- `.github/workflows/perievent-tests.yml`

### Secondary, historical, or local-draft surfaces

- `fp_analysis_app/app.py` and `components.py` are a secondary browser/upload app path. They are not the packaged desktop entrypoint, but `app.py` still contains manual annotation save behavior absent from `app_dev.py`.
- `fp_analysis_app/preprocessing.py`, `postprocessing.py`, and `make_mp4.py` are inherited utilities used by older sleep-scoring-oriented flows or specialized tasks. Confirm call sites before editing or removing them.
- `fp_analysis_app/sketch*.py`, `preprocessing_sketch.py`, and similarly named local files are experiments or prototypes, not the production entrypoint.
- Untracked files such as `event_analysis_dev.py` and cross-correlation sketches may contain ongoing local work. Do not assume they are disposable.
- `setup.py`, `environment.yml`, and `change_log.txt` originated in the earlier sleep-scoring lineage. The obsolete changelog was removed during treaty migration; the remaining packaging metadata should be modernized only as a deliberate task.

## Tests and Fixtures

The focused suite is [`tests/test_perievent_analysis.py`](tests/test_perievent_analysis.py). It uses `unittest` and covers:

- sleep-bout table detection and transition conversion
- NE-only MAT metadata fallback
- event filtering and perievent extraction
- AUC and peak/decay metrics
- mean-trace and cross-correlation workbook layouts
- multi-subject append and same-subject overwrite behavior
- setup-folder naming and `data_description.txt`
- selective export and remembered checklist choices

Run it with:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis
```

Canonical local fixtures:

- `data/F268.mat`
- `data/Transitions_F268.xlsx`

The fixture-dependent integration class skips when those files are absent; synthetic export tests still run. GitHub Actions runs the same module on Python 3.10 with a minimal dependency set and `MPLBACKEND=Agg`.

## User Data Expectations

### MAT input

Normal fiber-photometry input contains:

- `fp_signal_names`: one signal name or a list/array of names
- `fp_frequency`: numeric sampling frequency
- one numeric array for every listed signal name

Supported visualization fallback:

- `ne`: numeric NE signal
- `ne_frequency`: numeric NE sampling frequency

Other optional fields, such as sleep labels and video timing metadata, are used by specialized visualization paths when present.

### Annotation input

Annotation input accepts `.xlsx` and `.csv` files. The file extension selects the
matching pandas reader, and both file types use the same event-format detection
and filtering rules.

Two annotation spreadsheet formats are supported:

```text
transition table -> event names as columns -> event times in seconds
sleep-bout table -> sleep_scores/start/end/duration -> transition events
```

Event-table format uses one column per event type with event times in seconds. Empty cells are ignored, and events outside the valid baseline/analysis boundaries are filtered.

Sleep-bout tables are detected by case-insensitive columns:

- `sleep_scores`
- `start`
- `end`
- `duration`

Sleep scores may use `0-3` or `1-4`; consecutive state changes become transition events such as `wake_nrem`.

## Generated and Local Artifacts

- `cache/` and the operating-system temp cache hold callback/export payloads.
- `fp_analysis_app/assets/figures/`, `videos/`, and `spreadsheets/` hold generated outputs or fallback exports.
- `data/` contains local samples and can be large.
- `build/` and `dist/` are packaging outputs.
- `.worktrees/` contains linked Git worktrees.

These locations can contain valuable local evidence. Inspect scope before cleanup.

## Practical Mental Model

For most product work, read files in this order:

1. [`AGENTS.md`](AGENTS.md)
2. [`next_steps.md`](next_steps.md)
3. [`run_desktop_app.py`](run_desktop_app.py)
4. [`fp_analysis_app/app_dev.py`](fp_analysis_app/app_dev.py)
5. The focused helper for the task:
   - export orchestration: `analysis_export.py` / `export_settings.py`
   - calculations and workbook sheets: `event_analysis.py`
   - visualization: `make_figure.py` / `mat_utils.py`
   - UI: `components_dev.py`
6. [`tests/test_perievent_analysis.py`](tests/test_perievent_analysis.py)
7. [`work_log.md`](work_log.md) for recent decisions and verification

## Questions Worth Clarifying Later

- Should manual annotation save/export behavior from `app.py` be migrated into `app_dev.py`, or should the secondary app remain supported?
- Should `environment.yml` and `setup.py` be replaced or corrected to reflect the current `fp_analysis` product rather than its sleep-scoring ancestry?
- The README documents a stale-results UI workaround after changing the second signal. Is that still reproducible in the current beta, and which callback should own the refresh?

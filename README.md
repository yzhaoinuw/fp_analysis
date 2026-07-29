# FP Analysis

[![Agent Collab Treaty](https://img.shields.io/badge/Agent_Collab_Treaty-adopted-6d81f1?style=flat-square)](https://github.com/yzhaoinuw/agent_collab_treaty)

## Installation
1. Go to the [fp_analysis folder](https://uofr-my.sharepoint.com/:f:/r/personal/yzhao38_ur_rochester_edu/Documents/fp_analysis_project?csf=1&web=1&e=JIeIs6) on OneDrive. Contact Yue if you can't access it.
2. Download the current **fp_analysis_app_vX.Y.Z-windows.zip** package. Note that if you unzip it to the same location where the zip file is, you may end up with a nested folder. If this happens, move the inner app folder somewhere else and delete the empty outer folder.
3. Examine the unzipped **fp_analysis_app_vX.Y** folder. The folder name stays on the major/minor release line because patch-level source updates are applied in place. It should contain **_internal**, **fp_analysis_app**, **run_fp_analysis_app.exe**, and **unblock_app.cmd**.
4. If the package contains `unblock_app.cmd`, double-click it to unblock the downloaded files and start the app. For an older package without that helper, open PowerShell and run:
```bash
cd PATH_TO_YOUR_APP_FOLDER
Get-ChildItem -Recurse | Unblock-File
```
The first line navigates to the unzipped app folder. You need to supply the actual path to the app folder on your computer. The second line unblocks the dependencies related to webview, which provides the window to the app interface, as Windows may block them when the zip file is downloaded. Without unblocking them, the app may not start.


## Features And Usage 
To open the app, double click "run_fp_analysis_app.exe" and it will open the app's home page in a tab in your web browser. You don't need internet connection to run the app. The app only uses the web browser as the interface.

### Startup Source Updates

Starting with a full package that includes the shared updater dependency, the app checks its latest GitHub Release for a verified source-update zip before loading the interface. Ordinary launches make at most one release check per 24 hours using a small per-user state file; an explicit `--check-update` run bypasses that interval. Internet access is optional: if GitHub is unavailable or no compatible source asset exists, startup continues with the installed version.

Source-update zips can replace only approved files under `fp_analysis_app/`. They verify downloaded hashes and known installed-file baselines, support jumping over compatible skipped versions, and refuse to overwrite unrecognized local edits. Dependency, packaging, build, generated-data, deletion, and rename changes still require a normal full packaged release.

### Visualization
Click "**Click here to select File**". After you select a file, the app will briefly validate the file selected and then show you the figures of the signals. Once the figures are shown, you can navigate and zoom in or out on the figures.

#### Zooming and Navigation
**Zooming** is done by scrolling your mouse around the center of interest. **Navigation** can be done using the left/right arrow key or just dragging the figure. To zoom or navigate along X-Axis only, move your cursor to the last figure and then operate. To zoom in or navigate along Y-Axis only, move your cursor to the left edge of the figure of interest and operate. 

### Analysis Spreadsheet Export
After annotations are loaded, go to the analysis page, choose the signal(s) and analysis windows, and click `Show Results`. This generates the analysis figures and prepares spreadsheet-ready results without writing spreadsheet files.

To export spreadsheets, click `Save Spreadsheets`, check the analysis types to save, and confirm. The app writes only the selected workbook types.

#### Analysis Plots And Metrics

Each event tab shows the selected event type over the chosen baseline and analysis windows:

- Raw perievent traces: one trace per event occurrence, aligned to the event at time 0.
- Mean trace: the average raw perievent response across all included occurrences.
- Perievent heatmap: one row per included event occurrence, with color showing signal amplitude over time.
- Normalized perievent traces: baseline-normalized traces used for the metric plots and spreadsheet values.
- AUC: the app's post-event response summary, currently calculated as the mean normalized signal value across the analysis window.
- Positive peak value: the signal value at the first detected positive peak. If no positive peak is detected, the value is `NaN`.
- Negative peak value: the signal value at the first detected negative peak. If no negative peak is detected, the value is `NaN`.
- First peak time: seconds from the event to the first detected positive peak.
- Decay time: seconds from the first detected positive peak until the response returns below its baseline mean; if that return is not found, the last analysis-window time is used.
- Cross-correlation: when two signals are selected, the app also plots their mean normalized cross-correlation across event occurrences.

The app first tries to save the spreadsheets next to the selected `.mat` file. If that location is not writable, it saves them to the app spreadsheet folder under `fp_analysis_app/assets/spreadsheets/`.

Spreadsheet exports are grouped into a setup-specific subfolder based on:

- the selected signal or signal pair
- the baseline window size
- the analysis window size

For the same analysis setup, different subjects append into the same workbook set. If the same subject is exported again with the same setup, that subject's columns are updated instead of duplicated. Each export folder also includes a `data_description.txt` file that records the analyzed data source, setup details, and saved analysis types.

### Annotation File Formats
Annotation files may be `.xlsx` or `.csv`. The app accepts two event formats:

```text
Transition table
  columns = event names
  cells   = event times in seconds
  example = wake_sws, sws_REM, REM_wake

Sleep-bout table
  columns = sleep_scores, start, end, duration
  action  = converted internally into transition events
  example = wake_nrem, nrem_rem, rem_wake
```

In both formats, empty cells are ignored and events too close to the beginning or end of the signal are skipped based on the selected baseline and analysis windows.

### Known Issues
1. When adding a second or removing a second signal, clicking Show Results may not update the results even when the analyses are done (indicated by the Show Results button becomes available again), when this happens, switch to any other tab. That should help update the results. Fix to this issue is in progress.


## Input File 
The input files to the app must be .mat (matlab) files preprocessed using [sleep_data_preprocessing_app](https://github.com/yzhaoinuw/sleep_data_preprocessing_app) (although called sleep data, it also works with fp data), and contain the following fields. 
### Required Fields
| Field Name            | Data Type      |
| ----------------------|----------------|
| **_fp_signal_names_** | *N* x 1 string |
| **fp_frequency**      | float          |
| **_signal_A_**        | 1 x *N* single |
|  ...                  | 1 x *N* single | 

**Explanations**
1. **_signal_A_**  is just an example name for a fp signal. You can name it by its real signal name such as NE, but it must be listed as a name in **_fp_signal_names_**, in order to be visualized by the app.

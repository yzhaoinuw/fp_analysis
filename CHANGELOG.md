# Changelog

## v0.6.0

- check for safe source-only updates before desktop startup while preserving the
  installed version if validation or replacement fails
- distribute a repeatable Windows package with an unblock-and-start helper and
  verified fresh-extraction startup
- detect and plot negative peaks alongside positive peaks, and include both peak
  values in spreadsheet exports
- show event timestamps on the full recording and support rectangle deletion,
  save, and undo for event annotations
- improve analysis-window controls, heatmap labels, signal-selection guidance,
  and analysis status messages

## v0.5.0

- skip event times whose selected analysis window would exceed the available signal samples
- document the two accepted annotation spreadsheet formats: transition tables and sleep-bout tables
- keep the M67 demo script path aligned with current example data

## v0.5.0-beta

- separate analysis generation from spreadsheet export with a dedicated `Save Spreadsheets` flow
- let users choose which spreadsheet workbook types to save from a checklist modal
- keep spreadsheet save controls disabled while workbook exports are running
- remember the last selected spreadsheet export types across files
- reset stale analysis controls when switching to a new `.mat` file

## v0.4.1

- import sleep-bout spreadsheets by converting 1-based or 0-based state codes into transition events
- keep single-signal visualization imports working by attaching event legend entries to the first subplot row

## v0.4.0

- add grouped spreadsheet export by analysis setup
- save analysis spreadsheets beside the input `.mat` file when possible
- fall back to the app spreadsheet folder when the input folder is not writable

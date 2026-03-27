# Agent Instructions For fp_analysis

Read `PROJECT_MEMORY.md` before doing any meaningful work in this repo.

## Required Startup Checklist

Before running commands, editing files, or answering environment questions:

1. Open `PROJECT_MEMORY.md`.
2. Use the repo's known-good Python interpreter explicitly:
   - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe`
3. Prefer this test command when validating the perievent suite:
   - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis`
4. Do not assume `python`, `py`, or `conda` are on `PATH`.
5. In PowerShell, do not assume `&&` works. Use separate commands or PowerShell-native sequencing.

## Repo Reminders

- The active desktop runtime path is `run_desktop_app.py` -> `fp_analysis_app/app_dev.py`.
- For desktop export behavior, inspect `fp_analysis_app/app_dev.py` before `fp_analysis_app/app.py`.
- This repo may intentionally contain many untracked or draft files. Do not treat a dirty worktree as unexpected.
- When making focused changes, stage only the files relevant to the task.

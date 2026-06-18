# Work Log

Prepend new session notes to this file. The live log holds at most the five most recent unique calendar dates. When a sixth date is added, move the oldest five dates together into `work_log_archive/work_log_<earliest>_to_<latest>.md`.

If today's date already appears at the top, add another `###` session under it rather than creating a duplicate date heading. Each substantive session needs compact model metadata and a `- Verification:` subsection containing commands that were actually run.

## 2026-06-18

### Adopt the Agent Collab Treaty documentation (gpt-5)

- Replaced treaty placeholders with repository-specific runtime, test, branch, data-contract, active-versus-legacy, and packaging guidance.
- Migrated durable context from the pre-treaty `PROJECT_MEMORY.md`, completed export plan, Git history, and archived agent instructions into the treaty overview, roadmap, and work-log archive.
- Retired parallel agent-memory documents and the unrelated legacy sleep-scoring changelog while keeping the user-facing `README.md` and product `CHANGELOG.md`.
- Added the treaty adoption badge to the README.
- Recorded that the annotation dialog advertises CSV files while the active reader still uses Excel parsing, leaving CSV support as an explicit beta follow-up.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 30 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate . --migration-hints` - passed.
  - `git diff --check` - clean.

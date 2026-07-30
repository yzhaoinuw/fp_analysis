# Work Log

Prepend new session notes to this file. The live log holds at most the five most recent unique calendar dates. When a sixth date is added, move the oldest five dates together into `work_log_archive/work_log_<earliest>_to_<latest>.md`.

If today's date already appears at the top, add another `###` session under it rather than creating a duplicate date heading. Each substantive session needs compact model metadata and a `- Verification:` subsection containing commands that were actually run.

## 2026-07-30

### Move packaged-app installation to GitHub Releases (gpt-5)

- Replaced the OneDrive installation path with direct public GitHub Releases
  instructions: Windows users now look in a release's Assets for the app ZIP
  ending in `_full.zip` and explicitly avoid GitHub's generated source archives.
- Changed future full-package output to the stable
  `fp_analysis_app_vX.Y_full.zip` convention while retaining the existing
  `fp_analysis_app_vX.Y` extracted folder. Confirmed that the exact patch
  version remains visible in the native window title, `--smoke`, and
  `--check-update` output.
- Audited the updater 0.2.0 installed in the distributed `v0.6.0` baseline. Its
  normal redirect discovery constructs the exact
  `fp_analysis_app_update_<tag>.zip` name, so the compatible prefix-style update
  asset remains unchanged. Recorded suffix-style discovery as a shared-updater
  and next-full-baseline task.
- Relabeled the existing verified `v0.6.0` Windows archive without rebuilding
  it. The new `fp_analysis_app_v0.6_full.zip` retains SHA-256
  `75CD2BEC810B0F1A91601EC3D9BE1EEB870F1993CE84755BD12CCC194386A7E1`,
  contains 3,335 entries under the single `fp_analysis_app_v0.6` root, and has
  corrected manifest/checksum sidecars in both the task worktree and the
  primary checkout's `release_artifacts/`.
- Uploaded and verified all four replacement assets before deleting the four
  stale `fp_analysis_app_v0.6.0-windows.zip` assets from GitHub. The deleted
  assets remain recoverable from the unchanged local copies. The release title
  and tag remain exactly `v0.6.0`, and the release body now gives direct
  `_full.zip` installation instructions.
- Verification:
  - PowerShell parser checks for `make_full_app_zip.ps1` and `smoke_check_release.ps1` - passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 85 tests passed with 8 private-fixture skips.
  - `$env:FP_ANALYSIS_SKIP_UPDATE='1'; C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe run_desktop_app.py --smoke` - printed `smoke ok: v0.6.0`.
  - Active analysis-module import smoke - printed `import ok v0.6.0`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe packaging\windows\package_folder_name.py --full-zip-name v0.6.0` - printed `fp_analysis_app_v0.6_full.zip`.
  - Local ZIP/hash/manifest/root audit - confirmed byte identity with the original archive, corrected sidecar references, the expected stable root, and 3,335 entries.
  - GitHub release page and public API audit - confirmed exactly the four replacement assets, matching local SHA-256 digests, tag-only title, published/latest production status, and updated install text.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Restore the Perievent Tests workflow (gpt-5)

- Traced the repeated GitHub Actions failures to the curated CI dependency
  install omitting `pywebview`, even though the startup-update tests import
  `run_desktop_app.py` and therefore its top-level `webview` dependency.
- Added the runtime's existing `pywebview==6.1` pin to the Perievent Tests
  workflow without expanding the workflow to the full application dependency
  set.
- Pushed workflow fix `65bbd35` to `dev`; GitHub run
  `30515206787` completed successfully after the preceding runs had failed with
  `ModuleNotFoundError: No module named 'webview'`.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_startup_update.TestStartupUpdate.test_launcher_forces_explicit_update_checks` - 1 test passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 84 tests passed.
  - Active-module, launcher, and `webview` import smoke - printed `import ok`.
  - `git diff --check` - passed.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.
  - `gh run watch 30515206787 --exit-status` - Perievent Tests passed on Ubuntu in 45 seconds.

## 2026-07-29

### Correct the v0.6.0 local handoff and release title (gpt-5)

- Copied the clean `v0.6.0` Windows ZIP and its three sidecars from the
  dedicated `main_release` worktree into the primary checkout's
  `release_artifacts/` folder so the distributable is visible in the expected
  local location.
- Renamed the GitHub Release title from `FP Analysis v0.6.0` to exactly
  `v0.6.0` for consistency while leaving the release tag, notes, assets, and
  package bytes unchanged.
- Verification:
  - SHA-256 comparison of all four source/destination artifacts - byte-for-byte matches.
  - Primary artifact ZIP SHA-256 - `75CD2BEC810B0F1A91601EC3D9BE1EEB870F1993CE84755BD12CCC194386A7E1`.
  - GitHub release inspection - confirmed title/tag `v0.6.0`, non-draft, non-prerelease, with all four assets still uploaded.

### Publish the canonical v0.6.0 user release (gpt-5)

- Committed the CSV annotation import, stable major/minor package folder,
  runtime version, release notes, and deferred follow-up plan as release commit
  `03987f3235c1eb8949f1d9f505e5775d31823160`.
- Pushed `dev`, fast-forwarded and pushed `main`, replaced the obsolete local
  `v0.6.0` tag with an annotated tag at the release commit, and removed the
  obsolete local-only `v0.6.1` tag so that version is available for a future
  update. Neither legacy tag had a GitHub Release or remote ref.
- Built the full Windows package from clean tracked `main` bytes. The artifact
  retains the full `fp_analysis_app_v0.6.0-windows.zip` filename while its only
  archive root is the update-stable `fp_analysis_app_v0.6` folder.
- Published a non-draft, non-prerelease GitHub Release with the Windows ZIP,
  SHA-256 sidecar, provenance manifest, and frozen build-environment
  requirements. GitHub's latest-release redirect now resolves to `v0.6.0`.
- Verification:
  - Clean `powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1` - passed dependency checks, 84 tests with 8 private-fixture skips, PyInstaller, release-structure smoke, packaged smoke, fresh-extraction smoke, hashing, and sidecar generation.
  - The clean manifest records `version: v0.6.0`, `package_folder: fp_analysis_app_v0.6`, `git_commit: 03987f3...`, `tracked_worktree_dirty: false`, `source_export: git-blob-bytes`, and updater 0.2.0.
  - Local SHA-256 recomputation and the uploaded GitHub asset digest matched `75CD2BEC810B0F1A91601EC3D9BE1EEB870F1993CE84755BD12CCC194386A7E1`.
  - ZIP inspection - confirmed one top-level folder, `fp_analysis_app_v0.6`, containing 3,335 entries.
  - GitHub release inspection - confirmed all four assets uploaded and release `https://github.com/yzhaoinuw/fp_analysis/releases/tag/v0.6.0` is published, non-draft, and non-prerelease.
  - Latest-release HEAD request - returned `302` to `https://github.com/yzhaoinuw/fp_analysis/releases/tag/v0.6.0`.
  - Remote-ref inspection - confirmed `origin/dev`, `origin/main`, and `v0.6.0^{}` matched `03987f3`; no `v0.6.1` remote ref exists.

### Prepare the canonical v0.6.0 release (gpt-5)

- Set the runtime version to `v0.6.0` and consolidated the final user-facing
  release notes across the updater, annotation, analysis, and package-folder
  changes.
- Moved the stale two-signal analysis refresh and first source-only updater
  trial into explicitly deferred future-update work, with the distributed
  `v0.6.0` package as the new updater 0.2.0 baseline.
- Confirmed that the obsolete local `v0.6.0` and `v0.6.1` tags have no matching
  remote refs or GitHub Releases; release delivery will replace `v0.6.0` and
  remove `v0.6.1`.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 84 tests passed.
  - `$env:FP_ANALYSIS_SKIP_UPDATE='1'; C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe run_desktop_app.py --smoke` - printed `smoke ok: v0.6.0`.
  - Active analysis-module import smoke - printed `import ok v0.6.0`.
  - Local/remote tag and GitHub Release inspection - confirmed only local legacy `v0.6.0` and `v0.6.1` tags, with no corresponding GitHub Releases.
  - `git diff --check` - passed with only line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Keep the packaged app folder on the release line (gpt-5)

- Split Windows package naming so full-version artifact files retain release
  provenance while the extracted and `dist/` app folder uses the stable
  major/minor release line, such as `fp_analysis_app_v0.6`.
- Added one shared naming helper used by both the PowerShell builder and
  PyInstaller spec, recorded the package folder in the manifest, and added
  regression coverage for stable patch/prerelease naming and invalid versions.
- Updated the installation, packaging, and changelog wording to distinguish the
  full-version ZIP name from its major/minor root folder.
- Produced an ignored dirty-worktree rehearsal artifact at
  `release_artifacts/fp_analysis_app_v0.6.0-dev2-windows.zip`; its only archive
  root is `fp_analysis_app_v0.6`, and its manifest records the same package
  folder. The rehearsal SHA-256 is
  `E1A2A4309B2C198CDF6A599986ADC02CDE844EA2C5F7C6D5F6AD0FA6519B70E1`.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_packaging` - 4 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fp_analysis_dist\python.exe packaging\windows\package_folder_name.py v0.6.0-dev2` - printed `fp_analysis_app_v0.6`.
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1 -AllowDirty` - passed dependency checks, 84 tests, PyInstaller, release-structure smoke, packaged smoke, fresh-extraction smoke, hashing, and sidecar generation.
  - ZIP root and manifest inspection - confirmed `fp_analysis_app_v0.6` while the artifact retained the full `v0.6.0-dev2` version.
  - `git diff --check` - passed with only line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Support CSV annotation imports (gpt-5)

- Matched the active desktop file-dialog contract by routing `.csv` annotation
  paths through `pandas.read_csv()` while retaining the existing Excel reader
  for `.xlsx` paths.
- Added focused file-backed coverage for transition-table and sleep-bout CSV
  imports, including case-insensitive extension handling, plus an Excel-reader
  regression check.
- Updated the README, changelog, project overview, and active next steps to
  record CSV as supported and close the annotation-file mismatch.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestAnnotationFileImport` - 3 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 82 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev, fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - passed.
  - `git diff --check` - passed with only line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

## 2026-07-28

### Publish the `v0.6.0-dev2` updater 0.2.0 baseline (gpt-5)

- Committed the updater integration and release metadata, pushed `auto-update`, and fast-forwarded `dev` and `main` to release commit `d65283828e0fbb55f36e6fb0935dbc26f37b7a75`.
- Built the full Windows package from clean tracked `main` bytes with updater 0.2.0, then confirmed the provenance manifest records `tracked_worktree_dirty: false` and `source_export: git-blob-bytes`.
- Published the annotated `v0.6.0-dev2` tag and a non-draft GitHub prerelease with the Windows ZIP, checksum, provenance manifest, and frozen build-environment requirements.
- Kept the following normal source-update release version undecided because historical tag `v0.6.1` is already taken; the two existing app issues remain deferred for closer review.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 79 tests passed.
  - Import and `py_compile` checks for the active analysis modules, launcher, updater config, and version module - passed.
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1` - passed dependency checks, 79 tests, PyInstaller, structure smoke, packaged smoke, fresh-extraction smoke, hashing, and sidecar generation.
  - Artifact SHA-256 recomputation matched `902912718AEA3B8C42F1F2EEA5CDB8791937EDE17685F5054D017E6519DA99A6`.
  - GitHub release inspection - confirmed `v0.6.0-dev2` is a published prerelease with all four assets uploaded and the ZIP digest matching the local artifact.
  - Remote-ref inspection - confirmed `auto-update`, `dev`, and `main` matched the release commit at publication and the annotated tag exists on origin.
  - `git diff --check` and `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Adopt durable startup checks from updater 0.2.0 (gpt-5)

- Compared the existing updater pin at `adb2221` with shared-package `main` commit `85bb68e` and adopted only the new downstream requirements: API-free latest-release discovery, an app-specific per-user check-state file, 24-hour throttling and persisted rate-limit backoff, update-available status, and forced explicit checks.
- Pinned updater 0.2.0 at full commit `85bb68e42155ad8b661f669df8d281fbe683a045` in both the runtime requirements and CI, then installed that exact revision in the `fiber_photometry` and `fp_analysis_dist` environments.
- Moved normal discovery to GitHub's ordinary `/releases/latest` redirect, stored durable state under the user's local application-data folder, and made `--check-update` pass `force_check=True` so package gates and troubleshooting are never suppressed by the normal interval.
- Recorded that normal latest-release discovery excludes GitHub prereleases; prerelease-only gates should use the direct update-zip override.
- Kept `tools/build_update_asset.py` unchanged because it already forwards the shared builder's multiple-baseline and schema-2 arguments. No Python config-merge path was declared because this app has no updater-managed user-editable Python configuration.
- Confirmed that the published `v0.6.0-dev` package contains updater 0.1.0; updater 0.2.0 therefore requires a new full package before the following source-only release can test it.
- Produced an ignored dirty-worktree rehearsal ZIP with updater 0.2.0 and matching SHA-256 `AE5A68A7E2A2B8B7A5C303B6B98F2E358A1CD6803199B4753749C8C5FD76EEB9`; it is packaging evidence, not a release candidate.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m pip install --no-deps --disable-pip-version-check --force-reinstall "desktop-app-source-updater @ git+https://github.com/yzhaoinuw/desktop_app_source_updater.git@85bb68e42155ad8b661f669df8d281fbe683a045"` - installed updater 0.2.0 from the pinned commit.
  - The equivalent install command in `fp_analysis_dist` - installed the same updater version and commit in the packaging environment.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 79 tests passed.
  - Import and `py_compile` checks for the active analysis modules, launcher, updater config, and updater tests - passed.
  - `$env:FP_ANALYSIS_SKIP_UPDATE='1'; C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe run_desktop_app.py --check-update` - returned the expected disabled status and exit code 1.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe tools\build_update_asset.py --help` - exposed the shared multiple-installed-baseline and schema-2 config-merge options.
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1 -AllowDirty` - passed dependency checks, 79 tests, PyInstaller, structure smoke, packaged smoke, fresh-extraction smoke, hashing, and sidecar generation with updater 0.2.0 in the manifest.
  - `git diff --check` and `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

## 2026-07-24

### Publish the `v0.6.0-dev` updater baseline (gpt-5)

- Committed and pushed the shared-updater migration and lean Windows packaging pipeline, then fast-forwarded `auto-update`, `dev`, and `main` to the verified release commit.
- Used `v0.6.0-dev` as a deliberately temporary full-package baseline so the next source-changing release can exercise the real startup updater.
- Built the full Windows package from clean `main` commit `f8a65b9`, including exact Git-blob runtime bytes, packaged startup validation, and a fresh-extraction `unblock_app.cmd --smoke` check.
- Published the annotated `v0.6.0-dev` tag and GitHub prerelease with the Windows ZIP, SHA-256 sidecar, manifest, and build-environment snapshot.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 78 tests passed.
  - `$env:FP_ANALYSIS_SKIP_UPDATE='1'; C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe run_desktop_app.py --smoke` - printed `smoke ok: v0.6.0-dev`.
  - Updater version comparison check - confirmed a later `v0.6.2`-style release compares newer than `v0.6.0-dev`.
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1` - passed dependency checks, 78 tests with 8 private-fixture cases skipped in the release worktree, PyInstaller, structural smoke, packaged smoke, fresh-extraction smoke, hashing, and sidecar generation.
  - Artifact SHA-256 recomputation matched `ACC77CFF862BEBE96B00741A1AFFE583555D59EA09841CF52AA03123E1F2BE27`.
  - GitHub release inspection - confirmed a published, non-draft prerelease with all four assets uploaded and the ZIP digest matching the local artifact.
  - Remote-ref inspection - confirmed the release tag peels to `f8a65b9` and `main`, `dev`, and `auto-update` matched that commit at publication.
  - `git diff --check` and `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Build a lean Windows packaging pipeline on `auto-update` (gpt-5)

- Fast-forwarded the existing `auto-update` branch to the current `dev` tip at `8606bfb` and switched the main checkout there without disturbing the uncommitted updater migration.
- Assessed the `sleep_scoring` pipeline and adopted only the portable PyInstaller spec, exact tracked-byte runtime export, full-build orchestration, packaged/fresh-extraction smoke checks, unblock-and-start helper, and ZIP/hash/manifest/environment outputs.
- Deliberately omitted Torch/model splitting, a manual source-folder replacement ZIP, post-build baseline repair, and automatic GitHub publication because they do not currently solve an `fp_analysis` requirement.
- Added `--smoke` and `--check-update` launcher modes, packaging regression coverage, and a clean-worktree release gate while retaining `-AllowDirty` for local rehearsals.
- The first rehearsal caught missing Conda runtime DLLs at packaged startup; the portable spec now collects the seven required core DLLs explicitly, and the rebuilt executable passed.
- Produced a local dirty-worktree rehearsal artifact at `release_artifacts/fp_analysis_app_v0.5.0-windows.zip` with matching SHA-256 `853785BA1504F72E4F3C032A27FAABBAB83AAE78A4E6266B4D3D4BC27AF43825`. It is ignored local evidence, not a release candidate.
- Verification:
  - `git branch -f auto-update dev; git switch auto-update` - switched to `auto-update` at the current `dev` commit with the task changes preserved.
  - `C:\Users\yzhao\miniconda3\envs\fp_analysis_dist\python.exe -m pip install --no-deps --disable-pip-version-check "desktop-app-source-updater @ git+https://github.com/yzhaoinuw/desktop_app_source_updater.git@adb2221393ab6fab106f7cef19baf6157e039bd3"` - installed the pinned updater in the packaging environment.
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1 -AllowDirty` - passed environment checks, 76 tests, PyInstaller, structural smoke, packaged `--smoke`, compression, hashing, and sidecar generation after the DLL fix.
  - Fresh extraction followed by `unblock_app.cmd --smoke` - printed `smoke ok: v0.5.0`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 78 tests passed.
  - Source and packaged `run_fp_analysis_app.exe --smoke` checks - both printed `smoke ok: v0.5.0`.
  - Artifact SHA-256 recomputation matched the generated sidecar.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Integrate the shared startup updater (gpt-5)

- Replaced the copied runtime updater with the pinned `desktop_app_source_updater` dependency at commit `adb2221393ab6fab106f7cef19baf6157e039bd3`.
- Kept app-specific release settings and generated-data exclusions in `startup_update_config.py`, and reduced `tools/build_update_asset.py` to a thin wrapper around the shared builder.
- Extended the focused CI job to install the pinned package and run both the perievent and startup-update suites.
- Documented that this dependency migration needs one normal full packaged release before a later source-only release can exercise the real GitHub Release update path.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m pip install --no-deps --disable-pip-version-check --force-reinstall "desktop-app-source-updater @ git+https://github.com/yzhaoinuw/desktop_app_source_updater.git@adb2221393ab6fab106f7cef19baf6157e039bd3"` - installed version `0.1.0` from the pinned commit.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update` - 76 tests passed.
  - `$env:FP_ANALYSIS_SKIP_UPDATE='1'; C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import run_desktop_app; import desktop_app_source_updater, startup_update_config; import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe tools\build_update_asset.py --help` - printed the shared builder options through the app wrapper.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m pip check` - no broken requirements.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Tighten analysis-page status messaging (gpt-5)

- Removed the empty validation row's reserved height and reduced the status margins so analysis guidance sits directly below the controls.
- Reworded the valid-settings prompt to `Click Show Results to see the analysis results`.
- Simplified both primary and fallback save confirmations to `Spreadsheets are saved to '<path>'.`
- Verification:
  - Local Dash component preview - confirmed the prompt renders 2 px below the controls with the requested wording.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestAnalysisPageSignalHighlight tests.test_perievent_analysis.TestSelectiveAnalysisWorkbookExport.test_selective_export_creates_only_selected_workbooks` - 9 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 65 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev, fp_analysis_app.analysis_export, fp_analysis_app.components_dev; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Accept and publish detected peak-value spreadsheet exports (gpt-5)

- Marked the final analysis-workflow item accepted after the user confirmed the new positive- and negative-peak spreadsheet exports work.
- Closed the completed nine-item analysis-workflow thread and removed it from `next_steps.md`; dated implementation and acceptance evidence remains in the work log.
- Prepared the focused export implementation, tests, README wording, and coordination docs for commit and push on `negative-peaks`.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 64 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev, fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

## 2026-07-23

### Replace legacy max-value export with detected peak values (gpt-5)

- Replaced the active `max_peak_magnitude` export with separate `positive_peak_value` and `negative_peak_value` selections and payloads.
- Added `*_positive_peak_value_bw<baseline>_aw<analysis>.xlsx` and `*_negative_peak_value_bw<baseline>_aw<analysis>.xlsx` workbook names and matching user-facing checklist labels.
- Removed the legacy maximum calculation from analysis results; exported values now come from the first detected positive or negative peak index and remain `NaN` when no matching peak is found.
- Updated the secondary combined spreadsheet helper, README compatibility wording, fixture expectations, workbook alignment coverage, and selective-export coverage.
- Item 9 remains open for the user spreadsheet acceptance gate on `negative-peaks`.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestPerieventPeakAnalysis tests.test_perievent_analysis.TestSelectiveAnalysisWorkbookExport` - 8 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 64 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.app_dev, fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Accept and publish negative peak analysis plots (gpt-5)

- Marked item 8 accepted after the user confirmed the desktop analysis plots render and work.
- Negative peaks are found by applying the same `find_peaks` settings to the sign-flipped post-event reaction signal, then reading the detected index from the original signal so the reported value remains negative.
- Prepared the focused analysis, tests, and coordination docs for commit and push on `negative-peaks`; item 9 export correction remains next on the same branch.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 63 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Implement detected positive and negative peak plots (gpt-5)

- Created and switched to the `negative-peaks` branch from the current `dev` tip so analysis plot work and the following export correction can share one focused branch.
- Added first-detected positive and negative peak values using the same `find_peaks` settings on the reaction signal and its sign-flipped form.
- Replaced the legacy max-value analysis panel with separate positive- and negative-peak panels while retaining the legacy export payload until item 9.
- Added synthetic coverage for positive, negative, and missing peaks plus the six-panel analysis-figure contract.
- A preview render could not be completed because the current Matplotlib environment exits in layout/render operations for both the unchanged five-panel control and the new six-panel figure; item 8 remains open for the planned user visual gate.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestPerieventPeakAnalysis` - 2 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 63 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Accept and publish perievent heatmap row improvements (gpt-5)

- Marked adaptive black row dividers and corrected event-index labels accepted after the user's visual inspection.
- Prepared the focused heatmap implementation, tests, live TODO state, and required work-log archive for commit and push on `dev`.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 61 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Fix perievent heatmap event-index labels (gpt-5)

- Fixed the pre-existing event-index formatter that created a tick for every row but left all except every fifth label blank, even on three-row heatmaps.
- Labeled every event occurrence on sparse heatmaps and limited dense heatmaps to approximately 15 correctly numbered labels.
- Preserved the existing upward-increasing event-index axis, where event 1 is the bottom heatmap row.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestPerieventHeatmapRowDividers` - 5 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 61 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

### Add adaptive heatmap row dividers (gpt-5)

- Added solid black horizontal dividers between every event-occurrence row in perievent heatmaps.
- Kept sparse heatmaps clearly separated while progressively reducing divider width and opacity above 20, 50, and 100 event rows.
- Added focused coverage for divider placement, color, single-row behavior, and high-count styling.
- Archived the previous five live work-log dates into `work_log_archive/work_log_2026-07-04_to_2026-07-17.md` before adding this July 23 entry.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis.TestPerieventHeatmapRowDividers` - 3 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis` - 59 tests passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -c "import fp_analysis_app.analysis_export, fp_analysis_app.event_analysis, fp_analysis_app.mat_utils; print('import ok')"` - printed `import ok`.
  - `git diff --check` - clean apart from Git line-ending conversion warnings.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

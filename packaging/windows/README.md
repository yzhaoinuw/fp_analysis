# Windows Packaging

This folder contains the minimum repeatable packaging flow needed to establish
an updater-enabled Windows baseline and validate later source-only releases.

## Full App Zip

Use a full package when dependencies, the launcher, the shared updater,
PyInstaller settings, file deletions/renames, or the runtime layout changed.
Adding or upgrading the updater therefore requires a full package before later
source-only updates can rely on that updater version.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\windows\make_full_app_zip.ps1
```

Defaults:

- build environment: `fp_analysis_dist`
- test environment: `fiber_photometry`
- PyInstaller spec: `packaging/windows/app.spec`

Prepare or refresh the build environment with:

```powershell
C:\Users\yzhao\miniconda3\envs\fp_analysis_dist\python.exe -m pip install -r .\packaging\windows\build-requirements.txt
```

For a local pipeline test before committing, add `-AllowDirty`. Release builds
must use a clean tracked worktree so the manifest and external
`fp_analysis_app/` bytes can be tied to one commit.

Output goes to `release_artifacts/`:

```text
fp_analysis_app_vX.Y.Z-windows.zip
fp_analysis_app_vX.Y.Z-windows.zip.manifest.json
fp_analysis_app_vX.Y.Z-windows.zip.sha256.txt
fp_analysis_app_vX.Y.Z-windows.zip.build_env_requirements.txt
```

The build:

1. checks the build environment and runs the focused unittest suites;
2. runs PyInstaller with portable, environment-derived paths;
3. places exact tracked `fp_analysis_app/` bytes beside the executable;
4. adds writable figure, spreadsheet, and video directories;
5. validates the release layout and runs the packaged executable with
   `--smoke`;
6. extracts the ZIP afresh and runs `unblock_app.cmd --smoke`;
7. creates the SHA-256 sidecar, build-environment snapshot, and manifest.

The full ZIP includes `unblock_app.cmd`, which unblocks the extracted files and
starts the app.

## Later Source-Only Update

Use the existing app wrapper around the shared updater builder:

```powershell
C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe tools\build_update_asset.py `
  --from-ref <full-package-baseline-tag-or-commit> `
  --to-ref <later-release-tag-or-commit>
```

Repeat `--from-ref` for every compatible packaged baseline that should be able
to jump to the new version. Attach the generated
`fp_analysis_app_update_<version>.zip` to the matching GitHub Release.

For the installed-user test, run the baseline executable with
`--check-update`. The command forces a release check even inside the normal
24-hour interval, exits without opening the GUI, reports the updater status,
imports the resulting side-by-side app version after a successful check, and
returns a nonzero exit code for failed, blocked, skipped, or disabled updates.
Normal discovery follows GitHub's `/releases/latest` redirect, which excludes
prereleases. Publish the source-update test as a normal release, or point
`FP_ANALYSIS_UPDATE_ZIP_URL` directly at the test asset during a prerelease-only
gate.

The full-package exporter writes exact Git-blob bytes into clean release builds,
so the source-update builder can use Git refs as the installed hash baselines.
If a distributed package is ever modified after this export step, capture its
installed hashes explicitly with the shared builder's
`--installed-baseline-manifest` option.

## Deliberately Not Adopted From sleep_scoring

- optional Torch/runtime splitting and model packaging;
- a manual source-folder replacement ZIP;
- a post-build package-byte alignment repair script;
- automatic GitHub Release publication.

Those pieces do not currently solve an `fp_analysis` requirement. Release
publication remains an explicit maintainer action after the generated package,
hash, manifest, tests, and smoke checks have been reviewed.

# Work Log

Prepend new session notes to the top of this file. The live log holds at most the five most recent unique calendar dates; each archive file holds exactly five. See [Work Log Discipline](treaty_conventions.md#work-log-discipline) and [Work Log Rotation And Dating](treaty_conventions.md#work-log-rotation-and-dating).

## 2026-10-21

### Test-drive Treaty v0.9.2 (Codex, GPT-6; effort and token budget not exposed)

- Used local `chore/treaty` from `e7b3f7a` to assess the released updater before integration. The dedicated 0.9.2 CLI updated the v0.6.0 template without conflicts, retained `docs_dir: .` and the repository's answers, and left project guidance and application source intact. The generic conventions now include the pre-push/merge log checkpoint and the planning heading rules.
- Kept unrelated untracked data, sketches, and linked worktrees available: temporarily excluded their exact paths during apply, then restored `.git/info/exclude` byte-for-byte. Used explicit staging. Preview preserved every tracked file hash; apply's extra `work_log.md` listing was newline-only, not content loss.
- Recorded the process, issue-by-issue evidence, and limits in [the adopter test-drive report](docs/treaty-v0.9.2-test-drive.md). Disposable probes confirmed refusal of hidden rejects in both modes, a working `!*.rej` exception with matching visible conflicts, explicit answers to a genuinely new question, layout-override refusal, clearer verification errors, and nonfatal pointer/version warnings. On the same nested fixture, 0.9.1 reported three sections at risk while 0.9.2 correctly reported one.
- The prose fixes arrived, but their ability to prevent missed logs over future sessions is not established. No app planning changed; `next_steps.md` remains intact. Branch is intentionally local for review; no push, PR, integration merge, tag, or release was performed.
- Verification:
  - Released Treaty 0.9.2 `validate .`, `diff .`, update preview, and apply; zero conflicts in the real adoption and zero conventions drift afterward.
  - Disposable CLI/library probes and 0.9.1 comparison, with raw evidence under untracked `.worktrees/treaty-evidence-v0.9.2/`; details and untested paths are in the report.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` — 85 passed outside the sandbox after the initial sandbox run hit temporary-Git/updater restrictions.
  - Active analysis-module import smoke — printed `import ok`.
  - `git diff --check`, unmerged-index inspection, and line-anchored conflict-marker scan — passed.

## 2026-08-04

### Make the app licensed and citable (claude-opus-5)

- Added an MIT `LICENSE` and a `CITATION.cff`, closing the two gaps that made
  this the only shipped app of the four that could not be cited or legally
  reused. Without a license the repository defaults to "all rights reserved,"
  which blocks reuse rather than protecting anything. Copyright year is 2025,
  this repository's creation year, following the house pattern of per-repo
  creation years rather than a shared year.
- Gave the README an opening paragraph describing what the app is and who it is
  for, plus Citation, Funding, and License sections. Funding wording matches
  `sleep_scoring`.
- Narrowed the packaged-build claim in both the README and the CFF abstract
  after review: the startup updater only applies compatible source-update
  assets and cannot update dependencies, the launcher, or packaging, so the app
  "checks GitHub Releases for compatible source updates" rather than updating
  itself.
- Decided against publishing a citation-only release. Zenodo is now enabled for
  the repository but did not retroactively archive `v0.6.0`; archival begins
  with the next newly published GitHub Release. The CFF `version` and
  `date-released` bump therefore belongs to that release candidate alongside
  the app's other version surfaces, and the concept DOI and README badge follow
  after publication. Recorded both under the existing first source-update
  release trial in `next_steps.md`.
- Verification:
  - `cffconvert --validate -i CITATION.cff` - printed `Citation metadata are valid according to schema version 1.2.0.`
  - `git diff --check` - clean.
  - `treaty validate .` - passed.
  - Not run in this session: the focused unittest suite and desktop smoke
    checks. This macOS checkout has no `fiber_photometry` environment, and the
    change touches only documentation and repository metadata. CI on `dev` is
    green for the same commit range.

### Reconcile the citation PR with the treaty migration (gpt-5)

- Fast-forwarded remote `main` to the previously accepted treaty migration at
  `76c9d73`, then merged the citation PR head into `dev` without rewriting
  either history.
- After final approval, fast-forwarded `main` to the validated `dev` merge at
  `02d5543`; GitHub marked PR #1 merged, and local, tracking, and remote
  `main`/`dev` refs were verified identical.
- Resolved the overlapping work-log update by retaining the treaty-managed
  header, the August 1 treaty record, and the citation record under the current
  August 4 date. The independently created July archive was byte-identical on
  both branches, so it does not appear in the final PR diff.
- Kept all unrelated tracked and untracked local artifacts outside the merge.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 85 tests passed.
  - Active analysis-module import smoke - printed `import ok`.
  - `CITATION.cff` validation against the official CFF 1.2.0 schema - passed.
  - `git diff --cached --check` - passed.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .` - passed.

## 2026-08-01

### Migrate the collaboration treaty to v0.6.0 (gpt-5)

- Migrated the tracked Copier adoption from template `v0.3.2` to `v0.6.0` on the dedicated `chore/treaty-v0.6.0` branch, preserving the active desktop runtime, event/export, packaging, updater, and local-data boundaries.
- Kept `AGENTS.md` as a lean project-specific map, added the upstream-managed `treaty_conventions.md`, and added an authored-versus-derived map to `project_overview.md`.
- Recorded the repository's real release, Git-ownership, verification, and no-pre-commit answers; resolved all three expected Copier conflicts without changing application code or generated/local artifacts.
- Rotated the previous five live dates into `work_log_archive/work_log_2026-07-23_to_2026-07-30.md` before adding this date.
- Opened and verified upstream issue `agent_collab_treaty#17` for a newly observed update gap: older managed adopters silently receive defaults for newly introduced questions, including an incorrect `uses_precommit: true` in this repository.
- The user accepted the validated migration for local fast-forward integration into `dev` and `main`; remote publication remains a separate action.
- Verification:
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe --version` - reported CLI and pinned template `v0.6.0`; the source checkout was verified at stable release commit `cdfb3c84822ac549dcae3d279c356c64490ea424`.
  - Clean-tree `treaty.exe update . --dry-run` - exited 0 outside the sandbox, left status clean, and preserved hashes for every existing managed file.
  - Applied `treaty.exe update .` - exited 1 and named the three expected customized-doc conflicts; all were resolved and no unmerged index entries remain.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` - 85 tests passed with 8 private-fixture skips.
  - Active analysis-module import smoke - printed `import ok`.
  - `$env:FP_ANALYSIS_SKIP_UPDATE='1'; C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe run_desktop_app.py --smoke` - printed `smoke ok: v0.6.0`.
  - `treaty.exe validate .`, `treaty.exe diff .`, and `git diff --cached --check` - passed; the final treaty diff has no removed headings, and `treaty_conventions.md` is untouched against v0.6.0.
  - GitHub issue inspection - verified `https://github.com/yzhaoinuw/agent_collab_treaty/issues/17`, including the `Codex (GPT-5)` signature.

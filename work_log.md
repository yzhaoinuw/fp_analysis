# Work Log

Prepend new session notes to the top of this file. The live log holds at most the five most recent unique calendar dates; each archive file holds exactly five. See [Work Log Discipline](treaty_conventions.md#work-log-discipline) and [Work Log Rotation And Dating](treaty_conventions.md#work-log-rotation-and-dating).

## 2026-11-17

### Refresh Treaty to v0.11.1 (Codex; model/version, effort and token budget not exposed)

- Updated the released CLI first, then adopted template v0.11.1 from v0.9.2 on the existing local `chore/treaty` branch. Preview with `--dry-run --diff` and apply agreed on the answers-file pin, maintained conventions, and archive README; neither conflicted. Existing project answers and the flat `docs_dir: .` layout remain intact. No application code or product planning changed.
- The new conventions require upgrading the CLI before the template, running validation afterward, checking for Treaty updates on log rotation, and using the correct GitHub anchors in Currently Hot. The archive README also carries the rotation reminder. No relocation was requested.
- Re-ran the original [upstream #31](https://github.com/yzhaoinuw/agent_collab_treaty/issues/31) fixture against CLI 0.11.1: preview and apply listed the same three content changes; the CRLF-only work-log rewrite was identified separately and normalized history was identical. The printed staging hint named every affected file and did not use blanket staging. Recorded this recheck in the [original adopter report](docs/treaty-v0.9.2-test-drive.md#recheck-with-v0111-on-2026-11-17).
- Preserved the pre-existing photometry-investigation edits using a path-limited stash during the clean-tree update; they are restored separately as uncommitted work after this Treaty-only commit. Local exclude contents are restored byte-for-byte, unrelated artifacts remain in place, and no push, PR, merge, release, or new upstream comment was performed.
- Verification:
  - Released CLI 0.11.1 `validate .`, `diff .`, full-diff preview, and apply; preview preserved all tracked working-file SHA-256 hashes, and final conventions have zero drift.
  - Original #31 disposable fixture with `core.autocrlf=true`: matching content lists, separate newline reporting, identical normalized log text, and explicit staging handoff — passed.
  - `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` — 85 tests passed; active analysis-module import smoke printed `import ok`.
  - Whitespace, unmerged/conflict-marker, relative-link/anchor, and original-note preservation checks — passed.

## 2026-10-22

### Publish the Treaty adopter assessment (Codex, GPT-6; effort and token budget not exposed)

- At the user's request, posted targeted evidence to upstream issues #15, #17, #22, #23, #25, #26, #27, #28, and #30. Kept the distinction between tested CLI behavior, delivered guidance, and unproven long-term effects. Existing issue states were unchanged; the [test-drive report](docs/treaty-v0.9.2-test-drive.md#upstream-feedback-posted-2026-10-22) records direct comment links.
- Reproduced the newline-only summary discrepancy in a clean disposable clone before opening [upstream #31](https://github.com/yzhaoinuw/agent_collab_treaty/issues/31). Preview omitted `work_log.md`, apply listed it, and CRLF counts changed from 0 to 77 while normalized text and the Git content diff stayed unchanged. Classified it as reporting polish rather than content loss. Added the blanket-staging handoff suggestion to existing #22 rather than opening a duplicate workflow issue.
- Verified the destination and adopter repositories are public and owned by the authenticated account. After destination checks resolved the initial automatic-review objection, the connector's write returned HTTP 403; authenticated `gh` successfully posted the authorized feedback. Every comment was read back and matched its prepared body. No branch push, PR, or integration merge was performed; local documentation remains on `chore/treaty`.
- Verification:
  - Read the current issue bodies and comments, and searched the repository's issue history before posting or opening #31.
  - Disposable newline fixture: clean pre-update status, preview/apply exit 0, differing file lists, identical normalized log text, and empty `git diff -- work_log.md`.
  - GitHub readback: all nine exact comment bodies matched; #31 exists with the intended title, reproduction, and scope.
  - Treaty 0.9.2 `validate .` and `git diff --check` — passed after this documentation update. Application code was unchanged; the previous 85-test pass remains applicable.

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

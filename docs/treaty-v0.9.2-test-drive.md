# Treaty v0.9.2 adopter test drive

Date: 2026-10-21 (America/New_York). Agent: Codex, GPT-6; effort and token budget not exposed.

## Outcome

Updated `fp_analysis` from template v0.6.0 to v0.9.2 on local branch `chore/treaty`, based on `e7b3f7a17d3ed6e51d48f0cb4941db5bbe43587a`. The real update completed without conflicts, retained the flat layout (`docs_dir: .`), and preserved the project answers, including `uses_precommit: false`. `AGENTS.md`, `project_overview.md`, `next_steps.md`, application code, and existing log content survived unchanged. The maintained conventions match a pristine v0.9.2 render.

The branch is for review before integration. No push, PR, merge, tag, or application release was performed.

The tested CLI fixes worked. The new guidance also arrived, but its effect on agent behavior requires later sessions. This was an adopter test, not an exhaustive audit of Copier or every Treaty code path.

## Release and environment

The [latest GitHub Release](https://github.com/yzhaoinuw/agent_collab_treaty/releases/tag/v0.9.2) was verified through the GitHub API; its target commit is `2379618354c18a6e21733258e526285895311ba5`. PyPI also supplied version 0.9.2. The [v0.9.1 to v0.9.2 comparison](https://github.com/yzhaoinuw/agent_collab_treaty/compare/v0.9.1...v0.9.2) identifies the nested-diff correction and two diagnostic improvements. The adopted update also includes v0.9.1's safety checks, warning tier, answer overrides, and conventions changes.

`treaty` was absent from the shell's PATH. The conda environment contained CLI 0.3.2, which did not support `--version`; the separate Treaty development environment reported 0.9.1. Neither was used to apply this update.

Installed the released 0.9.2 wheel into `.worktrees/treaty-cli-v0.9.2`, with access to the existing `fiber_photometry` dependencies: Copier 9.15.2, Typer 0.26.7, and plumbum 2.0.1, under Python 3.10. A first attempt to resolve a fully separate dependency stack was slow and was interrupted; using the compatible existing dependencies avoided changing either established environment. This does not test the latest available Copier release.

## Update process and experience

1. Read `AGENTS.md`, the existing update conventions, current planning pointers, and the two newest work-log dates. Verified local date, branch/worktree state, and remote `main`/`dev`; both remote tips matched the starting commit. Created `chore/treaty` from the committed `dev` checkout.
2. Ran baseline `treaty validate .` and `treaty diff .`. Validation passed. Diff showed 22 sections of potential exposure in the customized `AGENTS.md` and `project_overview.md`, and zero drift in conventions. That exposure is conditional on upstream revising those sections; it is not a prediction of 22 actual conflicts.
3. Ran `treaty update . --dry-run --data uses_precommit=false`. It reported v0.6.0 to v0.9.2, the new `docs_dir: .` answer, two updated files, and no conflicts. SHA-256 hashes of every tracked working-tree file were identical before and after preview. Passing the already-recorded false answer made the intended project setting explicit.
4. The checkout contained unrelated untracked data, sketches, presentations, and linked worktrees. During apply only, added their exact root paths to `.git/info/exclude`, asserted that Git saw a clean tree, then ran `treaty update . --data uses_precommit=false`. Restored the original exclude file byte-for-byte in a `finally` block. No `status.showUntrackedFiles=no` setting or global `*.rej` ignore was used.
5. Apply exited 0. Its summary additionally listed `work_log.md`; normalized text and Git content diff confirmed this was only newline conversion. The content changes were confined to the answers file and conventions. The verification command was YAML-wrapped without changing its parsed value.
6. Reviewed the update, checked unmerged paths and line-anchored conflict markers, exercised the targeted probes below, and wrote this report plus a work-log entry. Staged only the four intended files rather than copying the CLI's generic `git add -A` suggestion into this checkout.

The clean-tree requirement was the main operational friction. Temporary local excludes worked while leaving `.rej` recovery visible and keeping unrelated artifacts available afterward. The no-conflict update itself was straightforward. The CLI still requires separately managing the installed package version and adopted template version; upgrading one does not upgrade the other.

## Assessment of the intended fixes

All fault-injection checks ran in disposable copies, not the working project. Synthetic fixtures are identified explicitly.

| Fix | Evidence observed | Assessment and limit |
| --- | --- | --- |
| Hidden reject recovery, issue #25 (v0.9.1) | In a clone of this repo, repo-local `status.showUntrackedFiles=no` made both preview and apply refuse with exit 1 before changing project documents. A deny-all `*` in local excludes produced the same refusal in both modes. | Front-door protection works for both known hiding mechanisms. The separate post-merge orphan-`.rej` backstop was not fault-injected. |
| Narrow ignore-exception advice, #25 follow-up (v0.9.2) | The refusal suggested `!*.rej`. With `*` followed by `!*.rej`, preview and apply both reached the merge. Deliberately changing `uses_precommit` to true produced exactly one visible `AGENTS.md` conflict and exit 1 in both modes. Project-specific text remained in the conflicted file; overview, next steps, and log text survived. | The suggested narrow exception works without removing the deny-all rule. Tested the rule in `.git/info/exclude`; the new diagnostic recommends `.gitignore`. |
| Answer overrides, #17 / #23.1 | A disposable two-version template introduced a genuinely new boolean question whose default was true. `--data uses_precommit=false` previewed `None -> False` without writing, then apply recorded false and preserved the existing instructions. The real-repo clone also previewed and applied an override to an existing answer. | Explicit noninteractive answers work for new and existing questions. Omitting `--data` still accepts defaults by design; automatic prompting for only new questions was not added. |
| Unsafe layout override guard | `--data docs_dir=treaty_docs` exited 1 in both preview and apply and directed the caller to `treaty relocate`. | Guard works. No layout migration was needed or performed in this repo. |
| Nested adopter-owned diff, #30 (v0.9.2) | Ran the released diff/report functions on the same synthetic `docs/agents/` fixture with modified log, next steps, and conventions. CLI/library 0.9.1 reported 3 sections across 3 files; 0.9.2 reported 1 section across 1 file and marked both logs as owned content. | The false-positive report is fixed, while maintained conventions still count as exposure. This repo's real flat-layout report remains 22 sections before and after. A real nested Copier migration was not part of this test. |
| Inline verification diagnostic, #15 follow-up (v0.9.2) | In the clone, changed one heading to `- Verification: inspected evidence`. Validation exited 1, pointed to that line, and said to put the heading on its own line and indent evidence below. Restoring the heading cleared the error. | The message now explains the visible formatting problem. Inline evidence remains invalid by policy; this was a diagnostic fix. |
| Claude pointer warning, #27 (v0.9.1) | A synthetic local `CLAUDE.md` saying only “Please read AGENTS.md” produced `claude-pointer-missing-import` with exit 0. Replacing it with `@AGENTS.md` cleared the warning. | Warning detection and nonfatal severity work. Actual Claude Code loading was not exercised. |
| CLI/template version skew, #22 / #23 (v0.9.1) | Temporarily changed only the clone's pin to synthetic `v99.0.0`. Validation warned `cli-older-than-template`, named the installed CLI version, suggested upgrading, and exited 0. | Validation warning works. This does not test running an update against a future template. |
| Push/merge logging trigger, #28; content decisions and planning headings, #26 | New conventions contain the push/merge log checkpoint, examples of content-editing decisions, and the active/paused heading rule. Final `treaty diff` shows conventions fully match the template. | Delivery verified. These are prose instructions, not enforcement. No push/merge or multi-session behavioral trial was performed, so prevention of missed logs remains unproven. |

## Verification and remaining rough edges

- Focused app suite: `C:\Users\yzhao\miniconda3\envs\fiber_photometry\python.exe -m unittest tests.test_perievent_analysis tests.test_startup_update tests.test_packaging` — 85 tests passed outside the sandbox. The first sandbox run had 5 failures and 4 errors involving updater behavior and temporary Git repositories; the unchanged rerun outside the sandbox passed.
- Active analysis-module import smoke printed `import ok`.
- Treaty 0.9.2 `validate .` passed before and after adoption. Final `diff .` has no removed project headings and no maintained-conventions drift. The 22 customized sections remain expected.
- `git diff --check`, unmerged-index inspection, and line-anchored conflict-marker checks passed. Existing untracked root paths remained present; app/source files were unchanged. The tests did not hash large local datasets.
- The CLI's apply summary can show a newline-only file as updated when preview did not list it. Inspect the content diff before treating this as a preview/apply disagreement.
- The generic successful-update handoff says `git add -A && git commit`. In this repository, restored untracked artifacts make explicit staging necessary.
- `--data` provides a route to explicit answers, but does not discover which newly introduced defaults suit the adopter. Continue checking the answer summary.
- Local raw outputs, fixture repositories, and probe scripts are retained under `.worktrees/treaty-evidence-v0.9.2` and adjacent `.worktrees/treaty_*.py` files. They are untracked local evidence, not required application artifacts.

No new app follow-up was created, so `next_steps.md` was left intact. At the end of the original test drive, nothing had been posted to an issue or sent to another chat. The subsequently authorized upstream feedback is recorded below.

## Upstream feedback posted 2026-10-22

At the maintainer's request, posted issue-specific assessments and read each comment back to verify its exact body. Existing issue states were left unchanged.

| Thread | Posted assessment |
| --- | --- |
| [#25](https://github.com/yzhaoinuw/agent_collab_treaty/issues/25#issuecomment-6332928677) | Both safety refusals, the working narrow exception, matching visible conflicts, and the untested post-merge backstop. |
| [#17](https://github.com/yzhaoinuw/agent_collab_treaty/issues/17#issuecomment-6332932310) | Explicit answers to a genuinely new question and existing answers; layout-override guard. |
| [#30](https://github.com/yzhaoinuw/agent_collab_treaty/issues/30#issuecomment-6332932488) | Same nested fixture reports three at-risk sections in 0.9.1 and one in 0.9.2. |
| [#15](https://github.com/yzhaoinuw/agent_collab_treaty/issues/15#issuecomment-6332932661) | Inline-verification message points to the problematic line and explains the correction. |
| [#27](https://github.com/yzhaoinuw/agent_collab_treaty/issues/27#issuecomment-6332932842) | Missing-import warning is nonfatal and clears with the explicit import; Claude Code runtime not tested. |
| [#28](https://github.com/yzhaoinuw/agent_collab_treaty/issues/28#issuecomment-6332933082) | New log checkpoints and decision examples arrived; longer-term behavioral effectiveness remains unproven. |
| [#26](https://github.com/yzhaoinuw/agent_collab_treaty/issues/26#issuecomment-6332933367) | Conventions invariant arrived; remaining template and behavioral work remains distinct. |
| [#23](https://github.com/yzhaoinuw/agent_collab_treaty/issues/23#issuecomment-6332933592) | Answer-override and version-warning evidence, without claiming first-run/placeholder issues were resolved. |
| [#22](https://github.com/yzhaoinuw/agent_collab_treaty/issues/22#issuecomment-6332933785) | Successful flat-layout update and temporary-exclude workflow; suggestion to replace the blanket staging handoff. |

Opened [#31: Windows update summary lists newline-only files that dry-run omits](https://github.com/yzhaoinuw/agent_collab_treaty/issues/31) after independently reproducing it in a clean disposable clone. Under `core.autocrlf=true`, refreshed the index for an LF `work_log.md` and confirmed empty status before updating. Preview omitted the log; apply listed it. The log went from 0 to 77 CRLF endings, normalized bytes stayed identical, and its content diff was empty. This is a minor reporting issue, not evidence of content loss. The issue includes standalone reproduction steps and the limits of the finding.

Posting required a tooling fallback: initial automatic review required destination/disclosure checks; the connector verified both repositories as public and owned by the authenticated administrator, then its write attempt returned HTTP 403. The authenticated GitHub CLI posted the comments and new issue successfully. Readback used the connector. No project branch was pushed or merged.

## Recheck with v0.11.1 on 2026-11-17

Updated this adopter from template v0.9.2 to the latest released v0.11.1 with CLI 0.11.1, Python 3.10, Copier 9.15.2, Typer 0.26.7, and plumbum 2.0.1. The CLI runs from the reused isolated `.worktrees/treaty-cli-v0.9.2` environment; its directory name is historical. Full-diff preview and apply both exited 0 and agreed on three content changes: `.copier-answers.yml`, `treaty_conventions.md`, and `work_log_archive/README.md`. Project answers and the flat layout were preserved; no conflicts or maintained-conventions drift remained. All 85 focused app tests and the import smoke passed.

[Issue #31 was closed as fixed in v0.9.3](https://github.com/yzhaoinuw/agent_collab_treaty/issues/31#issuecomment-6333143905). Independently re-ran the original public `fp_analysis` baseline fixture (`e7b3f7a`, LF working log under `core.autocrlf=true`) with CLI 0.11.1. Preview and apply now report the same three content updates. Apply puts `work_log.md` under **Line endings rewritten, content unchanged**, and the log's normalized bytes are identical. The suggested `git add` explicitly names the three content files and the newline-only file, replacing the blanket handoff. This verifies the previously reported defect is fixed on its original fixture.

The fuller preview, validation-after-update guidance, and upgrade-the-CLI-first guidance are also present. This recheck did not rerun all nine earlier probes or test whether prose guidance prevents omissions over future sessions. New installs default to `agent_worklog/`; this existing adopter stayed flat because no relocation was requested. Raw outputs are retained locally under `.worktrees/treaty-evidence-v0.11.1/`. No new upstream comment was posted during this refresh.

# Tasks

## 1. Documentation

- [x] 1.1 Review root/package READMEs, authored `docs/` pages, and skill bodies/references; remove release/prerequisite/compatibility numbers, preview/candidate setup detours, and normal-user TestPyPI instructions. Verify a targeted search leaves only executable contract values or excluded metadata; keep package manifests, lockfiles, runtime checks, and historical specs unchanged.
- [x] 1.2 Reduce the Studio README to installation, launch, one example, and links; make `docs/studio.md` the canonical tutorial/runtime/hosting guide and shorten `docs/how-to/local-api.md` to manual startup plus essential transport rules. Verify the complete install → launch → example → run → inspect path is clear, advanced safety guidance remains reachable, and strict MkDocs passes.
- [x] 1.3 Consolidate contributor installation into `docs/development/install.md`, correct the extra/CI instructions in `docs/how-to/development.md`, and replace repeated library-registration recipes with links to their existing owner pages. Verify commands match project configuration, referenced pages exist, and any navigation edits pass strict MkDocs.

## 2. Skill checks

- [x] 2.1 Replace exact frontmatter-layout assertions in `skills/tests/check_skills.py` with value checks using installed YAML tooling; retain identity, payload, eval-input, and link validation. Verify both single-line and multiline descriptions pass and missing required metadata/broken links still fail, then run the structural check.
- [x] 2.2 Keep the documented-example smoke in `check_runtime.py`; move unique fitted-state/checkpoint and tuning assertions into the existing pytest structure and remove duplicated assertions only after mapping their retained coverage. Update affected descriptions/CI commands, run both skill scripts and focused pytest, and verify the starter, manual grader/evals/fixtures, and ignored workspaces remain intact.

## 3. Bounded support-code cleanup

- [x] 3.1 Remove unread draft `semanticRevision` state and replace the unused observation event array with minimal truncation accounting. Add/run focused checks for save/layout/build revisions, bounded metric/terminal displays, replay cursors/gaps, reconnect, and terminal completion; keep raw persisted history unchanged.
- [x] 3.2 Reuse the current graph for connection validation in `model/graph.ts`, `components/studio.tsx`, and `components/canvas.tsx`, updating all callers together. Add/run model and relevant browser regressions for preceding producers, overwritten keys, domain/order rejection, evaluator defaults, and unchanged connection options; introduce no global cache.
- [x] 3.3 Compute the execution catalog once in `src/nexuml/api/worker.py` while preserving every response field. Add a focused API test proving one catalog evaluation and equivalent response content, run API catalog tests, and verify touched code remains readable without unrelated formatting or dependency changes.

## 4. Integration and merge handoff

- [x] 4.1 Run strict docs, affected Python lint/format/type checks and pytest, Studio typecheck/lint/unit/build, and the existing isolated package/browser workflow for both project and tool modes using matching built artifacts. Verify the documented CPU example, save/reopen, training/reconnect, and artifact inspection work; record unavailable checks explicitly without starting real cluster jobs, publishing, or changing pending platform gates.
- [x] 4.2 Condense `openspec/changes/add-local-nexuml-studio/verification.md` into current evidence, commands, links, and limitations; report documentation/code reductions and any retained exceptions. Verify old evidence is not represented as a current-head pass, all links resolve, `git diff --check` passes, and `openspec validate simplify-documentation-and-support-code --strict` succeeds.

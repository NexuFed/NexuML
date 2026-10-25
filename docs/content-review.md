# Editorial migration review

Scope: `migrate-starlight-docs` editorial portions of tasks 4.1–4.5. No task checkboxes
were changed by this slice. Source of truth: approved proposal/design sections 2–3,
documentation-journey, scenario-learning, decorator-discovery, and first-run specs, plus
`docs/migration/route-map.json` and connected implementation.

## Delivered coverage

41 owned MDX pages, approximately 18,400 whitespace-delimited words, cover all 38 mapped
owned editorial destinations plus the requested new first-run, Why NexuML, and Studio-hosting
pages. Original authored material is approximately 20,200 words before deduplication.
The original Markdown files remain intact for parity. No redirect-only placeholder was
copied as a competing article. Required title/description frontmatter is present.

| Legacy material | Canonical editorial destination / retained details |
| --- | --- |
| `index.md` | `/`: short purpose and recommended path, no README include or API dump |
| `start/index.md`, `learn/index.md` | `/start/`: one first-run route; external tutorials remain reachable |
| `start/install.md` | `/start/install/`: normal/core installs, extras, explicit GHCR tag selection, CUDA/DALI, root rules |
| `start/train-cifar-resnet.md`, `learn/cifar-resnet-walkthrough.md` | `/start/cifar-resnet/`: optional download/tracking/evaluation prerequisites, explicit config, output discovery |
| `studio.md` | `/studio/`: same synthetic CPU draft, local editing/run/auth/process limits; hosted proxy material split to `/how-to/studio-hosting/` |
| `tutorials.md` | `/tutorials/`: original external repository, tutorial-owned components, current learning stages versus later direction |
| `explanation/index.md`, `explanation/architecture.md`, `learn/mental-model.md`, `learn/scenarios.md`, `explanation/tensordict.md`, `learn/from-lightning.md` | `/explanation/architecture/`: one definition → persistence → materialization → TensorDict → Lightning → evaluation → export flow; `#scenarios`, `#tensor-data-flow`, `#coming-from-lightning`; unique Lightning table/checkpoint distinctions retained |
| `explanation/diagrams.md` | `/explanation/diagrams/`: configured Mermaid output, internal scenario-name filename, log-root rules, build/train behavior and diagnostic-warning semantics |
| `explanation/library-discovery.md`, `learn/decorators-and-discovery.md`, `how-to/custom-library.md`, `how-to/register-library.md`, `how-to/library-cli.md` | `/how-to/custom-library/`: `#discovery`, `#installed-libraries`, `#local-library-roots`; package entry points, all library commands, role registry checks, verbose errors, no cache, duplicate identities, dependency-install distinction |
| `how-to/run-scenarios.md`, `how-to/cli-lifecycle.md`, `how-to/define-scenario.md`, `how-to/custom-scenario.md`, `how-to/train.md` | `/how-to/run-scenarios/`: exact source-support matrix, complete registered recipe at `#define-a-recipe`, resolve/build, overrides at `#train`, actual lifecycle, output discovery, smoke and specialized-task links |
| `how-to/scenario-file.md` | `/how-to/scenario-file/`: complete executable file, trusted-code warning, artifact paths, hypothesis/parent/tags, tuning exports, provenance limits |
| `how-to/custom-layer.md` | `/how-to/custom-layer/`: direct PyTorch factories versus semantic definitions, JSON-safe/importability trust constraints, build context/runtime split, validation/schema, complete custom component |
| `how-to/custom-data-source.md` | `/how-to/custom-data-source/`: full dataset/runtime example, fixed-label contract, sample assertions, source versus surrounding data policy |
| `how-to/custom-eval-algorithm.md` | `/how-to/custom-eval-algorithm/`: full accumulator example, key annotations, consumed-context fallback/required/unknown/no-input distinctions |
| `how-to/data-loading.md`, `how-to/auto-batch-size.md` | respective same-slug routes: Torch/DALI/tensor-shards, native/fallback limitations, training/loader batch precedence, exhaustive CUDA probe and all safety policies |
| `how-to/evaluate.md`, `how-to/checkpoints.md` | respective same-slug routes: model versus reporting state, no invented evaluate command, test metric exposure, create/resume/selective-weight reuse distinction and glob semantics |
| `how-to/tracking.md`, `how-to/tune.md`, `how-to/autoresearch.md` | respective same-slug routes: all logger integrations, actual output discovery, bounded searches, trusted files/skills, scorer/data/latency constraints, final-test protection and evidence limits |
| `how-to/export-dataset.md` | same-slug route: raw/intermediate views, every export format, config/metadata contract, reuse/skip stages, S3/index boundaries, storage shards versus runtime batches |
| `how-to/export.md`, `how-to/deploy.md` | respective same-slug routes: real live/checkpoint export, initialized-weight warning, full package layout, two reload paths in Starlight tabs, inference, alternative formats, infrastructure ownership |
| `how-to/training-backends/index.md`, `ray.md`, `jobs.md` | same route family: local/Ray/native placement, schemas/API, read-only capacity and unknowns, worker/template derivation, shared/S3 handoff, all distributed semantic limits and UID-safe launch/inspect/cleanup warnings |
| `how-to/local-api.md` | same-slug route: authenticated OpenAPI endpoint, runtime/interface/catalog contracts, conflicts/revisions/order/layout/filesystem boundaries, first-frame WS authentication, bounded replay, operation/process ownership and remote cancellation limits |
| `how-to/index.md` | `/how-to/`: compact task groups with no competing scenario/library walkthroughs |
| `reference/index.md`, `backends.md`, `environment.md`, `registry.md`, `scenario-spec.md`, `tuning-file.md` | same route family: compact lookup entry points, both generated API trees, exact storage/root distinctions, source-derived ScenarioSpec Autodoc, specialized tuning-file semantics |
| `development/index.md`, `development/install.md`, `how-to/development.md`, `development/release.md` | `/development/`, merged `/development/install/`, `/development/release/`: setup, every useful test/coverage/static/skills/Studio/package check, OpenChamber, CI/trust, dependency/PR/release conventions, exact-commit protected publishing and recovery |
| `how-to/github-pages.md` | `/development/documentation/` is parent-owned; not copied/edited in this slice |
| `reference/cli.md` | `/reference/cli/` is parent-owned; no CLI content/component edits here |
| `reference/decorators.md` | generated `/reference/api/nexuml/core/discovery/`; no authored duplicate decorator page |

`/explanation/why-nexuml/` distinguishes implemented capabilities from direction and
explicitly preserves infrastructure, distributed-finalization, and trusted-operator limits.
The three template download links retain the prescribed root-like canonical paths;
public copying/download validation remains parent-owned. Internal links are Markdown
single-leading-slash canonical routes, never a fixed deployment mount/version/hostname.
Global Changelog is not linked through a docs-prefixed Markdown route.

## Connected-source consistency

- `src/nexuml/cli/main.py`: verified `resolve --output`, positional `build`, `train --config`,
  `--max-epochs`, repeatable `-O`, exclusive sources, actual completion/test-result output,
  optional diagram export, checkpoint export behavior, `smoke`, and library/registry commands.
- `src/nexuml/cli/overrides.py` and `core/types.py`: `training.accelerator=cpu` stays a
  string; `training.devices=1` becomes an integer; both are real training fields. No recipe
  or runtime defaults were changed.
- Synthetic recipe/data/model/training/evaluation helpers under `library/src/nexuml_library/scenarios/`:
  verified recipe key versus internal underscore name, 128-feature input, 8-dimensional latent,
  encoder/decoder/loss stages, generated data, reconstruction evaluation, and absent explicit
  logging/callbacks/exports. First-run text promises the explicit YAML and actual terminal
  results, not a made-up checkpoint/diagram. Lightning may create its default checkpoint;
  the guide discovers existing files instead of promising a filename/policy.
- `training/lightning.py`, `core/compiler.py`, `core/pipeline.py`, `core/components.py`,
  `core/discovery.py`, and `core/serialization.py`: checked definition/runtime and shape/key
  boundaries, canonical lifecycle, log root, optimizer LR application, exact identity
  restoration, local-root layout/config, and context/registry semantics.
- `execution/definitions.py` and `execution/artifacts.py`: current local execution dispatches
  train-package **and ONNX/SafeTensors** configured exports. The legacy guide's claim that
  only train-package automation works was stale; migrated content reflects actual dispatch.
  Direct `NexuSession.run()` is distinguished from configured execution-layer export orchestration.
- `core/scenario_loader.py` and `core/provenance.py`: checked trusted file loading,
  uppercase/function exports, project-root detection, snapshot filenames/timing/limitations.
- `tuning/optuna_tuner.py` and the CLI tuning handler exposed stale legacy claims:
  storage is a filesystem path converted to a SQLite `.db`, not a general database URL;
  absent a file `TUNING_SPEC`, CLI constructs defaults instead of automatically using
  `ScenarioSpec.tuning`; categorical/conditional/derived samples route to a factory;
  restricted string expressions reject arithmetic, so arithmetic derivation uses a callable;
  evaluation results are directly available to the tuner, not exclusively through mirroring.
  These are documented limitations, not runtime fixes. Reference also flags the CLI's
  single-best-trial assumption rather than promising multi-objective CLI acceptance.
- CIFAR recipe: retained as optional with actual enabled tracking/callback/export behavior
  and extra dependencies, rather than pretending it is identical to the synthetic default.

## Source-only checks completed

- Parsed all **41 owned MDX pages** using installed `@mdx-js/mdx.compile()` after removing
  frontmatter. Includes the Autodoc and built-in Tabs/TabItem syntax. No raw braces/angle
  syntax errors, MkDocs snippets/admonitions, or malformed MDX were found.
- Checked **27 authored cross-page fragments** against source headings using
  `github-slugger`: all resolved. This includes every required consolidation anchor.
- Python read-only inventory checker: all **38 owned mapped destinations** exist against
  the **57 legacy route entries**; all owned frontmatter is present; all authored internal
  links use canonical routes; generated API destinations map to existing source modules;
  download links match the inventory; no duplicate complete page bodies.
- Parsed **40 Python fenced examples** with `ast.parse`: all syntactically valid.
  This is syntax evidence, not execution, typing, or model-quality evidence.
- Confirmed no legacy `docs/**/*.md` files were modified by this editorial slice.
  Parent-owned CLI/development-documentation pages were excluded from owned-source checks.

The first preliminary ad hoc source scan was intentionally corrected: matching the literal
`/NexuML/` anywhere incorrectly classified legitimate external GitHub repository URLs,
and the scan initially included the parent-owned CLI placeholder. The final URL-specific,
ownership-filtered scan passes; no content workaround was used.

## Remaining integration evidence / unknowns

- **Not run:** Astro build/check/typecheck, Python runtime tests, training, installed-wheel
  CPU smoke, browser/visual/accessibility checks, live external URL/network checks, generated
  reference/link checks, deployment, or cluster operations. This avoids the requested
  concurrent Astro cache races and stays within ownership.
- Parent must execute the exact first-run sequence in its normal-user CPU environment:
  discovery; `resolve synthetic-linear-ae-reconstruction --output configs/first-run.yaml`;
  `build configs/first-run.yaml`; `train --config configs/first-run.yaml --max-epochs 1
  -O training.accelerator=cpu -O training.devices=1`. Verify actual result/artifact claims.
- Source parsing does not establish Starlight aside/tab/Autodoc/Mermaid rendering, API symbol
  fragments, prefix rewriting, navigation, search, or template download behavior. Parent owns
  those integrated gates and the documentation contributor page/global navigation.
- External tutorial/registry/documentation links are retained from existing material or point
  to verified repository files; their live availability was not probed in this slice.
- Tuning source limitations above may merit separate runtime backlog work, but no design,
  runtime, test, or public contract was altered to hide them. The small file search uses
  numeric scalar LR overrides; the structural example explicitly supplies its factory.

Editorial acceptance: complete within ownership, pending parent integration/runtime gates.

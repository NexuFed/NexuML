# NexuML agent skills

Skills for **users of installed NexuML**, not contributors to the framework.
Work in the user's own library/project; no NexuML source checkout, MCP or API
server is required.

| Skill | Responsibility |
| --- | --- |
| [nexuml](nexuml/SKILL.md) | Select the runtime, inspect installed definitions, compose and execute scenarios. |
| [nexuml-library](nexuml-library/SKILL.md) | Build reusable library packages, datasets, layers, losses and evaluation algorithms. |
| [nexuml-reproduce-paper](nexuml-reproduce-paper/SKILL.md) | Reproduce data, algorithm, training and evaluation faithfully, with source-linked evidence. |
| [nexuml-autoresearch](nexuml-autoresearch/SKILL.md) | Search configurations/architectures under explicit budgets and selection rules. |

New algorithm development combines library authoring and experimentation. A
framework bug is a report/blocker, not permission to patch installed NexuML.

## Install the skills separately

Use your agent's skill installer to select these directories, or copy individual
skill directories into its supported skills location. For agents supporting
project-local `.agents/skills/`, install each directory there, preserving its
`SKILL.md`, references and assets. Check the host's documentation for other paths
and commands. Review existing files before replacing a skill.

Only the selected skill files are needed; users need not clone the framework.
Installing the Python distribution does **not** install these agent skills.
Each skill contains its essential safety/runtime rules and works independently;
the `nexuml` skill provides fuller operating guidance when available.

## Select one Python runtime

For component development and research, prefer the user's project environment:
declare compatible NexuML and library dependencies in package metadata, use
`uv sync` for an existing uv project, or install into an existing environment
with `uv pip install`. Install the user's library editably during development.

`uv tool install nexuml` is an alternative for CLI use. Its isolated interpreter
must also have the custom library and its dependencies. Never merge it implicitly
with a project's `.venv`. The optional `library` extra supplies the base library;
`tuning` supplies Optuna. Neither is required for every custom library. Consult
release-appropriate installation instructions before choosing an index or version.

## References and checks

The examples target the typed NexuML 0.2 API. Inspect the actual installed version,
schemas and CLI help; do not silently translate legacy configurations or assume
unreleased interfaces exist.

- [Public CLI/configuration boundaries](nexuml/references/cli-and-config.md)
- [Library component contracts](nexuml-library/references/contracts.md)
- [Offline external-library starter](nexuml-library/assets/minimal-library/README.md)
- [Research/tuning boundaries](nexuml-autoresearch/references/tuning.md)
- [Hypothesis selection and resumable research](nexuml-autoresearch/references/research-state.md)

Run structural checks from this repository with:

```sh
python skills/tests/check_skills.py
```

With compatible installed NexuML, `python skills/tests/check_runtime.py` also
checks key preservation, fitted-score lifecycle/checkpoint state and tuning
parameter/factory semantics. It uses test doubles; no study or training is run.

Repository-only cases live in `skills/tests/evals/`, outside the independently
installed skill payloads. Read-only cases assess decisions/commands; workflow
cases assess naturally selected skills and executable external-project artifacts.
The starter's contract checks remain a useful installable example, not agent
evaluation fixtures. See [evaluation protocol](tests/README.md) for boundaries.
Larger paper reproductions, GPU/distributed runs and description optimization
need their own scope and budget. Skill-creator workspaces are local review
evidence, not shipped skills or a production experiment database.

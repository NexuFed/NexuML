# Environment roots

NexuML uses environment roots as optional prefixes for relative project paths. They do not replace the path fields in a scenario.

## `NEXUML_DATA_ROOT`

The base library's scenario data-root helper resolves a relative dataset path under this variable:

```bash
export NEXUML_DATA_ROOT=/mnt/datasets
```

For example, a library helper that requests `cifar10` resolves to `/mnt/datasets/cifar10`. Absolute dataset paths remain unchanged. Custom `DataSourceDefinition` implementations decide whether/how they use a global data root; the core does not rewrite every arbitrary `root` field automatically.

## `NEXUML_LOGS_ROOT`

Core log-path helpers prefix **relative** log/artifact paths with this variable:

```bash
export NEXUML_LOGS_ROOT=/mnt/experiments
```

If a configured path is `.experiments/diagrams`, it resolves to:

```text
/mnt/experiments/.experiments/diagrams
```

Absolute paths are left unchanged. Relative `file:` URIs handled by the logging helper are similarly resolved; remote/non-file URIs are preserved.

Because each logger/callback/export has its own path field, do not assume one universal hard-coded directory tree. Check the relevant spec/helper or generated API for the artifact you are configuring.

## Typical shell setup

```bash
export NEXUML_DATA_ROOT=/data/nexuml
export NEXUML_LOGS_ROOT=/experiments/nexuml
```

Then keep scenario paths relative when you want them to move with those roots.

## Implementation references

- [`nexuml.core.log_paths`](api/nexuml/core/log_paths.md)
- base-library scenario data-root helper under `nexuml_library.scenarios.data.roots`

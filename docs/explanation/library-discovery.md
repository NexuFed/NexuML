# Library discovery

NexuML discovers decorated component definitions and scenario recipes from Python packages at process startup. Discovery gives persisted YAML a stable identity boundary without forcing normal Python code to construct components by string.

## Discovery sources

NexuML scans:

1. the optional built-in `nexuml_library` package when installed;
2. installed packages exposing the `nexuml.libraries` entry-point group;
3. local package roots registered with `nexuml library add`.

```text
NexuML process
   ↓
scan installed/local library packages
   ↓
collect decorated definitions + scenario recipes
   ↓
component/scenario registries
   ↓
CLI inspection and YAML restoration
```

The entry-point value names an importable package; NexuML scans its tree without a
central `register_all()` function. Configured local roots are rescanned on each
invocation, so new modules need no re-registration. For setup procedures, see
[Register a library](../how-to/register-library.md) and
[Manage local roots](../how-to/library-cli.md).

## Resilient scanning

Import/registration failures are collected as discovery errors instead of hiding unrelated valid components. Use `--verbose` registry output to see tracebacks for failed modules.

Duplicate `(kind, name, version)` component identities are rejected rather than resolved by import order.

## Python vs persistence

Python code should still do this:

```python
LayerSpec(component=MyEncoder(width=128), ...)
```

not a registry lookup by string. At YAML boundaries, `MyEncoder` is lowered to its decorated stable identity and restored through discovery later.

## See also

- [Components and discovery](../learn/decorators-and-discovery.md)
- [Register a library](../how-to/register-library.md)
- [Registry inspection](../reference/registry.md)

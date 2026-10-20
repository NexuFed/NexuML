"""Existing backend descriptions shared by CLI and local API."""


def backend_rows() -> list[tuple[str, str, str]]:
    """Describe existing backends without pretending every dependency is installed.

    Returns:
        Sorted category/name/import-target rows (availability is checked on use).
    """
    from nexuml.core.registry import get_component_registry
    from nexuml.data.export import get_export_backend, list_export_backends

    rows = []
    for name in sorted(list_export_backends()):
        backend = get_export_backend(name)
        rows.append(("data-export", name, f"{backend.__module__}.{backend.__name__}"))
    for entry in get_component_registry().entries(kind="loader_backend"):
        rows.append(("data-loader", entry.name, entry.import_target))
    for entry in get_component_registry().entries(kind="execution_backend"):
        rows.append(("execution", entry.name, entry.import_target))
    rows.extend(
        [
            ("training", "lightning", "nexuml.training.lightning.NexuSession"),
            ("tracking", "tensorboard", "nexuml.tracking.logger"),
            ("tracking", "dvclive", "nexuml.tracking.logger"),
            ("tracking", "mlflow", "nexuml.tracking.logger"),
            ("eval-storage", "memory", "nexuml.evaluation.storage"),
            ("eval-storage", "memmap", "nexuml.evaluation.storage"),
            ("pipeline-export", "package", "nexuml.core.export.export_package"),
            ("pipeline-export", "safetensors", "nexuml.core.export.export_safetensors"),
            ("pipeline-export", "onnx", "nexuml.core.export.export_onnx"),
        ]
    )
    return sorted(rows)

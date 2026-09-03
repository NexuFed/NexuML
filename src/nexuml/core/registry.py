"""Identity registry for typed component definitions."""

from dataclasses import dataclass

from nexuml.core.components import ComponentDefinition
from nexuml.core.discovery import DiscoveryError


@dataclass(frozen=True, slots=True)
class ComponentEntry:
    """Registered stable identity and its concrete definition type."""

    kind: str
    name: str
    version: str
    definition_type: type[ComponentDefinition]
    import_target: str


class ComponentRegistry:
    """Map stable component identities to concrete definition types."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], ComponentEntry] = {}
        self._by_type: dict[type[ComponentDefinition], ComponentEntry] = {}
        self._errors: list[DiscoveryError] = []
        self._loaded = False

    @property
    def errors(self) -> list[DiscoveryError]:
        self.ensure_loaded()
        return list(self._errors)

    def register(
        self,
        name: str,
        definition_type: type[ComponentDefinition],
        *,
        kind: str | None = None,
        version: str | None = None,
    ) -> None:
        if not isinstance(definition_type, type) or not issubclass(
            definition_type, ComponentDefinition
        ):
            raise TypeError("registered components must inherit ComponentDefinition")

        resolved_kind = kind or definition_type.kind
        resolved_version = version or definition_type.component_version
        key = (resolved_kind, name, resolved_version)
        target = f"{definition_type.__module__}.{definition_type.__qualname__}"
        existing = self._entries.get(key)
        if existing is not None:
            if existing.definition_type is definition_type:
                return
            raise ValueError(
                "Component registry conflict for "
                f"{resolved_kind!r}/{name!r}/{resolved_version!r}: "
                f"{existing.import_target} and {target}"
            )

        previous = self._by_type.get(definition_type)
        if previous is not None:
            raise ValueError(
                f"Component type {target} is already registered as "
                f"{previous.kind!r}/{previous.name!r}/{previous.version!r}"
            )

        entry = ComponentEntry(
            kind=resolved_kind,
            name=name,
            version=resolved_version,
            definition_type=definition_type,
            import_target=target,
        )
        self._entries[key] = entry
        self._by_type[definition_type] = entry

    def get_entry(self, kind: str, name: str, version: str = "1") -> ComponentEntry:
        self.ensure_loaded()
        try:
            return self._entries[(kind, name, version)]
        except KeyError as exc:
            available = ", ".join(
                f"{entry.name}@{entry.version}" for entry in self.entries(kind=kind)
            )
            raise KeyError(
                f"Unknown component {kind!r}/{name!r}/{version!r}. Available: [{available}]"
            ) from exc

    def get_type(self, kind: str, name: str, version: str = "1") -> type[ComponentDefinition]:
        return self.get_entry(kind, name, version).definition_type

    def entry_for_type(self, definition_type: type[ComponentDefinition]) -> ComponentEntry:
        self.ensure_loaded()
        try:
            return self._by_type[definition_type]
        except KeyError as exc:
            target = f"{definition_type.__module__}.{definition_type.__qualname__}"
            raise KeyError(f"Component type {target} is not registered") from exc

    def entries(self, *, kind: str | None = None) -> tuple[ComponentEntry, ...]:
        self.ensure_loaded()
        values = self._entries.values()
        if kind is not None:
            values = (entry for entry in values if entry.kind == kind)
        return tuple(sorted(values, key=lambda entry: (entry.kind, entry.name, entry.version)))

    def scan(self, package_paths: list[str] | None = None) -> None:
        from nexuml.core.discovery import Scanner, discover_library_packages, register_items

        scanner = Scanner()
        for package_path in package_paths or discover_library_packages():
            scanner.scan_package(package_path)

        self._errors = list(scanner.errors)
        for kind in ("layer", "data_source", "eval_algorithm", "loader_backend"):
            register_items(scanner.by_kind(kind), self.register, self._errors)
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.scan()


_default_registry: ComponentRegistry | None = None


def get_component_registry() -> ComponentRegistry:
    """Return the process-wide component identity registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ComponentRegistry()
    return _default_registry

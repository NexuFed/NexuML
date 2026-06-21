"""CLI override utilities: apply key.path=value overrides to a ScenarioSpec."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def _coerce_value(raw: str) -> Any:
    """Parse a CLI string into a typed Python value.

    Returns:
        int, float, bool, list, or the original string if no coercion applies.
    """
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.startswith(("[", "{")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def apply_overrides(scenario: Any, overrides: list[str]) -> Any:
    """Apply ``key.path=value`` overrides to a ScenarioSpec in-place.

    Dot-separated paths traverse Pydantic model attributes and dict keys.
    Values are coerced to int/float/bool/list before assignment.
    Prints a Rich table summarising each change.

    Returns:
        The mutated *scenario* instance.

    Raises:
        ValueError: If an override string is malformed or has an empty path segment.
        KeyError: If a path segment does not exist on the scenario.
    """
    if not overrides:
        return scenario

    rows: list[tuple[str, str, str]] = []
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Invalid override (expected key=value): {spec!r}")
        key, _, raw = spec.partition("=")
        parts = key.split(".")
        if not all(parts):
            raise ValueError(f"Invalid override path (empty segment): {spec!r}")

        obj = scenario
        try:
            for part in parts[:-1]:
                if isinstance(obj, dict):
                    obj = obj[part]
                else:
                    obj = getattr(obj, part)
        except (AttributeError, KeyError) as exc:
            raise KeyError(f"Override path not found: {key!r}") from exc

        attr = parts[-1]
        new_value = _coerce_value(raw)

        if isinstance(obj, dict):
            old_value = obj.get(attr, "<unset>")
            obj[attr] = new_value
        else:
            if not hasattr(obj, attr):
                raise KeyError(f"Override path not found: {key!r}")
            old_value = getattr(obj, attr)
            setattr(obj, attr, new_value)

        rows.append((key, str(old_value), str(new_value)))

    table = Table(title="Applied overrides")
    table.add_column("Key", style="cyan")
    table.add_column("Old Value", style="yellow")
    table.add_column("New Value", style="green")
    for row in rows:
        table.add_row(*row)
    console.print(table)

    return scenario

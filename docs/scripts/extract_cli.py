"""Extract the real CLI's public Click tree without executing command callbacks."""

from __future__ import annotations

import argparse
import inspect
import json
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

from typer import Context
from typer.core import TyperGroup
from typer.main import get_command

from nexuml.cli.main import app


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_default(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"<dynamic: {module}.{name}>"
    raise TypeError(f"Unsupported CLI default type: {type(value).__qualname__}")


def extract_cli() -> list[dict[str, Any]]:
    """Return public commands and parameters from a fresh conversion of the real app."""
    commands = []

    def visit(command: Any, name: str, parent: Any = None) -> None:
        if command.hidden:
            return
        with Context(command, info_name=name, parent=parent, **command.context_settings) as ctx:
            parameters = []
            for parameter in command.get_params(ctx):
                if getattr(parameter, "hidden", False):
                    continue
                parameters.append(
                    {
                        "name": parameter.name,
                        "kind": parameter.param_type_name,
                        "opts": parameter.opts,
                        "secondary_opts": parameter.secondary_opts,
                        "type": parameter.type.get_metavar(param=parameter, ctx=ctx)
                        or parameter.type.name,
                        "required": parameter.required,
                        "multiple": parameter.multiple,
                        "nargs": parameter.nargs,
                        "default": _json_default(parameter.get_default(ctx, call=False)),
                        "help": inspect.cleandoc(getattr(parameter, "help", None) or ""),
                    }
                )
            commands.append(
                {
                    "path": ctx.command_path,
                    "help": inspect.cleandoc(command.help or ""),
                    "parameters": parameters,
                }
            )
            if isinstance(command, TyperGroup):
                for child_name in command.list_commands(ctx):
                    child = command.get_command(ctx, child_name)
                    if child is not None:
                        visit(child, child_name, ctx)

    visit(get_command(app), "nexuml")
    return commands


def main() -> None:
    """Write extracted JSON to an explicitly selected generated-output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.dumps(extract_cli(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

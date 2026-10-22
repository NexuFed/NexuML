"""Validate editorial metadata and route coverage before generated-page compilation."""

import json
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2] / "docs"


def check_content(root: Path = ROOT) -> None:
    """Reject missing editorial metadata and missing canonical migration destinations.

    Raises:
        ValueError: If frontmatter or an inventoried destination is missing.
    """
    content = root / "src/content/docs"
    for page in content.rglob("*.mdx"):
        text = page.read_text()
        if not text.startswith("---\n") or "\n---\n" not in text[4:]:
            raise ValueError(f"{page}: MDX frontmatter required")
        metadata = YAML(typ="safe").load(text[4:].split("\n---\n", 1)[0])
        for field in ("title", "description"):
            if (
                not isinstance(metadata, dict)
                or not isinstance(metadata.get(field), str)
                or not metadata[field].strip()
            ):
                raise ValueError(f"{page}: nonempty {field} required")
    mapping = json.loads((root / "migration/route-map.json").read_text())
    for destination in set(mapping["pages"].values()):
        route = destination.split("#")[0].rstrip("/")
        if route.startswith("reference/api/"):
            continue
        if not any((content / path).is_file() for path in (f"{route}.mdx", f"{route}/index.mdx")):
            if not route and (content / "index.mdx").is_file():
                continue
            raise ValueError(f"Missing canonical route: {destination}")


if __name__ == "__main__":
    check_content()
    print("Editorial metadata and migration coverage pass")

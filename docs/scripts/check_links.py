"""Check rendered local links, fragments, CSS URLs, and assets without network access."""

from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse


class Page(HTMLParser):
    """Collect actual rendered anchors and resource requests."""

    def __init__(self, text: str, not_found: bool = False):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.not_found = not_found
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if self.not_found and tag == "link" and values.get("rel") == "canonical":
            return
        if value := values.get("id"):
            self.ids.add(value)
        for name in ("href", "src", "poster"):
            if value := values.get(name):
                self.links.append(value)
        if srcset := values.get("srcset"):
            self.links.extend(item.strip().split()[0] for item in srcset.split(","))


def check(root: Path, site: str = "https://nexufed.github.io") -> list[str]:
    """Return all broken rendered paths/fragments within an assembled static tree."""
    root = root.resolve()
    pages = {path: Page(path.read_text(), path.name == "404.html") for path in root.rglob("*.html")}
    failures = []
    documents = [(path, page.links) for path, page in pages.items()]
    documents.extend(
        (path, re.findall(r"url\([\"']?([^\"')]+)", path.read_text()))
        for path in root.rglob("*.css")
    )
    for source, links in documents:
        relative = source.relative_to(root).as_posix()
        current = f"{site}/{relative}"
        for link in links:
            if not link or link.startswith(("data:", "mailto:", "tel:", "blob:")):
                continue
            url = urlparse(urljoin(current, link))
            if url.scheme not in ("http", "https"):
                failures.append(f"{relative}: unsafe URL {link}")
                continue
            if url.netloc != urlparse(site).netloc:
                continue
            target = root / unquote(url.path).lstrip("/")
            if not target.resolve().is_relative_to(root):
                failures.append(f"{relative}: path escapes output: {link}")
                continue
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                failures.append(f"{relative}: missing {link}")
            elif url.fragment and target.suffix == ".html":
                page = pages.get(target)
                if page is None:
                    page = Page(target.read_text())
                if unquote(url.fragment) not in page.ids:
                    failures.append(f"{relative}: missing fragment {link}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--site", default="https://nexufed.github.io")
    args = parser.parse_args()
    errors = check(args.root, args.site)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Rendered links, fragments, and assets pass")

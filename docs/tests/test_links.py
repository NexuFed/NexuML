"""The link gate must reject broken compiled content, not only broken source MDX."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_links", ROOT / "docs/scripts/check_links.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rendered_links_and_css_assets(tmp_path):
    root = tmp_path / "NexuML/dev"
    root.mkdir(parents=True)
    (root / "index.html").write_text(
        '<h1 id="ready">Ready</h1><a href="#ready">Go</a><link href="style.css" rel="stylesheet">'
    )
    (root / "style.css").write_text('body { background: url("image.webp"); }')
    (root / "image.webp").write_bytes(b"fixture")
    assert module.check(tmp_path) == []
    (root / "image.webp").unlink()
    assert any("image.webp" in e for e in module.check(tmp_path))
    (root / "index.html").write_text(
        '<a href="#not-here">Broken</a><a href="/missing/">Missing</a>'
    )
    errors = module.check(tmp_path)
    assert any("missing fragment" in e for e in errors)
    assert any("missing /missing/" in e for e in errors)

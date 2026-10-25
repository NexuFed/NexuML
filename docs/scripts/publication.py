"""Compose an existing documentation archive; optionally stage it with a fast-forward push.

Only staging invokes a write to Git. Builds and their exact-revision validation belong
to the caller. No checkout, reset, force push, or modification of source inputs occurs.
"""

# CLI helpers use concise summaries rather than full public-library API sections.
# ruff: noqa: DOC201, DOC501

import argparse
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
STABLE = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")
SHA = re.compile(r"[a-f0-9]{40}\Z")
BRANCHES = {"main", "dev"}
GLOBAL = {
    "changelog",
    "_astro",
    "pagefind",
    "404.html",
    "sitemap-index.xml",
    "sitemap-0.xml",
    "favicon.png",
    "build.json",
}
RESERVED = GLOBAL | BRANCHES | {"api", "latest", "versions.json", "publication.json", ".git"}


def git(repository, *args, check=True, input=None, env=None):
    """Run Git without a shell, preserving diagnostics on failure."""
    result = subprocess.run(
        ["git", "-C", str(repository), *args], input=input, capture_output=True, env=env
    )
    if check and result.returncode:
        raise ValueError(f"git {' '.join(args)}: {result.stderr.decode(errors='replace').strip()}")
    return result


def safe_path(value):
    """Reject root, traversal and symlink components before using a filesystem path."""
    path = Path(value).absolute()
    if {"..", ".git"}.intersection(Path(value).parts) or path == Path(path.anchor):
        raise ValueError(f"Unsafe path: {value}")
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError(f"Symlink path component: {part}")
    return path


def validate_tree(root, legacy_latest=False):
    """Accept ordinary directories/files, with only the old root latest symlink allowed."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Missing or unsafe directory: {root}")
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if name == ".git":
                raise ValueError(f"Git internals are not publication content: {path}")
            if stat.S_ISLNK(mode):
                target = os.readlink(path)
                if not (
                    legacy_latest
                    and path == root / "latest"
                    and STABLE.fullmatch(target)
                    and (root / target).is_dir()
                    and not (root / target).is_symlink()
                ):
                    raise ValueError(f"Unsupported symlink: {path}")
            elif not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f"Unsupported file type: {path}")


def manifest(root, names=None):
    """Hash relative paths, types, full permission modes and file bytes (not timestamps)."""
    result = {}

    def visit(path):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            result[relative] = ["link", mode, os.readlink(path)]
        elif path.is_dir():
            result[relative] = ["directory", mode]
            for child in sorted(path.iterdir()):
                visit(child)
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[relative] = ["file", mode, digest.hexdigest()]

    if names is None:
        visit(root)
    else:
        for name in sorted(names):
            if (root / name).exists():
                visit(root / name)
    return result


def tree_digest(root, names=None):
    """Return a deterministic SHA-256 of the complete tree manifest."""
    return hashlib.sha256(
        json.dumps(manifest(root, names), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def routes(root):
    """Inventory actual directory index pages, relative to the documentation root."""
    found = sorted(
        "" if page.parent == root else page.parent.relative_to(root).as_posix() + "/"
        for page in root.rglob("index.html")
        if page.is_file()
    )
    for path in found:
        route(path)
    return found


def route(value, fragment=False):
    """Validate a documentation-relative route, optionally ending in a mapped fragment."""
    parsed = urlsplit(value)
    path = parsed.path
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or (parsed.fragment and not fragment)
        or path.startswith("/")
        or "\\" in value
        or "%" in path
        or any(p in {".", "..", ".git"} for p in path.split("/"))
        or "//" in path
        or (path and not path.endswith("/"))
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"Unsafe route: {value!r}")
    return path


def hosting(mount, site):
    """Validate explicit origin/mount configuration; never infer it from content."""
    parsed = urlsplit(site)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or any(c.isspace() for c in site)
    ):
        raise ValueError("--site must be an HTTP(S) origin without credentials or a path")
    if not re.fullmatch(r"/(?:[A-Za-z0-9_-]+/)*", mount):
        raise ValueError("--mount must be / or slash-delimited safe path segments")
    return mount, site.rstrip("/")


def verify_source(repository, version, revision, source_ref=None):
    """Verify branch ancestry, or the exact peeled stable tag commit."""
    if version not in BRANCHES and not STABLE.fullmatch(version):
        raise ValueError(
            "Version must be main, dev or a stable vX.Y.Z (no prerelease/leading zeros)"
        )
    if not SHA.fullmatch(revision):
        raise ValueError("Revision must be a full lowercase 40-character commit SHA")
    ref = source_ref or (f"refs/heads/{version}" if version in BRANCHES else f"refs/tags/{version}")
    if version in BRANCHES and ref not in {
        f"refs/heads/{version}",
        f"refs/remotes/origin/{version}",
    }:
        raise ValueError("Branch source ref must match its publication version")
    if version not in BRANCHES and ref != f"refs/tags/{version}":
        raise ValueError("A release source ref must be its exact refs/tags/vX.Y.Z")
    if (
        not ref.startswith("refs/")
        or git(repository, "check-ref-format", ref, check=False).returncode
    ):
        raise ValueError("Source ref must be a fully qualified Git ref")
    actual = (
        git(repository, "rev-parse", "--verify", f"{revision}^{{commit}}").stdout.decode().strip()
    )
    tip = git(repository, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.decode().strip()
    if actual != revision:
        raise ValueError("Revision must identify a commit, not a tag object")
    if version in BRANCHES:
        if not is_ancestor(repository, revision, tip):
            raise ValueError(
                "Branch candidate revision is not an ancestor of the trusted source ref"
            )
    elif revision != tip:
        raise ValueError("Release revision does not match the trusted tag ref")


def is_ancestor(repository, older, newer):
    """Fail closed on missing history instead of mistaking a Git error for divergence."""
    result = git(repository, "merge-base", "--is-ancestor", older, newer, check=False)
    if result.returncode not in {0, 1}:
        raise ValueError(result.stderr.decode(errors="replace"))
    return result.returncode == 0


def source_order(repository, incoming, recorded):
    """Return same/newer/stale, rejecting divergent publication histories."""
    if incoming == recorded:
        return "same"
    if is_ancestor(repository, recorded, incoming):
        return "newer"
    if is_ancestor(repository, incoming, recorded):
        return "stale"
    raise ValueError("Divergent branch history; publication requires maintainer investigation")


def load_archive(archive):
    """Validate mike metadata and independently inventory every stable directory."""
    versions = json.loads((archive / "versions.json").read_text())
    if not isinstance(versions, list):
        raise ValueError("versions.json must be a mike-compatible array")
    seen = set()
    selected = None
    for entry in versions:
        if not isinstance(entry, dict):
            raise ValueError("Every mike version entry must be an object")
        name = entry["version"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            or name in RESERVED - BRANCHES
            or name in {".", ".."}
            or name in seen
            or not (archive / name).is_dir()
            or not isinstance(entry.get("aliases"), list)
            or not all(isinstance(alias, str) for alias in entry["aliases"])
        ):
            raise ValueError(f"Invalid/missing archive version: {entry!r}")
        seen.add(name)
        if "latest" in entry["aliases"]:
            if selected is not None or not STABLE.fullmatch(name):
                raise ValueError("latest must identify exactly one stable release")
            selected = name
    if (archive / "latest").is_symlink():
        target = os.readlink(archive / "latest")
        if selected is not None and target != selected:
            raise ValueError("latest symlink disagrees with versions.json")
        selected = target
    for path in sorted(archive.iterdir()):
        if (
            (STABLE.fullmatch(path.name) or path.name in BRANCHES)
            and path.is_dir()
            and path.name not in seen
        ):
            versions.append({"version": path.name, "title": path.name, "aliases": []})
            seen.add(path.name)
    stable = [name for name in seen if STABLE.fullmatch(name)]
    selected = selected or (max(stable, key=semver) if stable else None)
    if (archive / "latest").exists() and not (archive / "latest").is_symlink():
        if not selected or tree_digest(archive / "latest") != tree_digest(archive / selected):
            raise ValueError("Materialized latest does not match its selected release")
    state_file = archive / "publication.json"
    state: dict[str, Any] = (
        json.loads(state_file.read_text())
        if state_file.exists()
        else {
            "schemaVersion": 1,
            "versions": {},
            "global": {"revision": None, "digest": tree_digest(archive, GLOBAL)},
        }
    )
    if (
        not isinstance(state, dict)
        or state.get("schemaVersion") != 1
        or not isinstance(state.get("versions"), dict)
        or not isinstance(state.get("global"), dict)
    ):
        raise ValueError("Unsupported publication.json schema")
    for name in seen:
        snapshot = archive / name
        if not (snapshot / "index.html").is_file():
            raise ValueError(f"Archive snapshot has no home index.html: {name}")
        actual = {"revision": None, "digest": tree_digest(snapshot), "routes": routes(snapshot)}
        if name in state["versions"]:
            recorded = state["versions"][name]
            if not isinstance(recorded, dict):
                raise ValueError(f"Invalid publication record: {name}")
            if recorded.get("revision") is not None and not SHA.fullmatch(recorded["revision"]):
                raise ValueError(f"Invalid recorded revision: {name}")
            if (
                recorded.get("digest") != actual["digest"]
                or recorded.get("routes") != actual["routes"]
            ):
                raise ValueError(f"Archive snapshot no longer matches publication.json: {name}")
        else:
            state["versions"][name] = actual
    if set(state["versions"]) != seen:
        raise ValueError("publication.json references a missing/unlisted archive version")
    global_state = state["global"]
    if global_state.get("digest") != tree_digest(archive, GLOBAL) or (
        global_state.get("revision") is not None and not SHA.fullmatch(global_state["revision"])
    ):
        raise ValueError("Global archive no longer matches publication.json")
    return versions, state, selected


def semver(version):
    """Sort only stable numeric release identifiers."""
    match = STABLE.fullmatch(version)
    assert match is not None
    return tuple(map(int, match.groups()))


def redirect(target, site):
    """Provide a static redirect with an accessible fallback, not an HTTP 301."""
    escaped = html.escape(target, quote=True)
    canonical = html.escape(site + target, quote=True)
    script_target = json.dumps(target).replace("<", "\\u003c")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width">'
        '<title>Documentation moved</title><meta name="robots" content="noindex">'
        f'<link rel="canonical" href="{canonical}">'
        f'<meta http-equiv="refresh" content="0;url={escaped}"></head><body>'
        f'<p>This documentation moved. <a href="{escaped}">Continue to documentation</a>.</p>'
        f"<script>const target = new URL({script_target}, window.location.href);"
        "target.search = window.location.search;"
        "target.hash = window.location.hash || target.hash;"
        "window.location.replace(target.href);</script></body></html>\n"
    )


def mappings():
    """Load the reviewed authored route inventory, including consolidation fragments."""
    pages = json.loads((ROOT / "docs/migration/route-map.json").read_text())["pages"]
    result = {}
    for source, target in pages.items():
        old = str(PurePosixPath(source).with_suffix(""))
        old = "" if old == "index" else old.removesuffix("/index") + "/"
        route(old)
        route(target, fragment=True)
        result[old] = target
    return result


def destination(available, original, mapped):
    """Prefer an actual old page, then its mapped equivalent/nearest section/home."""
    if original in available:
        return original
    path = route(mapped, fragment=True)
    if path in available:
        return mapped
    while path:
        path = path.rstrip("/").rpartition("/")[0]
        path = path + "/" if path else ""
        if path in available:
            return path
    return ""


def reserved(path):
    """Keep global routes and version namespaces out of root documentation aliases."""
    first = path.split("/", 1)[0]
    return first in RESERVED or bool(STABLE.fullmatch(first)) or first.startswith(".")


def write_redirect(root, path, target, site):
    """Write only the owned route index; unrelated files alongside it remain untouched."""
    page = root / path / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(redirect(target, site))
    return page.relative_to(root).as_posix()


def remove(path):
    """Remove an output-owned path, never an input path."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def write_json(path, value):
    """Write deterministic shared metadata."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n")


def compose(
    *,
    archive,
    candidate,
    output,
    version,
    revision,
    mount,
    site,
    repository,
    changelog=None,
    source_ref=None,
):
    """Build a separate, preservation-checked tree from mandatory existing archive data."""
    archive, candidate, output, repository = map(
        safe_path, (archive, candidate, output, repository)
    )
    changelog = safe_path(changelog) if changelog is not None else None
    inputs = [archive, candidate] + ([changelog] if changelog is not None else [])
    paths = [*inputs, output]
    for i, path in enumerate(paths):
        for other in paths[i + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError(f"Overlapping publication paths: {path}, {other}")
    if output.exists() or output == repository or output in repository.parents:
        raise ValueError("Output must be a new directory, not the repository or its ancestor")
    if not output.parent.is_dir():
        raise ValueError(
            "Output parent must already exist; composition writes only the output tree"
        )
    mount, site = hosting(mount, site)
    verify_source(repository, version, revision, source_ref)
    validate_tree(archive, legacy_latest=True)
    validate_tree(candidate)
    if not (candidate / "index.html").is_file():
        raise ValueError("Versioned candidate must contain index.html")
    if changelog is not None:
        if version != "main":
            raise ValueError("Only main publication may supply --changelog")
        validate_tree(changelog)
        if {p.name for p in changelog.iterdir()} - GLOBAL:
            raise ValueError(
                "Global build contains paths outside its owned changelog/support paths"
            )
        if not (changelog / "changelog/index.html").is_file():
            raise ValueError("Global candidate must contain changelog/index.html")
    versions, state, selected = load_archive(archive)
    before = manifest(archive)
    records = [state["versions"].get(version, {})]
    if changelog is not None:
        records.append(state["global"])
    stale = version in BRANCHES and any(
        source_order(repository, revision, record["revision"]) == "stale"
        for record in records
        if record.get("revision") is not None
    )
    # Copying breaks hardlinks; the one allowed legacy symlink is materialized separately.
    try:
        shutil.copytree(
            archive,
            output,
            ignore=lambda directory, names: (
                ["latest"]
                if Path(directory) == archive and (archive / "latest").is_symlink()
                else []
            ),
        )
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise
    allowed_trees = {"latest"} if (archive / "latest").is_symlink() else set()
    allowed_files = {"versions.json", "publication.json", ".nojekyll"}
    status = "stale" if stale else "composed"
    try:
        if not stale:
            with tempfile.TemporaryDirectory(prefix=".publication-", dir=output) as temporary:
                prepared = Path(temporary) / "snapshot"
                shutil.copytree(candidate, prepared)
                available = set(routes(prepared))
                for old, mapped in mappings().items():
                    if old not in available and not reserved(old):
                        write_redirect(
                            prepared,
                            old,
                            f"{mount}{version}/" + destination(available, old, mapped),
                            site,
                        )
                digest = tree_digest(prepared)
                record = state["versions"].get(version)
                if (archive / version).exists() and (
                    version not in BRANCHES or (record and record["revision"] == revision)
                ):
                    if not record or record["revision"] != revision or record["digest"] != digest:
                        raise ValueError(f"Immutable/conflicting retry for {version}")
                    status = "idempotent"
                else:
                    remove(output / version)
                    shutil.move(str(prepared), output / version)
                    allowed_trees.add(version)
                    state["versions"][version] = {
                        "revision": revision,
                        "digest": digest,
                        "routes": routes(output / version),
                    }
                if changelog is not None:
                    digest = tree_digest(changelog, GLOBAL)
                    if (
                        state["global"]["revision"] == revision
                        and state["global"]["digest"] != digest
                    ):
                        raise ValueError("Conflicting global build for the same revision")
                    for name in GLOBAL:
                        remove(output / name)
                        source = changelog / name
                        if source.is_dir():
                            shutil.copytree(source, output / name)
                        elif source.exists():
                            shutil.copy2(source, output / name)
                    allowed_trees.update(GLOBAL)
                    state["global"] = {"revision": revision, "digest": digest}
            if version not in {entry["version"] for entry in versions}:
                versions.append({"version": version, "title": version, "aliases": []})
            if version not in BRANCHES:
                selected = max((v for v in state["versions"] if STABLE.fullmatch(v)), key=semver)
            for entry in versions:
                if entry["version"] in BRANCHES:
                    entry["title"] = f"{entry['version']} (unreleased)"
                entry["aliases"] = [alias for alias in entry["aliases"] if alias != "latest"]
                if entry["version"] == selected:
                    entry["aliases"].append("latest")
            versions.sort(
                key=lambda entry: (
                    3
                    if entry["version"] == "main"
                    else 2
                    if entry["version"] == "dev"
                    else 1
                    if STABLE.fullmatch(entry["version"])
                    else 0,
                    semver(entry["version"]) if STABLE.fullmatch(entry["version"]) else (),
                    entry["version"],
                ),
                reverse=True,
            )
            write_json(output / "versions.json", versions)
            write_json(output / "publication.json", state)
            target_version = selected or ("main" if "main" in state["versions"] else "dev")
            available = set(routes(output / target_version))
            for old in sorted(set(mappings()) | available):
                if not reserved(old):
                    target = destination(available, old, mappings().get(old, old))
                    allowed_files.add(
                        write_redirect(output, old, f"{mount}{target_version}/" + target, site)
                    )
        if selected and (
            not (output / "latest").exists()
            or tree_digest(output / "latest") != tree_digest(output / selected)
        ):
            remove(output / "latest")
            shutil.copytree(output / selected, output / "latest")
            allowed_trees.add("latest")
        if not (output / ".nojekyll").exists():
            (output / ".nojekyll").touch()
        validate_tree(output)
        after = manifest(output)
        for path in before.keys() | after.keys():
            if before.get(path) == after.get(path):
                continue
            if path in allowed_files or any(
                path == p or path.startswith(p + "/") for p in allowed_trees
            ):
                continue
            if (
                path not in before
                and after[path][0] == "directory"
                and any(p.startswith(path + "/") for p in allowed_files)
            ):
                continue
            raise ValueError(f"Publication changed an unowned archive path: {path}")
        for entry in versions:
            name = entry["version"]
            if name != version and (archive / name).is_dir():
                if tree_digest(archive / name) != tree_digest(output / name):
                    raise ValueError(f"Publication modified an immutable snapshot: {name}")
    except Exception:
        shutil.rmtree(output)
        raise
    return {"status": status, "version": version, "revision": revision, "latest": selected}


def extract_archive(repository, revision, output):
    """Read exact Git blobs/modes, without export-ignore/export-subst or a checkout."""
    output.mkdir()
    listing = git(repository, "ls-tree", "-rz", revision).stdout
    process = subprocess.Popen(
        ["git", "-C", str(repository), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for entry in listing.split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            name = raw_path.decode("utf-8")
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(p in {".", "..", ".git"} for p in relative.parts):
                raise ValueError(f"Unsafe Git archive path: {name}")
            if kind != "blob" or mode not in {"100644", "100755", "120000"}:
                raise ValueError(f"Unsupported Git archive entry: {name}")
            if mode == "120000" and name != "latest":
                raise ValueError(f"Unsupported archive symlink: {name}")
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            process.stdin.write(oid.encode() + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError(f"Cannot read archive blob: {oid}")
            content = process.stdout.read(int(header[2]))
            if process.stdout.read(1) != b"\n":
                raise ValueError("Truncated Git blob stream")
            if mode == "120000":
                if not STABLE.fullmatch(content.decode()):
                    raise ValueError("Unsafe latest symlink target")
                path.symlink_to(content.decode())
            else:
                path.write_bytes(content)
                path.chmod(0o755 if mode == "100755" else 0o644)
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    if process.returncode:
        raise ValueError("Git blob extraction failed")


def stage(*, repository, remote, branch="gh-pages", attempts=4, **options):
    """Refetch/recompose each competing archive update, then push without force."""
    repository = safe_path(repository)
    verify_source(repository, options["version"], options["revision"], options.get("source_ref"))
    if not 1 <= attempts <= 10:
        raise ValueError("Staging attempts must be between 1 and 10")
    if git(repository, "check-ref-format", f"refs/heads/{branch}", check=False).returncode:
        raise ValueError("Invalid archive branch")
    if remote.startswith("-"):
        raise ValueError("Invalid remote")
    # Resolve named remotes before entering the isolated staging repository.
    configured = git(repository, "remote", "get-url", remote, check=False)
    if configured.returncode == 0:
        remote = configured.stdout.decode().strip()
    elif ":" not in remote and not Path(remote).is_absolute():
        remote = str((repository / remote).resolve())
    base = Path(os.environ.get("TMPDIR", "/tmp/opencode"))
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docs-stage-", dir=base) as temporary:
        temporary = Path(temporary)
        staging = temporary / "git"
        git(temporary, "init", "--bare", str(staging))
        previous = None
        for attempt in range(1, attempts + 1):
            git(staging, "fetch", "--no-tags", remote, f"refs/heads/{branch}")
            head = git(staging, "rev-parse", "FETCH_HEAD").stdout.decode().strip()
            if head == previous:
                raise ValueError("Push failed without a competing archive update; not retrying")
            directory = temporary / str(attempt)
            directory.mkdir()
            archive = directory / "archive"
            output = directory / "output"
            extract_archive(staging, head, archive)
            report = compose(repository=repository, archive=archive, output=output, **options)
            # Git cannot record arbitrary POSIX permissions. Reject, rather than silently lose them.
            for path, entry in manifest(output).items():
                if entry[1] not in ({0o755} if entry[0] == "directory" else {0o644, 0o755}):
                    raise ValueError(f"Git cannot preserve this file mode: {path}")
            git(staging, f"--work-tree={output}", "add", "--force", "--all", "--", ".")
            tree = git(staging, "write-tree").stdout.decode().strip()
            roundtrip = directory / "roundtrip"
            extract_archive(staging, tree, roundtrip)
            if manifest(roundtrip) != manifest(output):
                raise ValueError("Git staging changed publication bytes, modes or paths")
            old_tree = git(staging, "rev-parse", f"{head}^{{tree}}").stdout.decode().strip()
            if tree == old_tree:
                return {**report, "archiveCommit": head, "attempts": attempt, "staged": False}
            identity = {
                **os.environ,
                "GIT_AUTHOR_NAME": "Documentation publisher",
                "GIT_AUTHOR_EMAIL": "docs@users.noreply.github.com",
                "GIT_COMMITTER_NAME": "Documentation publisher",
                "GIT_COMMITTER_EMAIL": "docs@users.noreply.github.com",
            }
            commit = (
                git(
                    staging,
                    "commit-tree",
                    tree,
                    "-p",
                    head,
                    input=f"Publish docs {options['version']} at {options['revision']}\n".encode(),
                    env=identity,
                )
                .stdout.decode()
                .strip()
            )
            pushed = git(staging, "push", remote, f"{commit}:refs/heads/{branch}", check=False)
            if not pushed.returncode:
                return {**report, "archiveCommit": commit, "attempts": attempt, "staged": True}
            previous = head
        raise ValueError(
            f"Archive staging conflicted after {attempts} attempts; rerun validated candidate"
        )


def main(argv=None):
    """Expose pure composition by default and explicit opt-in Git staging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--changelog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mount", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source-ref", help="Trusted, locally fetched matching branch/tag ref")
    parser.add_argument("--stage", action="store_true", help="Opt in to fast-forward archive push")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="gh-pages")
    parser.add_argument("--attempts", type=int, default=4)
    args = vars(parser.parse_args(argv))
    staging = args.pop("stage")
    try:
        if staging:
            if args.pop("archive") is not None or args.pop("output") is not None:
                parser.error("--stage fetches its own archive; do not pass --archive/--output")
            report = stage(**args)
        else:
            for name in ("remote", "branch", "attempts"):
                args.pop(name)
            if args["archive"] is None or args["output"] is None:
                parser.error("Local composition requires --archive and --output")
            report = compose(**args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Publication refused: {error}\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

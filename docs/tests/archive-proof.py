"""Dry-run new builds against the real retained archive, without remote writes."""

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
publication = runpy.run_path(str(ROOT / "docs/scripts/publication.py"))
verify = runpy.run_path(str(ROOT / "docs/scripts/verify_candidate.py"))["verify"]


def git(repository, *args):
    return (
        subprocess.check_output(["git", "-C", str(repository), *args], stderr=subprocess.PIPE)
        .decode()
        .strip()
    )


def prove(archive_ref, report):
    identity = json.loads((ROOT / "docs/.output/versioned/build.json").read_text())
    verify(ROOT / "docs/.output", identity["revision"], "dev")
    Path("/tmp/opencode").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docs-real-archive-", dir="/tmp/opencode") as temporary:
        temporary = Path(temporary)
        source = temporary / "source.git"
        subprocess.run(
            ["git", "clone", "--quiet", "--shared", "--bare", str(ROOT), str(source)], check=True
        )
        git(source, "branch", "docs-proof", identity["revision"])
        archive = temporary / "archive"
        publication["extract_archive"](ROOT, archive_ref, archive)
        original = publication["manifest"](archive)
        dev = temporary / "dev-candidate"
        shutil.copytree(ROOT / "docs/.output/versioned", dev)
        global_build = temporary / "global-candidate"
        shutil.copytree(ROOT / "docs/.output/changelog", global_build)
        common = dict(
            repository=source,
            revision=identity["revision"],
            site=identity["site"],
            mount=identity["mount"],
        )
        main = temporary / "main"
        main_result = publication["compose"](
            archive=archive,
            candidate=dev,
            changelog=global_build,
            output=main,
            version="dev",
            source_ref="refs/heads/docs-proof",
            **common,
        )
        historical = [
            p.name
            for p in archive.iterdir()
            if p.is_dir() and publication["STABLE"].fullmatch(p.name)
        ]
        hashes = {}
        for name in historical:
            before = publication["tree_digest"](archive / name)
            after = publication["tree_digest"](main / name)
            assert before == after, name
            hashes[name] = before
        assert publication["manifest"](archive) == original
        assert not (main / "latest").is_symlink()
        assert publication["tree_digest"](main / "latest") == publication["tree_digest"](
            main / main_result["latest"]
        )
        version = "v999.0.0"
        git(source, "tag", version, identity["revision"])
        env = {**os.environ, "DOCS_VERSION": version}
        subprocess.run(
            ["node", "docs/scripts/build.mjs", "versioned"], cwd=ROOT, env=env, check=True
        )
        tagged = temporary / "tagged"
        tag_result = publication["compose"](
            archive=main,
            candidate=ROOT / "docs/.output/versioned",
            output=tagged,
            version=version,
            **common,
        )
        for name in historical + ["dev"]:
            assert publication["tree_digest"](main / name) == publication["tree_digest"](
                tagged / name
            ), name
        assert publication["tree_digest"](main, publication["GLOBAL"]) == publication[
            "tree_digest"
        ](tagged, publication["GLOBAL"])
        assert tag_result["latest"] == version
        assert (tagged / "latest/sitemap.xml").is_file()
        assert (tagged / "latest/how-to/training-backends/templates/job.yaml").is_file()
        assert not any(path.is_symlink() for path in tagged.rglob("*"))
        result = {
            "archiveCommit": git(ROOT, "rev-parse", archive_ref),
            "sourceCommit": identity["revision"],
            "site": identity["site"],
            "mount": identity["mount"],
            "retainedReleases": hashes,
            "mainLatest": main_result["latest"],
            "simulatedTag": version,
            "tagLatest": tag_result["latest"],
            "globalPreservedByTag": True,
            "archiveInputUnchanged": True,
            "remoteWrites": False,
        }
        report.write_text(json.dumps(result, indent=2) + "\n")
    subprocess.run(["node", "docs/scripts/build.mjs", "all"], cwd=ROOT, check=True)
    print(f"Real-archive main/tag preservation proof passed: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-ref", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    prove(args.archive_ref, args.report)

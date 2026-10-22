"""Publication invariants using real trees and local bare Git repositories only."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("publication", ROOT / "docs/scripts/publication.py")
assert SPEC is not None and SPEC.loader is not None
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def run(repository, *args):
    """Return stdout from a real fixture Git invocation."""
    return (
        subprocess.check_output(["git", "-C", str(repository), *args], stderr=subprocess.PIPE)
        .decode()
        .strip()
    )


def write(root, name, content="fixture", mode=0o644):
    """Return a fixture file with explicit bytes and permissions."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(mode)
    return path


class PublicationTests(unittest.TestCase):
    """Exercise composition and non-forced staging against ordinary local data."""

    def setUp(self):
        Path("/tmp/opencode").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="publication-test-", dir="/tmp/opencode"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        run(self.repo, "init", "-b", "main")
        run(self.repo, "config", "user.name", "Publication test")
        run(self.repo, "config", "user.email", "test@example.invalid")
        self.revisions = []
        for number in range(3):
            write(self.repo, "source.txt", str(number))
            run(self.repo, "add", ".")
            run(self.repo, "commit", "-m", f"Source {number}")
            self.revisions.append(run(self.repo, "rev-parse", "HEAD"))
        self.archive = self.root / "archive"
        self.archive.mkdir()
        for version in ["dev", "v0.1.0", "v0.2.1"]:
            write(self.archive, f"{version}/index.html", version)
            write(self.archive, f"{version}/start/install/index.html", f"install {version}")
            write(self.archive, f"{version}/assets/site.css", "old css")
            write(self.archive, f"{version}/downloads/job.yaml", "old download")
            write(self.archive, f"{version}/executable", "old executable", 0o755)
        (self.archive / "latest").symlink_to("v0.2.1")
        write(
            self.archive,
            "versions.json",
            json.dumps(
                [
                    {"version": "dev", "title": "Development", "aliases": []},
                    {"version": "v0.2.1", "title": "Stable", "aliases": ["latest", "stable"]},
                    {"version": "v0.1.0", "title": "v0.1.0", "aliases": []},
                ]
            ),
        )
        write(self.archive, "index.html", "old entry")
        write(self.archive, ".nojekyll", "")
        write(self.archive, "unknown/subdir/keep.dat", "do not delete")
        write(self.archive, "robots.txt", "retained robots")
        self.candidate = self.root / "candidate"
        self.candidate.mkdir()
        write(self.candidate, "index.html", "new home")
        write(self.candidate, "start/install/index.html", "new install")
        write(self.candidate, "how-to/custom-library/index.html", "new library")
        write(self.candidate, "reference/api/nexuml/core/discovery/index.html", "generated api")
        write(self.candidate, "_astro/site.js", "new asset")
        write(self.candidate, "downloads/job.yaml", "new download")
        self.global_build = self.root / "global"
        self.global_build.mkdir()
        write(self.global_build, "changelog/index.html", "new updates")
        write(self.global_build, "changelog/2026/announcement/index.html", "announcement")
        write(self.global_build, "_astro/global.js", "global asset")
        write(self.global_build, "pagefind/search.js", "global search")
        write(self.global_build, "404.html", "global not found")
        write(self.global_build, "sitemap-index.xml", "global sitemap index")
        write(self.global_build, "sitemap-0.xml", "global sitemap")
        write(self.global_build, "favicon.png", "global favicon")
        write(self.global_build, "build.json", "global build identity")
        self.counter = 0

    def compose(self, version="dev", revision=None, archive=None, changelog=True, **kwargs):
        self.counter += 1
        output = self.root / f"output-{self.counter}"
        revision = revision or self.revisions[-1]
        if version != "dev":
            if not run(self.repo, "tag", "--list", version):
                run(self.repo, "tag", version, revision)
            changelog = False
        report = publication.compose(
            archive=archive or self.archive,
            candidate=self.candidate,
            output=output,
            version=version,
            revision=revision,
            mount=kwargs.pop("mount", "/NexuML/"),
            site=kwargs.pop("site", "https://nexufed.github.io"),
            repository=self.repo,
            changelog=self.global_build if changelog else None,
            **kwargs,
        )
        return output, report

    def test_main_preserves_every_old_byte_mode_and_unknown_path(self):
        write(self.archive, "v0.1.0/private-mode", "preserve all permission bits", 0o640)
        (self.archive / "v0.1.0/assets").chmod(0o750)
        # A release missing from mike metadata is still immutable archive data.
        write(self.archive, "v0.9.0/index.html", "unlisted release")
        archive_before = publication.manifest(self.archive)
        candidate_before = publication.manifest(self.candidate)
        global_before = publication.manifest(self.global_build)
        output, report = self.compose()
        self.assertEqual(report["latest"], "v0.2.1")
        for name in ["v0.1.0", "v0.2.1", "v0.9.0", "unknown"]:
            self.assertEqual(
                publication.manifest(self.archive / name), publication.manifest(output / name)
            )
        self.assertEqual(
            (output / "robots.txt").read_bytes(), (self.archive / "robots.txt").read_bytes()
        )
        self.assertEqual(archive_before, publication.manifest(self.archive))
        self.assertEqual(candidate_before, publication.manifest(self.candidate))
        self.assertEqual(global_before, publication.manifest(self.global_build))
        state = json.loads((output / "publication.json").read_text())
        self.assertEqual(state["schemaVersion"], 1)
        self.assertEqual(state["versions"]["dev"]["revision"], self.revisions[-1])
        self.assertIsNone(state["versions"]["v0.9.0"]["revision"])
        self.assertEqual(state["global"]["revision"], self.revisions[-1])
        self.assertEqual(state["versions"]["dev"]["routes"], publication.routes(output / "dev"))
        self.assertIn("reference/api/nexuml/core/discovery/", state["versions"]["dev"]["routes"])
        versions = json.loads((output / "versions.json").read_text())
        self.assertEqual(versions[0]["version"], "dev")
        self.assertEqual(next(v for v in versions if "latest" in v["aliases"])["version"], "v0.2.1")
        self.assertIn("stable", next(v for v in versions if v["version"] == "v0.2.1")["aliases"])

    def test_latest_is_full_ordinary_copy_including_assets_downloads_and_modes(self):
        os.link(self.candidate / "_astro/site.js", self.candidate / "_astro/hardlinked.js")
        output, _ = self.compose()
        self.assertEqual(
            publication.manifest(output / "latest"), publication.manifest(output / "v0.2.1")
        )
        self.assertTrue((output / "latest/assets/site.css").is_file())
        self.assertTrue((output / "latest/downloads/job.yaml").is_file())
        self.assertEqual(stat.S_IMODE((output / "latest/executable").stat().st_mode), 0o755)
        for path in output.rglob("*"):
            self.assertFalse(path.is_symlink())
            if path.is_file():
                self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual((self.candidate / "_astro/site.js").stat().st_nlink, 2)

    def test_newer_main_advances_dev_global_without_changing_latest(self):
        old, _ = self.compose(revision=self.revisions[0])
        write(self.candidate, "index.html", "newer main docs")
        write(self.global_build, "changelog/index.html", "newer main editorial content")
        output, report = self.compose(archive=old)
        self.assertEqual(report["status"], "composed")
        self.assertEqual((output / "dev/index.html").read_text(), "newer main docs")
        self.assertEqual(
            (output / "changelog/index.html").read_text(), "newer main editorial content"
        )
        for name in ["v0.1.0", "v0.2.1", "latest"]:
            self.assertEqual(publication.manifest(output / name), publication.manifest(old / name))

    def test_tag_preserves_dev_global_and_support_then_older_tag_does_not_roll_back(self):
        main, _ = self.compose()
        newer, _ = self.compose("v0.10.0", archive=main)
        old, report = self.compose("v0.3.0", archive=newer, revision=self.revisions[0])
        self.assertEqual(report["latest"], "v0.10.0")
        self.assertEqual(
            publication.manifest(old / "latest"), publication.manifest(newer / "v0.10.0")
        )
        for name in publication.GLOBAL | {"dev", "v0.1.0", "v0.2.1", "v0.10.0"}:
            self.assertEqual(
                publication.manifest(main / name)
                if name in publication.GLOBAL | {"dev", "v0.1.0", "v0.2.1"}
                else publication.manifest(newer / name),
                publication.manifest(old / name),
            )
        versions = json.loads((old / "versions.json").read_text())
        self.assertEqual(
            [v["version"] for v in versions], ["dev", "v0.10.0", "v0.3.0", "v0.2.1", "v0.1.0"]
        )
        state = json.loads((old / "publication.json").read_text())
        self.assertEqual(state["global"]["revision"], self.revisions[-1])
        self.assertEqual(state["versions"]["v0.3.0"]["revision"], self.revisions[0])

    def test_release_idempotency_requires_both_exact_revision_and_digest(self):
        output, _ = self.compose("v1.0.0")
        retry, report = self.compose("v1.0.0", archive=output)
        self.assertEqual(report["status"], "idempotent")
        self.assertEqual(publication.manifest(output), publication.manifest(retry))
        write(self.candidate, "downloads/job.yaml", "different build for same commit")
        before = publication.manifest(output)
        with self.assertRaisesRegex(ValueError, "conflicting retry"):
            self.compose("v1.0.0", archive=output)
        self.assertEqual(before, publication.manifest(output))
        write(self.candidate, "downloads/job.yaml", "new download")
        run(self.repo, "tag", "--force", "v1.0.0", self.revisions[0])
        with self.assertRaisesRegex(ValueError, "conflicting retry"):
            self.compose("v1.0.0", archive=output, revision=self.revisions[0])
        with self.assertRaisesRegex(ValueError, "conflicting retry"):
            self.compose("v0.2.1")

    def test_main_idempotency_conflicting_build_stale_skip_and_divergence(self):
        output, _ = self.compose()
        retry, report = self.compose(archive=output)
        self.assertEqual(report["status"], "idempotent")
        self.assertEqual(publication.manifest(output), publication.manifest(retry))
        stale, report = self.compose(archive=output, revision=self.revisions[0])
        self.assertEqual(report["status"], "stale")
        self.assertEqual(publication.manifest(output), publication.manifest(stale))
        write(self.global_build, "pagefind/search.js", "conflicting build")
        with self.assertRaisesRegex(ValueError, "Conflicting global"):
            self.compose(archive=output)
        write(self.global_build, "pagefind/search.js", "global search")
        write(self.candidate, "index.html", "conflicting docs build")
        with self.assertRaisesRegex(ValueError, "conflicting retry"):
            self.compose(archive=output)
        run(self.repo, "switch", "-c", "diverged", self.revisions[0])
        write(self.repo, "branch.txt", "divergent")
        run(self.repo, "add", ".")
        run(self.repo, "commit", "-m", "Divergent main fixture")
        divergent = run(self.repo, "rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "Divergent main"):
            self.compose(archive=output, revision=divergent, source_ref="refs/heads/diverged")

    def test_redirects_choose_actual_stable_then_mapped_sections_and_keep_query_hash(self):
        main, _ = self.compose()
        legacy = (main / "start/install/index.html").read_text()
        self.assertIn('href="/NexuML/v0.2.1/start/install/"', legacy)
        self.assertNotIn("/dev/", legacy)
        merged = (main / "dev/learn/decorators-and-discovery/index.html").read_text()
        self.assertIn("/NexuML/dev/how-to/custom-library/#discovery", merged)
        self.assertIn('rel="canonical" href="https://nexufed.github.io/', merged)
        self.assertIn('http-equiv="refresh"', merged)
        self.assertIn("Continue to documentation</a>", merged)
        self.assertIn("target.search = window.location.search", merged)
        self.assertIn("target.hash = window.location.hash || target.hash", merged)
        self.assertNotIn("301", merged)
        # A page absent in the selected old release must fall back to a real route.
        self.assertIn(
            'href="/NexuML/v0.2.1/"', (main / "learn/mental-model/index.html").read_text()
        )
        tagged, _ = self.compose("v1.0.0", archive=main)
        api = tagged / "reference/api/nexuml/core/discovery/index.html"
        self.assertIn('href="/NexuML/v1.0.0/reference/api/nexuml/core/discovery/"', api.read_text())
        self.assertEqual((tagged / "changelog/index.html").read_text(), "new updates")
        self.assertEqual((tagged / "404.html").read_text(), "global not found")
        self.assertFalse((tagged / "api").exists())
        frozen_before = publication.manifest(tagged / "v1.0.0")
        later, _ = self.compose(archive=tagged)
        self.assertEqual(frozen_before, publication.manifest(later / "v1.0.0"))

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute redirect JavaScript")
    def test_redirect_javascript_really_preserves_query_fragment_and_mapped_default(self):
        page = publication.redirect(
            "/NexuML/v1.0.0/how-to/custom-library/#discovery", "https://nexufed.github.io"
        )
        script = page.split("<script>", 1)[1].split("</script>", 1)[0]
        for incoming_hash, expected_hash in [
            ("#original-heading", "#original-heading"),
            ("", "#discovery"),
        ]:
            with self.subTest(incoming_hash=incoming_hash):
                result = subprocess.check_output(
                    [
                        "node",
                        "--input-type=module",
                        "-e",
                        'import vm from "node:vm";'
                        'const location = {href: "https://nexufed.github.io/NexuML/old/",'
                        'search: "?q=hello%20world&x=1", hash: process.argv[1],'
                        "replace: value => console.log(value)};"
                        "vm.runInNewContext(process.argv[2], {URL, window: {location}});",
                        incoming_hash,
                        script,
                    ],
                    text=True,
                ).strip()
                self.assertEqual(
                    result,
                    "https://nexufed.github.io/NexuML/v1.0.0/"
                    "how-to/custom-library/?q=hello%20world&x=1" + expected_hash,
                )

    def test_root_mount_and_alternate_origin(self):
        output, _ = self.compose("v1.0.0", mount="/", site="https://docs.example.invalid/")
        html = (output / "index.html").read_text()
        self.assertIn('href="/v1.0.0/"', html)
        self.assertIn('href="https://docs.example.invalid/v1.0.0/"', html)
        self.assertNotIn("NexuML", html)
        self.assertNotIn("nexufed.github.io", html)

    def test_tree_digest_detects_paths_types_modes_and_bytes_not_timestamps(self):
        tree = self.root / "digest"
        tree.mkdir()
        file = write(tree, "file", "bytes")
        original = publication.tree_digest(tree)
        os.utime(file, (1, 1))
        self.assertEqual(original, publication.tree_digest(tree))
        file.chmod(0o755)
        self.assertNotEqual(original, publication.tree_digest(tree))
        file.chmod(0o644)
        file.rename(tree / "renamed")
        self.assertNotEqual(original, publication.tree_digest(tree))
        (tree / "renamed").rename(file)
        file.write_text("other bytes")
        self.assertNotEqual(original, publication.tree_digest(tree))
        file.unlink()
        file.mkdir()
        self.assertNotEqual(original, publication.tree_digest(tree))

    def test_archive_tampering_and_missing_manifest_fail_closed(self):
        output, _ = self.compose()
        write(output, "v0.1.0/index.html", "tampered")
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            self.compose(archive=output)
        (self.archive / "versions.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.compose()

    def test_unsafe_inputs_and_overlapping_paths_are_rejected_before_writes(self):
        args = dict(
            archive=self.archive,
            candidate=self.candidate,
            output=self.root / "unsafe-output",
            version="dev",
            revision=self.revisions[-1],
            mount="/NexuML/",
            site="https://nexufed.github.io",
            repository=self.repo,
        )
        cases = [
            {"output": self.archive / "nested"},
            {"output": self.candidate},
            {"output": self.root},
            {"output": Path("/")},
            {"output": self.repo / ".git/publication"},
            {"output": self.root / "absent-parent/output"},
            {"archive": self.root / "missing"},
            {"candidate": self.archive},
            {"version": "v1.0.0-rc1"},
            {"version": "v01.0.0"},
            {"revision": "bad-sha"},
            {"mount": "/NexuML/../"},
            {"site": "https://host.invalid/path/"},
            {"site": "https://user:password@host.invalid"},
            {"source_ref": "refs/heads/missing"},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises((ValueError, OSError)):
                publication.compose(**(args | changes))
        link = self.root / "link"
        link.symlink_to(self.candidate)
        with self.assertRaisesRegex(ValueError, "Symlink path component"):
            publication.compose(**(args | {"candidate": link}))
        write(self.candidate, "target", "target")
        (self.candidate / "escape").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "Unsupported symlink"):
            publication.compose(**args)
        (self.candidate / "escape").unlink()
        (self.archive / "unknown/link").symlink_to("../v0.1.0")
        with self.assertRaisesRegex(ValueError, "Unsupported symlink"):
            publication.compose(**args)
        self.assertFalse((self.root / "unsafe-output").exists())

    def test_tag_source_mismatch_and_global_ownership_rejected(self):
        run(self.repo, "tag", "v1.0.0", self.revisions[0])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.compose("v1.0.0")
        args = dict(
            archive=self.archive,
            candidate=self.candidate,
            output=self.root / "refused",
            version="v1.0.0",
            revision=self.revisions[0],
            mount="/NexuML/",
            site="https://nexufed.github.io",
            repository=self.repo,
            changelog=self.global_build,
        )
        with self.assertRaisesRegex(ValueError, "Only main"):
            publication.compose(**args)
        write(self.global_build, "foreign-root/index.html", "unowned global path")
        with self.assertRaisesRegex(ValueError, "outside its owned"):
            self.compose()

    def stage_fixture(self):
        remote = self.root / "pages.git"
        run(self.root, "init", "--bare", str(remote))
        seed = self.root / "seed"
        shutil.copytree(self.archive, seed, symlinks=True)
        run(seed, "init", "-b", "gh-pages")
        run(seed, "config", "user.name", "Fixture")
        run(seed, "config", "user.email", "fixture@example.invalid")
        run(seed, "add", ".")
        run(seed, "commit", "-m", "Original archive")
        run(seed, "push", str(remote), "HEAD:refs/heads/gh-pages")
        return remote

    def stage(self, remote, version="dev", revision=None, **kwargs):
        revision = revision or self.revisions[-1]
        if version != "dev" and not run(self.repo, "tag", "--list", version):
            run(self.repo, "tag", version, revision)
        return publication.stage(
            repository=self.repo,
            remote=str(remote),
            candidate=self.candidate,
            changelog=self.global_build if version == "dev" else None,
            version=version,
            revision=revision,
            mount="/NexuML/",
            site="https://nexufed.github.io",
            **kwargs,
        )

    def extracted(self, remote):
        destination = self.root / "extracted"
        publication.extract_archive(remote, "refs/heads/gh-pages", destination)
        return destination

    def test_competing_staged_tags_refetch_recompose_and_both_survive(self):
        remote = self.stage_fixture()
        original = run(remote, "rev-parse", "refs/heads/gh-pages")
        run(self.repo, "tag", "v1.1.0", self.revisions[0])
        run(self.repo, "tag", "v1.2.0", self.revisions[-1])
        original_source = publication.manifest(self.repo)
        actual_git = publication.git
        competing = []

        def interleave(repository, *args, **kwargs):
            if args[0] == "push" and not competing:
                competing.append(True)
                # Real nested publication wins the remote before the outer real push.
                competing.append(self.stage(remote, "v1.1.0", revision=self.revisions[0]))
            return actual_git(repository, *args, **kwargs)

        with patch.object(publication, "git", side_effect=interleave):
            report = self.stage(remote, "v1.2.0")
        self.assertEqual(report["attempts"], 2)
        self.assertTrue(competing[1]["staged"])
        self.assertEqual(original_source, publication.manifest(self.repo))
        tree = self.extracted(remote)
        self.assertTrue((tree / "v1.1.0/index.html").is_file())
        self.assertTrue((tree / "v1.2.0/index.html").is_file())
        self.assertEqual(
            publication.manifest(tree / "latest"), publication.manifest(tree / "v1.2.0")
        )
        self.assertEqual((tree / "dev/index.html").read_text(), "dev")
        self.assertEqual((tree / "unknown/subdir/keep.dat").read_text(), "do not delete")
        self.assertTrue(publication.is_ancestor(remote, original, report["archiveCommit"]))
        retry = self.stage(remote, "v1.2.0")
        self.assertFalse(retry["staged"])
        self.assertEqual(retry["archiveCommit"], report["archiveCommit"])

    def test_stale_staged_main_cannot_overwrite_competing_new_main(self):
        remote = self.stage_fixture()
        actual_git = publication.git
        competing = []

        def interleave(repository, *args, **kwargs):
            if args[0] == "push" and not competing:
                competing.append(True)
                competing.append(self.stage(remote))
            return actual_git(repository, *args, **kwargs)

        with patch.object(publication, "git", side_effect=interleave):
            report = self.stage(remote, revision=self.revisions[0])
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(report["status"], "stale")
        self.assertFalse(report["staged"])
        self.assertEqual(report["archiveCommit"], competing[1]["archiveCommit"])
        tree = self.extracted(remote)
        state = json.loads((tree / "publication.json").read_text())
        self.assertEqual(state["versions"]["dev"]["revision"], self.revisions[-1])
        self.assertEqual(state["global"]["revision"], self.revisions[-1])
        self.assertEqual((tree / "changelog/index.html").read_text(), "new updates")

    def test_competing_main_and_tag_preserve_both_owned_outputs(self):
        remote = self.stage_fixture()
        actual_git = publication.git
        competing = []

        def interleave(repository, *args, **kwargs):
            if args[0] == "push" and not competing:
                competing.append(True)
                competing.append(self.stage(remote, "v1.0.0"))
            return actual_git(repository, *args, **kwargs)

        with patch.object(publication, "git", side_effect=interleave):
            report = self.stage(remote)
        self.assertEqual(report["attempts"], 2)
        tree = self.extracted(remote)
        self.assertEqual((tree / "changelog/index.html").read_text(), "new updates")
        self.assertEqual((tree / "v1.0.0/index.html").read_text(), "new home")
        self.assertEqual((tree / "dev/index.html").read_text(), "new home")
        self.assertEqual(
            publication.manifest(tree / "latest"), publication.manifest(tree / "v1.0.0")
        )

    def test_staging_refuses_missing_archive_and_push_denial_without_force(self):
        remote = self.root / "empty.git"
        run(self.root, "init", "--bare", str(remote))
        with self.assertRaisesRegex(ValueError, "git fetch"):
            self.stage(remote)
        remote = self.stage_fixture()
        write(remote, "hooks/pre-receive", "#!/bin/sh\nexit 1\n", 0o755)
        before = run(remote, "rev-parse", "refs/heads/gh-pages")
        with self.assertRaisesRegex(ValueError, "without a competing"):
            self.stage(remote)
        self.assertEqual(before, run(remote, "rev-parse", "refs/heads/gh-pages"))

    def test_retry_limit_keeps_winning_release_when_losing_candidate_exhausts(self):
        remote = self.stage_fixture()
        actual_git = publication.git
        competing = []

        def interleave(repository, *args, **kwargs):
            if args[0] == "push" and not competing:
                competing.append(True)
                competing.append(self.stage(remote, "v1.0.0"))
            return actual_git(repository, *args, **kwargs)

        with patch.object(publication, "git", side_effect=interleave):
            with self.assertRaisesRegex(ValueError, "after 1 attempts"):
                self.stage(remote, "v2.0.0", attempts=1)
        tree = self.extracted(remote)
        self.assertTrue((tree / "v1.0.0/index.html").is_file())
        self.assertFalse((tree / "v2.0.0").exists())
        self.assertEqual(
            run(remote, "rev-parse", "refs/heads/gh-pages"), competing[1]["archiveCommit"]
        )

    def test_cli_composition_records_exact_revision(self):
        output = self.root / "cli-output"
        result = subprocess.run(
            [
                "python3",
                str(ROOT / "docs/scripts/publication.py"),
                "--archive",
                str(self.archive),
                "--candidate",
                str(self.candidate),
                "--changelog",
                str(self.global_build),
                "--output",
                str(output),
                "--version",
                "dev",
                "--revision",
                self.revisions[-1],
                "--mount",
                "/NexuML/",
                "--site",
                "https://nexufed.github.io",
                "--repository",
                str(self.repo),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["revision"], self.revisions[-1])
        self.assertTrue((output / "publication.json").is_file())


if __name__ == "__main__":
    unittest.main()

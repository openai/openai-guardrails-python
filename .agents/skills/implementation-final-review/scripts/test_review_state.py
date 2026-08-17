#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from review_state import _component, _content_fingerprint, _load_pathspec_file, review_state


class ReviewStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self._git("init", "-q")
        self._git("config", "user.email", "review-state@example.test")
        self._git("config", "user.name", "Review State Test")
        (self.repo / ".gitignore").write_text("plans/private.md\n")
        (self.repo / "src").mkdir()
        (self.repo / "tests").mkdir()
        (self.repo / "plans").mkdir()
        (self.repo / "src" / "runtime.py").write_text("VALUE = 1\n")
        (self.repo / "tests" / "test_runtime.py").write_text("assert True\n")
        self._git("add", ".")
        self._git("commit", "-qm", "initial")
        self.base = self._git("rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _git(self, *args: str) -> str:
        return subprocess.check_output(("git", "-C", str(self.repo), *args), text=True)

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(Path(__file__).with_name("review_state.py")),
                "--repo",
                str(self.repo),
                "--base",
                self.base,
                *args,
            ),
            capture_output=True,
            text=True,
        )

    def test_equivalent_pathspecs_have_the_same_content_fingerprint(self) -> None:
        (self.repo / "src" / "runtime.py").write_text("VALUE = 2\n")
        explicit = review_state(self.repo, self.base, ("src/runtime.py",))
        directory = review_state(self.repo, self.base, ("src",))
        with_ignored_artifact = review_state(
            self.repo, self.base, ("src/runtime.py", "plans/private.md")
        )

        self.assertEqual(explicit["content_fingerprint"], directory["content_fingerprint"])
        self.assertEqual(
            explicit["content_fingerprint"], with_ignored_artifact["content_fingerprint"]
        )

    def test_component_fingerprints_invalidate_only_changed_content(self) -> None:
        runtime = self.repo / "src" / "runtime.py"
        tests = self.repo / "tests" / "test_runtime.py"
        runtime.write_text("VALUE = 2\n")
        tests.write_text("assert 2 == 2\n")
        components = {"runtime": ("src",), "tests-examples": ("tests",)}
        before = review_state(self.repo, self.base, ("src", "tests"), components)

        tests.write_text("assert 2 != 1\n")
        after = review_state(self.repo, self.base, ("src", "tests"), components)

        self.assertEqual(
            before["components"]["runtime"]["content_fingerprint"],
            after["components"]["runtime"]["content_fingerprint"],
        )
        self.assertNotEqual(
            before["components"]["tests-examples"]["content_fingerprint"],
            after["components"]["tests-examples"]["content_fingerprint"],
        )
        self.assertNotEqual(before["content_fingerprint"], after["content_fingerprint"])

    def test_unfiltered_workspace_accounts_for_changes_outside_manifest(self) -> None:
        (self.repo / "src" / "runtime.py").write_text("VALUE = 2\n")
        (self.repo / "tests" / "test_runtime.py").write_text("assert 2 == 2\n")

        state = review_state(self.repo, self.base, ("src",))

        self.assertEqual([entry["path"] for entry in state["workspace"]], ["src/runtime.py"])
        self.assertEqual(
            [entry["path"] for entry in state["unfiltered"]["workspace"]],
            ["src/runtime.py", "tests/test_runtime.py"],
        )
        self.assertRegex(state["unfiltered"]["status_sha256"], r"^[0-9a-f]{64}$")

    def test_complete_diff_includes_task_owned_untracked_files(self) -> None:
        new_test = self.repo / "tests" / "test_new.py"
        new_test.write_text("assert 2 == 2\n")
        complete_diff = self.root / "complete.diff"

        state = review_state(
            self.repo,
            self.base,
            ("tests",),
            complete_diff_output=complete_diff,
        )

        diff = complete_diff.read_bytes()
        self.assertIn(b"diff --git a/tests/test_new.py b/tests/test_new.py", diff)
        self.assertIn(b"+assert 2 == 2", diff)
        self.assertEqual(state["complete_diff_sha256"], hashlib.sha256(diff).hexdigest())
        self.assertEqual(
            state["complete_diff_paths"],
            ["tests/test_new.py"],
        )
        self.assertNotEqual(state["complete_diff_sha256"], state["tracked_diff_sha256"])

    def test_exact_manifest_path_includes_ignored_untracked_file(self) -> None:
        ignored = self.repo / "plans" / "private.md"
        ignored.write_text("shipped fixture\n")
        complete_diff = self.root / "complete.diff"

        state = review_state(
            self.repo,
            self.base,
            ("plans/private.md",),
            {"release-metadata": ("plans/private.md",)},
            complete_diff_output=complete_diff,
        )

        self.assertEqual(state["complete_diff_paths"], ["plans/private.md"])
        self.assertEqual(state["unfiltered"]["workspace"], state["workspace"])
        self.assertEqual(
            state["components"]["release-metadata"]["workspace"],
            state["workspace"],
        )
        self.assertIn(b"+shipped fixture", complete_diff.read_bytes())

    def test_directory_pathspec_does_not_promote_ignored_operational_files(self) -> None:
        (self.repo / "plans" / "private.md").write_text("operational plan\n")

        state = review_state(self.repo, self.base, ("plans",))

        self.assertEqual(state["workspace"], [])
        self.assertEqual(state["complete_diff_paths"], [])

    def test_literal_filename_with_pathspec_metacharacters_is_exact(self) -> None:
        (self.repo / "plans" / "[a].md").write_text("literal\n")
        (self.repo / "plans" / "a.md").write_text("glob match\n")

        state = review_state(self.repo, self.base, ("plans/[a].md",))

        self.assertEqual(state["complete_diff_paths"], ["plans/[a].md"])

    def test_existing_magic_prefixed_filename_is_exact(self) -> None:
        """Treat an existing magic-prefixed filename as an exact path."""
        (self.repo / ":(glob)literal").write_text("literal filename\n")
        (self.repo / "literal").write_text("glob match\n")

        state = review_state(self.repo, self.base, (":(glob)literal",))

        self.assertEqual(state["complete_diff_paths"], [":(glob)literal"])

    def test_deleted_magic_prefixed_filename_is_exact_from_base(self) -> None:
        """Treat a deleted base filename with magic syntax as exact."""
        magic_prefixed = self.repo / ":(glob)literal"
        magic_prefixed.write_text("deleted literal filename\n")
        (self.repo / "literal").write_text("glob match\n")
        self._git("add", ".")
        self._git("commit", "-qm", "add magic-prefixed filename")
        self.base = self._git("rev-parse", "HEAD").strip()
        self._git("rm", "-q", "--", ":(literal):(glob)literal")

        state = review_state(self.repo, self.base, (":(glob)literal",))

        self.assertEqual(state["complete_diff_paths"], [":(glob)literal"])

    def test_explicit_glob_magic_preserves_pattern_semantics(self) -> None:
        (self.repo / "plans" / "[a].md").write_text("literal\n")
        (self.repo / "plans" / "a.md").write_text("glob match\n")

        state = review_state(self.repo, self.base, (":(glob)plans/[a].md",))

        self.assertEqual(state["complete_diff_paths"], ["plans/[a].md", "plans/a.md"])

    def test_dirty_submodule_content_changes_fingerprint(self) -> None:
        """Fingerprint dirty submodule content, not only its status."""
        source = self.repo / ".fixtures" / "dependency-source"
        source.mkdir(parents=True)
        subprocess.run(("git", "init", "-q", str(source)), check=True)
        subprocess.run(
            ("git", "-C", str(source), "config", "user.email", "submodule@example.test"),
            check=True,
        )
        subprocess.run(
            ("git", "-C", str(source), "config", "user.name", "Submodule Test"),
            check=True,
        )
        (source / "tracked.txt").write_text("committed\n")
        subprocess.run(("git", "-C", str(source), "add", "."), check=True)
        subprocess.run(("git", "-C", str(source), "commit", "-qm", "initial"), check=True)
        with (self.repo / ".gitignore").open("a") as gitignore:
            gitignore.write(".fixtures/\n")
        self._git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(source),
            "vendor/dependency",
        )
        self._git("add", ".")
        self._git("commit", "-qm", "add dependency")
        self.base = self._git("rev-parse", "HEAD").strip()
        tracked = self.repo / "vendor" / "dependency" / "tracked.txt"
        tracked.write_text("first dirty body\n")
        before = review_state(self.repo, self.base, ("vendor/dependency",))

        tracked.write_text("second dirty body\n")
        after = review_state(self.repo, self.base, ("vendor/dependency",))

        self.assertEqual(
            before["workspace"][0]["status_sha256"],
            after["workspace"][0]["status_sha256"],
        )
        self.assertEqual(before["complete_diff_sha256"], after["complete_diff_sha256"])
        self.assertNotEqual(
            before["workspace"][0]["worktree_sha256"],
            after["workspace"][0]["worktree_sha256"],
        )
        self.assertNotEqual(before["content_fingerprint"], after["content_fingerprint"])
        self.assertNotEqual(before["repository_fingerprint"], after["repository_fingerprint"])

    def test_dirty_nested_submodule_content_changes_fingerprint(self) -> None:
        """Recursively fingerprint dirty nested submodule content."""
        leaf_source = self.repo / ".fixtures" / "leaf-source"
        leaf_source.mkdir(parents=True)
        subprocess.run(("git", "init", "-q", str(leaf_source)), check=True)
        subprocess.run(
            ("git", "-C", str(leaf_source), "config", "user.email", "leaf@example.test"),
            check=True,
        )
        subprocess.run(
            ("git", "-C", str(leaf_source), "config", "user.name", "Leaf Test"),
            check=True,
        )
        (leaf_source / "tracked.txt").write_text("committed\n")
        subprocess.run(("git", "-C", str(leaf_source), "add", "."), check=True)
        subprocess.run(
            ("git", "-C", str(leaf_source), "commit", "-qm", "initial"),
            check=True,
        )

        parent_source = self.repo / ".fixtures" / "parent-source"
        parent_source.mkdir()
        subprocess.run(("git", "init", "-q", str(parent_source)), check=True)
        subprocess.run(
            ("git", "-C", str(parent_source), "config", "user.email", "parent@example.test"),
            check=True,
        )
        subprocess.run(
            ("git", "-C", str(parent_source), "config", "user.name", "Parent Test"),
            check=True,
        )
        subprocess.run(
            (
                "git",
                "-C",
                str(parent_source),
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "-q",
                str(leaf_source),
                "nested",
            ),
            check=True,
        )
        subprocess.run(("git", "-C", str(parent_source), "commit", "-qam", "initial"), check=True)

        with (self.repo / ".gitignore").open("a") as gitignore:
            gitignore.write(".fixtures/\n")
        self._git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(parent_source),
            "vendor/dependency",
        )
        self._git(
            "-C",
            "vendor/dependency",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "-q",
        )
        self._git("add", ".")
        self._git("commit", "-qm", "add nested dependency")
        self.base = self._git("rev-parse", "HEAD").strip()
        tracked = self.repo / "vendor" / "dependency" / "nested" / "tracked.txt"
        tracked.write_text("first dirty body\n")
        before = review_state(self.repo, self.base, ("vendor/dependency",))

        tracked.write_text("second dirty body\n")
        after = review_state(self.repo, self.base, ("vendor/dependency",))

        self.assertEqual(
            before["workspace"][0]["status_sha256"],
            after["workspace"][0]["status_sha256"],
        )
        self.assertNotEqual(
            before["workspace"][0]["worktree_sha256"],
            after["workspace"][0]["worktree_sha256"],
        )
        self.assertNotEqual(before["content_fingerprint"], after["content_fingerprint"])
        self.assertNotEqual(before["repository_fingerprint"], after["repository_fingerprint"])

    @unittest.skipIf(os.name == "nt", "Non-UTF-8 filenames require POSIX filesystem bytes.")
    def test_non_utf8_filename_has_stable_fingerprint(self) -> None:
        """Preserve surrogateescaped Git path bytes in review artifacts."""
        raw_relative_path = b"tests/non-utf8-\xff.py"
        git = (b"git", b"-C", os.fsencode(self.repo))
        blob = subprocess.check_output(
            (*git, b"hash-object", b"-w", b"--stdin"),
            input=b"assert True\n",
        ).strip()
        subprocess.run(
            (
                *git,
                b"update-index",
                b"--add",
                b"--cacheinfo",
                b"100644," + blob + b"," + raw_relative_path,
            ),
            check=True,
        )
        self._git("commit", "-qm", "add non-UTF-8 filename")
        self.base = self._git("rev-parse", "HEAD").strip()
        subprocess.run(
            (*git, b"update-index", b"--force-remove", b"--", raw_relative_path),
            check=True,
        )
        relative_path = os.fsdecode(raw_relative_path)

        state = review_state(self.repo, self.base, ("tests",))

        self.assertEqual(state["complete_diff_paths"], [relative_path])
        self.assertEqual(
            _content_fingerprint(state["base"], state["workspace"]),
            state["content_fingerprint"],
        )
        json.dumps(state, ensure_ascii=True)

        completed = self._run_cli("--pathspec", "tests")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        cli_state = json.loads(completed.stdout)
        self.assertEqual(cli_state["complete_diff_paths"], [relative_path])
        self.assertEqual(cli_state["content_fingerprint"], state["content_fingerprint"])

    def test_complete_diff_output_must_be_outside_repository(self) -> None:
        """Reject an operational diff artifact inside the worktree."""
        complete_diff = self.repo / "complete.diff"

        with self.assertRaisesRegex(ValueError, "outside the repository"):
            review_state(
                self.repo,
                self.base,
                complete_diff_output=complete_diff,
            )

        self.assertFalse(complete_diff.exists())

    def test_cli_writes_complete_diff_output(self) -> None:
        (self.repo / "tests" / "test_new.py").write_text("assert True\n")
        complete_diff = self.root / "complete.diff"

        completed = self._run_cli(
            "--pathspec",
            "tests",
            "--complete-diff-output",
            str(complete_diff),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        state = json.loads(completed.stdout)
        self.assertEqual(
            state["complete_diff_sha256"],
            hashlib.sha256(complete_diff.read_bytes()).hexdigest(),
        )

    def test_repository_fingerprint_includes_outside_manifest_state_and_content(self) -> None:
        (self.repo / "src" / "runtime.py").write_text("VALUE = 2\n")
        before = review_state(self.repo, self.base, ("src",))

        outside = self.repo / "outside.txt"
        outside.write_text("first\n")
        after_add = review_state(self.repo, self.base, ("src",))
        outside.write_text("second\n")
        after_content = review_state(self.repo, self.base, ("src",))

        self.assertEqual(before["content_fingerprint"], after_add["content_fingerprint"])
        self.assertEqual(after_add["content_fingerprint"], after_content["content_fingerprint"])
        self.assertNotEqual(before["repository_fingerprint"], after_add["repository_fingerprint"])
        self.assertNotEqual(
            after_add["repository_fingerprint"], after_content["repository_fingerprint"]
        )

    def test_pathspec_file_preserves_literal_values_and_deduplicates(self) -> None:
        manifest = self.repo / "paths.txt"
        manifest.write_text("src\n\n#literal\n lead.py\nsrc\n")

        self.assertEqual(_load_pathspec_file(manifest), ("src", "#literal", " lead.py"))

    def test_direct_pathspec_preserves_leading_space(self) -> None:
        (self.repo / " lead.py").write_text("VALUE = 2\n")

        completed = self._run_cli("--pathspec", " lead.py")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        state = json.loads(completed.stdout)
        self.assertEqual([entry["path"] for entry in state["workspace"]], [" lead.py"])

    def test_empty_direct_pathspec_fails_closed(self) -> None:
        completed = self._run_cli("--pathspec", "")

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Pathspecs must not be empty", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_invalid_manifest_files_are_parser_errors(self) -> None:
        cases = (
            ("--pathspec-file", str(self.repo / "missing.paths")),
            ("--pathspec-file", str(self.repo)),
            ("--component-pathspec-file", "runtime="),
            ("--component-pathspec-file", f"runtime={self.repo / 'missing.paths'}"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                completed = self._run_cli(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("error:", completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)

    def test_invalid_repository_is_a_parser_error(self) -> None:
        missing_repo = self.repo / "missing-repo"
        completed = subprocess.run(
            (
                sys.executable,
                str(Path(__file__).with_name("review_state.py")),
                "--repo",
                str(missing_repo),
                "--base",
                self.base,
            ),
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Git command failed", completed.stderr)
        self.assertNotIn("fatal:", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_invalid_base_is_a_parser_error(self) -> None:
        completed = self._run_cli("--base", "missing-revision")

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Git command failed", completed.stderr)
        self.assertNotIn("fatal:", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_non_ancestor_base_is_a_parser_error(self) -> None:
        self._git("checkout", "-qb", "sibling")
        (self.repo / "src" / "runtime.py").write_text("VALUE = 2\n")
        self._git("commit", "-qam", "sibling change")
        sibling = self._git("rev-parse", "HEAD").strip()
        self._git("checkout", "-qb", "current", self.base)
        (self.repo / "tests" / "test_runtime.py").write_text("assert 2 == 2\n")
        self._git("commit", "-qam", "head change")

        completed = self._run_cli("--base", sibling)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Base must be an ancestor of HEAD", completed.stderr)
        self.assertNotIn("fatal:", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_component_manifests_must_cover_combined_content(self) -> None:
        (self.repo / "src" / "runtime.py").write_text("VALUE = 2\n")
        (self.repo / "tests" / "test_runtime.py").write_text("assert 2 == 2\n")

        with self.assertRaisesRegex(ValueError, "missing=.*test_runtime.py"):
            review_state(self.repo, self.base, ("src", "tests"), {"runtime": ("src",)})

    def test_component_manifests_must_not_overlap(self) -> None:
        (self.repo / "src" / "runtime.py").write_text("VALUE = 2\n")

        with self.assertRaisesRegex(ValueError, "overlapping=.*runtime.py"):
            review_state(
                self.repo,
                self.base,
                ("src",),
                {"runtime": ("src",), "tests-examples": ("src/runtime.py",)},
            )

    def test_components_define_combined_scope_when_pathspecs_are_omitted(self) -> None:
        (self.repo / "src" / "runtime.py").write_text("VALUE = 2\n")
        state = review_state(self.repo, self.base, components={"runtime": ("src",)})

        self.assertEqual(state["pathspecs"], ["src"])
        self.assertEqual([entry["path"] for entry in state["workspace"]], ["src/runtime.py"])

    def test_component_cli_value(self) -> None:
        self.assertEqual(_component("runtime=src"), ("runtime", "src"))


if __name__ == "__main__":
    unittest.main()

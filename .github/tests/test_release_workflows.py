"""Check the release credential and event boundaries without calling GitHub.

Run with: uv run python -m unittest discover -s .github/tests -v
PyYAML is supplied by the repository's MkDocs development dependency.
"""

import re
import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def load_workflow(name: str) -> dict:
    """Read workflow scalars as strings, preserving the YAML 1.2 `on` key."""
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


class ReleaseWorkflowTests(unittest.TestCase):
    """Protect the App migration and its existing publication boundary."""

    def test_release_credentials_are_main_only_and_repository_scoped(self) -> None:
        """Only the canonical main job receives a least-privilege App token."""
        workflow = load_workflow("create-releases.yml")
        self.assertEqual(workflow["on"], {"push": {"branches": ["main"]}})
        self.assertEqual(workflow["permissions"], {})
        release = workflow["jobs"]["release"]
        self.assertEqual(release.get("permissions", {}), {})
        self.assertEqual(release["environment"], "release")
        self.assertEqual(release["if"], "github.ref == 'refs/heads/main' && github.repository == 'openai/openai-guardrails-python'")
        token = release["steps"][0]
        self.assertRegex(token["uses"], r"^actions/create-github-app-token@[0-9a-f]{40}$")
        self.assertEqual(
            token["with"],
            {
                "client-id": "${{ vars.OPENAI_SDKS_APP_CLIENT_ID }}",
                "private-key": "${{ secrets.OPENAI_SDKS_APP_PRIVATE_KEY }}",
                "owner": "${{ github.repository_owner }}",
                "repositories": "${{ github.event.repository.name }}",
                "permission-contents": "write",
                "permission-issues": "write",
                "permission-pull-requests": "write",
            },
        )

    def test_release_uses_app_events_without_a_second_publish_dispatch(self) -> None:
        """App-authored PRs and releases use the existing CI and publish events."""
        steps = load_workflow("create-releases.yml")["jobs"]["release"]["steps"]
        self.assertEqual(len(steps), 2, "Keep publishing out of the release job: the release event owns the handoff.")
        self.assertTrue(steps[1]["uses"].startswith("googleapis/release-please-action@"))
        self.assertEqual(steps[1]["with"]["token"], "${{ steps." + steps[0]["id"] + ".outputs.token }}")
        self.assertEqual(steps[1]["with"]["target-branch"], "main")
        self.assertIn("pull_request", load_workflow("ci.yml")["on"])
        self.assertIn("merge_group", load_workflow("ci.yml")["on"])
        self.assertEqual(load_workflow("publish.yml")["on"], {"workflow_dispatch": "", "release": {"types": ["published"]}})

    def test_publication_keeps_release_and_main_ancestry_guards(self) -> None:
        """Both automatic publishing and manual recovery require a release tag."""
        build = load_workflow("publish.yml")["jobs"]["build"]
        self.assertEqual(
            build["if"],
            "github.repository == 'openai/openai-guardrails-python' && github.ref_type == 'tag' && startsWith(github.ref, 'refs/tags/v')",
        )
        steps = build["steps"]
        validation = steps[0]["with"]["script"]
        self.assertIn("github.rest.repos.getReleaseByTag", validation)
        self.assertIn("release.draft || !release.published_at", validation)
        self.assertIn("throw new Error", validation)
        self.assertEqual(steps[1]["with"], {"persist-credentials": "false", "fetch-depth": "0"})
        self.assertEqual(steps[2]["run"], "git merge-base --is-ancestor HEAD refs/remotes/origin/main")

    def test_build_and_trusted_publishing_credentials_stay_separate(self) -> None:
        """Package code cannot receive the App key or publishing credentials."""
        workflow = load_workflow("publish.yml")
        self.assertEqual(workflow["permissions"], {})
        build, publish = workflow["jobs"]["build"], workflow["jobs"]["publish"]
        self.assertEqual(build["permissions"], {"contents": "read"})
        self.assertNotIn("environment", build)
        self.assertEqual(publish["needs"], "build")
        self.assertEqual(publish["environment"]["name"], "pypi")
        self.assertEqual(publish["permissions"], {"id-token": "write"})
        self.assertEqual(len(publish["steps"]), 2)
        self.assertTrue(publish["steps"][0]["uses"].startswith("actions/download-artifact@"))
        self.assertTrue(publish["steps"][1]["uses"].startswith("pypa/gh-action-pypi-publish@"))
        self.assertEqual(publish["steps"][1]["with"], {"attestations": "true"})
        self.assertIsNone(re.search(r"secrets\.|OPENAI_SDKS_APP|create-github-app-token", (WORKFLOWS / "publish.yml").read_text()))

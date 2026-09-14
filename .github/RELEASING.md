# Releasing

Releases use release-please with a short-lived installation token from the
`openai-sdks` GitHub App. App-created release PRs trigger normal CI, and published
GitHub Releases trigger the existing PyPI publishing workflow.

## One-time setup

The App must be installed for this repository. The existing `release` environment
holds the `OPENAI_SDKS_APP_CLIENT_ID` variable and `OPENAI_SDKS_APP_PRIVATE_KEY`
secret. Keep it restricted to the exact `main` branch with admin bypass disabled;
preserve configured environment approvals. Verify secret presence by metadata
only; never print or copy its value.

The token action is pinned to a full commit SHA and requests only this repository
with Contents, Issues, and Pull requests write permissions. It revokes the token
at job completion; installation tokens also expire after one hour. The release
job's `GITHUB_TOKEN` has no granted permissions. Keep required PR reviews, checks,
and the merge queue in place; the App does not need a default-branch bypass.

Keep the existing PyPI trusted publisher for `openai/openai-guardrails-python`,
workflow `publish.yml`, environment `pypi`. Restrict `pypi` deployments to `v*`
tags and require SDK-team approval. Protect release tags against unauthorized
creation, updates, and deletion while allowing the release workflow to create
new tags. These controls live in GitHub settings, not workflow YAML.

## Routine releases

1. Use Conventional Commit titles when squash-merging PRs into `main`: `fix:`
   produces a patch bump, `feat:` a minor bump, and breaking changes a minor bump
   while the package is pre-1.0. Release notes use the sections in
   `release-please-config.json`.
2. After a push to `main`, approve the `release` environment run if prompted.
   Release-please opens or updates a release PR containing the version in `pyproject.toml`,
   `CHANGELOG.md`, and `.release-please-manifest.json`. The manifest starts at
   the existing `v0.3.2` release. The package reads its runtime version from
   installed metadata, so no source version constant needs updating.
3. App-created or updated release PRs trigger the normal PR workflows. Review the
   proposed version and changelog, wait for required CI, and run the repository's
   final release review before merging the release PR through the merge queue.
4. Approve the ensuing `release` environment run if prompted. Release-please creates
   the `vX.Y.Z` tag and publishes a GitHub Release using the App token. The
   `release: published` event starts `publish.yml` at that tag, including for
   releases published directly by a maintainer. Do not also dispatch publishing:
   the release event is the single automatic handoff.
5. The publishing workflow requires a published GitHub Release and verifies
   that the checked-out tag commit is an ancestor of `main` before installing
   dependencies. It builds the wheel and sdist without OIDC permission, then
   transfers the artifacts to an upload-only job in the `pypi` environment.
   Approve that deployment after checking the release tag and build. The PyPI
   action uses OIDC trusted publishing and explicitly enables PEP 740 attestations.
6. Confirm the publishing run succeeded and inspect the files and attestations
   on PyPI. Configuration alone does not prove a particular upload succeeded.

## Recovery

If release creation succeeds but no publishing run appears, first check Actions
for an existing run for that tag. If none exists, manually run **Publish to PyPI**
with the published release tag selected. Select a `v*` tag containing the
dispatch-enabled workflow; branch runs are skipped. Rerunning release-please
alone does not republish an already-created release. Never manually dispatch
while that tag's publishing run is queued or active, and inspect PyPI before
retrying a completed run.

If publishing fails, rerun the failed job from the original publishing run
while its artifacts remain available (one day), or rerun the whole original run
to rebuild from its release tag. Do not create a new version solely to retry a
failed upload. If an upload partially succeeded, inspect PyPI before retrying:
existing files cannot be overwritten and the workflow does not skip them.

Documentation continues deploying independently on pushes to `main`. Coordinate
documentation for new behavior with its package release, as required by `AGENTS.md`.

PEP 740 publishing attestations are enabled; this workflow does not separately
generate GitHub artifact attestations or SLSA build provenance.

## Local workflow validation

After `make sync`, run `uv run python -m unittest discover -s .github/tests -v`
to check release credential scope, the single automatic publication handoff,
and the publishing guards. These configuration checks do not execute a
production release or verify a PyPI upload.

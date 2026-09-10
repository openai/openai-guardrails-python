# Releasing

Releases use release-please with the repository's `GITHUB_TOKEN`. No SDK GitHub
App installation, client ID, or private key is required.

## One-time setup

Allow GitHub Actions to create pull requests in this repository's Actions
settings. The release job grants its token Contents, Issues, Pull requests, and
Actions write permissions; other jobs keep their own scoped permissions.
Require SDK-team approval for the `release` environment and restrict it to the
exact `main` branch.

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
2. Approve the `release` environment run after a push to `main`. Release-please
   opens or updates a release PR containing the version in `pyproject.toml`,
   `CHANGELOG.md`, and `.release-please-manifest.json`. The manifest starts at
   the existing `v0.3.2` release. The package reads its runtime version from
   installed metadata, so no source version constant needs updating.
3. For a token-created or updated PR, approve its pending workflow runs using
   **Approve workflows to run** in the PR merge box. Review the proposed version
   and changelog, wait for required CI, and run the repository's final release
   review before merging the release PR.
4. Approve the ensuing `release` environment run. Release-please creates the
   `vX.Y.Z` tag and publishes a GitHub Release, then explicitly dispatches
   `publish.yml` at that tag. `GITHUB_TOKEN`-created releases do not trigger
   release-event workflows; `workflow_dispatch` provides the handoff. Releases
   published directly by a maintainer still use the `release: published` event.
5. The publishing workflow requires a published GitHub Release and verifies
   that the checked-out tag commit is an ancestor of `main` before installing
   dependencies. It builds the wheel and sdist without OIDC permission, then
   transfers the artifacts to an upload-only job in the `pypi` environment.
   Approve that deployment after checking the release tag and build. The PyPI
   action uses OIDC trusted publishing and explicitly enables PEP 740 attestations.
6. Confirm the publishing run succeeded and inspect the files and attestations
   on PyPI. Configuration alone does not prove a particular upload succeeded.

## Recovery

If release creation succeeds but dispatch fails, manually run **Publish to
PyPI** from the Actions UI with the published release tag selected. Select a
`v*` tag containing the dispatch-enabled workflow; branch runs are skipped.
Rerunning release-please alone may not dispatch an already-created release.

If publishing fails, rerun the failed job from the original publishing run
while its artifacts remain available (one day), or rerun the whole original run
to rebuild from its release tag. Do not create a new version solely to retry a
failed upload. If an upload partially succeeded, inspect PyPI before retrying:
existing files cannot be overwritten and the workflow does not skip them.

Documentation continues deploying independently on pushes to `main`. Coordinate
documentation for new behavior with its package release, as required by `AGENTS.md`.

PEP 740 publishing attestations are enabled; this workflow does not separately
generate GitHub artifact attestations or SLSA build provenance.

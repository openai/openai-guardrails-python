# Releasing

Releases use release-please, following the GitHub App setup in `openai/openai-python`.

## One-time setup

Before enabling the workflow, install the OpenAI SDKs GitHub App on this repository
with Contents, Issues, and Pull requests write permissions. Make
`OPENAI_SDKS_APP_CLIENT_ID` available as an Actions variable and
`OPENAI_SDKS_APP_PRIVATE_KEY` as an Actions secret to the `release` environment.
Restrict that environment to `main`. The App token lets release PRs run CI and
published GitHub Releases trigger the separate publishing workflow; substituting
`GITHUB_TOKEN` prevents those downstream workflow runs.

Keep the existing PyPI trusted publisher for `openai/openai-guardrails-python`,
workflow `publish.yml`, environment `pypi`. This migration does not rename that
OIDC identity. Check the environment's approval rules and tag deployment rules
before releasing; these settings live outside the repository.

## Routine releases

1. Use Conventional Commit titles when squash-merging PRs into `main`: `fix:`
   produces a patch bump, `feat:` a minor bump, and breaking changes a minor bump
   while the package is pre-1.0. Release notes use the sections in
   `release-please-config.json`.
2. On pushes to `main`, release-please opens or updates a release PR containing
   the version in `pyproject.toml`, `CHANGELOG.md`, and
   `.release-please-manifest.json`. The manifest starts at the existing `v0.3.2`
   release. The package reads its runtime version from installed metadata, so no
   source version constant needs updating.
3. Review the proposed version and changelog, check CI, and run the repository's
   final release review before merging the release PR. Merging authorizes
   release-please to create the `vX.Y.Z` tag and publish a GitHub Release.
4. That release event runs `publish.yml` against the release tag. It builds the
   wheel and sdist in a job without OIDC permission, then transfers the artifacts
   to an upload-only job in the `pypi` environment. The PyPI action uses OIDC
   trusted publishing and explicitly enables PEP 740 attestations.
5. Confirm the publishing run succeeded and inspect the files and attestations
   on PyPI. Configuration alone does not prove a particular upload succeeded.

If publishing fails, rerun the failed job from the original release workflow run
while its artifacts remain available (one day), or rerun the whole original run
to rebuild from its release tag. Do not create a new version solely to retry a
failed upload. If an upload partially succeeded, inspect PyPI before retrying:
existing files cannot be overwritten and the workflow does not skip them.

Documentation continues deploying independently on pushes to `main`. Coordinate
documentation for new behavior with its package release, as required by `AGENTS.md`.

PEP 740 publishing attestations are enabled; this workflow does not separately
generate GitHub artifact attestations or SLSA build provenance.

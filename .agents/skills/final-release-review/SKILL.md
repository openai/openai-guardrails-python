---
name: final-release-review
description: Review an openai-guardrails-python release plan or final release candidate against the previous remote tag, determine the compatible release type, audit runtime and packaging risk, inspect current CI, and produce an English ship-or-block report. Use for pre-release readiness checks, not ordinary PR review or implementation.
---

# Final Release Review

## Purpose

Review `BASE_TAG...TARGET` in one of two modes:

- **Pre-release planning:** Use when reviewing the next release from `origin/main` before a release branch or version bump exists. Recommend the minimum compatible release type.
- **Final candidate:** Use when the target is a release branch, the package version has been bumped beyond the base release, or the user explicitly requests a final release gate. Verify that the checked-out candidate and release metadata agree.

This is a read-only review workflow. Never push, create or edit a pull request, publish a release, create a tag, or otherwise mutate GitHub. Never use `gh`.

The final report must be in English even when the request is in another language.

Documentation coverage is out of scope. Do not search for documentation PRs, assess documentation completeness, or include a documentation coverage section. A concrete docs-build or packaging regression introduced by the candidate remains in scope as code or release infrastructure risk.

## Establish the review inputs

1. Confirm the repository root, current branch, `HEAD`, remote target, and clean status.
2. Refresh remote tags and resolve the latest release tag:

   ```bash
   BASE_TAG="$(.agents/skills/final-release-review/scripts/find_latest_release_tag.sh origin 'v*')"
   ```

3. Refresh the requested target with read-only Git operations. Default to:
   - `origin/main` for pre-release planning;
   - the checked-out release branch `HEAD` for a final candidate.
4. Require `BASE_TAG` to be an ancestor of the refreshed target. If it is not, stop and resolve the release lineage instead of reviewing an unrelated three-dot diff.
5. Record the exact target commit. Keep uncommitted working-tree content outside the reviewed release diff.
6. For a final candidate, require:
   - the checkout to be on the expected release branch rather than detached;
   - `HEAD` to equal the refreshed remote release branch;
   - a clean working tree;
   - the branch name, `pyproject.toml` version, and intended release version to agree;
   - release-owned metadata changes to be committed.

If these conditions are not satisfied, do not silently review a nearby local or remote commit as the candidate.

## Map and audit the release diff

Inspect the full base-to-target comparison:

```bash
git diff --stat "${BASE_TAG}"..."${TARGET}"
git diff --dirstat=files,0 "${BASE_TAG}"..."${TARGET}"
git log --oneline --reverse "${BASE_TAG}".."${TARGET}"
git diff --name-status "${BASE_TAG}"..."${TARGET}"
```

Separate repository workflow/tooling additions from shipped runtime, tests, examples, packaging, and build behavior. Large diff size is a discovery signal, not a release blocker.

For shipped behavior, compare the candidate with the released base rather than reviewing the candidate in isolation. Trace only relevant boundaries:

- documented imports and exports;
- public signatures, positional meaning, defaults, enums, and Pydantic validation;
- guardrail registry names, configuration, result shape, and failure behavior;
- sync/async, Chat Completions/Responses, and streaming/non-streaming parity when the shared path changes;
- tripwire ordering, suppression, exception behavior, cleanup, cancellation, and sensitive-data handling;
- dependencies, Python support, extras, distribution contents, import behavior, and build/publish workflows;
- serialized evaluation inputs, MCP schemas, environment variables, and other durable caller-visible formats.

Read changed tests as behavioral evidence, not proof by themselves. Re-evaluate material review comments against the final candidate content. Use a minimal public-path probe only when static evidence and existing tests cannot settle a decision-relevant question.

## Determine release compatibility

- Use `patch` for compatible bug fixes, security hardening that preserves the supported contract, dependency repairs, and internal improvements.
- Require `minor` for a breaking non-beta public contract change or any backward-compatible addition to supported public functionality.
- Determine the minimum required release type independently from the intended version.
- In planning mode, unchanged version metadata is not a blocker.
- In final-candidate mode, a patch candidate that requires a minor release is blocking.

Treat stricter handling of previously malformed, unsafe, or unsupported input as a compatible patch only when valid supported inputs and public configuration remain usable.

## Check CI and packaging readiness

Use current read-only remote evidence. Keep code quality, CI state, artifact readiness, and publication state distinct.

- Inspect checks for the exact runtime head and candidate head when available.
- Do not treat green CI as proof that a semantic finding is resolved.
- Do not block solely because the version-only candidate commit lacks a separate check run when its runtime parent is green and the candidate delta is proven release-only.
- When version, dependency, or build metadata changed, build the wheel and sdist locally when practical and inspect their name, version, Python requirement, and dependency metadata.
- Do not rerun the full local suite merely to duplicate a green remote matrix for the exact runtime content. Run focused or broad checks only when they can change the release decision.
- Verify public package or GitHub Release state only when the report makes a claim about what is currently published.

## Deterministic release gate

Default to **GREEN LIGHT TO SHIP** unless a blocker is proven.

Use **BLOCKED** only for concrete evidence of at least one of these conditions:

- a regression or bug introduced in `BASE_TAG...TARGET` on a supported path;
- an under-versioned final candidate;
- inconsistent branch, commit, version, or artifact metadata;
- a breaking public API, configuration, protocol, or durable-state change without a usable migration or compatibility path;
- unresolved data loss, corruption, security impact, or sensitive-data exposure;
- a broken release-critical build, package, import, installation, or publish path.

The following are not blockers by themselves:

- large or complex diffs;
- speculative risk without a reachable failure;
- missing local duplication of green remote checks;
- missing documentation or documentation coverage work;
- absent check runs on a proven version-only commit.

Every reported risk must include impact, base-versus-target evidence, affected files, and an actionable next step or preservation condition. Use:

- **LOW** for verified compatible changes and operational considerations;
- **MODERATE** for concrete unresolved regression signals that do not yet prove a blocker;
- **HIGH** for confirmed release blockers.

Any target, version, dependency, artifact, or release-owned metadata change after the review invalidates the gate.

## Required output

Return the report in English. Do not include a documentation coverage section. Do not include Key Changes or release-note copy unless the user separately requests it.

Use this structure:

```markdown
COMPLETE

## Release readiness review

`<base-tag>` -> `<target-ref>` (`<target-sha>`)

Diff: https://github.com/openai/openai-guardrails-python/compare/<base-tag>...<target-sha>

### Release intent

- Review mode: <pre-release planning | final candidate>
- Intended release: <version/type or unspecified>
- Minimum required release type: <patch | minor>
- Recommended release type: <patch | minor; planning mode only>
- Versioning verdict: <compatible | compatible plan | revise plan to minor | under-versioned>

### Release call

**<🟢 GREEN LIGHT TO SHIP | 🔴 BLOCKED>**

<One concise rationale and candidate consistency statement.>

### Scope summary

<File and line statistics, key shipped areas, and separation from repository-only tooling.>

### Risk assessment

1. **<Finding or verified release consideration>**
   - Risk: **<🟢 LOW | 🟡 MODERATE | 🔴 HIGH>**. <Impact.>
   - Evidence: <Specific base-versus-target evidence.>
   - Files: `<paths>`
   - Action: <Exact next step or preservation condition.>

### CI and packaging readiness

- <Exact runtime and candidate check state.>
- <Artifact or metadata verification.>
- <Working-tree and candidate consistency.>
- <Any intentionally omitted local duplication.>

<If blocked, add an Unblock checklist with exact exit criteria.>

This <green|blocked> gate applies only to `<target-sha>`. Any candidate, version, dependency, or release-owned metadata change invalidates the result and requires another review.
```

Omit `Recommended release type` in final-candidate mode. Omit the unblock checklist when green. Keep routine command output, test counts, and low-value inventories out of the report.

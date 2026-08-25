# Contributor Guide

This guide defines the required workflow for agents and contributors working in the OpenAI Guardrails Python repository.

## Policies and mandatory rules

### Repository skills

Repository skills live under `.agents/skills/`. A reference such as `$<skill-name>` is a repository instruction reference, not a request for manual user invocation. When a rule requires a skill, read `.agents/skills/<skill-name>/SKILL.md` completely before taking task actions, follow it, and resolve referenced files relative to that skill directory.

This repository defines these repository skills:

- `$code-change-verification`
- `$final-release-review`
- `$implementation-final-review`
- `$implementation-kickoff`
- `$implementation-strategy`
- `$maintainer-review`
- `$pr-draft-summary`

#### `$final-release-review`

Use `$final-release-review` when the user asks for pre-release planning or a final release-candidate review. Compare the target with the previous remote release tag, determine the minimum compatible release type, audit runtime and packaging risk, inspect current CI, and issue the English ship-or-block report required by the skill. Documentation coverage is intentionally out of scope. This is a read-only workflow and never authorizes GitHub mutation.

#### `$implementation-kickoff`

Use `$implementation-kickoff` only when the user explicitly invokes it. It may create a dedicated worktree, a local branch, and one local commit as described by the skill. It never authorizes a push, pull-request creation, or any other GitHub mutation.

#### `$implementation-strategy`

Before changing or reviewing runtime code, exported APIs, external configuration, persisted schemas, wire formats, or other caller-visible behavior, use `$implementation-strategy` to define the compatibility boundary and the smallest coherent implementation.

Before coding, record an implementation scope contract with:

1. Required behavior.
2. Compatibility requirements.
3. Intentionally unsupported cases and their failure behavior.
4. A supported alternative, or `none`.

Repeat the strategy check before each review-feedback batch that would widen supported behavior, add a compatibility branch, change lifecycle or protocol ownership, or expand test permutations.

Independent reviewers dispatched by `$implementation-final-review` inherit the implementer's recorded implementation scope contract. The implementer remains responsible for rerunning `$implementation-strategy` before any review-feedback batch that widens the supported contract or changes a durable boundary.

#### `$implementation-final-review`

After implementing runtime code, tests, examples, build or test behavior, or behavior-impacting docs and completing focused checks, run `$implementation-final-review` before broad verification and before declaring the task complete. Repository instructions authorize this automatic invocation without a separate user mention.

Do not invoke it for planning, investigation, report-only work, repository metadata changes, or documentation without behavior impact. Its clean-review gate does not replace verification.

#### `$code-change-verification`

Run `$code-change-verification` after final review and before marking work complete when changes affect:

- `src/guardrails/` or `mcp_server/` runtime code.
- `tests/`.
- `examples/`.
- Build, packaging, docs-build, or test configuration such as `pyproject.toml`, `uv.lock`, `Makefile`, `mkdocs.yml`, `docs/scripts/`, or CI workflows.

Skip it for repository metadata or editorial documentation-only changes, including `.agents/`, `AGENTS.md`, `README.md`, and ordinary prose under `docs/`, unless the user explicitly requests the full stack or the change affects executable examples, generated reference content, or build behavior.

Treat this skill as the final broad gate. During iterative review, prefer focused tests and narrowly targeted static checks.

Immediately before the broad stack, use available read-only task or process evidence to check for another repository-wide test, typecheck, build, examples, or integration command on the same host. When concrete contention exists, continue useful non-heavy work and retry later. Do not add a repository lock, host-wide mutex, sentinel file, or user-triggered `finalize` step. Lack of host telemetry alone is not a blocker.

#### `$maintainer-review`

Use `$maintainer-review` when the user asks for a maintainer-level review of an issue, pull request, proposed fix, or competing implementations. Separate evidence for the user need from code quality and repository readiness. Use read-only remote access only.

#### `$pr-draft-summary`

Before the final response for a task that changed runtime code, tests, examples, build/test configuration, or behavior-impacting docs, use `$pr-draft-summary` after review and verification. It produces copy-ready text only and does not authorize creating a branch, committing, pushing, or opening a pull request.

Skip it for repository metadata, editorial docs, conversation-only work, or when the user explicitly says not to include a PR draft.

### Work status reporting

- Use `RUNNING` only in commentary while autonomous work remains and no user action is required.
- Use `COMPLETE` in the final response only when every applicable implementation, review, verification, commit, and handoff step is complete.
- Use `NEEDS_DECISION` in the final response only when progress requires a concrete user choice, expanded authority, or unresolved external condition. State the exact decision or condition instead of asking the user to say "continue".

### Git and GitHub safety

- Work in the current checkout and branch unless the user explicitly asks for or approves a different branch or worktree.
- Preserve unrelated and user-owned changes. Never remove or overwrite an existing worktree or branch to make room.
- Agent workflows must never push, open or edit pull requests, post comments or reviews, merge, tag, publish releases, or otherwise mutate GitHub.
- Do not run `gh`. Use an approved read-only GitHub mechanism when remote evidence is required.
- Local branch creation, staging, and commits require explicit user authorization or explicit invocation of `$implementation-kickoff`.
- Stop after local verification and the requested local handoff.

### Scope discipline and complexity reset

- Implement the narrowest explicitly requested behavior.
- Prefer adapting the required case into an existing pipeline over creating a parallel schema, validation path, resolver, or source of truth.
- Every new abstraction, state field, compatibility branch, configuration option, or dependency must map to a requirement, released contract, durable boundary, or verified runtime risk.
- Do not treat every Python shape or third-party protocol variant as supported merely because it is constructible.
- When a second related review finding would add another condition, protocol hop, compatibility case, or test permutation to the same abstraction, stop patching and run the complexity-reset workflow in `$implementation-strategy`.
- Keep unrelated refactors and pre-existing failures out of the patch.

### Documentation release timing

When a feature or bug fix introduces behavior that is not yet available in the latest published release, do not include `docs/` changes that describe that unreleased behavior in the feature or bug-fix pull request, and do not expect those changes as part of that pull request. Handle them in a separate docs-only pull request so maintainers can coordinate its merge timing with the release that makes the documentation accurate. This exception applies only when the documentation would be incorrect for the latest published release; documentation already accurate for released behavior remains part of the normal change scope.

Determine whether documentation is required separately from deciding which pull request should carry it. When required `docs/` content would describe behavior unavailable in the latest published release, classify it as separately timed documentation work rather than a missing deliverable or blocking finding. This timing rule takes precedence over general documentation-completeness requirements in review rules and repository skills. It applies to `docs/` content, not automatically to examples or code-level documentation that ships with the changed API.

### Documentation verification tiers

Use the narrowest tier that covers the complete diff:

- Editorial: spelling, terminology, punctuation, or formatting without changed behavior, runnable code, navigation, links, anchors, or generated content. Inspect the diff, run targeted searches, and run `git diff --check`.
- Content: new or materially changed behavior guidance or runnable snippets without structural changes. Verify claims against code and authoritative sources, validate changed snippets when practical, and run `make build-docs` after content is stable.
- Structural: added, removed, renamed, or moved pages; changes to `mkdocs.yml`, documentation scripts, plugins, or generated reference inputs. Run focused generators and `make build-docs` after structure is stable. Use `$code-change-verification` when build or test configuration is affected.

Existing warnings from a successful documentation build are not findings for an unrelated docs change. Evaluate the exit status and identify new errors, broken references, or warnings caused by the diff instead of reviewing the complete warning stream line by line.

Do not edit generated output. Use `make build-full-docs` only for translation-tooling changes, explicit localization work, or a requested broad localization audit.

## Code review rules

### Finding threshold and supported scope

- Report a runtime defect only when changed code causes concrete incorrect behavior on a supported path. State the trigger and the caller-visible, compatibility, security, persistence, or lifecycle consequence; omit the finding when no such consequence can be established.
- Treat added abstractions, state, validation, compatibility handling, fallback behavior, dependencies, or parallel paths as actionable only when the machinery does not map to the task, a released contract, supported durable state, or a verified runtime or platform risk. Identify the exact unnecessary machinery and recommend the smallest safe removal or direct replacement.
- Flag validation, compatibility handling, fallback behavior, or tests added only for synthetic or unsupported values when no ordinary supported producer, released contract, durable boundary, or actual untrusted-input path can produce the value with a concrete consequence. Constructibility in Python, manually corrupted typed objects, monkeypatched state, and direct helper calls that bypass the owning public or wire boundary are not sufficient justification.
- Do not duplicate client-side runtime validation solely for values already excluded by the public type contract or authoritatively rejected by the upstream provider. Add fail-fast library validation only when delayed rejection creates a concrete library-owned problem before the authoritative rejection, such as an irreversible side effect, persistent corruption, security or privacy exposure, avoidable billable work, repeated resource consumption, or an error that arrives too late or is too opaque for reasonable correction.
- Do not report a defect merely because another semantic choice appears cleaner, more symmetric, or easier to explain. When repository evidence does not select one contract, report only a concrete inconsistency with an already supported path or established caller-visible expectation.
- Flag a new public option, callback, class, compatibility branch, or parallel execution path when the exact required outcome, including lifecycle and compatibility constraints, is already available through a reasonable supported API or composition path. Name that path and recommend removal or narrower reuse of the existing source of truth.
- Report compatibility findings only against behavior shipped in the latest release, an explicitly supported public contract, or a durable external state or protocol boundary. Do not require compatibility shims for unreleased branch-local helpers, same-branch tests, or intentionally unsupported intermediate persisted formats.

### Contract and lifecycle coverage

- For every added or modified public field, configuration value, event, serialized value, or wire value, inspect all supported construction, forwarding, adapter, and consumption paths. Flag partial implementations where normal, specialized, default, missing-value, or error paths silently drop, reshape, or reject the value inconsistently. Include intended public imports and generated package surfaces when they are part of the changed contract.
- Require parity across streaming and non-streaming, sync and async, Chat Completions and Responses, OpenAI and Azure, or direct and Agents SDK paths only when the accepted requirement or existing contract covers those paths. Do not report missing parity solely for API symmetry or conceptual similarity.
- When changed code mutates shared state across an `await`, callback, retry, cancellation, cleanup, or rollback boundary, check whether stale or failing work can overwrite, revert, or dispose state owned by surviving work. Report the concrete interleaving and the missing ownership, generation, identity, transaction, revalidation, or serialization invariant at the actual mutation boundary; sequential happy-path tests are insufficient.
- When a new validation or failure path can run after resources or observable state are acquired, verify cleanup explicitly and preserve the primary failure. Report concrete leaked resources, stale state, lost handlers, or survivor corruption rather than assuming normal teardown runs after failed construction or context entry.
- Flag persisted, resumed, serialized, provider-controlled, or manifest data that is treated as authority for a host-owned runtime, security, identity, or cleanup decision unless the supported trust boundary explicitly grants that authority. Preserve trusted current configuration and validate untrusted state before it can affect side effects, replay, or resource ownership.

### Test and documentation evidence

- Treat tests as contract evidence only when they exercise the highest stable caller-visible boundary that controls the observable result and derive expected behavior from the requirement, released behavior, a worked example, a baseline, or another independent oracle. Do not accept helper-only call-shape assertions or expected values recomputed with the implementation's own logic when another layer owns the outcome.
- Require representative regression coverage for accepted behavior and intentionally unsupported categories. For concurrency findings, require controlled completion ordering plus assertions about the surviving operation and final shared state. Do not request exhaustive tests for every constructible permutation.
- Report missing documentation or examples only when the patch makes existing guidance materially false, unsafe, or misleading; correct use depends on a non-obvious migration, compatibility boundary, constraint, or operational warning; or the accepted feature would otherwise be practically unusable. Do not report optional completeness or discoverability improvements as blocking findings.
- Decide `docs/` delivery timing separately from documentation necessity. If required `docs/` content would describe behavior unavailable in the latest published release, apply the documentation release timing policy: record it as separately timed work and do not report its absence as a blocking finding for the feature or bug-fix pull request.
- Do not report formatting, lint, full-suite status, commit history, or pull-request description quality as code findings; those are CI or repository-readiness conditions.

### Review scope

- Review the complete diff from the merge base of the intended target branch, or from the latest release tag when it is the compatibility baseline, not only the latest incremental fix. Passing tests do not justify branch-local machinery that no longer matches the original requirement.
- Keep findings scoped to consequences introduced, exposed, or worsened by the patch. Do not block on unrelated cleanup, pre-existing bugs, optional refactors, or speculative extensibility merely discovered while reading adjacent code. A pre-existing condition is in scope when the patch newly reaches it on a supported path, relies on it for correctness, or otherwise makes its consequence part of the changed behavior.
- Require a broader refactor only when concrete evidence shows the focused change would otherwise remain incorrect, unsafe, incompatible, or dependent on duplicated sources of truth that can observably diverge.

## Project structure

- `src/guardrails/`: Public package and runtime implementation.
- `src/guardrails/checks/`: Built-in guardrail checks.
- `src/guardrails/evals/`: Evaluation runtime.
- `mcp_server/`: Workspace package for the MCP server.
- `tests/unit/`: Hermetic unit tests.
- `tests/integration/`: Repository integration tests; these are still expected to avoid live services unless explicitly marked and authorized.
- `examples/`: User-facing examples.
- `docs/`: MkDocs documentation source.
- `pyproject.toml`, `uv.lock`: Dependencies, packaging, and tool configuration.
- `Makefile`: Common development commands.
- `.github/workflows/`: CI, docs, and publication workflows.

Use `uv run ...` for Python commands so local execution uses the repository environment.

## Guardrails runtime boundaries

### Public API and compatibility

- Treat documented imports from `guardrails` and exported symbols in `src/guardrails/__init__.py` as compatibility contracts.
- Preserve positional argument meaning for released public constructors and functions. Append optional parameters when possible.
- Keep sync and async clients behaviorally aligned: `GuardrailsOpenAI`, `GuardrailsAsyncOpenAI`, `GuardrailsAzureOpenAI`, and `GuardrailsAsyncAzureOpenAI`.
- Review Chat Completions and Responses API paths together when shared guardrail behavior changes.
- Review streaming and non-streaming paths together when input checks, output checks, suppression, response wrapping, cancellation, or exception behavior changes.
- Preserve caller-visible OpenAI client behavior that this package intentionally proxies, including response types, streaming ownership, and supported keyword forwarding.
- Treat guardrail configuration, Pydantic models, registry names, CLI flags, serialized evaluation inputs, and MCP schemas as external configuration or wire contracts when released.
- Keep import-time behavior free of live network calls and optional-resource failures.

### Guardrail execution and failures

- Preserve stage ordering and the distinction between input, output, and tool guardrails.
- Keep `GuardrailTripwireTriggered` and Agents SDK tripwire behavior consistent with documented suppression and propagation rules.
- Validate malformed configuration before model or provider side effects whenever possible.
- Trace cleanup and ownership across success, tripwire, provider error, cancellation, partial stream consumption, and early iterator close.
- Do not retain sensitive user input, model output, credentials, or provider payloads in exception chains, logs, telemetry, or test artifacts unless the public contract explicitly requires it.
- Treat remote or model-backed checks as paid and side-effecting. Unit and integration tests must remain hermetic by default.

### Tests

- Mirror source behavior under `tests/unit/` and reserve `tests/integration/` for cross-module behavior.
- Prefer DAMP tests that read as specifications. Use fixtures and parametrization where they improve clarity.
- Add regression tests for required behavior and representative failure paths. Use Hypothesis for non-trivial input domains when it materially improves coverage.
- Async tests use `pytest.mark.asyncio`. Do not use sleeps or real network calls in ordinary tests.
- Test public behavior rather than branch-local helper structure.
- For changes to a shared guardrail path, cover relevant sync/async, streaming/non-streaming, and Chat Completions/Responses variants without duplicating mechanically equivalent cases.

## Development workflow

1. Inspect the current status and applicable instructions without modifying user-owned work.
2. Use `$implementation-strategy` when the change affects runtime or caller-visible contracts.
3. Implement the narrowest coherent change and add focused tests.
4. Run focused formatting, static checks, and tests while iterating.
5. Run `$implementation-final-review` when applicable.
6. Run `$code-change-verification` when applicable.
7. Run `$pr-draft-summary` when applicable.
8. Create a local commit only when authorized. Never push or mutate GitHub.

### Common commands

Install or refresh dependencies when needed:

```bash
make sync
```

Run focused tests:

```bash
uv run pytest -q tests/unit/test_<area>.py
```

Run formatting and linting:

```bash
make format
make lint
```

Run both configured type checkers:

```bash
uv run mypy src tests
uv run pyright
```

Run the test suite and coverage gate:

```bash
make tests
make coverage
```

Build documentation:

```bash
make build-docs
```

### Code style

- Target Python 3.11+ and fully type public APIs.
- Use descriptive names and straightforward control flow.
- Prefer pure functions and immutable values when they reduce state and lifecycle complexity.
- Use specific exception handling and preserve useful context with explicit chaining.
- Use `logging`, not `print`, in library code. Never log secrets or raw sensitive content.
- Follow the Ruff and mypy configuration in `pyproject.toml`; repository configuration overrides generic style preferences.
- Public functions and classes require concise Google-style docstrings.
- Keep comments focused on why, not a restatement of the code.

## Commit and PR text

- Use concise, imperative commit subjects. Conventional prefixes such as `fix:`, `feat:`, `docs:`, and `chore:` are preferred when they clarify intent.
- Keep commits focused and include tests with behavior changes.
- Copy-ready GitHub text must use `#123` for this repository and `owner/repo#123` for cross-repository references. Do not wrap native issue or pull-request references in Markdown links.
- Never include local paths, internal review artifacts, task IDs, or Codex-only directives in copy-ready external text.

## Review baseline

- The implementation satisfies the explicit requirement without unsupported scope expansion.
- Released public behavior and durable formats remain compatible or have an explicitly approved migration.
- Sync/async, streaming/non-streaming, and Chat Completions/Responses parity are covered where affected.
- Failure, cancellation, cleanup, and sensitive-data paths are reviewed where affected.
- Tests cover new behavior and representative edge cases.
- Applicable formatting, lint, type checking, tests, and documentation checks pass.
- The final diff contains no unrelated changes and no untracked task deliverables are omitted.

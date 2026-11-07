# Contributor Guide

## Policies and mandatory rules

### Security review

Before pushing or opening a pull request, review the complete diff for security
impact alongside the normal code review. For affected paths:

- Trace untrusted configuration, provider responses, tool arguments, and evaluation
  inputs to network, file, subprocess, and deserialization operations; verify the
  trust boundary before side effects.
- Check that credentials, prompts, model outputs, and tool data cannot leak through
  logs, exception chains, telemetry, or test artifacts. Use synthetic regression
  cases and keep tests hermetic.
- Review dependency changes for necessity, source integrity, and known advisories.
  For CI or release changes, verify action SHA pins, least-privilege token grants,
  trusted refs, and isolation of publishing credentials from untrusted code.
- Run the applicable validation below and inspect required security-check results.
  Fix introduced findings before handoff; do not disable checks, weaken protections,
  or dismiss findings merely to make CI pass.

Record the reviewed surfaces, validation, and unresolved risks in the handoff,
without sensitive details. Route suspected vulnerabilities privately according to
[SECURITY.md](SECURITY.md); do not put exploit details or secrets in public issues,
pull requests, or logs. If a security decision requires additional authority,
complete independent work and identify the exact remaining decision privately.

### Repository skills

`$<skill-name>` references are instructions, not requests for manual invocation. When applicable below, read the linked skill completely before task actions and follow it; resolve its references relative to its directory.

| Skill | Required use / limits |
| --- | --- |
| [final-release-review](.agents/skills/final-release-review/SKILL.md) | Requested pre-release planning or final candidate review: compare the previous remote release, determine minimum compatible release type, audit runtime/packaging and CI, and produce the English ship-or-block report. Documentation coverage excluded; read-only, no GitHub mutations. |
| [implementation-kickoff](.agents/skills/implementation-kickoff/SKILL.md) | Explicit invocation only. May create a dedicated worktree, local branch, and one local commit; no push, PR, or other GitHub mutation. |
| [implementation-strategy](.agents/skills/implementation-strategy/SKILL.md) | Before changing or reviewing runtime, exported APIs, external configuration, persisted schemas, wire formats, or caller-visible behavior. Before coding, record required behavior, compatibility, unsupported cases/failure behavior, and supported alternative (or `none`). Reviewers inherit this contract. Implementer repeats strategy before feedback batches widening support, adding compatibility branches, changing lifecycle/protocol ownership, or expanding test permutations. |
| [implementation-final-review](.agents/skills/implementation-final-review/SKILL.md) | After runtime, tests, examples, build/test behavior, or behavior-impacting docs changes and focused checks; before broad verification and completion. Automatically authorized. Skip planning, investigation, reports, metadata, and docs without behavior impact. Clean review does not replace verification. |
| [code-change-verification](.agents/skills/code-change-verification/SKILL.md) | Final broad gate after review for runtime (`src/guardrails/`, `mcp_server/`), tests, examples, build/packaging/docs-build/test configuration (`pyproject.toml`, `uv.lock`, `Makefile`, `mkdocs.yml`, `docs/scripts/`, CI). Skip metadata/editorial docs (`.agents/`, `AGENTS.md`, `README.md`, ordinary `docs/` prose) unless full-stack verification is requested or executable examples, generated reference content, or build behavior changes. Follow its host-capacity checks; use focused checks while iterating. |
| [maintainer-review](.agents/skills/maintainer-review/SKILL.md) | Requested maintainer review of issues, PRs, fixes, or competing implementations. Separate user-need evidence, code quality, and readiness; read-only remote access. |
| [pr-draft-summary](.agents/skills/pr-draft-summary/SKILL.md) | After review/verification, before the final response for runtime, tests, examples, build/test configuration, or behavior-impacting docs. Copy-ready text only; no Git mutations. Skip metadata, editorial docs, conversation-only work, or explicit opt-out. |

### Work status reporting

- Use `RUNNING` only in commentary while autonomous work remains and no user action is required.
- Use `COMPLETE` in the final response only when every applicable implementation, review, verification, commit, and handoff step is complete.
- Use `NEEDS_DECISION` in the final response only when progress requires a concrete user choice, expanded authority, or unresolved external condition. State the exact decision or condition instead of asking the user to say "continue".

### Git and GitHub safety

- Work in the current checkout and branch unless the user explicitly asks for or approves a different branch or worktree.
- Preserve unrelated and user-owned changes. Never remove or overwrite an existing worktree or branch to make room.
- Perform Git and GitHub actions within the user's authorized task scope, including creating branches, committing, pushing, and managing pull requests. Do not request authorization again for actions already authorized in the conversation.
- Use the available GitHub connector or Git tooling for remote operations.
- Complete required review and verification before pushing or opening or updating a pull request. Monitor CI, fix in-scope failures, and request review in the root `#sdk-reviews` channel once checks pass.
- Complete the authorized delivery and handoff steps; stop at local verification only when the user requests a local-only task.

### Scope discipline and complexity reset

- Implement the narrowest explicitly requested behavior.
- Prefer adapting the required case into an existing pipeline over creating a parallel schema, validation path, resolver, or source of truth.
- Every new abstraction, state field, compatibility branch, configuration option, or dependency must map to a requirement, released contract, durable boundary, or verified runtime risk.
- Do not treat every Python shape or third-party protocol variant as supported merely because it is constructible.
- When a second related review finding would add another condition, protocol hop, compatibility case, or test permutation to the same abstraction, stop patching and run the complexity-reset workflow in `$implementation-strategy`.
- Keep unrelated refactors and pre-existing failures out of the patch.

### Documentation release timing

`docs/` content describing behavior unavailable in the latest published release belongs in a separate docs-only PR, timed by maintainers with that release. Record it as separately timed work, not a missing deliverable or blocking finding in the feature/fix PR. Determine documentation necessity separately from delivery timing.

This rule overrides general documentation-completeness rules and skills. Documentation accurate for released behavior remains in normal scope. The exception does not automatically cover examples or code-level documentation shipped with the API.

### Documentation verification tiers

Use the narrowest tier that covers the complete diff:

- Editorial: spelling, terminology, punctuation, or formatting without changed behavior, runnable code, navigation, links, anchors, or generated content. Inspect the diff, run targeted searches, and run `git diff --check`.
- Content: new or materially changed behavior guidance or runnable snippets without structural changes. Verify claims against code and authoritative sources, validate changed snippets when practical, and run `make build-docs` after content is stable.
- Structural: added, removed, renamed, or moved pages; changes to `mkdocs.yml`, documentation scripts, plugins, or generated reference inputs. Run focused generators and `make build-docs` after structure is stable. Use `$code-change-verification` when build or test configuration is affected.

Existing warnings from a successful documentation build are not findings for an unrelated docs change. Evaluate the exit status and identify new errors, broken references, or warnings caused by the diff instead of reviewing the complete warning stream line by line.

Do not edit generated output. Use `make build-full-docs` only for translation-tooling changes, explicit localization work, or a requested broad localization audit.

## Code review rules

### Finding threshold and supported scope

- Runtime findings require concrete incorrect behavior on a supported path. State the trigger and caller-visible, compatibility, security, persistence, or lifecycle consequence; otherwise omit.
- Flag abstractions, state, validation, compatibility, fallbacks, dependencies, or parallel paths lacking a task requirement, released contract, supported durable boundary/state, or verified runtime/platform risk. Name the unnecessary machinery and smallest safe removal/replacement. Flag new public options, callbacks, classes, compatibility branches, or execution paths when a reasonable supported API/composition already meets the exact requirement, including lifecycle/compatibility constraints. Name that path and recommend removal or narrower reuse.
- Synthetic/unsupported values do not justify machinery or tests without an ordinary supported producer, released contract, durable boundary, or actual untrusted-input path and concrete consequence. Python constructibility, corrupted typed objects, monkeypatches, and helpers bypassing public/wire owners are insufficient.
- Do not duplicate validation for values excluded by public types or authoritatively rejected upstream. Fail fast only for a concrete library-owned problem before rejection: irreversible effects, persistent corruption, security/privacy exposure, avoidable billing, repeated resource use, or errors too late/opaque for reasonable correction.
- Cleaner or more symmetric semantics alone are not defects. If evidence does not select a contract, require concrete inconsistency with supported behavior or established caller expectations.
- Compatibility findings require latest-release behavior, an explicitly supported public contract, or durable external state/protocol boundaries. No shims for unreleased helpers, same-branch tests, or intentionally unsupported intermediate persisted formats.

### Contract and lifecycle coverage

- For added or modified public fields, configuration, events, serialized/wire values, inspect every supported construction, forwarding, adapter, and consumption path, including intended imports/generated surfaces. Flag inconsistent dropping, reshaping, or rejection across normal, specialized, default, missing, and error paths.
- Require streaming/non-streaming, sync/async, Chat Completions/Responses, OpenAI/Azure, and direct/Agents SDK parity only where the accepted requirement or existing contract applies; symmetry alone is insufficient.
- For shared-state mutations across await, callback, retry, cancellation, cleanup, or rollback, check whether stale/failing work can overwrite, revert, or dispose surviving work's state. Identify the concrete interleaving and missing ownership, generation, identity, transaction, revalidation, or serialization invariant at the mutation boundary. Sequential happy-path tests are insufficient.
- For validation/failure after resource or observable-state acquisition, verify cleanup and primary-failure preservation explicitly. Report concrete leaks, stale state, lost handlers, or survivor corruption; do not assume teardown after failed construction/context entry.
- Persisted, resumed, serialized, provider-controlled, or manifest data must not authorize host-owned runtime, security, identity, or cleanup decisions unless the supported trust boundary grants authority. Preserve trusted current configuration; validate untrusted state before side effects, replay, or ownership changes.

### Test and documentation evidence

- Tests must exercise the highest stable caller-visible boundary controlling the result, with expectations from requirements, released behavior, worked examples, baselines, or independent oracles. Helper call-shape assertions or implementation-derived expectations are insufficient when another layer owns the outcome.
- Require representative regressions for accepted behavior and intentionally unsupported categories, not every constructible permutation. Concurrency tests need controlled completion order and assertions on surviving work and final shared state.
- Missing docs/examples block only when existing guidance becomes materially false/unsafe/misleading, correct use requires a non-obvious migration/compatibility boundary/constraint/operational warning, or the accepted feature is otherwise practically unusable. Optional completeness/discoverability does not block. Apply [documentation release timing](#documentation-release-timing) separately: required unreleased `docs/` content is separately timed work, not a blocker for the feature/fix PR.
- Formatting, lint, full-suite status, commit history, and PR descriptions are CI/readiness conditions, not code findings.

### Review scope

- Review the complete diff from the intended target's merge base; use the latest release tag for compatibility. Passing tests do not justify branch-local machinery outside the original requirement.
- Findings must concern consequences introduced, exposed, or worsened by the patch. Pre-existing conditions qualify only when newly reached on a supported path, relied on for correctness, or otherwise made consequential by the change. Exclude unrelated cleanup, pre-existing bugs, optional refactors, and speculative extensibility.
- Require broader refactoring only with evidence the focused change remains incorrect, unsafe, incompatible, or dependent on duplicated sources of truth that can observably diverge.

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

Inspect status/instructions without modifying user-owned work; apply the [skill gates](#repository-skills) in order: strategy → focused implementation/tests/checks → final review → broad verification → PR summary → authorized delivery and handoff. Keep commits focused and include tests with behavior changes.

### Common commands

Use the repository environment (`uv run`) for Python commands. See [Makefile](Makefile) for command definitions.

| Purpose | Command |
| --- | --- |
| Install/refresh dependencies | `make sync` |
| Focused tests | `uv run pytest -q tests/unit/test_<area>.py` |
| Format / lint | `make format` / `make lint` |
| Both type checkers | `uv run mypy src tests` and `uv run pyright` |
| Test suite / coverage gate | `make tests` / `make coverage` |
| Documentation | `make build-docs` |

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
- Copy-ready GitHub text must use `#123` for this repository and `owner/repo#123` for cross-repository references. Do not wrap native issue or pull-request references in Markdown links.
- Never include local paths, internal review artifacts, task IDs, or Codex-only directives in copy-ready external text.

## Review baseline

Apply the [code review rules](#code-review-rules), [runtime boundaries](#guardrails-runtime-boundaries), and applicable [skill gates](#repository-skills). Confirm the requirement is met without unsupported scope, released behavior/durable formats remain compatible or have an approved migration, affected parity/failure/cleanup/security paths and representative tests are covered, and all applicable checks pass. Audit the final diff for unrelated changes and omitted task-owned untracked deliverables.

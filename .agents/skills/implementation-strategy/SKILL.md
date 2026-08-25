---
name: implementation-strategy
description: Choose compatibility-aware scope for runtime and API changes in openai-guardrails-python. Use before initial implementation and each review-feedback batch to decide whether to patch, reset the design, preserve compatibility, or reject unsupported cases.
---

# Implementation Strategy

## Workflow

1. Identify the surface you are changing or reviewing: released public API, unreleased branch-local API, internal helper, persisted schema, wire protocol, CLI/config/env surface, or docs/examples only.
2. Determine the latest release tag to use as the compatibility baseline from `origin` first, and only fall back to local tags when remote tags are unavailable:
   ```bash
   BASE_TAG="$(.agents/skills/final-release-review/scripts/find_latest_release_tag.sh origin 'v*' 2>/dev/null || git tag -l 'v*' --sort=-v:refname | head -n1)"
   echo "$BASE_TAG"
   ```
   Report a local-tag fallback as potentially stale.
3. Record the implementation scope contract below before coding.
4. Identify the nearest existing implementation pipeline and the functions, types, or modules that are the source of truth for each affected concern. Prefer adapting the required input into that pipeline over creating parallel schema, metadata, validation, naming, or execution machinery.
5. Choose the smallest coherent change using the core decision rules. Add compatibility machinery only for a required supported boundary.
6. Before editing each review-feedback batch, run the review gate against the complete branch diff, not only the latest revision.
7. Before handoff, run the effectiveness check. If any answer is no, revise the design.

## Implementation scope contract

Record these four items in the plan or working notes, and update them before widening or narrowing the implementation:

1. **Required behavior:** The smallest user-visible scenario that must work.
2. **Compatibility requirements:** Supported released behavior or a durable boundary that must remain usable.
3. **Intentionally unsupported cases:** Nearby inputs or shapes to reject, including when and how rejection occurs.
4. **Supported alternative:** An existing wrapper, override, adapter, configuration, or lower-level API; state `none` when absent.

If the intentionally unsupported cases cannot be stated clearly, do not start by adding a general resolver. First define a narrower behavior contract. If no adequate supported alternative exists, add one only when the task requires it; do not invent one speculatively.

A released-version reproducer proves reachability, not support. Treat the exact shape as a compatibility requirement only when intentionally covered by public documentation, examples, tests, or typing; required by a durable boundary; or backed by concrete user reliance or maintainer intent. Otherwise record the risk and prefer early rejection with an existing supported alternative.

## Review-feedback gate

Repeat this gate before editing each new feedback batch:

```text
Review checkpoint:
- Root cause and required behavior:
- Compatibility evidence and unsupported cases:
- Source of truth:
- Behavior-space change: narrows / unchanged / widens
- Action: focused patch / complexity reset / reject as unsupported
```

Classify each finding as a required-behavior defect, supported compatibility requirement, another combination of the same implementation dimensions, or unrelated issue. Widening the behavior space requires new contract evidence.

If a second related finding would add another condition, protocol hop, compatibility case, or test permutation to the same abstraction, stop patching and run the complexity reset. Continue only when concrete evidence puts the exact case in the required or supported contract.

After a reset spec is frozen, classify each later finding as a violation of that spec, an evidence-backed reason to revise it, an intentionally unsupported case, or an unrelated issue. Do not resume incremental patching merely because the new finding is locally fixable.

Example: if successive findings require traversing a direct wrapper, partial, nested wrapper, descriptor, and bound method, do not add another hop. Unless arbitrary wrapper graphs are supported, retain the required plain callable behavior and reject ambiguous wrappers before invocation.

## Core decision rules

- Preserve released public APIs, documented behavior, and supported durable boundaries, or provide an explicit migration path.
- Rewrite branch-local interfaces, internal helpers, same-branch tests, and post-release additions on `main` directly unless they already define a supported durable boundary.
- Unreleased persisted schema versions may be renumbered or squashed when intermediate snapshots are intentionally unsupported; update the support set and tests together.
- Do not equate a broad Python or third-party protocol with support for every representable shape.
- Prefer the nearest existing pipeline and one source of truth for schema, documentation, validation, identity, and invocation.
- Treat an interface as everything a caller must know to use the behavior correctly, including ordering, errors, lifecycle, configuration, and performance constraints when relevant; do not judge its size from the signature alone.
- Apply the deletion test before retaining a new abstraction: keep it when removing it would distribute required complexity across callers, but remove it when the complexity itself would disappear.
- Add a replaceable boundary only for demonstrated variation, ownership, or testability. One hypothetical adapter or a test-only indirection is not enough when the existing pipeline already provides a stable boundary.
- Add abstractions, state, classifications, branches, configuration, dependencies, or parallel paths only for a stated requirement, supported contract, or verified risk.
- Prefer deletion or direct replacement for unreleased code. Treat branch-local implementation and tests as disposable.
- Prefer an actionable construction- or validation-time error plus an existing alternative over partial protocol emulation.
- Keep unrelated refactors and pre-existing failures out of the patch.
- Test the required behavior, the nearest supported path, and one representative case per unsupported category rather than every constructible permutation.
- Call out changes to supported released behavior or durable formats in the plan and handoff.

## Complexity reset

Stop extending the current design when:

- Related findings keep combining the same dimensions, such as wrappers, descriptors, generics, binding, context injection, sync/async classification, or provider variants.
- The patch interprets a host-language or third-party protocol, or separately infers representations that can drift.
- A narrow requirement needs recursive resolution, cached modes, new state, or unrelated subsystem changes.
- Tests enumerate mechanics or the full diff keeps growing while the required scenario remains small.

When a trigger fires:

1. Stop editing and freeze the current revision for analysis instead of addressing comments one by one.
2. Group findings by root cause and re-read the original requirement, scope contract, and supported release or durable boundaries.
3. Write a candidate finding-derived reset spec using those inputs. Do not treat accumulated review explanations, branch-local machinery, or same-branch tests as requirements.
4. Audit the candidate spec against every affected entry point and the nearest existing supported paths. Revise it as needed, then freeze it before resuming edits.
5. Compare the complete diff with the intended merge base or latest release tag, and map each abstraction, branch, and test to the frozen spec as `retain`, `replace`, or `delete`.
6. Delete machinery with no mapping, narrow the contract, and reject unsupported cases before side effects.
7. Rebuild tests around required behavior, supported compatibility, cross-entry-point consistency, and representative unsupported categories.
8. Evaluate later findings against the frozen spec. Stop and record new contract evidence before changing the spec or widening the behavior space.

Use this compact reset spec in the plan or working notes:

```text
Finding-derived reset spec:
- Original required outcome:
- Supported release or durable boundaries:
- Grouped findings and common root cause:
- Invariants across affected entry points:
- Allowed states and behavior:
- Rejected states, failure timing, and side-effect boundary:
- Trusted and untrusted boundaries:
- Single sources of truth:
- Persistence, resume, cleanup, or other lifecycle semantics:
- Non-goals and supported alternatives:
- Representative test categories:
- Diff reset: retain / replace / delete:
```

The candidate spec is a falsifiable design hypothesis, not a record of the current implementation. The audit may correct it before it is frozen. Once frozen, require explicit evidence to revise it and re-run the complete diff mapping after any revision.

Do not wait for the user or reviewer to request this reset when the signals are already present.

## Effectiveness check

Before declaring the design complete, answer all of these with concrete evidence:

- Can the required behavior be described without naming internal helper types or reflection mechanics?
- Does the implementation reuse the nearest existing pipeline rather than maintain a parallel interpretation?
- Does every new abstraction and branch map to the scope contract or a verified risk?
- Would deleting each new abstraction merely push required complexity into multiple callers, and does each new boundary correspond to demonstrated variation, ownership, or testability?
- Are unsupported neighboring cases rejected before side effects with an existing alternative identified?
- Do tests exercise the highest stable caller boundary that reproduces the required behavior, with expected values independent of the implementation logic?
- Do the complete diff and tests cover the contract without making every constructible permutation supported?
- Does the latest review revision shrink or preserve the behavior space rather than widen it without evidence?
- When a complexity reset occurred, does every retained abstraction, branch, and test map to the frozen reset spec, with later findings classified against it?

## Guardrails-specific decision rules

- Treat released guardrail configuration, registry names, CLI flags, evaluation inputs, and MCP schemas as compatibility-sensitive external boundaries.
- Preserve documented imports from `guardrails` and caller-visible OpenAI client behavior intentionally proxied by the wrapper.
- Evaluate sync and async clients together when shared construction, forwarding, or guardrail behavior changes.
- Evaluate Chat Completions and Responses API paths together when a shared guardrail stage, context, suppression option, or exception path changes.
- Evaluate streaming and non-streaming paths together when output checks, buffering, cancellation, partial consumption, response wrapping, or early close behavior changes.
- Preserve stage ordering and the distinction between input, output, and tool guardrails. Reject invalid configuration before provider or model side effects whenever possible.
- Keep `GuardrailTripwireTriggered`, Agents SDK tripwire behavior, and suppression semantics aligned with their released contracts.
- Do not expose new behavior only through module globals. Prefer an explicit typed configuration field or public parameter with deliberate default and `None` semantics.
- Append new optional fields or constructor parameters to released public dataclasses and constructors. Do not insert them before existing fields without a compatibility layer and regression coverage for the prior positional call shape.
- Treat thresholds, model choices, and remote checks as API design. Do not silently introduce paid calls, live-service dependencies, or sensitive-data retention into an existing local or hermetic path.

## When to stop and confirm

- The change would alter supported behavior shipped in the latest release tag, or concrete evidence shows material reliance on behavior that the release incidentally accepted.
- The change would modify durable external data, protocol formats, or serialized state.
- The correct solution would materially expand beyond the requested outcome or require unrelated architectural work.
- A complexity reset trigger fires and the narrower replacement would change an already released supported contract rather than branch-local code.
- The user explicitly asked for backward compatibility, deprecation, or migration support.

## Output expectations

When this skill materially affects the implementation approach, state the decision briefly in your reasoning or handoff, for example:

- `Compatibility boundary: latest release tag v0.x.y; branch-local interface rewrite, no shim needed.`
- `Implementation scope contract: support X; preserve Y; reject Z before side effects; use supported alternative W, or none exists.`
- `Complexity reset: repeated edge-case combinations show the approach is too broad; redesign from the original requirement instead of adding another branch.`
- `Finding-derived reset spec: findings F1-F3 expose invariant X across entry points A-C; freeze that contract, delete unmapped machinery, and review later findings against it.`

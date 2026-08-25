---
name: code-change-verification
description: Run the mandatory verification stack when changes affect runtime code, tests, examples, or build and test behavior in the OpenAI Guardrails Python repository.
---

# Code Change Verification

## Overview

Mark eligible work complete only after formatting, linting, both configured type checkers, and the test suite pass. Use this skill for runtime code, tests, examples, packaging, and build or test configuration. Skip it for repository metadata or editorial docs unless the user requests the full stack.

This is a post-review final gate. When `$implementation-final-review` applies, wait for its clean-review condition before running this broad stack.

## Quick start

1. Keep this skill at `./.agents/skills/code-change-verification`.
2. Codex on macOS/Linux: `/usr/bin/env -u OPENAI_API_KEY bash .agents/skills/code-change-verification/scripts/run.sh`.
3. Other macOS/Linux environments: `bash .agents/skills/code-change-verification/scripts/run.sh`.
4. Windows: `powershell -ExecutionPolicy Bypass -File .agents/skills/code-change-verification/scripts/run.ps1`.
5. The scripts run `make format` first.
6. They then run `make lint`, `uv run mypy src tests`, `uv run pyright`, and `make tests`.
7. If any command fails, fix the issue and rerun the complete script.
8. Confirm completion only when every command succeeds and the final diff contains no formatter-created surprises.

## Start condition and host capacity

- During iterative review, use focused tests and a narrowly targeted static check only when the changed boundary requires one.
- Immediately before starting the complete stack, use available read-only task or process evidence to check whether another repository-wide test, typecheck, build, examples runner, or integration command is already active on the same host.
- When concrete contention is visible, continue useful non-heavy work and check again later. Do not create or wait on a repository lock, host-wide mutex, sentinel file, or user-triggered `finalize` step.
- Start automatically once review is clean, the diff is stable, and observable host capacity is available. Lack of host telemetry alone is not a blocker.

## Environment and API safety

The verification scripts assume repository dependencies are already installed. Run `make sync` only for a fresh checkout, after dependency files change, or when dependency resolution fails before checks start.

Repository verification and all child processes must remain in the normal Codex workspace sandbox. Never request elevated sandbox permissions for the verification wrapper, and never retry it with broader host access after a failure.

The ordinary test suite must remain hermetic. In Codex, remove inherited `OPENAI_API_KEY` from the wrapper environment. Do not run live provider calls as part of this verification stack. If a future test requires a live service, keep it out of the default stack and require explicit user approval, the repository's service-account credential boundary, and a separate report.

## Manual workflow

Run these commands from the repository root in this order:

```bash
make format
make lint
uv run mypy src tests
uv run pyright
make tests
```

Do not skip a failing step. After a fix, rerun the complete sequence so the final result applies to one stable diff.

## Resources

### `scripts/run.sh`

Runs the required macOS/Linux sequence, executing independent read-only gates concurrently after formatting and stopping sibling processes when one fails.

### `scripts/run.ps1`

Runs the same required sequence on Windows PowerShell and reports the failing command.

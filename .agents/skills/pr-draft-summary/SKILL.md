---
name: pr-draft-summary
description: Create the required PR-ready summary block, branch suggestion, title, and draft description for openai-guardrails-python after runtime, tests, examples, build/test configuration, or behavior-impacting docs change.
---

# PR Draft Summary

## Purpose

Produce a concise, copy-ready branch suggestion, PR title, and description for OpenAI Guardrails Python. This skill creates text only. It never authorizes creating a branch, committing, pushing, opening a pull request, or mutating GitHub.

## When to trigger

- Run after applicable final review and verification when the task changed `src/guardrails/`, `mcp_server/`, `tests/`, `examples/`, build/test configuration, or behavior-impacting docs.
- Run for eligible local-only or uncommitted work even when the user did not ask to open a pull request.
- Skip for repository metadata, editorial docs, conversation-only work, or when the user explicitly asks not to include a PR draft.

## Inputs to collect automatically

- Current branch: `git branch --show-current`.
- Repository state: `git status --short`.
- Untracked files: `git ls-files --others --exclude-standard`.
- Staged and unstaged paths and statistics.
- Base reference: the branch upstream when configured, otherwise local `main`, otherwise `origin/main`.
- Merge base and commits ahead of that base.
- Latest local release tag when compatibility context matters; label it potentially stale when remote tags were not refreshed.
- Category signals:
  - Runtime: `src/guardrails/`, `mcp_server/`.
  - Tests: `tests/`.
  - Examples: `examples/`.
  - Docs: `docs/`, `mkdocs.yml`.
  - Build/test configuration: `pyproject.toml`, `uv.lock`, `Makefile`, `.github/`.

Do not ask the user for information that can be derived from the repository.

## Workflow

1. Resolve the base and inspect committed, staged, unstaged, and untracked task content.
2. If there are no task changes and no commits ahead of the base, report that no code changes were detected and do not emit the PR block.
3. Classify the work as feature, fix, refactor/performance, docs-with-impact, or repository tooling.
4. Flag backward-compatibility risk only when the diff changes a released public API, external configuration, persisted data, serialized state, or wire protocol.
5. Summarize the complete change in one to three sentences. Include untracked deliverables because ordinary diff statistics omit them.
6. Choose a branch name:
   - Keep the current non-`main`, non-detached branch when it already describes the task.
   - Otherwise suggest `feat/<slug>`, `fix/<slug>`, `docs/<slug>`, or `chore/<slug>`.
   - Never suggest `HEAD`.
7. Use an imperative title with a conventional prefix when useful.
8. Start the description with `This pull request adds ...`, `fixes ...`, `improves ...`, or `updates ...`.
9. Explain the motivation, complete behavioral change, and compatibility considerations. Do not list tests unless the user asks.
10. Normalize GitHub references: `#123` for this repository and `owner/repo#123` for another repository. Remove Markdown-linked issue/PR labels, bare issue/PR URLs, local paths, Codex citations, and app directives.
11. Return the following block in English.

## Output format

```markdown
# Pull Request Draft

## Branch name suggestion

git checkout -b <kebab-case branch>

## Title

<single-line imperative title>

## Description

<copy-ready description beginning with "This pull request ...">
```

Keep the block tight and avoid repeating the same information in multiple sections.

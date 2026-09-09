# Contributing to OpenAI Guardrails Python

Thanks for contributing! This guide covers local setup and the checks expected for
changes. Read [AGENTS.md](AGENTS.md) for the repository's detailed workflow and
review policies, including the restrictions that apply to automated agents.

## Report a problem or propose a change

For bugs, include the package and Python versions, a minimal reproducible example,
and the expected and actual behavior. Remove credentials and sensitive data from
examples, logs, and configuration files.

Discuss substantial API or architectural changes with maintainers before
implementing them. Keep each contribution focused on one outcome and leave
unrelated cleanup for separate work.

For security vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening
a public issue.

## Set up your development environment

You need Python 3.11 or newer, Git, `uv`, and `make`. From your local clone, create
a linked worktree for your change, keeping the primary checkout on the default
branch:

```bash
git fetch origin
git worktree add -b your-change ../guardrails-your-change origin/main
cd ../guardrails-your-change
make sync
```

`make sync` installs the workspace packages, all extras, and development tools.
Run Python commands with `uv run` to use this environment.

Install the spaCy model used by Contains PII, as CI does:

```bash
uv pip install --python .venv/bin/python \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Ordinary tests should use mocks and local fixtures. Examples and model-backed
checks may call paid services; configure credentials only when intentionally
running those examples, and never commit them.

## Find the right place for your change

- `src/guardrails/`: Public package and runtime implementation.
- `src/guardrails/checks/`: Built-in guardrail checks.
- `src/guardrails/evals/`: Evaluation framework.
- `mcp_server/`: MCP server workspace package.
- `tests/unit/`: Isolated unit tests.
- `tests/integration/`: Cross-module tests that avoid live services by default.
- `examples/`: User-facing examples.
- `docs/`: Documentation source.

## Implement and test

Define the required behavior and compatibility boundaries before coding. Prefer
the smallest coherent change that fits the existing implementation.

- Preserve released public imports, constructor argument meanings, configuration
  formats, and the OpenAI client behavior this package proxies.
- For shared runtime changes, check affected sync and async clients, streaming
  and non-streaming responses, and Chat Completions and Responses API paths.
- Cover the required behavior and representative failures with regression tests
  at the public boundary. Keep tests deterministic and independent of live APIs.
- Use type annotations for public APIs and concise Google-style docstrings.
  Follow the Ruff and type-checker configuration in `pyproject.toml`.
- Keep secrets and sensitive payloads out of logs, exceptions, and test artifacts.

Run focused tests while iterating, for example:

```bash
uv run pytest -q tests/unit/test_types.py
```

Before handing off a code change, run the applicable checks:

```bash
make format
make lint
uv run mypy src tests
uv run pyright
make tests
make coverage
```

`make format` modifies files; inspect its changes before including them.
`make coverage` enforces the configured 95% coverage threshold.

For documentation content or structure changes, build the documentation:

```bash
make build-docs
```

For editorial-only changes, inspect the diff and run `git diff --check`.
Do not edit generated documentation. Documentation under `docs/` that describes
unreleased behavior belongs in a separately timed documentation change, so its
publication can be coordinated with the release.

## Prepare your contribution for review

Describe the problem, the resulting behavior, and the checks you ran. Reference
related issues, explain compatibility considerations, and identify any remaining
limitations. Include regression tests with behavior changes and keep the diff
free of unrelated edits.

Use concise, imperative commit subjects, such as `fix: preserve stream cleanup`.
Follow the review and verification gates in [AGENTS.md](AGENTS.md). Automated
agents must stop at the authorized local handoff and follow that file's Git and
GitHub restrictions.

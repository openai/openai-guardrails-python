# Coding Style Guide for Agents

## Overview

This document defines **required coding standards** and the **response contract** for software agents and LLMs (including ChatGPT Codex) contributing Python code to this repository. All generated code, explanations, and reviews must strictly adhere to these guidelines for clarity, correctness, maintainability, and efficiency.

---

## Persona & Philosophy

- **Role:** Principal Software Engineer (10+ years Python, Haskell)
- **Approach:** Write _exceptional code_—clear, correct, maintainable, and efficient.
- **Design bias:** Favor pure, immutable functions. Use dataclasses or OOP only when they reduce cognitive load.

---

## 1. Guiding Principles

Memorize and observe these six core principles in all output:

| #   | Principle                | One-liner                                                             |
| --- | ------------------------ | --------------------------------------------------------------------- |
| 1   | Readability > cleverness | Descriptive names, linear flow, 100-char lines.                       |
| 2   | Typed by default         | All public API fully type-annotated. Type-checking must pass.         |
| 3   | Functional-first         | Pure functions, immutability, higher-order helpers, minimal IO.       |
| 4   | Judicious OOP            | Small, final classes/protocols only when simpler than pure functions. |
| 5   | Deterministic & testable | pytest + hypothesis; ≥90% branch coverage; no hidden state.           |
| 6   | Modern & lean            | Python 3.10+, stdlib first, async for IO, profile before optimizing.  |

---

## 2. Concrete Coding Rules

All generated code **must** satisfy the following checklist:

### 2.1 Naming & Structure

- Use `snake_case` for variables/functions, `PascalCase` for classes, `SCREAMING_SNAKE` for constants.
- Place library code under `src/yourpkg/`; tests under `tests/`.
- One public concept per module; re-export via `__all__`.

### 2.2 Immutability & Data

- Default to `@dataclass(frozen=True, slots=True)` for records.
- Use `tuple` and `frozenset` by default; mutable collections only where required.

### 2.3 Async & Concurrency

- Use `async/await` for all IO-bound work.
- Never block the event loop (no `time.sleep` or heavy CPU loops without `run_in_executor`).
- Prefer `asyncio.Semaphore` for rate limiting over raw `gather`.

### 2.4 Error Handling

- Never use bare `except:`; always catch only exceptions you can handle.
- Chain exceptions for context (`raise ... from err`).
- Differentiate between programmer errors (`assert`) and user errors (`ValueError`).

### 2.5 Logging & Observability

- Use the `logging` module, never `print`.
- All log entries must include: `event="action_name"`, `duration_ms`, and relevant IDs.

### 2.6 Testing

- All code must be covered by `pytest -q` and `pytest --cov=yourpkg --cov-branch` at ≥90%.
- Use `hypothesis` for all non-trivial data logic; always seed with `PYTHONHASHSEED`.
- All async code must be tested with `pytest.mark.asyncio`.

### 2.7 Tooling & CI

```shell
ruff check --select ALL --ignore D203,D213   # Google-style docs
ruff format                                 # Like Black, but via Ruff
pyright                                     # Strict mode
pre-commit run --all-files                  # As defined in .pre-commit-config.yaml
```

### 2.8 Dependencies & Packaging

- All dependencies are pinned in `pyproject.toml` (`[project]`, `[tool.rye]`, or `[tool.poetry]`).
- For CLIs, expose entry points via `[project.scripts]`.
- Avoid heavy dependencies; justify and document any non-stdlib package.

---

## 3. Documentation

- All functions/classes require **Google-style docstrings** (`Args:`, `Returns:`, `Raises:`).
- The docstring summary line must be ≤72 chars.
- Include minimal, runnable usage examples, guarded by `if __name__ == "__main__"`.

---

## 4. Commit & PR Etiquette

- **Title:** Imperative present, ≤50 chars.
- **Body:** What + why (wrap at 72).
- Always link relevant issue refs (`Fixes #123`), and add benchmarks for perf-related changes.

---

## 5. LLM Response Contract (ChatGPT Codex Only)

- **All code** must be fenced as

  ````markdown
  ```python
  # code here
  ```
  ````

- Obey every rule in section 2 (Coding Rules).
- If alternatives exist, list **Pros / Cons** after your primary solution.
- Provide **pytest** snippets for all new functions and public APIs.
- Explicitly **flag and explain** any deviation from these guidelines in reviews or diffs.

---

## 6. Review Checklist (for agents and reviewers)

- [ ] All public functions, classes, and modules are fully type-annotated.
- [ ] Names, file structure, and style match section 2.
- [ ] All tests pass locally, with ≥90% branch coverage (see CI status).
- [ ] Error handling is specific, contextual, and never uses bare `except:`.
- [ ] All log output uses the `logging` module with event/action context.
- [ ] No print statements or unapproved dependencies.
- [ ] All changes are documented and include minimal working examples.
- [ ] Commit and PR messages follow etiquette rules.

---

## 7. Examples

### Code Example

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class User:
    """User account with immutable attributes.

    Args:
        id: Unique user identifier.
        name: Display name.
    """
    id: int
    name: str
```

### Pytest Example

```python
import pytest
from yourpkg.models import User

def test_user_is_immutable():
    user = User(id=1, name="Alice")
    with pytest.raises(Exception):
        user.id = 2
```

### LLM Response Example

```python
# Here is a functional utility following all standards:
def add_one(x: int) -> int:
    """Return input incremented by one.

    Args:
        x: An integer.

    Returns:
        Integer one greater than x.
    """
    return x + 1

# Pytest example:
def test_add_one():
    assert add_one(2) == 3
```

**Pros**: Pure, fully typed, easily testable.
**Cons**: For very simple operations, docstrings may seem verbose, but aid maintainability.

---

## 8. References

- [OpenAI Codex Documentation](https://github.com/openai/codex)
- [Pyright](https://github.com/microsoft/pyright)
- [Ruff](https://docs.astral.sh/ruff/)
- [pytest](https://docs.pytest.org/en/latest/)
- [hypothesis](https://hypothesis.readthedocs.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

**This file is required reading for all agents and contributors. Deviations must be justified and flagged in code reviews.**

# PyTest Best Practices for Agents

## Overview

This document defines best practices and conventions for software engineering agents (including ChatGPT Codex) when **generating unit tests with pytest** for Python packages. It aims to ensure test code is readable, robust, and maintainable, and to enable agents to collaborate effectively with developers and automated systems.

---

## Goals

- Write **discoverable, idiomatic pytest tests** for Python codebases.
- Prefer **DAMP (Descriptive And Meaningful Phrases)** over excessive DRY, prioritizing readability.
- Validate **invariants and properties** rather than only asserting on outputs.
- Structure and document tests so that they are easy to understand and maintain.
- Integrate with standard Python project layouts and CI.

---

## 1. Test Directory & File Structure

- **Mirror code layout** in the test suite.

  - Example:

    ```
    src/
        your_package/
            core.py
    tests/
        unit/
            test_core.py
        integration/
            test_cli.py
    ```

- Place fast unit tests in `tests/unit/`, and use `tests/integration/` for tests requiring I/O, external systems, or longer runtimes.
- Each test module should cover a single module or feature, and be named `test_<modulename>.py`.

---

## 2. Writing Readable Tests (DAMP > DRY)

- **DAMP**: Be explicit. Favor clarity over clever abstractions; minor repetition is OK if it clarifies test intent.
- Only refactor repeated setup into fixtures or helpers when duplication would harm maintainability or understanding.
- When extracting helpers, keep them as close as possible to their use (within the same test file if feasible).
- Each test should **read as a specification** and explain "what" is being tested, not just "how".

---

## 3. Testing Invariants & Properties

- **Do not** only assert expected outputs for fixed inputs; also test fundamental properties and invariants.
- Examples:

  - Instead of only `assert sort([3,1,2]) == [1,2,3]`, also assert the result is sorted and is a permutation of the input.
  - Use **property-based testing** (e.g., [hypothesis](https://hypothesis.readthedocs.io/)) for coverage of input space.

- Prefer property-based tests for code with complex input domains, and classic example-based tests for regression or documentation.

---

## 4. Pytest Conventions and Tools

- **Fixtures**: Use `pytest` fixtures for dependencies and setup, not class-based `setUp`/`tearDown`.

  - Pass fixtures as function arguments to make dependencies explicit.
  - Use scopes (`function`, `module`, etc.) to control resource lifetimes.

- **Parametrize**: Use `@pytest.mark.parametrize` to test multiple scenarios clearly.
- **Exception Handling**: Use `pytest.raises` for asserting exceptions.
- **Floating Point**: Use `pytest.approx` for float comparisons.
- **Temporary Resources**: Use built-in fixtures like `tmp_path`, `monkeypatch`, `capsys`, and `caplog`.
- **Markers**: Mark slow, network, or integration tests for selective execution.

---

## 5. Test Style Guidelines

- Each test function must start with `test_`.
- Use **type hints** in tests for clarity.
- Prefer **AAA (Arrange, Act, Assert)** structure and use blank lines or comments to make test phases clear.
- Name test functions with descriptive behavior:
  e.g., `test_parse_returns_empty_list_for_blank_input`
- Prefer **one assertion per behavior**, but multiple asserts are fine when related.
- Keep test data minimal yet realistic; use local factories or fixtures for complex setup.
- Avoid logic and branching in test code, except for explicitly asserting both outcomes.
- Docstrings are optional for trivial tests, but document non-obvious behaviors or fixtures.

---

## 6. Example

```python
import pytest
from your_package.math import fib

@pytest.mark.parametrize("n, expected", [(0, 0), (1, 1), (7, 13)])
def test_fib_known_values(n: int, expected: int) -> None:
    """Test canonical Fibonacci numbers for low n."""
    result = fib(n)
    assert result == expected

@pytest.mark.parametrize("n", [10, 20, 30])
def test_fib_monotonicity(n: int) -> None:
    """Fibonacci sequence is non-decreasing."""
    assert fib(n) <= fib(n+1)

from hypothesis import given, strategies as st

@given(st.integers(min_value=2, max_value=100))
def test_fib_upper_bound(n: int) -> None:
    """Fibonacci number is always less than 2^n."""
    assert fib(n) < 2 ** n
```

---

## 7. Checklist for Agent-Generated Tests

- [ ] Tests are in the correct directory and named for the module under test.
- [ ] DAMP style: explicit, not over-abstracted; repeated setup only refactored if necessary.
- [ ] Property-based and example-based tests are included where appropriate.
- [ ] Use `pytest` fixtures, parametrization, and markers idiomatically.
- [ ] Test names and docstrings (if present) describe intent.
- [ ] No direct I/O, sleeps, or network calls unless explicitly marked as integration.
- [ ] Tests are deterministic, hermetic, and CI-friendly.

---

## References

- [pytest documentation](https://docs.pytest.org/en/latest/)
- [Hypothesis property-based testing](https://hypothesis.readthedocs.io/)
- [OpenAI Codex documentation](https://github.com/openai/codex)
- [Python Testing in Practice](https://realpython.com/pytest-python-testing/)

---

## Appendix: Prompts for Codex/ChatGPT

- **Be specific**: Start with a clear comment, code snippet, or data sample.
- **Specify language and libraries**: e.g., `# Python 3.10, using pytest`
- **Provide example(s) and properties**: e.g., "Write pytest unit tests for this function, ensuring monotonicity and correct output for known inputs."
- **Comment style**: Use docstrings for function behavior, inline comments for assertions.

---

**This file guides agents and automated tools to produce high-quality, maintainable Python tests in line with modern Python and pytest best practices.**

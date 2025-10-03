.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format: 
	uv run ruff format
	uv run ruff check --fix

.PHONY: lint
lint: 
	uv run ruff check

.PHONY: mypy
mypy: 
	uv run mypy src
	uv run mypy tests
	uv run mypy evals

.PHONY: tests
tests: 
	uv run pytest 

.PHONY: coverage
coverage:
	
	uv run coverage run --include="src/guardrails/*" -m pytest
	uv run coverage xml --include="src/guardrails/*" -o coverage.xml
	uv run coverage report --include="src/guardrails/*" -m --fail-under=95

.PHONY: snapshots-fix
snapshots-fix: 
	uv run pytest --inline-snapshot=fix 

.PHONY: snapshots-create 
snapshots-create: 
	uv run pytest --inline-snapshot=create 

.PHONY: build-docs
build-docs:
	uv run mkdocs build
 
.PHONY: build-full-docs
build-full-docs:
	uv run docs/scripts/translate_docs.py
	uv run mkdocs build

.PHONY: serve-docs
serve-docs:
	uv run mkdocs serve
.PHONY: deploy-docs
deploy-docs:
	uv run mkdocs gh-deploy --force --verbose

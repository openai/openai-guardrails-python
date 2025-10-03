
# OpenAI Guardrails Python — Docs Hosting Setup (MkDocs → GitHub Pages)

Target site: **https://openai.github.io/openai-guardrails-python/**  
Repository: **https://github.com/openai/openai-guardrails-python**

This guide is copy‑pasteable by an AI coding IDE to create files and apply changes automatically.

---

## Overview

- Build with **MkDocs** (Material theme recommended).
- Deploy with **GitHub Pages** using **GitHub Actions**.
- Automatic deployment on push to `main` (after tests succeed).
- Optional: PR preview URLs for docs changes.

---

## 0) Prerequisites

- Python 3.10+ available in CI (we’ll use 3.11).
- MkDocs and any plugins installed in CI (local `make serve-docs` can continue to use your existing dev env).
- The repository contains your docs in a `docs/` folder (create if missing).

---

## 1) Project layout (expected)

```
openai-guardrails-python/
├─ docs/                     # Markdown docs live here
│  └─ index.md
├─ mkdocs.yml                # MkDocs configuration (see below)
└─ .github/
   └─ workflows/
      ├─ docs.yml            # Deploy workflow
      └─ docs-preview.yml    # (optional) PR preview workflow
```

If `docs/` or `.github/workflows/` folders are missing, create them.

---

## 2) MkDocs configuration

Create or update **`mkdocs.yml`** in the repo root with the following minimal content (merge with your existing file if you have one):

```yaml
site_name: OpenAI Guardrails Python
site_url: https://openai.github.io/openai-guardrails-python/
repo_url: https://github.com/openai/openai-guardrails-python
repo_name: openai/openai-guardrails-python

theme:
  name: material
  features:
    - navigation.instant
    - navigation.tracking
    - content.code.copy

nav:
  - Home: index.md

markdown_extensions:
  - admonition
  - codehilite
  - toc:
      permalink: true
```

> **Note:** If you use extra MkDocs plugins locally (e.g., `mkdocstrings[python]`, `mkdocs-section-index`, `mkdocs-literate-nav`), add them here and in the CI install step below.

---

## 3) Docs index

Create **`docs/index.md`** if missing:

```markdown
# OpenAI Guardrails Python

Welcome to the docs for **openai-guardrails-python**.
```

(Replace with your actual content.)

---

## 4) GitHub Pages settings

In the GitHub repo:

1. Go to **Settings → Pages**.
2. Set **Source** to **GitHub Actions** (not a branch).
3. No custom domain is required since we publish under `openai.github.io/openai-guardrails-python/`.

---

## 5) CI: Deploy docs to GitHub Pages

Create **`.github/workflows/docs.yml`** with the following content:

```yaml
name: Deploy docs

on:
  # Deploy after the main test workflow finishes successfully
  workflow_run:
    workflows: ["Tests"]          # <-- set to your repo's test workflow name
    branches: [ main ]
    types: [ completed ]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy_docs:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install MkDocs and plugins
        run: |
          pip install --upgrade pip
          pip install mkdocs mkdocs-material
          # Add any plugins you use:
          # pip install mkdocstrings[python] mkdocs-section-index mkdocs-literate-nav

      - name: Build site
        run: mkdocs build --strict --site-dir site

      - name: Configure Pages
        uses: actions/configure-pages@v5

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

**Alternative trigger:** If you prefer a simpler “deploy on push to `main`” flow, replace the `on:` block with:
```yaml
on:
  push:
    branches: [ main ]
  workflow_dispatch:
```

---

## 6) Local development

You can keep using your existing `make serve-docs`. A direct command is:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview.

---

## 7) Verification checklist

- Push to `main` → The **Deploy docs** workflow runs and publishes the site.
- Visit **https://openai.github.io/openai-guardrails-python/** and confirm:
  - Styles load correctly (no 404s for assets).
  - Internal links and nav work.
  - Canonical URLs include `/openai-guardrails-python/` (set via `site_url`).
- In **Settings → Pages**, you should see the latest successful deployment linked.

---

## 8) Troubleshooting

- **404s on CSS/JS or broken links under `/openai-guardrails-python/`**  
  Ensure `site_url` is set to `https://openai.github.io/openai-guardrails-python/` in `mkdocs.yml`.

- **Workflow never triggers**  
  If using `workflow_run`, confirm the referenced workflow name (e.g., `"Tests"`) matches your CI workflow’s `name:` exactly, or switch to a simple `push` trigger.

- **MkDocs plugin missing in CI**  
  Add the plugin to both `pip install ...` in the workflow and to `mkdocs.yml`.

- **Custom domain not needed**  
  We’re publishing under the project path on `openai.github.io`—no `CNAME` file is required.

---

## 9) Summary

- Keep docs in `docs/` and configure `mkdocs.yml` with the correct `site_url`.
- Use `docs.yml` to automatically build and deploy on each push (or after tests pass).

Once merged to `main`, the site will publish to:
**https://openai.github.io/openai-guardrails-python/**

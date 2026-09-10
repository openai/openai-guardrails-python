# Security Policy

## Report a vulnerability privately

For suspected security vulnerabilities in the `openai-guardrails` Python package
or this repository's code, dependencies, build, and release artifacts, follow
OpenAI's [Coordinated Vulnerability Disclosure Policy](https://openai.com/security/disclosure/)
and use the private reporting channels listed there. OpenAI's
[security.txt](https://cdn.openai.com/security.txt) lists the disclosure contact
and PGP encryption key. The disclosure policy governs program scope and eligibility.

Do not disclose suspected vulnerabilities or sensitive details in public GitHub
issues, pull requests, discussions, or logs. If you are unsure whether an issue
is security-related, use the private disclosure channel first.

## What to include

Identify the repository as `openai/openai-guardrails-python` and include, when available:

- The installed `openai-guardrails` version or affected commit, Python version,
  operating system, and relevant dependency versions (such as `openai` or the
  Agents SDK).
- The affected component and security impact, including the expected and observed
  behavior and any prerequisites.
- A minimal example using synthetic data and a sanitized guardrail configuration.
  Include the affected check, stage, and relevant settings, such as thresholds
  and tripwire suppression.
- The relevant execution path: sync or async, streaming or non-streaming,
  Chat Completions or Responses, OpenAI or Azure, or an Agents SDK integration.
  For evaluation or tooling issues, identify the command and input format instead.
- Sanitized diagnostic output needed to understand the issue. A partial report
  is welcome; do not delay reporting to gather every detail or test more systems.

## Redact sensitive data

Before sharing code, configurations, logs, tracebacks, screenshots, or evaluation
artifacts, inspect them for credentials and private content:

- Remove API keys, access tokens, authorization headers, cookies, passwords,
  private keys, and secrets in environment variables or URLs.
- Replace personal or customer data, confidential prompts and model outputs,
  conversation history, tool arguments and results, and evaluation datasets with
  synthetic placeholders.
- Redact private endpoints, account and project identifiers, local paths, and
  proprietary configuration details unless essential to explain the issue.

Preserve the structure needed to understand the report and describe what was
redacted. Do not attach raw `.env` files, credential stores, production logs, or
full request/response dumps. If sensitive material is essential, describe that
need in the private report and agree on secure handling before sending it.
Revoke or rotate any exposed credentials promptly; deleting them from a report
does not invalidate them.

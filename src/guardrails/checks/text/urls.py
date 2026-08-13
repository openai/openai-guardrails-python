"""URL detection guardrail.

This guardrail detects URLs in text and validates them against an allow list of
permitted domains, IP addresses, and full URLs. It provides security features
to prevent credential injection, typosquatting attacks, and unauthorized schemes.

The guardrail uses regex patterns for URL detection and Pydantic for robust
URL parsing and validation.

Example Usage:
    Default configuration:
        config = URLConfig(url_allow_list=["example.com"])

    Custom configuration:
        config = URLConfig(
            url_allow_list=["company.com", "10.0.0.0/8"],
            allowed_schemes={"http", "https"},
            allow_subdomains=True
        )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from ipaddress import AddressValueError, ip_address, ip_network
from typing import Any
from urllib.parse import ParseResult, urlparse

from pydantic import BaseModel, Field, field_validator

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["urls"]

DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
}

SCHEME_PREFIX_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")
WWW_PREFIX = "www."


def _strip_www_prefix(host: str) -> str:
    """Remove a single leading ``www.`` label from a host."""
    return host.removeprefix(WWW_PREFIX)


@dataclass(frozen=True, slots=True)
class UrlDetectionResult:
    """Result structure for URL detection and filtering."""

    detected: list[str]
    allowed: list[str]
    blocked: list[str]
    blocked_reasons: list[str] = field(default_factory=list)


class URLConfig(BaseModel):
    """Direct URL configuration with explicit parameters."""

    url_allow_list: list[str] = Field(
        default_factory=list,
        description="Allowed URLs, domains, or IP addresses",
    )
    allowed_schemes: set[str] = Field(
        default={"https"},
        description="Allowed URL schemes/protocols (default: HTTPS only for security)",
    )
    block_userinfo: bool = Field(
        default=True,
        description="Block URLs with userinfo (user:pass@domain) to prevent credential injection",
    )
    allow_subdomains: bool = Field(
        default=False,
        description="Allow subdomains of allowed domains (e.g. api.example.com if example.com is allowed)",
    )

    @field_validator("allowed_schemes", mode="before")
    @classmethod
    def normalize_allowed_schemes(cls, value: Any) -> set[str]:
        """Normalize allowed schemes to bare identifiers without delimiters."""
        if value is None:
            return {"https"}

        if isinstance(value, str):
            raw_values = [value]
        else:
            raw_values = list(value)

        normalized: set[str] = set()
        for entry in raw_values:
            if not isinstance(entry, str):
                raise TypeError("allowed_schemes entries must be strings")
            cleaned = entry.strip().lower()
            if not cleaned:
                continue
            if cleaned.endswith("://"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.removesuffix(":")
            if cleaned:
                normalized.add(cleaned)

        if not normalized:
            raise ValueError("allowed_schemes must include at least one scheme")

        return normalized


def _detect_urls(text: str) -> list[str]:
    """Detect URLs using regex patterns with deduplication."""
    PUNCTUATION_CLEANUP = r"[.,;:!?)\]]+$"

    detected_urls = []
    scheme_patterns = [
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        r'ftp://[^\s<>"{}|\\^`\[\]]+',
        r'data:[^\s<>"{}|\\^`\[\]]+',
        r'javascript:[^\s<>"{}|\\^`\[\]]+',
        r'vbscript:[^\s<>"{}|\\^`\[\]]+',
    ]

    scheme_urls = set()
    for pattern in scheme_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = re.sub(PUNCTUATION_CLEANUP, "", match)
            if cleaned:
                detected_urls.append(cleaned)
                if "://" in cleaned:
                    domain_part = cleaned.split("://", 1)[1].split("/")[0].split("?")[0].split("#")[0]
                    scheme_urls.add(domain_part.lower())

    domain_pattern = r"\b(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}(?:/[^\s]*)?"
    domain_matches = re.findall(domain_pattern, text, re.IGNORECASE)
    for match in domain_matches:
        cleaned = re.sub(PUNCTUATION_CLEANUP, "", match)
        if cleaned:
            domain_part = cleaned.split("/")[0].split("?")[0].split("#")[0].lower()
            if domain_part not in scheme_urls:
                detected_urls.append(cleaned)

    ip_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]+)?(?:/[^\s]*)?"
    ip_matches = re.findall(ip_pattern, text, re.IGNORECASE)
    for match in ip_matches:
        cleaned = re.sub(PUNCTUATION_CLEANUP, "", match)
        if cleaned:
            ip_part = cleaned.split("/")[0].split("?")[0].split("#")[0].lower()
            if ip_part not in scheme_urls:
                detected_urls.append(cleaned)

    final_urls = []
    scheme_url_domains = set()
    for url in detected_urls:
        if "://" in url:
            try:
                parsed = urlparse(url)
                if parsed.hostname:
                    scheme_url_domains.add(parsed.hostname.lower())
                    bare_domain = _strip_www_prefix(parsed.hostname.lower())
                    scheme_url_domains.add(bare_domain)
            except (ValueError, UnicodeError):
                pass
            final_urls.append(url)

    for url in detected_urls:
        if "://" not in url:
            url_lower = _strip_www_prefix(url.lower())
            if url_lower not in scheme_url_domains:
                final_urls.append(url)

    return list(dict.fromkeys([url for url in final_urls if url]))


def _validate_url_security(url_string: str, config: URLConfig) -> tuple[ParseResult | None, str, bool]:
    """Validate URL security properties using urllib.parse."""
    try:
        has_explicit_scheme = False
        if "://" in url_string:
            parsed_url = urlparse(url_string)
            original_scheme = parsed_url.scheme
            has_explicit_scheme = True
        elif ":" in url_string and url_string.split(":", 1)[0] in {"data", "javascript", "vbscript", "mailto"}:
            parsed_url = urlparse(url_string)
            original_scheme = parsed_url.scheme
            has_explicit_scheme = True
        else:
            parsed_url = urlparse(f"http://{url_string}")
            original_scheme = None
            has_explicit_scheme = False

        if not parsed_url.scheme:
            return None, "Invalid URL format", False

        special_schemes = {"data", "javascript", "vbscript", "mailto"}
        if parsed_url.scheme not in special_schemes and not parsed_url.netloc:
            return None, "Invalid URL format", False

        if has_explicit_scheme and original_scheme not in config.allowed_schemes:
            return None, f"Blocked scheme: {original_scheme}", has_explicit_scheme

        if config.block_userinfo and (parsed_url.username or parsed_url.password):
            return None, "Contains userinfo (potential credential injection)", has_explicit_scheme

        return parsed_url, "", has_explicit_scheme

    except (ValueError, UnicodeError, AttributeError) as e:
        return None, f"Invalid URL format: {str(e)}", False
    except Exception as e:
        return None, f"URL parsing error: {type(e).__name__}: {str(e)}", False


def _safe_get_port(parsed: ParseResult, scheme: str) -> int | None:
    """Safely extract port from ParseResult, handling malformed ports."""
    try:
        return parsed.port or DEFAULT_PORTS.get(scheme.lower())
    except ValueError:
        return None


def _is_url_allowed(
    parsed_url: ParseResult,
    allow_list: list[str],
    allow_subdomains: bool,
    url_had_explicit_scheme: bool,
) -> bool:
    """Check if parsed URL matches any entry in the allow list."""
    if not allow_list:
        return False

    url_host = parsed_url.hostname
    if not url_host:
        return False

    url_host = url_host.lower()
    url_domain = _strip_www_prefix(url_host)
    scheme_lower = parsed_url.scheme.lower() if parsed_url.scheme else ""
    url_port = _safe_get_port(parsed_url, scheme_lower)
    try:
        _ = parsed_url.port
    except ValueError:
        return False
    url_path = parsed_url.path or "/"
    url_query = parsed_url.query
    url_fragment = parsed_url.fragment

    try:
        url_ip = ip_address(url_host)
    except (AddressValueError, ValueError):
        url_ip = None

    for allowed_entry in allow_list:
        allowed_entry = allowed_entry.lower().strip()

        has_explicit_scheme = bool(SCHEME_PREFIX_RE.match(allowed_entry))
        if has_explicit_scheme:
            parsed_allowed = urlparse(allowed_entry)
        else:
            parsed_allowed = urlparse(f"//{allowed_entry}")
        allowed_host = (parsed_allowed.hostname or "").lower()
        allowed_scheme = parsed_allowed.scheme.lower() if parsed_allowed.scheme else ""
        try:
            allowed_port_explicit = parsed_allowed.port
        except ValueError:
            allowed_port_explicit = None
        allowed_port = _safe_get_port(parsed_allowed, allowed_scheme)
        allowed_path = parsed_allowed.path
        allowed_query = parsed_allowed.query
        allowed_fragment = parsed_allowed.fragment

        try:
            allowed_ip = ip_address(allowed_host)
        except (AddressValueError, ValueError):
            allowed_ip = None

        if allowed_ip is not None:
            if url_ip is None:
                continue
            if has_explicit_scheme and url_had_explicit_scheme and allowed_scheme and allowed_scheme != scheme_lower:
                continue
            if allowed_port_explicit is not None and allowed_port != url_port:
                continue
            if allowed_ip == url_ip:
                return True

            network_spec = allowed_host
            if parsed_allowed.path not in ("", "/"):
                network_spec = f"{network_spec}{parsed_allowed.path}"
            try:
                if network_spec and "/" in network_spec and url_ip in ip_network(network_spec, strict=False):
                    return True
            except (AddressValueError, ValueError):
                pass
            continue

        if not allowed_host:
            continue

        allowed_domain = _strip_www_prefix(allowed_host)

        if allowed_port_explicit is not None and allowed_port != url_port:
            continue

        host_matches = url_domain == allowed_domain or (allow_subdomains and url_domain.endswith(f".{allowed_domain}"))
        if not host_matches:
            continue

        if has_explicit_scheme and url_had_explicit_scheme and allowed_scheme and allowed_scheme != scheme_lower:
            continue

        if allowed_path not in ("", "/"):
            normalized_allowed_path = allowed_path.rstrip("/")
            if url_path != allowed_path and url_path != normalized_allowed_path and not url_path.startswith(f"{normalized_allowed_path}/"):
                continue

        if allowed_query and allowed_query != url_query:
            continue

        if allowed_fragment and allowed_fragment != url_fragment:
            continue

        return True

    return False


async def urls(ctx: Any, data: str, config: URLConfig) -> GuardrailResult:
    """Detects URLs using regex patterns, validates them with Pydantic, and checks against the allow list."""
    _ = ctx
    detected_urls = _detect_urls(data)

    allowed, blocked = [], []
    blocked_reasons = []

    for url_string in detected_urls:
        parsed_url, error_reason, url_had_explicit_scheme = _validate_url_security(url_string, config)

        if parsed_url is None:
            blocked.append(url_string)
            blocked_reasons.append(f"{url_string}: {error_reason}")
            continue

        hostless_schemes = {"data", "javascript", "vbscript", "mailto"}
        if parsed_url.scheme in hostless_schemes:
            allowed.append(url_string)
        elif _is_url_allowed(parsed_url, config.url_allow_list, config.allow_subdomains, url_had_explicit_scheme):
            allowed.append(url_string)
        else:
            blocked.append(url_string)
            blocked_reasons.append(f"{url_string}: Not in allow list")

    return GuardrailResult(
        tripwire_triggered=bool(blocked),
        info={
            "guardrail_name": "URL Filter",
            "config": {
                "allowed_schemes": list(config.allowed_schemes),
                "block_userinfo": config.block_userinfo,
                "allow_subdomains": config.allow_subdomains,
                "url_allow_list": config.url_allow_list,
            },
            "detected": detected_urls,
            "allowed": allowed,
            "blocked": blocked,
            "blocked_reasons": blocked_reasons,
        },
    )


default_spec_registry.register(
    name="URL Filter",
    check_fn=urls,
    description="URL filtering using regex + Pydantic with direct configuration.",
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="RegEx"),
)

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

from pydantic import BaseModel, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["urls"]


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


def _detect_urls(text: str) -> list[str]:
    """Detect URLs using regex."""
    # Pattern for cleaning trailing punctuation (] must be escaped)
    PUNCTUATION_CLEANUP = r"[.,;:!?)\]]+$"

    detected_urls = []

    # Pattern 1: URLs with schemes (highest priority)
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
            # Clean trailing punctuation
            cleaned = re.sub(PUNCTUATION_CLEANUP, "", match)
            if cleaned:
                detected_urls.append(cleaned)
                # Track the domain part to avoid duplicates
                if "://" in cleaned:
                    domain_part = cleaned.split("://", 1)[1].split("/")[0].split("?")[0].split("#")[0]
                    scheme_urls.add(domain_part.lower())

    # Pattern 2: Domain-like patterns (scheme-less) - but skip if already found with scheme
    domain_pattern = r"\b(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}(?:/[^\s]*)?"
    domain_matches = re.findall(domain_pattern, text, re.IGNORECASE)

    for match in domain_matches:
        # Clean trailing punctuation
        cleaned = re.sub(PUNCTUATION_CLEANUP, "", match)
        if cleaned:
            # Extract just the domain part for comparison
            domain_part = cleaned.split("/")[0].split("?")[0].split("#")[0].lower()
            # Only add if we haven't already found this domain with a scheme
            if domain_part not in scheme_urls:
                detected_urls.append(cleaned)

    # Pattern 3: IP addresses - similar deduplication
    ip_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]+)?(?:/[^\s]*)?"
    ip_matches = re.findall(ip_pattern, text, re.IGNORECASE)

    for match in ip_matches:
        # Clean trailing punctuation
        cleaned = re.sub(PUNCTUATION_CLEANUP, "", match)
        if cleaned:
            # Extract IP part for comparison
            ip_part = cleaned.split("/")[0].split("?")[0].split("#")[0].lower()
            if ip_part not in scheme_urls:
                detected_urls.append(cleaned)

    # Advanced deduplication: Remove domains that are already part of full URLs
    final_urls = []
    scheme_url_domains = set()

    # First pass: collect all domains from scheme-ful URLs
    for url in detected_urls:
        if "://" in url:
            try:
                parsed = urlparse(url)
                if parsed.hostname:
                    scheme_url_domains.add(parsed.hostname.lower())
                    # Also add www-stripped version
                    bare_domain = parsed.hostname.lower().replace("www.", "")
                    scheme_url_domains.add(bare_domain)
            except (ValueError, UnicodeError):
                # Skip URLs with parsing errors (malformed URLs, encoding issues)
                # This is expected for edge cases and doesn't require logging
                pass
            final_urls.append(url)

    # Second pass: only add scheme-less URLs if their domain isn't already covered
    for url in detected_urls:
        if "://" not in url:
            # Check if this domain is already covered by a full URL
            url_lower = url.lower().replace("www.", "")
            if url_lower not in scheme_url_domains:
                final_urls.append(url)

    # Remove empty URLs and return unique list
    return list(dict.fromkeys([url for url in final_urls if url]))


def _validate_url_security(url_string: str, config: URLConfig) -> tuple[ParseResult | None, str]:
    """Validate URL using stdlib urllib.parse."""
    try:
        # Parse URL - preserve original scheme for validation
        if "://" in url_string:
            # Standard URL with double-slash scheme (http://, https://, ftp://, etc.)
            parsed_url = urlparse(url_string)
            original_scheme = parsed_url.scheme
        elif ":" in url_string and url_string.split(":", 1)[0] in {"data", "javascript", "vbscript", "mailto"}:
            # Special single-colon schemes
            parsed_url = urlparse(url_string)
            original_scheme = parsed_url.scheme
        else:
            # Add http scheme for parsing, but remember this is a default
            parsed_url = urlparse(f"http://{url_string}")
            original_scheme = "http"  # Default scheme for scheme-less URLs

        # Basic validation: must have scheme and netloc (except for special schemes)
        if not parsed_url.scheme:
            return None, "Invalid URL format"

        # Special schemes like data: and javascript: don't need netloc
        special_schemes = {"data", "javascript", "vbscript", "mailto"}
        if original_scheme not in special_schemes and not parsed_url.netloc:
            return None, "Invalid URL format"

        # Security validations - use original scheme
        if original_scheme not in config.allowed_schemes:
            return None, f"Blocked scheme: {original_scheme}"

        if config.block_userinfo and parsed_url.username:
            return None, "Contains userinfo (potential credential injection)"

        # Everything else (IPs, localhost, private IPs) goes through allow list logic
        return parsed_url, ""

    except (ValueError, UnicodeError, AttributeError) as e:
        # Common URL parsing errors:
        # - ValueError: Invalid URL structure, invalid port, etc.
        # - UnicodeError: Invalid encoding in URL
        # - AttributeError: Unexpected URL structure
        return None, f"Invalid URL format: {str(e)}"
    except Exception as e:
        # Catch any unexpected errors but provide debugging info
        return None, f"URL parsing error: {type(e).__name__}: {str(e)}"


def _is_url_allowed(parsed_url: ParseResult, allow_list: list[str], allow_subdomains: bool) -> bool:
    """Check if URL is allowed."""
    if not allow_list:
        return False

    url_host = parsed_url.hostname
    if not url_host:
        return False

    url_host = url_host.lower()

    for allowed_entry in allow_list:
        allowed_entry = allowed_entry.lower().strip()

        # Handle IP addresses and CIDR blocks
        try:
            ip_address(allowed_entry.split("/")[0])
            if allowed_entry == url_host or ("/" in allowed_entry and ip_address(url_host) in ip_network(allowed_entry, strict=False)):
                return True
            continue
        except (AddressValueError, ValueError):
            pass

        # Handle domain matching
        allowed_domain = allowed_entry.replace("www.", "")
        url_domain = url_host.replace("www.", "")

        # Exact match always allowed
        if url_domain == allowed_domain:
            return True

        # Subdomain matching if enabled
        if allow_subdomains and url_domain.endswith(f".{allowed_domain}"):
            return True

    return False


async def urls(ctx: Any, data: str, config: URLConfig) -> GuardrailResult:
    """Detects URLs using regex patterns, validates them with Pydantic, and checks against the allow list.

    Args:
        ctx: Context object.
        data: Text to scan for URLs.
        config: Configuration object.
    """
    _ = ctx

    # Detect URLs using regex patterns
    detected_urls = _detect_urls(data)

    allowed, blocked = [], []
    blocked_reasons = []

    for url_string in detected_urls:
        # Validate URL with security checks
        parsed_url, error_reason = _validate_url_security(url_string, config)

        if parsed_url is None:
            blocked.append(url_string)
            blocked_reasons.append(f"{url_string}: {error_reason}")
            continue

        # Check against allow list
        # Special schemes (data:, javascript:, mailto:) don't have meaningful hosts
        # so they only need scheme validation, not host-based allow list checking
        hostless_schemes = {"data", "javascript", "vbscript", "mailto"}
        if parsed_url.scheme in hostless_schemes:
            # For hostless schemes, only scheme permission matters (no allow list needed)
            # They were already validated for scheme permission in _validate_url_security
            allowed.append(url_string)
        elif _is_url_allowed(parsed_url, config.url_allow_list, config.allow_subdomains):
            allowed.append(url_string)
        else:
            blocked.append(url_string)
            blocked_reasons.append(f"{url_string}: Not in allow list")

    return GuardrailResult(
        tripwire_triggered=bool(blocked),
        info={
            "guardrail_name": "URL Filter (Direct Config)",
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
            "checked_text": data,
        },
    )


# Register the URL filter
default_spec_registry.register(
    name="URL Filter",
    check_fn=urls,
    description="URL filtering using regex + Pydantic with direct configuration.",
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="RegEx"),
)

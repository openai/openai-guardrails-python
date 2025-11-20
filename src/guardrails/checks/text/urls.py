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
            # Support inputs like "https://", "HTTPS:", or " https "
            if cleaned.endswith("://"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.removesuffix(":")
            if cleaned:
                normalized.add(cleaned)

        if not normalized:
            raise ValueError("allowed_schemes must include at least one scheme")

        return normalized


def _detect_urls(text: str) -> list[str]:
    """Detect URLs using regex patterns with deduplication.

    Detects URLs with explicit schemes (http, https, ftp, data, javascript,
    vbscript), domain-like patterns without schemes, and IP addresses.
    Deduplicates to avoid returning both scheme-ful and scheme-less versions
    of the same URL.

    Args:
        text: The text to scan for URLs.

    Returns:
        List of unique URL strings found in the text, with trailing
        punctuation removed.
    """
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
    """Validate URL security properties using urllib.parse.

    Checks URL structure, validates the scheme is allowed, and ensures no
    credentials are embedded in userinfo if block_userinfo is enabled.

    Args:
        url_string: The URL string to validate.
        config: Configuration specifying allowed schemes and userinfo policy.

    Returns:
        A tuple of (parsed_url, error_reason). If validation succeeds,
        parsed_url is a ParseResult and error_reason is empty. If validation
        fails, parsed_url is None and error_reason describes the failure.
    """
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

        if config.block_userinfo and (parsed_url.username or parsed_url.password):
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


def _safe_get_port(parsed: ParseResult, scheme: str) -> int | None:
    """Safely extract port from ParseResult, handling malformed ports.

    Args:
        parsed: The parsed URL.
        scheme: The URL scheme (for default port lookup).

    Returns:
        The port number, the default port for the scheme, or None if invalid.
    """
    try:
        return parsed.port or DEFAULT_PORTS.get(scheme.lower())
    except ValueError:
        # Port is out of range (0-65535) or malformed
        return None


def _is_url_allowed(parsed_url: ParseResult, allow_list: list[str], allow_subdomains: bool) -> bool:
    """Check if parsed URL matches any entry in the allow list.

    Supports domain names, IP addresses, CIDR blocks, and full URLs with
    paths/ports/query strings. Allow list entries without explicit schemes
    match any scheme. Entries with schemes must match exactly.

    Args:
        parsed_url: The parsed URL to check.
        allow_list: List of allowed URL patterns (domains, IPs, CIDR, full URLs).
        allow_subdomains: If True, subdomains of allowed domains are permitted.

    Returns:
        True if the URL matches any allow list entry, False otherwise.
    """
    if not allow_list:
        return False

    url_host = parsed_url.hostname
    if not url_host:
        return False

    url_host = url_host.lower()
    url_domain = url_host.replace("www.", "")
    scheme_lower = parsed_url.scheme.lower() if parsed_url.scheme else ""
    url_port = _safe_get_port(parsed_url, scheme_lower)
    # If port is invalid (None from _safe_get_port due to ValueError), reject the URL
    if url_port is None and parsed_url.netloc and ":" in parsed_url.netloc:
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
        allowed_port = _safe_get_port(parsed_allowed, allowed_scheme)
        allowed_path = parsed_allowed.path
        allowed_query = parsed_allowed.query
        allowed_fragment = parsed_allowed.fragment

        # Handle IP addresses and CIDR blocks (including schemes)
        try:
            allowed_ip = ip_address(allowed_host)
        except (AddressValueError, ValueError):
            allowed_ip = None

        if allowed_ip is not None:
            if url_ip is None:
                continue
            # Scheme matching for IPs: if allow list entry has explicit scheme, it must match exactly
            if has_explicit_scheme:
                allowed_scheme = parsed_allowed.scheme.lower() if parsed_allowed.scheme else ""
                if allowed_scheme and allowed_scheme != scheme_lower:
                    continue
            if allowed_port is not None and allowed_port != url_port:
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
                # Path segment might not represent a CIDR mask; ignore.
                pass
            continue

        if not allowed_host:
            continue

        allowed_domain = allowed_host.replace("www.", "")

        if allowed_port is not None and allowed_port != url_port:
            continue

        host_matches = url_domain == allowed_domain or (
            allow_subdomains and url_domain.endswith(f".{allowed_domain}")
        )
        if not host_matches:
            continue

        # Scheme matching: if allow list entry has explicit scheme, it must match exactly
        if has_explicit_scheme:
            allowed_scheme = parsed_allowed.scheme.lower() if parsed_allowed.scheme else ""
            if allowed_scheme and allowed_scheme != scheme_lower:
                continue

        # Path matching with segment boundary respect
        if allowed_path not in ("", "/"):
            # Ensure path matching respects segment boundaries to prevent
            # "/api" from matching "/api2" or "/api-v2"
            if url_path != allowed_path and not url_path.startswith(f"{allowed_path}/"):
                continue

        if allowed_query and allowed_query != url_query:
            continue

        if allowed_fragment and allowed_fragment != url_fragment:
            continue

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


# Register the URL filter
default_spec_registry.register(
    name="URL Filter",
    check_fn=urls,
    description="URL filtering using regex + Pydantic with direct configuration.",
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="RegEx"),
)

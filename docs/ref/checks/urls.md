# URL Filter

Advanced URL detection and filtering guardrail that prevents access to unauthorized domains. Uses comprehensive regex patterns and robust URL parsing to detect various URL formats, validates them against security policies, and filters based on a configurable allow list.

**Key Security Features:**

- Prevents credential injection attacks (`user:pass@domain`)
- Blocks typosquatting and look-alike domains  
- Restricts dangerous schemes (`javascript:`, `data:`)
- Supports IP addresses and CIDR ranges
- Configurable subdomain matching

## Configuration

```json
{
    "name": "URL Filter",
    "config": {
        "url_allow_list": ["example.com", "192.168.1.100", "https://api.service.com/v1"],
        "allowed_schemes": ["https"],
        "block_userinfo": true,
        "allow_subdomains": false
    }
}
```

### Parameters

- **`url_allow_list`** (optional): List of allowed domains, IP addresses, CIDR ranges, or full URLs.
    - Default: `[]` (blocks all URLs)

- **`allowed_schemes`** (optional): Set of allowed URL schemes/protocols. 
    - Default: `["https"]` (HTTPS-only for security)

- **`block_userinfo`** (optional): Whether to block URLs containing userinfo (`user:pass@domain`) to prevent credential injection attacks.
    - `true` (default): Blocks URLs containing userinfo
    - `false`: Allows URLs containing userinfo


- **`allow_subdomains`** (optional): Whether to allow subdomains of allowed domains.
    - `false` (default): Only exact domain matches (e.g., `example.com` allows `example.com` and `www.example.com`)
    - `true`: Allows subdomains (e.g., `example.com` allows `api.example.com`)

## Implementation Notes

- Detects URLs, domains, and IP addresses using regex patterns
- Validates URL schemes and security policies  
- Supports exact domain matching or subdomain inclusion
- Handles IP addresses and CIDR ranges

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "URL Filter (Direct Config)",
    "config": {
        "allowed_schemes": ["https"],
        "block_userinfo": true,
        "allow_subdomains": false,
        "url_allow_list": ["example.com"]
    },
    "detected": ["https://example.com", "https://user:pass@malicious.com"],
    "allowed": ["https://example.com"],
    "blocked": ["https://user:pass@malicious.com"],
    "blocked_reasons": ["https://user:pass@malicious.com: Contains userinfo (potential credential injection)"],
    "checked_text": "Visit https://example.com or login at https://user:pass@malicious.com"
}
```

### Response Fields

- **`guardrail_name`**: Name of the guardrail that was executed
- **`config`**: Applied configuration including allow list, schemes, userinfo blocking, and subdomain settings
- **`detected`**: All URLs detected in the text using regex patterns
- **`allowed`**: URLs that passed all security checks and allow list validation
- **`blocked`**: URLs that were blocked due to security policies or allow list restrictions
- **`blocked_reasons`**: Detailed explanations for why each URL was blocked
- **`checked_text`**: Original input text that was scanned
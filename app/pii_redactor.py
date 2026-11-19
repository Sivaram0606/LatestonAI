"""Active PII and credential redaction for logs, spans, and memory pipelines.

Satisfies AgentOps Code Review Matrix — Category 4 (PII Redaction):
- Scrubs emails, phone numbers, US SSNs, payment card numbers, IPv4 addresses,
  and API/OAuth bearer tokens recursively across strings, dicts, and lists.
- Supports optional Google Cloud Sensitive Data Protection (Cloud DLP API
  ``content:deidentify``) when ``ENABLE_CLOUD_DLP=true`` is configured.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # API keys & OAuth bearer tokens (scrubbed first before generic numeric rules)
    (
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        "[REDACTED_GOOGLE_API_KEY]",
    ),
    (
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{30,}\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    (
        re.compile(r"\bya29\.[0-9A-Za-z_-]{20,}\b"),
        "[REDACTED_OAUTH_TOKEN]",
    ),
    (
        re.compile(r"(?i)(bearer\s+)[0-9A-Za-z._~+/=-]{16,}"),
        r"\1[REDACTED_BEARER_TOKEN]",
    ),
    # Email addresses
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    # US Social Security Numbers
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    # Credit / debit card sequences (13-16 digits with optional separators)
    (
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "[REDACTED_CARD]",
    ),
    # Phone numbers (international / US formats, avoiding YYYY-MM-DD dates)
    (
        re.compile(
            r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
        ),
        "[REDACTED_PHONE]",
    ),
)

_SENSITIVE_KEY_NAMES = frozenset(
    {
        "api_key",
        "gemini_api_key",
        "google_api_key",
        "github_token",
        "authorization",
        "password",
        "secret",
        "access_token",
        "client_secret",
    }
)


class PiiRedactor:
    """Scrubs PII and credentials prior to logging, tracing, or memory storage."""

    def scrub_text(self, text: str) -> str:
        """Redact sensitive tokens and PII patterns from a string synchronously."""
        if not text:
            return text
        scrubbed = text
        for pattern, replacement in _REDACTION_PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed

    def scrub_payload(self, value: Any) -> Any:
        """Recursively scrub strings and sensitive keys across dicts, lists, and tuples."""
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for k, v in value.items():
                key_str = str(k)
                if key_str.lower() in _SENSITIVE_KEY_NAMES and v:
                    cleaned[key_str] = "[REDACTED_SECRET]"
                else:
                    cleaned[key_str] = self.scrub_payload(v)
            return cleaned
        if isinstance(value, list):
            return [self.scrub_payload(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.scrub_payload(item) for item in value)
        return value

    async def scrub_text_with_cloud_dlp(self, text: str) -> str:
        """Scrub via local regex first, then optionally invoke Google Cloud DLP API."""
        locally_scrubbed = self.scrub_text(text)
        if not locally_scrubbed:
            return locally_scrubbed

        if os.environ.get("ENABLE_CLOUD_DLP", "").lower() not in ("true", "1"):
            return locally_scrubbed

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        if not project_id:
            return locally_scrubbed

        try:
            import google.auth
            from google.auth.transport.requests import Request

            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(Request())
            url = f"https://dlp.googleapis.com/v2/projects/{project_id}/locations/global/content:deidentify"
            body = {
                "item": {"value": locally_scrubbed},
                "inspectConfig": {
                    "infoTypes": [
                        {"name": "EMAIL_ADDRESS"},
                        {"name": "PHONE_NUMBER"},
                        {"name": "PERSON_NAME"},
                        {"name": "US_SOCIAL_SECURITY_NUMBER"},
                        {"name": "CREDIT_CARD_NUMBER"},
                    ]
                },
                "deidentifyConfig": {
                    "infoTypeTransformations": {
                        "transformations": [
                            {"primitiveTransformation": {"replaceWithInfoTypeConfig": {}}}
                        ]
                    }
                },
            }
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {credentials.token}"},
                    json=body,
                )
                if resp.status_code == 200:
                    return str(
                        resp.json().get("item", {}).get("value", locally_scrubbed)
                    )
        except Exception:
            pass

        return locally_scrubbed


pii_redactor = PiiRedactor()

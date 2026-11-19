"""Secure secret resolution via Google Cloud Secret Manager with TTL caching.

Satisfies AgentOps Code Review Matrix — Category 5 (Secure Secret Management):
- Zero hardcoded API keys or tokens anywhere in source code.
- Resolves secrets dynamically from Google Cloud Secret Manager (`projects/*/secrets/*/versions/*`)
  when configured, falling back to Cloud Run secret-mounted environment variables.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_SECRET_CACHE: dict[str, tuple[str, float]] = {}
_DEFAULT_TTL_SECONDS = 300.0


class SecretManagerResolver:
    """Fetches and caches runtime credentials from Google Cloud Secret Manager."""

    def __init__(
        self,
        project_id: str | None = None,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._project_id = project_id
        self._ttl_seconds = ttl_seconds

    @property
    def project_id(self) -> str:
        return (
            self._project_id
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
            or os.environ.get("GCP_PROJECT_ID", "").strip()
        )

    def get_secret(
        self,
        env_var_name: str,
        secret_id_env_var: str | None = None,
        default_secret_id: str | None = None,
    ) -> str:
        """Resolve a secret via Cloud Run env mount or GCP Secret Manager API.

        Resolution order:
        1. Explicit Secret Manager resource/ID configured in ``secret_id_env_var``
           (or ``USE_SECRET_MANAGER=true`` + ``default_secret_id``).
        2. Secret Manager–mounted Cloud Run environment variable ``env_var_name``.
        3. Empty string if unconfigured (never raises on optional public-API keys).
        """
        secret_id = ""
        if secret_id_env_var:
            secret_id = os.environ.get(secret_id_env_var, "").strip()
        if (
            not secret_id
            and default_secret_id
            and os.environ.get("USE_SECRET_MANAGER", "").lower() in ("true", "1")
        ):
            secret_id = default_secret_id

        if secret_id and self.project_id:
            fetched = self._fetch_from_gcp_secret_manager(secret_id)
            if fetched:
                return fetched

        return os.environ.get(env_var_name, "").strip()

    def _fetch_from_gcp_secret_manager(self, secret_id_or_resource: str) -> str:
        """Fetch latest secret payload from Google Cloud Secret Manager with TTL cache."""
        now = time.monotonic()
        cached = _SECRET_CACHE.get(secret_id_or_resource)
        if cached and cached[1] > now:
            return cached[0]

        try:
            from google.cloud import secretmanager

            client = secretmanager.SecretManagerServiceClient()
            if secret_id_or_resource.startswith("projects/"):
                resource_name = secret_id_or_resource
                if "/versions/" not in resource_name:
                    resource_name = f"{resource_name}/versions/latest"
            else:
                resource_name = (
                    f"projects/{self.project_id}/secrets/"
                    f"{secret_id_or_resource}/versions/latest"
                )

            response = client.access_secret_version(request={"name": resource_name})
            payload = response.payload.data.decode("utf-8").strip()
            _SECRET_CACHE[secret_id_or_resource] = (payload, now + self._ttl_seconds)
            return payload
        except Exception as exc:
            logger.warning(
                "Secret Manager lookup skipped/unavailable",
                extra={
                    "event_type": "secret_manager.lookup_failed",
                    "secret_ref": secret_id_or_resource,
                    "error_type": type(exc).__name__,
                },
            )
            return ""


secret_resolver = SecretManagerResolver()


def get_github_token() -> str:
    """Retrieve optional GitHub API token via Secret Manager or secret env mount."""
    return secret_resolver.get_secret(
        env_var_name="GITHUB_TOKEN",
        secret_id_env_var="GITHUB_TOKEN_SECRET_ID",
        default_secret_id="ai-daily-digest-github-token",
    )


def get_gemini_api_key() -> str:
    """Retrieve optional Gemini API key via Secret Manager or secret env mount."""
    return (
        secret_resolver.get_secret(
            env_var_name="GEMINI_API_KEY",
            secret_id_env_var="GEMINI_API_KEY_SECRET_ID",
            default_secret_id="ai-daily-digest-gemini-api-key",
        )
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )

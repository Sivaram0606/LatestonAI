"""FastAPI server exposing Daily Digest REST APIs, HITL gates, OTel headers, and React UI."""

from __future__ import annotations

import datetime
import logging
import os
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# Ensure local and GCE metadata endpoints bypass outbound HTTP proxies
_no_proxy_hosts = "127.0.0.1,localhost,metadata.google.internal,169.254.169.254"
for _var in ("NO_PROXY", "no_proxy"):
    _existing = os.environ.get(_var, "").strip()
    if "metadata.google.internal" not in _existing:
        os.environ[_var] = f"{_existing},{_no_proxy_hosts}".strip(",")

from app.agent import MODEL  # noqa: E402
from app.digest_service import (  # noqa: E402
    clear_digest_cache,
    generate_daily_briefing,
)
from app.observability import (  # noqa: E402
    configure_observability,
    get_current_trace_id,
    trace_span,
)
from app.persistence import briefing_repository  # noqa: E402
from app.schemas import DailyBriefingResponse  # noqa: E402
from app.secrets import get_gemini_api_key  # noqa: E402
from app.tools import parse_target_date  # noqa: E402

configure_observability()
logger = logging.getLogger("app.server")

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

api = FastAPI(
    title="AI Daily Digest — AgentOps Multi-Agent Briefing API",
    description="Google ADK multi-agent backend with AgentOps guardrails, tracing, FTS5 memory, and HITL gates.",
    version="0.2.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@api.middleware("http")
async def attach_otel_trace_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    with trace_span(
        f"http.{request.method.lower()}",
        {"http.path": request.url.path, "http.method": request.method},
    ):
        trace_id = get_current_trace_id()
        response: Response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response


class RuntimeLLMConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_vertexai: bool = True
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    gemini_api_key: str = ""


class PublishDigestHITLRequest(BaseModel):
    """High-stakes broadcast request gated by explicit Human-in-the-Loop confirmation."""

    model_config = ConfigDict(extra="forbid")

    target_date: str = Field(description="Target briefing date in YYYY-MM-DD format.")
    distribution_channel: Literal[
        "org_announcement", "engineering_digest", "public_feed"
    ] = "engineering_digest"
    reviewer_note: str = Field(min_length=3, max_length=500)
    human_approved: bool = Field(
        default=False,
        description="Explicit human confirmation flag required to execute publication.",
    )


@api.get("/api/health")
async def health_check() -> dict[str, object]:
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() in (
        "true",
        "1",
    )
    has_api_key = bool(get_gemini_api_key())
    return {
        "status": "ok",
        "model": MODEL,
        "auth_mode": "vertex_ai" if use_vertex else "gemini_api_key",
        "google_cloud_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        "google_cloud_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        "has_gemini_api_key": has_api_key,
        "server_utc_date": datetime.datetime.now(datetime.UTC).date().isoformat(),
    }


@api.post("/api/config")
async def update_runtime_config(payload: RuntimeLLMConfigRequest) -> dict[str, object]:
    """Update Vertex AI or Secret Manager / API key runtime config and reset cache."""
    if payload.gemini_api_key.strip() and not payload.use_vertexai:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
        os.environ["GEMINI_API_KEY"] = payload.gemini_api_key.strip()
        os.environ["GOOGLE_API_KEY"] = payload.gemini_api_key.strip()
    else:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        if payload.google_cloud_project.strip():
            os.environ["GOOGLE_CLOUD_PROJECT"] = payload.google_cloud_project.strip()
        if payload.google_cloud_location.strip():
            os.environ["GOOGLE_CLOUD_LOCATION"] = payload.google_cloud_location.strip()
        if payload.gemini_api_key.strip():
            os.environ["GEMINI_API_KEY"] = payload.gemini_api_key.strip()

    clear_digest_cache()
    await briefing_repository.clear_cached_briefings()
    logger.info(
        "Runtime LLM configuration updated",
        extra={"event_type": "config.updated", "use_vertexai": payload.use_vertexai},
    )
    return await health_check()


@api.get("/api/digest", response_model=DailyBriefingResponse)
async def get_daily_digest(
    date: str = Query(
        default="today",
        description="Target date in YYYY-MM-DD format (or 'today').",
    ),
    refresh: bool = Query(
        default=False,
        description="Bypass cache and re-run the multi-agent pipeline.",
    ),
    analysis_depth: Literal["standard", "deep"] = Query(
        default="standard",
        description="Model routing tier: 'standard' (Flash) or 'deep' (Pro strategic planning).",
    ),
) -> DailyBriefingResponse:
    return await generate_daily_briefing(
        target_date=date,
        force_refresh=refresh,
        analysis_depth=analysis_depth,
    )


@api.get("/api/archive/search")
async def search_archive(
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, object]:
    """Search persisted historical briefings via SQLite FTS5 BM25 ranking."""
    matches = await briefing_repository.search_briefings(query=q, limit=limit)
    return {"status": "ok", "query": q, "count": len(matches), "matches": matches}


@api.post("/api/archive/publish")
async def publish_digest_with_hitl_gate(
    payload: PublishDigestHITLRequest,
) -> dict[str, object]:
    """High-stakes broadcast endpoint requiring explicit Human-in-the-Loop confirmation."""
    day, _ = parse_target_date(payload.target_date)
    if not payload.human_approved:
        raise HTTPException(
            status_code=409,
            detail={
                "status": "pending_human_approval",
                "requires_confirmation": True,
                "target_date": day.isoformat(),
                "distribution_channel": payload.distribution_channel,
                "message": (
                    "Human-in-the-Loop confirmation required: resubmit with "
                    "`human_approved=true` after reviewing the daily briefing."
                ),
            },
        )

    archive_id = f"pub-{day.isoformat()}-{uuid.uuid4().hex[:8]}"
    published_at = await briefing_repository.record_published_archive(
        archive_id=archive_id,
        target_date=day.isoformat(),
        channel=payload.distribution_channel,
        reviewer_note=payload.reviewer_note,
    )
    return {
        "status": "published",
        "archive_id": archive_id,
        "target_date": day.isoformat(),
        "distribution_channel": payload.distribution_channel,
        "published_at": published_at,
    }


if FRONTEND_DIST.is_dir():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        api.mount(
            "/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets"
        )

    @api.get("/{full_path:path}")
    async def serve_spa(full_path: str) -> FileResponse:
        candidate = (FRONTEND_DIST / full_path).resolve()
        if (
            full_path
            and str(candidate).startswith(str(FRONTEND_DIST.resolve()))
            and candidate.is_file()
        ):
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(api, host="0.0.0.0", port=port)

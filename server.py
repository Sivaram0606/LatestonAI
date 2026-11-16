"""FastAPI server exposing Daily Digest REST APIs and serving the React UI."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
from app.schemas import DailyBriefingResponse  # noqa: E402

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

api = FastAPI(
    title="AI Daily Digest — Multi-Agent Briefing API",
    description="Google ADK multi-agent backend pulling popular AI APIs and summarizing daily intelligence.",
    version="0.1.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class RuntimeLLMConfigRequest(BaseModel):
    use_vertexai: bool = True
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    gemini_api_key: str = ""


@api.get("/api/health")
async def health_check() -> dict[str, object]:
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() in (
        "true",
        "1",
    )
    has_api_key = bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
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
    """Update Vertex AI or Gemini API key configuration in-memory and clear digest cache."""
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
    return await health_check()


@api.get("/api/digest", response_model=DailyBriefingResponse)
async def get_daily_digest(
    date: str = Query(
        default="today",
        description="Target date in YYYY-MM-DD format (or 'today').",
    ),
    refresh: bool = Query(
        default=False,
        description="Bypass in-memory cache and re-run the multi-agent pipeline.",
    ),
) -> DailyBriefingResponse:
    return await generate_daily_briefing(target_date=date, force_refresh=refresh)


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

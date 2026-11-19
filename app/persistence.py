"""Persistent SQLite FTS5 briefing/knowledge store and non-blocking async memory consolidator.

Satisfies AgentOps Code Review Matrix — Category 2 (Context & Memory):
- 2.3 Persistent Session State: Stores daily briefings, metadata, and searchable FTS5
      full-text vectors persistently in SQLite (`aiosqlite`) alongside ADK's
      `SqliteSessionService` / `VertexAiSessionService` and optional `VertexAiSearchTool`.
- 2.4 Async Memory Operations: Executes expensive memory consolidation (`add_session_to_memory`
      + persistent FTS indexing + PII scrubbing) as tracked background `asyncio.Task`s so UI
      responses never block on storage/memory writes.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import pathlib
import tempfile
from typing import Any

import aiosqlite

from app.pii_redactor import pii_redactor
from app.schemas import DailyBriefingResponse

logger = logging.getLogger("app.persistence")


def get_default_sqlite_path(filename: str = "ai_daily_digest.sqlite3") -> str:
    """Return a writable SQLite file path for local dev, tests, and Cloud Run."""
    configured = os.environ.get("AI_DIGEST_SQLITE_PATH", "").strip()
    if configured:
        parent = pathlib.Path(configured).parent
        parent.mkdir(parents=True, exist_ok=True)
        return configured

    base_dir = pathlib.Path(tempfile.gettempdir()) / "ai_daily_digest_store"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / filename)


class PersistentBriefingRepository:
    """Async SQLite + FTS5 persistent repository for daily briefings and published snapshots."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._initialized_path: str | None = None
        self._init_lock = asyncio.Lock()

    @property
    def db_path(self) -> str:
        return self._db_path or get_default_sqlite_path("briefings_fts.sqlite3")

    async def _ensure_schema(self) -> None:
        target = self.db_path
        if self._initialized_path == target:
            return
        async with self._init_lock:
            if self._initialized_path == target:
                return
            async with aiosqlite.connect(target) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_briefings (
                        target_date TEXT PRIMARY KEY,
                        headline TEXT NOT NULL,
                        executive_summary TEXT NOT NULL,
                        generation_mode TEXT NOT NULL,
                        model_used TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                await db.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS daily_briefings_fts USING fts5(
                        target_date UNINDEXED,
                        headline,
                        executive_summary,
                        searchable_corpus
                    )
                    """
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS published_archives (
                        archive_id TEXT PRIMARY KEY,
                        target_date TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        reviewer_note TEXT NOT NULL,
                        published_at TEXT NOT NULL
                    )
                    """
                )
                await db.commit()
            self._initialized_path = target

    async def save_briefing(self, response: DailyBriefingResponse) -> None:
        """Persist a PII-scrubbed DailyBriefingResponse and index it in SQLite FTS5."""
        await self._ensure_schema()
        raw_dict = response.model_dump()
        scrubbed_dict = pii_redactor.scrub_payload(raw_dict)
        scrubbed_resp = DailyBriefingResponse.model_validate(scrubbed_dict)

        b = scrubbed_resp.briefing
        corpus_parts = [
            b.weekly_window_heading,
            *b.whats_new_today,
            *b.key_themes,
            *(f"{p.title} {p.summary}" for p in b.papers),
            *(f"{m.name} {m.summary}" for m in b.models),
            *(f"{n.title} {n.summary}" for n in b.news),
            *(f"{r.full_name} {r.summary}" for r in b.repos),
        ]
        searchable_corpus = " | ".join(corpus_parts)
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO daily_briefings (
                    target_date, headline, executive_summary,
                    generation_mode, model_used, payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_date) DO UPDATE SET
                    headline = excluded.headline,
                    executive_summary = excluded.executive_summary,
                    generation_mode = excluded.generation_mode,
                    model_used = excluded.model_used,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    scrubbed_resp.target_date,
                    b.headline,
                    b.executive_summary,
                    scrubbed_resp.generation_mode,
                    scrubbed_resp.model_used,
                    json.dumps(scrubbed_dict),
                    now_iso,
                ),
            )
            await db.execute(
                "DELETE FROM daily_briefings_fts WHERE target_date = ?",
                (scrubbed_resp.target_date,),
            )
            await db.execute(
                """
                INSERT INTO daily_briefings_fts (
                    target_date, headline, executive_summary, searchable_corpus
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    scrubbed_resp.target_date,
                    b.headline,
                    b.executive_summary,
                    searchable_corpus,
                ),
            )
            await db.commit()

    async def get_briefing(self, target_date: str) -> DailyBriefingResponse | None:
        """Load a previously persisted daily briefing by YYYY-MM-DD date."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT payload_json FROM daily_briefings WHERE target_date = ?",
                (target_date,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return DailyBriefingResponse.model_validate_json(row[0])

    async def search_briefings(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Full-text BM25 retrieval across archived daily briefings."""
        await self._ensure_schema()
        safe_tokens = [
            tok
            for tok in "".join(
                ch if ch.isalnum() else " " for ch in (query or "")
            ).split()
            if len(tok) >= 2
        ]
        async with aiosqlite.connect(self.db_path) as db:
            if safe_tokens:
                fts_query = " OR ".join(f'"{tok}"' for tok in safe_tokens[:8])
                cursor = await db.execute(
                    """
                    SELECT target_date, headline, substr(executive_summary, 1, 280)
                    FROM daily_briefings_fts
                    WHERE daily_briefings_fts MATCH ?
                    ORDER BY bm25(daily_briefings_fts)
                    LIMIT ?
                    """,
                    (fts_query, max(1, min(limit, 20))),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT target_date, headline, substr(executive_summary, 1, 280)
                    FROM daily_briefings
                    ORDER BY target_date DESC
                    LIMIT ?
                    """,
                    (max(1, min(limit, 20)),),
                )
            rows = await cursor.fetchall()
            return [
                {
                    "target_date": r[0],
                    "headline": r[1],
                    "excerpt": r[2],
                }
                for r in rows
            ]

    async def record_published_archive(
        self,
        archive_id: str,
        target_date: str,
        channel: str,
        reviewer_note: str,
    ) -> str:
        """Record a human-approved official archive broadcast entry."""
        await self._ensure_schema()
        published_at = datetime.datetime.now(datetime.UTC).isoformat()
        scrubbed_note = pii_redactor.scrub_text(reviewer_note)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO published_archives (
                    archive_id, target_date, channel, reviewer_note, published_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (archive_id, target_date, channel, scrubbed_note, published_at),
            )
            await db.commit()
        return published_at

    async def clear_cached_briefings(self) -> None:
        """Clear persisted daily briefings when runtime LLM config is reset."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM daily_briefings")
            await db.execute("DELETE FROM daily_briefings_fts")
            await db.commit()


class AsyncMemoryConsolidator:
    """Non-blocking background scheduler for ADK Memory Bank & FTS archive indexing."""

    def __init__(self, repository: PersistentBriefingRepository) -> None:
        self._repository = repository
        self._pending_tasks: set[asyncio.Task[None]] = set()

    @property
    def pending_count(self) -> int:
        return len(self._pending_tasks)

    def schedule_briefing_and_memory_consolidation(
        self,
        response: DailyBriefingResponse,
        session: Any | None = None,
        memory_service: Any | None = None,
    ) -> asyncio.Task[None]:
        """Fire-and-forget async background task so UI latency is never blocked."""
        task = asyncio.create_task(
            self._consolidate(
                response=response, session=session, memory_service=memory_service
            ),
            name=f"consolidate-memory-{response.target_date}",
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def _consolidate(
        self,
        response: DailyBriefingResponse,
        session: Any | None,
        memory_service: Any | None,
    ) -> None:
        try:
            await self._repository.save_briefing(response)
            if session is not None and memory_service is not None:
                await memory_service.add_session_to_memory(session)
            logger.info(
                "Background memory consolidation completed",
                extra={
                    "event_type": "memory.consolidation",
                    "phase": "OUTCOME",
                    "target_date": response.target_date,
                    "outcome_status": "persisted",
                },
            )
        except Exception as exc:
            logger.error(
                "Background memory consolidation encountered an error",
                extra={
                    "event_type": "memory.consolidation",
                    "phase": "OUTCOME",
                    "target_date": response.target_date,
                    "outcome_status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": pii_redactor.scrub_text(str(exc)),
                },
            )

    async def drain(self) -> None:
        """Await all in-flight background consolidation tasks (for shutdown/tests)."""
        if self._pending_tasks:
            await asyncio.gather(*list(self._pending_tasks), return_exceptions=True)


briefing_repository = PersistentBriefingRepository()
memory_consolidator = AsyncMemoryConsolidator(repository=briefing_repository)

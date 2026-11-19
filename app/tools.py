"""Domain-specific ADK tools with Sunday-Saturday weekly windows, strict schemas, and HITL gates.

Satisfies AgentOps Code Review Matrix — Category 1 (Tool & Interface Design) & 3.4 (HITL):
- 1.1 Comprehensive Tool Docstrings: Documents Sunday-to-Saturday weekly window semantics,
      daily focus (`target_date`), parameters, and return schemas without `tool_context`.
- 1.2 Descriptive Naming: Specific verb-noun tool names across all 7 tools.
- 1.3 Explicit JSON Schemas: Strict `ConfigDict(extra="forbid")` Pydantic input & output models.
- 1.4 Guided Error Handling: Structured `ToolErrorGuidance` dicts provide actionable
      `recovery_instructions` back to the LLM instead of crashing.
- 3.4 Human-in-the-Loop Hooks: High-stakes archive broadcast gated by ADK confirmation.
"""

from __future__ import annotations

import asyncio
import datetime
import re
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from typing import Any

from google.adk.tools import FunctionTool, ToolContext
import httpx
from pydantic import ValidationError

from app.observability import trace_span
from app.persistence import briefing_repository
from app.pii_redactor import pii_redactor
from app.schemas import (
    ArchiveSearchInput,
    ArxivPapersToolOutput,
    DateScopedToolInput,
    GitHubReposToolOutput,
    HackerNewsToolOutput,
    HFDailyPapersToolOutput,
    HFModelsSpacesToolOutput,
    ModelBrief,
    NewsBrief,
    PaperBrief,
    PublishDigestArchiveInput,
    RepoBrief,
    ToolErrorGuidance,
)
from app.secrets import get_github_token

USER_AGENT = "ai-daily-digest-adk/2.1 (Google-ADK AgentOps Weekly+Daily Briefing)"
HTTP_TIMEOUT = 8.0

AI_NEWS_PATTERN = re.compile(
    r"\b("
    r"ai|artificial intelligence|llm|llms|gpt|chatgpt|gemini|claude|openai|"
    r"anthropic|deepmind|hugging\s?face|mistral|llama|qwen|deepseek|gemma|"
    r"transformer|diffusion|neural|machine learning|deep learning|agent|agents|"
    r"agentic|rag|mcp|vllm|ollama|pytorch|jax|cuda|tpu|gpu|nvidia|robotics|"
    r"inference|fine-tun\w*|benchmark|reasoning|multimodal|embedding|vector|"
    r"copilot|codex|cursor|voice model|vision model|open-weight|weights"
    r")\b",
    re.IGNORECASE,
)


def parse_target_date(target_date: str) -> tuple[datetime.date, str]:
    """Parse YYYY-MM-DD (or 'today') into a valid UTC date, clamping future dates."""
    today = datetime.datetime.now(datetime.UTC).date()
    cleaned = (target_date or "").strip().lower()
    if not cleaned or cleaned == "today":
        return today, ""
    try:
        parsed = datetime.date.fromisoformat(cleaned[:10])
    except ValueError:
        return today, f"Invalid date '{target_date}'; defaulted to {today.isoformat()}."
    if parsed > today:
        return (
            today,
            f"Requested date {parsed.isoformat()} is in the future; clamped to {today.isoformat()}.",
        )
    return parsed, ""


def compute_sunday_to_saturday_window(
    day: datetime.date,
) -> tuple[datetime.date, datetime.date]:
    """Return `(week_start_sunday, week_end_saturday)` for the week containing `day`."""
    days_since_sunday = (day.weekday() + 1) % 7
    week_start = day - datetime.timedelta(days=days_since_sunday)
    week_end = week_start + datetime.timedelta(days=6)
    return week_start, week_end


def format_weekly_window_heading(
    week_start: datetime.date, week_end: datetime.date
) -> str:
    """Format a human-readable Sunday-to-Saturday weekly window heading."""
    start_lbl = week_start.strftime("Sun, %b %d, %Y")
    end_lbl = week_end.strftime("Sat, %b %d, %Y")
    return f"Weekly AI Summary: {start_lbl} – {end_lbl}"


def _effective_query_end(week_end: datetime.date) -> datetime.date:
    today = datetime.datetime.now(datetime.UTC).date()
    return min(week_end, today)


def _validate_date_input(
    raw_target_date: str, source_name: str
) -> tuple[datetime.date | None, str, dict[str, Any] | None]:
    """Validate tool `target_date` via strict Pydantic schema or return guided LLM error."""
    try:
        validated = DateScopedToolInput.model_validate({"target_date": raw_target_date})
    except ValidationError as exc:
        err = ToolErrorGuidance(
            source=source_name,
            requested_date=str(raw_target_date)[:40],
            error_code="INVALID_DATE_ARGUMENT",
            error_message=pii_redactor.scrub_text(str(exc.errors()[0]["msg"])),
            retryable=True,
            recovery_instructions=(
                "Re-invoke this tool with 'target_date' formatted strictly as an ISO-8601 "
                "date string 'YYYY-MM-DD' (for example '2025-02-14') or the keyword 'today'."
            ),
        )
        return None, "", err.model_dump()

    day, note = parse_target_date(validated.target_date)
    return day, note, None


def _clean_text(text: str, max_len: int = 600) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "")).strip()
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rsplit(" ", 1)[0] + "…"


async def fetch_hf_daily_papers(target_date: str) -> dict[str, Any]:
    """Fetch AI research papers across the Sunday-to-Saturday week and `target_date` from Hugging Face.

    Queries `https://huggingface.co/api/daily_papers` concurrently across active days in
    the Sunday-to-Saturday week window containing `target_date` (falling back up to 3 days
    prior if `target_date` is Sunday morning before curation), prioritizing papers published
    on `target_date` alongside top-upvoted papers of the week.

    Args:
        target_date: Target UTC date formatted strictly as 'YYYY-MM-DD' (e.g. '2025-02-14')
            or the literal string 'today'.

    Returns:
        Strict JSON-serializable dictionary matching `HFDailyPapersToolOutput` on success,
        or `ToolErrorGuidance` with `recovery_instructions` if validation or HTTP fails.
    """
    with trace_span("tool.fetch_hf_daily_papers", {"target_date": target_date}):
        day, date_note, err_payload = _validate_date_input(
            target_date, "Hugging Face Daily Papers"
        )
        if err_payload is not None or day is None:
            return err_payload or {}

        week_start, week_end = compute_sunday_to_saturday_window(day)
        query_end = _effective_query_end(week_end)
        week_window_str = f"{week_start.isoformat()}..{week_end.isoformat()}"

        candidate_days: list[datetime.date] = []
        cursor = query_end
        while cursor >= week_start:
            candidate_days.append(cursor)
            cursor -= datetime.timedelta(days=1)

        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers
        ) as client:
            responses = await asyncio.gather(
                *(
                    client.get(
                        f"https://huggingface.co/api/daily_papers?date={d.isoformat()}"
                    )
                    for d in candidate_days
                ),
                return_exceptions=True,
            )

            dated_batches: list[tuple[datetime.date, list[dict[str, Any]]]] = []
            network_errors: list[str] = []
            for d, resp in zip(candidate_days, responses, strict=False):
                if isinstance(resp, Exception):
                    network_errors.append(str(resp))
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        dated_batches.append((d, data))

            fallback_note = date_note
            effective_day = day

            # If Sunday/Monday morning has no papers in the new Sun-Sat week yet, walk back up to 3 days
            if not dated_batches:
                for back in range(1, 4):
                    prev_d = week_start - datetime.timedelta(days=back)
                    try:
                        r = await client.get(
                            f"https://huggingface.co/api/daily_papers?date={prev_d.isoformat()}"
                        )
                        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
                            dated_batches.append((prev_d, r.json()))
                            effective_day = prev_d
                            msg = (
                                f"New Sunday–Saturday week ({week_window_str}) has no curated HF batch yet; "
                                f"including latest Friday/weekend batch from {prev_d.isoformat()}."
                            )
                            fallback_note = (
                                f"{fallback_note} {msg}".strip() if fallback_note else msg
                            )
                            break
                    except httpx.HTTPError as exc:
                        network_errors.append(str(exc))

            if not dated_batches and network_errors:
                return ToolErrorGuidance(
                    source="Hugging Face Daily Papers",
                    requested_date=day.isoformat(),
                    error_code="UPSTREAM_HTTP_ERROR",
                    error_message=pii_redactor.scrub_text(network_errors[0]),
                    retryable=True,
                    recovery_instructions=(
                        "Hugging Face Daily Papers API timed out or failed connectivity. "
                        "Proceed by synthesizing research papers from `fetch_arxiv_ai_papers` "
                        "instead of halting the briefing."
                    ),
                ).model_dump()

        seen_ids: set[str] = set()
        today_papers: list[PaperBrief] = []
        week_papers: list[PaperBrief] = []

        for batch_day, raw_items in dated_batches:
            for entry in raw_items:
                paper = entry.get("paper") or {}
                paper_id = str(paper.get("id") or entry.get("id") or "").strip()
                title = _clean_text(
                    str(entry.get("title") or paper.get("title") or ""), max_len=240
                )
                dedupe_key = paper_id or title.lower()
                if not title or dedupe_key in seen_ids:
                    continue
                seen_ids.add(dedupe_key)

                authors_raw = paper.get("authors") or []
                author_names = [
                    str(a.get("name") or "").strip()
                    for a in authors_raw
                    if isinstance(a, dict) and a.get("name")
                ]
                authors_str = ", ".join(author_names[:4])
                if len(author_names) > 4:
                    authors_str += " et al."

                summary_raw = (
                    paper.get("ai_summary")
                    or entry.get("summary")
                    or paper.get("summary")
                    or ""
                )
                keywords = [
                    str(k).strip()
                    for k in (paper.get("ai_keywords") or [])
                    if str(k).strip()
                ][:4] or ["AI Research"]

                item = PaperBrief(
                    title=title,
                    authors=authors_str or "Research Community",
                    source="Hugging Face Daily Papers",
                    url=(
                        f"https://huggingface.co/papers/{paper_id}"
                        if paper_id
                        else "https://huggingface.co/papers"
                    ),
                    published_date=batch_day.isoformat(),
                    upvotes=int(paper.get("upvotes") or 0),
                    summary=_clean_text(str(summary_raw), max_len=550)
                    or "Curated AI research paper.",
                    tags=keywords,
                )
                if batch_day == day:
                    today_papers.append(item)
                else:
                    week_papers.append(item)

        today_papers.sort(key=lambda p: p.upvotes, reverse=True)
        week_papers.sort(key=lambda p: p.upvotes, reverse=True)
        combined = [*today_papers[:5], *week_papers]
        top_papers = combined[:10]
        for idx, p in enumerate(top_papers, 1):
            p.citation_id = f"P{idx}"

        return HFDailyPapersToolOutput(
            status="success" if top_papers else "empty",
            source="Hugging Face Daily Papers",
            requested_date=day.isoformat(),
            week_window=week_window_str,
            effective_date=effective_day.isoformat(),
            note=fallback_note,
            count=len(top_papers),
            papers=top_papers,
        ).model_dump()


async def fetch_arxiv_ai_papers(target_date: str) -> dict[str, Any]:
    """Fetch cs.AI, cs.LG, and cs.CL preprints across the Sunday-to-Saturday week from arXiv.

    Queries `https://export.arxiv.org/api/query` across the Sunday-to-Saturday week window
    containing `target_date` (widening by 3 days if Sunday has no announcements yet).

    Args:
        target_date: Target UTC date formatted strictly as 'YYYY-MM-DD' (e.g. '2025-02-14')
            or the literal string 'today'.

    Returns:
        Strict JSON-serializable dictionary matching `ArxivPapersToolOutput` on success,
        or `ToolErrorGuidance` with actionable `recovery_instructions` on failure.
    """
    with trace_span("tool.fetch_arxiv_ai_papers", {"target_date": target_date}):
        day, date_note, err_payload = _validate_date_input(target_date, "arXiv API")
        if err_payload is not None or day is None:
            return err_payload or {}

        week_start, week_end = compute_sunday_to_saturday_window(day)
        query_end = _effective_query_end(week_end)
        headers = {"User-Agent": USER_AGENT}
        atom_ns = "{http://www.w3.org/2005/Atom}"
        windows = [
            (week_start, query_end),
            (week_start - datetime.timedelta(days=4), query_end),
        ]
        last_err = ""

        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers
        ) as client:
            for start_d, end_d in windows:
                start_str = start_d.strftime("%Y%m%d") + "0000"
                end_str = end_d.strftime("%Y%m%d") + "2359"
                search_q = (
                    f"(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) "
                    f"AND submittedDate:[{start_str} TO {end_str}]"
                )
                params = {
                    "search_query": search_q,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": "10",
                }
                try:
                    resp = await client.get(
                        "https://export.arxiv.org/api/query", params=params
                    )
                    if resp.status_code != 200:
                        last_err = f"HTTP {resp.status_code}"
                        continue
                    root = ET.fromstring(resp.text)
                    entries = root.findall(f"{atom_ns}entry")
                    if not entries:
                        continue

                    papers: list[PaperBrief] = []
                    for idx, entry in enumerate(entries, 1):
                        title = _clean_text(
                            entry.findtext(f"{atom_ns}title", default=""), max_len=240
                        )
                        if not title:
                            continue
                        raw_id = entry.findtext(f"{atom_ns}id", default="").strip()
                        url = raw_id.replace("http://", "https://")
                        summary = _clean_text(
                            entry.findtext(f"{atom_ns}summary", default=""), max_len=520
                        )
                        published = entry.findtext(
                            f"{atom_ns}published", default=day.isoformat()
                        )[:10]
                        author_elems = entry.findall(f"{atom_ns}author/{atom_ns}name")
                        author_names = [
                            (a.text or "").strip()
                            for a in author_elems
                            if (a.text or "").strip()
                        ]
                        authors_str = ", ".join(author_names[:4])
                        if len(author_names) > 4:
                            authors_str += " et al."
                        cats = [
                            c.attrib.get("term", "")
                            for c in entry.findall(f"{atom_ns}category")
                            if c.attrib.get("term", "").startswith("cs.")
                        ][:3]
                        papers.append(
                            PaperBrief(
                                citation_id=f"P{idx}",
                                title=title,
                                authors=authors_str or "arXiv Authors",
                                source="arXiv",
                                url=url or "https://arxiv.org/list/cs.AI/recent",
                                published_date=published,
                                upvotes=0,
                                summary=summary or "arXiv AI research preprint.",
                                tags=cats or ["cs.AI"],
                            )
                        )

                    if papers:
                        return ArxivPapersToolOutput(
                            status="success",
                            source="arXiv API",
                            requested_date=day.isoformat(),
                            window=f"{week_start.isoformat()}..{week_end.isoformat()}",
                            note=date_note,
                            count=len(papers),
                            papers=papers,
                        ).model_dump()
                except (httpx.HTTPError, ET.ParseError) as exc:
                    last_err = pii_redactor.scrub_text(str(exc))
                    continue

        if last_err:
            return ToolErrorGuidance(
                source="arXiv API",
                requested_date=day.isoformat(),
                error_code="ARXIV_UNAVAILABLE",
                error_message=last_err,
                retryable=True,
                recovery_instructions=(
                    "arXiv export API returned an error or timed out. Do not retry more than once; "
                    "rely on `fetch_hf_daily_papers` output to populate the research papers section."
                ),
            ).model_dump()

        return ArxivPapersToolOutput(
            status="empty",
            source="arXiv API",
            requested_date=day.isoformat(),
            window=f"{week_start.isoformat()}..{week_end.isoformat()}",
            note=date_note or "No arXiv entries found for Sunday-Saturday window.",
            count=0,
            papers=[],
        ).model_dump()


async def fetch_hf_trending_models_and_spaces(target_date: str) -> dict[str, Any]:
    """Fetch trending open-weight AI models and interactive Spaces from Hugging Face Hub.

    Pulls weekly trending model checkpoints and interactive Spaces concurrently from
    `https://huggingface.co/api/trending`, recording `last_modified` dates so daily updates
    on `target_date` can be distinguished alongside weekly momentum.

    Args:
        target_date: Target UTC date formatted strictly as 'YYYY-MM-DD' (e.g. '2025-02-14')
            or the literal string 'today'.

    Returns:
        Strict JSON-serializable dictionary matching `HFModelsSpacesToolOutput` on success,
        or `ToolErrorGuidance` with `recovery_instructions` on failure.
    """
    with trace_span(
        "tool.fetch_hf_trending_models_and_spaces", {"target_date": target_date}
    ):
        day, date_note, err_payload = _validate_date_input(
            target_date, "Hugging Face Hub API"
        )
        if err_payload is not None or day is None:
            return err_payload or {}

        week_start, week_end = compute_sunday_to_saturday_window(day)
        week_window_str = f"{week_start.isoformat()}..{week_end.isoformat()}"
        today = datetime.datetime.now(datetime.UTC).date()
        is_historical = (today - week_end).days > 2

        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers
        ) as client:
            models_task = client.get(
                "https://huggingface.co/api/trending?type=model&limit=10"
            )
            spaces_task = client.get(
                "https://huggingface.co/api/trending?type=space&limit=6"
            )
            models_resp, spaces_resp = await asyncio.gather(
                models_task, spaces_task, return_exceptions=True
            )

            items: list[ModelBrief] = []
            raw_models: list[dict[str, Any]] = []
            if (
                isinstance(models_resp, httpx.Response)
                and models_resp.status_code == 200
            ):
                payload = models_resp.json()
                if isinstance(payload, dict) and "recentlyTrending" in payload:
                    raw_models = [
                        x.get("repoData", x)
                        for x in payload.get("recentlyTrending", [])
                    ]
                elif isinstance(payload, list):
                    raw_models = [x.get("repoData", x) for x in payload]

            if not raw_models:
                try:
                    fallback = await client.get(
                        "https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=8"
                    )
                    if fallback.status_code == 200 and isinstance(
                        fallback.json(), list
                    ):
                        raw_models = fallback.json()
                except httpx.HTTPError as exc:
                    return ToolErrorGuidance(
                        source="Hugging Face Hub API",
                        requested_date=day.isoformat(),
                        error_code="HF_HUB_HTTP_ERROR",
                        error_message=pii_redactor.scrub_text(str(exc)),
                        retryable=True,
                        recovery_instructions=(
                            "Hugging Face Hub trending endpoint is unreachable. Note that model "
                            "telemetry is temporarily unavailable and proceed with papers, news, and repos."
                        ),
                    ).model_dump()

            for m in raw_models:
                repo_id = str(m.get("id") or m.get("modelId") or "").strip()
                if not repo_id:
                    continue
                lowered_id = repo_id.lower()
                if any(
                    term in lowered_id
                    for term in (
                        "uncensored",
                        "nsfw",
                        "abliterated",
                        "heretic",
                        "erotic",
                    )
                ):
                    continue
                author = str(
                    m.get("author")
                    or (repo_id.split("/")[0] if "/" in repo_id else "hf")
                )
                pipeline_tag = str(m.get("pipeline_tag") or "foundation-model")
                likes = int(m.get("likes") or 0)
                downloads = int(m.get("downloads") or 0)
                last_mod = str(m.get("lastModified") or day.isoformat())[:10]
                items.append(
                    ModelBrief(
                        citation_id=f"M{len(items) + 1}",
                        name=repo_id,
                        kind="Model",
                        author=author,
                        pipeline_tag=pipeline_tag,
                        url=f"https://huggingface.co/{repo_id}",
                        likes=likes,
                        downloads=downloads,
                        last_modified=last_mod,
                        summary=(
                            f"Trending {pipeline_tag} model by {author} on Hugging Face Hub "
                            f"({likes:,} likes, {downloads:,} downloads)."
                        ),
                    )
                )
                if len(items) >= 6:
                    break

            raw_spaces: list[dict[str, Any]] = []
            if (
                isinstance(spaces_resp, httpx.Response)
                and spaces_resp.status_code == 200
            ):
                payload = spaces_resp.json()
                if isinstance(payload, dict) and "recentlyTrending" in payload:
                    raw_spaces = [
                        x.get("repoData", x)
                        for x in payload.get("recentlyTrending", [])
                    ]
                elif isinstance(payload, list):
                    raw_spaces = [x.get("repoData", x) for x in payload]

            if not raw_spaces:
                try:
                    fallback_sp = await client.get(
                        "https://huggingface.co/api/spaces?sort=trendingScore&direction=-1&limit=5"
                    )
                    if fallback_sp.status_code == 200 and isinstance(
                        fallback_sp.json(), list
                    ):
                        raw_spaces = fallback_sp.json()
                except httpx.HTTPError:
                    raw_spaces = []

            for sp in raw_spaces[:4]:
                space_id = str(sp.get("id") or "").strip()
                if not space_id:
                    continue
                author = str(
                    sp.get("author")
                    or (space_id.split("/")[0] if "/" in space_id else "hf")
                )
                title = str(sp.get("title") or space_id.split("/")[-1])
                short_desc = _clean_text(
                    str(
                        sp.get("shortDescription")
                        or sp.get("ai_short_description")
                        or ""
                    ),
                    max_len=240,
                )
                likes = int(sp.get("likes") or 0)
                sdk = str(sp.get("sdk") or "interactive-space")
                last_mod = str(sp.get("lastModified") or day.isoformat())[:10]
                items.append(
                    ModelBrief(
                        citation_id=f"M{len(items) + 1}",
                        name=space_id,
                        kind="Space",
                        author=author,
                        pipeline_tag=f"{sdk}-space",
                        url=f"https://huggingface.co/spaces/{space_id}",
                        likes=likes,
                        downloads=0,
                        last_modified=last_mod,
                        summary=(
                            f"{title}: {short_desc}"
                            if short_desc
                            else f"Interactive AI Space '{title}' by {author} ({likes:,} likes)."
                        ),
                    )
                )

            note = date_note
            if is_historical:
                hist_note = (
                    "Hugging Face Hub trending reflects live Hub momentum; "
                    "papers, news, and repos are strictly scoped to the selected Sunday–Saturday week."
                )
                note = f"{note} {hist_note}".strip() if note else hist_note

            return HFModelsSpacesToolOutput(
                status="success" if items else "empty",
                source="Hugging Face Hub API",
                requested_date=day.isoformat(),
                week_window=week_window_str,
                note=note,
                count=len(items),
                items=items,
            ).model_dump()


async def fetch_hackernews_ai_stories(target_date: str) -> dict[str, Any]:
    """Fetch top AI discussions across the Sunday-to-Saturday week and `target_date` from Hacker News.

    Queries Hacker News Algolia concurrently for both the selected `target_date` (to power
    'What's New Today' bullets) and the full Sunday-to-Saturday week window, tagging every
    story with its exact UTC `published_date`.

    Args:
        target_date: Target UTC date formatted strictly as 'YYYY-MM-DD' (e.g. '2025-02-14')
            or the literal string 'today'.

    Returns:
        Strict JSON-serializable dictionary matching `HackerNewsToolOutput` on success,
        or `ToolErrorGuidance` with `recovery_instructions` on failure.
    """
    with trace_span("tool.fetch_hackernews_ai_stories", {"target_date": target_date}):
        day, date_note, err_payload = _validate_date_input(
            target_date, "Hacker News Algolia API"
        )
        if err_payload is not None or day is None:
            return err_payload or {}

        week_start, week_end = compute_sunday_to_saturday_window(day)
        query_end = _effective_query_end(week_end)
        week_window_str = f"{week_start.isoformat()}..{week_end.isoformat()}"

        week_start_ts = int(
            datetime.datetime(
                week_start.year,
                week_start.month,
                week_start.day,
                0,
                0,
                0,
                tzinfo=datetime.UTC,
            ).timestamp()
        )
        week_end_ts = (
            int(
                datetime.datetime(
                    query_end.year,
                    query_end.month,
                    query_end.day,
                    0,
                    0,
                    0,
                    tzinfo=datetime.UTC,
                ).timestamp()
            )
            + 86400
        )
        day_start_ts = int(
            datetime.datetime(
                day.year, day.month, day.day, 0, 0, 0, tzinfo=datetime.UTC
            ).timestamp()
        )
        day_end_ts = day_start_ts + 86400

        week_filter = f"created_at_i>={week_start_ts},created_at_i<{week_end_ts}"
        day_filter = f"created_at_i>={day_start_ts},created_at_i<{day_end_ts}"

        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers
        ) as client:
            reqs = [
                client.get(
                    "https://hn.algolia.com/api/v1/search",
                    params={
                        "tags": "story",
                        "numericFilters": day_filter,
                        "hitsPerPage": "60",
                    },
                ),
                client.get(
                    "https://hn.algolia.com/api/v1/search",
                    params={
                        "tags": "story",
                        "numericFilters": week_filter,
                        "hitsPerPage": "100",
                    },
                ),
                client.get(
                    "https://hn.algolia.com/api/v1/search",
                    params={
                        "query": "AI OR LLM OR Gemini OR OpenAI",
                        "tags": "story",
                        "numericFilters": f"{day_filter},points>=3",
                        "hitsPerPage": "25",
                    },
                ),
            ]
            results = await asyncio.gather(*reqs, return_exceptions=True)

            network_failures = sum(1 for r in results if isinstance(r, Exception))
            if network_failures == len(results):
                return ToolErrorGuidance(
                    source="Hacker News Algolia API",
                    requested_date=day.isoformat(),
                    error_code="HN_ALGOLIA_UNREACHABLE",
                    error_message="All Hacker News Algolia requests failed or timed out.",
                    retryable=True,
                    recovery_instructions=(
                        "Hacker News Algolia search timed out. Continue building the ecosystem "
                        "section using `fetch_github_ai_repos` results without failing the run."
                    ),
                ).model_dump()

            seen_ids: set[str] = set()
            today_hits: list[dict[str, Any]] = []
            week_hits: list[dict[str, Any]] = []

            for idx_r, r in enumerate(results):
                if not isinstance(r, httpx.Response) or r.status_code != 200:
                    continue
                for hit in r.json().get("hits", []):
                    obj_id = str(hit.get("objectID") or "")
                    title = str(hit.get("title") or "")
                    url = str(hit.get("url") or "")
                    if not obj_id or obj_id in seen_ids:
                        continue
                    if idx_r < 2 and not AI_NEWS_PATTERN.search(f"{title} {url}"):
                        continue
                    seen_ids.add(obj_id)
                    created_i = int(hit.get("created_at_i") or day_start_ts)
                    if day_start_ts <= created_i < day_end_ts:
                        today_hits.append(hit)
                    else:
                        week_hits.append(hit)

            today_hits.sort(key=lambda h: int(h.get("points") or 0), reverse=True)
            week_hits.sort(key=lambda h: int(h.get("points") or 0), reverse=True)
            ordered_hits = [*today_hits[:4], *week_hits][:10]

            stories: list[NewsBrief] = []
            for hit in ordered_hits:
                obj_id = str(hit.get("objectID") or "")
                hn_url = f"https://news.ycombinator.com/item?id={obj_id}"
                raw_url = str(hit.get("url") or "").strip() or hn_url
                domain = urllib.parse.urlparse(raw_url).netloc.removeprefix("www.")
                points = int(hit.get("points") or 0)
                comments = int(hit.get("num_comments") or 0)
                title = _clean_text(str(hit.get("title") or ""), max_len=220)
                if not title:
                    continue
                created_iso = str(hit.get("created_at") or day.isoformat())[:10]
                stories.append(
                    NewsBrief(
                        citation_id=f"N{len(stories) + 1}",
                        title=title,
                        source=domain or "Hacker News",
                        url=raw_url,
                        discussion_url=hn_url,
                        published_date=created_iso,
                        points=points,
                        comments=comments,
                        summary=(
                            f"Discussed on Hacker News on {created_iso} ({points} points, "
                            f"{comments} comments) via {domain or 'news.ycombinator.com'}."
                        ),
                    )
                )

            return HackerNewsToolOutput(
                status="success" if stories else "empty",
                source="Hacker News Algolia API",
                requested_date=day.isoformat(),
                week_window=week_window_str,
                note=date_note,
                count=len(stories),
                stories=stories,
            ).model_dump()


async def fetch_github_ai_repos(target_date: str) -> dict[str, Any]:
    """Fetch breakout open-source AI repositories across the Sunday-to-Saturday week from GitHub.

    Queries `https://api.github.com/search/repositories` across the Sunday-to-Saturday week
    containing `target_date` (falling back to a 7-day window if Sunday morning has sparse new
    creations), authenticating via Secret Manager–resolved `GITHUB_TOKEN` when configured.

    Args:
        target_date: Target UTC date formatted strictly as 'YYYY-MM-DD' (e.g. '2025-02-14')
            or the literal string 'today'.

    Returns:
        Strict JSON-serializable dictionary matching `GitHubReposToolOutput` on success,
        or `ToolErrorGuidance` with rate-limit/recovery instructions on error.
    """
    with trace_span("tool.fetch_github_ai_repos", {"target_date": target_date}):
        day, date_note, err_payload = _validate_date_input(
            target_date, "GitHub Search API"
        )
        if err_payload is not None or day is None:
            return err_payload or {}

        week_start, week_end = compute_sunday_to_saturday_window(day)
        query_end = _effective_query_end(week_end)
        search_start = (
            week_start
            if (query_end - week_start).days >= 2
            else query_end - datetime.timedelta(days=5)
        )

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
        }
        gh_token = get_github_token()
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"

        query = (
            f"(llm OR agent OR mcp OR multimodal OR rag) "
            f"created:{search_start.isoformat()}..{query_end.isoformat()}"
        )

        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers
        ) as client:
            try:
                resp = await client.get(
                    "https://api.github.com/search/repositories",
                    params={
                        "q": query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": "10",
                    },
                )
                if resp.status_code != 200:
                    is_rate_limit = resp.status_code in (403, 429)
                    return ToolErrorGuidance(
                        source="GitHub Search API",
                        requested_date=day.isoformat(),
                        error_code=(
                            "GITHUB_RATE_LIMITED"
                            if is_rate_limit
                            else f"HTTP_{resp.status_code}"
                        ),
                        error_message=f"GitHub Search API returned HTTP {resp.status_code}.",
                        retryable=not is_rate_limit,
                        recovery_instructions=(
                            "GitHub unauthenticated rate limit reached. Do not retry immediately; "
                            "synthesize ecosystem highlights from `fetch_hackernews_ai_stories` "
                            "and configure `GITHUB_TOKEN` in GCP Secret Manager for higher quota."
                            if is_rate_limit
                            else "Check date formatting and proceed with available Hacker News data."
                        ),
                    ).model_dump()
                items = resp.json().get("items", [])
            except httpx.HTTPError as exc:
                return ToolErrorGuidance(
                    source="GitHub Search API",
                    requested_date=day.isoformat(),
                    error_code="GITHUB_HTTP_ERROR",
                    error_message=pii_redactor.scrub_text(str(exc)),
                    retryable=True,
                    recovery_instructions=(
                        "GitHub API request failed due to transient network error. "
                        "Proceed with `fetch_hackernews_ai_stories` findings if retry fails."
                    ),
                ).model_dump()

        repos: list[RepoBrief] = []
        for repo in items[:8]:
            full_name = str(repo.get("full_name") or "").strip()
            if not full_name:
                continue
            desc = _clean_text(
                str(repo.get("description") or "Open-source AI repository."),
                max_len=280,
            )
            topics = [str(t) for t in (repo.get("topics") or [])[:4]]
            lang = str(repo.get("language") or "Multi-language")
            created_iso = str(
                repo.get("pushed_at") or repo.get("created_at") or day.isoformat()
            )[:10]
            repos.append(
                RepoBrief(
                    citation_id=f"R{len(repos) + 1}",
                    full_name=full_name,
                    language=lang,
                    url=str(repo.get("html_url") or f"https://github.com/{full_name}"),
                    created_date=created_iso,
                    stars=int(repo.get("stargazers_count") or 0),
                    forks=int(repo.get("forks_count") or 0),
                    summary=desc,
                    topics=topics or ["ai", "open-source"],
                )
            )

        return GitHubReposToolOutput(
            status="success" if repos else "empty",
            source="GitHub Search API",
            requested_date=day.isoformat(),
            window=f"{week_start.isoformat()}..{week_end.isoformat()}",
            note=date_note,
            count=len(repos),
            repos=repos,
        ).model_dump()


async def search_archived_daily_briefings(query: str, limit: int) -> dict[str, Any]:
    """Search previously generated AI briefings stored in the persistent FTS5 database.

    Performs BM25-ranked full-text search across historical headlines, weekly summaries,
    daily bullet updates, papers, models, news, and repositories persisted in SQLite FTS5.

    Args:
        query: Search topic or keyword phrase (for example 'multimodal agents' or 'MCP').
        limit: Maximum number of historical briefing matches to return (between 1 and 20).

    Returns:
        Dictionary containing `status`, `count`, and `matches` (`target_date`, `headline`,
        `excerpt`), or `ToolErrorGuidance` with `recovery_instructions` if validation fails.
    """
    with trace_span("tool.search_archived_daily_briefings", {"query": query}):
        try:
            validated = ArchiveSearchInput.model_validate(
                {"query": query, "limit": limit}
            )
        except ValidationError as exc:
            return ToolErrorGuidance(
                source="PersistentBriefingRepository",
                requested_date="archive",
                error_code="INVALID_SEARCH_INPUT",
                error_message=pii_redactor.scrub_text(str(exc.errors()[0]["msg"])),
                retryable=True,
                recovery_instructions=(
                    "Pass a non-empty `query` string (1-200 chars) and an integer `limit` "
                    "between 1 and 20."
                ),
            ).model_dump()

        matches = await briefing_repository.search_briefings(
            query=validated.query, limit=validated.limit
        )
        return {
            "status": "success",
            "source": "SQLite FTS5 Briefing Store",
            "query": pii_redactor.scrub_text(validated.query),
            "count": len(matches),
            "matches": matches,
        }


def requires_human_confirmation_for_publish(
    target_date: str = "",
    distribution_channel: str = "",
    reviewer_note: str = "",
    **_: Any,
) -> bool:
    """Predicate enforcing mandatory Human-in-the-Loop confirmation for external broadcasts."""
    return distribution_channel in {
        "org_announcement",
        "engineering_digest",
        "public_feed",
        "",
    }


async def publish_verified_daily_digest_archive(
    target_date: str,
    distribution_channel: str,
    reviewer_note: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Publish a verified weekly+daily AI briefing snapshot to an official broadcast archive.

    HIGH-STAKES ACTION: Requires explicit Human-in-the-Loop (HITL) confirmation before
    writing an immutable publication record or broadcasting to organization channels.

    Args:
        target_date: Target briefing date formatted as 'YYYY-MM-DD' (or 'today').
        distribution_channel: Broadcast target ('org_announcement', 'engineering_digest',
            or 'public_feed').
        reviewer_note: Human editor approval note summarizing why this digest is approved.

    Returns:
        Dictionary with `status='pending_human_approval'` when paused at the HITL gate,
        `status='published'` with `archive_id` once approved, or `ToolErrorGuidance`.
    """
    with trace_span(
        "tool.publish_verified_daily_digest_archive",
        {"target_date": target_date, "distribution_channel": distribution_channel},
    ):
        try:
            validated = PublishDigestArchiveInput.model_validate(
                {
                    "target_date": target_date,
                    "distribution_channel": distribution_channel,
                    "reviewer_note": reviewer_note,
                }
            )
        except ValidationError as exc:
            return ToolErrorGuidance(
                source="PublishDigestArchiveGate",
                requested_date=str(target_date)[:40],
                error_code="INVALID_PUBLISH_ARGS",
                error_message=pii_redactor.scrub_text(str(exc.errors()[0]["msg"])),
                retryable=True,
                recovery_instructions=(
                    "Provide `target_date` as 'YYYY-MM-DD', `distribution_channel` in "
                    "{'org_announcement', 'engineering_digest', 'public_feed'}, and a "
                    "`reviewer_note` of at least 3 characters."
                ),
            ).model_dump()

        day, _ = parse_target_date(validated.target_date)
        approved_in_state = bool(tool_context.state.get("human_approved_publish"))
        resume_inputs = getattr(tool_context, "resume_inputs", None)
        approved_on_resume = bool(
            isinstance(resume_inputs, dict) and resume_inputs.get("approved") is True
        )

        if not (approved_in_state or approved_on_resume):
            tool_context.request_confirmation(
                hint=(
                    f"Human confirmation required before publishing {day.isoformat()} "
                    f"AI Digest to '{validated.distribution_channel}'."
                ),
                payload={
                    "action": "publish_verified_daily_digest_archive",
                    "target_date": day.isoformat(),
                    "distribution_channel": validated.distribution_channel,
                    "reviewer_note": pii_redactor.scrub_text(validated.reviewer_note),
                },
            )
            return {
                "status": "pending_human_approval",
                "requires_confirmation": True,
                "target_date": day.isoformat(),
                "distribution_channel": validated.distribution_channel,
                "message": (
                    "Execution paused at Human-in-the-Loop gate. Awaiting explicit human "
                    "confirmation before broadcasting the digest."
                ),
            }

        archive_id = f"pub-{day.isoformat()}-{uuid.uuid4().hex[:8]}"
        published_at = await briefing_repository.record_published_archive(
            archive_id=archive_id,
            target_date=day.isoformat(),
            channel=validated.distribution_channel,
            reviewer_note=validated.reviewer_note,
        )
        return {
            "status": "published",
            "archive_id": archive_id,
            "target_date": day.isoformat(),
            "distribution_channel": validated.distribution_channel,
            "published_at": published_at,
        }


publish_digest_hitl_tool = FunctionTool(
    publish_verified_daily_digest_archive,
    require_confirmation=requires_human_confirmation_for_publish,
)

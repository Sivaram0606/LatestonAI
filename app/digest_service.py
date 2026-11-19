"""Service layer orchestrating Sunday-Saturday weekly + daily ADK synthesis, citations, and async memory."""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import time
import uuid
from collections import Counter
from typing import Any

from google.adk.runners import Runner
from google.genai import types

from app.agent import MODEL, build_agentops_plugins, create_fast_editorial_agent
from app.app_utils import services
from app.guardrails import evaluate_briefing_self_check, normalize_citation_brackets
from app.model_router import TaskTier, model_router
from app.observability import (
    configure_observability,
    get_current_trace_id,
    trace_span,
)
from app.persistence import briefing_repository, memory_consolidator
from app.pii_redactor import pii_redactor
from app.schemas import (
    AgentStepTrace,
    DailyBriefingLLMOutput,
    DailyBriefingResponse,
    EditorialSynthesisLLMOutput,
    ModelBrief,
    NewsBrief,
    PaperBrief,
    RepoBrief,
)
from app.tools import (
    compute_sunday_to_saturday_window,
    fetch_arxiv_ai_papers,
    fetch_github_ai_repos,
    fetch_hackernews_ai_stories,
    fetch_hf_daily_papers,
    fetch_hf_trending_models_and_spaces,
    format_weekly_window_heading,
    parse_target_date,
)

configure_observability()
logger = logging.getLogger("app.digest_service")

_DIGEST_CACHE: dict[str, DailyBriefingResponse] = {}
_CITE_RE = re.compile(r"\[([PMNR]\d+)\]")


def clear_digest_cache() -> None:
    """Clear in-memory cached briefings (e.g. after updating LLM config)."""
    _DIGEST_CACHE.clear()


def _prune_dangling_citations(text: str, valid_ids: set[str]) -> str:
    """Normalize grouped citations and strip any out-of-range citation tokens."""
    expanded = normalize_citation_brackets(text)
    if not valid_ids:
        return expanded

    def _keep_valid(match: re.Match[str]) -> str:
        cid = match.group(1)
        return f"[{cid}]" if cid in valid_ids else ""

    cleaned = _CITE_RE.sub(_keep_valid, expanded)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _build_deterministic_whats_new_bullets(
    target_day: str,
    papers: list[PaperBrief],
    models: list[ModelBrief],
    news: list[NewsBrief],
    repos: list[RepoBrief],
) -> list[str]:
    """Build cited bullet-point updates highlighting what is new on `target_day`."""
    bullets: list[str] = []

    today_papers = [p for p in papers if p.published_date == target_day]
    p_pick = today_papers[0] if today_papers else (papers[0] if papers else None)
    if p_pick:
        date_tag = "New paper today" if p_pick.published_date == target_day else "Featured research update"
        bullets.append(
            f"{date_tag}: \"{p_pick.title}\" ({p_pick.source}"
            + (f", {p_pick.upvotes} upvotes" if p_pick.upvotes else "")
            + f") — {p_pick.summary[:165]} [{p_pick.citation_id}]"
        )
    if len(today_papers) > 1:
        p2 = today_papers[1]
        bullets.append(
            f"Additional preprint on {target_day}: \"{p2.title}\" by {p2.authors} — "
            f"{p2.summary[:150]} [{p2.citation_id}]"
        )

    today_models = [m for m in models if m.last_modified == target_day]
    m_pick = today_models[0] if today_models else (models[0] if models else None)
    if m_pick:
        bullets.append(
            f"Hugging Face {m_pick.kind.lower()} momentum: {m_pick.name} ({m_pick.pipeline_tag}, "
            f"{m_pick.likes:,} likes) is surging on the Hub [{m_pick.citation_id}]"
        )

    today_news = [n for n in news if n.published_date == target_day]
    n_pick = today_news[0] if today_news else (news[0] if news else None)
    if n_pick:
        bullets.append(
            f"Community discussion ({n_pick.points} pts, {n_pick.comments} comments): "
            f"\"{n_pick.title}\" via {n_pick.source} [{n_pick.citation_id}]"
        )

    today_repos = [r for r in repos if r.created_date == target_day]
    r_pick = today_repos[0] if today_repos else (repos[0] if repos else None)
    if r_pick:
        bullets.append(
            f"Open-source release traction: {r_pick.full_name} ({r_pick.language}, "
            f"{r_pick.stars:,} stars) — {r_pick.summary[:150]} [{r_pick.citation_id}]"
        )

    return bullets[:6]


def build_deterministic_briefing_from_raw(
    target_day: str,
    hf_papers_res: dict[str, Any],
    arxiv_res: dict[str, Any],
    models_res: dict[str, Any],
    hn_res: dict[str, Any],
    gh_res: dict[str, Any],
) -> DailyBriefingLLMOutput:
    """Build canonical items with `[P#]`/`[M#]`/`[N#]`/`[R#]` citations, Sun-Sat heading, and daily bullets."""
    day_obj, _ = parse_target_date(target_day)
    week_start, week_end = compute_sunday_to_saturday_window(day_obj)
    weekly_heading = format_weekly_window_heading(week_start, week_end)

    seen_titles: set[str] = set()
    papers: list[PaperBrief] = []

    for raw_p in [
        *(hf_papers_res.get("papers") or []),
        *(arxiv_res.get("papers") or []),
    ]:
        norm_title = str(raw_p.get("title") or "").strip().lower()
        if not norm_title or norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)
        papers.append(
            PaperBrief(
                citation_id=f"P{len(papers) + 1}",
                title=str(raw_p.get("title") or ""),
                authors=str(raw_p.get("authors") or "Research Community"),
                source=str(raw_p.get("source") or "AI Research"),
                url=str(raw_p.get("url") or "https://huggingface.co/papers"),
                published_date=str(raw_p.get("published_date") or target_day),
                upvotes=int(raw_p.get("upvotes") or 0),
                summary=str(raw_p.get("summary") or "Research preprint."),
                tags=[str(t) for t in (raw_p.get("tags") or ["AI"])[:4]],
            )
        )
        if len(papers) >= 8:
            break

    models: list[ModelBrief] = [
        ModelBrief(
            citation_id=f"M{idx}",
            name=str(m.get("name") or ""),
            kind=str(m.get("kind") or "Model"),
            author=str(m.get("author") or "hf"),
            pipeline_tag=str(m.get("pipeline_tag") or "foundation-model"),
            url=str(m.get("url") or "https://huggingface.co/models"),
            likes=int(m.get("likes") or 0),
            downloads=int(m.get("downloads") or 0),
            last_modified=str(m.get("last_modified") or ""),
            summary=str(m.get("summary") or ""),
        )
        for idx, m in enumerate((models_res.get("items") or [])[:8], 1)
    ]

    news: list[NewsBrief] = [
        NewsBrief(
            citation_id=f"N{idx}",
            title=str(s.get("title") or ""),
            source=str(s.get("source") or "Hacker News"),
            url=str(s.get("url") or ""),
            discussion_url=str(s.get("discussion_url") or s.get("url") or ""),
            published_date=str(s.get("published_date") or target_day),
            points=int(s.get("points") or 0),
            comments=int(s.get("comments") or 0),
            summary=str(s.get("summary") or ""),
        )
        for idx, s in enumerate((hn_res.get("stories") or [])[:8], 1)
    ]

    repos: list[RepoBrief] = [
        RepoBrief(
            citation_id=f"R{idx}",
            full_name=str(r.get("full_name") or ""),
            language=str(r.get("language") or "Python"),
            url=str(r.get("url") or ""),
            created_date=str(r.get("created_date") or ""),
            stars=int(r.get("stars") or 0),
            forks=int(r.get("forks") or 0),
            summary=str(r.get("summary") or ""),
            topics=[str(t) for t in (r.get("topics") or ["ai"])[:4]],
        )
        for idx, r in enumerate((gh_res.get("repos") or [])[:8], 1)
    ]

    tag_counter: Counter[str] = Counter()
    for p in papers:
        for t in p.tags:
            if t.lower() not in {"ai research", "cs.ai", "cs.lg", "cs.cl"}:
                tag_counter[t] += 2
    for m in models:
        if m.pipeline_tag:
            tag_counter[m.pipeline_tag.replace("-", " ").title()] += 1
    for r in repos:
        for t in r.topics:
            if t.lower() not in {"ai", "open-source"}:
                tag_counter[t] += 1

    top_tags = [k for k, _ in tag_counter.most_common(4)]
    key_themes = [
        f"Weekly research spotlight: {papers[0].title[:70]}"
        if papers
        else "Foundation model research",
        f"Hub momentum led by {models[0].name} ({models[0].likes:,} likes)"
        if models
        else "Open-weight model & Space activity",
        f"Community focus: {news[0].title[:70]}"
        if news
        else "Developer tooling and inference optimization",
        f"Open-source surge: {repos[0].full_name} ({repos[0].stars:,} stars)"
        if repos
        else "Agentic frameworks and developer SDKs",
    ]
    if top_tags:
        key_themes.append("Core technical threads: " + ", ".join(top_tags))

    week_range_text = f"{week_start.isoformat()} to {week_end.isoformat()}"
    lead_paper = (
        f"Across the Sunday-Saturday week ({week_range_text}), research momentum is anchored by "
        f"'{papers[0].title}' ({papers[0].source}"
        + (f", {papers[0].upvotes} upvotes" if papers[0].upvotes else "")
        + f") [{papers[0].citation_id}]"
        + (
            f" alongside '{papers[1].title}' [{papers[1].citation_id}]."
            if len(papers) > 1
            else "."
        )
        + f" {papers[0].summary[:200]}"
        if papers
        else f"Research feeds across {week_range_text} were queried on Hugging Face and arXiv."
    )
    lead_models = (
        f"On the Hugging Face Hub this week, {len(models)} standout models and interactive Spaces "
        f"are led by {models[0].name} ({models[0].pipeline_tag}, {models[0].likes:,} likes) "
        f"[{models[0].citation_id}]"
        + (
            f" and {models[1].name} [{models[1].citation_id}]."
            if len(models) > 1
            else "."
        )
        if models
        else "Hugging Face Hub model and Space feeds were scanned for weekly releases."
    )
    lead_ecosystem = (
        f"Across engineering discussions and open-source repositories during {week_range_text}, "
        + (
            f"Hacker News highlighted '{news[0].title}' ({news[0].points} pts, "
            f"{news[0].comments} comments) [{news[0].citation_id}], "
            if news
            else ""
        )
        + (
            f"while GitHub saw strong breakout velocity around {repos[0].full_name} "
            f"({repos[0].stars:,} stars, {repos[0].language}) [{repos[0].citation_id}]: "
            f"{repos[0].summary[:170]}"
            if repos
            else "alongside active open-source AI repositories."
        )
    )

    headline_subject = (
        papers[0].title.split(":")[0]
        if papers
        else (news[0].title if news else f"AI Ecosystem Week of {week_start.isoformat()}")
    )
    headline = (
        f"Week of {week_start.strftime('%b %d')}-{week_end.strftime('%b %d, %Y')} "
        f"({target_day}): {headline_subject}"
    )

    whats_new_bullets = _build_deterministic_whats_new_bullets(
        target_day=target_day,
        papers=papers,
        models=models,
        news=news,
        repos=repos,
    )

    return DailyBriefingLLMOutput(
        weekly_window_heading=weekly_heading,
        headline=headline[:150],
        executive_summary=f"{lead_paper}\n\n{lead_models}\n\n{lead_ecosystem}",
        whats_new_today=whats_new_bullets,
        key_themes=key_themes[:5],
        papers=papers,
        models=models,
        news=news,
        repos=repos,
    )


def _format_compact_digest_context(
    base: DailyBriefingLLMOutput, target_date: str
) -> str:
    """Format compact cited context distinguishing latest/today items within the Sun-Sat week."""
    paper_dates = [p.published_date for p in base.papers if p.published_date]
    news_dates = [n.published_date for n in base.news if n.published_date]
    latest_paper_day = max(paper_dates) if paper_dates else target_date
    latest_news_day = max(news_dates) if news_dates else target_date

    lines: list[str] = ["=== RESEARCH PAPERS (Section: Papers) ==="]
    for p in base.papers:
        is_today = (
            " [NEW TODAY]"
            if p.published_date in {target_date, latest_paper_day}
            else ""
        )
        lines.append(
            f"[{p.citation_id}] ({p.published_date}){is_today} {p.title} "
            f"({p.source}, {p.upvotes} upvotes) — {p.summary[:220]}"
        )
    lines.append("\n=== MODELS & SPACES (Section: Models & Spaces) ===")
    for m in base.models:
        is_today = " [UPDATED TODAY]" if m.last_modified == target_date else ""
        lines.append(
            f"[{m.citation_id}] ({m.last_modified or 'weekly'}){is_today} {m.name} "
            f"[{m.kind}, {m.pipeline_tag}, {m.likes} likes, {m.downloads} dl] — {m.summary[:170]}"
        )
    lines.append("\n=== COMMUNITY NEWS (Section: Community News) ===")
    for n in base.news:
        is_today = (
            " [NEW TODAY]"
            if n.published_date in {target_date, latest_news_day}
            else ""
        )
        lines.append(
            f"[{n.citation_id}] ({n.published_date}){is_today} {n.title} "
            f"({n.source}, {n.points} pts, {n.comments} comments) — {n.summary[:150]}"
        )
    lines.append("\n=== GITHUB REPOS (Section: Breakout Repositories) ===")
    for r in base.repos:
        is_today = " [ACTIVE TODAY]" if r.created_date == target_date else ""
        lines.append(
            f"[{r.citation_id}] ({r.created_date or 'weekly'}){is_today} {r.full_name} "
            f"({r.language}, {r.stars} stars, topics={','.join(r.topics)}) — {r.summary[:170]}"
        )
    return "\n".join(lines)


async def _run_adk_editorial_pass(
    normalized_date: str,
    weekly_window_heading: str,
    context_text: str,
    minimal_thinking: bool = True,
    tier: TaskTier = TaskTier.STANDARD_SYNTHESIS,
) -> tuple[EditorialSynthesisLLMOutput, int, Any, str]:
    """Execute single-pass ADK Runner editorial synthesis with persistent session + plugins."""
    t0 = time.perf_counter()
    session_service = services.get_session_service()
    memory_service = services.get_memory_service()
    session_id = f"digest-{normalized_date}-{uuid.uuid4().hex[:8]}"
    user_id = "web_dashboard_user"

    route = model_router.resolve(tier=tier, prefer_minimal_thinking=minimal_thinking)
    agent = create_fast_editorial_agent(minimal_thinking=minimal_thinking, tier=tier)
    runner = Runner(
        agent=agent,
        app_name="app",
        plugins=build_agentops_plugins(),
        session_service=session_service,
        memory_service=memory_service,
        auto_create_session=True,
    )

    await session_service.create_session(
        app_name="app",
        user_id=user_id,
        session_id=session_id,
        state={
            "target_date": normalized_date,
            "weekly_window_heading": weekly_window_heading,
            "collected_digest_context": pii_redactor.scrub_text(context_text),
        },
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        f"Synthesize the Sunday-to-Saturday weekly summary ({weekly_window_heading}) "
                        f"and bulleted What's New Today updates for {normalized_date} with citations."
                    )
                )
            ],
        ),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text

    session = await session_service.get_session(
        app_name="app", user_id=user_id, session_id=session_id
    )
    raw_output = session.state.get("editorial_synthesis") if session else None

    if isinstance(raw_output, EditorialSynthesisLLMOutput):
        editorial = raw_output
    elif isinstance(raw_output, dict):
        editorial = EditorialSynthesisLLMOutput.model_validate(raw_output)
    elif isinstance(raw_output, str) and raw_output.strip():
        editorial = EditorialSynthesisLLMOutput.model_validate_json(raw_output)
    elif final_text.strip():
        editorial = EditorialSynthesisLLMOutput.model_validate_json(final_text)
    else:
        raise RuntimeError(
            "ADK synthesizer finished without structured editorial output."
        )

    dur_ms = max(1, int((time.perf_counter() - t0) * 1000))
    return editorial, dur_ms, session, route.model_name


def _merge_editorial_into_briefing(
    base: DailyBriefingLLMOutput, editorial: EditorialSynthesisLLMOutput
) -> DailyBriefingLLMOutput:
    """Merge Gemini's weekly summary, daily bullets, and item takeaways with verified citations."""
    papers = [
        p.model_copy(
            update={
                "summary": pii_redactor.scrub_text(editorial.paper_summaries[i].strip())
                if i < len(editorial.paper_summaries)
                and editorial.paper_summaries[i].strip()
                else p.summary
            }
        )
        for i, p in enumerate(base.papers)
    ]
    models = [
        m.model_copy(
            update={
                "summary": pii_redactor.scrub_text(editorial.model_summaries[i].strip())
                if i < len(editorial.model_summaries)
                and editorial.model_summaries[i].strip()
                else m.summary
            }
        )
        for i, m in enumerate(base.models)
    ]
    news = [
        n.model_copy(
            update={
                "summary": pii_redactor.scrub_text(editorial.news_summaries[i].strip())
                if i < len(editorial.news_summaries)
                and editorial.news_summaries[i].strip()
                else n.summary
            }
        )
        for i, n in enumerate(base.news)
    ]
    repos = [
        r.model_copy(
            update={
                "summary": pii_redactor.scrub_text(editorial.repo_summaries[i].strip())
                if i < len(editorial.repo_summaries)
                and editorial.repo_summaries[i].strip()
                else r.summary
            }
        )
        for i, r in enumerate(base.repos)
    ]

    valid_ids: set[str] = {
        *(p.citation_id for p in papers),
        *(m.citation_id for m in models),
        *(n.citation_id for n in news),
        *(r.citation_id for r in repos),
    }
    fallback_cites = [
        cid for cid in ("P1", "M1", "N1", "R1") if cid in valid_ids
    ]

    exec_summary = _prune_dangling_citations(
        pii_redactor.scrub_text(editorial.executive_summary.strip()), valid_ids
    )
    if not exec_summary:
        exec_summary = base.executive_summary
    elif fallback_cites and not _CITE_RE.search(exec_summary):
        exec_summary = (
            exec_summary
            + " Key references: "
            + " ".join(f"[{c}]" for c in fallback_cites)
            + "."
        )

    cleaned_bullets: list[str] = []
    for idx_b, raw_b in enumerate(editorial.whats_new_today):
        stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_b or "").strip()
        if not stripped:
            continue
        sanitized = _prune_dangling_citations(
            pii_redactor.scrub_text(stripped), valid_ids
        )
        if fallback_cites and not _CITE_RE.search(sanitized):
            sanitized = f"{sanitized} [{fallback_cites[idx_b % len(fallback_cites)]}]"
        cleaned_bullets.append(sanitized)

    return DailyBriefingLLMOutput(
        weekly_window_heading=base.weekly_window_heading,
        headline=pii_redactor.scrub_text(editorial.headline.strip()) or base.headline,
        executive_summary=exec_summary,
        whats_new_today=cleaned_bullets[:6] or base.whats_new_today,
        key_themes=[
            pii_redactor.scrub_text(t.strip())
            for t in editorial.key_themes
            if t.strip()
        ][:5]
        or base.key_themes,
        papers=papers,
        models=models,
        news=news,
        repos=repos,
    )


async def _timed_call(coro: Any) -> tuple[dict[str, Any], int]:
    t0 = time.perf_counter()
    res = await coro
    return res, max(1, int((time.perf_counter() - t0) * 1000))


async def generate_daily_briefing(
    target_date: str,
    force_refresh: bool = False,
    analysis_depth: str = "standard",
) -> DailyBriefingResponse:
    """Generate or retrieve weekly (Sun-Sat) + daily AI briefing with citations & OTel tracing."""
    day, date_note = parse_target_date(target_date)
    normalized_date = day.isoformat()
    week_start, week_end = compute_sunday_to_saturday_window(day)

    with trace_span(
        "adk.generate_daily_briefing",
        {
            "target_date": normalized_date,
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "force_refresh": force_refresh,
            "analysis_depth": analysis_depth,
        },
    ):
        trace_id = get_current_trace_id()
        logger.info(
            "Weekly + daily briefing pipeline intent initiated",
            extra={
                "event_type": "digest.pipeline",
                "phase": "INTENT",
                "target_date": normalized_date,
                "week_window": f"{week_start.isoformat()}..{week_end.isoformat()}",
                "force_refresh": force_refresh,
                "analysis_depth": analysis_depth,
            },
        )

        if not force_refresh and normalized_date in _DIGEST_CACHE:
            cached_mem = _DIGEST_CACHE[normalized_date]
            if cached_mem.briefing.weekly_window_heading:
                logger.info(
                    "Briefing served from in-memory cache",
                    extra={
                        "event_type": "digest.pipeline",
                        "phase": "OUTCOME",
                        "target_date": normalized_date,
                        "outcome_status": "cache_hit_memory",
                    },
                )
                return cached_mem

        if not force_refresh:
            persisted = await briefing_repository.get_briefing(normalized_date)
            if (
                persisted
                and persisted.generation_mode == "adk_multi_agent"
                and persisted.briefing.weekly_window_heading
            ):
                _DIGEST_CACHE[normalized_date] = persisted
                logger.info(
                    "Briefing restored from persistent SQLite store",
                    extra={
                        "event_type": "digest.pipeline",
                        "phase": "OUTCOME",
                        "target_date": normalized_date,
                        "outcome_status": "cache_hit_sqlite",
                    },
                )
                return persisted

        with trace_span("adk.collect_external_apis", {"target_date": normalized_date}):
            (
                (hf_papers_res, hf_ms),
                (arxiv_res, arxiv_ms),
                (models_res, models_ms),
                (hn_res, hn_ms),
                (gh_res, gh_ms),
            ) = await asyncio.gather(
                _timed_call(fetch_hf_daily_papers(normalized_date)),
                _timed_call(fetch_arxiv_ai_papers(normalized_date)),
                _timed_call(fetch_hf_trending_models_and_spaces(normalized_date)),
                _timed_call(fetch_hackernews_ai_stories(normalized_date)),
                _timed_call(fetch_github_ai_repos(normalized_date)),
            )

        base_briefing = build_deterministic_briefing_from_raw(
            target_day=normalized_date,
            hf_papers_res=hf_papers_res,
            arxiv_res=arxiv_res,
            models_res=models_res,
            hn_res=hn_res,
            gh_res=gh_res,
        )

        allowed_urls: set[str] = {
            *(p.url for p in base_briefing.papers),
            *(m.url for m in base_briefing.models),
            *(n.url for n in base_briefing.news),
            *(r.url for r in base_briefing.repos),
        }

        notes = [
            n
            for n in (
                date_note,
                hf_papers_res.get("note", ""),
                models_res.get("note", ""),
            )
            if n
        ]

        collector_traces = [
            AgentStepTrace(
                agent_name="papers_research_agent",
                role="Research Papers Collector (Sun-Sat Week + Daily HF & arXiv)",
                status="completed",
                duration_ms=max(hf_ms, arxiv_ms),
                items_collected=len(base_briefing.papers),
                detail=(
                    f"Fetched {hf_papers_res.get('count', 0)} HF Papers + "
                    f"{arxiv_res.get('count', 0)} arXiv preprints across Sun-Sat window."
                ),
            ),
            AgentStepTrace(
                agent_name="models_spaces_agent",
                role="Models & Spaces Scout (Hugging Face Hub API)",
                status="completed",
                duration_ms=models_ms,
                items_collected=len(base_briefing.models),
                detail=f"Fetched {models_res.get('count', 0)} trending Models & Spaces from HF Hub.",
            ),
            AgentStepTrace(
                agent_name="news_repos_agent",
                role="Ecosystem News & Repos Scout (Sun-Sat + Daily HN & GitHub)",
                status="completed",
                duration_ms=max(hn_ms, gh_ms),
                items_collected=len(base_briefing.news) + len(base_briefing.repos),
                detail=(
                    f"Fetched {hn_res.get('count', 0)} HN stories + "
                    f"{gh_res.get('count', 0)} GitHub breakout repos."
                ),
            ),
        ]

        total_items = (
            len(base_briefing.papers)
            + len(base_briefing.models)
            + len(base_briefing.news)
            + len(base_briefing.repos)
        )

        selected_tier = (
            TaskTier.DEEP_STRATEGIC_PLANNING
            if analysis_depth == "deep"
            else TaskTier.STANDARD_SYNTHESIS
        )

        try:
            context_text = _format_compact_digest_context(
                base_briefing, target_date=normalized_date
            )
            with trace_span(
                "adk.editorial_synthesis",
                {"tier": str(selected_tier), "target_date": normalized_date},
            ):
                try:
                    editorial, synth_ms, session, resolved_model = (
                        await _run_adk_editorial_pass(
                            normalized_date=normalized_date,
                            weekly_window_heading=base_briefing.weekly_window_heading,
                            context_text=context_text,
                            minimal_thinking=True,
                            tier=selected_tier,
                        )
                    )
                except Exception as first_exc:
                    if selected_tier == TaskTier.DEEP_STRATEGIC_PLANNING or (
                        "thinking" in str(first_exc).lower()
                    ):
                        editorial, synth_ms, session, resolved_model = (
                            await _run_adk_editorial_pass(
                                normalized_date=normalized_date,
                                weekly_window_heading=base_briefing.weekly_window_heading,
                                context_text=context_text,
                                minimal_thinking=False,
                                tier=TaskTier.STANDARD_SYNTHESIS,
                            )
                        )
                    else:
                        raise

            final_briefing = _merge_editorial_into_briefing(base_briefing, editorial)
            with trace_span("adk.self_eval_guardrail"):
                verdict = evaluate_briefing_self_check(
                    final_briefing, allowed_urls=allowed_urls
                )

            traces = [
                *collector_traces,
                AgentStepTrace(
                    agent_name="digest_synthesizer_agent",
                    role=f"Editor-in-Chief Synthesizer ({resolved_model})",
                    status="completed",
                    duration_ms=synth_ms,
                    items_collected=total_items,
                    detail=(
                        f"Synthesized Sun-Sat weekly + daily bullets with {resolved_model} "
                        f"({verdict.verified_citation_count} citations verified)."
                    ),
                ),
            ]
            response = DailyBriefingResponse(
                target_date=normalized_date,
                week_start_date=week_start.isoformat(),
                week_end_date=week_end.isoformat(),
                generated_at=datetime.datetime.now(datetime.UTC).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                ),
                generation_mode="adk_multi_agent",
                model_used=resolved_model,
                notice=" ".join(notes),
                briefing=final_briefing,
                agent_trace=traces,
                self_eval=verdict,
                trace_id=trace_id,
            )
            _DIGEST_CACHE[normalized_date] = response

            memory_consolidator.schedule_briefing_and_memory_consolidation(
                response=response,
                session=session,
                memory_service=services.get_memory_service(),
            )
            logger.info(
                "Weekly + daily briefing synthesized via ADK pipeline",
                extra={
                    "event_type": "digest.pipeline",
                    "phase": "OUTCOME",
                    "target_date": normalized_date,
                    "week_window": f"{week_start.isoformat()}..{week_end.isoformat()}",
                    "outcome_status": "adk_multi_agent",
                    "model_used": resolved_model,
                    "grounding_score": verdict.grounding_score,
                    "verified_citations": verdict.verified_citation_count,
                },
            )
            return response
        except Exception as adk_exc:
            root_exc: BaseException = adk_exc
            while hasattr(root_exc, "exceptions") and root_exc.exceptions:
                root_exc = root_exc.exceptions[0]
            short_err = pii_redactor.scrub_text(str(root_exc).splitlines()[0][:160])

            logger.warning(
                "ADK LLM synthesis unavailable; returning deterministic weekly+daily briefing",
                extra={
                    "event_type": "digest.pipeline",
                    "phase": "OUTCOME",
                    "target_date": normalized_date,
                    "outcome_status": "live_api_fallback",
                    "error_message": short_err,
                },
            )
            notes.append(
                f"Live AI APIs succeeded; Gemini ({MODEL}) synthesis fell back to deterministic "
                f"aggregation ({short_err}). Configure your GCP Project + ADC or Secret Manager "
                f"credentials in 'LLM Settings' for full generative editorial prose."
            )
            verdict = evaluate_briefing_self_check(
                base_briefing, allowed_urls=allowed_urls
            )
            traces = [
                *collector_traces,
                AgentStepTrace(
                    agent_name="digest_synthesizer_agent",
                    role="Deterministic Fallback Synthesizer (Live API Data)",
                    status="fallback",
                    duration_ms=5,
                    items_collected=total_items,
                    detail="Structured live API weekly + daily updates while awaiting LLM credentials.",
                ),
            ]
            return DailyBriefingResponse(
                target_date=normalized_date,
                week_start_date=week_start.isoformat(),
                week_end_date=week_end.isoformat(),
                generated_at=datetime.datetime.now(datetime.UTC).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                ),
                generation_mode="live_api_fallback",
                model_used=f"{MODEL} (fallback active)",
                notice=" ".join(notes),
                briefing=base_briefing,
                agent_trace=traces,
                self_eval=verdict,
                trace_id=trace_id,
            )

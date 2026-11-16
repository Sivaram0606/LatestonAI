"""Service layer running the fast parallel-collector + ADK synthesis pipeline."""

from __future__ import annotations

import asyncio
from collections import Counter
import datetime
import logging
import time
import uuid
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import MODEL, create_fast_editorial_agent
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
    fetch_arxiv_ai_papers,
    fetch_github_ai_repos,
    fetch_hackernews_ai_stories,
    fetch_hf_daily_papers,
    fetch_hf_trending_models_and_spaces,
    parse_target_date,
)

logger = logging.getLogger(__name__)

_DIGEST_CACHE: dict[str, DailyBriefingResponse] = {}


def clear_digest_cache() -> None:
    """Clear in-memory cached daily briefings (e.g. after updating LLM config)."""
    _DIGEST_CACHE.clear()


def build_deterministic_briefing_from_raw(
    target_day: str,
    hf_papers_res: dict[str, Any],
    arxiv_res: dict[str, Any],
    models_res: dict[str, Any],
    hn_res: dict[str, Any],
    gh_res: dict[str, Any],
) -> DailyBriefingLLMOutput:
    """Build canonical items and baseline briefing directly from live API payloads."""
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
            name=str(m.get("name") or ""),
            kind=str(m.get("kind") or "Model"),
            author=str(m.get("author") or "hf"),
            pipeline_tag=str(m.get("pipeline_tag") or "foundation-model"),
            url=str(m.get("url") or "https://huggingface.co/models"),
            likes=int(m.get("likes") or 0),
            downloads=int(m.get("downloads") or 0),
            summary=str(m.get("summary") or ""),
        )
        for m in (models_res.get("items") or [])[:8]
    ]

    news: list[NewsBrief] = [
        NewsBrief(
            title=str(s.get("title") or ""),
            source=str(s.get("source") or "Hacker News"),
            url=str(s.get("url") or ""),
            discussion_url=str(s.get("discussion_url") or s.get("url") or ""),
            points=int(s.get("points") or 0),
            comments=int(s.get("comments") or 0),
            summary=str(s.get("summary") or ""),
        )
        for s in (hn_res.get("stories") or [])[:8]
    ]

    repos: list[RepoBrief] = [
        RepoBrief(
            full_name=str(r.get("full_name") or ""),
            language=str(r.get("language") or "Python"),
            url=str(r.get("url") or ""),
            stars=int(r.get("stars") or 0),
            forks=int(r.get("forks") or 0),
            summary=str(r.get("summary") or ""),
            topics=[str(t) for t in (r.get("topics") or ["ai"])[:4]],
        )
        for r in (gh_res.get("repos") or [])[:8]
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
        f"Research spotlight: {papers[0].title[:75]}" if papers else "Foundation model research",
        f"Hub momentum led by {models[0].name} ({models[0].likes:,} likes)"
        if models
        else "Open-weight model & Space activity",
        f"Community focus: {news[0].title[:75]}"
        if news
        else "Developer tooling and inference optimization",
        f"Open-source surge: {repos[0].full_name} ({repos[0].stars:,} stars)"
        if repos
        else "Agentic frameworks and developer SDKs",
    ]
    if top_tags:
        key_themes.append("Core technical threads: " + ", ".join(top_tags))

    lead_paper = (
        f"In research, '{papers[0].title}' ({papers[0].source}"
        + (f", {papers[0].upvotes} upvotes" if papers[0].upvotes else "")
        + f") leads {len(papers)} curated papers for {target_day}. {papers[0].summary[:220]}"
        if papers
        else f"Research feeds for {target_day} were queried across Hugging Face Daily Papers and arXiv."
    )
    lead_models = (
        f"Across the Hugging Face Hub, {len(models)} trending models and interactive Spaces stand out, "
        f"headlined by {models[0].name} ({models[0].pipeline_tag}, {models[0].likes:,} likes)"
        + (f" alongside {models[1].name}." if len(models) > 1 else ".")
        if models
        else "Hugging Face Hub model and Space feeds were scanned for active releases."
    )
    lead_ecosystem = (
        f"In community and open-source engineering, Hacker News surfaced {len(news)} notable AI discussions"
        + (
            f"—most prominently '{news[0].title}' ({news[0].points} pts, {news[0].comments} comments)—"
            if news
            else "—"
        )
        + (
            f"while GitHub saw strong traction around {repos[0].full_name} "
            f"({repos[0].stars:,} stars, {repos[0].language}): {repos[0].summary[:180]}"
            if repos
            else "alongside active open-source AI repositories."
        )
    )

    headline_subject = (
        papers[0].title.split(":")[0]
        if papers
        else (news[0].title if news else f"AI Ecosystem Briefing for {target_day}")
    )
    headline = f"{target_day} AI Pulse: {headline_subject}"

    return DailyBriefingLLMOutput(
        headline=headline[:140],
        executive_summary=f"{lead_paper}\n\n{lead_models}\n\n{lead_ecosystem}",
        key_themes=key_themes[:5],
        papers=papers,
        models=models,
        news=news,
        repos=repos,
    )


def _format_compact_digest_context(base: DailyBriefingLLMOutput) -> str:
    """Format compact bullet context so the ADK synthesizer processes minimal input tokens."""
    lines: list[str] = ["=== PAPERS ==="]
    for idx, p in enumerate(base.papers, 1):
        lines.append(
            f"{idx}. {p.title} ({p.source}, {p.upvotes} upvotes) — {p.summary[:240]}"
        )
    lines.append("\n=== MODELS & SPACES ===")
    for idx, m in enumerate(base.models, 1):
        lines.append(
            f"{idx}. {m.name} [{m.kind}, {m.pipeline_tag}, {m.likes} likes, {m.downloads} downloads] — {m.summary[:180]}"
        )
    lines.append("\n=== COMMUNITY NEWS ===")
    for idx, n in enumerate(base.news, 1):
        lines.append(
            f"{idx}. {n.title} ({n.source}, {n.points} pts, {n.comments} comments) — {n.summary[:160]}"
        )
    lines.append("\n=== GITHUB REPOS ===")
    for idx, r in enumerate(base.repos, 1):
        lines.append(
            f"{idx}. {r.full_name} ({r.language}, {r.stars} stars, topics={','.join(r.topics)}) — {r.summary[:180]}"
        )
    return "\n".join(lines)


async def _run_adk_editorial_pass(
    normalized_date: str,
    context_text: str,
    minimal_thinking: bool = True,
) -> tuple[EditorialSynthesisLLMOutput, int]:
    """Execute single-pass ADK Runner editorial synthesis with Gemini."""
    t0 = time.perf_counter()
    session_service = InMemorySessionService()
    session_id = f"digest-{normalized_date}-{uuid.uuid4().hex[:8]}"
    user_id = "web_dashboard_user"

    agent = create_fast_editorial_agent(minimal_thinking=minimal_thinking)
    runner = Runner(
        agent=agent,
        app_name="app",
        session_service=session_service,
        auto_create_session=True,
    )

    await session_service.create_session(
        app_name="app",
        user_id=user_id,
        session_id=session_id,
        state={
            "target_date": normalized_date,
            "collected_digest_context": context_text,
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
                    text=f"Synthesize the daily AI intelligence briefing for {normalized_date}."
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
        raise RuntimeError("ADK synthesizer finished without structured editorial output.")

    dur_ms = max(1, int((time.perf_counter() - t0) * 1000))
    return editorial, dur_ms


def _merge_editorial_into_briefing(
    base: DailyBriefingLLMOutput, editorial: EditorialSynthesisLLMOutput
) -> DailyBriefingLLMOutput:
    """Merge Gemini's editorial narrative and concise item takeaways onto canonical API items."""
    papers = [
        p.model_copy(
            update={
                "summary": editorial.paper_summaries[i].strip()
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
                "summary": editorial.model_summaries[i].strip()
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
                "summary": editorial.news_summaries[i].strip()
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
                "summary": editorial.repo_summaries[i].strip()
                if i < len(editorial.repo_summaries)
                and editorial.repo_summaries[i].strip()
                else r.summary
            }
        )
        for i, r in enumerate(base.repos)
    ]

    return DailyBriefingLLMOutput(
        headline=editorial.headline.strip() or base.headline,
        executive_summary=editorial.executive_summary.strip() or base.executive_summary,
        key_themes=[t.strip() for t in editorial.key_themes if t.strip()][:5]
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
    target_date: str, force_refresh: bool = False
) -> DailyBriefingResponse:
    """Generate or return cached daily AI briefing for target_date."""
    day, date_note = parse_target_date(target_date)
    normalized_date = day.isoformat()

    if not force_refresh and normalized_date in _DIGEST_CACHE:
        return _DIGEST_CACHE[normalized_date]

    # Stage 1: Fetch all 5 live AI APIs concurrently (~1-2s total, zero LLM round-trip waste)
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
            role="Research Papers Collector (Hugging Face Daily Papers + arXiv)",
            status="completed",
            duration_ms=max(hf_ms, arxiv_ms),
            items_collected=len(base_briefing.papers),
            detail=(
                f"Fetched {hf_papers_res.get('count', 0)} HF Daily Papers + "
                f"{arxiv_res.get('count', 0)} arXiv preprints concurrently."
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
            role="Ecosystem News & Repos Scout (Hacker News + GitHub API)",
            status="completed",
            duration_ms=max(hn_ms, gh_ms),
            items_collected=len(base_briefing.news) + len(base_briefing.repos),
            detail=(
                f"Fetched {hn_res.get('count', 0)} HN AI stories + "
                f"{gh_res.get('count', 0)} GitHub breakout repos concurrently."
            ),
        ),
    ]

    total_items = (
        len(base_briefing.papers)
        + len(base_briefing.models)
        + len(base_briefing.news)
        + len(base_briefing.repos)
    )

    # Stage 2: Single-pass ADK Runner editorial synthesis with Gemini (MINIMAL thinking)
    try:
        context_text = _format_compact_digest_context(base_briefing)
        try:
            editorial, synth_ms = await _run_adk_editorial_pass(
                normalized_date, context_text, minimal_thinking=True
            )
        except Exception as first_exc:
            if "thinking" in str(first_exc).lower():
                editorial, synth_ms = await _run_adk_editorial_pass(
                    normalized_date, context_text, minimal_thinking=False
                )
            else:
                raise

        final_briefing = _merge_editorial_into_briefing(base_briefing, editorial)
        traces = [
            *collector_traces,
            AgentStepTrace(
                agent_name="digest_synthesizer_agent",
                role="Editor-in-Chief Synthesizer (ADK Gemini Single-Pass)",
                status="completed",
                duration_ms=synth_ms,
                items_collected=total_items,
                detail=f"Synthesized structured briefing with {MODEL} (minimal thinking).",
            ),
        ]
        response = DailyBriefingResponse(
            target_date=normalized_date,
            generated_at=datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            generation_mode="adk_multi_agent",
            model_used=MODEL,
            notice=" ".join(notes),
            briefing=final_briefing,
            agent_trace=traces,
        )
        _DIGEST_CACHE[normalized_date] = response
        return response
    except Exception as adk_exc:
        logger.warning(
            "ADK LLM synthesis unavailable (%s); returning deterministic live-API briefing.",
            adk_exc,
        )
        root_exc: BaseException = adk_exc
        while hasattr(root_exc, "exceptions") and getattr(root_exc, "exceptions"):
            root_exc = root_exc.exceptions[0]
        short_err = str(root_exc).splitlines()[0][:160]
        notes.append(
            f"Live AI APIs succeeded; Gemini ({MODEL}) synthesis fell back to deterministic "
            f"aggregation ({short_err}). Configure your GCP Project + ADC or GEMINI_API_KEY "
            f"in 'LLM Settings' for full generative editorial prose."
        )
        traces = [
            *collector_traces,
            AgentStepTrace(
                agent_name="digest_synthesizer_agent",
                role="Deterministic Fallback Synthesizer (Live API Data)",
                status="fallback",
                duration_ms=5,
                items_collected=total_items,
                detail="Structured live API results while awaiting Vertex AI / Gemini credentials.",
            ),
        ]
        return DailyBriefingResponse(
            target_date=normalized_date,
            generated_at=datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            generation_mode="live_api_fallback",
            model_used=f"{MODEL} (fallback active)",
            notice=" ".join(notes),
            briefing=base_briefing,
            agent_trace=traces,
        )

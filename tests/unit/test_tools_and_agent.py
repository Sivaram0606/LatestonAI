"""Unit tests for Sun-Sat weekly window, daily bullets, citations, and multi-agent topology."""

import datetime

from google.adk.agents import ParallelAgent, SequentialAgent

from app.agent import MODEL, create_fast_editorial_agent, create_root_agent
from app.digest_service import (
    _merge_editorial_into_briefing,
    build_deterministic_briefing_from_raw,
)
from app.guardrails import evaluate_briefing_self_check
from app.schemas import EditorialSynthesisLLMOutput
from app.tools import compute_sunday_to_saturday_window, parse_target_date


def test_parse_target_date_and_sunday_saturday_window() -> None:
    today = datetime.datetime.now(datetime.UTC).date()
    d_today, note_today = parse_target_date("today")
    assert d_today == today
    assert note_today == ""

    d_past, note_past = parse_target_date("2025-02-14")
    assert d_past == datetime.date(2025, 2, 14)
    assert note_past == ""

    # 2025-02-14 is Friday -> Sunday 2025-02-09 through Saturday 2025-02-15
    sun, sat = compute_sunday_to_saturday_window(d_past)
    assert sun == datetime.date(2025, 2, 9)
    assert sat == datetime.date(2025, 2, 15)
    assert sun.weekday() == 6  # Sunday
    assert sat.weekday() == 5  # Saturday

    # Boundary check when target date itself is Sunday or Saturday
    assert compute_sunday_to_saturday_window(datetime.date(2026, 11, 15)) == (
        datetime.date(2026, 11, 15),
        datetime.date(2026, 11, 21),
    )
    assert compute_sunday_to_saturday_window(datetime.date(2026, 11, 21)) == (
        datetime.date(2026, 11, 15),
        datetime.date(2026, 11, 21),
    )

    future_str = (today + datetime.timedelta(days=10)).isoformat()
    d_future, note_future = parse_target_date(future_str)
    assert d_future == today
    assert "future" in note_future.lower()


def test_adk_multi_agent_topology() -> None:
    root = create_root_agent()
    assert isinstance(root, SequentialAgent)
    assert root.name == "daily_digest_pipeline"
    assert len(root.sub_agents) == 2

    parallel_stage = root.sub_agents[0]
    synthesizer_stage = root.sub_agents[1]
    assert isinstance(parallel_stage, ParallelAgent)
    assert len(parallel_stage.sub_agents) == 3
    collector_names = {a.name for a in parallel_stage.sub_agents}
    assert collector_names == {
        "papers_research_agent",
        "models_spaces_agent",
        "news_repos_agent",
    }
    assert synthesizer_stage.name == "digest_synthesizer_agent"
    assert MODEL == "gemini-3.7-flash"


def test_weekly_summary_whats_new_bullets_and_section_citations() -> None:
    briefing = build_deterministic_briefing_from_raw(
        target_day="2025-02-14",
        hf_papers_res={
            "papers": [
                {
                    "title": "Scaling Test-Time Compute for Agentic Reasoning",
                    "authors": "Alice Chen, Bob Smith",
                    "source": "Hugging Face Daily Papers",
                    "url": "https://huggingface.co/papers/2502.00001",
                    "published_date": "2025-02-14",
                    "upvotes": 128,
                    "summary": "Demonstrates parallel verification search for tool-using LLM agents.",
                    "tags": ["Agents", "Reasoning"],
                }
            ]
        },
        arxiv_res={
            "papers": [
                {
                    "title": "scaling test-time compute for agentic reasoning",
                    "authors": "Alice Chen",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/2502.00001",
                    "published_date": "2025-02-14",
                    "upvotes": 0,
                    "summary": "Duplicate title check.",
                    "tags": ["cs.AI"],
                },
                {
                    "title": "Sparse Multimodal Attention on Edge Accelerators",
                    "authors": "Priya Nair",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/2502.00002",
                    "published_date": "2025-02-12",
                    "upvotes": 0,
                    "summary": "Reduces KV-cache memory by 4x for vision-language models.",
                    "tags": ["Multimodal", "Efficiency"],
                },
            ]
        },
        models_res={
            "items": [
                {
                    "name": "google/gemma-next-instruct",
                    "kind": "Model",
                    "author": "google",
                    "pipeline_tag": "text-generation",
                    "url": "https://huggingface.co/google/gemma-next-instruct",
                    "likes": 1420,
                    "downloads": 85000,
                    "last_modified": "2025-02-14",
                    "summary": "Open-weight instruction-tuned model.",
                }
            ]
        },
        hn_res={
            "stories": [
                {
                    "title": "Show HN: Open-source MCP server for local agent sandboxes",
                    "source": "github.com",
                    "url": "https://github.com/example/mcp-sandbox",
                    "discussion_url": "https://news.ycombinator.com/item?id=12345",
                    "published_date": "2025-02-14",
                    "points": 310,
                    "comments": 94,
                    "summary": "Discussed on Hacker News.",
                }
            ]
        },
        gh_res={
            "repos": [
                {
                    "full_name": "example/agent-orchestrator",
                    "language": "Python",
                    "url": "https://github.com/example/agent-orchestrator",
                    "created_date": "2025-02-14",
                    "stars": 940,
                    "forks": 72,
                    "summary": "Lightweight multi-agent workflow engine.",
                    "topics": ["agents", "mcp"],
                }
            ]
        },
    )

    assert "Sun, Feb 09, 2025 – Sat, Feb 15, 2025" in briefing.weekly_window_heading
    assert len(briefing.papers) == 2
    assert briefing.papers[0].citation_id == "P1"
    assert briefing.papers[1].citation_id == "P2"
    assert briefing.models[0].citation_id == "M1"
    assert briefing.news[0].citation_id == "N1"
    assert briefing.repos[0].citation_id == "R1"
    assert len(briefing.whats_new_today) >= 4
    assert "[P1]" in briefing.executive_summary
    assert any("[P1]" in b for b in briefing.whats_new_today)

    fast_agent = create_fast_editorial_agent()
    assert fast_agent.name == "digest_synthesizer_agent"

    merged = _merge_editorial_into_briefing(
        briefing,
        EditorialSynthesisLLMOutput(
            headline="Breakthrough Week in Agentic Test-Time Compute",
            executive_summary=(
                "Weekly breakthroughs span parallel verification [P1, P2] and new open weights [M1].\n\n"
                "Community discussions highlighted MCP sandboxes [N1] and orchestration [R1] [P99]."
            ),
            whats_new_today=[
                "- New test-time compute paper landed today [P1]",
                "• Gemma-Next instruct checkpoint updated on the Hub [M1]",
                "Show HN MCP sandbox crossed 300 points today [N1]",
                "Agent orchestrator repo surged on GitHub [R1]",
            ],
            key_themes=["Test-time compute", "Sparse attention", "MCP sandboxes"],
            paper_summaries=["Takeaway 1.", "Takeaway 2."],
            model_summaries=["Model takeaway."],
            news_summaries=["News takeaway."],
            repo_summaries=["Repo takeaway."],
        ),
    )
    assert merged.headline == "Breakthrough Week in Agentic Test-Time Compute"
    assert "[P1] [P2]" in merged.executive_summary
    assert "[P99]" not in merged.executive_summary  # dangling citation pruned
    assert not merged.whats_new_today[0].startswith("-")
    assert "[P1]" in merged.whats_new_today[0]

    verdict = evaluate_briefing_self_check(merged)
    assert verdict.passed is True
    assert verdict.verified_citation_count >= 5

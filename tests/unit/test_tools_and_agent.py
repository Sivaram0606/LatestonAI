"""Unit tests for date parsing, multi-agent topology, and fallback briefing synthesis."""

import datetime

from google.adk.agents import ParallelAgent, SequentialAgent

from app.agent import MODEL, create_fast_editorial_agent, create_root_agent
from app.digest_service import (
    _merge_editorial_into_briefing,
    build_deterministic_briefing_from_raw,
)
from app.schemas import EditorialSynthesisLLMOutput
from app.tools import parse_target_date


def test_parse_target_date_valid_and_future() -> None:
    today = datetime.datetime.now(datetime.UTC).date()
    d_today, note_today = parse_target_date("today")
    assert d_today == today
    assert note_today == ""

    d_past, note_past = parse_target_date("2025-02-14")
    assert d_past == datetime.date(2025, 2, 14)
    assert note_past == ""

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


def test_deterministic_briefing_deduplicates_and_populates_all_sections() -> None:
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
                    "published_date": "2025-02-14",
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
                    "stars": 940,
                    "forks": 72,
                    "summary": "Lightweight multi-agent workflow engine.",
                    "topics": ["agents", "mcp"],
                }
            ]
        },
    )

    assert len(briefing.papers) == 2  # duplicate title removed
    assert len(briefing.models) == 1
    assert len(briefing.news) == 1
    assert len(briefing.repos) == 1
    assert len(briefing.key_themes) >= 3
    assert "2025-02-14" in briefing.headline

    fast_agent = create_fast_editorial_agent()
    assert fast_agent.name == "digest_synthesizer_agent"

    merged = _merge_editorial_into_briefing(
        briefing,
        EditorialSynthesisLLMOutput(
            headline="Breakthrough Day in Agentic Test-Time Compute",
            executive_summary="Paragraph one.\n\nParagraph two.",
            key_themes=["Test-time compute", "Sparse attention", "MCP sandboxes"],
            paper_summaries=["Takeaway 1.", "Takeaway 2."],
            model_summaries=["Model takeaway."],
            news_summaries=["News takeaway."],
            repo_summaries=["Repo takeaway."],
        ),
    )
    assert merged.headline == "Breakthrough Day in Agentic Test-Time Compute"
    assert merged.papers[0].summary == "Takeaway 1."
    assert merged.papers[0].url == "https://huggingface.co/papers/2502.00001"

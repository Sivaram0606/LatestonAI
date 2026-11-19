"""Strict Pydantic input/output schemas for tools, agents, citations, self-eval, and REST APIs.

Satisfies AgentOps Code Review Matrix — Category 1 (Explicit JSON Schemas):
- Enforces `ConfigDict(extra="forbid")` on tool inputs and structured outputs to
  strictly validate tool arguments, weekly Sunday-Saturday windows, daily bullet
  updates, and section citation IDs (`[P1]`, `[M1]`, `[N1]`, `[R1]`).
"""

from __future__ import annotations

import datetime
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Strict Tool Input Schemas
# ---------------------------------------------------------------------------
class DateScopedToolInput(BaseModel):
    """Validated input schema for date-scoped AI intelligence collector tools."""

    model_config = ConfigDict(extra="forbid")

    target_date: str = Field(
        description=(
            "Target UTC date formatted strictly as 'YYYY-MM-DD' (e.g., '2025-02-14') "
            "or the literal keyword 'today'."
        )
    )

    @field_validator("target_date")
    @classmethod
    def validate_target_date(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if cleaned.lower() == "today":
            return "today"
        candidate = cleaned[:10]
        if not _ISO_DATE_RE.match(candidate):
            raise ValueError(
                f"Invalid target_date '{value}'. Expected 'YYYY-MM-DD' (e.g. '2025-02-14') or 'today'."
            )
        datetime.date.fromisoformat(candidate)
        return candidate


class ArchiveSearchInput(BaseModel):
    """Validated input schema for querying archived briefings in persistent storage."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=200,
        description="Keywords or topic phrase to search across archived AI briefings.",
    )
    limit: int = Field(
        ge=1,
        le=20,
        description="Maximum number of matching archived briefings to return (1 to 20).",
    )


class PublishDigestArchiveInput(BaseModel):
    """Validated input schema for high-stakes external digest publication (HITL-gated)."""

    model_config = ConfigDict(extra="forbid")

    target_date: str = Field(
        description="Target briefing date in 'YYYY-MM-DD' format to publish officially."
    )
    distribution_channel: Literal[
        "org_announcement", "engineering_digest", "public_feed"
    ] = Field(
        description="Destination broadcast channel requiring human editorial approval."
    )
    reviewer_note: str = Field(
        min_length=3,
        max_length=500,
        description="Human reviewer sign-off note accompanying the published digest.",
    )

    @field_validator("target_date")
    @classmethod
    def validate_publish_date(cls, value: str) -> str:
        return DateScopedToolInput.validate_target_date(value)


# ---------------------------------------------------------------------------
# Guided Tool Error & Output Schemas
# ---------------------------------------------------------------------------
class ToolErrorGuidance(BaseModel):
    """Guided error response instructing the LLM how to recover without crashing."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    source: str = Field(description="Upstream API or tool component name.")
    requested_date: str = Field(description="Requested target_date string.")
    error_code: str = Field(description="Machine-readable error classification.")
    error_message: str = Field(description="Sanitized human-readable failure reason.")
    retryable: bool = Field(
        description="Whether retrying with adjusted args may succeed."
    )
    recovery_instructions: str = Field(
        description="Actionable step-by-step instructions for the LLM to recover gracefully."
    )


class PaperBrief(BaseModel):
    """Summarized AI research paper with section citation anchor."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(
        default="P1",
        description="Deterministic section citation ID such as 'P1', 'P2'.",
    )
    title: str = Field(description="Full title of the research paper.")
    authors: str = Field(description="Lead authors or institution summary.")
    source: str = Field(
        description="Source API, e.g. 'Hugging Face Daily Papers' or 'arXiv'."
    )
    url: str = Field(description="Direct canonical URL to the paper page.")
    published_date: str = Field(
        description="Publication or curation date (YYYY-MM-DD)."
    )
    upvotes: int = Field(
        description="Community upvotes or score (0 if not applicable)."
    )
    summary: str = Field(
        description="Concise 2-3 sentence synthesis of the core contribution and why it matters."
    )
    tags: list[str] = Field(
        description="2-4 short topic tags such as 'Agents', 'Reasoning', 'Multimodal'."
    )


class ModelBrief(BaseModel):
    """Summarized trending AI model or interactive Space with section citation anchor."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(
        default="M1",
        description="Deterministic section citation ID such as 'M1', 'M2'.",
    )
    name: str = Field(
        description="Model or Space identifier (e.g. 'org/model-name')."
    )
    kind: str = Field(description="Either 'Model' or 'Space'.")
    author: str = Field(description="Creator or organization name.")
    pipeline_tag: str = Field(
        description="Task category such as 'text-generation', 'image-text-to-text', or 'space-demo'."
    )
    url: str = Field(description="Direct URL to the model or Space on Hugging Face.")
    likes: int = Field(description="Total or trending likes count.")
    downloads: int = Field(description="Download count (0 for Spaces).")
    last_modified: str = Field(
        default="",
        description="Last modified ISO date (YYYY-MM-DD) when available.",
    )
    summary: str = Field(
        description="Concise explanation of capabilities, architecture, or interactive demo."
    )


class NewsBrief(BaseModel):
    """Summarized AI tech news or community discussion item with section citation anchor."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(
        default="N1",
        description="Deterministic section citation ID such as 'N1', 'N2'.",
    )
    title: str = Field(description="Headline of the news story or discussion.")
    source: str = Field(description="Origin source domain or 'Hacker News'.")
    url: str = Field(
        description="Direct article URL (or HN discussion URL if self-post)."
    )
    discussion_url: str = Field(description="Community discussion URL.")
    published_date: str = Field(
        default="",
        description="UTC publication date (YYYY-MM-DD) of the story.",
    )
    points: int = Field(description="Upvotes / points on the story.")
    comments: int = Field(description="Number of community comments.")
    summary: str = Field(
        description="1-2 sentence takeaway on why this story is notable for AI practitioners."
    )


class RepoBrief(BaseModel):
    """Summarized trending open-source AI repository with section citation anchor."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(
        default="R1",
        description="Deterministic section citation ID such as 'R1', 'R2'.",
    )
    full_name: str = Field(description="GitHub owner/repo identifier.")
    language: str = Field(description="Primary programming language.")
    url: str = Field(description="Direct GitHub repository URL.")
    created_date: str = Field(
        default="",
        description="Repository creation or breakout ISO date (YYYY-MM-DD).",
    )
    stars: int = Field(description="GitHub star count.")
    forks: int = Field(description="GitHub fork count.")
    summary: str = Field(
        description="Concise summary of what problem this repository solves and its approach."
    )
    topics: list[str] = Field(description="Up to 4 topic tags.")


class HFDailyPapersToolOutput(BaseModel):
    """Strict output schema for `fetch_hf_daily_papers`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "empty"]
    source: str
    requested_date: str
    week_window: str
    effective_date: str
    note: str
    count: int
    papers: list[PaperBrief]


class ArxivPapersToolOutput(BaseModel):
    """Strict output schema for `fetch_arxiv_ai_papers`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "empty"]
    source: str
    requested_date: str
    window: str
    note: str
    count: int
    papers: list[PaperBrief]


class HFModelsSpacesToolOutput(BaseModel):
    """Strict output schema for `fetch_hf_trending_models_and_spaces`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "empty"]
    source: str
    requested_date: str
    week_window: str
    note: str
    count: int
    items: list[ModelBrief]


class HackerNewsToolOutput(BaseModel):
    """Strict output schema for `fetch_hackernews_ai_stories`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "empty"]
    source: str
    requested_date: str
    week_window: str
    note: str
    count: int
    stories: list[NewsBrief]


class GitHubReposToolOutput(BaseModel):
    """Strict output schema for `fetch_github_ai_repos`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "empty"]
    source: str
    requested_date: str
    window: str
    note: str
    count: int
    repos: list[RepoBrief]


# ---------------------------------------------------------------------------
# Structured Agent Synthesis & Self-Eval Schemas
# ---------------------------------------------------------------------------
class DailyBriefingLLMOutput(BaseModel):
    """Structured weekly (Sunday-Saturday) + daily ('What's New Today') AI briefing."""

    model_config = ConfigDict(extra="forbid")

    weekly_window_heading: str = Field(
        default="",
        description=(
            "Heading displaying the Sunday-to-Saturday week window as one unified range, "
            "e.g. 'Weekly AI Summary: Sun, Nov 15, 2026 - Sat, Nov 21, 2026'."
        ),
    )
    headline: str = Field(
        description="Editorial headline capturing the biggest AI themes of the Sunday-Saturday week."
    )
    executive_summary: str = Field(
        description=(
            "2-3 cohesive paragraphs synthesizing the Sunday-to-Saturday weekly breakthroughs "
            "with inline bracketed citations ([P1], [M1], [N1], [R1]) referring to the sections below."
        )
    )
    whats_new_today: list[str] = Field(
        default_factory=list,
        description=(
            "4-6 concise bullet-point updates highlighting what is new on the selected day, "
            "each containing inline citations ([P1], [M1], [N1], [R1]) referring to the sections below."
        ),
    )
    key_themes: list[str] = Field(
        description="3-5 crisp bullet themes summarizing the week's AI landscape."
    )
    papers: list[PaperBrief] = Field(
        description="Top curated AI research papers ([P1]..[Pn])."
    )
    models: list[ModelBrief] = Field(
        description="Top trending AI models and interactive Spaces ([M1]..[Mn])."
    )
    news: list[NewsBrief] = Field(
        description="Top AI news and community discussions ([N1]..[Nn])."
    )
    repos: list[RepoBrief] = Field(
        description="Top breakout open-source AI repositories ([R1]..[Rn])."
    )


class EditorialSynthesisLLMOutput(BaseModel):
    """Compact single-pass ADK synthesis with Sunday-Saturday summary, daily bullets, and citations."""

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(
        description="Sharp editorial headline capturing the Sunday-to-Saturday week's defining AI breakthroughs."
    )
    executive_summary: str = Field(
        description=(
            "2-3 cohesive weekly summary paragraphs (separated by double newlines) synthesizing "
            "the Sunday-to-Saturday window with inline section citations like [P1], [M1], [N1], [R1]."
        )
    )
    whats_new_today: list[str] = Field(
        description=(
            "4-6 crisp bullet points (without leading '-' characters) summarizing what is new today "
            "(on target_date), with each bullet citing at least one [P#], [M#], [N#], or [R#] item below."
        )
    )
    key_themes: list[str] = Field(
        description="4-5 crisp bullet themes summarizing the Sunday-to-Saturday AI landscape."
    )
    paper_summaries: list[str] = Field(
        description="Concise 1-2 sentence technical takeaway for each paper, in exact input order."
    )
    model_summaries: list[str] = Field(
        description="Concise 1-2 sentence capability/use-case takeaway for each model/Space, in exact input order."
    )
    news_summaries: list[str] = Field(
        description="Concise 1-sentence practitioner takeaway for each news story, in exact input order."
    )
    repo_summaries: list[str] = Field(
        description="Concise 1-sentence takeaway on what each GitHub repo solves, in exact input order."
    )


class SelfEvalVerdict(BaseModel):
    """Automated grounding, citation integrity, and policy self-evaluation verdict."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(
        description="Whether all grounding, citation, and safety checks passed."
    )
    grounding_score: float = Field(
        description="Normalized quality/grounding score between 0.0 and 1.0."
    )
    verified_url_count: int = Field(
        description="Count of item URLs verified against live tool provenance."
    )
    flagged_url_count: int = Field(
        description="Count of item URLs flagged as unverified."
    )
    verified_citation_count: int = Field(
        default=0,
        description="Count of inline [P#]/[M#]/[N#]/[R#] citations verified against section items.",
    )
    findings: list[str] = Field(
        description="Human-readable audit notes from the self-evaluation pass."
    )


class AgentStepTrace(BaseModel):
    """Telemetry trace for an individual agent or tool step in the pipeline."""

    agent_name: str
    role: str
    status: str
    duration_ms: int
    items_collected: int
    detail: str


class DailyBriefingResponse(BaseModel):
    """Full API response returned to the React frontend."""

    target_date: str
    week_start_date: str = ""
    week_end_date: str = ""
    generated_at: str
    generation_mode: str
    model_used: str
    notice: str
    briefing: DailyBriefingLLMOutput
    agent_trace: list[AgentStepTrace]
    self_eval: SelfEvalVerdict | None = None
    trace_id: str = ""

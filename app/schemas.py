"""Pydantic schemas for the AI Daily Digest multi-agent pipeline and REST API."""

from pydantic import BaseModel, Field


class PaperBrief(BaseModel):
    """Summarized AI research paper for the daily digest."""

    title: str = Field(description="Full title of the research paper.")
    authors: str = Field(description="Lead authors or institution summary.")
    source: str = Field(
        description="Source API, e.g. 'Hugging Face Daily Papers' or 'arXiv'."
    )
    url: str = Field(description="Direct canonical URL to the paper page.")
    published_date: str = Field(description="Publication or curation date (YYYY-MM-DD).")
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
    """Summarized trending AI model or interactive Space."""

    name: str = Field(description="Model or Space identifier (e.g. 'org/model-name').")
    kind: str = Field(description="Either 'Model' or 'Space'.")
    author: str = Field(description="Creator or organization name.")
    pipeline_tag: str = Field(
        description="Task category such as 'text-generation', 'image-text-to-text', or 'space-demo'."
    )
    url: str = Field(description="Direct URL to the model or Space on Hugging Face.")
    likes: int = Field(description="Total or trending likes count.")
    downloads: int = Field(description="Download count (0 for Spaces).")
    summary: str = Field(
        description="Concise explanation of capabilities, architecture, or interactive demo."
    )


class NewsBrief(BaseModel):
    """Summarized AI tech news or community discussion item."""

    title: str = Field(description="Headline of the news story or discussion.")
    source: str = Field(description="Origin source domain or 'Hacker News'.")
    url: str = Field(description="Direct article URL (or HN discussion URL if self-post).")
    discussion_url: str = Field(description="Community discussion URL.")
    points: int = Field(description="Upvotes / points on the story.")
    comments: int = Field(description="Number of community comments.")
    summary: str = Field(
        description="1-2 sentence takeaway on why this story is notable for AI practitioners."
    )


class RepoBrief(BaseModel):
    """Summarized trending open-source AI repository."""

    full_name: str = Field(description="GitHub owner/repo identifier.")
    language: str = Field(description="Primary programming language.")
    url: str = Field(description="Direct GitHub repository URL.")
    stars: int = Field(description="GitHub star count.")
    forks: int = Field(description="GitHub fork count.")
    summary: str = Field(
        description="Concise summary of what problem this repository solves and its approach."
    )
    topics: list[str] = Field(description="Up to 4 topic tags.")


class DailyBriefingLLMOutput(BaseModel):
    """Structured daily AI briefing produced by the ADK synthesizer agent."""

    headline: str = Field(
        description="Editorial headline capturing the biggest AI themes of the day."
    )
    executive_summary: str = Field(
        description=(
            "2-3 cohesive paragraphs synthesizing key breakthroughs across papers, "
            "models, industry news, and open-source tools for the target date."
        )
    )
    key_themes: list[str] = Field(
        description="3-5 crisp bullet themes summarizing the day's AI landscape."
    )
    papers: list[PaperBrief] = Field(
        description="Top curated AI research papers for the day."
    )
    models: list[ModelBrief] = Field(
        description="Top trending AI models and interactive Spaces."
    )
    news: list[NewsBrief] = Field(
        description="Top AI news and community discussions for the day."
    )
    repos: list[RepoBrief] = Field(
        description="Top breakout open-source AI repositories."
    )


class EditorialSynthesisLLMOutput(BaseModel):
    """Compact single-pass ADK editorial synthesis without repeating item metadata."""

    headline: str = Field(
        description="Sharp editorial headline capturing the day's defining AI breakthroughs."
    )
    executive_summary: str = Field(
        description=(
            "2-3 cohesive paragraphs (separated by double newlines) synthesizing key "
            "breakthroughs across papers, models/Spaces, community news, and open-source repos."
        )
    )
    key_themes: list[str] = Field(
        description="4-5 crisp bullet themes summarizing the day's AI landscape."
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
    generated_at: str
    generation_mode: str
    model_used: str
    notice: str
    briefing: DailyBriefingLLMOutput
    agent_trace: list[AgentStepTrace]

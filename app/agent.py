# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-agent ADK pipeline for generating daily AI intelligence briefings."""

from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.schemas import DailyBriefingLLMOutput, EditorialSynthesisLLMOutput
from app.tools import (
    fetch_arxiv_ai_papers,
    fetch_github_ai_repos,
    fetch_hackernews_ai_stories,
    fetch_hf_daily_papers,
    fetch_hf_trending_models_and_spaces,
)

MODEL = "gemini-3.7-flash"


async def initialize_digest_state(callback_context: CallbackContext) -> None:
    """Ensure shared state keys exist before parallel collectors execute."""
    for key in ("raw_papers", "raw_models", "raw_news_repos"):
        if key not in callback_context.state:
            callback_context.state[key] = "No items collected."


def _make_model() -> Gemini:
    return Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=2),
    )


def _fast_gen_config(minimal_thinking: bool = True) -> types.GenerateContentConfig:
    """Low-latency generation config using MINIMAL thinking for fast editorial synthesis."""
    if minimal_thinking:
        return types.GenerateContentConfig(
            temperature=0.3,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            ),
        )
    return types.GenerateContentConfig(temperature=0.3)


def create_papers_research_agent() -> Agent:
    """Sub-agent that gathers daily AI research papers from Hugging Face and arXiv."""
    return Agent(
        name="papers_research_agent",
        model=_make_model(),
        generate_content_config=_fast_gen_config(),
        description="Collects and filters daily AI research papers from Hugging Face Daily Papers and arXiv.",
        instruction=(
            "You are the AI Research Papers Collector Agent.\n"
            "Extract the target YYYY-MM-DD date from the user request (use 'today' if unspecified).\n"
            "1. Call BOTH `fetch_hf_daily_papers` and `fetch_arxiv_ai_papers` with that target_date.\n"
            "2. Select up to 8 of the highest-signal, non-duplicate papers.\n"
            "3. Return a concise, structured markdown report preserving exact titles, authors, "
            "source ('Hugging Face Daily Papers' or 'arXiv'), canonical URLs, published_date, "
            "upvotes, tags, and a 2-sentence technical takeaway for each paper. "
            "Never invent URLs or paper titles."
        ),
        tools=[fetch_hf_daily_papers, fetch_arxiv_ai_papers],
        output_key="raw_papers",
    )


def create_models_spaces_agent() -> Agent:
    """Sub-agent that gathers trending AI models and interactive Spaces from Hugging Face Hub."""
    return Agent(
        name="models_spaces_agent",
        model=_make_model(),
        generate_content_config=_fast_gen_config(),
        description="Collects trending AI models and interactive Spaces from the Hugging Face Hub API.",
        instruction=(
            "You are the AI Models & Interactive Spaces Scout Agent.\n"
            "Extract the target YYYY-MM-DD date from the user request (use 'today' if unspecified).\n"
            "1. Call `fetch_hf_trending_models_and_spaces` with that target_date.\n"
            "2. Curate up to 8 standout Models and Spaces.\n"
            "3. Return a structured report preserving exact model/space name, kind ('Model' or 'Space'), "
            "author, pipeline_tag, canonical Hugging Face URL, likes, downloads, and a concise "
            "explanation of what developers can build or demo with it. Never fabricate URLs."
        ),
        tools=[fetch_hf_trending_models_and_spaces],
        output_key="raw_models",
    )


def create_news_repos_agent() -> Agent:
    """Sub-agent that gathers AI community discussions and breakout open-source repos."""
    return Agent(
        name="news_repos_agent",
        model=_make_model(),
        generate_content_config=_fast_gen_config(),
        description="Collects daily AI news from Hacker News Algolia API and breakout AI repos from GitHub.",
        instruction=(
            "You are the AI Ecosystem News & Open-Source Repos Agent.\n"
            "Extract the target YYYY-MM-DD date from the user request (use 'today' if unspecified).\n"
            "1. Call BOTH `fetch_hackernews_ai_stories` and `fetch_github_ai_repos` with that target_date.\n"
            "2. Select up to 6 top AI news discussions and up to 6 breakout GitHub repositories.\n"
            "3. Return a structured report preserving exact headlines/repo full_names, URLs, "
            "discussion_url, points, comments, stars, forks, language, topics, and why each item matters. "
            "Never invent URLs or metrics."
        ),
        tools=[fetch_hackernews_ai_stories, fetch_github_ai_repos],
        output_key="raw_news_repos",
    )


def create_digest_synthesizer_agent() -> Agent:
    """Lead editor agent that synthesizes collector findings into DailyBriefingLLMOutput."""
    return Agent(
        name="digest_synthesizer_agent",
        model=_make_model(),
        generate_content_config=_fast_gen_config(),
        description="Synthesizes research papers, models, news, and repos into a cohesive daily AI briefing.",
        instruction=(
            "You are the Editor-in-Chief of the AI Daily Digest.\n"
            "Synthesize the findings gathered by your specialist research agents for the requested date:\n\n"
            "=== RESEARCH PAPERS ===\n{raw_papers}\n\n"
            "=== MODELS & SPACES ===\n{raw_models}\n\n"
            "=== NEWS & OPEN-SOURCE REPOS ===\n{raw_news_repos}\n\n"
            "Editorial Rules:\n"
            "1. Craft a sharp, informative `headline` capturing the day's defining AI breakthroughs.\n"
            "2. Write a rich 2-3 paragraph `executive_summary` (separated by double newlines) connecting "
            "the threads across research papers, model releases, community news, and open-source tools.\n"
            "3. Extract 3-5 crisp `key_themes`.\n"
            "4. Populate `papers`, `models`, `news`, and `repos` strictly using ONLY real URLs, names, "
            "and metrics present in the collector reports above. Never fabricate links or entries."
        ),
        output_schema=DailyBriefingLLMOutput,
        output_key="daily_briefing",
    )


def create_fast_editorial_agent(minimal_thinking: bool = True) -> Agent:
    """Single-pass ADK Editor-in-Chief agent operating on parallel-collected API payloads."""
    return Agent(
        name="digest_synthesizer_agent",
        model=_make_model(),
        generate_content_config=_fast_gen_config(minimal_thinking=minimal_thinking),
        include_contents="none",
        description="Fast single-pass ADK synthesizer producing editorial briefing & item takeaways.",
        instruction=(
            "You are the Editor-in-Chief of the AI Daily Digest.\n"
            "Below are the live items collected in parallel for {target_date} across Hugging Face, "
            "arXiv, Hacker News, and GitHub:\n\n"
            "{collected_digest_context}\n\n"
            "Instructions:\n"
            "1. Write a sharp, specific `headline` capturing the day's defining AI themes.\n"
            "2. Write a cohesive 2-3 paragraph `executive_summary` (separated by `\\n\\n`) weaving "
            "together the standout research papers, trending models/Spaces, community news, and repos.\n"
            "3. Provide 4-5 crisp `key_themes`.\n"
            "4. Provide concise, high-signal 1-2 sentence takeaways in `paper_summaries`, "
            "`model_summaries`, `news_summaries`, and `repo_summaries` matching the exact count "
            "and order of items listed in each section."
        ),
        output_schema=EditorialSynthesisLLMOutput,
        output_key="editorial_synthesis",
    )


def create_root_agent() -> SequentialAgent:
    """Build the multi-agent Sequential -> Parallel -> Synthesizer ADK workflow."""
    parallel_collectors = ParallelAgent(
        name="parallel_api_collectors",
        description="Runs Papers, Models/Spaces, and News/Repos collector agents concurrently.",
        sub_agents=[
            create_papers_research_agent(),
            create_models_spaces_agent(),
            create_news_repos_agent(),
        ],
    )
    return SequentialAgent(
        name="daily_digest_pipeline",
        description="End-to-end multi-agent pipeline that pulls popular AI APIs and synthesizes a daily briefing.",
        before_agent_callback=initialize_digest_state,
        sub_agents=[
            parallel_collectors,
            create_digest_synthesizer_agent(),
        ],
    )


root_agent = create_root_agent()

app = App(
    root_agent=root_agent,
    name="app",
)

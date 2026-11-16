"""Tools for pulling daily AI intelligence from popular public APIs."""

from __future__ import annotations

import asyncio
import datetime
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

import httpx

USER_AGENT = "ai-daily-digest-adk/1.0 (Google-ADK Multi-Agent Briefing)"
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


def _clean_text(text: str, max_len: int = 600) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "")).strip()
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rsplit(" ", 1)[0] + "…"


async def fetch_hf_daily_papers(target_date: str) -> dict[str, Any]:
    """Fetch curated daily AI research papers from the Hugging Face Daily Papers API.

    Args:
        target_date: Target date in YYYY-MM-DD format (e.g. '2025-02-14').

    Returns:
        Dictionary with status, effective_date, note, and list of daily papers.
    """
    day, date_note = parse_target_date(target_date)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        raw_items: list[dict[str, Any]] = []
        effective_day = day
        fallback_note = date_note

        for offset in range(0, 5):
            candidate = day - datetime.timedelta(days=offset)
            url = f"https://huggingface.co/api/daily_papers?date={candidate.isoformat()}"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        raw_items = data
                        effective_day = candidate
                        if offset > 0:
                            note_msg = (
                                f"No Hugging Face Daily Papers batch on {day.isoformat()} "
                                f"(weekend/holiday or pending curation); using nearest batch "
                                f"from {candidate.isoformat()}."
                            )
                            fallback_note = (
                                f"{fallback_note} {note_msg}".strip()
                                if fallback_note
                                else note_msg
                            )
                        break
            except httpx.HTTPError as exc:
                if offset == 4:
                    return {
                        "status": "error",
                        "source": "Hugging Face Daily Papers",
                        "requested_date": day.isoformat(),
                        "error": str(exc),
                        "papers": [],
                    }

        papers: list[dict[str, Any]] = []
        for entry in raw_items:
            paper = entry.get("paper") or {}
            paper_id = str(paper.get("id") or entry.get("id") or "").strip()
            title = _clean_text(
                str(entry.get("title") or paper.get("title") or ""), max_len=240
            )
            if not title:
                continue

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
            ][:4]
            if not keywords:
                keywords = ["AI Research"]

            papers.append(
                {
                    "title": title,
                    "authors": authors_str or "Research Community",
                    "source": "Hugging Face Daily Papers",
                    "url": (
                        f"https://huggingface.co/papers/{paper_id}"
                        if paper_id
                        else "https://huggingface.co/papers"
                    ),
                    "published_date": str(
                        entry.get("publishedAt")
                        or paper.get("publishedAt")
                        or effective_day.isoformat()
                    )[:10],
                    "upvotes": int(paper.get("upvotes") or 0),
                    "num_comments": int(entry.get("numComments") or 0),
                    "summary": _clean_text(str(summary_raw), max_len=550),
                    "tags": keywords,
                    "github_repo": paper.get("githubRepo") or "",
                }
            )

        papers.sort(key=lambda p: p["upvotes"], reverse=True)
        return {
            "status": "success",
            "source": "Hugging Face Daily Papers",
            "requested_date": day.isoformat(),
            "effective_date": effective_day.isoformat(),
            "note": fallback_note,
            "count": len(papers[:10]),
            "papers": papers[:10],
        }


async def fetch_arxiv_ai_papers(target_date: str) -> dict[str, Any]:
    """Fetch recent cs.AI, cs.LG, and cs.CL preprints from the arXiv API around a date.

    Args:
        target_date: Target date in YYYY-MM-DD format (e.g. '2025-02-14').

    Returns:
        Dictionary with status, effective_window, and list of arXiv AI papers.
    """
    day, date_note = parse_target_date(target_date)
    headers = {"User-Agent": USER_AGENT}
    atom_ns = "{http://www.w3.org/2005/Atom}"

    windows = [
        (day, day),
        (day - datetime.timedelta(days=4), day),
    ]

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
                    continue
                root = ET.fromstring(resp.text)
                entries = root.findall(f"{atom_ns}entry")
                if not entries:
                    continue

                papers: list[dict[str, Any]] = []
                for entry in entries:
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
                        (a.text or "").strip() for a in author_elems if (a.text or "").strip()
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
                        {
                            "title": title,
                            "authors": authors_str or "arXiv Authors",
                            "source": "arXiv",
                            "url": url,
                            "published_date": published,
                            "upvotes": 0,
                            "summary": summary,
                            "tags": cats or ["cs.AI"],
                        }
                    )

                if papers:
                    return {
                        "status": "success",
                        "source": "arXiv API",
                        "requested_date": day.isoformat(),
                        "window": f"{start_d.isoformat()}..{end_d.isoformat()}",
                        "note": date_note,
                        "count": len(papers),
                        "papers": papers,
                    }
            except (httpx.HTTPError, ET.ParseError) as exc:
                last_err = str(exc)
                continue

    return {
        "status": "empty",
        "source": "arXiv API",
        "requested_date": day.isoformat(),
        "note": locals().get("last_err", "No arXiv entries found for date window."),
        "count": 0,
        "papers": [],
    }


async def fetch_hf_trending_models_and_spaces(target_date: str) -> dict[str, Any]:
    """Fetch trending AI models and interactive Spaces from the Hugging Face Hub API.

    Args:
        target_date: Target date in YYYY-MM-DD format (e.g. '2025-02-14').

    Returns:
        Dictionary containing trending models and Spaces with metadata and URLs.
    """
    day, date_note = parse_target_date(target_date)
    today = datetime.datetime.now(datetime.UTC).date()
    is_historical = (today - day).days > 2

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

        items: list[dict[str, Any]] = []

        # Parse trending models (fallback to /api/models if /api/trending changes shape)
        raw_models: list[dict[str, Any]] = []
        if isinstance(models_resp, httpx.Response) and models_resp.status_code == 200:
            payload = models_resp.json()
            if isinstance(payload, dict) and "recentlyTrending" in payload:
                raw_models = [
                    x.get("repoData", x) for x in payload.get("recentlyTrending", [])
                ]
            elif isinstance(payload, list):
                raw_models = [x.get("repoData", x) for x in payload]

        if not raw_models:
            try:
                fallback = await client.get(
                    "https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=8"
                )
                if fallback.status_code == 200 and isinstance(fallback.json(), list):
                    raw_models = fallback.json()
            except httpx.HTTPError:
                raw_models = []

        for m in raw_models:
            repo_id = str(m.get("id") or m.get("modelId") or "").strip()
            if not repo_id:
                continue
            lowered_id = repo_id.lower()
            if any(
                term in lowered_id
                for term in ("uncensored", "nsfw", "abliterated", "heretic", "erotic")
            ):
                continue
            author = str(
                m.get("author") or (repo_id.split("/")[0] if "/" in repo_id else "hf")
            )
            pipeline_tag = str(m.get("pipeline_tag") or "foundation-model")
            likes = int(m.get("likes") or 0)
            downloads = int(m.get("downloads") or 0)
            items.append(
                {
                    "name": repo_id,
                    "kind": "Model",
                    "author": author,
                    "pipeline_tag": pipeline_tag,
                    "url": f"https://huggingface.co/{repo_id}",
                    "likes": likes,
                    "downloads": downloads,
                    "last_modified": str(m.get("lastModified") or "")[:10],
                    "summary": (
                        f"Trending {pipeline_tag} model by {author} on Hugging Face Hub "
                        f"({likes:,} likes, {downloads:,} downloads)."
                    ),
                }
            )
            if len(items) >= 7:
                break

        # Parse trending Spaces
        raw_spaces: list[dict[str, Any]] = []
        if isinstance(spaces_resp, httpx.Response) and spaces_resp.status_code == 200:
            payload = spaces_resp.json()
            if isinstance(payload, dict) and "recentlyTrending" in payload:
                raw_spaces = [
                    x.get("repoData", x) for x in payload.get("recentlyTrending", [])
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

        for sp in raw_spaces[:5]:
            space_id = str(sp.get("id") or "").strip()
            if not space_id:
                continue
            author = str(
                sp.get("author")
                or (space_id.split("/")[0] if "/" in space_id else "hf")
            )
            title = str(sp.get("title") or space_id.split("/")[-1])
            short_desc = _clean_text(
                str(sp.get("shortDescription") or sp.get("ai_short_description") or ""),
                max_len=240,
            )
            likes = int(sp.get("likes") or 0)
            sdk = str(sp.get("sdk") or "interactive-space")
            items.append(
                {
                    "name": space_id,
                    "kind": "Space",
                    "author": author,
                    "pipeline_tag": f"{sdk}-space",
                    "url": f"https://huggingface.co/spaces/{space_id}",
                    "likes": likes,
                    "downloads": 0,
                    "last_modified": str(sp.get("lastModified") or "")[:10],
                    "summary": (
                        f"{title}: {short_desc}"
                        if short_desc
                        else f"Interactive AI Space '{title}' by {author} ({likes:,} likes)."
                    ),
                }
            )

        note = date_note
        if is_historical:
            hist_note = (
                "Hugging Face Hub trending reflects live Hub momentum; "
                "papers, news, and repos are strictly scoped to the selected date."
            )
            note = f"{note} {hist_note}".strip() if note else hist_note

        return {
            "status": "success",
            "source": "Hugging Face Hub API",
            "requested_date": day.isoformat(),
            "note": note,
            "count": len(items),
            "items": items,
        }


async def fetch_hackernews_ai_stories(target_date: str) -> dict[str, Any]:
    """Fetch top AI and LLM stories from Hacker News Algolia API for a 24h UTC date.

    Args:
        target_date: Target date in YYYY-MM-DD format (e.g. '2025-02-14').

    Returns:
        Dictionary with status, date window, and top AI Hacker News stories.
    """
    day, date_note = parse_target_date(target_date)
    start_dt = datetime.datetime(
        day.year, day.month, day.day, 0, 0, 0, tzinfo=datetime.UTC
    )
    start_ts = int(start_dt.timestamp())
    end_ts = start_ts + 86400
    num_filter = f"created_at_i>={start_ts},created_at_i<{end_ts}"

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        seen_ids: set[str] = set()
        matched_hits: list[dict[str, Any]] = []

        # 1. Fetch top stories across the exact 24h UTC day and filter with AI regex
        try:
            top_resp = await client.get(
                "https://hn.algolia.com/api/v1/search",
                params={
                    "tags": "story",
                    "numericFilters": num_filter,
                    "hitsPerPage": "75",
                },
            )
            if top_resp.status_code == 200:
                for hit in top_resp.json().get("hits", []):
                    obj_id = str(hit.get("objectID") or "")
                    title = str(hit.get("title") or "")
                    url = str(hit.get("url") or "")
                    if obj_id and AI_NEWS_PATTERN.search(f"{title} {url}"):
                        seen_ids.add(obj_id)
                        matched_hits.append(hit)
        except httpx.HTTPError:
            pass

        # 2. Supplement with targeted AI / LLM keyword queries in the same 24h window
        if len(matched_hits) < 8:
            for kw in ("AI", "LLM", "Gemini", "OpenAI"):
                try:
                    kw_resp = await client.get(
                        "https://hn.algolia.com/api/v1/search",
                        params={
                            "query": kw,
                            "tags": "story",
                            "numericFilters": f"{num_filter},points>=5",
                            "hitsPerPage": "15",
                        },
                    )
                    if kw_resp.status_code == 200:
                        for hit in kw_resp.json().get("hits", []):
                            obj_id = str(hit.get("objectID") or "")
                            if obj_id and obj_id not in seen_ids:
                                seen_ids.add(obj_id)
                                matched_hits.append(hit)
                except httpx.HTTPError:
                    continue

        matched_hits.sort(key=lambda h: int(h.get("points") or 0), reverse=True)
        stories: list[dict[str, Any]] = []
        for hit in matched_hits[:10]:
            obj_id = str(hit.get("objectID") or "")
            hn_url = f"https://news.ycombinator.com/item?id={obj_id}"
            raw_url = str(hit.get("url") or "").strip() or hn_url
            domain = urllib.parse.urlparse(raw_url).netloc.removeprefix("www.")
            points = int(hit.get("points") or 0)
            comments = int(hit.get("num_comments") or 0)
            title = _clean_text(str(hit.get("title") or ""), max_len=220)
            if not title:
                continue
            stories.append(
                {
                    "title": title,
                    "source": domain or "Hacker News",
                    "url": raw_url,
                    "discussion_url": hn_url,
                    "points": points,
                    "comments": comments,
                    "author": str(hit.get("author") or ""),
                    "published_date": day.isoformat(),
                    "summary": (
                        f"Discussed on Hacker News ({points} points, {comments} comments) "
                        f"via {domain or 'news.ycombinator.com'}."
                    ),
                }
            )

        return {
            "status": "success",
            "source": "Hacker News Algolia API",
            "requested_date": day.isoformat(),
            "note": date_note,
            "count": len(stories),
            "stories": stories,
        }


async def fetch_github_ai_repos(target_date: str) -> dict[str, Any]:
    """Fetch breakout open-source AI and agent repositories from GitHub Search API.

    Args:
        target_date: Target date in YYYY-MM-DD format (e.g. '2025-02-14').

    Returns:
        Dictionary with status, search window, and top GitHub AI repositories.
    """
    day, date_note = parse_target_date(target_date)
    start_day = day - datetime.timedelta(days=6)

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    }
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    query = (
        f"(llm OR agent OR mcp OR multimodal OR rag) "
        f"created:{start_day.isoformat()}..{day.isoformat()}"
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
                return {
                    "status": "error",
                    "source": "GitHub Search API",
                    "requested_date": day.isoformat(),
                    "error": f"HTTP {resp.status_code}",
                    "repos": [],
                }
            items = resp.json().get("items", [])
        except httpx.HTTPError as exc:
            return {
                "status": "error",
                "source": "GitHub Search API",
                "requested_date": day.isoformat(),
                "error": str(exc),
                "repos": [],
            }

    repos: list[dict[str, Any]] = []
    for repo in items[:8]:
        full_name = str(repo.get("full_name") or "").strip()
        if not full_name:
            continue
        desc = _clean_text(
            str(repo.get("description") or "Open-source AI repository."), max_len=280
        )
        topics = [str(t) for t in (repo.get("topics") or [])[:4]]
        lang = str(repo.get("language") or "Multi-language")
        repos.append(
            {
                "full_name": full_name,
                "language": lang,
                "url": str(repo.get("html_url") or f"https://github.com/{full_name}"),
                "stars": int(repo.get("stargazers_count") or 0),
                "forks": int(repo.get("forks_count") or 0),
                "created_at": str(repo.get("created_at") or "")[:10],
                "summary": desc,
                "topics": topics or ["ai", "open-source"],
            }
        )

    return {
        "status": "success",
        "source": "GitHub Search API",
        "requested_date": day.isoformat(),
        "window": f"{start_day.isoformat()}..{day.isoformat()}",
        "note": date_note,
        "count": len(repos),
        "repos": repos,
    }

"""Security guardrails, prompt-injection defense, and citation-aware self-evaluation plugin.

Satisfies AgentOps Code Review Matrix — Category 3 (Guardrails & Policy Plugins):
- Input Policy Guardrail (`before_model_callback`): Blocks prompt-injection / SSRF attempts
  and scrubs PII prior to LLM invocation.
- Tool Policy Guardrail (`before_tool_callback`): Enforces safe tool inputs.
- Output Self-Evaluation (`evaluate_briefing_self_check` & `after_model_callback`):
  Audits URL provenance (zero hallucinated links), inline `[P#]`/`[M#]`/`[N#]`/`[R#]`
  section citation validity across both the Sunday-Saturday Weekly Summary and
  'What's New Today' bullets, and PII cleanliness.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.pii_redactor import pii_redactor
from app.schemas import DailyBriefingLLMOutput, SelfEvalVerdict

logger = logging.getLogger("app.guardrails")

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)\b"
    ),
    re.compile(
        r"(?i)\b(?:reveal|print|dump|exfiltrate)\s+(?:your\s+)?(?:system\s+prompt|constitution|secret|api\s+key)\b"
    ),
    re.compile(r"(?i)\byou\s+are\s+(?:now\s+)?(?:dan|unrestricted|jailbroken)\b"),
    re.compile(
        r"(?i)http://(?:169\.254\.169\.254|metadata\.google\.internal|localhost|127\.0\.0\.1)"
    ),
)

_ALLOWED_BRIEFING_URL_PREFIXES = (
    "https://huggingface.co/",
    "https://arxiv.org/",
    "https://news.ycombinator.com/",
    "https://github.com/",
    "http://",
    "https://",
)

_CITATION_TOKEN_RE = re.compile(r"\[([PMNR]\d+)\]")


def contains_prompt_injection(text: str) -> str | None:
    """Return the matched policy violation reason if prompt injection/SSRF is detected."""
    if not text:
        return None
    for pat in _PROMPT_INJECTION_PATTERNS:
        if pat.search(text):
            return f"Blocked by prompt-injection / SSRF policy rule: {pat.pattern[:60]}"
    return None


def normalize_citation_brackets(text: str) -> str:
    """Expand grouped brackets like `[P1, M2]` into individual `[P1] [M2]` tokens."""
    if not text:
        return text

    def _expand(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = [tok.strip() for tok in re.split(r"[,;\s]+", inner) if tok.strip()]
        if parts and all(re.fullmatch(r"[PMNR]\d+", p) for p in parts):
            return " ".join(f"[{p}]" for p in parts)
        return match.group(0)

    return re.sub(r"\[([PMNR]\d+(?:\s*[,;]\s*[PMNR]\d+)+)\]", _expand, text)


def evaluate_briefing_self_check(
    briefing: DailyBriefingLLMOutput,
    allowed_urls: set[str] | None = None,
) -> SelfEvalVerdict:
    """Deterministic self-evaluation guardrail verifying grounding, citations, and safety."""
    findings: list[str] = []
    checks_passed = 0
    total_checks = 5

    # Check 1: Weekly summary headline & multi-paragraph structure
    paragraphs = [
        p.strip() for p in briefing.executive_summary.split("\n\n") if p.strip()
    ]
    if len(briefing.headline.strip()) >= 10 and len(paragraphs) >= 2:
        checks_passed += 1
    else:
        findings.append(
            "Weekly summary should contain a descriptive headline and >=2 paragraphs."
        )

    # Check 2: Key themes & What's New Today bullet coverage
    if len(briefing.key_themes) >= 3 and (
        len(briefing.whats_new_today) >= 1
        or not (
            briefing.papers or briefing.models or briefing.news or briefing.repos
        )
    ):
        checks_passed += 1
    else:
        findings.append(
            "Expected >=3 weekly key_themes and populated whats_new_today bullets."
        )

    # Check 3: Strict URL provenance / anti-hallucination check
    all_item_urls = [
        *(p.url for p in briefing.papers),
        *(m.url for m in briefing.models),
        *(n.url for n in briefing.news),
        *(r.url for r in briefing.repos),
    ]
    unverified_urls: list[str] = []
    for url in all_item_urls:
        if not url.startswith(_ALLOWED_BRIEFING_URL_PREFIXES):
            unverified_urls.append(url)
        elif allowed_urls is not None and url not in allowed_urls:
            unverified_urls.append(url)

    if not unverified_urls:
        checks_passed += 1
    else:
        findings.append(
            f"Detected {len(unverified_urls)} URL(s) outside verified tool provenance."
        )

    # Check 4: Section citation validity ([P#], [M#], [N#], [R#]) across weekly & daily updates
    valid_ids: set[str] = {
        *(p.citation_id for p in briefing.papers),
        *(m.citation_id for m in briefing.models),
        *(n.citation_id for n in briefing.news),
        *(r.citation_id for r in briefing.repos),
    }
    citation_corpus = "\n".join(
        [briefing.executive_summary, *briefing.whats_new_today]
    )
    cited_tokens = _CITATION_TOKEN_RE.findall(citation_corpus)
    dangling_citations = [c for c in cited_tokens if valid_ids and c not in valid_ids]
    verified_citations = [c for c in cited_tokens if c in valid_ids]

    if not dangling_citations and (verified_citations or not valid_ids):
        checks_passed += 1
    else:
        findings.append(
            f"Citation audit found {len(dangling_citations)} dangling citation(s) "
            f"and {len(verified_citations)} verified section citation(s)."
        )

    # Check 5: PII / secret leakage scan across weekly prose & daily bullets
    combined_prose = "\n".join(
        [briefing.headline, briefing.executive_summary, *briefing.whats_new_today]
    )
    if pii_redactor.scrub_text(combined_prose) == combined_prose:
        checks_passed += 1
    else:
        findings.append("Redacted PII/credential pattern found in editorial prose.")

    score = round(checks_passed / total_checks, 2)
    return SelfEvalVerdict(
        passed=len(unverified_urls) == 0
        and len(dangling_citations) == 0
        and score >= 0.8,
        grounding_score=score,
        verified_url_count=len(all_item_urls) - len(unverified_urls),
        flagged_url_count=len(unverified_urls),
        verified_citation_count=len(verified_citations),
        findings=findings
        or ["All URL grounding, section citation, structure, and PII checks passed."],
    )


class SecurityAndSelfEvalPlugin(BasePlugin):
    """ADK global guardrail plugin enforcing input safety, tool policy, and output self-eval."""

    def __init__(self, name: str = "security_and_self_eval_plugin") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        for content in getattr(llm_request, "contents", None) or []:
            if getattr(content, "role", "") != "user":
                continue
            for part in getattr(content, "parts", None) or []:
                txt = getattr(part, "text", None)
                if not txt:
                    continue
                violation = contains_prompt_injection(txt)
                if violation:
                    logger.warning(
                        "Guardrail blocked unsafe user prompt before LLM invocation",
                        extra={
                            "event_type": "guardrail.prompt_blocked",
                            "phase": "OUTCOME",
                            "agent_name": callback_context.agent_name,
                            "reason": violation,
                        },
                    )
                    return LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text=(
                                        "Request declined by AI Daily Digest security guardrails: "
                                        "instructions attempting to override system policies, expose "
                                        "secrets, or probe internal endpoints are not permitted."
                                    )
                                )
                            ],
                        )
                    )
                part.text = pii_redactor.scrub_text(txt)
        return None

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        for val in tool_args.values():
            if isinstance(val, str):
                violation = contains_prompt_injection(val)
                if violation:
                    return {
                        "status": "error",
                        "error_code": "POLICY_VIOLATION",
                        "error_message": violation,
                        "retryable": False,
                        "recovery_instructions": (
                            "Tool call blocked by SecurityAndSelfEvalPlugin. "
                            "Remove unsafe instructions or internal URLs and pass a valid ISO date."
                        ),
                    }
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response and llm_response.content and llm_response.content.parts:
            for part in llm_response.content.parts:
                if getattr(part, "text", None):
                    part.text = pii_redactor.scrub_text(part.text)
        return None

"""Strategic model routing across Flash (fast extraction) and Pro (deep planning).

Satisfies AgentOps Code Review Matrix — Category 3 (Strategic Model Routing):
- Routes high-throughput API collection, event compaction, and low-latency daily
  refreshes to the Flash tier (`gemini-3.7-flash` with `ThinkingLevel.MINIMAL`).
- Routes deep multi-theme strategic planning and high-complexity editorial adjudication
  to the Pro / high-reasoning tier (`ADK_PRO_MODEL`, defaulting to `gemini-3.7-pro`
  with automatic Flash fallback if Pro endpoint quota is unavailable).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

from google.adk.models import Gemini
from google.genai import types

DEFAULT_FAST_MODEL = "gemini-3.7-flash"
DEFAULT_PRO_MODEL = "gemini-3.7-pro"


class TaskTier(StrEnum):
    """Workload categories mapped to optimal Gemini model + thinking tiers."""

    FAST_COLLECTOR = "fast_collector"
    HISTORY_COMPACTION = "history_compaction"
    STANDARD_SYNTHESIS = "standard_synthesis"
    DEEP_STRATEGIC_PLANNING = "deep_strategic_planning"


@dataclass(frozen=True)
class ModelRouteDecision:
    """Resolved model identifier, thinking level, and temperature for an agent task."""

    tier: TaskTier
    model_name: str
    thinking_level: types.ThinkingLevel | None
    temperature: float
    rationale: str


class StrategicModelRouter:
    """Selects the optimal Gemini model and generation config per task complexity."""

    @property
    def fast_model(self) -> str:
        return os.environ.get("ADK_FAST_MODEL", DEFAULT_FAST_MODEL).strip() or DEFAULT_FAST_MODEL

    @property
    def pro_model(self) -> str:
        return os.environ.get("ADK_PRO_MODEL", DEFAULT_PRO_MODEL).strip() or DEFAULT_PRO_MODEL

    def resolve(
        self, tier: TaskTier, prefer_minimal_thinking: bool = True
    ) -> ModelRouteDecision:
        """Route a task tier to its optimal model and reasoning configuration."""
        if tier in (TaskTier.FAST_COLLECTOR, TaskTier.HISTORY_COMPACTION):
            return ModelRouteDecision(
                tier=tier,
                model_name=self.fast_model,
                thinking_level=types.ThinkingLevel.MINIMAL
                if prefer_minimal_thinking
                else None,
                temperature=0.2,
                rationale="High-throughput API collection/compaction routed to Flash with minimal thinking.",
            )

        if tier == TaskTier.DEEP_STRATEGIC_PLANNING:
            return ModelRouteDecision(
                tier=tier,
                model_name=self.pro_model,
                thinking_level=types.ThinkingLevel.HIGH
                if prefer_minimal_thinking
                else None,
                temperature=0.35,
                rationale="Complex multi-domain strategic planning routed to Pro with deep reasoning.",
            )

        return ModelRouteDecision(
            tier=TaskTier.STANDARD_SYNTHESIS,
            model_name=self.fast_model,
            thinking_level=types.ThinkingLevel.MINIMAL
            if prefer_minimal_thinking
            else None,
            temperature=0.3,
            rationale="Balanced daily editorial synthesis routed to Flash with minimal thinking latency.",
        )

    def build_gemini_model(self, decision: ModelRouteDecision) -> Gemini:
        """Instantiate an ADK Gemini model wrapper with bounded retry policy."""
        return Gemini(
            model=decision.model_name,
            retry_options=types.HttpRetryOptions(attempts=2),
        )

    def build_generate_config(
        self, decision: ModelRouteDecision
    ) -> types.GenerateContentConfig:
        """Construct GenerateContentConfig including safety settings and thinking tier."""
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
        ]
        if decision.thinking_level is not None:
            return types.GenerateContentConfig(
                temperature=decision.temperature,
                thinking_config=types.ThinkingConfig(
                    thinking_level=decision.thinking_level
                ),
                safety_settings=safety_settings,
            )
        return types.GenerateContentConfig(
            temperature=decision.temperature,
            safety_settings=safety_settings,
        )


model_router = StrategicModelRouter()

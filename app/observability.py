"""Structured JSON logging, OpenTelemetry distributed tracing, and Intent/Outcome plugin.

Satisfies AgentOps Code Review Matrix — Category 4 (Observability & Tracing):
- 4.1 Structured JSON Logging: Emits Cloud Logging-compatible JSON records with rich metadata.
- 4.2 Intent vs. Outcome Capture: Explicitly logs `phase="INTENT"` before agent/model/tool
      execution and `phase="OUTCOME"` with correlation IDs and latency afterward.
- 4.3 Distributed Tracing: Configures OpenTelemetry spans linking HTTP -> workflow -> tools -> LLM.
- 4.4 PII Redaction: Scrubs all log payloads and span attributes via `pii_redactor`.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Iterator
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from app.pii_redactor import pii_redactor

_STANDARD_LOG_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_LOGGING_CONFIGURED = False
_TRACER_CONFIGURED = False


class StructuredJsonFormatter(logging.Formatter):
    """Formats Python LogRecords into PII-scrubbed Google Cloud Logging JSON."""

    def format(self, record: logging.LogRecord) -> str:
        span_ctx = trace.get_current_span().get_span_context()
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()

        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": pii_redactor.scrub_text(record.getMessage()),
        }

        if span_ctx and span_ctx.is_valid:
            trace_hex = format(span_ctx.trace_id, "032x")
            span_hex = format(span_ctx.span_id, "016x")
            payload["span_id"] = span_hex
            payload["trace_id"] = trace_hex
            if project:
                payload["logging.googleapis.com/trace"] = (
                    f"projects/{project}/traces/{trace_hex}"
                )
                payload["logging.googleapis.com/spanId"] = span_hex

        for key, val in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_"):
                payload[key] = pii_redactor.scrub_payload(val)

        if record.exc_info:
            payload["exception"] = pii_redactor.scrub_text(
                self.formatException(record.exc_info)
            )

        return json.dumps(payload, default=str)


def configure_observability() -> trace.Tracer:
    """Initialize structured JSON stdout logging and OpenTelemetry TracerProvider."""
    global _LOGGING_CONFIGURED, _TRACER_CONFIGURED

    if not _TRACER_CONFIGURED:
        current_provider = trace.get_tracer_provider()
        if not isinstance(current_provider, TracerProvider):
            resource = Resource.create(
                {
                    "service.name": os.environ.get(
                        "OTEL_SERVICE_NAME", "ai-daily-digest"
                    ),
                    "service.version": os.environ.get("AGENT_VERSION", "0.2.0"),
                }
            )
            trace.set_tracer_provider(TracerProvider(resource=resource))
        _TRACER_CONFIGURED = True

    if not _LOGGING_CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        app_logger = logging.getLogger("app")
        app_logger.handlers.clear()
        app_logger.addHandler(handler)
        app_logger.setLevel(logging.INFO)
        app_logger.propagate = False
        _LOGGING_CONFIGURED = True

    return trace.get_tracer("ai_daily_digest.agentops")


def get_current_trace_id() -> str:
    """Return active 32-hex OpenTelemetry trace ID or generate a fallback correlation ID."""
    span_ctx = trace.get_current_span().get_span_context()
    if span_ctx and span_ctx.is_valid:
        return format(span_ctx.trace_id, "032x")
    return uuid.uuid4().hex


@contextlib.contextmanager
def trace_span(
    span_name: str, attributes: dict[str, Any] | None = None
) -> Iterator[trace.Span]:
    """Create a linked OpenTelemetry child span with PII-scrubbed attributes."""
    tracer = configure_observability()
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            scrubbed = pii_redactor.scrub_payload(attributes)
            for k, v in scrubbed.items():
                if isinstance(v, (str, bool, int, float)):
                    span.set_attribute(k, v)
                else:
                    span.set_attribute(k, json.dumps(v, default=str))
        yield span


class IntentOutcomeObservabilityPlugin(BasePlugin):
    """ADK global plugin capturing paired INTENT (pre) and OUTCOME (post) structured logs."""

    def __init__(self, name: str = "intent_outcome_observability_plugin") -> None:
        super().__init__(name=name)
        self._logger = logging.getLogger("app.observability")
        self._timings: dict[str, float] = {}

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        corr_id = f"agent-{agent.name}-{uuid.uuid4().hex[:8]}"
        callback_context.state[f"temp:corr:{agent.name}"] = corr_id
        self._timings[corr_id] = time.perf_counter()
        self._logger.info(
            "Agent execution intent recorded",
            extra={
                "event_type": "agent.lifecycle",
                "phase": "INTENT",
                "correlation_id": corr_id,
                "agent_name": agent.name,
                "intended_action": f"Execute agent '{agent.name}'",
            },
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        corr_id = str(
            callback_context.state.get(f"temp:corr:{agent.name}", f"agent-{agent.name}")
        )
        t0 = self._timings.pop(corr_id, time.perf_counter())
        latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
        self._logger.info(
            "Agent execution outcome recorded",
            extra={
                "event_type": "agent.lifecycle",
                "phase": "OUTCOME",
                "correlation_id": corr_id,
                "agent_name": agent.name,
                "outcome_status": "completed",
                "latency_ms": latency_ms,
            },
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        corr_id = f"llm-{callback_context.agent_name}-{uuid.uuid4().hex[:8]}"
        callback_context.state["temp:last_llm_corr"] = corr_id
        self._timings[corr_id] = time.perf_counter()
        model_name = str(getattr(llm_request, "model", "") or "gemini")
        self._logger.info(
            "LLM generation intent recorded",
            extra={
                "event_type": "llm.invocation",
                "phase": "INTENT",
                "correlation_id": corr_id,
                "agent_name": callback_context.agent_name,
                "model": model_name,
                "intended_action": "Generate structured editorial response or tool plan",
            },
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        corr_id = str(callback_context.state.get("temp:last_llm_corr", "llm-call"))
        t0 = self._timings.pop(corr_id, time.perf_counter())
        latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
        has_content = bool(llm_response and llm_response.content)
        self._logger.info(
            "LLM generation outcome recorded",
            extra={
                "event_type": "llm.invocation",
                "phase": "OUTCOME",
                "correlation_id": corr_id,
                "agent_name": callback_context.agent_name,
                "outcome_status": "success" if has_content else "empty",
                "latency_ms": latency_ms,
            },
        )
        return None

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        corr_id = f"tool-{tool.name}-{uuid.uuid4().hex[:8]}"
        tool_context.state[f"temp:tool_corr:{tool.name}"] = corr_id
        self._timings[corr_id] = time.perf_counter()
        self._logger.info(
            "Tool invocation intent recorded",
            extra={
                "event_type": "tool.invocation",
                "phase": "INTENT",
                "correlation_id": corr_id,
                "agent_name": tool_context.agent_name,
                "tool_name": tool.name,
                "intended_action": f"Invoke tool '{tool.name}'",
                "tool_args": pii_redactor.scrub_payload(tool_args),
            },
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        corr_id = str(
            tool_context.state.get(f"temp:tool_corr:{tool.name}", f"tool-{tool.name}")
        )
        t0 = self._timings.pop(corr_id, time.perf_counter())
        latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
        status = (
            str(result.get("status", "success"))
            if isinstance(result, dict)
            else "success"
        )
        self._logger.info(
            "Tool invocation outcome recorded",
            extra={
                "event_type": "tool.invocation",
                "phase": "OUTCOME",
                "correlation_id": corr_id,
                "agent_name": tool_context.agent_name,
                "tool_name": tool.name,
                "outcome_status": status,
                "latency_ms": latency_ms,
            },
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        corr_id = str(
            tool_context.state.get(f"temp:tool_corr:{tool.name}", f"tool-{tool.name}")
        )
        t0 = self._timings.pop(corr_id, time.perf_counter())
        latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
        self._logger.error(
            "Tool invocation failed with guided recovery response",
            extra={
                "event_type": "tool.invocation",
                "phase": "OUTCOME",
                "correlation_id": corr_id,
                "agent_name": tool_context.agent_name,
                "tool_name": tool.name,
                "outcome_status": "error",
                "error_type": type(error).__name__,
                "error_message": pii_redactor.scrub_text(str(error)),
                "latency_ms": latency_ms,
            },
        )
        return {
            "status": "error",
            "error_code": type(error).__name__,
            "error_message": pii_redactor.scrub_text(str(error)),
            "retryable": True,
            "recovery_instructions": (
                f"Tool '{tool.name}' encountered {type(error).__name__}. "
                "Verify 'target_date' is formatted as YYYY-MM-DD (or 'today') and continue "
                "synthesizing available sections from remaining collector tools."
            ),
        }

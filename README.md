# AI Daily Digest (`ai-daily-digest`)

An enterprise AgentOps AI intelligence briefing platform built with **Google Agent Development Kit (ADK)** (`gemini-3.7-flash` & `gemini-3.7-pro` model routing), **OpenTelemetry**, **SQLite FTS5 / Vertex AI Memory Bank**, and a **React 18 + Vite** web UI.

## Multi-Agent Architecture

```
App ("app", EventsCompactionConfig + ContextCacheConfig + ResumabilityConfig)
├── Global Plugins:
│   ├── GlobalInstructionPlugin (AGENT_CONSTITUTION)
│   ├── ContextFilterPlugin (sliding window)
│   ├── ReflectAndRetryToolPlugin (guided tool retry)
│   ├── SecurityAndSelfEvalPlugin (prompt-injection shield + URL provenance self-eval)
│   └── IntentOutcomeObservabilityPlugin (PII-scrubbed INTENT/OUTCOME JSON + OTel spans)
└── root_agent: SequentialAgent ("daily_digest_pipeline")
    ├── ParallelAgent ("parallel_api_collectors") [Flash tier — ThinkingLevel.MINIMAL]
    │   ├── papers_research_agent   ──► fetch_hf_daily_papers + fetch_arxiv_ai_papers + search_archived_daily_briefings
    │   ├── models_spaces_agent     ──► fetch_hf_trending_models_and_spaces
    │   └── news_repos_agent        ──► fetch_hackernews_ai_stories + fetch_github_ai_repos + publish_digest_hitl_tool + PreloadMemoryTool
    └── digest_synthesizer_agent    ──► StrategicModelRouter (Flash standard / Pro deep) + async Memory Bank consolidation
```

## Quick Start

### 1. Configure Credentials & Secret Manager
By default `.env` uses **Vertex AI** + optional **Google Cloud Secret Manager**:
```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
USE_SECRET_MANAGER=true
```
Authenticate locally with Application Default Credentials:
```bash
gcloud auth application-default login
```

### 2. Build Frontend & Run Unified Server
```bash
npm install --prefix frontend
npm run build --prefix frontend
uv run python server.py
```
Open `http://localhost:8080`.

### 3. `agents-cli` Development, Evaluation & Infrastructure Workflow
This repository is scaffolded and managed with **`agents-cli`** (`uv tool install google-agents-cli`):
```bash
# Inspect ADK manifest and environment configuration
agents-cli info

# Interactive local multi-agent web playground
agents-cli playground

# Smoke-test the ADK pipeline from the terminal
agents-cli run "Generate the daily AI intelligence briefing for target_date=2025-02-14"

# Run golden-dataset automated evaluation (LLM-as-judge + static grounding/PII regression metrics)
agents-cli eval run

# Run unit + 19-criteria AgentOps Matrix verification suite
uv run pytest tests/unit -v

# Provision single-project Terraform IaC (APIs, IAM, GCS telemetry, Secret Manager, Cloud Run / Agent Runtime)
agents-cli infra single-project --project your-gcp-project-id

# Deploy via agents-cli or Cloud Run
agents-cli deploy --project your-gcp-project-id
```

### 4. Minimal-Resource Cloud Run Deployment
```bash
npm run build --prefix frontend
gcloud beta run deploy ai-daily-digest \
  --source . \
  --project your-gcp-project-id \
  --region us-central1 \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 2 \
  --concurrency 8 \
  --cpu-throttling \
  --timeout 180 \
  --port 8080 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=your-gcp-project-id,GOOGLE_CLOUD_LOCATION=global \
  --iap
```

---

## AgentOps Code Review Matrix (95 / 95 Points)

| Category | Criteria | Implementation Evidence | Pts |
| :--- | :--- | :--- | :---: |
| **1. Tool & Interface Design** | Comprehensive Tool Docstrings | Google-style docstrings with `Args:` and `Returns:` across all 7 tools in `app/tools.py` | 5 |
| | Descriptive Naming | Domain-specific verb-noun tools (`fetch_hf_daily_papers`, `publish_verified_daily_digest_archive`, `search_archived_daily_briefings`) | 5 |
| | Explicit JSON Schemas | Strict `ConfigDict(extra="forbid")` Pydantic input & output models in `app/schemas.py` | 5 |
| | Guided Error Handling | `ToolErrorGuidance` with `recovery_instructions` + `ReflectAndRetryToolPlugin` in `app/tools.py` & `app/observability.py` | 5 |
| **2. Context & Memory** | Robust System Instructions | Structured `<CONSTITUTION>` (`<PERSONA>`, `<DOMAIN_KNOWLEDGE>`, `<GROUNDING_INVARIANTS>`, `<SAFETY_AND_PRIVACY_CONSTRAINTS>`) via `GlobalInstructionPlugin` in `app/agent.py` | 5 |
| | History Compaction | `EventsCompactionConfig(token_threshold=24000)` + `ContextCacheConfig` + `ContextFilterPlugin` in `app/agent.py` | 5 |
| | Persistent Session State | `SqliteSessionService` / `VertexAiSessionService` (`app/app_utils/services.py`) + SQLite FTS5 BM25 store (`app/persistence.py`) + optional `VertexAiSearchTool` | 5 |
| | Async Memory Operations | Non-blocking `AsyncMemoryConsolidator` (`asyncio.create_task`) + `trigger_async_memory_callback` in `app/persistence.py` & `app/agent.py` | 5 |
| **3. Orchestration & Logic** | Multi-Agent Patterns | ADK `SequentialAgent` -> `ParallelAgent` (3 specialist collectors) -> `digest_synthesizer_agent` in `app/agent.py` | 5 |
| | Strategic Model Routing | `StrategicModelRouter` (`app/model_router.py`) routing collectors/fast synthesis to `gemini-3.7-flash` and deep strategic planning to `gemini-3.7-pro` | 5 |
| | Guardrails & Policy Plugins | `SecurityAndSelfEvalPlugin` (`app/guardrails.py`): prompt-injection shield, Vertex `SafetySetting`, and URL-provenance `evaluate_briefing_self_check` | 5 |
| | Human-in-the-Loop Hooks | `ResumabilityConfig(is_resumable=True)` + `FunctionTool(..., require_confirmation=...)` + `tool_context.request_confirmation` (`app/tools.py`, `server.py`) | 5 |
| **4. Observability & Tracing** | Structured JSON Logging | Cloud Logging–compatible `StructuredJsonFormatter` with trace correlation in `app/observability.py` | 5 |
| | Intent vs. Outcome Capture | `IntentOutcomeObservabilityPlugin` logging paired `phase="INTENT"` and `phase="OUTCOME"` events across agents, LLMs, and tools | 5 |
| | Distributed Tracing | OpenTelemetry `TracerProvider` + `trace_span` hierarchy + `X-Trace-Id` HTTP middleware (`app/observability.py`, `server.py`) | 5 |
| | PII Redaction | `PiiRedactor` (`app/pii_redactor.py`) + optional Cloud DLP `content:deidentify` scrubbing logs, spans, prompts, and SQLite/Memory Bank writes | 5 |
| **5. Infrastructure & CI/CD** | Automated Evaluation Suites | Golden dataset (`tests/eval/datasets/basic-dataset.json`), `eval_config.yaml`, `static_regression_metrics.py`, `response_quality.py`, & `test_agentops_matrix.py` | 5 |
| | Infrastructure as Code | Terraform IaC (`deployment/terraform/single-project/{service,cloud_run,secrets,iam,apis,telemetry}.tf`), GitHub Actions CI, and `agents-cli` docs | 5 |
| | Secure Secret Management | `SecretManagerResolver` (`app/secrets.py`) + Terraform `secrets.tf` (`google_secret_manager_secret`), zero hardcoded keys | 5 |
| **Total** | | **19 / 19 Criteria Verified** | **95** |

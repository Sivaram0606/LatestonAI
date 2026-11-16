# AI Daily Digest (`ai-daily-digest`)

An agentic AI intelligence briefing system built with **Google Agent Development Kit (ADK)** (`gemini-3.7-flash`) and a **React + Vite** web UI. For any selected date (`YYYY-MM-DD`), specialist agents pull live data in parallel across popular AI APIs and synthesize a structured daily briefing.

## Multi-Agent Architecture

```
root_agent: SequentialAgent ("daily_digest_pipeline")
├── ParallelAgent ("parallel_api_collectors")
│   ├── papers_research_agent   ──► Hugging Face Daily Papers API + arXiv API (cs.AI / cs.LG / cs.CL)
│   ├── models_spaces_agent     ──► Hugging Face Hub Trending Models & Interactive Spaces API
│   └── news_repos_agent        ──► Hacker News Algolia API (24h UTC window) + GitHub Search API
└── digest_synthesizer_agent    ──► Synthesizes structured DailyBriefingLLMOutput JSON
```

## Quick Start

### 1. Configure LLM Credentials (`.env` or via UI "LLM Settings")
By default `.env` uses **Vertex AI**:
```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
```
Authenticate with ADC if needed:
```bash
gcloud auth application-default login
```
Alternatively, set `GOOGLE_GENAI_USE_VERTEXAI=false` and `GEMINI_API_KEY=your-key` (or configure at runtime from the **LLM Settings** button in the top bar). Even before LLM credentials are configured, the backend still pulls live daily data from all external AI APIs and renders a deterministic fallback briefing.

### 2. Build Frontend & Start Unified Server
```bash
npm install --prefix frontend
npm run build --prefix frontend
uv run python server.py
```
Open `http://localhost:8080`.

### 3. Development Mode (Hot-Reload UI + API)
```bash
# Terminal 1 — FastAPI + ADK backend on :8080
uv run python server.py

# Terminal 2 — Vite React dev server on :5173 (proxies /api -> :8080)
npm run dev --prefix frontend
```

### 4. ADK CLI & Tests
```bash
uv run pytest tests/unit
agents-cli playground
```

### 5. Deploy to Google Cloud Run (Minimal Resources)
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
  --allow-unauthenticated
```

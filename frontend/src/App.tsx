import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BookOpen,
  Bot,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Cpu,
  ExternalLink,
  GitFork,
  Heart,
  Loader2,
  MessageSquare,
  Moon,
  Newspaper,
  RefreshCw,
  Search,
  Settings,
  Star,
  Sun,
  ThumbsUp,
} from "lucide-react";
import { DailyBriefingResponse, HealthStatus } from "./types";

type SectionTab = "all" | "papers" | "models" | "news" | "repos";

function formatReadableDate(isoDate: string): string {
  const parts = isoDate.split("-").map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) return isoDate;
  const dt = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
    timeZone: "UTC",
  }).format(dt);
}

function shiftIsoDate(isoDate: string, deltaDays: number): string {
  const parts = isoDate.split("-").map(Number);
  const dt = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2] + deltaDays));
  return dt.toISOString().slice(0, 10);
}

function getTodayUtcIso(): string {
  return new Date().toISOString().slice(0, 10);
}

export function App(): React.ReactElement {
  const [todayUtc, setTodayUtc] = useState<string>(getTodayUtcIso);
  const [selectedDate, setSelectedDate] = useState<string>(getTodayUtcIso);
  const [digest, setDigest] = useState<DailyBriefingResponse | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [activeTab, setActiveTab] = useState<SectionTab>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [traceExpanded, setTraceExpanded] = useState<boolean>(true);
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("ai-digest-theme");
    if (saved === "dark" || saved === "light") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });

  const [configOpen, setConfigOpen] = useState<boolean>(false);
  const [cfgUseVertex, setCfgUseVertex] = useState<boolean>(true);
  const [cfgProject, setCfgProject] = useState<string>("");
  const [cfgLocation, setCfgLocation] = useState<string>("global");
  const [cfgApiKey, setCfgApiKey] = useState<string>("");
  const [savingCfg, setSavingCfg] = useState<boolean>(false);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("ai-digest-theme", theme);
  }, [theme]);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((h: HealthStatus | null) => {
        if (!h) return;
        setHealth(h);
        if (h.server_utc_date) {
          setTodayUtc(h.server_utc_date);
          setSelectedDate(h.server_utc_date);
        }
        setCfgUseVertex(h.auth_mode === "vertex_ai");
        setCfgProject(h.google_cloud_project || "");
        setCfgLocation(h.google_cloud_location || "global");
      })
      .catch(() => {});
  }, []);

  const loadDigest = useCallback(async (dateStr: string, refresh = false) => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        date: dateStr,
        refresh: String(refresh),
      });
      const resp = await fetch(`/api/digest?${params.toString()}`);
      if (!resp.ok) {
        throw new Error(`API returned HTTP ${resp.status}`);
      }
      const data: DailyBriefingResponse = await resp.json();
      setDigest(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to pull daily briefing from backend."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDigest(selectedDate, false);
  }, [selectedDate, loadDigest]);

  const handleSaveConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavingCfg(true);
    try {
      const resp = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          use_vertexai: cfgUseVertex,
          google_cloud_project: cfgProject,
          google_cloud_location: cfgLocation,
          gemini_api_key: cfgApiKey,
        }),
      });
      if (resp.ok) {
        const updated: HealthStatus = await resp.json();
        setHealth(updated);
        setConfigOpen(false);
        void loadDigest(selectedDate, true);
      }
    } finally {
      setSavingCfg(false);
    }
  };

  const q = searchQuery.trim().toLowerCase();

  const filteredPapers = useMemo(() => {
    const list = digest?.briefing.papers ?? [];
    if (!q) return list;
    return list.filter((p) =>
      `${p.title} ${p.authors} ${p.summary} ${p.tags.join(" ")}`
        .toLowerCase()
        .includes(q)
    );
  }, [digest, q]);

  const filteredModels = useMemo(() => {
    const list = digest?.briefing.models ?? [];
    if (!q) return list;
    return list.filter((m) =>
      `${m.name} ${m.author} ${m.pipeline_tag} ${m.summary}`
        .toLowerCase()
        .includes(q)
    );
  }, [digest, q]);

  const filteredNews = useMemo(() => {
    const list = digest?.briefing.news ?? [];
    if (!q) return list;
    return list.filter((n) =>
      `${n.title} ${n.source} ${n.summary}`.toLowerCase().includes(q)
    );
  }, [digest, q]);

  const filteredRepos = useMemo(() => {
    const list = digest?.briefing.repos ?? [];
    if (!q) return list;
    return list.filter((r) =>
      `${r.full_name} ${r.language} ${r.summary} ${r.topics.join(" ")}`
        .toLowerCase()
        .includes(q)
    );
  }, [digest, q]);

  const canGoNext = selectedDate < todayUtc;

  return (
    <div>
      <header className="masthead">
        <div className="masthead-inner">
          <div className="brand-group">
            <div className="brand-mark" aria-hidden="true">
              <Bot size={20} />
            </div>
            <div>
              <div className="brand-title">AI Daily Digest</div>
              <div className="brand-subtitle">
                Google ADK Multi-Agent Briefing
              </div>
            </div>
          </div>

          <div className="controls-bar">
            <div className="date-cluster">
              <button
                id="btn-prev-date"
                type="button"
                className="btn btn-icon"
                disabled={loading}
                onClick={() => setSelectedDate(shiftIsoDate(selectedDate, -1))}
                title="Previous day"
                aria-label="Previous day"
              >
                <ChevronLeft size={16} />
              </button>
              <input
                id="digest-date-picker"
                type="date"
                className="date-input"
                value={selectedDate}
                max={todayUtc}
                disabled={loading}
                onChange={(e) => {
                  if (e.target.value) setSelectedDate(e.target.value);
                }}
                aria-label="Select briefing date"
              />
              <button
                id="btn-next-date"
                type="button"
                className="btn btn-icon"
                disabled={loading || !canGoNext}
                onClick={() => setSelectedDate(shiftIsoDate(selectedDate, 1))}
                title="Next day"
                aria-label="Next day"
              >
                <ChevronRight size={16} />
              </button>
            </div>

            <button
              id="btn-today-date"
              type="button"
              className="btn"
              disabled={loading || selectedDate === todayUtc}
              onClick={() => setSelectedDate(todayUtc)}
            >
              Today
            </button>

            <button
              id="btn-refresh-digest"
              type="button"
              className="btn btn-primary"
              disabled={loading}
              onClick={() => void loadDigest(selectedDate, true)}
            >
              <RefreshCw size={15} className={loading ? "spin" : ""} />
              {loading ? "Running Agents…" : "Refresh"}
            </button>

            <button
              id="btn-llm-settings"
              type="button"
              className="btn"
              disabled={loading}
              onClick={() => setConfigOpen(true)}
              title="Configure Vertex AI or Gemini API Key"
            >
              <Settings size={15} />
              LLM Settings
            </button>

            <button
              id="btn-theme-toggle"
              type="button"
              className="btn"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label="Toggle dark mode"
              title="Toggle light/dark mode"
            >
              {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </div>
        </div>
      </header>

      <main className="page-container">
        {error && (
          <div className="notice-banner" role="alert">
            <span>{error}</span>
            <button
              id="btn-retry-error"
              type="button"
              className="btn"
              disabled={loading}
              onClick={() => void loadDigest(selectedDate, true)}
            >
              Retry
            </button>
          </div>
        )}

        {loading && (
          <div className="loading-state" aria-live="polite">
            <Loader2
              size={30}
              className="spin"
              style={{ color: "var(--accent)", marginBottom: "0.75rem" }}
            />
            <div style={{ fontWeight: 600, fontSize: "1.05rem" }}>
              Running ADK Parallel Collector &amp; Synthesizer Agents for{" "}
              {formatReadableDate(selectedDate)}…
            </div>
            <p
              style={{
                color: "var(--text-muted)",
                fontSize: "0.86rem",
                marginTop: "0.4rem",
              }}
            >
              Querying Hugging Face Daily Papers &amp; Hub API, arXiv preprints,
              Hacker News Algolia, and GitHub AI repositories.
            </p>
          </div>
        )}

        {!loading && digest && (
          <>
            {digest.notice && (
              <div className="notice-banner">
                <span>{digest.notice}</span>
                <button
                  id="btn-banner-configure"
                  type="button"
                  className="btn"
                  onClick={() => setConfigOpen(true)}
                >
                  Configure
                </button>
              </div>
            )}

            {/* Agent Pipeline Telemetry */}
            <section
              className="trace-panel"
              aria-label="Agentic workflow trace"
            >
              <div
                id="btn-toggle-trace"
                className="trace-header"
                onClick={() => setTraceExpanded(!traceExpanded)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    setTraceExpanded(!traceExpanded);
                  }
                }}
              >
                <div className="trace-title-group">
                  <Activity size={16} style={{ color: "var(--accent)" }} />
                  <span>ADK Multi-Agent Pipeline Execution</span>
                  <span className="badge badge-neutral">
                    {digest.generation_mode === "adk_multi_agent"
                      ? `ADK · ${digest.model_used}`
                      : "Live API Parallel Mode"}
                  </span>
                </div>
                {traceExpanded ? (
                  <ChevronUp size={16} />
                ) : (
                  <ChevronDown size={16} />
                )}
              </div>

              {traceExpanded && (
                <div className="trace-grid">
                  {digest.agent_trace.map((step) => (
                    <div key={step.agent_name} className="trace-step">
                      <div className="trace-step-top">
                        <span
                          className="mono-tag"
                          style={{ fontWeight: 600, color: "var(--text-primary)" }}
                        >
                          <span className="status-dot" /> {step.agent_name}
                        </span>
                        <span
                          className="mono-tag"
                          style={{ color: "var(--text-muted)" }}
                        >
                          {step.duration_ms} ms
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: "0.78rem",
                          color: "var(--text-secondary)",
                          marginBottom: "0.2rem",
                        }}
                      >
                        {step.role}
                      </div>
                      <div
                        className="mono-tag"
                        style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}
                      >
                        {step.detail} ({step.items_collected} items)
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* Executive Summary Hero */}
            <article className="hero-card">
              <div className="hero-meta">
                <span className="badge">
                  {formatReadableDate(digest.target_date)}
                </span>
                <span>ISO {digest.target_date}</span>
                <span>•</span>
                <span>Generated {digest.generated_at}</span>
              </div>

              <h1 className="hero-headline">{digest.briefing.headline}</h1>

              <div className="executive-prose">
                {digest.briefing.executive_summary
                  .split(/\n\s*\n/)
                  .filter(Boolean)
                  .map((para, idx) => (
                    <p key={idx}>{para}</p>
                  ))}
              </div>

              {digest.briefing.key_themes.length > 0 && (
                <div className="themes-list" aria-label="Key daily themes">
                  {digest.briefing.key_themes.map((themeItem, idx) => (
                    <span key={idx} className="theme-chip">
                      {themeItem}
                    </span>
                  ))}
                </div>
              )}

              <div className="kpi-strip">
                <div className="kpi-box">
                  <span className="kpi-value">
                    {digest.briefing.papers.length}
                  </span>
                  <span className="kpi-label">Research Papers</span>
                </div>
                <div className="kpi-box">
                  <span className="kpi-value">
                    {digest.briefing.models.length}
                  </span>
                  <span className="kpi-label">Trending Models &amp; Spaces</span>
                </div>
                <div className="kpi-box">
                  <span className="kpi-value">
                    {digest.briefing.news.length}
                  </span>
                  <span className="kpi-label">Community Discussions</span>
                </div>
                <div className="kpi-box">
                  <span className="kpi-value">
                    {digest.briefing.repos.length}
                  </span>
                  <span className="kpi-label">Breakout GitHub Repos</span>
                </div>
              </div>
            </article>

            {/* Category Navigation & Search */}
            <div className="toolbar-row relative z-30">
              <div className="tabs-group" role="tablist">
                {(
                  [
                    ["all", "All Briefing Sections"],
                    ["papers", `Papers (${filteredPapers.length})`],
                    ["models", `Models & Spaces (${filteredModels.length})`],
                    ["news", `News (${filteredNews.length})`],
                    ["repos", `Repos (${filteredRepos.length})`],
                  ] as const
                ).map(([tabKey, label]) => (
                  <button
                    id={`tab-${tabKey}`}
                    key={tabKey}
                    type="button"
                    role="tab"
                    aria-selected={activeTab === tabKey}
                    className={`tab-btn ${activeTab === tabKey ? "active" : ""}`}
                    onClick={() => setActiveTab(tabKey)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="search-box">
                <Search size={15} style={{ color: "var(--text-muted)" }} />
                <input
                  id="digest-filter-input"
                  type="search"
                  placeholder="Filter papers, models, news, repos…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  aria-label="Filter briefing items"
                />
              </div>
            </div>

            {/* 1. Research Papers */}
            {(activeTab === "all" || activeTab === "papers") && (
              <section className="digest-section">
                <div className="section-header">
                  <h2 className="section-title">
                    <BookOpen size={19} style={{ color: "var(--accent)" }} />
                    Research Papers
                  </h2>
                  <span className="mono-tag" style={{ color: "var(--text-muted)" }}>
                    Hugging Face Daily Papers &amp; arXiv
                  </span>
                </div>

                {filteredPapers.length === 0 ? (
                  <p style={{ color: "var(--text-muted)" }}>
                    No matching research papers for this filter.
                  </p>
                ) : (
                  <div className="cards-grid">
                    {filteredPapers.map((paper, idx) => (
                      <article key={`${paper.url}-${idx}`} className="item-card">
                        <div>
                          <div className="card-top-meta">
                            <span>{paper.source}</span>
                            <span>
                              {formatReadableDate(paper.published_date)}
                            </span>
                          </div>
                          <h3 className="card-title">
                            <a
                              href={paper.url}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {paper.title}
                            </a>
                          </h3>
                          <div className="card-subtitle">{paper.authors}</div>
                          <p className="card-summary">{paper.summary}</p>
                        </div>

                        <div className="card-footer">
                          <div className="tag-row">
                            {paper.upvotes > 0 && (
                              <span className="mini-tag">
                                <ThumbsUp
                                  size={11}
                                  style={{ marginRight: 4, verticalAlign: "-1px" }}
                                />
                                {paper.upvotes}
                              </span>
                            )}
                            {paper.tags.slice(0, 3).map((t) => (
                              <span key={t} className="mini-tag">
                                {t}
                              </span>
                            ))}
                          </div>
                          <a
                            className="ext-link"
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Paper <ExternalLink size={13} />
                          </a>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* 2. Models & Spaces */}
            {(activeTab === "all" || activeTab === "models") && (
              <section className="digest-section">
                <div className="section-header">
                  <h2 className="section-title">
                    <Cpu size={19} style={{ color: "var(--accent)" }} />
                    Models &amp; Interactive Spaces
                  </h2>
                  <span className="mono-tag" style={{ color: "var(--text-muted)" }}>
                    Hugging Face Hub API
                  </span>
                </div>

                {filteredModels.length === 0 ? (
                  <p style={{ color: "var(--text-muted)" }}>
                    No matching models or Spaces for this filter.
                  </p>
                ) : (
                  <div className="cards-grid">
                    {filteredModels.map((model, idx) => (
                      <article key={`${model.url}-${idx}`} className="item-card">
                        <div>
                          <div className="card-top-meta">
                            <span className="badge badge-neutral">
                              {model.kind}
                            </span>
                            <span>{model.pipeline_tag}</span>
                          </div>
                          <h3
                            className="card-title"
                            style={{
                              fontFamily: "var(--font-mono)",
                              fontSize: "0.94rem",
                              wordBreak: "break-word",
                            }}
                          >
                            <a
                              href={model.url}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {model.name}
                            </a>
                          </h3>
                          <div className="card-subtitle">by {model.author}</div>
                          <p className="card-summary">{model.summary}</p>
                        </div>

                        <div className="card-footer">
                          <div className="tag-row">
                            <span className="mini-tag">
                              <Heart
                                size={11}
                                style={{ marginRight: 4, verticalAlign: "-1px" }}
                              />
                              {model.likes.toLocaleString()}
                            </span>
                            {model.downloads > 0 && (
                              <span className="mini-tag">
                                {model.downloads.toLocaleString()} dl
                              </span>
                            )}
                          </div>
                          <a
                            className="ext-link"
                            href={model.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Open Hub <ExternalLink size={13} />
                          </a>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* 3. AI News & Discussions */}
            {(activeTab === "all" || activeTab === "news") && (
              <section className="digest-section">
                <div className="section-header">
                  <h2 className="section-title">
                    <Newspaper size={19} style={{ color: "var(--accent)" }} />
                    Community News &amp; Discussions
                  </h2>
                  <span className="mono-tag" style={{ color: "var(--text-muted)" }}>
                    Hacker News Algolia 24h Window
                  </span>
                </div>

                {filteredNews.length === 0 ? (
                  <p style={{ color: "var(--text-muted)" }}>
                    No matching news stories found for this date/filter.
                  </p>
                ) : (
                  <div className="cards-grid">
                    {filteredNews.map((story, idx) => (
                      <article key={`${story.discussion_url}-${idx}`} className="item-card">
                        <div>
                          <div className="card-top-meta">
                            <span>{story.source}</span>
                            <span>{story.points} pts</span>
                          </div>
                          <h3 className="card-title">
                            <a
                              href={story.url}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {story.title}
                            </a>
                          </h3>
                          <p className="card-summary">{story.summary}</p>
                        </div>

                        <div className="card-footer">
                          <a
                            className="mini-tag"
                            href={story.discussion_url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            <MessageSquare
                              size={11}
                              style={{ marginRight: 4, verticalAlign: "-1px" }}
                            />
                            {story.comments} comments
                          </a>
                          <a
                            className="ext-link"
                            href={story.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Read <ExternalLink size={13} />
                          </a>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* 4. Open-Source AI Repositories */}
            {(activeTab === "all" || activeTab === "repos") && (
              <section className="digest-section">
                <div className="section-header">
                  <h2 className="section-title">
                    <GitFork size={19} style={{ color: "var(--accent)" }} />
                    Breakout Open-Source Repositories
                  </h2>
                  <span className="mono-tag" style={{ color: "var(--text-muted)" }}>
                    GitHub Search API
                  </span>
                </div>

                {filteredRepos.length === 0 ? (
                  <p style={{ color: "var(--text-muted)" }}>
                    No matching GitHub repositories found for this filter.
                  </p>
                ) : (
                  <div className="cards-grid">
                    {filteredRepos.map((repo, idx) => (
                      <article key={`${repo.url}-${idx}`} className="item-card">
                        <div>
                          <div className="card-top-meta">
                            <span>{repo.language}</span>
                            <span>
                              <Star
                                size={12}
                                style={{ marginRight: 3, verticalAlign: "-1px" }}
                              />
                              {repo.stars.toLocaleString()}
                            </span>
                          </div>
                          <h3
                            className="card-title"
                            style={{
                              fontFamily: "var(--font-mono)",
                              fontSize: "0.95rem",
                            }}
                          >
                            <a
                              href={repo.url}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {repo.full_name}
                            </a>
                          </h3>
                          <p className="card-summary">{repo.summary}</p>
                        </div>

                        <div className="card-footer">
                          <div className="tag-row">
                            <span className="mini-tag">
                              {repo.forks.toLocaleString()} forks
                            </span>
                            {repo.topics.slice(0, 3).map((t) => (
                              <span key={t} className="mini-tag">
                                {t}
                              </span>
                            ))}
                          </div>
                          <a
                            className="ext-link"
                            href={repo.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            GitHub <ExternalLink size={13} />
                          </a>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>
            )}
          </>
        )}
      </main>

      {/* Runtime LLM Settings Modal */}
      {configOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setConfigOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="llm-settings-title"
        >
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h2
              id="llm-settings-title"
              style={{ marginTop: 0, fontSize: "1.15rem" }}
            >
              ADK Agent LLM Configuration
            </h2>
            <p
              style={{
                fontSize: "0.84rem",
                color: "var(--text-secondary)",
                marginBottom: "1rem",
              }}
            >
              Model: <code>{health?.model || "gemini-3.7-flash"}</code>. Update
              Vertex AI project/location or supply a Gemini Developer API key.
            </p>

            <form onSubmit={(e) => void handleSaveConfig(e)}>
              <div className="form-field">
                <label htmlFor="cfg-mode">Backend Mode</label>
                <select
                  id="cfg-mode"
                  value={cfgUseVertex ? "vertex" : "apikey"}
                  onChange={(e) => setCfgUseVertex(e.target.value === "vertex")}
                >
                  <option value="vertex">
                    Vertex AI (Google Cloud Project + ADC)
                  </option>
                  <option value="apikey">
                    Gemini Developer API (GEMINI_API_KEY)
                  </option>
                </select>
              </div>

              {cfgUseVertex ? (
                <>
                  <div className="form-field">
                    <label htmlFor="cfg-project">GOOGLE_CLOUD_PROJECT</label>
                    <input
                      id="cfg-project"
                      type="text"
                      placeholder="my-gcp-project-id"
                      value={cfgProject}
                      onChange={(e) => setCfgProject(e.target.value)}
                    />
                  </div>
                  <div className="form-field">
                    <label htmlFor="cfg-location">GOOGLE_CLOUD_LOCATION</label>
                    <input
                      id="cfg-location"
                      type="text"
                      placeholder="global"
                      value={cfgLocation}
                      onChange={(e) => setCfgLocation(e.target.value)}
                    />
                  </div>
                </>
              ) : (
                <div className="form-field">
                  <label htmlFor="cfg-apikey">GEMINI_API_KEY</label>
                  <input
                    id="cfg-apikey"
                    type="password"
                    placeholder="AIza..."
                    value={cfgApiKey}
                    onChange={(e) => setCfgApiKey(e.target.value)}
                  />
                </div>
              )}

              <div
                style={{
                  display: "flex",
                  justifyContent: "flex-end",
                  gap: "0.5rem",
                  marginTop: "1.25rem",
                }}
              >
                <button
                  id="btn-cancel-config"
                  type="button"
                  className="btn"
                  disabled={savingCfg}
                  onClick={() => setConfigOpen(false)}
                >
                  Cancel
                </button>
                <button
                  id="btn-save-config"
                  type="submit"
                  className="btn btn-primary"
                  disabled={savingCfg}
                >
                  {savingCfg ? "Applying…" : "Save & Re-Run Digest"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

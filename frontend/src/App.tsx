import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BookOpen,
  Bot,
  Calendar,
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
  Sparkles,
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
    weekday: "short",
    month: "short",
    day: "2-digit",
    year: "numeric",
    timeZone: "UTC",
  }).format(dt);
}

interface WeekWindowInfo {
  offset: number;
  sundayIso: string;
  saturdayIso: string;
  queryDateIso: string;
  rangeLabel: string;
  fullHeading: string;
}

function computeWeekWindow(
  anchorTodayIso: string,
  weekOffset: number
): WeekWindowInfo {
  const parts = anchorTodayIso.split("-").map(Number);
  const y = parts[0] || 2026;
  const m = (parts[1] || 1) - 1;
  const d = parts[2] || 1;
  const anchorDt = new Date(Date.UTC(y, m, d));
  const dayOfWeek = anchorDt.getUTCDay(); // 0 = Sunday .. 6 = Saturday

  const sun = new Date(Date.UTC(y, m, d - dayOfWeek + weekOffset * 7));
  const sat = new Date(Date.UTC(y, m, d - dayOfWeek + weekOffset * 7 + 6));
  const sundayIso = sun.toISOString().slice(0, 10);
  const saturdayIso = sat.toISOString().slice(0, 10);
  const queryDateIso = weekOffset === 0 ? anchorTodayIso : saturdayIso;

  const shortFmt = new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "2-digit",
    timeZone: "UTC",
  });
  const endFmt = new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "2-digit",
    year: "numeric",
    timeZone: "UTC",
  });
  const fullFmt = new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "2-digit",
    year: "numeric",
    timeZone: "UTC",
  });

  const rangeLabel = `${shortFmt.format(sun)} – ${endFmt.format(sat)}`;
  const fullHeading = `Weekly AI Summary: ${fullFmt.format(sun)} – ${fullFmt.format(sat)}`;
  return {
    offset: weekOffset,
    sundayIso,
    saturdayIso,
    queryDateIso,
    rangeLabel,
    fullHeading,
  };
}

const WEEK_PRESETS: ReadonlyArray<{ offset: number; label: string }> = [
  { offset: 0, label: "This Week" },
  { offset: -1, label: "Previous Week" },
  { offset: -2, label: "-2 Weeks" },
  { offset: -3, label: "-3 Weeks" },
];

function getTodayUtcIso(): string {
  return new Date().toISOString().slice(0, 10);
}

const CITATION_SPLIT_RE = /(\[[PMNR]\d+(?:\s*,\s*[PMNR]\d+)*\])/g;
const SINGLE_CITE_RE = /^[PMNR]\d+$/;

export function App(): React.ReactElement {
  const [todayUtc, setTodayUtc] = useState<string>(getTodayUtcIso);
  const [selectedWeekOffset, setSelectedWeekOffset] = useState<number>(0);
  const [digest, setDigest] = useState<DailyBriefingResponse | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [activeTab, setActiveTab] = useState<SectionTab>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [traceExpanded, setTraceExpanded] = useState<boolean>(true);
  const [highlightedCite, setHighlightedCite] = useState<string>("");
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
        }
        setCfgUseVertex(h.auth_mode === "vertex_ai");
        setCfgProject(h.google_cloud_project || "");
        setCfgLocation(h.google_cloud_location || "global");
      })
      .catch(() => {});
  }, []);

  const activeWeek = useMemo(
    () => computeWeekWindow(todayUtc, selectedWeekOffset),
    [todayUtc, selectedWeekOffset]
  );
  const selectedDate = activeWeek.queryDateIso;

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
          : "Failed to pull weekly & daily briefing from backend."
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

  // Map citation IDs (P1, M1, N1, R1) to human-readable tooltips & target sections
  const citationMeta = useMemo(() => {
    const map = new Map<
      string,
      { label: string; section: SectionTab; sectionDomId: string }
    >();
    if (!digest) return map;
    digest.briefing.papers.forEach((p, i) => {
      const id = p.citation_id || `P${i + 1}`;
      map.set(id, {
        label: `${id} · Research Paper: ${p.title}`,
        section: "papers",
        sectionDomId: "section-papers",
      });
    });
    digest.briefing.models.forEach((m, i) => {
      const id = m.citation_id || `M${i + 1}`;
      map.set(id, {
        label: `${id} · ${m.kind}: ${m.name}`,
        section: "models",
        sectionDomId: "section-models",
      });
    });
    digest.briefing.news.forEach((n, i) => {
      const id = n.citation_id || `N${i + 1}`;
      map.set(id, {
        label: `${id} · Community News: ${n.title}`,
        section: "news",
        sectionDomId: "section-news",
      });
    });
    digest.briefing.repos.forEach((r, i) => {
      const id = r.citation_id || `R${i + 1}`;
      map.set(id, {
        label: `${id} · GitHub Repo: ${r.full_name}`,
        section: "repos",
        sectionDomId: "section-repos",
      });
    });
    return map;
  }, [digest]);

  const jumpToCitation = useCallback(
    (citeId: string) => {
      const meta = citationMeta.get(citeId);
      if (searchQuery) {
        setSearchQuery("");
      }
      if (activeTab !== "all" && meta && activeTab !== meta.section) {
        setActiveTab("all");
      }
      setHighlightedCite(citeId);
      window.setTimeout(() => {
        const el =
          document.getElementById(`cite-${citeId}`) ||
          (meta ? document.getElementById(meta.sectionDomId) : null);
        el?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 40);
      window.setTimeout(() => {
        setHighlightedCite((prev) => (prev === citeId ? "" : prev));
      }, 2800);
    },
    [activeTab, citationMeta, searchQuery]
  );

  const jumpToSection = useCallback(
    (sectionDomId: string, tab: SectionTab) => {
      if (activeTab !== "all" && activeTab !== tab) {
        setActiveTab("all");
      }
      window.setTimeout(() => {
        document
          .getElementById(sectionDomId)
          ?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 30);
    },
    [activeTab]
  );

  const renderTextWithCitations = useCallback(
    (text: string): React.ReactNode => {
      const parts = text.split(CITATION_SPLIT_RE);
      return parts.map((segment, idx) => {
        if (segment.startsWith("[") && segment.endsWith("]")) {
          const inner = segment.slice(1, -1);
          const ids = inner
            .split(",")
            .map((s) => s.trim())
            .filter((s) => SINGLE_CITE_RE.test(s));
          if (ids.length > 0) {
            return (
              <React.Fragment key={idx}>
                {ids.map((cid) => {
                  const meta = citationMeta.get(cid);
                  return (
                    <button
                      key={`${idx}-${cid}`}
                      type="button"
                      className="citation-chip"
                      title={meta?.label || `Jump to ${cid} below`}
                      onClick={() => jumpToCitation(cid)}
                    >
                      [{cid}]
                    </button>
                  );
                })}
              </React.Fragment>
            );
          }
        }
        return <React.Fragment key={idx}>{segment}</React.Fragment>;
      });
    },
    [citationMeta, jumpToCitation]
  );

  const q = searchQuery.trim().toLowerCase();

  const filteredPapers = useMemo(() => {
    const list = (digest?.briefing.papers ?? []).map((p, idx) => ({
      ...p,
      citation_id: p.citation_id || `P${idx + 1}`,
    }));
    if (!q) return list;
    return list.filter((p) =>
      `${p.citation_id} ${p.title} ${p.authors} ${p.summary} ${p.tags.join(" ")}`
        .toLowerCase()
        .includes(q)
    );
  }, [digest, q]);

  const filteredModels = useMemo(() => {
    const list = (digest?.briefing.models ?? []).map((m, idx) => ({
      ...m,
      citation_id: m.citation_id || `M${idx + 1}`,
    }));
    if (!q) return list;
    return list.filter((m) =>
      `${m.citation_id} ${m.name} ${m.author} ${m.pipeline_tag} ${m.summary}`
        .toLowerCase()
        .includes(q)
    );
  }, [digest, q]);

  const filteredNews = useMemo(() => {
    const list = (digest?.briefing.news ?? []).map((n, idx) => ({
      ...n,
      citation_id: n.citation_id || `N${idx + 1}`,
    }));
    if (!q) return list;
    return list.filter((n) =>
      `${n.citation_id} ${n.title} ${n.source} ${n.summary}`
        .toLowerCase()
        .includes(q)
    );
  }, [digest, q]);

  const filteredRepos = useMemo(() => {
    const list = (digest?.briefing.repos ?? []).map((r, idx) => ({
      ...r,
      citation_id: r.citation_id || `R${idx + 1}`,
    }));
    if (!q) return list;
    return list.filter((r) =>
      `${r.citation_id} ${r.full_name} ${r.language} ${r.summary} ${r.topics.join(" ")}`
        .toLowerCase()
        .includes(q)
    );
  }, [digest, q]);

  const canGoNextWeek = selectedWeekOffset < 0;
  const weeklyHeading =
    digest?.briefing.weekly_window_heading || activeWeek.fullHeading;
  const whatsNewBullets = digest?.briefing.whats_new_today ?? [];

  return (
    <div>
      <header className="masthead">
        <div className="masthead-inner">
          <div className="brand-group">
            <div className="brand-mark" aria-hidden="true">
              <Bot size={20} />
            </div>
            <div>
              <div className="brand-title">AI Weekly &amp; Daily Digest</div>
              <div className="brand-subtitle">
                Google ADK Multi-Agent Briefing (Sun–Sat Window + Daily Updates)
              </div>
            </div>
          </div>

          <div className="controls-bar">
            {/* Sunday-to-Saturday Weekly Window Stepper */}
            <div
              className="date-cluster"
              role="group"
              aria-label="Sunday to Saturday week selector"
            >
              <button
                id="btn-prev-week"
                type="button"
                className="btn btn-icon"
                disabled={loading}
                onClick={() => setSelectedWeekOffset((w) => w - 1)}
                title="Previous Sunday–Saturday week"
                aria-label="Previous Sunday–Saturday week"
              >
                <ChevronLeft size={16} />
              </button>
              <span className="week-range-badge" id="active-week-range">
                <Calendar size={14} style={{ color: "var(--accent)" }} />
                {activeWeek.rangeLabel}
              </span>
              <button
                id="btn-next-week"
                type="button"
                className="btn btn-icon"
                disabled={loading || !canGoNextWeek}
                onClick={() => setSelectedWeekOffset((w) => Math.min(0, w + 1))}
                title="Next Sunday–Saturday week"
                aria-label="Next Sunday–Saturday week"
              >
                <ChevronRight size={16} />
              </button>
            </div>

            {/* Quick Preset Week Selector: This Week, Previous Week, -2 Weeks, -3 Weeks */}
            <div
              className="week-presets-group"
              role="group"
              aria-label="Preset weeks"
            >
              {WEEK_PRESETS.map((preset) => {
                const info = computeWeekWindow(todayUtc, preset.offset);
                const isActive = selectedWeekOffset === preset.offset;
                return (
                  <button
                    id={`btn-week-${preset.offset === 0 ? "this" : Math.abs(preset.offset)}`}
                    key={preset.offset}
                    type="button"
                    disabled={loading}
                    className={`week-preset-btn ${isActive ? "active" : ""}`}
                    onClick={() => setSelectedWeekOffset(preset.offset)}
                    title={info.rangeLabel}
                  >
                    {preset.label}
                  </button>
                );
              })}
            </div>

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
              {activeWeek.fullHeading}…
            </div>
            <p
              style={{
                color: "var(--text-muted)",
                fontSize: "0.86rem",
                marginTop: "0.4rem",
              }}
            >
              Querying Sunday–Saturday &amp; daily updates across Hugging Face,
              arXiv, Hacker News, and GitHub.
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
                  {digest.self_eval && (
                    <span className="badge badge-neutral">
                      Self-Eval {(digest.self_eval.grounding_score * 100).toFixed(0)}%
                    </span>
                  )}
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
                          style={{
                            fontWeight: 600,
                            color: "var(--text-primary)",
                          }}
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
                        style={{
                          color: "var(--text-muted)",
                          fontSize: "0.72rem",
                        }}
                      >
                        {step.detail} ({step.items_collected} items)
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* 1. Weekly Summary Hero (Sunday to Saturday Window Heading) */}
            <article className="hero-card" id="weekly-summary-section">
              <h2 className="weekly-window-heading">
                <Calendar size={15} />
                {weeklyHeading}
              </h2>

              <div className="hero-meta">
                <span>
                  Sun–Sat Window: {activeWeek.sundayIso} →{" "}
                  {activeWeek.saturdayIso}
                </span>
                <span>•</span>
                <span>Generated {digest.generated_at}</span>
              </div>

              <h1 className="hero-headline">{digest.briefing.headline}</h1>

              <div className="executive-prose">
                {digest.briefing.executive_summary
                  .split(/\n\s*\n/)
                  .filter(Boolean)
                  .map((para, idx) => (
                    <p key={idx}>{renderTextWithCitations(para)}</p>
                  ))}
              </div>

              {/* Section Citation Index Jump Bar */}
              <div
                className="section-jump-bar"
                aria-label="Cited sections below"
              >
                <span
                  className="mono-tag"
                  style={{ color: "var(--text-muted)" }}
                >
                  Referenced Sections:
                </span>
                {digest.briefing.papers.length > 0 && (
                  <button
                    type="button"
                    className="section-jump-pill"
                    onClick={() => jumpToSection("section-papers", "papers")}
                  >
                    Papers [P1–P{digest.briefing.papers.length}]
                  </button>
                )}
                {digest.briefing.models.length > 0 && (
                  <button
                    type="button"
                    className="section-jump-pill"
                    onClick={() => jumpToSection("section-models", "models")}
                  >
                    Models &amp; Spaces [M1–M{digest.briefing.models.length}]
                  </button>
                )}
                {digest.briefing.news.length > 0 && (
                  <button
                    type="button"
                    className="section-jump-pill"
                    onClick={() => jumpToSection("section-news", "news")}
                  >
                    Community News [N1–N{digest.briefing.news.length}]
                  </button>
                )}
                {digest.briefing.repos.length > 0 && (
                  <button
                    type="button"
                    className="section-jump-pill"
                    onClick={() => jumpToSection("section-repos", "repos")}
                  >
                    GitHub Repos [R1–R{digest.briefing.repos.length}]
                  </button>
                )}
              </div>

              {digest.briefing.key_themes.length > 0 && (
                <div className="themes-list" aria-label="Key weekly themes">
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
                  <span className="kpi-label">Research Papers [P]</span>
                </div>
                <div className="kpi-box">
                  <span className="kpi-value">
                    {digest.briefing.models.length}
                  </span>
                  <span className="kpi-label">Models &amp; Spaces [M]</span>
                </div>
                <div className="kpi-box">
                  <span className="kpi-value">
                    {digest.briefing.news.length}
                  </span>
                  <span className="kpi-label">Community News [N]</span>
                </div>
                <div className="kpi-box">
                  <span className="kpi-value">
                    {digest.briefing.repos.length}
                  </span>
                  <span className="kpi-label">Breakout Repos [R]</span>
                </div>
              </div>
            </article>

            {/* 2. Separate "What's New Today" Bullet-Point Section */}
            <section
              id="whats-new-today"
              className="whats-new-card"
              aria-labelledby="whats-new-heading"
            >
              <div className="whats-new-header">
                <h2 id="whats-new-heading" className="whats-new-title">
                  <Sparkles size={19} style={{ color: "var(--accent)" }} />
                  {selectedWeekOffset === 0
                    ? `What's New Today — ${formatReadableDate(todayUtc)}`
                    : `What's New Today · Closing Highlights (${activeWeek.rangeLabel})`}
                </h2>
                <span className="badge">
                  Daily Updates · Click [P/M/N/R] to jump to source card
                </span>
              </div>

              {whatsNewBullets.length === 0 ? (
                <p style={{ color: "var(--text-muted)", margin: 0 }}>
                  No standalone daily updates recorded yet for{" "}
                  {digest.target_date}.
                </p>
              ) : (
                <ul className="whats-new-bullets">
                  {whatsNewBullets.map((bullet, idx) => (
                    <li key={idx}>{renderTextWithCitations(bullet)}</li>
                  ))}
                </ul>
              )}
            </section>

            {/* Category Navigation & Search */}
            <div className="toolbar-row relative z-30">
              <div className="tabs-group" role="tablist">
                {(
                  [
                    ["all", "All Referenced Sections"],
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
                  placeholder="Filter papers, models, news, repos, or [P1]…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  aria-label="Filter briefing items"
                />
              </div>
            </div>

            {/* Section 1: Research Papers */}
            {(activeTab === "all" || activeTab === "papers") && (
              <section id="section-papers" className="digest-section">
                <div className="section-header">
                  <h2 className="section-title">
                    <BookOpen size={19} style={{ color: "var(--accent)" }} />
                    Research Papers
                  </h2>
                  <span
                    className="mono-tag"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Citations [P1–P{digest.briefing.papers.length}] · Hugging
                    Face &amp; arXiv
                  </span>
                </div>

                {filteredPapers.length === 0 ? (
                  <p style={{ color: "var(--text-muted)" }}>
                    No matching research papers for this filter.
                  </p>
                ) : (
                  <div className="cards-grid">
                    {filteredPapers.map((paper) => (
                      <article
                        id={`cite-${paper.citation_id}`}
                        key={`${paper.citation_id}-${paper.url}`}
                        className={`item-card ${
                          highlightedCite === paper.citation_id
                            ? "cited-highlight"
                            : ""
                        }`}
                      >
                        <div>
                          <div className="card-top-meta">
                            <span className="card-cite-anchor">
                              [{paper.citation_id}]
                            </span>
                            <span>
                              {paper.source} ·{" "}
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
                                  style={{
                                    marginRight: 4,
                                    verticalAlign: "-1px",
                                  }}
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

            {/* Section 2: Models & Spaces */}
            {(activeTab === "all" || activeTab === "models") && (
              <section id="section-models" className="digest-section">
                <div className="section-header">
                  <h2 className="section-title">
                    <Cpu size={19} style={{ color: "var(--accent)" }} />
                    Models &amp; Interactive Spaces
                  </h2>
                  <span
                    className="mono-tag"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Citations [M1–M{digest.briefing.models.length}] · Hugging
                    Face Hub
                  </span>
                </div>

                {filteredModels.length === 0 ? (
                  <p style={{ color: "var(--text-muted)" }}>
                    No matching models or Spaces for this filter.
                  </p>
                ) : (
                  <div className="cards-grid">
                    {filteredModels.map((model) => (
                      <article
                        id={`cite-${model.citation_id}`}
                        key={`${model.citation_id}-${model.url}`}
                        className={`item-card ${
                          highlightedCite === model.citation_id
                            ? "cited-highlight"
                            : ""
                        }`}
                      >
                        <div>
                          <div className="card-top-meta">
                            <span className="card-cite-anchor">
                              [{model.citation_id}]
                            </span>
                            <span>
                              {model.kind} · {model.pipeline_tag}
                            </span>
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
                                style={{
                                  marginRight: 4,
                                  verticalAlign: "-1px",
                                }}
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

            {/* Section 3: AI News & Discussions */}
            {(activeTab === "all" || activeTab === "news") && (
              <section id="section-news" className="digest-section">
                <div className="section-header">
                  <h2 className="section-title">
                    <Newspaper size={19} style={{ color: "var(--accent)" }} />
                    Community News &amp; Discussions
                  </h2>
                  <span
                    className="mono-tag"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Citations [N1–N{digest.briefing.news.length}] · Hacker News
                    Algolia
                  </span>
                </div>

                {filteredNews.length === 0 ? (
                  <p style={{ color: "var(--text-muted)" }}>
                    No matching news stories found for this date/filter.
                  </p>
                ) : (
                  <div className="cards-grid">
                    {filteredNews.map((story) => (
                      <article
                        id={`cite-${story.citation_id}`}
                        key={`${story.citation_id}-${story.discussion_url}`}
                        className={`item-card ${
                          highlightedCite === story.citation_id
                            ? "cited-highlight"
                            : ""
                        }`}
                      >
                        <div>
                          <div className="card-top-meta">
                            <span className="card-cite-anchor">
                              [{story.citation_id}]
                            </span>
                            <span>
                              {story.source} · {story.points} pts
                            </span>
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

            {/* Section 4: Open-Source AI Repositories */}
            {(activeTab === "all" || activeTab === "repos") && (
              <section id="section-repos" className="digest-section">
                <div className="section-header">
                  <h2 className="section-title">
                    <GitFork size={19} style={{ color: "var(--accent)" }} />
                    Breakout Open-Source Repositories
                  </h2>
                  <span
                    className="mono-tag"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Citations [R1–R{digest.briefing.repos.length}] · GitHub
                    Search API
                  </span>
                </div>

                {filteredRepos.length === 0 ? (
                  <p style={{ color: "var(--text-muted)" }}>
                    No matching GitHub repositories found for this filter.
                  </p>
                ) : (
                  <div className="cards-grid">
                    {filteredRepos.map((repo) => (
                      <article
                        id={`cite-${repo.citation_id}`}
                        key={`${repo.citation_id}-${repo.url}`}
                        className={`item-card ${
                          highlightedCite === repo.citation_id
                            ? "cited-highlight"
                            : ""
                        }`}
                      >
                        <div>
                          <div className="card-top-meta">
                            <span className="card-cite-anchor">
                              [{repo.citation_id}]
                            </span>
                            <span>
                              {repo.language} ·{" "}
                              <Star
                                size={12}
                                style={{
                                  marginRight: 3,
                                  verticalAlign: "-1px",
                                }}
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

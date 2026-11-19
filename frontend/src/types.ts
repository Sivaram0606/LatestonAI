export interface PaperBrief {
  citation_id?: string;
  title: string;
  authors: string;
  source: string;
  url: string;
  published_date: string;
  upvotes: number;
  summary: string;
  tags: string[];
}

export interface ModelBrief {
  citation_id?: string;
  name: string;
  kind: string;
  author: string;
  pipeline_tag: string;
  url: string;
  likes: number;
  downloads: number;
  last_modified?: string;
  summary: string;
}

export interface NewsBrief {
  citation_id?: string;
  title: string;
  source: string;
  url: string;
  discussion_url: string;
  published_date?: string;
  points: number;
  comments: number;
  summary: string;
}

export interface RepoBrief {
  citation_id?: string;
  full_name: string;
  language: string;
  url: string;
  created_date?: string;
  stars: number;
  forks: number;
  summary: string;
  topics: string[];
}

export interface DailyBriefing {
  weekly_window_heading?: string;
  headline: string;
  executive_summary: string;
  whats_new_today?: string[];
  key_themes: string[];
  papers: PaperBrief[];
  models: ModelBrief[];
  news: NewsBrief[];
  repos: RepoBrief[];
}

export interface SelfEvalVerdict {
  passed: boolean;
  grounding_score: number;
  verified_url_count: number;
  flagged_url_count: number;
  verified_citation_count?: number;
  findings: string[];
}

export interface AgentStepTrace {
  agent_name: string;
  role: string;
  status: string;
  duration_ms: number;
  items_collected: number;
  detail: string;
}

export interface DailyBriefingResponse {
  target_date: string;
  week_start_date?: string;
  week_end_date?: string;
  generated_at: string;
  generation_mode: string;
  model_used: string;
  notice: string;
  briefing: DailyBriefing;
  agent_trace: AgentStepTrace[];
  self_eval?: SelfEvalVerdict | null;
  trace_id?: string;
}

export interface HealthStatus {
  status: string;
  model: string;
  auth_mode: string;
  google_cloud_project: string;
  google_cloud_location: string;
  has_gemini_api_key: boolean;
  server_utc_date: string;
}

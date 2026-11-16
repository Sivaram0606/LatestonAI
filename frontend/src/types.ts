export interface PaperBrief {
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
  name: string;
  kind: string;
  author: string;
  pipeline_tag: string;
  url: string;
  likes: number;
  downloads: number;
  summary: string;
}

export interface NewsBrief {
  title: string;
  source: string;
  url: string;
  discussion_url: string;
  points: number;
  comments: number;
  summary: string;
}

export interface RepoBrief {
  full_name: string;
  language: string;
  url: string;
  stars: number;
  forks: number;
  summary: string;
  topics: string[];
}

export interface DailyBriefing {
  headline: string;
  executive_summary: string;
  key_themes: string[];
  papers: PaperBrief[];
  models: ModelBrief[];
  news: NewsBrief[];
  repos: RepoBrief[];
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
  generated_at: string;
  generation_mode: string;
  model_used: string;
  notice: string;
  briefing: DailyBriefing;
  agent_trace: AgentStepTrace[];
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

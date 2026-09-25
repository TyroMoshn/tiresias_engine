export interface AuthUser {
  user_id: string;
  username: string;
  display_name: string;
  role: number;
  role_name: string;
  site_source: 'e621' | 'direct' | 'sandbox' | string;
  site_user_id?: number | null;
  has_password: boolean;
  owner_id?: string | null;
  sandbox_profiles?: Array<{ user_id: string; username: string; display_name: string }>;
}

export interface AuthResponse {
  success: boolean;
  user_id: string;
  username: string;
  display_name: string;
  role: number;
  role_name: string;
  site_source: string;
  site_user_id?: number | null;
  has_password: boolean;
  token?: string | null;
  owner_id?: string | null;
  sandbox_profiles?: Array<{ user_id: string; username: string; display_name: string }>;
  message?: string;
}

import { Language } from './i18n/types';

export type { Language };

export interface ServerConfig {
  serverUrl: string;
  selectedTarget: 'local' | 'laptop' | 'vps' | 'custom';
  customUrl: string;
  userId: string;
  authToken?: string | null;
  authUser?: AuthUser | null;
  adminKey?: string;
  useSessionUser: boolean;
  detectedSessionUser: string;
  autoSeen: boolean;
  themeEmoji: string;
  testerMode?: boolean;
  testProfiles?: string[];
  enableTelemetry?: boolean;
  detectedSiteUser?: { id: number; name: string } | null;
  language?: Language;
}


export interface PostMetadata {
  id: number;
  score: number;
  score_val?: number;
  fav_count: number;
  comment_count?: number;
  rating: 's' | 'q' | 'e' | string;
  file_ext?: string;
  is_video?: boolean;
  preview_url?: string;
  sample_url?: string;
  file_url?: string;
  tags?: string[];
  artist_tags?: string[];
  character_tags?: string[];
  copyright_tags?: string[];
  general_tags?: string[];
  meta_tags?: string[];
  similarity?: number;
  reason?: string;
  affinity?: number;
  added_at?: number;
}

export interface FeedRequest {
  user_id?: string;
  limit?: number;
  cursor?: number;
  min_score?: number;
  ratings?: string[];
  media_types?: string[];
  exploration_weight?: number;
}

export interface FeedResponse {
  items: PostMetadata[];
  total?: number;
  has_more?: boolean;
  next_cursor?: number;
}

export interface SimilarResponse {
  source_post_id: number;
  items: PostMetadata[];
}

export interface Board {
  id: string;
  user_id: string;
  title: string;
  description?: string;
  cover_post_id?: number;
  cover_url?: string;
  post_count: number;
  created_at: string;
  updated_at: string;
  tags?: string[];
  is_public?: boolean;
}

export interface BoardDetail {
  board: Board;
  posts: PostMetadata[];
  salient_tags: Array<{ tag: string; count?: number; post_count?: number; score?: number }>;
  total: number;
}

export interface BoardCreateRequest {
  user_id: string;
  title: string;
  description?: string;
  cover_post_id?: number;
  is_public?: boolean;
}

export interface BoardUpdateRequest {
  name?: string;
  title?: string;
  description?: string;
  cover_post_id?: number;
  is_public?: boolean;
}

export interface BoardPostsAddRequest {
  post_ids: number[];
}

export interface FeedbackRequest {
  user_id: string;
  post_id: number;
  action: 'like' | 'dislike' | 'hide' | 'skip' | 'save' | 'undo_like' | 'undo_hide';
  context?: string;
}

export interface SeenBatchRequest {
  user_id: string;
  post_ids: number[];
}

export interface HealthResponse {
  status: string;
  version?: string;
  device?: string;
  vectors_count?: number;
  uptime_seconds?: number;
  database?: string;
  faiss_index?: string;
  diagnostics_status?: string;
  sanity_warnings?: string[];
  telemetry_errors_count?: number;
  maintenance?: boolean;
}

export interface ClientErrorEntry {
  id: number;
  timestamp: string;
  user_id: string;
  error_type: string;
  message: string;
  stack?: string;
  url?: string;
  source_file?: string;
  lineno?: number;
  colno?: number;
  metadata?: Record<string, any>;
}

export interface ClientErrorsResponse {
  total_errors: number;
  errors: ClientErrorEntry[];
}

export interface DiagnosticsReport {
  status: string;
  audit_timestamp: number;
  audit_duration_ms: number;
  total_warnings: number;
  warnings: string[];
  subsystems: Record<string, { status: string; warnings: string[] }>;
}

export interface ArchetypeOverrideRequest {
  user_id: string;
  locked_archetype: number | null;
  forced_weight: number | null;
}

export interface ArchetypeOverrideResponse {
  success: boolean;
  user_id: string;
  taste_archetype_id: number;
  locked_archetype: number | null;
  forced_weight: number | null;
  effective_archetype_id: number;
  taste_coherence: number;
  archetype_influence: number;
  message: string;
}


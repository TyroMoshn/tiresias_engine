import { loadConfig, saveConfig, getActiveUserId } from './storage';
import {
  AuthResponse,
  AuthUser,
  Board,
  BoardCreateRequest,
  BoardDetail,
  FeedRequest,
  FeedResponse,
  FeedbackRequest,
  HealthResponse,
  PostMetadata,
  SeenBatchRequest,
  SimilarResponse,
  ClientErrorsResponse,
  DiagnosticsReport,
  ArchetypeOverrideResponse,
} from './types';

async function getBaseUrl(): Promise<string> {
  const cfg = await loadConfig();
  return cfg.serverUrl || 'http://127.0.0.1:8000';
}

async function getUserId(): Promise<string> {
  const cfg = await loadConfig();
  return getActiveUserId(cfg);
}

/**
 * Universal request wrapper.
 * When called inside a Content Script on https://e926.net, routes requests through
 * the extension background worker via runtime.sendMessage to bypass Mixed Content
 * (HTTPS -> HTTP) and the host page's Content Security Policy (CSP).
 */
async function requestJson<T = any>(
  endpoint: string,
  options: {
    method?: string;
    headers?: Record<string, string>;
    body?: any;
    params?: Record<string, string | number | undefined>;
  } = {}
): Promise<T> {
  const cfg = await loadConfig();
  const base = cfg.serverUrl || 'http://127.0.0.1:8000';
  let fullUrl = `${base.replace(/\/+$/, '')}/${endpoint.replace(/^\/+/, '')}`;
  if (options.params) {
    const urlObj = new URL(fullUrl);
    for (const [k, v] of Object.entries(options.params)) {
      if (v !== undefined) {
        urlObj.searchParams.set(k, String(v));
      }
    }
    fullUrl = urlObj.toString();
  }

  const method = options.method || 'GET';
  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(options.headers || {}),
  };

  // Automatically attach authorization token if user is authenticated
  if (cfg.authToken && !headers['Authorization']) {
    headers['Authorization'] = `Bearer ${cfg.authToken}`;
  }
  // Automatically attach master admin key if set
  if (cfg.adminKey && !headers['X-Admin-Key']) {
    headers['X-Admin-Key'] = cfg.adminKey;
  }

  let body = options.body;
  if (body !== undefined && typeof body !== 'string') {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(body);
  }

  const isContentScript =
    typeof window !== 'undefined' &&
    typeof document !== 'undefined' &&
    !window.location.protocol.startsWith('moz-extension');

  const runtime =
    typeof browser !== 'undefined' && browser.runtime
      ? browser.runtime
      : typeof chrome !== 'undefined'
      ? chrome.runtime
      : null;

  if (isContentScript && runtime?.sendMessage) {
    const response: any = await new Promise((resolve) => {
      try {
        runtime.sendMessage(
          {
            type: 'TIRESIAS_FETCH',
            url: fullUrl,
            options: { method, headers, body },
          },
          (res: any) => resolve(res)
        );
      } catch (err) {
        resolve({ ok: false, error: String(err) });
      }
    });

    if (!response) {
      throw new Error('Фоновый воркер расширения не ответил на сетевой запрос.');
    }
    if (!response.ok) {
      let detail = response.data?.detail || response.error || `HTTP ${response.status}`;
      if (typeof detail === 'object') {
        if (Array.isArray(detail)) {
          detail = detail.map((d: any) => d.msg || d.message || JSON.stringify(d)).join(', ');
        } else {
          detail = JSON.stringify(detail);
        }
      }
      const err: any = new Error(String(detail));
      err.status = response.status;
      if (response.status === 503) {
        err.isMaintenance = true;
      }
      throw err;
    }
    return response.data as T;
  }

  // Direct fetch (popup, background context)
  const resp = await fetch(fullUrl, { method, headers, body });
  if (!resp.ok) {
    let errData: any = {};
    try {
      errData = await resp.json();
    } catch {}
    let detail = errData?.detail || `HTTP ${resp.status}`;
    if (typeof detail === 'object') {
      if (Array.isArray(detail)) {
        detail = detail.map((d: any) => d.msg || d.message || JSON.stringify(d)).join(', ');
      } else {
        detail = JSON.stringify(detail);
      }
    }
    const err: any = new Error(String(detail));
    err.status = resp.status;
    if (resp.status === 503) {
      err.isMaintenance = true;
    }
    throw err;
  }
  const ct = resp.headers.get('content-type') || '';
  if (ct.includes('application/json')) {
    return (await resp.json()) as T;
  }
  return (await resp.text()) as unknown as T;
}

export const TiresiasApi = {
  async checkHealth(): Promise<{ ok: boolean; isMaintenance?: boolean; data?: HealthResponse; error?: string; latencyMs: number }> {
    const start = performance.now();
    try {
      const data = await requestJson<HealthResponse>('/api/v1/system/health', { method: 'GET' });
      const latencyMs = Math.round(performance.now() - start);
      const isMaint = data.status === 'maintenance' || !!data.maintenance;
      return { ok: true, isMaintenance: isMaint, data, latencyMs };
    } catch (e: any) {
      const latencyMs = Math.round(performance.now() - start);
      if (e?.isMaintenance) {
        return { ok: true, isMaintenance: true, error: 'Maintenance mode', latencyMs };
      }
      return { ok: false, error: e?.message || 'Connection refused', latencyMs };
    }
  },

  async getSuppressionStatus(): Promise<any> {
    return await requestJson('/api/v1/system/suppression', { method: 'GET' });
  },

  async reloadSuppression(): Promise<any> {
    return await requestJson('/api/v1/system/suppression/reload', { method: 'POST' });
  },

  async enableMaintenance(): Promise<any> {
    return await requestJson('/api/v1/system/maintenance/enable', { method: 'POST' });
  },

  async disableMaintenance(): Promise<any> {
    return await requestJson('/api/v1/system/maintenance/disable', { method: 'POST' });
  },

  async getUserProfile(userId?: string): Promise<any> {
    const uid = userId || (await getUserId());
    return await requestJson(`/api/v1/user/${encodeURIComponent(uid)}/profile`, { method: 'GET' });
  },

  async getFeedbackHistory(
    signalType?: 'like' | 'dislike' | 'hide',
    limit = 50,
    offset = 0,
    userId?: string
  ): Promise<{ items: any[]; total: number; limit: number; offset: number; user_id: string; signal_type: string | null }> {
    const uid = userId || (await getUserId());
    return await requestJson(`/api/v1/user/${encodeURIComponent(uid)}/feedback`, {
      method: 'GET',
      params: {
        signal_type: signalType,
        limit,
        offset,
      },
    });
  },

  async overrideArchetype(req: { userId?: string; lockedArchetype: number | null; forcedWeight: number | null }): Promise<ArchetypeOverrideResponse> {
    const uid = req.userId || (await getUserId());
    return await requestJson<ArchetypeOverrideResponse>('/api/v1/user/archetype/override', {
      method: 'POST',
      body: {
        user_id: uid,
        locked_archetype: req.lockedArchetype,
        forced_weight: req.forcedWeight,
      },
    });
  },

  async resetUserHistory(options?: string | { userId?: string; userIds?: string[]; resetAll?: boolean }): Promise<any> {
    let body: any = {};
    if (typeof options === 'string') {
      body = { user_id: options };
    } else if (options) {
      if (options.resetAll) {
        body = { reset_all: true };
      } else if (options.userIds) {
        body = { user_ids: options.userIds };
      } else if (options.userId) {
        body = { user_id: options.userId };
      }
    } else {
      const uid = await getUserId();
      body = { user_id: uid };
    }
    return await requestJson('/api/v1/user/reset', {
      method: 'POST',
      body,
    });
  },

  async getFeed(req: FeedRequest = {}): Promise<FeedResponse> {
    const userId = req.user_id || (await getUserId());
    const payload: FeedRequest = {
      user_id: userId,
      limit: req.limit ?? 30,
      cursor: req.cursor ?? 0,
      min_score: req.min_score ?? 0,
      ratings: req.ratings ?? ['s'],
      media_types: req.media_types ?? ['image', 'video'],
      exploration_weight: req.exploration_weight ?? 0.2,
    };
    return await requestJson<FeedResponse>('/api/v1/recommend/feed', {
      method: 'POST',
      body: payload,
    });
  },

  async getSimilar(postId: number, limit = 12, ratings?: string[], mediaTypes?: string[]): Promise<SimilarResponse> {
    const payload = {
      post_id: postId,
      limit,
      ratings: ratings ?? ['s'],
      media_types: mediaTypes ?? ['image', 'video'],
    };
    return await requestJson<SimilarResponse>('/api/v1/recommend/similar', {
      method: 'POST',
      body: payload,
    });
  },

  async sendFeedback(postId: number, action: 'like' | 'dislike' | 'hide' | 'skip' | 'save' | 'undo_like' | 'undo_hide', context?: string): Promise<void> {
    const userId = await getUserId();
    const payload = {
      user_id: userId,
      post_id: postId,
      signal_type: action,
      action,
      context,
    };
    try {
      await requestJson('/api/v1/feedback', { method: 'POST', body: payload });
    } catch (err) {
      console.warn('[Tiresias] Feedback failed:', err);
    }
  },

  async undoFeedback(postId: number, action: 'like' | 'hide'): Promise<void> {
    await this.sendFeedback(postId, action === 'like' ? 'undo_like' : 'undo_hide');
  },

  async sendSeenBatch(postIds: number[]): Promise<void> {
    if (!postIds || postIds.length === 0) return;
    const userId = await getUserId();
    const payload: SeenBatchRequest = {
      user_id: userId,
      post_ids: postIds,
    };
    try {
      await requestJson('/api/v1/feedback/seen', { method: 'POST', body: payload });
    } catch (err) {
      console.warn('[Tiresias] Seen batch failed:', err);
    }
  },

  async listBoards(): Promise<Board[]> {
    const userId = await getUserId();
    try {
      const boards = await requestJson<Board[]>('/api/v1/boards', {
        method: 'GET',
        params: { user_id: userId },
      });
      try {
        localStorage.setItem(`tiresias_cache_boards_${userId}`, JSON.stringify(boards));
      } catch {}
      return boards;
    } catch (err) {
      try {
        const cached = localStorage.getItem(`tiresias_cache_boards_${userId}`);
        if (cached) {
          console.info('[Tiresias] Loaded boards from offline cache');
          return JSON.parse(cached);
        }
      } catch {}
      throw err;
    }
  },

  async getBoardDetail(
    boardId: string,
    sortBy: 'added_at' | 'score' | 'fav_count' | 'affinity' = 'added_at',
    order: 'asc' | 'desc' = 'desc',
    limit = 50,
    offset = 0
  ): Promise<BoardDetail> {
    try {
      const detail = await requestJson<BoardDetail>(`/api/v1/boards/${encodeURIComponent(boardId)}`, {
        method: 'GET',
        params: { sort_by: sortBy, order, limit, offset },
      });
      try {
        localStorage.setItem(`tiresias_cache_board_${boardId}`, JSON.stringify(detail));
      } catch {}
      return detail;
    } catch (err) {
      try {
        const cached = localStorage.getItem(`tiresias_cache_board_${boardId}`);
        if (cached) {
          console.info(`[Tiresias] Loaded board ${boardId} from offline cache`);
          return JSON.parse(cached);
        }
      } catch {}
      throw err;
    }
  },

  async createBoard(title: string, description?: string, coverPostId?: number, isPublic: boolean = false): Promise<Board> {
    const userId = await getUserId();
    const payload: any = {
      user_id: userId,
      name: title,
      title: title,
      description,
      cover_post_id: coverPostId,
      is_public: isPublic,
    };
    return await requestJson<Board>('/api/v1/boards', {
      method: 'POST',
      body: payload,
    });
  },

  async updateBoard(boardId: string, updates: { name?: string; description?: string; cover_post_id?: number; is_public?: boolean }): Promise<Board> {
    return await requestJson<Board>(`/api/v1/boards/${encodeURIComponent(boardId)}`, {
      method: 'PATCH',
      body: updates,
    });
  },

  async deleteBoard(boardId: string): Promise<void> {
    await requestJson(`/api/v1/boards/${encodeURIComponent(boardId)}`, {
      method: 'DELETE',
    });
  },

  async addPostsToBoard(boardId: string, postIds: number[]): Promise<void> {
    await requestJson(`/api/v1/boards/${encodeURIComponent(boardId)}/posts`, {
      method: 'POST',
      body: { post_ids: postIds },
    });
  },

  async removePostFromBoard(boardId: string, postId: number): Promise<void> {
    await requestJson(`/api/v1/boards/${encodeURIComponent(boardId)}/posts/${postId}`, {
      method: 'DELETE',
    });
  },

  async getBoardRecommendations(boardId: string, limit = 30, ratings?: string[], minScore = 0): Promise<FeedResponse> {
    const userId = await getUserId();
    const payload = {
      user_id: userId,
      limit,
      ratings: ratings && ratings.length > 0 ? ratings : undefined,
      min_score: minScore,
    };
    return await requestJson<FeedResponse>(`/api/v1/boards/${encodeURIComponent(boardId)}/recommend`, {
      method: 'POST',
      body: payload,
    });
  },

  async reportClientError(report: {
    user_id?: string;
    error_type?: string;
    message: string;
    stack?: string;
    url?: string;
    source_file?: string;
    lineno?: number;
    colno?: number;
    metadata?: Record<string, any>;
  }): Promise<void> {
    try {
      const userId = await getUserId();
      await requestJson('/api/v1/system/client-errors', {
        method: 'POST',
        body: {
          user_id: report.user_id || userId,
          error_type: report.error_type || 'js_error',
          message: report.message,
          stack: report.stack,
          url: report.url || (typeof window !== 'undefined' ? window.location.href : undefined),
          source_file: report.source_file,
          lineno: report.lineno,
          colno: report.colno,
          metadata: report.metadata,
        },
      });
    } catch {
      // Avoid recursive failure
    }
  },

  async getClientErrors(limit = 50): Promise<ClientErrorsResponse> {
    return await requestJson<ClientErrorsResponse>('/api/v1/system/client-errors', {
      params: { limit },
    });
  },

  async clearClientErrors(): Promise<void> {
    await requestJson('/api/v1/system/client-errors', {
      method: 'DELETE',
    });
  },

  async getDiagnostics(): Promise<DiagnosticsReport> {
    return await requestJson<DiagnosticsReport>('/api/v1/system/diagnostics');
  },

  async runDiagnostics(): Promise<DiagnosticsReport> {
    return await requestJson<DiagnosticsReport>('/api/v1/system/diagnostics/run', {
      method: 'POST',
    });
  },

  // ---------------------------------------------------------------------------
  // Authentication & Account Management
  // ---------------------------------------------------------------------------

  async sessionHandshake(siteUserId: number, username: string, deviceInfo = ''): Promise<AuthResponse> {
    const res = await requestJson<AuthResponse>('/api/v1/auth/session-handshake', {
      method: 'POST',
      body: {
        site: 'e621',
        site_user_id: siteUserId,
        username,
        device_info: deviceInfo,
      },
    });
    if (res.token) {
      await saveConfig({
        authToken: res.token,
        authUser: res,
        userId: res.user_id,
        useSessionUser: true,
        detectedSessionUser: res.username,
      });
    }
    return res;
  },

  async login(username: string, password: string, deviceInfo = ''): Promise<AuthResponse> {
    const res = await requestJson<AuthResponse>('/api/v1/auth/login', {
      method: 'POST',
      body: {
        username,
        password,
        device_info: deviceInfo,
      },
    });
    if (res.token) {
      await saveConfig({
        authToken: res.token,
        authUser: res,
        userId: res.user_id,
        useSessionUser: false,
      });
    }
    return res;
  },

  async register(username: string, password: string, displayName = '', deviceInfo = ''): Promise<AuthResponse> {
    const res = await requestJson<AuthResponse>('/api/v1/auth/register', {
      method: 'POST',
      body: {
        username,
        password,
        display_name: displayName,
        device_info: deviceInfo,
      },
    });
    if (res.token) {
      await saveConfig({
        authToken: res.token,
        authUser: res,
        userId: res.user_id,
        useSessionUser: false,
      });
    }
    return res;
  },

  async logout(): Promise<void> {
    try {
      await requestJson('/api/v1/auth/logout', { method: 'POST' });
    } catch {}
    await saveConfig({
      authToken: null,
      authUser: null,
      userId: 'default_user',
    });
  },

  async getMe(): Promise<AuthResponse> {
    const res = await requestJson<AuthResponse>('/api/v1/auth/me');
    await saveConfig({ authUser: res, userId: res.user_id });
    return res;
  },

  async setPassword(newPassword: string, newUsername?: string): Promise<void> {
    await requestJson('/api/v1/auth/password', {
      method: 'POST',
      body: { new_password: newPassword, new_username: newUsername || undefined },
    });
    const me = await this.getMe();
    await saveConfig({ authUser: me });
  },

  async linkSession(siteUserId: number, siteUsername: string): Promise<AuthResponse> {
    const res = await requestJson<AuthResponse>('/api/v1/auth/link-session', {
      method: 'POST',
      body: { site_user_id: siteUserId, site_username: siteUsername },
    });
    await saveConfig({ authUser: res });
    return res;
  },

  async createSandboxProfile(profileName: string): Promise<any> {
    const res = await requestJson('/api/v1/auth/sandbox/create', {
      method: 'POST',
      body: { profile_name: profileName },
    });
    await this.getMe();
    return res;
  },

  async deleteSandboxProfile(profileId: string): Promise<any> {
    const res = await requestJson(`/api/v1/auth/sandbox/${encodeURIComponent(profileId)}`, {
      method: 'DELETE',
    });
    await this.getMe();
    return res;
  },
};

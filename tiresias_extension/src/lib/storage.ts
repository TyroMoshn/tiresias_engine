import { ServerConfig } from './types';

const DEFAULT_CONFIG: ServerConfig = {
  serverUrl: 'http://127.0.0.1:8000',
  selectedTarget: 'local',
  customUrl: 'http://127.0.0.1:8000',
  userId: 'default_user',
  authToken: null,
  authUser: null,
  adminKey: '',
  useSessionUser: true,
  detectedSessionUser: '',
  autoSeen: true,
  themeEmoji: '🔮',
  testerMode: false,
  testProfiles: ['user_demo', 'tester_demo', 'admin_demo'],
  language: 'ru',
};


export function getActiveUserId(cfg: ServerConfig): string {
  if (cfg.authUser) {
    if (cfg.userId === cfg.authUser.user_id) {
      return cfg.authUser.user_id;
    }
    const isSandbox = cfg.authUser.sandbox_profiles?.some(
      (sp) => sp.user_id === cfg.userId
    );
    if (isSandbox && cfg.userId) {
      return cfg.userId;
    }
    // Also allow any sandbox/test profile if user is tester or admin
    if (cfg.authUser.role >= 20 && cfg.userId && (cfg.userId.startsWith('test:') || cfg.testProfiles?.includes(cfg.userId))) {
      return cfg.userId;
    }
    return cfg.authUser.user_id;
  }

  if (cfg.useSessionUser && cfg.detectedSessionUser) {
    return cfg.detectedSessionUser;
  }
  return cfg.userId || 'default_user';
}

const STORAGE_KEY = 'tiresias_settings';

function getStorageArea() {
  if (typeof browser !== 'undefined' && browser.storage?.local) {
    return browser.storage.local;
  }
  if (typeof chrome !== 'undefined' && chrome.storage?.local) {
    return chrome.storage.local;
  }
  return null;
}

export async function loadConfig(): Promise<ServerConfig> {
  const storage = getStorageArea();
  if (storage) {
    try {
      const res = await storage.get(STORAGE_KEY);
      if (res && res[STORAGE_KEY]) {
        return { ...DEFAULT_CONFIG, ...res[STORAGE_KEY] };
      }
    } catch (e) {
      console.warn('[Tiresias] Failed to read from extension storage:', e);
    }
  }

  // Fallback to localStorage (e.g. during local dev or mock)
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      return { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
    }
  } catch {}

  return { ...DEFAULT_CONFIG };
}

export async function saveConfig(patch: Partial<ServerConfig>): Promise<ServerConfig> {
  const current = await loadConfig();
  const updated: ServerConfig = { ...current, ...patch };

  // Sync serverUrl based on selectedTarget
  if (updated.selectedTarget === 'local') {
    updated.serverUrl = 'http://127.0.0.1:8000';
  } else if (updated.selectedTarget === 'vps' || updated.selectedTarget === 'custom') {
    if (updated.customUrl) {
      updated.serverUrl = updated.customUrl.replace(/\/+$/, '');
    }
  }

  const storage = getStorageArea();
  if (storage) {
    try {
      await storage.set({ [STORAGE_KEY]: updated });
    } catch (e) {
      console.warn('[Tiresias] Failed to write to extension storage:', e);
    }
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  } catch {}

  return updated;
}

import { h, Fragment } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { TiresiasApi } from '../../lib/api';
import { loadConfig, saveConfig, getActiveUserId } from '../../lib/storage';
import { AuthUser, HealthResponse, ServerConfig } from '../../lib/types';
import {
  Button,
  Input,
  SegmentedControl,
  Badge,
  ToastContainer,
  toast,
} from '../../components/ui';
import { useTranslation } from '../../lib/i18n';

// DEMO ACCOUNTS FLAG: Toggle to false before production release to completely hide demo accounts
const SHOW_DEMO_ACCOUNTS = true;

export function App() {
  const { t, language, setLanguage } = useTranslation();
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; data?: HealthResponse; latencyMs: number; error?: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    loadConfig().then(async (cfg) => {
      setConfig(cfg);
      checkConnection();

      // Refresh /auth/me if token exists
      if (cfg.authToken) {
        try {
          const me = await TiresiasApi.getMe();
          const updated = await saveConfig({ authUser: me, userId: me.user_id });
          setConfig(updated);
        } catch {}
      }
    });
  }, []);

  const checkConnection = async () => {
    const res = await TiresiasApi.checkHealth();
    setHealth(res);
  };

  const openPage = async (path: string) => {
    let host = 'e926.net';
    try {
      if (typeof chrome !== 'undefined' && chrome.tabs) {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab?.url && tab.url.includes('e621.net')) {
          host = 'e621.net';
        }
      }
    } catch {}
    const url = `https://${host}${path}`;
    if (typeof browser !== 'undefined' && browser.tabs) {
      browser.tabs.create({ url });
    } else if (typeof chrome !== 'undefined' && chrome.tabs) {
      chrome.tabs.create({ url });
    } else {
      window.open(url, '_blank');
    }
  };

  // Auth Handlers
  const handleSessionLogin = async () => {
    setLoading(true);
    setAuthError(null);
    try {
      let siteId = config?.detectedSiteUser?.id;
      let siteName = config?.detectedSiteUser?.name;

      if (!siteId || !siteName) {
        // Query active tab to detect user from page
        if (typeof chrome !== 'undefined' && chrome.tabs) {
          const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
          if (tab?.id) {
            try {
              const [res] = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: () => {
                  const el = document.getElementById('site-user');
                  if (!el) return null;
                  try {
                    const raw = atob(el.textContent?.trim() || '');
                    const bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0));
                    const json = new TextDecoder('utf-8').decode(bytes);
                    const parsed = JSON.parse(json);
                    return { id: Number(parsed.id), name: String(parsed.name || ''), isAnonymous: Boolean(parsed.is_anonymous) };
                  } catch {
                    return null;
                  }
                },
              });
              if (res?.result && !res.result.isAnonymous && res.result.id > 0) {
                siteId = res.result.id;
                siteName = res.result.name;
              }
            } catch {}
          }
        }
      }

      if (!siteId || !siteName) {
        setAuthError(t('popup.sessionNotFound'));
        toast.error(t('popup.sessionNotFound'));
        setLoading(false);
        return;
      }

      const res = await TiresiasApi.sessionHandshake(siteId, siteName, 'Extension Popup');
      const updated = await saveConfig({
        authToken: res.token,
        authUser: res,
        userId: res.user_id,
        useSessionUser: true,
        detectedSiteUser: { id: siteId, name: siteName },
      });
      setConfig(updated);
      toast.success(t('settings.welcomeUser', { name: res.display_name }));
    } catch (e: any) {
      const msg = e?.message || t('popup.sessionHandshakeError');
      setAuthError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordAuth = async (e: any) => {
    e.preventDefault();
    if (!authUsername.trim() || !authPassword.trim()) {
      setAuthError(t('popup.enterCredentialsError'));
      return;
    }
    setLoading(true);
    setAuthError(null);
    try {
      let res: any;
      if (authMode === 'login') {
        res = await TiresiasApi.login(authUsername.trim(), authPassword.trim(), 'Extension Popup');
      } else {
        res = await TiresiasApi.register(authUsername.trim(), authPassword.trim(), authUsername.trim(), 'Extension Popup');
      }
      const updated = await saveConfig({
        authToken: res.token,
        authUser: res,
        userId: res.user_id,
        useSessionUser: false,
      });
      setConfig(updated);
      setAuthUsername('');
      setAuthPassword('');
      toast.success(t('settings.welcomeUser', { name: res.display_name }));
    } catch (e: any) {
      const msg = e?.message || t('popup.authError');
      setAuthError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickDemoLogin = async (username: string, pass: string) => {
    setLoading(true);
    setAuthError(null);
    try {
      const res = await TiresiasApi.login(username, pass, 'Demo Button');
      const updated = await saveConfig({
        authToken: res.token,
        authUser: res,
        userId: res.user_id,
        useSessionUser: false,
      });
      setConfig(updated);
      toast.success(t('settings.welcomeUser', { name: res.display_name }));
    } catch (e: any) {
      const msg = e?.message || t('popup.demoLoginError', { username });
      setAuthError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    setLoading(true);
    try {
      await TiresiasApi.logout();
      const updated = await loadConfig();
      setConfig(updated);
    } finally {
      setLoading(false);
    }
  };

  if (!config) {
    return <div style={{ padding: 20, width: 320, background: 'var(--tiresias-bg-base)', color: '#fff' }}>{t('common.loading')}</div>;
  }

  const authUser = config.authUser;
  const userRole = authUser ? authUser.role : 0;
  const activeUserId = getActiveUserId(config);
  const isSandboxActive = activeUserId !== authUser?.user_id;

  const roleBadgeText = userRole >= 100 ? t('settings.roleAdmin') : userRole >= 20 ? t('settings.roleTester') : t('settings.roleUser');
  const roleBadgeColor = userRole >= 100 ? '#b388ff' : userRole >= 20 ? 'var(--tiresias-tester-cyan)' : 'var(--tiresias-info)';

  return (
    <div
      style={{
        width: 330,
        boxSizing: 'border-box',
        background: 'var(--tiresias-bg-base)',
        color: '#fff',
        fontFamily: 'Verdana, Helvetica, Arial, sans-serif',
        padding: '14px 16px',
        fontSize: 13,
        position: 'relative',
      }}
    >
      <ToastContainer />

      {/* Header Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--tiresias-border)', paddingBottom: 10, marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 22 }}>🔮</span>
          <div>
            <div style={{ fontWeight: 'bold', fontSize: 15, color: 'var(--tiresias-gold)', letterSpacing: 0.5 }}>{t('popup.title')}</div>
            <div style={{ fontSize: 10, color: 'var(--tiresias-text-muted)' }}>{t('popup.subtitle')}</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <SegmentedControl
            size="sm"
            options={[
              { label: 'RU', value: 'ru' },
              { label: 'EN', value: 'en' },
            ]}
            value={language}
            onChange={(val) => setLanguage(val as any)}
          />
          <span
            className={`tiresias-nav-status ${health?.ok ? 'online' : 'offline'}`}
            style={{ width: 8, height: 8 }}
            title={health?.ok ? t('settings.serverOnline', { latency: health.latencyMs }) : t('settings.serverOffline')}
          />
          <span style={{ fontSize: 10, color: health?.ok ? 'var(--tiresias-success)' : 'var(--tiresias-danger)', fontWeight: 'bold' }}>
            {health?.ok ? t('popup.online') : t('popup.offline')}
          </span>
        </div>
      </div>

      {/* Error alert */}
      {authError && (
        <div style={{ background: 'var(--tiresias-danger-subtle)', border: '1px solid var(--tiresias-danger)', color: 'var(--tiresias-danger-light)', padding: '6px 10px', borderRadius: 4, fontSize: 11, marginBottom: 10 }}>
          {authError}
        </div>
      )}

      {/* =================================================================== */}
      {/* LOGGED IN VIEW                                                      */}
      {/* =================================================================== */}
      {authUser ? (
        <div>
          {/* User Card */}
          <div
            style={{
              background: 'var(--tiresias-bg-header)',
              border: '1px solid var(--tiresias-border)',
              borderRadius: 'var(--tiresias-radius-md)',
              padding: '10px 12px',
              marginBottom: 12,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontWeight: 'bold', fontSize: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>👤</span>
                <span>{authUser.display_name}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                <span
                  style={{
                    fontSize: 9,
                    fontWeight: 'bold',
                    padding: '1px 6px',
                    borderRadius: 8,
                    background: 'rgba(255,255,255,0.08)',
                    color: roleBadgeColor,
                    border: `1px solid ${roleBadgeColor}`,
                  }}
                >
                  {roleBadgeText}
                </span>
                {isSandboxActive && (
                  <span style={{ fontSize: 10, color: 'var(--tiresias-tester-cyan)', fontWeight: 'bold' }}>
                    🧪 {activeUserId.replace(/^test:[^:]+:/, '')}
                  </span>
                )}
              </div>
            </div>

            <Button
              variant="secondary"
              size="sm"
              style={{ padding: '3px 8px', fontSize: 11 }}
              onClick={handleLogout}
              disabled={loading}
            >
              {t('popup.logout')}
            </Button>
          </div>

          {/* Primary Navigation Buttons */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <Button
              variant="primary"
              style={{
                width: '100%',
                justifyContent: 'center',
                gap: 8,
                padding: '10px',
                fontSize: 13,
                fontWeight: 'bold',
              }}
              onClick={() => openPage('/feed')}
            >
              <span style={{ fontSize: 16 }}>🔮</span>
              <span>{t('popup.recommendations')}</span>
            </Button>

            <Button
              variant="secondary"
              style={{
                width: '100%',
                justifyContent: 'center',
                gap: 8,
                padding: '9px',
                fontSize: 13,
              }}
              onClick={() => openPage('/boards')}
            >
              <span style={{ fontSize: 16 }}>📂</span>
              <span>{t('popup.boards')}</span>
            </Button>

            <Button
              variant="secondary"
              style={{
                width: '100%',
                justifyContent: 'center',
                gap: 8,
                padding: '9px',
                fontSize: 13,
              }}
              onClick={() => openPage('/activity')}
            >
              <span style={{ fontSize: 16 }}>📊</span>
              <span>{t('popup.activity')}</span>
            </Button>

            <Button
              variant="secondary"
              style={{
                width: '100%',
                justifyContent: 'center',
                gap: 8,
                padding: '9px',
                borderColor: 'var(--tiresias-gold)',
                color: 'var(--tiresias-gold)',
                fontSize: 13,
                fontWeight: 500,
              }}
              onClick={() => openPage('/tiresias/settings')}
            >
              <span style={{ fontSize: 16 }}>⚙️</span>
              <span>{t('popup.settings')}</span>
            </Button>
          </div>
        </div>
      ) : (
        /* =================================================================== */
        /* LOGGED OUT / AUTH FORM                                              */
        /* =================================================================== */
        <div>
          {/* Primary One-Click Action: Session Login */}
          <Button
            variant="primary"
            style={{
              width: '100%',
              padding: '10px',
              background: 'var(--tiresias-gold)',
              color: 'var(--tiresias-text-inverse)',
              fontWeight: 'bold',
              fontSize: 13,
              justifyContent: 'center',
              marginBottom: 12,
            }}
            onClick={handleSessionLogin}
            disabled={loading}
          >
            <span>🦊</span>
            <span>{t('popup.sessionLogin')}</span>
          </Button>

          <div style={{ textAlign: 'center', margin: '8px 0', fontSize: 11, color: 'var(--tiresias-text-muted)' }}>
            {t('popup.orCredentials')}
          </div>

          {/* Login / Register Form */}
          <form onSubmit={handlePasswordAuth} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <Input
              type="text"
              placeholder={t('popup.loginPlaceholder')}
              value={authUsername}
              onInput={(e: any) => setAuthUsername(e.target.value)}
              size="sm"
            />
            <Input
              type="password"
              placeholder={t('popup.passwordPlaceholder')}
              value={authPassword}
              onInput={(e: any) => setAuthPassword(e.target.value)}
              size="sm"
            />
            <div style={{ display: 'flex', gap: 6, marginTop: 2 }}>
              <Button
                type="submit"
                variant="primary"
                disabled={loading}
                style={{ flex: 1 }}
              >
                {loading ? '...' : authMode === 'login' ? t('popup.loginBtn') : t('popup.registerBtn')}
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={() => {
                  setAuthMode(authMode === 'login' ? 'register' : 'login');
                  setAuthError(null);
                }}
              >
                {authMode === 'login' ? t('popup.switchRegister') : t('popup.switchLogin')}
              </Button>
            </div>
          </form>

          {/* DEMO ACCOUNTS BLOCK (Conditionally rendered, safe to drop) */}
          {SHOW_DEMO_ACCOUNTS && (
            <details style={{ marginTop: 14, borderTop: '1px solid var(--tiresias-border)', paddingTop: 8 }}>
              <summary style={{ fontSize: 11, color: 'var(--tiresias-text-muted)', cursor: 'pointer', outline: 'none' }}>
                {t('popup.demoTitle')}
              </summary>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, marginTop: 8 }}>
                <Button
                  variant="secondary"
                  size="sm"
                  style={{ padding: '5px 2px', fontSize: 10, color: 'var(--tiresias-info)' }}
                  onClick={() => handleQuickDemoLogin('user_demo', 'user123')}
                  disabled={loading}
                >
                  👤 user
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  style={{ padding: '5px 2px', fontSize: 10, color: 'var(--tiresias-tester-cyan)' }}
                  onClick={() => handleQuickDemoLogin('tester_demo', 'tester123')}
                  disabled={loading}
                >
                  🧪 tester
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  style={{ padding: '5px 2px', fontSize: 10, color: '#b388ff' }}
                  onClick={() => handleQuickDemoLogin('admin_demo', 'admin123')}
                  disabled={loading}
                >
                  🛡️ admin
                </Button>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

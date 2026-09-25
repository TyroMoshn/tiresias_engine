import { h, Fragment } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import { TiresiasApi } from '../lib/api';
import { loadConfig, saveConfig, getActiveUserId } from '../lib/storage';
import { AuthUser, ClientErrorEntry, DiagnosticsReport, HealthResponse, ServerConfig } from '../lib/types';
import {
  Button,
  IconButton,
  Input,
  Select,
  FormGroup,
  SegmentedControl,
  Badge,
  toast,
} from './ui';
import { useTranslation } from '../lib/i18n';

export function SettingsPage() {
  const { t, language, setLanguage } = useTranslation();
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; data?: HealthResponse; latencyMs: number; error?: string } | null>(null);
  const [activeTab, setActiveTab] = useState<'profile' | 'tester' | 'admin'>('profile');
  const [loading, setLoading] = useState(true);

  // Profile / Password states
  const [newPassword, setNewPassword] = useState('');
  const [newUsername, setNewUsername] = useState('');
  const [savingPassword, setSavingPassword] = useState(false);
  const [linkingSession, setLinkingSession] = useState(false);
  const [confirmResetText, setConfirmResetText] = useState('');
  const [resettingHistory, setResettingHistory] = useState(false);

  // Tester states
  const [sandboxName, setSandboxName] = useState('');
  const [creatingSandbox, setCreatingSandbox] = useState(false);
  const [profileStats, setProfileStats] = useState<any>(null);
  const [lockedArch, setLockedArch] = useState<number | string>(27);
  const [forcedWeight, setForcedWeight] = useState(100);
  const [savingArch, setSavingArch] = useState(false);
  const [confirmDeleteProfileId, setConfirmDeleteProfileId] = useState<string | null>(null);

  // Admin states
  const [customServerUrl, setCustomServerUrl] = useState('');
  const [testingConnection, setTestingConnection] = useState(false);
  const [togglingMaintenance, setTogglingMaintenance] = useState(false);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsReport | null>(null);
  const [runningDiagnostics, setRunningDiagnostics] = useState(false);
  const [clientErrors, setClientErrors] = useState<ClientErrorEntry[]>([]);
  const [loadingErrors, setLoadingErrors] = useState(false);
  const [suppressionStatus, setSuppressionStatus] = useState<any>(null);
  const [reloadingSuppression, setReloadingSuppression] = useState(false);
  const [adminKeyInput, setAdminKeyInput] = useState('');
  const [confirmWipeAll, setConfirmWipeAll] = useState(false);

  useEffect(() => {
    initPage();
  }, []);

  const initPage = async () => {
    setLoading(true);
    const cfg = await loadConfig();
    setConfig(cfg);
    setCustomServerUrl(cfg.customUrl || '');
    setAdminKeyInput(cfg.adminKey || '');

    // Health check
    const hRes = await TiresiasApi.checkHealth();
    setHealth(hRes);

    // Refresh /auth/me if logged in
    if (cfg.authToken) {
      try {
        const me = await TiresiasApi.getMe();
        const updated = await saveConfig({ authUser: me });
        setConfig(updated);
        await loadUserData(me.user_id, updated);
      } catch {
        await loadUserData(getActiveUserId(cfg), cfg);
      }
    } else {
      await loadUserData(getActiveUserId(cfg), cfg);
    }
    setLoading(false);
  };

  const loadUserData = async (userId: string, cfg: ServerConfig) => {
    const role = cfg.authUser?.role ?? 10;
    if (role >= 20) {
      try {
        const pStats = await TiresiasApi.getUserProfile(userId);
        setProfileStats(pStats);
        if (pStats.locked_archetype !== null && pStats.locked_archetype !== undefined) {
          setLockedArch(pStats.locked_archetype);
        } else if (pStats.taste_archetype_id !== undefined) {
          setLockedArch(pStats.taste_archetype_id);
        }
        if (pStats.forced_archetype_weight !== null && pStats.forced_archetype_weight !== undefined) {
          setForcedWeight(Math.round(pStats.forced_archetype_weight * 100));
        }
      } catch {}
    }
  };

  // ---------------------------------------------------------------------------
  // Profile Handlers
  // ---------------------------------------------------------------------------

  const handleSetPassword = async () => {
    if (!newPassword || newPassword.length < 4) {
      toast.error(t('settings.passwordLengthError'));
      return;
    }
    setSavingPassword(true);
    try {
      await TiresiasApi.setPassword(newPassword, newUsername.trim() || undefined);
      const me = await TiresiasApi.getMe();
      await saveConfig({ authUser: me });
      setConfig((prev) => (prev ? { ...prev, authUser: me } : prev));
      setNewPassword('');
      setNewUsername('');
      toast.success(t('settings.passwordSaveSuccess'));
    } catch (e: any) {
      toast.error(e?.message || t('settings.passwordLengthError'));
    } finally {
      setSavingPassword(false);
    }
  };

  const handleLinkSession = async () => {
    if (!config?.detectedSiteUser) {
      toast.error(t('settings.sessionNotFound'));
      return;
    }
    setLinkingSession(true);
    try {
      const res = await TiresiasApi.linkSession(config.detectedSiteUser.id, config.detectedSiteUser.name);
      await saveConfig({ authUser: res });
      setConfig((prev) => (prev ? { ...prev, authUser: res } : prev));
      toast.success(t('settings.sessionLinkedSuccess', { name: config.detectedSiteUser.name, id: config.detectedSiteUser.id }));
    } catch (e: any) {
      toast.error(e?.message || 'Error linking session');
    } finally {
      setLinkingSession(false);
    }
  };

  const handleSessionLogin = async () => {
    if (!config?.detectedSiteUser) {
      toast.error(t('settings.sessionNotFound'));
      return;
    }
    setLoading(true);
    try {
      const res = await TiresiasApi.sessionHandshake(config.detectedSiteUser.id, config.detectedSiteUser.name, 'SettingsPage');
      await saveConfig({ authUser: res, authToken: res.token, userId: res.user_id, useSessionUser: true });
      await initPage();
      toast.success(t('settings.welcomeUser', { name: res.display_name }));
    } catch (e: any) {
      toast.error(e?.message || t('settings.sessionLoginFailed'));
    } finally {
      setLoading(false);
    }
  };

  const handleResetHistory = async () => {
    if (confirmResetText !== t('settings.confirmResetWord')) {
      toast.error(t('settings.resetConfirmError'));
      return;
    }
    setResettingHistory(true);
    try {
      const activeId = getActiveUserId(config!);
      await TiresiasApi.resetUserHistory({ userId: activeId, resetAll: false });
      setConfirmResetText('');
      toast.success(t('settings.resetSuccess'));
      await loadUserData(activeId, config!);
    } catch (e: any) {
      toast.error(e?.message || 'Error resetting history');
    } finally {
      setResettingHistory(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Tester Sandbox Handlers
  // ---------------------------------------------------------------------------

  const handleSwitchProfile = async (targetUserId: string) => {
    const updated = await saveConfig({ userId: targetUserId });
    setConfig(updated);
    await loadUserData(targetUserId, updated);
  };

  const handleCreateSandbox = async () => {
    const clean = sandboxName.trim();
    if (!clean) return;
    setCreatingSandbox(true);
    try {
      const res = await TiresiasApi.createSandboxProfile(clean);
      const me = await TiresiasApi.getMe();
      const updated = await saveConfig({ authUser: me });
      setConfig(updated);
      setSandboxName('');
      if (res?.profile?.user_id) {
        await handleSwitchProfile(res.profile.user_id);
      }
      toast.success(t('settings.sandboxCreated', { name: clean }));
    } catch (e: any) {
      toast.error(e?.message || 'Error creating sandbox');
    } finally {
      setCreatingSandbox(false);
    }
  };

  const handleDeleteSandbox = async (profileId: string) => {
    setConfirmDeleteProfileId(null);
    try {
      await TiresiasApi.deleteSandboxProfile(profileId);
      const me = await TiresiasApi.getMe();
      let nextId = me.user_id;
      const updated = await saveConfig({ authUser: me, userId: nextId });
      setConfig(updated);
      await loadUserData(nextId, updated);
      toast.info(t('settings.sandboxDeleted'));
    } catch (e: any) {
      toast.error(e?.message || 'Error deleting sandbox');
    }
  };

  const handleSaveArchetypeOverride = async () => {
    setSavingArch(true);
    try {
      const activeId = getActiveUserId(config!);
      const archId = typeof lockedArch === 'number' ? lockedArch : parseInt(String(lockedArch), 10);
      const weight = forcedWeight / 100;
      await TiresiasApi.overrideArchetype({
        userId: activeId,
        lockedArchetype: isNaN(archId) ? null : archId,
        forcedWeight: weight,
      });
      toast.success(t('settings.archetypeOverrideSuccess'));
      await loadUserData(activeId, config!);
    } catch (e: any) {
      toast.error(e?.message || 'Error saving archetype');
    } finally {
      setSavingArch(false);
    }
  };

  const handleUnlockArchetype = async () => {
    setSavingArch(true);
    try {
      const activeId = getActiveUserId(config!);
      await TiresiasApi.overrideArchetype({
        userId: activeId,
        lockedArchetype: null,
        forcedWeight: null,
      });
      toast.success(t('settings.archetypeUnlockSuccess'));
      await loadUserData(activeId, config!);
    } catch (e: any) {
      toast.error(e?.message || 'Error unlocking archetype');
    } finally {
      setSavingArch(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Admin Handlers
  // ---------------------------------------------------------------------------

  const handleSwitchServer = async (target: 'local' | 'laptop' | 'vps' | 'custom', customUrl?: string) => {
    setTestingConnection(true);

    const targetUrl = customUrl !== undefined ? customUrl.trim() : (target === 'local' ? 'http://127.0.0.1:8000' : (config?.customUrl || config?.serverUrl || '').trim());
    if ((customUrl !== undefined || target !== 'local') && targetUrl) {
      try {
        const fullUrl = targetUrl.includes('://') ? targetUrl : `http://${targetUrl}`;
        const origin = new URL(fullUrl).origin + '/*';
        const permApi = typeof browser !== 'undefined' && (browser as any).permissions ? (browser as any).permissions : (typeof chrome !== 'undefined' ? chrome.permissions : null);
        if (permApi?.request) {
          await permApi.request({ origins: [origin] });
        }
      } catch (permErr) {
        console.warn('[Tiresias] Dynamic permission request failed or rejected:', permErr);
      }
    }

    const patch: Partial<ServerConfig> = { selectedTarget: target };
    if (customUrl !== undefined) patch.customUrl = customUrl;
    const updated = await saveConfig(patch);
    setConfig(updated);
    const hRes = await TiresiasApi.checkHealth();
    setHealth(hRes);
    setTestingConnection(false);
    if (hRes.ok) {
      toast.success(t('settings.serverConnected', { url: updated.serverUrl, latency: hRes.latencyMs }));
    } else {
      toast.error(t('settings.serverUnavailable', { error: hRes.error || 'Connection refused' }));
    }
  };

  const handleToggleMaintenance = async () => {
    setTogglingMaintenance(true);
    try {
      const isMaint = Boolean(health?.data?.maintenance);
      if (isMaint) {
        await TiresiasApi.disableMaintenance();
        toast.success(t('settings.maintenanceDisabledMsg'));
      } else {
        await TiresiasApi.enableMaintenance();
        toast.info(t('settings.maintenanceEnabledMsg'));
      }
      const hRes = await TiresiasApi.checkHealth();
      setHealth(hRes);
    } catch (e: any) {
      toast.error(e?.message || 'Error toggling maintenance');
    } finally {
      setTogglingMaintenance(false);
    }
  };

  const handleRunDiagnostics = async () => {
    setRunningDiagnostics(true);
    try {
      const rep = await TiresiasApi.runDiagnostics();
      setDiagnostics(rep);
      toast.success(t('settings.diagnosticsCompleted'));
    } catch (e: any) {
      toast.error(e?.message || 'Error running diagnostics');
    } finally {
      setRunningDiagnostics(false);
    }
  };

  const handleLoadErrors = async () => {
    setLoadingErrors(true);
    try {
      const res = await TiresiasApi.getClientErrors(50);
      setClientErrors(res.errors || []);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to load error log');
    } finally {
      setLoadingErrors(false);
    }
  };

  const handleClearErrors = async () => {
    try {
      await TiresiasApi.clearClientErrors();
      setClientErrors([]);
      toast.success(t('settings.clientErrorsCleared'));
      const hRes = await TiresiasApi.checkHealth();
      setHealth(hRes);
    } catch (e: any) {
      toast.error(e?.message || 'Error clearing errors');
    }
  };

  const handleReloadSuppression = async () => {
    setReloadingSuppression(true);
    try {
      await TiresiasApi.reloadSuppression();
      const st = await TiresiasApi.getSuppressionStatus();
      setSuppressionStatus(st);
      toast.success(t('settings.suppressionReloaded'));
    } catch (e: any) {
      toast.error(e?.message || 'Error reloading suppression');
    } finally {
      setReloadingSuppression(false);
    }
  };

  const handleWipeAll = async () => {
    if (!adminKeyInput) {
      toast.error(t('settings.adminKeyRequired'));
      return;
    }
    if (!confirmWipeAll) {
      setConfirmWipeAll(true);
      return;
    }
    try {
      await saveConfig({ adminKey: adminKeyInput });
      await TiresiasApi.resetUserHistory({ resetAll: true });
      setConfirmWipeAll(false);
      toast.success(t('settings.globalResetSuccess'));
      await initPage();
    } catch (e: any) {
      toast.error(e?.message || 'Error executing global reset');
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  if (loading && !config) {
    return (
      <div className="tiresias-page-wrapper" style={{ padding: 40, textAlign: 'center' }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>🔮</div>
        <div style={{ color: 'var(--tiresias-gold)', fontWeight: 'bold' }}>{t('settings.loadingTitle')}</div>
      </div>
    );
  }

  const authUser = config?.authUser;
  const userRole = authUser ? authUser.role : 0;
  const isTesterOrAdmin = userRole >= 20;
  const isAdmin = userRole >= 100;
  const activeUserId = config ? getActiveUserId(config) : 'default_user';

  // Guest view
  if (!authUser || userRole === 0) {
    return (
      <div className="tiresias-page-wrapper" style={{ maxWidth: 640, margin: '40px auto', padding: 24 }}>
        <div className="tiresias-header-bar">
          <div className="tiresias-header-title">
            <span style={{ fontSize: 24 }}>🔮</span> {t('settings.title')}
          </div>
          <SegmentedControl
            size="sm"
            options={[
              { label: 'RU', value: 'ru' },
              { label: 'EN', value: 'en' },
            ]}
            value={language}
            onChange={(val) => setLanguage(val as any)}
          />
        </div>

        <div style={{ background: 'var(--tiresias-card)', padding: 24, borderRadius: 'var(--tiresias-radius-lg)', marginTop: 16, textAlign: 'center' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>🔒</div>
          <h3 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-gold)' }}>{t('settings.guestTitle')}</h3>
          <p style={{ color: 'var(--tiresias-text-muted)', fontSize: 13, lineHeight: 1.5, marginBottom: 20 }}>
            {t('settings.guestDesc')}
          </p>

          {config?.detectedSiteUser ? (
            <Button
              variant="primary"
              style={{ fontWeight: 'bold', padding: '10px 20px', fontSize: 14 }}
              onClick={handleSessionLogin}
            >
              {t('settings.loginViaSessionBtn', { name: config.detectedSiteUser.name })}
            </Button>
          ) : (
            <div style={{ color: 'var(--tiresias-text-muted)', fontSize: 12 }}>
              {t('settings.guestHint')}
            </div>
          )}
        </div>
      </div>
    );
  }

  const roleBadgeText = userRole >= 100 ? t('settings.roleAdmin') : userRole >= 20 ? t('settings.roleTester') : t('settings.roleUser');
  const roleBadgeColor = userRole >= 100 ? '#b388ff' : userRole >= 20 ? 'var(--tiresias-tester-cyan)' : 'var(--tiresias-info)';

  return (
    <div className="tiresias-page-wrapper" style={{ maxWidth: 860, margin: '20px auto', padding: '0 16px 40px 16px' }}>
      {/* Header Bar */}
      <div className="tiresias-header-bar" style={{ marginBottom: 16 }}>
        <div className="tiresias-header-title">
          <span style={{ fontSize: 24 }}>⚙️</span> {t('settings.title')}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <SegmentedControl
            size="sm"
            options={[
              { label: 'RU', value: 'ru' },
              { label: 'EN', value: 'en' },
            ]}
            value={language}
            onChange={(val) => setLanguage(val as any)}
          />
          <span style={{ fontSize: 12, color: 'var(--tiresias-text-muted)' }}>{authUser.display_name}</span>
          <span
            style={{
              fontSize: 10,
              fontWeight: 'bold',
              padding: '2px 8px',
              borderRadius: 10,
              background: 'rgba(255,255,255,0.08)',
              color: roleBadgeColor,
              border: `1px solid ${roleBadgeColor}`,
            }}
          >
            {roleBadgeText}
          </span>
          <span
            className={`tiresias-nav-status ${health?.ok ? 'online' : 'offline'}`}
            title={health?.ok ? t('settings.serverOnline', { latency: health.latencyMs }) : t('settings.serverOffline')}
          />
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="tiresias-tabs" style={{ marginBottom: 20 }}>
        <button
          type="button"
          className={`tiresias-tab ${activeTab === 'profile' ? 'active' : ''}`}
          onClick={() => setActiveTab('profile')}
        >
          {t('settings.tabProfile')}
        </button>
        {isTesterOrAdmin && (
          <button
            type="button"
            className={`tiresias-tab ${activeTab === 'tester' ? 'active' : ''}`}
            onClick={() => setActiveTab('tester')}
          >
            {t('settings.tabTester')}
          </button>
        )}
        {isAdmin && (
          <button
            type="button"
            className={`tiresias-tab ${activeTab === 'admin' ? 'active' : ''}`}
            onClick={() => setActiveTab('admin')}
          >
            {t('settings.tabAdmin')}
          </button>
        )}
      </div>

      {/* =================================================================== */}
      {/* TAB 1: ПРОФИЛЬ & ДОСТУП                                              */}
      {/* =================================================================== */}
      {activeTab === 'profile' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Card: Language Selection */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>🌐 {t('common.language')}</h4>
            <div style={{ maxWidth: 300 }}>
              <SegmentedControl
                options={[
                  { label: 'Русский (RU)', value: 'ru' },
                  { label: 'English (EN)', value: 'en' },
                ]}
                value={language}
                onChange={(val) => setLanguage(val as any)}
              />
            </div>
          </div>

          {/* Card: Account Info */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 12px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.cardAccount')}</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, fontSize: 13 }}>
              <div>
                <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('settings.userId')} </span>
                <code style={{ color: 'var(--tiresias-tester-cyan)' }}>{authUser.user_id}</code>
              </div>
              <div>
                <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('settings.username')} </span>
                <b>{authUser.username || '—'}</b>
              </div>
              <div>
                <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('settings.displayName')} </span>
                <b>{authUser.display_name}</b>
              </div>
              <div>
                <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('settings.source')} </span>
                <span>{authUser.site_source === 'e621' ? t('settings.sourceSession') : t('settings.sourceDirect')}</span>
              </div>
            </div>
          </div>

          {/* Card: Bidirectional Session Linking */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.cardLinking')}</h4>
            {authUser.site_source === 'e621' || authUser.site_user_id ? (
              <div style={{ fontSize: 13, color: 'var(--tiresias-success)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>✓</span> {t('settings.linkedSuccess', { id: authUser.site_user_id || 'сессия' })}
              </div>
            ) : (
              <div>
                <p style={{ margin: '0 0 12px 0', fontSize: 13, color: 'var(--tiresias-text-muted)', lineHeight: 1.4 }}>
                  {t('settings.linkingDesc')}
                </p>
                {config?.detectedSiteUser ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 13 }}>
                      {t('settings.detectedUser', { name: config.detectedSiteUser.name, id: config.detectedSiteUser.id })}
                    </span>
                    <Button
                      variant="primary"
                      disabled={linkingSession}
                      loading={linkingSession}
                      onClick={handleLinkSession}
                    >
                      {t('settings.linkBtn')}
                    </Button>
                  </div>
                ) : (
                  <div style={{ fontSize: 12, color: 'var(--tiresias-text-muted)' }}>
                    {t('settings.noSessionDetected')}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Card: Credentials / Password Management */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.cardCredentials')}</h4>
            <p style={{ margin: '0 0 12px 0', fontSize: 13, color: 'var(--tiresias-text-muted)', lineHeight: 1.4 }}>
              {!authUser.has_password
                ? t('settings.credentialsCreateDesc')
                : t('settings.credentialsUpdateDesc')}
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxWidth: 360 }}>
              {!authUser.has_password && (
                <Input
                  type="text"
                  placeholder={t('settings.loginPlaceholder', { name: authUser.display_name })}
                  value={newUsername}
                  onInput={(e: any) => setNewUsername(e.target.value)}
                />
              )}
              <Input
                type="password"
                placeholder={t('settings.passwordPlaceholder')}
                value={newPassword}
                onInput={(e: any) => setNewPassword(e.target.value)}
              />
              <Button
                variant="primary"
                disabled={savingPassword || !newPassword}
                loading={savingPassword}
                onClick={handleSetPassword}
              >
                {!authUser.has_password ? t('settings.saveCredentialsBtn') : t('settings.updatePasswordBtn')}
              </Button>
            </div>
          </div>

          {/* Card: Danger Zone */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid rgba(228, 95, 95, 0.4)' }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-danger)', fontSize: 15 }}>{t('settings.cardDanger')}</h4>
            <p style={{ margin: '0 0 12px 0', fontSize: 13, color: 'var(--tiresias-text-muted)' }}>
              {t('settings.dangerDesc')}
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <Input
                type="text"
                placeholder={t('settings.confirmResetPlaceholder')}
                style={{ maxWidth: 280 }}
                value={confirmResetText}
                onInput={(e: any) => setConfirmResetText(e.target.value)}
              />
              <Button
                variant="danger"
                disabled={resettingHistory || confirmResetText !== t('settings.confirmResetWord')}
                loading={resettingHistory}
                onClick={handleResetHistory}
              >
                {t('settings.resetBtn')}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* TAB 2: ПЕСОЧНИЦЫ ТЕСТИРОВЩИКА (role >= 20)                          */}
      {/* =================================================================== */}
      {activeTab === 'tester' && isTesterOrAdmin && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Card: Sandbox Profiles Management */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.sandboxesTitle')}</h4>
            <p style={{ margin: '0 0 14px 0', fontSize: 13, color: 'var(--tiresias-text-muted)', lineHeight: 1.4 }}>
              {t('settings.sandboxesDesc')}
            </p>

            {/* Create profile form */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 16, maxWidth: 420 }}>
              <Input
                type="text"
                placeholder={t('settings.sandboxPlaceholder')}
                value={sandboxName}
                onInput={(e: any) => setSandboxName(e.target.value)}
                style={{ flex: 1 }}
              />
              <Button
                variant="primary"
                disabled={creatingSandbox || !sandboxName.trim()}
                loading={creatingSandbox}
                onClick={handleCreateSandbox}
                style={{ whiteSpace: 'nowrap' }}
              >
                {t('settings.createSandboxBtn')}
              </Button>
            </div>

            {/* Profiles list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {/* Main Tester Profile */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 12px',
                  borderRadius: 6,
                  background: activeUserId === authUser.user_id ? 'rgba(78, 201, 176, 0.12)' : 'rgba(255,255,255,0.03)',
                  border: `1px solid ${activeUserId === authUser.user_id ? 'var(--tiresias-tester-cyan)' : 'rgba(255,255,255,0.08)'}`,
                }}
              >
                <div>
                  <div style={{ fontWeight: 'bold', fontSize: 13 }}>
                    👤 {authUser.display_name} <span style={{ fontSize: 11, color: 'var(--tiresias-text-muted)' }}>{t('settings.mainProfile')}</span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)' }}>ID: {authUser.user_id}</div>
                </div>
                <div>
                  {activeUserId === authUser.user_id ? (
                    <Badge variant="rating-s" size="sm">
                      {t('settings.activeProfile')}
                    </Badge>
                  ) : (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => handleSwitchProfile(authUser.user_id)}
                    >
                      {t('settings.activateBtn')}
                    </Button>
                  )}
                </div>
              </div>

              {/* Sub-profiles */}
              {(authUser.sandbox_profiles || []).map((sp) => {
                const isActive = activeUserId === sp.user_id;
                return (
                  <div
                    key={sp.user_id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 12px',
                      borderRadius: 6,
                      background: isActive ? 'rgba(78, 201, 176, 0.12)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${isActive ? 'var(--tiresias-tester-cyan)' : 'rgba(255,255,255,0.08)'}`,
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 'bold', fontSize: 13, color: 'var(--tiresias-tester-cyan)' }}>
                        🧪 {sp.display_name || sp.username}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)' }}>ID: {sp.user_id}</div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      {isActive ? (
                        <Badge variant="rating-s" size="sm">
                          {t('settings.activeProfile')}
                        </Badge>
                      ) : (
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => handleSwitchProfile(sp.user_id)}
                        >
                          {t('settings.activateBtn')}
                        </Button>
                      )}
                      {confirmDeleteProfileId === sp.user_id ? (
                        <div style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
                          <Button
                            variant="danger"
                            size="sm"
                            style={{ fontSize: 11, padding: '3px 8px', fontWeight: 'bold' }}
                            onClick={() => handleDeleteSandbox(sp.user_id)}
                          >
                            {t('settings.deleteSandboxConfirm')}
                          </Button>
                          <Button
                            variant="secondary"
                            size="sm"
                            style={{ fontSize: 11, padding: '3px 6px' }}
                            onClick={() => setConfirmDeleteProfileId(null)}
                          >
                            ✕
                          </Button>
                        </div>
                      ) : (
                        <IconButton
                          icon="🗑️"
                          title={t('settings.deleteSandboxTitle')}
                          variant="danger"
                          size="sm"
                          onClick={() => setConfirmDeleteProfileId(sp.user_id)}
                        />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Card: Archetype Lab */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.labTitle')}</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, fontSize: 13, marginBottom: 14 }}>
              <div>
                <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('settings.labActiveProfile')} </span>
                <b style={{ color: 'var(--tiresias-tester-cyan)' }}>{activeUserId}</b>
              </div>
              <div>
                <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('settings.labCurrentArchetype')} </span>
                <b>#{profileStats?.taste_archetype_id ?? '—'}</b>
              </div>
              <div>
                <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('settings.labCoherence')} </span>
                <b>{profileStats?.taste_coherence ? (profileStats.taste_coherence * 100).toFixed(1) + '%' : '—'}</b>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, maxWidth: 440 }}>
              <FormGroup label={t('settings.lockArchetypeLabel')}>
                <Input
                  type="number"
                  min="0"
                  max="63"
                  value={lockedArch}
                  onInput={(e: any) => setLockedArch(e.target.value)}
                />
              </FormGroup>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--tiresias-text-muted)', marginBottom: 4 }}>
                  <span>{t('settings.archetypeWeightLabel')}</span>
                  <b>{forcedWeight}%</b>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  value={forcedWeight}
                  style={{ width: '100%' }}
                  onInput={(e: any) => setForcedWeight(parseInt(e.target.value, 10))}
                />
              </div>

              <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                <Button
                  variant="primary"
                  disabled={savingArch}
                  loading={savingArch}
                  onClick={handleSaveArchetypeOverride}
                >
                  {t('settings.lockBtn')}
                </Button>
                <Button
                  variant="secondary"
                  disabled={savingArch}
                  onClick={handleUnlockArchetype}
                >
                  {t('settings.unlockBtn')}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* TAB 3: АДМИНИСТРАТОР (role >= 100)                                  */}
      {/* =================================================================== */}
      {activeTab === 'admin' && isAdmin && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Card: Server Target Configuration */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.serverTargetTitle')}</h4>
            <p style={{ margin: '0 0 14px 0', fontSize: 13, color: 'var(--tiresias-text-muted)', lineHeight: 1.4 }}>
              {t('settings.serverTargetDesc')}
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10, marginBottom: 14 }}>
              {[
                { id: 'local', name: t('settings.targetLocal'), url: 'http://127.0.0.1:8000' },
                { id: 'vps', name: t('settings.targetVps'), url: config?.customUrl || 'http://127.0.0.1:8000' },
              ].map((srv) => {
                const isSelected = config?.selectedTarget === srv.id;
                return (
                  <button
                    key={srv.id}
                    type="button"
                    className="tiresias-button"
                    style={{
                      padding: '10px 12px',
                      textAlign: 'left',
                      background: isSelected ? 'rgba(232, 196, 70, 0.15)' : 'rgba(255,255,255,0.03)',
                      borderColor: isSelected ? 'var(--tiresias-gold)' : 'var(--tiresias-border)',
                      color: isSelected ? 'var(--tiresias-gold)' : 'inherit',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                    }}
                    onClick={() => handleSwitchServer(srv.id as any)}
                  >
                    <div style={{ fontWeight: 'bold', fontSize: 13 }}>{srv.name}</div>
                    <div style={{ fontSize: 10, color: 'var(--tiresias-text-muted)', marginTop: 2 }}>{srv.url}</div>
                  </button>
                );
              })}
            </div>

            {config?.selectedTarget === 'vps' && (
              <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                <Input
                  type="text"
                  placeholder={t('settings.customUrlPlaceholder')}
                  value={customServerUrl}
                  onInput={(e: any) => setCustomServerUrl(e.target.value)}
                  style={{ flex: 1 }}
                />
                <Button
                  variant="primary"
                  onClick={() => handleSwitchServer('vps', customServerUrl.trim())}
                >
                  {t('settings.applyBtn')}
                </Button>
              </div>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13, marginBottom: 12 }}>
              <span>{t('settings.activeServer')} <b>{config?.serverUrl}</b></span>
              <Button
                variant="secondary"
                size="sm"
                disabled={testingConnection}
                loading={testingConnection}
                onClick={() => handleSwitchServer(config?.selectedTarget || 'local')}
              >
                {t('settings.testConnectionBtn')}
              </Button>
            </div>

            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '8px 12px', borderRadius: 4, fontSize: 11, color: 'var(--tiresias-text-muted)', lineHeight: 1.4 }}>
              💡 <b style={{ color: 'var(--tiresias-accent)' }}>{t('settings.serverLogsTipTitle')}</b> {t('settings.serverLogsTip')}
            </div>
          </div>

          {/* Card: Maintenance Mode */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.maintenanceTitle')}</h4>
            <p style={{ margin: '0 0 12px 0', fontSize: 13, color: 'var(--tiresias-text-muted)', lineHeight: 1.4 }}>
              {t('settings.maintenanceDesc')}
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 13 }}>
                {t('settings.status')}{' '}
                <b style={{ color: health?.data?.maintenance ? 'var(--tiresias-danger)' : 'var(--tiresias-success)' }}>
                  {health?.data?.maintenance ? t('settings.maintenanceStatusActive') : t('settings.maintenanceStatusNormal')}
                </b>
              </span>
              <Button
                variant={health?.data?.maintenance ? 'success' : 'danger'}
                disabled={togglingMaintenance}
                loading={togglingMaintenance}
                onClick={handleToggleMaintenance}
              >
                {health?.data?.maintenance ? t('settings.disableMaintenance') : t('settings.enableMaintenance')}
              </Button>
            </div>
          </div>

          {/* Card: Suppression Rules */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.suppressionTitle')}</h4>
            <p style={{ margin: '0 0 12px 0', fontSize: 13, color: 'var(--tiresias-text-muted)', lineHeight: 1.4 }}>
              {t('settings.suppressionDesc')}
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <Button
                variant="primary"
                disabled={reloadingSuppression}
                loading={reloadingSuppression}
                onClick={handleReloadSuppression}
              >
                {t('settings.reloadSuppressionBtn')}
              </Button>
              {suppressionStatus && (
                <span style={{ fontSize: 12, color: 'var(--tiresias-text-muted)' }}>
                  {t('settings.loadedCategories', { count: Object.keys(suppressionStatus?.categories || {}).length })}
                </span>
              )}
            </div>
          </div>

          {/* Card: Diagnostics (Sanity Checks) */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.diagnosticsTitle')}</h4>
            <p style={{ margin: '0 0 12px 0', fontSize: 13, color: 'var(--tiresias-text-muted)' }}>
              {t('settings.diagnosticsDesc')}
            </p>
            <Button
              variant="primary"
              disabled={runningDiagnostics}
              loading={runningDiagnostics}
              style={{ marginBottom: 12 }}
              onClick={handleRunDiagnostics}
            >
              {t('settings.runDiagnosticsBtn')}
            </Button>

            {diagnostics && (
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: 12, borderRadius: 6, fontSize: 12, fontFamily: 'monospace' }}>
                <div>{t('settings.status')} <b style={{ color: diagnostics.status === 'healthy' ? 'var(--tiresias-success)' : 'var(--tiresias-danger)' }}>{diagnostics.status}</b></div>
                <div>{t('settings.auditDuration', { duration: diagnostics.audit_duration_ms?.toFixed(2) || '0' })}</div>
                <div>{t('settings.warnings', { count: diagnostics.total_warnings })}</div>
                {diagnostics.warnings && diagnostics.warnings.length > 0 && (
                  <ul style={{ margin: '8px 0 0 0', paddingLeft: 16, color: 'var(--tiresias-danger-light)' }}>
                    {diagnostics.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>

          {/* Card: JS Client Errors */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)' }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-gold)', fontSize: 15 }}>{t('settings.clientErrorsTitle')}</h4>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <Button
                variant="secondary"
                disabled={loadingErrors}
                loading={loadingErrors}
                onClick={handleLoadErrors}
              >
                {t('settings.loadErrorsBtn')}
              </Button>
              <Button
                variant="danger"
                onClick={handleClearErrors}
              >
                {t('settings.clearErrorsBtn')}
              </Button>
            </div>

            {clientErrors.length > 0 ? (
              <div style={{ maxHeight: 200, overflowY: 'auto', background: 'rgba(0,0,0,0.3)', padding: 8, borderRadius: 6 }}>
                {clientErrors.map((err) => (
                  <div key={err.id} style={{ fontSize: 11, borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '4px 0' }}>
                    <span style={{ color: 'var(--tiresias-danger-light)' }}>[{err.error_type}]</span> <b>{err.message}</b>
                    <div style={{ color: 'var(--tiresias-text-muted)' }}>{err.timestamp} • {err.source_file || 'inline'}:{err.lineno || 0}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: 'var(--tiresias-text-muted)' }}>{t('settings.noErrors')}</div>
            )}
          </div>

          {/* Card: Super-Admin Database Reset */}
          <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-danger)' }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--tiresias-danger)', fontSize: 15 }}>{t('settings.globalResetTitle')}</h4>
            <p style={{ margin: '0 0 12px 0', fontSize: 13, color: 'var(--tiresias-text-muted)' }}>
              {t('settings.globalResetDesc')}
            </p>
            <div style={{ display: 'flex', gap: 8, maxWidth: 440 }}>
              <Input
                type="password"
                placeholder={t('settings.adminKeyPlaceholder')}
                value={adminKeyInput}
                onInput={(e: any) => setAdminKeyInput(e.target.value)}
                style={{ flex: 1 }}
              />
              {!confirmWipeAll ? (
                <Button
                  variant="danger"
                  style={{ whiteSpace: 'nowrap' }}
                  onClick={handleWipeAll}
                >
                  {t('settings.wipeAllBtn')}
                </Button>
              ) : (
                <div style={{ display: 'inline-flex', gap: 6 }}>
                  <Button
                    variant="danger"
                    style={{ whiteSpace: 'nowrap' }}
                    onClick={handleWipeAll}
                  >
                    {t('settings.confirmWipeAllBtn')}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => setConfirmWipeAll(false)}
                  >
                    ✕
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

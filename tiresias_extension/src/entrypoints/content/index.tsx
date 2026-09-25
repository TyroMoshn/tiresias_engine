import { defineContentScript } from 'wxt/utils/define-content-script';
import { render, h, Fragment } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import '../../styles/theme.css';
import { TiresiasApi } from '../../lib/api';
import { ToastContainer } from '../../components/ui';
import { t } from '../../lib/i18n';
import { FeedPage } from '../../components/FeedPage';
import { BoardsPage } from '../../components/BoardsPage';
import { ActivityPage } from '../../components/ActivityPage';
import { SettingsPage } from '../../components/SettingsPage';
import { SimilarDrawer } from '../../components/SimilarDrawer';
import { BoardPickerModal } from '../../components/BoardPickerModal';
import { initSeenTracker } from './seenTracker';
import { initThumbnailActions } from './thumbnailActions';
import { initTelemetry } from '../../lib/telemetry';
import { PostMetadata } from '../../lib/types';

function GlobalOverlays() {
  const [similarPostId, setSimilarPostId] = useState<number | null>(null);
  const [boardTargetPost, setBoardTargetPost] = useState<PostMetadata | null>(null);

  useEffect(() => {
    const handleSimilar = (e: any) => {
      if (e.detail?.postId) setSimilarPostId(e.detail.postId);
    };
    const handleBoardPicker = (e: any) => {
      if (e.detail?.post) setBoardTargetPost(e.detail.post);
    };

    window.addEventListener('tiresias:show-similar', handleSimilar);
    window.addEventListener('tiresias:show-board-picker', handleBoardPicker);

    return () => {
      window.removeEventListener('tiresias:show-similar', handleSimilar);
      window.removeEventListener('tiresias:show-board-picker', handleBoardPicker);
    };
  }, []);

  return (
    <>
      <ToastContainer />
      {similarPostId && (
        <SimilarDrawer
          postId={similarPostId}
          onClose={() => setSimilarPostId(null)}
          onAddToBoard={(post) => setBoardTargetPost(post)}
        />
      )}
      {boardTargetPost && (
        <BoardPickerModal
          post={boardTargetPost}
          onClose={() => setBoardTargetPost(null)}
        />
      )}
    </>
  );
}

export default defineContentScript({
  matches: ['*://*.e926.net/*', '*://*.e621.net/*'],
  runAt: 'document_end',
  main() {
    initTelemetry('content');
    console.log('[Tiresias] Content script active');

    // Mount global modal host
    const overlayHost = document.createElement('div');
    overlayHost.id = 'tiresias-global-overlays';
    document.body.appendChild(overlayHost);
    render(h(GlobalOverlays, {}), overlayHost);

    // Initialize thumbnail enhancements and seen observer
    initThumbnailActions();
    initSeenTracker();

    // Detect active e621 / e926 session user and perform silent handshake
    const detected = detectSiteUser();
    if (detected && !detected.isAnonymous && detected.id > 0) {
      import('../../lib/storage').then(async ({ loadConfig, saveConfig }) => {
        try {
          await saveConfig({ detectedSiteUser: { id: detected.id, name: detected.name } });
          const cfg = await loadConfig();
          const activeSiteId = cfg.authUser?.site_user_id;
          if (cfg.useSessionUser !== false && (!cfg.authToken || activeSiteId !== detected.id)) {
            console.log(`[Tiresias] Silent session handshake for ${detected.name} (#${detected.id})...`);
            await TiresiasApi.sessionHandshake(detected.id, detected.name, 'Browser ContentScript');
          }
        } catch (err) {
          console.warn('[Tiresias] Session handshake failed:', err);
        }
      });
    }

    // Inject navigation tabs into e926 header
    injectNavigation();

    // Handle virtual routes
    handleRouting();
    window.addEventListener('popstate', () => handleRouting());
  },
});

function injectNavigation() {
  if (document.getElementById('tiresias-nav-group') || document.getElementById('tiresias-nav-feed')) return;

  const navHelp = document.querySelector('nav.navigation menu.nav-help, menu.nav-help');
  const navPrimary = document.querySelector('nav.navigation menu.nav-primary, menu.nav-primary');
  const targetParent = navHelp || navPrimary;
  if (!targetParent) return;

  const liFeed = document.createElement('li');
  liFeed.className = 'tiresias-nav-item tiresias-nav-first';
  liFeed.innerHTML = `
    <a href="/feed" class="tiresias-nav-link" id="tiresias-nav-feed">
      <span class="tiresias-nav-badge">🔮</span> <span class="tiresias-nav-text" id="tiresias-nav-text-feed">${t('nav.feed')}</span>
      <span class="tiresias-nav-status checking" id="tiresias-nav-status" title="${t('nav.statusChecking')}"></span>
    </a>
  `;

  const liBoards = document.createElement('li');
  liBoards.className = 'tiresias-nav-item';
  liBoards.innerHTML = `
    <a href="/boards" class="tiresias-nav-link" id="tiresias-nav-boards">
      <span class="tiresias-nav-badge">📂</span> <span class="tiresias-nav-text" id="tiresias-nav-text-boards">${t('nav.boards')}</span>
    </a>
  `;

  const liActivity = document.createElement('li');
  liActivity.className = 'tiresias-nav-item';
  liActivity.innerHTML = `
    <a href="/activity" class="tiresias-nav-link" id="tiresias-nav-activity">
      <span class="tiresias-nav-badge">📊</span> <span class="tiresias-nav-text" id="tiresias-nav-text-activity">${t('nav.activity')}</span>
    </a>
  `;

  const liSettings = document.createElement('li');
  liSettings.className = 'tiresias-nav-item tiresias-nav-last';
  liSettings.innerHTML = `
    <a href="/tiresias/settings" class="tiresias-nav-link" id="tiresias-nav-settings">
      <span class="tiresias-nav-badge">⚙️</span> <span class="tiresias-nav-text" id="tiresias-nav-text-settings">${t('nav.settings')}</span>
    </a>
  `;

  // Intercept click to navigate smoothly via history API
  liFeed.querySelector('a')?.addEventListener('click', (e) => {
    e.preventDefault();
    history.pushState(null, '', '/feed');
    handleRouting();
  });

  liBoards.querySelector('a')?.addEventListener('click', (e) => {
    e.preventDefault();
    history.pushState(null, '', '/boards');
    handleRouting();
  });

  liActivity.querySelector('a')?.addEventListener('click', (e) => {
    e.preventDefault();
    history.pushState(null, '', '/activity');
    handleRouting();
  });

  liSettings.querySelector('a')?.addEventListener('click', (e) => {
    e.preventDefault();
    history.pushState(null, '', '/tiresias/settings');
    handleRouting();
  });

  targetParent.appendChild(liFeed);
  targetParent.appendChild(liBoards);
  targetParent.appendChild(liActivity);
  targetParent.appendChild(liSettings);

  // Dynamic right margin offset to prevent overlapping avatar and username
  updateNavOffset();
  window.addEventListener('resize', updateNavOffset);
  const navControls = document.querySelector('nav.navigation menu.nav-controls, menu.nav-controls');
  if (navControls && typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => updateNavOffset()).observe(navControls);
  }

  // Health check for status dot
  checkServerConnection();
  setInterval(() => checkServerConnection(), 30000);

  // Re-evaluate navigation labels and document title when language changes
  window.addEventListener('tiresias:language-changed', () => {
    updateNavLabels();
  });
}

function updateNavLabels() {
  const feedText = document.getElementById('tiresias-nav-text-feed');
  if (feedText) feedText.textContent = t('nav.feed');
  const boardsText = document.getElementById('tiresias-nav-text-boards');
  if (boardsText) boardsText.textContent = t('nav.boards');
  const activityText = document.getElementById('tiresias-nav-text-activity');
  if (activityText) activityText.textContent = t('nav.activity');
  const settingsText = document.getElementById('tiresias-nav-text-settings');
  if (settingsText) settingsText.textContent = t('nav.settings');
  checkServerConnection();
  updateDocumentTitle();
}

function updateNavOffset() {
  const navControls = document.querySelector('nav.navigation menu.nav-controls, menu.nav-controls');
  if (navControls) {
    const width = Math.ceil(navControls.getBoundingClientRect().width);
    const offset = Math.max(width + 24, 180);
    document.documentElement.style.setProperty('--tiresias-nav-offset', `${offset}px`);
  }
}

async function checkServerConnection() {
  const statusDot = document.getElementById('tiresias-nav-status');
  if (!statusDot) return;

  const res = await TiresiasApi.checkHealth();
  if (res.ok) {
    statusDot.className = 'tiresias-nav-status online';
    statusDot.title = t('nav.statusOnline', { latency: res.latencyMs });
  } else {
    statusDot.className = 'tiresias-nav-status offline';
    statusDot.title = t('nav.statusOffline', { error: res.error || 'Unavailable' });
  }
}

let originalPageContent: string | null = null;

function updateDocumentTitle() {
  const path = window.location.pathname;
  const host = window.location.host || 'e926';
  if (path === '/feed' || path === '/tiresias/feed') {
    document.title = t('nav.pageTitleFeed', { host });
  } else if (path === '/boards' || path === '/tiresias/boards') {
    document.title = t('nav.pageTitleBoards', { host });
  } else if (
    path === '/tiresias/activity' ||
    path === '/activity' ||
    path === '/likes' ||
    path === '/dislikes' ||
    path === '/history'
  ) {
    document.title = t('nav.pageTitleActivity', { host });
  } else if (path === '/tiresias/settings' || path === '/settings') {
    document.title = t('nav.pageTitleSettings', { host });
  }
}

function handleRouting() {
  const path = window.location.pathname;
  const pageContainer = document.getElementById('page');

  if (path === '/feed' || path === '/tiresias/feed') {
    if (pageContainer) {
      if (originalPageContent === null) {
        originalPageContent = pageContainer.innerHTML;
      }
      pageContainer.innerHTML = '<div id="tiresias-page-root"></div>';
      const root = document.getElementById('tiresias-page-root');
      if (root) {
        render(h(FeedPage, {}), root);
      }
    }
    updateDocumentTitle();
  } else if (path === '/boards' || path === '/tiresias/boards') {
    if (pageContainer) {
      if (originalPageContent === null) {
        originalPageContent = pageContainer.innerHTML;
      }
      pageContainer.innerHTML = '<div id="tiresias-page-root"></div>';
      const root = document.getElementById('tiresias-page-root');
      if (root) {
        render(h(BoardsPage, {}), root);
      }
    }
    updateDocumentTitle();
  } else if (
    path === '/tiresias/activity' ||
    path === '/activity' ||
    path === '/likes' ||
    path === '/dislikes' ||
    path === '/history'
  ) {
    const initialTab = (path === '/dislikes') ? 'dislike' : 'like';
    if (pageContainer) {
      if (originalPageContent === null) {
        originalPageContent = pageContainer.innerHTML;
      }
      pageContainer.innerHTML = '<div id="tiresias-page-root"></div>';
      const root = document.getElementById('tiresias-page-root');
      if (root) {
        render(h(ActivityPage, { initialTab }), root);
      }
    }
    updateDocumentTitle();
  } else if (path === '/tiresias/settings' || path === '/settings') {
    if (pageContainer) {
      if (originalPageContent === null) {
        originalPageContent = pageContainer.innerHTML;
      }
      pageContainer.innerHTML = '<div id="tiresias-page-root"></div>';
      const root = document.getElementById('tiresias-page-root');
      if (root) {
        render(h(SettingsPage, {}), root);
      }
    }
    updateDocumentTitle();
  }
}


interface SiteUserInfo {
  id: number;
  name: string;
  isAnonymous: boolean;
}

function detectSiteUser(): SiteUserInfo | null {
  const siteUserScript = document.getElementById('site-user');
  if (siteUserScript?.textContent) {
    try {
      const base64 = siteUserScript.textContent.trim();
      const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
      const json = new TextDecoder().decode(bytes);
      const parsed = JSON.parse(json);
      if (parsed && parsed.id && parsed.name && !parsed.is?.anonymous) {
        return {
          id: Number(parsed.id),
          name: String(parsed.name),
          isAnonymous: false,
        };
      }
    } catch (e) {
      console.warn('[Tiresias] Could not parse #site-user data:', e);
    }
  }

  const navProfile = document.querySelector('a.nav-controls-profile[data-name], a.simple-avatar[data-name]');
  const name = navProfile?.getAttribute('data-name');
  if (name && name !== '<blank>' && name !== 'Sign In') {
    return { id: 0, name, isAnonymous: false };
  }
  return null;
}

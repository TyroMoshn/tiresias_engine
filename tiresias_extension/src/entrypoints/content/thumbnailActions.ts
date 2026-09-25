import { TiresiasApi } from '../../lib/api';
import { loadConfig } from '../../lib/storage';
import { PostMetadata } from '../../lib/types';
import { t } from '../../lib/i18n';

export function initThumbnailActions() {
  let activeThumbnail: HTMLElement | null = null;
  let hideTimer: any = null;
  let testerMode = false;

  const updateTesterMode = () => {
    loadConfig().then((cfg) => {
      testerMode = !!cfg.testerMode;
    });
  };
  updateTesterMode();

  if (typeof browser !== 'undefined' && browser.storage?.onChanged) {
    browser.storage.onChanged.addListener(updateTesterMode);
  } else if (typeof chrome !== 'undefined' && chrome.storage?.onChanged) {
    chrome.storage.onChanged.addListener(updateTesterMode);
  }

  // Create singleton floating action bar container
  let container = document.getElementById('tiresias-floating-actions');
  if (!container) {
    container = document.createElement('div');
    container.id = 'tiresias-floating-actions';
    container.className = 'tiresias-floating-actions';
    container.innerHTML = `
      <div id="tiresias-floating-diag" class="tiresias-floating-diag" style="display: none;"></div>
      <div class="tiresias-floating-bar">
        <button type="button" id="tiresias-act-like" class="tiresias-floating-btn" data-tooltip="${t('actions.like')}">👍</button>
        <button type="button" id="tiresias-act-hide" class="tiresias-floating-btn" data-tooltip="${t('actions.hide')}">👎</button>
        <button type="button" id="tiresias-act-board" class="tiresias-floating-btn" data-tooltip="${t('actions.board')}">📁</button>
        <button type="button" id="tiresias-act-remove-board" class="tiresias-floating-btn tiresias-act-danger" style="display: none;" data-tooltip="${t('actions.removeFromBoard')}">✕</button>
        <button type="button" id="tiresias-act-add-current-board" class="tiresias-floating-btn tiresias-act-primary" style="display: none;" data-tooltip="${t('actions.addToCurrentBoard')}">➕</button>
        <button type="button" id="tiresias-act-similar" class="tiresias-floating-btn" data-tooltip="${t('actions.similar')}">✨</button>
      </div>
    `;
    document.body.appendChild(container);
  }

  const diagEl = container.querySelector('#tiresias-floating-diag') as HTMLElement;
  const likeBtn = container.querySelector('#tiresias-act-like') as HTMLButtonElement;
  const hideBtn = container.querySelector('#tiresias-act-hide') as HTMLButtonElement;
  const boardBtn = container.querySelector('#tiresias-act-board') as HTMLButtonElement;
  const removeBoardBtn = container.querySelector('#tiresias-act-remove-board') as HTMLButtonElement;
  const addCurrentBoardBtn = container.querySelector('#tiresias-act-add-current-board') as HTMLButtonElement;
  const similarBtn = container.querySelector('#tiresias-act-similar') as HTMLButtonElement;

  const updateTooltips = () => {
    if (likeBtn) likeBtn.setAttribute('data-tooltip', t('actions.like'));
    if (hideBtn) hideBtn.setAttribute('data-tooltip', t('actions.hide'));
    if (boardBtn) boardBtn.setAttribute('data-tooltip', t('actions.board'));
    if (removeBoardBtn) removeBoardBtn.setAttribute('data-tooltip', t('actions.removeFromBoard'));
    if (addCurrentBoardBtn) addCurrentBoardBtn.setAttribute('data-tooltip', t('actions.addToCurrentBoard'));
    if (similarBtn) similarBtn.setAttribute('data-tooltip', t('actions.similar'));
  };

  updateTooltips();
  window.addEventListener('tiresias:language-changed', updateTooltips);

  function showFloatingBar(target: HTMLElement) {
    if (!container) return;
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }

    activeThumbnail = target;
    const idStr = target.dataset.id || target.getAttribute('id')?.replace('post_', '');
    if (!idStr) return;
    const postId = parseInt(idStr, 10);
    if (isNaN(postId)) return;

    // Check if post is currently liked
    if (target.classList.contains('tiresias-post-liked')) {
      likeBtn.classList.add('active');
    } else {
      likeBtn.classList.remove('active');
    }

    // Check if post is currently disliked/hidden
    if (target.classList.contains('tiresias-post-disliked') || target.classList.contains('tiresias-post-hidden')) {
      hideBtn.classList.add('disliked');
    } else {
      hideBtn.classList.remove('disliked');
    }

    // Contextual buttons (e.g. inside a board or board recommendations)
    const context = target.dataset.context;
    const boardId = target.dataset.boardId;
    if (context === 'board-post' && boardId) {
      removeBoardBtn.style.display = 'flex';
      addCurrentBoardBtn.style.display = 'none';
      boardBtn.style.display = 'none';
    } else if (context === 'board-rec' && boardId) {
      addCurrentBoardBtn.style.display = 'flex';
      removeBoardBtn.style.display = 'none';
      boardBtn.style.display = 'none';
    } else {
      removeBoardBtn.style.display = 'none';
      addCurrentBoardBtn.style.display = 'none';
      boardBtn.style.display = 'flex';
    }

    // Diagnostic information for Tester Mode
    if (testerMode && diagEl) {
      const score = target.dataset.score || target.dataset.scoreVal || '';
      const reason = target.dataset.reason || '';
      const rating = (target.dataset.rating || '').toUpperCase();
      const ext = target.dataset.fileExt || target.dataset.ext || '';
      let diagText = `#${postId}`;
      if (rating) diagText += ` [${rating}]`;
      if (ext) diagText += ` (${ext})`;
      if (score) diagText += ` ★${score}`;
      if (reason) diagText += ` • ${reason}`;
      diagEl.textContent = diagText;
      diagEl.style.display = 'block';
    } else if (diagEl) {
      diagEl.style.display = 'none';
    }

    // Position floating bar over the target thumbnail image (anchored at the bottom of the art, above footer buttons)
    const anchorEl = (target.querySelector('.tiresias-thumb-box, a.thm-link, .tiresias-card-thumb, img') as HTMLElement) || target;
    const rect = anchorEl.getBoundingClientRect();
    const barHeight = container.offsetHeight || 38;
    const barWidth = container.offsetWidth || 160;

    // Fixed positioning relative to viewport
    const left = Math.max(8, Math.min(window.innerWidth - barWidth - 8, rect.left + (rect.width - barWidth) / 2));
    let top = rect.bottom - barHeight - 6;
    if (top < 0) top = rect.top + 6;

    container.style.left = `${Math.round(left)}px`;
    container.style.top = `${Math.round(top)}px`;
    container.classList.add('visible');
  }

  function scheduleHide() {
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      if (container) {
        container.classList.remove('visible');
      }
      activeThumbnail = null;
    }, 140); // 140ms grace buffer for smooth pointer transition (WCAG 1.4.13)
  }

  // Event Delegation for hover over any thumbnail on the page
  document.addEventListener('pointerover', (e) => {
    const eventTarget = e.target as HTMLElement | null;
    // If hovering directly over card footer buttons (e.g. "✕ Убрать", "+ В доску", etc.), hide floating bar immediately
    if (
      eventTarget &&
      eventTarget.closest(
        'button, .tiresias-button, .tiresias-btn-icon, .tiresias-card-meta, .thm-desc, input, select'
      ) &&
      !container?.contains(eventTarget)
    ) {
      if (container) container.classList.remove('visible');
      activeThumbnail = null;
      return;
    }

    const target = eventTarget?.closest(
      'article.post-thumbnail, article.thumbnail, .tiresias-card-item'
    ) as HTMLElement | null;

    if (target) {
      showFloatingBar(target);
    }
  });

  document.addEventListener('pointerout', (e) => {
    const target = (e.target as HTMLElement)?.closest(
      'article.post-thumbnail, article.thumbnail, .tiresias-card-item'
    ) as HTMLElement | null;

    if (target) {
      scheduleHide();
    }
  });

  // Keep visible when hovering the action bar itself
  container.addEventListener('pointerenter', () => {
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  });

  container.addEventListener('pointerleave', () => {
    scheduleHide();
  });

  // Like Action (👍) - Toggle on/off
  likeBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!activeThumbnail) return;

    const idStr = activeThumbnail.dataset.id || activeThumbnail.getAttribute('id')?.replace('post_', '');
    const postId = parseInt(idStr || '0', 10);
    if (!postId) return;

    const isLiked = activeThumbnail.classList.contains('tiresias-post-liked');
    if (isLiked) {
      activeThumbnail.classList.remove('tiresias-post-liked');
      likeBtn.classList.remove('active');
      await TiresiasApi.undoFeedback(postId, 'like');
    } else {
      activeThumbnail.classList.add('tiresias-post-liked');
      likeBtn.classList.add('active');
      // Remove dislike if present
      if (activeThumbnail.classList.contains('tiresias-post-disliked')) {
        activeThumbnail.classList.remove('tiresias-post-disliked');
        hideBtn.classList.remove('disliked');
      }
      likeBtn.style.transform = 'scale(1.3)';
      setTimeout(() => { likeBtn.style.transform = ''; }, 180);
      await TiresiasApi.sendFeedback(postId, 'like');
    }
  });

  // Hide / Dislike Action (👎) - Toggle on/off
  hideBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!activeThumbnail) return;

    const idStr = activeThumbnail.dataset.id || activeThumbnail.getAttribute('id')?.replace('post_', '');
    const postId = parseInt(idStr || '0', 10);
    if (!postId) return;

    const isDisliked = activeThumbnail.classList.contains('tiresias-post-disliked') || activeThumbnail.classList.contains('tiresias-post-hidden');
    if (isDisliked) {
      activeThumbnail.classList.remove('tiresias-post-disliked');
      activeThumbnail.classList.remove('tiresias-post-hidden');
      hideBtn.classList.remove('disliked');
      await TiresiasApi.undoFeedback(postId, 'hide');
    } else {
      activeThumbnail.classList.add('tiresias-post-disliked');
      hideBtn.classList.add('disliked');
      // Remove like if present
      if (activeThumbnail.classList.contains('tiresias-post-liked')) {
        activeThumbnail.classList.remove('tiresias-post-liked');
        likeBtn.classList.remove('active');
      }
      hideBtn.style.transform = 'scale(1.3)';
      setTimeout(() => { hideBtn.style.transform = ''; }, 180);
      await TiresiasApi.sendFeedback(postId, 'hide');
    }
  });

  // Add to Board Action (📁)
  boardBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!activeThumbnail) return;

    const idStr = activeThumbnail.dataset.id || activeThumbnail.getAttribute('id')?.replace('post_', '');
    const postId = parseInt(idStr || '0', 10);
    if (!postId) return;

    const postMeta: PostMetadata = {
      id: postId,
      score: parseInt(activeThumbnail.dataset.score || activeThumbnail.dataset.scoreVal || '0', 10),
      fav_count: parseInt(activeThumbnail.dataset.favCount || '0', 10),
      rating: (activeThumbnail.dataset.rating || 's') as any,
      preview_url: activeThumbnail.querySelector('img')?.getAttribute('src') || undefined,
    };

    window.dispatchEvent(
      new CustomEvent('tiresias:show-board-picker', { detail: { post: postMeta } })
    );
  });

  // Similar Action (✨)
  similarBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!activeThumbnail) return;

    const idStr = activeThumbnail.dataset.id || activeThumbnail.getAttribute('id')?.replace('post_', '');
    const postId = parseInt(idStr || '0', 10);
    if (!postId) return;

    window.dispatchEvent(
      new CustomEvent('tiresias:show-similar', { detail: { postId } })
    );
  });

  // Remove from Board Action (✕)
  removeBoardBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!activeThumbnail) return;
    const idStr = activeThumbnail.dataset.id || activeThumbnail.getAttribute('id')?.replace('post_', '');
    const postId = parseInt(idStr || '0', 10);
    const boardId = activeThumbnail.dataset.boardId;
    if (postId && boardId) {
      window.dispatchEvent(
        new CustomEvent('tiresias:remove-from-board', { detail: { postId, boardId } })
      );
      if (container) container.classList.remove('visible');
    }
  });

  // Add to Current Board Action (➕)
  addCurrentBoardBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!activeThumbnail) return;
    const idStr = activeThumbnail.dataset.id || activeThumbnail.getAttribute('id')?.replace('post_', '');
    const postId = parseInt(idStr || '0', 10);
    const boardId = activeThumbnail.dataset.boardId;
    if (postId && boardId) {
      window.dispatchEvent(
        new CustomEvent('tiresias:add-to-current-board', { detail: { postId, boardId } })
      );
      addCurrentBoardBtn.textContent = '✓';
      setTimeout(() => { addCurrentBoardBtn.textContent = '➕'; }, 1200);
    }
  });
}

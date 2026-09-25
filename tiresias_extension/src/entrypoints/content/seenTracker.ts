import { TiresiasApi } from '../../lib/api';
import { loadConfig } from '../../lib/storage';

const seenQueue = new Set<number>();
let timer: any = null;
let isEnabled = true;

async function flushSeen() {
  if (seenQueue.size === 0) return;
  const ids = Array.from(seenQueue);
  seenQueue.clear();
  try {
    await TiresiasApi.sendSeenBatch(ids);
  } catch (e) {
    console.warn('[Tiresias] Error sending seen batch:', e);
  }
}

export async function initSeenTracker() {
  const cfg = await loadConfig();
  isEnabled = cfg.autoSeen;
  if (!isEnabled) return;

  const visibleTimers = new Map<number, any>();

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const target = entry.target as HTMLElement;
        const idStr = target.dataset.id || target.getAttribute('id')?.replace('post_', '');
        if (!idStr) return;
        const postId = parseInt(idStr, 10);
        if (isNaN(postId)) return;

        if (entry.isIntersecting && entry.intersectionRatio >= 0.5) {
          // Started viewing
          if (!visibleTimers.has(postId)) {
            const t = setTimeout(() => {
              seenQueue.add(postId);
              visibleTimers.delete(postId);
              if (!timer) {
                timer = setTimeout(() => {
                  timer = null;
                  flushSeen();
                }, 3000);
              }
            }, 1000); // 1.0 second dwell threshold
            visibleTimers.set(postId, t);
          }
        } else {
          // Scrolled away before threshold
          if (visibleTimers.has(postId)) {
            clearTimeout(visibleTimers.get(postId));
            visibleTimers.delete(postId);
          }
        }
      });
    },
    { threshold: [0.5] }
  );

  function observeAll() {
    document.querySelectorAll('article.post-thumbnail:not([data-tiresias-seen-observed]), .tiresias-card-item[data-id]:not([data-tiresias-seen-observed])').forEach((el) => {
      el.setAttribute('data-tiresias-seen-observed', 'true');
      observer.observe(el);
    });
  }

  observeAll();

  // MutationObserver for dynamic infinite scroll on booru and Tiresias feed
  const mutObserver = new MutationObserver(() => observeAll());
  const container = document.querySelector('#posts, .posts-container, #page') || document.body;
  if (container) {
    mutObserver.observe(container, { childList: true, subtree: true });
  }

  // Flush on unload
  window.addEventListener('beforeunload', () => flushSeen());
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flushSeen();
  });
}

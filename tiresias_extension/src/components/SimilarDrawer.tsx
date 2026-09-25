import { h } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { TiresiasApi } from '../lib/api';
import { PostMetadata } from '../lib/types';
import { Drawer, Badge } from './ui';
import { useTranslation } from '../lib/i18n';

interface Props {
  postId: number | null;
  onClose: () => void;
  onAddToBoard?: (post: PostMetadata) => void;
}

export function SimilarDrawer({ postId, onClose }: Props) {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [items, setItems] = useState<PostMetadata[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Settings
  const [imageContain, setImageContain] = useState(false);
  const [cardSize, setCardSize] = useState(150);
  const [showDesc, setShowDesc] = useState(true);

  useEffect(() => {
    const readSettings = () => {
      try {
        setImageContain(localStorage.getItem('e6.posts.contain') === 'true');
        const savedDesc = localStorage.getItem('e6.posts.show_desc');
        if (savedDesc !== null) setShowDesc(savedDesc === 'true');
        const savedSize = parseInt(localStorage.getItem('e6.posts.custom_size') || '', 10);
        if (!isNaN(savedSize) && savedSize >= 110 && savedSize <= 340) {
          setCardSize(savedSize);
        }
      } catch {}
    };

    readSettings();

    const handleSettingsChanged = (e: any) => {
      if (e?.detail) {
        if (typeof e.detail.imageContain === 'boolean') setImageContain(e.detail.imageContain);
        if (typeof e.detail.showDesc === 'boolean') setShowDesc(e.detail.showDesc);
        if (typeof e.detail.cardSize === 'number') setCardSize(e.detail.cardSize);
      } else {
        readSettings();
      }
    };

    window.addEventListener('tiresias:view_settings_changed', handleSettingsChanged);
    window.addEventListener('storage', readSettings);
    return () => {
      window.removeEventListener('tiresias:view_settings_changed', handleSettingsChanged);
      window.removeEventListener('storage', readSettings);
    };
  }, []);

  useEffect(() => {
    if (!postId) return;
    setLoading(true);
    setError(null);
    TiresiasApi.getSimilar(postId, 16, ['s', 'q', 'e'])
      .then(async (res) => {
        const rawItems = (res.items || []).map((x: any) => ({ ...x, id: x.id ?? x.post_id }));
        const pids = rawItems.map((x: any) => x.id).filter(Boolean);
        if (pids.length > 0) {
          try {
            const booruResp = await fetch(`/posts.json?tags=id:${pids.join(',')}`);
            if (booruResp.ok) {
              const bData = await booruResp.json();
              const map = new Map<number, any>();
              for (const bp of (bData.posts || [])) map.set(bp.id, bp);
              for (const it of rawItems) {
                const bp = map.get(it.id);
                if (bp) {
                  it.preview_url = bp.preview?.url || it.preview_url;
                  it.score = bp.score?.total ?? it.score;
                  it.score_val = bp.score?.total ?? it.score_val;
                  it.rating = bp.rating ?? it.rating;
                  it.fav_count = bp.fav_count ?? it.fav_count;
                  it.comment_count = bp.comment_count ?? (it as any).comment_count;
                }
              }
            }
          } catch (err) {
            console.warn('[Tiresias] Could not enrich similar items:', err);
          }
        }
        setItems(rawItems);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || t('similar.loadError'));
        setLoading(false);
      });
  }, [postId, t]);

  if (!postId) return null;

  return (
    <Drawer
      isOpen={!!postId}
      onClose={onClose}
      title={t('similar.title', { id: postId })}
      icon="✨"
    >
      {loading && (
        <div style={{ textAlign: 'center', padding: 40, color: 'var(--tiresias-accent)' }}>
          <div>{t('similar.searching')}</div>
        </div>
      )}

      {error && (
        <div style={{ padding: 12, background: 'var(--tiresias-danger-subtle)', border: '1px solid var(--tiresias-danger)', borderRadius: 'var(--tiresias-radius-sm)', color: 'var(--tiresias-danger-light)', marginBottom: 12 }}>
          {error}
        </div>
      )}

      {!loading && !error && items.length === 0 && (
        <div style={{ color: 'var(--tiresias-text-muted)', textAlign: 'center', padding: 30 }}>
          {t('similar.empty')}
        </div>
      )}

      {/* Unified e621 Thumbnail Grid */}
      <section
        className={`posts-container tiresias-posts-container ${!showDesc ? 'no-stats' : ''}`}
        style={{ '--thumb-image-size': `${cardSize}px`, overflowY: 'auto', maxHeight: 'calc(100vh - 120px)' } as any}
        data-st-contain={imageContain ? 'true' : 'false'}
        data-st-show-desc={showDesc ? 'true' : 'false'}
      >
        {items.map((item) => {
          const pid = item.id ?? (item as any).post_id;
          const host = typeof window !== 'undefined' ? window.location.host : 'e926.net';
          const scoreVal = item.score_val ?? Math.round((item.score || 0) * 100);

          return (
            <article
              key={pid}
              className="thumbnail post-thumbnail tiresias-post-thumb"
              data-id={pid}
              data-score={scoreVal}
              data-rating={item.rating || 's'}
              data-context="similar"
            >
              <a
                href={`https://${host}/posts/${pid}`}
                target="_blank"
                rel="noreferrer"
                className="thm-link"
              >
                {item.is_video && <div className="tiresias-video-badge">🎬</div>}
                {item.preview_url ? (
                  <img src={item.preview_url} alt={`Post #${pid}`} loading="lazy" />
                ) : (
                  <div className="tiresias-thumb-placeholder">#{pid}</div>
                )}
              </a>

              {showDesc && (
                <div className={`desc thm-desc thm-rating-${item.rating || 's'}`}>
                  <span className="thm-desc-a">
                    <span className="thm-desc-m thm-score" title={t('common.score')}>
                      ▲ {scoreVal}
                    </span>
                    <span className="thm-desc-m thm-favorites" title={t('common.favorites')}>
                      ★ {item.fav_count ?? 0}
                    </span>
                    {item.comment_count !== undefined && (
                      <span className="thm-desc-m thm-comments" title={t('common.comments')}>
                        💬 {item.comment_count}
                      </span>
                    )}
                  </span>
                  <span className="thm-desc-b thm-rating" title={`${t('common.rating')}: ${(item.rating || 's').toUpperCase()}`}>
                    {(item.rating || 's').toUpperCase()}
                  </span>
                </div>
              )}
            </article>
          );
        })}
      </section>
    </Drawer>
  );
}

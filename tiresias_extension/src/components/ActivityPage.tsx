import { h, Fragment } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { TiresiasApi } from '../lib/api';
import { PostMetadata, ServerConfig } from '../lib/types';
import { loadConfig } from '../lib/storage';
import { SimilarDrawer } from './SimilarDrawer';
import { Button, IconButton, Badge, toast } from './ui';
import { useTranslation } from '../lib/i18n';

interface ActivityItem {
  post_id: number;
  signal_type: string;
  created_at: number;
  score?: number;
  fav_count?: number;
  rating?: string;
  file_ext?: string;
  preview_url?: string;
  is_video?: boolean;
}

export function ActivityPage({ initialTab = 'like' }: { initialTab?: 'like' | 'dislike' }) {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<'like' | 'dislike'>(initialTab);
  const [profile, setProfile] = useState<any>(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [config, setConfig] = useState<ServerConfig | null>(null);

  // Settings
  const [imageContain, setImageContain] = useState(false);
  const [cardSize, setCardSize] = useState(180);
  const [showDesc, setShowDesc] = useState(true);

  const [items, setItems] = useState<ActivityItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [similarPostId, setSimilarPostId] = useState<number | null>(null);

  useEffect(() => {
    try {
      setImageContain(localStorage.getItem('e6.posts.contain') === 'true');
      const savedSize = parseInt(localStorage.getItem('e6.posts.custom_size') || '', 10);
      if (!isNaN(savedSize) && savedSize >= 110 && savedSize <= 340) {
        setCardSize(savedSize);
      }
      const savedDesc = localStorage.getItem('e6.posts.show_desc');
      if (savedDesc !== null) setShowDesc(savedDesc === 'true');
    } catch {}

    loadConfig().then((cfg) => setConfig(cfg)).catch(() => {});
  }, []);

  const loadProfile = async () => {
    setProfileLoading(true);
    try {
      const data = await TiresiasApi.getUserProfile();
      setProfile(data);
    } catch (e: any) {
      console.warn('[Tiresias] Could not load profile stats:', e);
    } finally {
      setProfileLoading(false);
    }
  };

  const loadHistory = async (tab: 'like' | 'dislike', newOffset = 0, append = false) => {
    setLoading(true);
    setError(null);
    try {
      const res = await TiresiasApi.getFeedbackHistory(tab, 50, newOffset);
      const rawList = res.items || [];
      const pids = rawList.map((x: any) => x.post_id).filter(Boolean);

      // Fast booru metadata enrichment for preview images
      if (pids.length > 0) {
        try {
          const booruResp = await fetch(`/posts.json?tags=id:${pids.join(',')}`);
          if (booruResp.ok) {
            const bData = await booruResp.json();
            const map = new Map<number, any>();
            for (const bp of (bData.posts || [])) map.set(bp.id, bp);
            for (const it of rawList) {
              const bp = map.get(it.post_id);
              if (bp) {
                it.preview_url = bp.preview?.url || it.preview_url;
                it.score = bp.score?.total ?? it.score;
                it.rating = bp.rating ?? it.rating;
                it.fav_count = bp.fav_count ?? it.fav_count;
              }
            }
          }
        } catch (err) {
          console.warn('[Tiresias] Could not enrich feedback preview:', err);
        }
      }

      if (append) {
        setItems((prev) => [...prev, ...rawList]);
      } else {
        setItems(rawList);
      }
      setTotal(res.total || 0);
      setOffset(newOffset);
    } catch (e: any) {
      setError(e.message || t('activity.loadError'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProfile();
  }, []);

  useEffect(() => {
    loadHistory(activeTab, 0, false);
  }, [activeTab]);

  const handleUndo = async (item: ActivityItem) => {
    try {
      const action = item.signal_type === 'like' ? 'like' : 'hide';
      await TiresiasApi.undoFeedback(item.post_id, action);
      setItems((prev) => prev.filter((x) => x.post_id !== item.post_id));
      setTotal((prev) => Math.max(0, prev - 1));
      loadProfile();
    } catch (e: any) {
      toast.error(t('activity.undoError', { error: e.message || 'Unknown' }));
    }
  };

  return (
    <div className="tiresias-page-wrapper">
      {/* Header Bar */}
      <div className="tiresias-header-bar">
        <div className="tiresias-header-title">
          <span>📊</span>
          <span>{t('activity.title')}</span>
        </div>
        <Button
          variant="secondary"
          onClick={() => {
            loadProfile();
            loadHistory(activeTab, 0, false);
          }}
          disabled={loading || profileLoading}
        >
          {t('activity.refreshBtn')}
        </Button>
      </div>

      {/* Profile Stats Card */}
      <div style={{
        background: 'var(--tiresias-card)',
        border: '1px solid var(--tiresias-border)',
        borderRadius: 'var(--tiresias-radius-lg)',
        padding: 18,
        marginBottom: 20,
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 14 }}>
          <div>
            <h3 style={{ margin: '0 0 6px 0', color: 'var(--tiresias-gold)', fontSize: 16 }}>
              {t('activity.cardTitle')}
            </h3>
            <div style={{ fontSize: 12, color: 'var(--tiresias-text-muted)' }}>
              {t('activity.cardSubtitle')}
            </div>
          </div>

          {profile && (
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <div style={{ background: 'var(--tiresias-bg-surface-dark)', border: '1px solid var(--tiresias-border)', borderRadius: 'var(--tiresias-radius-md)', padding: '6px 12px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)' }}>{t('activity.statLiked')}</div>
                <div style={{ fontSize: 15, fontWeight: 'bold', color: 'var(--tiresias-success-light)' }}>
                  {profile.likes_count ?? profile.feedback_counts?.like ?? 0}
                </div>
              </div>

              <div style={{ background: 'var(--tiresias-bg-surface-dark)', border: '1px solid var(--tiresias-border)', borderRadius: 'var(--tiresias-radius-md)', padding: '6px 12px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)' }}>{t('activity.statHidden')}</div>
                <div style={{ fontSize: 15, fontWeight: 'bold', color: 'var(--tiresias-danger)' }}>
                  {profile.hides_count ?? profile.feedback_counts?.dislike ?? 0}
                </div>
              </div>

              <div style={{ background: 'var(--tiresias-bg-surface-dark)', border: '1px solid var(--tiresias-border)', borderRadius: 'var(--tiresias-radius-md)', padding: '6px 12px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)' }}>{t('activity.statSeen')}</div>
                <div style={{ fontSize: 15, fontWeight: 'bold', color: 'var(--tiresias-gold)' }}>
                  {profile.seen_count ?? 0}
                </div>
              </div>

              <div style={{ background: 'var(--tiresias-bg-surface-dark)', border: '1px solid var(--tiresias-border)', borderRadius: 'var(--tiresias-radius-md)', padding: '6px 12px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)' }}>{t('activity.statApproval')}</div>
                <div style={{ fontSize: 15, fontWeight: 'bold', color: 'var(--tiresias-info)' }}>
                  {profile.approval_rate !== undefined ? `${profile.approval_rate}%` : '—'}
                </div>
              </div>

              <div style={{ background: 'var(--tiresias-bg-surface-dark)', border: '1px solid var(--tiresias-border)', borderRadius: 'var(--tiresias-radius-md)', padding: '6px 12px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)' }}>{t('activity.statAvgScore')}</div>
                <div style={{ fontSize: 15, fontWeight: 'bold', color: 'var(--tiresias-gold)' }}>
                  {profile.avg_liked_score !== undefined ? profile.avg_liked_score : '—'}
                </div>
              </div>

              {/* Advanced math details only for testers and admins */}
              {config?.authUser && config.authUser.role >= 20 && (
                <>
                  <div style={{ background: 'var(--tiresias-bg-surface-dark)', border: '1px solid var(--tiresias-tester-cyan)', borderRadius: 'var(--tiresias-radius-md)', padding: '6px 12px', textAlign: 'center' }}>
                    <div style={{ fontSize: 11, color: 'var(--tiresias-tester-cyan)' }}>{t('activity.statArchetype')}</div>
                    <div style={{ fontSize: 15, fontWeight: 'bold', color: 'var(--tiresias-gold)' }}>
                      #{profile.effective_archetype_id ?? profile.taste_archetype_id ?? 27}
                      {profile.locked_archetype !== null && profile.locked_archetype !== undefined && (
                        <span style={{ fontSize: 10, color: 'var(--tiresias-danger-light)', marginLeft: 4 }}>🔒</span>
                      )}
                    </div>
                  </div>

                  <div style={{ background: 'var(--tiresias-bg-surface-dark)', border: '1px solid var(--tiresias-tester-cyan)', borderRadius: 'var(--tiresias-radius-md)', padding: '6px 12px', textAlign: 'center' }}>
                    <div style={{ fontSize: 11, color: 'var(--tiresias-tester-cyan)' }}>{t('activity.statCoherence')}</div>
                    <div style={{ fontSize: 15, fontWeight: 'bold', color: 'var(--tiresias-success-light)' }}>
                      {Math.round((profile.taste_coherence ?? 0) * 100)}%
                    </div>
                  </div>

                  <div style={{ background: 'var(--tiresias-bg-surface-dark)', border: '1px solid var(--tiresias-tester-cyan)', borderRadius: 'var(--tiresias-radius-md)', padding: '6px 12px', textAlign: 'center' }}>
                    <div style={{ fontSize: 11, color: 'var(--tiresias-tester-cyan)' }}>{t('activity.statInfluence')}</div>
                    <div style={{ fontSize: 15, fontWeight: 'bold', color: 'var(--tiresias-info)' }}>
                      {Math.round((profile.archetype_influence ?? 1) * 100)}%
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Rating Distribution Bar */}
        {profile?.rating_distribution && (
          <div style={{ marginTop: 14, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--tiresias-text-muted)', marginBottom: 5 }}>
              <span>{t('activity.ratingDistTitle')}</span>
              <span>
                <span style={{ color: 'var(--tiresias-rating-s)' }}>Safe: {profile.rating_distribution.s_pct}%</span> ({profile.rating_distribution.s}) |{' '}
                <span style={{ color: 'var(--tiresias-rating-q)' }}>Quest: {profile.rating_distribution.q_pct}%</span> ({profile.rating_distribution.q}) |{' '}
                <span style={{ color: 'var(--tiresias-rating-e)' }}>Explicit: {profile.rating_distribution.e_pct}%</span> ({profile.rating_distribution.e})
              </span>
            </div>
            <div style={{ display: 'flex', height: 7, borderRadius: 'var(--tiresias-radius-sm)', overflow: 'hidden', background: 'var(--tiresias-bg-surface-dark)' }}>
              <div style={{ width: `${profile.rating_distribution.s_pct}%`, background: 'var(--tiresias-rating-s)' }} title={`Safe: ${profile.rating_distribution.s_pct}%`} />
              <div style={{ width: `${profile.rating_distribution.q_pct}%`, background: 'var(--tiresias-rating-q)' }} title={`Questionable: ${profile.rating_distribution.q_pct}%`} />
              <div style={{ width: `${profile.rating_distribution.e_pct}%`, background: 'var(--tiresias-rating-e)' }} title={`Explicit: ${profile.rating_distribution.e_pct}%`} />
            </div>
          </div>
        )}

        {/* Top Artists */}
        {profile?.top_artists && profile.top_artists.length > 0 && (
          <div style={{ marginTop: 12, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10 }}>
            <span style={{ fontSize: 11, color: 'var(--tiresias-tag-artist)', marginRight: 8, fontWeight: 'bold' }}>
              {t('activity.topArtists')}
            </span>
            <div style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 6 }}>
              {profile.top_artists.map((tagItem: any) => (
                <Badge key={tagItem.tag} variant="artist">
                  {tagItem.tag} ({tagItem.post_count})
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Top Characters & Species */}
        {profile?.top_characters && profile.top_characters.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <span style={{ fontSize: 11, color: 'var(--tiresias-tag-character)', marginRight: 8, fontWeight: 'bold' }}>
              {t('activity.topCharacters')}
            </span>
            <div style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 6 }}>
              {profile.top_characters.map((tagItem: any) => (
                <Badge key={tagItem.tag} variant="character">
                  {tagItem.tag} ({tagItem.post_count})
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Top Motifs / Topics */}
        {((profile?.top_motifs && profile.top_motifs.length > 0) || (profile?.top_tag_likes && profile.top_tag_likes.length > 0)) && (
          <div style={{ marginTop: 10 }}>
            <span style={{ fontSize: 11, color: 'var(--tiresias-accent)', marginRight: 8, fontWeight: 'bold' }}>
              {t('activity.topMotifs')}
            </span>
            <div style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 6 }}>
              {(profile.top_motifs || profile.top_tag_likes).slice(0, 14).map((tagItem: any) => (
                <Badge key={tagItem.tag} variant="general">
                  {tagItem.tag} ({tagItem.post_count ?? tagItem.likes ?? 1})
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="tiresias-tabs">
        <button
          type="button"
          className={`tiresias-tab ${activeTab === 'like' ? 'active' : ''}`}
          onClick={() => setActiveTab('like')}
        >
          {t('activity.tabLiked', { count: activeTab === 'like' ? total : profile?.feedback_counts?.like ?? '...' })}
        </button>
        <button
          type="button"
          className={`tiresias-tab ${activeTab === 'dislike' ? 'active' : ''}`}
          onClick={() => setActiveTab('dislike')}
        >
          {t('activity.tabDisliked', { count: activeTab === 'dislike' ? total : profile?.feedback_counts?.dislike ?? '...' })}
        </button>
      </div>

      {/* History Grid */}
      {error && (
        <div style={{ padding: 14, background: 'var(--tiresias-danger-subtle)', border: '1px solid var(--tiresias-danger)', borderRadius: 'var(--tiresias-radius-md)', marginBottom: 16 }}>
          {error}
        </div>
      )}

      {loading && items.length === 0 ? (
        <div style={{ textAlign: 'center', padding: 40, color: 'var(--tiresias-gold)' }}>
          {t('activity.loadingHistory')}
        </div>
      ) : items.length === 0 ? (
        <div style={{ textAlign: 'center', padding: 50, color: 'var(--tiresias-text-muted)' }}>
          {activeTab === 'like'
            ? t('activity.emptyLiked')
            : t('activity.emptyDisliked')}
        </div>
      ) : (
        <div>
          {/* Unified e621 Thumbnail Grid */}
          <section
            className={`posts-container tiresias-posts-container ${!showDesc ? 'no-stats' : ''}`}
            style={{ '--thumb-image-size': `${cardSize}px` } as any}
            data-st-contain={imageContain ? 'true' : 'false'}
            data-st-show-desc={showDesc ? 'true' : 'false'}
          >
            {items.map((item) => {
              const pid = item.post_id;
              const host = typeof window !== 'undefined' ? window.location.host : 'e926.net';
              const previewUrl = item.preview_url;
              const scoreVal = item.score ?? 0;
              return (
                <article
                  key={pid}
                  className="thumbnail post-thumbnail tiresias-post-thumb"
                  data-id={pid}
                  data-score={scoreVal}
                  data-rating={item.rating || 's'}
                  data-context="activity"
                >
                  <a href={`https://${host}/posts/${pid}`} target="_blank" rel="noreferrer" className="thm-link">
                    {item.is_video && <div className="tiresias-video-badge">🎬</div>}
                    {previewUrl ? (
                      <img src={previewUrl} alt={`Post #${pid}`} loading="lazy" />
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
                      </span>
                      <span className="thm-desc-b thm-rating" title={`${t('common.rating')}: ${(item.rating || 's').toUpperCase()}`}>
                        {(item.rating || 's').toUpperCase()}
                      </span>
                    </div>
                  )}

                  <div style={{ marginTop: 4, width: '100%' }}>
                    <Button
                      variant="secondary"
                      size="sm"
                      style={{ width: '100%', padding: '3px 0', fontSize: 11 }}
                      onClick={() => handleUndo(item)}
                      title={activeTab === 'like' ? t('activity.removeLikeTitle') : t('activity.restoreDislikeTitle')}
                    >
                      {activeTab === 'like' ? t('activity.removeLike') : t('activity.restoreDislike')}
                    </Button>
                  </div>
                </article>
              );
            })}
          </section>

          {/* Load More Button */}
          {offset + items.length < total && (
            <div style={{ textAlign: 'center', marginTop: 24 }}>
              <Button
                variant="primary"
                onClick={() => loadHistory(activeTab, offset + 50, true)}
                disabled={loading}
                style={{ padding: '8px 24px', fontSize: 13 }}
              >
                {loading ? t('common.loading') : t('activity.loadMore', { count: items.length, total })}
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Similar Drawer */}
      {similarPostId && (
        <SimilarDrawer
          postId={similarPostId}
          onClose={() => setSimilarPostId(null)}
        />
      )}
    </div>
  );
}

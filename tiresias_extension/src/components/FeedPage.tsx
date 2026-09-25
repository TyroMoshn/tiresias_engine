import { h, Fragment } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { TiresiasApi } from '../lib/api';
import { PostMetadata, ServerConfig } from '../lib/types';
import { loadConfig } from '../lib/storage';
import { Button, IconButton, Input, Toggle, Badge, toast } from './ui';
import { useTranslation } from '../lib/i18n';

export function FeedPage() {
  const { t } = useTranslation();
  const [posts, setPosts] = useState<PostMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isMaintenance, setIsMaintenance] = useState(false);
  const [config, setConfig] = useState<ServerConfig | null>(null);

  // Layout Settings (Native e621ng style)
  const [imageContain, setImageContain] = useState(false);
  const [cardSize, setCardSize] = useState(180);
  const [showDesc, setShowDesc] = useState(true);

  // Algorithm Settings
  const [minScore, setMinScore] = useState(10);
  const [exploration, setExploration] = useState(0.2);
  const [ratings, setRatings] = useState<string[]>(['s', 'q']);
  const [mediaTypes, setMediaTypes] = useState<string[]>(['image', 'video']);

  // User profile stats
  const [userProfile, setUserProfile] = useState<any>(null);

  // Active floating popover: 'none' | 'layout' | 'algorithm' | 'stats'
  const [activePopover, setActivePopover] = useState<'none' | 'layout' | 'algorithm' | 'stats'>('none');

  const sentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleDocClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (
        !target.closest('#layout-settings-container') &&
        !target.closest('#layout-settings-open') &&
        !target.closest('#advanced-search-container') &&
        !target.closest('#advanced-search-open') &&
        !target.closest('#stats-popover-container') &&
        !target.closest('#stats-popover-open')
      ) {
        setActivePopover('none');
      }
    };
    document.addEventListener('mousedown', handleDocClick);
    return () => document.removeEventListener('mousedown', handleDocClick);
  }, []);

  useEffect(() => {
    try {
      const savedContain = localStorage.getItem('e6.posts.contain') === 'true';
      setImageContain(savedContain);
      document.body.setAttribute('data-st-contain', savedContain ? 'true' : 'false');

      const savedSize = parseInt(localStorage.getItem('e6.posts.custom_size') || '', 10);
      if (!isNaN(savedSize) && savedSize >= 110 && savedSize <= 340) {
        setCardSize(savedSize);
      }

      const savedDesc = localStorage.getItem('e6.posts.show_desc');
      if (savedDesc !== null) {
        setShowDesc(savedDesc === 'true');
      }
    } catch {}

    loadConfig().then((cfg) => setConfig(cfg)).catch(() => {});

    TiresiasApi.getUserProfile()
      .then((p) => setUserProfile(p))
      .catch(() => {});
  }, []);

  const setContainMode = (contain: boolean) => {
    setImageContain(contain);
    try {
      localStorage.setItem('e6.posts.contain', contain ? 'true' : 'false');
      document.body.setAttribute('data-st-contain', contain ? 'true' : 'false');
      window.dispatchEvent(new CustomEvent('tiresias:view_settings_changed', { detail: { contain } }));
    } catch {}
  };

  const updateCardSize = (size: number) => {
    setCardSize(size);
    try {
      localStorage.setItem('e6.posts.custom_size', String(size));
      if (size <= 150) localStorage.setItem('e6.posts.size', 's');
      else if (size <= 220) localStorage.setItem('e6.posts.size', 'm');
      else localStorage.setItem('e6.posts.size', 'l');
      window.dispatchEvent(new CustomEvent('tiresias:view_settings_changed', { detail: { cardSize: size } }));
    } catch {}
  };

  const toggleShowDesc = (show: boolean) => {
    setShowDesc(show);
    try {
      localStorage.setItem('e6.posts.show_desc', show ? 'true' : 'false');
      window.dispatchEvent(new CustomEvent('tiresias:view_settings_changed', { detail: { showDesc: show } }));
    } catch {}
  };

  const loadFeed = async (newCursor = 0, append = false) => {
    if (loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await TiresiasApi.getFeed({
        cursor: newCursor,
        limit: 30,
        min_score: minScore,
        ratings,
        media_types: mediaTypes,
        exploration_weight: exploration,
      });

      const rawItems = res.items || [];
      const newItems: PostMetadata[] = rawItems.map((it: any) => ({
        ...it,
        id: it.id ?? it.post_id,
        score_val: it.score_val ?? Math.round(it.score * 100),
      }));

      // Enrich with official preview URLs and stats from host booru in a fast batch
      const pids = newItems.map((it) => it.id).filter(Boolean);
      if (pids.length > 0) {
        try {
          const booruResp = await fetch(`/posts.json?tags=id:${pids.join(',')}`);
          if (booruResp.ok) {
            const booruData = await booruResp.json();
            const metaMap = new Map<number, any>();
            for (const bp of (booruData.posts || [])) {
              metaMap.set(bp.id, bp);
            }
            for (const it of newItems) {
              const bp = metaMap.get(it.id);
              if (bp) {
                it.preview_url = bp.preview?.url || it.preview_url;
                it.score_val = bp.score?.total ?? it.score_val;
                it.fav_count = bp.fav_count ?? it.fav_count;
                it.comment_count = bp.comment_count ?? (it as any).comment_count;
                it.rating = bp.rating ?? it.rating;
              }
            }
          }
        } catch (err) {
          console.warn('[Tiresias] Could not fetch booru metadata:', err);
        }
      }

      if (append) {
        setPosts((prev) => {
          const existingIds = new Set(prev.map((p) => p.id));
          const uniqueItems = newItems.filter((p) => !existingIds.has(p.id));
          return [...prev, ...uniqueItems];
        });
      } else {
        setPosts(newItems);
      }
      setCursor(res.next_cursor || newCursor + newItems.length);
      setHasMore(res.has_more ?? newItems.length >= 30);
      setIsMaintenance(false);
    } catch (e: any) {
      if (e?.isMaintenance) {
        setIsMaintenance(true);
        setError(null);
      } else {
        setIsMaintenance(false);
        setError(e.message || t('feed.errorLoading'));
        toast.error(e.message || t('feed.errorLoading'));
      }
      setHasMore(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFeed(0, false);
  }, [minScore, exploration, ratings, mediaTypes]);

  // Infinite scroll intersection observer: automatically triggers next page
  useEffect(() => {
    if (!sentinelRef.current) return;
    if (loading || !hasMore || !!error || isMaintenance) return;
    if (posts.length >= 300) return; // Strict safety threshold

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !loading && hasMore && posts.length < 300) {
          loadFeed(cursor, true);
        }
      },
      { rootMargin: '300px' }
    );

    observer.observe(sentinelRef.current);
    return () => observer.disconnect();
  }, [cursor, loading, hasMore, error, isMaintenance, posts.length]);

  const toggleRating = (r: string) => {
    if (ratings.includes(r)) {
      if (ratings.length > 1) {
        setRatings(ratings.filter((x) => x !== r));
      }
    } else {
      setRatings([...ratings, r]);
    }
  };

  const toggleMediaType = (mt: string) => {
    if (mediaTypes.includes(mt)) {
      if (mediaTypes.length > 1) {
        setMediaTypes(mediaTypes.filter((x) => x !== mt));
      }
    } else {
      setMediaTypes([...mediaTypes, mt]);
    }
  };

  return (
    <div className="tiresias-page-wrapper" style={{ position: 'relative' }}>
      {/* Top Header Bar */}
      <div className="tiresias-header-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="tiresias-header-title">
            <span>🔮</span>
            <span>{t('feed.title')}</span>
          </div>
        </div>

        {/* e621ng Native Search Controls Bar */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <div className="search-controls" style={{ display: 'flex', gap: 4 }}>
            <IconButton
              id="layout-settings-open"
              icon={
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect width="7" height="7" x="14" y="3" rx="1" />
                  <rect width="7" height="7" x="14" y="14" rx="1" />
                  <rect width="7" height="18" x="3" y="3" rx="1" />
                </svg>
              }
              active={activePopover === 'layout'}
              title={t('feed.layoutSettingsTitle')}
              onClick={() => setActivePopover(activePopover === 'layout' ? 'none' : 'layout')}
            />
            <IconButton
              id="advanced-search-open"
              icon={
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10 18H21M3 18H6M6 18V20M6 18V16M20 12H21M3 12H16M16 12V14M16 12V10M14 6H21M3 6H10M10 6V8M10 6V4" />
                </svg>
              }
              active={activePopover === 'algorithm'}
              title={t('feed.algorithmSettingsTitle')}
              onClick={() => setActivePopover(activePopover === 'algorithm' ? 'none' : 'algorithm')}
            />
            <IconButton
              id="stats-popover-open"
              icon={
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="20" x2="18" y2="10" />
                  <line x1="12" y1="20" x2="12" y2="4" />
                  <line x1="6" y1="20" x2="6" y2="14" />
                </svg>
              }
              active={activePopover === 'stats'}
              title={t('feed.statsTitle')}
              onClick={() => setActivePopover(activePopover === 'stats' ? 'none' : 'stats')}
            />
          </div>

          <Button
            variant="primary"
            onClick={() => loadFeed(0, false)}
            disabled={loading}
            title={t('feed.refreshTitle')}
          >
            {t('feed.refreshBtn')}
          </Button>
        </div>
      </div>

      {/* Floating Layout Settings Popover (#layout-settings-container) */}
      <div
        id="layout-settings-container"
        className={`tiresias-popover-container ${activePopover === 'layout' ? 'active' : ''}`}
      >
        <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>{t('feed.layoutSettingsTitle')}</span>
          <IconButton
            id="layout-settings-close"
            icon="✕"
            title={t('feed.closeSettings')}
            size="sm"
            onClick={() => setActivePopover('none')}
          />
        </h3>

        {/* Image Crop */}
        <div className="ssc-entry" style={{ alignItems: 'center' }}>
          <div className="ssc-label">
            <label>{t('feed.crop')}</label>
          </div>
          <div className="ssc-body">
            <Toggle
              type="pill"
              checked={imageContain}
              onChange={setContainMode}
              offText="Crop"
              onText="Full"
              title={t('feed.cropTooltip')}
            />
          </div>
        </div>

        {/* Card Size Presets */}
        <div className="ssc-entry">
          <div className="ssc-label">
            <label>{t('feed.cardSize')}</label>
            <span style={{ fontSize: '0.8rem', color: 'var(--tiresias-gold)', marginLeft: 6, fontWeight: 'bold' }}>{cardSize}px</span>
          </div>
          <div className="ssc-body stm-toggle" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
            <input
              type="radio"
              id="ssc-card-small"
              name="ssc-card-size"
              checked={cardSize <= 150}
              onChange={() => updateCardSize(140)}
            />
            <label htmlFor="ssc-card-small" title="Small (140px)">
              {t('feed.cardSmall')}
            </label>

            <input
              type="radio"
              id="ssc-card-medium"
              name="ssc-card-size"
              checked={cardSize > 150 && cardSize <= 220}
              onChange={() => updateCardSize(190)}
            />
            <label htmlFor="ssc-card-medium" title="Medium (190px)">
              {t('feed.cardMedium')}
            </label>

            <input
              type="radio"
              id="ssc-card-large"
              name="ssc-card-size"
              checked={cardSize > 220}
              onChange={() => updateCardSize(250)}
            />
            <label htmlFor="ssc-card-large" title="Large (250px)">
              {t('feed.cardLarge')}
            </label>
          </div>
        </div>

        {/* Continuous Range Slider */}
        <div className="ssc-entry" style={{ gridTemplateColumns: '1fr', marginTop: '-0.25rem' }}>
          <input
            type="range"
            min="110"
            max="340"
            step="5"
            value={cardSize}
            onInput={(e) => updateCardSize(parseInt((e.target as HTMLInputElement).value, 10))}
            style={{ width: '100%', cursor: 'pointer' }}
          />
        </div>

        {/* Post Information (Score, Faves, Rating) */}
        <div className="ssc-entry" style={{ alignItems: 'center' }}>
          <div className="ssc-label">
            <label>{t('feed.postInfo')}</label>
          </div>
          <div className="ssc-body">
            <Toggle
              type="switch"
              checked={showDesc}
              onChange={toggleShowDesc}
              title={t('feed.postInfoTooltip')}
            />
          </div>
        </div>
      </div>

      {/* Floating Algorithm Settings Popover (#advanced-search-container) */}
      <div
        id="advanced-search-container"
        className={`tiresias-popover-container ${activePopover === 'algorithm' ? 'active' : ''}`}
      >
        <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>{t('feed.algorithmSettingsTitle')}</span>
          <IconButton
            id="advanced-search-close"
            icon="✕"
            title={t('feed.closeSettings')}
            size="sm"
            onClick={() => setActivePopover('none')}
          />
        </h3>

        {/* Rating */}
        <div className="ssc-entry">
          <div className="ssc-label">
            <label>{t('feed.rating')}</label>
          </div>
          <div className="ssc-body stm-toggle" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
            <input
              type="checkbox"
              id="ssc-rating-s"
              value="s"
              checked={ratings.includes('s')}
              onChange={() => toggleRating('s')}
            />
            <label htmlFor="ssc-rating-s" title={t('feed.ratingSafeTooltip')}>
              {t('feed.ratingSafe')}
            </label>

            <input
              type="checkbox"
              id="ssc-rating-q"
              value="q"
              checked={ratings.includes('q')}
              onChange={() => toggleRating('q')}
            />
            <label htmlFor="ssc-rating-q" title={t('feed.ratingQuestTooltip')}>
              {t('feed.ratingQuest')}
            </label>

            <input
              type="checkbox"
              id="ssc-rating-e"
              value="e"
              checked={ratings.includes('e')}
              onChange={() => toggleRating('e')}
            />
            <label htmlFor="ssc-rating-e" title={t('feed.ratingExplTooltip')}>
              {t('feed.ratingExpl')}
            </label>
          </div>
        </div>

        {/* Media Types */}
        <div className="ssc-entry">
          <div className="ssc-label">
            <label>{t('feed.mediaType')}</label>
          </div>
          <div className="ssc-body stm-toggle" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <input
              type="checkbox"
              id="ssc-media-image"
              value="image"
              checked={mediaTypes.includes('image')}
              onChange={() => toggleMediaType('image')}
            />
            <label htmlFor="ssc-media-image">
              {t('feed.mediaImages')}
            </label>

            <input
              type="checkbox"
              id="ssc-media-video"
              value="video"
              checked={mediaTypes.includes('video')}
              onChange={() => toggleMediaType('video')}
            />
            <label htmlFor="ssc-media-video">
              {t('feed.mediaVideos')}
            </label>
          </div>
        </div>

        {/* Exploration */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginTop: '0.35rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
            <span className="ssc-label" style={{ fontWeight: 'bold' }}>{t('feed.exploration')}</span>
            <span style={{ color: 'var(--tiresias-gold)', fontWeight: 'bold' }}>{Math.round(exploration * 100)}%</span>
          </div>
          <div style={{ width: '100%' }}>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={exploration}
              onInput={(e) => setExploration(parseFloat((e.target as HTMLInputElement).value))}
              style={{ width: '100%', cursor: 'pointer', margin: '0 0 2px 0', display: 'block' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--tiresias-text-muted)', padding: '0 1px' }}>
              <span>{t('feed.explorationAccurate')}</span>
              <span>{t('feed.explorationBalance')}</span>
              <span>{t('feed.explorationNovel')}</span>
            </div>
          </div>
        </div>

        {/* Min Score */}
        <div className="ssc-entry" style={{ gridTemplateColumns: '1fr', gap: '0.25rem', marginTop: '0.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', alignItems: 'center' }}>
            <span className="ssc-label">{t('feed.minScore')}</span>
            <Input
              type="number"
              size="sm"
              style={{ width: '75px', textAlign: 'right' }}
              value={minScore}
              onChange={(e: any) => setMinScore(parseInt(e.target.value, 10) || 0)}
            />
          </div>
        </div>

        {/* Apply Button */}
        <div style={{ marginTop: '0.35rem' }}>
          <Button
            variant="primary"
            style={{ width: '100%', justifyContent: 'center', fontSize: '0.85rem' }}
            onClick={() => {
              setActivePopover('none');
              loadFeed(0, false);
            }}
          >
            {t('feed.applyAndRefresh')}
          </Button>
        </div>
      </div>

      {/* Floating User Stats Popover (#stats-popover-container) */}
      <div
        id="stats-popover-container"
        className={`tiresias-popover-container ${activePopover === 'stats' ? 'active' : ''}`}
      >
        <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>{t('feed.statsTitle')}</span>
          <IconButton
            id="stats-popover-close"
            icon="✕"
            title={t('feed.closeSettings')}
            size="sm"
            onClick={() => setActivePopover('none')}
          />
        </h3>

        {userProfile ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {/* Counts */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, textAlign: 'center' }}>
              <div style={{ background: 'var(--tiresias-bg-surface-dark)', padding: '6px 4px', borderRadius: 'var(--tiresias-radius-sm)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--tiresias-text-muted)' }}>{t('feed.statsLikes')}</div>
                <div style={{ fontWeight: 'bold', fontSize: '1rem', color: 'var(--tiresias-success-light)' }}>{userProfile.likes_count ?? 0}</div>
              </div>
              <div style={{ background: 'var(--tiresias-bg-surface-dark)', padding: '6px 4px', borderRadius: 'var(--tiresias-radius-sm)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--tiresias-text-muted)' }}>{t('feed.statsHides')}</div>
                <div style={{ fontWeight: 'bold', fontSize: '1rem', color: 'var(--tiresias-danger)' }}>{userProfile.hides_count ?? 0}</div>
              </div>
              <div style={{ background: 'var(--tiresias-bg-surface-dark)', padding: '6px 4px', borderRadius: 'var(--tiresias-radius-sm)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--tiresias-text-muted)' }}>{t('feed.statsViews')}</div>
                <div style={{ fontWeight: 'bold', fontSize: '1rem', color: 'var(--tiresias-gold)' }}>{userProfile.seen_count ?? 0}</div>
              </div>
            </div>

            {/* Tester / Admin details */}
            {config?.authUser && config.authUser.role >= 20 && (
              <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                  <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('feed.tasteArchetype')}</span>
                  <span style={{ color: 'var(--tiresias-gold)', fontWeight: 'bold' }}>
                    #{userProfile.effective_archetype_id ?? userProfile.taste_archetype_id ?? 27}
                    {userProfile.locked_archetype !== null && (
                      <span style={{ fontSize: '0.65rem', marginLeft: 4, background: 'var(--tiresias-danger)', color: '#fff', padding: '1px 4px', borderRadius: 2 }}>{t('feed.archetypeFixed')}</span>
                    )}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                  <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('feed.coherence')}</span>
                  <span style={{ color: 'var(--tiresias-success-light)', fontWeight: 'bold' }}>{Math.round((userProfile.taste_coherence ?? 0) * 100)}%</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                  <span style={{ color: 'var(--tiresias-text-muted)' }}>{t('feed.archetypeInfluence')}</span>
                  <span style={{ color: 'var(--tiresias-info)', fontWeight: 'bold' }}>{Math.round((userProfile.archetype_influence ?? 1) * 100)}%</span>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
              <a
                href="/activity"
                className="tiresias-button"
                style={{ flex: 1, fontSize: '0.8rem', justifyContent: 'center' }}
              >
                {t('feed.activityLog')}
              </a>
              {config?.authUser && config.authUser.role >= 20 && (
                <a
                  href="/tiresias/settings"
                  className="tiresias-button"
                  style={{ fontSize: '0.8rem', padding: '6px 10px' }}
                  title={t('feed.testerSettings')}
                >
                  ⚙️
                </a>
              )}
            </div>
          </div>
        ) : (
          <div style={{ fontSize: '0.8rem', color: 'var(--tiresias-text-muted)' }}>{t('feed.loadingStats')}</div>
        )}
      </div>

      {/* Content Area with Unified e621 Thumbnail Grid */}
      <div className="tiresias-content-area" style={{ width: '100%' }}>
        {isMaintenance && (
          <div style={{
            padding: 14,
            background: 'rgba(232, 196, 70, 0.15)',
            border: '1px solid var(--tiresias-gold)',
            borderRadius: 'var(--tiresias-radius-md)',
            marginBottom: 16,
            display: 'flex',
            alignItems: 'center',
            gap: 12,
          }}>
            <span style={{ fontSize: 24 }}>🛠️</span>
            <div>
              <div style={{ fontWeight: 'bold', color: 'var(--tiresias-gold)', fontSize: 13 }}>
                {t('feed.maintenanceTitle')}
              </div>
              <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)', marginTop: 2 }}>
                {t('feed.maintenanceSubtitle')}
              </div>
            </div>
          </div>
        )}

        {error && (
          <div style={{
            padding: 14,
            background: 'var(--tiresias-danger-subtle)',
            border: '1px solid var(--tiresias-danger)',
            borderRadius: 'var(--tiresias-radius-md)',
            marginBottom: 16,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
          }}>
            <div>
              <strong style={{ color: 'var(--tiresias-danger)' }}>{t('common.serverError')}: </strong>
              <span>{error}</span>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setHasMore(true);
                loadFeed(0, false);
              }}
            >
              {t('common.retry')}
            </Button>
          </div>
        )}

        {/* Unified Grid */}
        <section
          className={`posts-container tiresias-posts-container ${!showDesc ? 'no-stats' : ''}`}
          id="posts"
          style={{ '--thumb-image-size': `${cardSize}px` } as any}
          data-st-contain={imageContain ? 'true' : 'false'}
          data-st-show-desc={showDesc ? 'true' : 'false'}
        >
          {posts.map((p) => {
            const pid = p.id ?? (p as any).post_id;
            const host = typeof window !== 'undefined' ? window.location.host : 'e926.net';
            const previewUrl = p.preview_url;
            const scoreVal = p.score_val ?? Math.round(p.score * 100);

            return (
              <article
                key={pid}
                className="thumbnail post-thumbnail tiresias-post-thumb"
                data-id={pid}
                data-score={scoreVal}
                data-rating={p.rating}
                data-file-ext={p.file_ext || ''}
                data-is-video={p.is_video ? 'true' : 'false'}
                data-reason={p.reason || ''}
                data-context="feed"
              >
                <a
                  href={`https://${host}/posts/${pid}`}
                  target="_blank"
                  rel="noreferrer"
                  className="thm-link"
                >
                  {p.is_video && (
                    <div className="tiresias-video-badge" title={`${t('common.video')} (${p.file_ext || 'video'})`}>
                      🎬
                    </div>
                  )}
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt={`Post #${pid}`}
                      loading="lazy"
                    />
                  ) : (
                    <div className="tiresias-thumb-placeholder">
                      #{pid}
                    </div>
                  )}
                </a>

                {/* Native e621ng Post Info Bar */}
                {showDesc && (
                  <div className={`desc thm-desc thm-rating-${p.rating || 's'}`}>
                    <span className="thm-desc-a">
                      <span className="thm-desc-m thm-score" title={t('common.score')}>
                        ▲ {scoreVal}
                      </span>
                      <span className="thm-desc-m thm-favorites" title={t('common.favorites')}>
                        ★ {p.fav_count ?? 0}
                      </span>
                      {p.comment_count !== undefined && (
                        <span className="thm-desc-m thm-comments" title={t('common.comments')}>
                          💬 {p.comment_count}
                        </span>
                      )}
                    </span>
                    <span className="thm-desc-b thm-rating" title={`${t('common.rating')}: ${(p.rating || 's').toUpperCase()}`}>
                      {(p.rating || 's').toUpperCase()}
                    </span>
                  </div>
                )}
              </article>
            );
          })}
        </section>

        {/* Safety Threshold Button for 300+ Posts */}
        {posts.length >= 300 && hasMore && (
          <div style={{ textAlign: 'center', padding: '24px 0' }}>
            <Button
              variant="primary"
              onClick={() => loadFeed(cursor, true)}
              disabled={loading}
              style={{ padding: '10px 24px', fontSize: 14 }}
            >
              {loading ? t('common.loading') : t('feed.loadMore', { count: posts.length })}
            </Button>
          </div>
        )}

        {loading && posts.length < 300 && (
          <div style={{ textAlign: 'center', padding: 30, color: 'var(--tiresias-gold)' }}>
            {t('feed.loadingFeed')}
          </div>
        )}

        {!hasMore && posts.length > 0 && (
          <div style={{ textAlign: 'center', padding: 25, color: 'var(--tiresias-text-muted)', fontSize: 13 }}>
            {t('feed.allLoaded', { count: posts.length })}
          </div>
        )}

        {/* Infinite scroll sentinel (active until 300 posts) */}
        {posts.length < 300 && <div ref={sentinelRef} style={{ height: 20 }} />}
      </div>
    </div>
  );
}

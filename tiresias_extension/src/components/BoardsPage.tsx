import { h, Fragment } from 'preact';
import { useEffect, useState, useRef } from 'preact/hooks';
import { TiresiasApi } from '../lib/api';
import { Board, BoardDetail, PostMetadata, ServerConfig } from '../lib/types';
import { loadConfig } from '../lib/storage';
import { SimilarDrawer } from './SimilarDrawer';
import {
  Modal,
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

export function BoardsPage() {
  const { t } = useTranslation();
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [boards, setBoards] = useState<Board[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedBoardId, setSelectedBoardId] = useState<string | null>(null);
  const [boardDetail, setBoardDetail] = useState<BoardDetail | null>(null);
  const [boardDetailLoading, setBoardDetailLoading] = useState(false);

  // Sorting
  const [sortBy, setSortBy] = useState<'added_desc' | 'added_asc' | 'score_desc' | 'score_asc' | 'id_desc'>('added_desc');

  // Layout settings (synced with Feed & dynamic)
  const [imageContain, setImageContain] = useState(false);
  const [cardSize, setCardSize] = useState(180);
  const [showDesc, setShowDesc] = useState(true);

  // Recommendations based on board (placed at bottom)
  const [boardRecs, setBoardRecs] = useState<PostMetadata[]>([]);
  const [boardRecsLoading, setBoardRecsLoading] = useState(false);
  const [hasMoreRecs, setHasMoreRecs] = useState(true);
  const [isRecsMaintenance, setIsRecsMaintenance] = useState(false);
  const recsSentinelRef = useRef<HTMLDivElement>(null);
  const [addedRecIds, setAddedRecIds] = useState<Set<number>>(new Set());

  // Recommendation algorithm parameters for board
  const [boardRatings, setBoardRatings] = useState<string[]>([]);
  const [boardMinScore, setBoardMinScore] = useState<number>(0);
  const [showBoardRecsSettings, setShowBoardRecsSettings] = useState(false);

  // 2-step inline delete confirmation: stores board_id awaiting second confirmation click
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [updatingPrivacy, setUpdatingPrivacy] = useState(false);

  // Create board modal state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newIsPublic, setNewIsPublic] = useState(false);

  // Similar drawer
  const [similarPostId, setSimilarPostId] = useState<number | null>(null);

  useEffect(() => {
    const readSettings = () => {
      try {
        const savedContain = localStorage.getItem('e6.posts.contain') === 'true';
        setImageContain(savedContain);
        const savedSize = parseInt(localStorage.getItem('e6.posts.custom_size') || '', 10);
        if (!isNaN(savedSize) && savedSize >= 110 && savedSize <= 340) {
          setCardSize(savedSize);
        }
        const savedDesc = localStorage.getItem('e6.posts.show_desc');
        if (savedDesc !== null) setShowDesc(savedDesc === 'true');
      } catch {}
    };

    readSettings();
    loadConfig().then((cfg) => setConfig(cfg)).catch(() => {});
    loadBoards();

    // Check URL state for /boards?id=...
    try {
      const params = new URLSearchParams(window.location.search);
      const urlBoardId = params.get('id');
      if (urlBoardId) {
        loadBoardDetail(urlBoardId, false);
      }
    } catch {}

    const handlePopState = () => {
      try {
        const params = new URLSearchParams(window.location.search);
        const urlBoardId = params.get('id');
        if (urlBoardId) {
          loadBoardDetail(urlBoardId, false);
        } else {
          setSelectedBoardId(null);
          setBoardDetail(null);
        }
      } catch {}
    };

    const handleSettingsChanged = (e: any) => {
      if (e?.detail) {
        if (typeof e.detail.imageContain === 'boolean') setImageContain(e.detail.imageContain);
        if (typeof e.detail.showDesc === 'boolean') setShowDesc(e.detail.showDesc);
        if (typeof e.detail.cardSize === 'number') setCardSize(e.detail.cardSize);
      } else {
        readSettings();
      }
    };

    window.addEventListener('popstate', handlePopState);
    window.addEventListener('tiresias:view_settings_changed', handleSettingsChanged);
    window.addEventListener('storage', readSettings);
    return () => {
      window.removeEventListener('popstate', handlePopState);
      window.removeEventListener('tiresias:view_settings_changed', handleSettingsChanged);
      window.removeEventListener('storage', readSettings);
    };
  }, []);

  const loadBoards = async () => {
    setLoading(true);
    try {
      const list = await TiresiasApi.listBoards();
      setBoards(list);
    } catch (e: any) {
      console.error('Failed to load boards', e);
      toast.error(t('boards.openError', { error: e.message || 'Unknown' }));
    } finally {
      setLoading(false);
    }
  };

  const loadBoardDetail = async (id: string, pushHistory = true) => {
    setSelectedBoardId(id);
    if (pushHistory && typeof window !== 'undefined') {
      window.history.pushState(null, '', '/boards?id=' + encodeURIComponent(id));
    }
    setBoardDetailLoading(true);
    setBoardRecs([]);
    setHasMoreRecs(true);
    setIsRecsMaintenance(false);
    setAddedRecIds(new Set());
    setConfirmDeleteId(null);
    setShowBoardRecsSettings(false);
    try {
      const detail = await TiresiasApi.getBoardDetail(id, 'added_at', 'desc', 100);
      const rawPosts = (detail.posts || []).map((p: any) => ({ ...p, id: p.id ?? p.post_id }));
      const pids = rawPosts.map((p: any) => p.id).filter(Boolean);
      if (pids.length > 0) {
        try {
          const booruResp = await fetch(`/posts.json?tags=id:${pids.join(',')}`);
          if (booruResp.ok) {
            const bData = await booruResp.json();
            const map = new Map<number, any>();
            for (const bp of (bData.posts || [])) map.set(bp.id, bp);
            for (const it of rawPosts) {
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
          console.warn('[Tiresias] Could not enrich board posts:', err);
        }
      }
      // Normalize board summary so boardDetail.board is guaranteed to exist
      const boardSummary = (detail as any).board || {
        id: (detail as any).board_id || (detail as any).id || id,
        board_id: (detail as any).board_id || (detail as any).id || id,
        user_id: (detail as any).user_id || '',
        title: (detail as any).name || (detail as any).title || 'Без названия',
        name: (detail as any).name || (detail as any).title || 'Без названия',
        description: (detail as any).description || '',
        is_public: Boolean((detail as any).is_public),
        post_count: (detail as any).post_count ?? rawPosts.length,
      };
      if (boardSummary.is_public === undefined && (detail as any).is_public !== undefined) {
        boardSummary.is_public = Boolean((detail as any).is_public);
      }
      (detail as any).board = boardSummary;
      detail.posts = rawPosts;
      setBoardDetail(detail);
    } catch (e: any) {
      toast.error(t('boards.openError', { error: e.message || 'Unknown' }));
    } finally {
      setBoardDetailLoading(false);
    }
  };

  const handleTogglePrivacy = async () => {
    if (!boardDetail || updatingPrivacy || !isOwner) return;
    const currentIsPublic = Boolean(boardDetail.board.is_public);
    const nextIsPublic = !currentIsPublic;
    setUpdatingPrivacy(true);
    try {
      const updated = await TiresiasApi.updateBoard(boardDetail.board.id, { is_public: nextIsPublic });
      setBoardDetail({
        ...boardDetail,
        board: {
          ...boardDetail.board,
          is_public: updated.is_public ?? nextIsPublic,
        },
      });
      loadBoards();
    } catch (e: any) {
      toast.error(t('boards.privacyError', { error: e.message || 'Unknown' }));
    } finally {
      setUpdatingPrivacy(false);
    }
  };

  const handleGenerateBoardRecs = async (more = false) => {
    if (!selectedBoardId) return;
    if (more && (boardRecsLoading || !hasMoreRecs)) return;
    setBoardRecsLoading(true);
    setIsRecsMaintenance(false);
    const currentCount = more ? boardRecs.length : 0;
    const requestedLimit = currentCount + 24;
    try {
      const activeRatings = boardRatings.length > 0 ? boardRatings : undefined;
      const res = await TiresiasApi.getBoardRecommendations(
        selectedBoardId,
        requestedLimit,
        activeRatings,
        boardMinScore
      );
      const rawRecs = (res.items || []).map((x: any) => ({ ...x, id: x.id ?? x.post_id }));
      const pids = rawRecs.map((x: any) => x.id).filter(Boolean);
      if (pids.length > 0) {
        try {
          const booruResp = await fetch(`/posts.json?tags=id:${pids.join(',')}`);
          if (booruResp.ok) {
            const bData = await booruResp.json();
            const map = new Map<number, any>();
            for (const bp of (bData.posts || [])) map.set(bp.id, bp);
            for (const it of rawRecs) {
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
          console.warn('[Tiresias] Could not enrich board recs:', err);
        }
      }
      if (more) {
        setBoardRecs((prev) => {
          const existing = new Set(prev.map((p) => p.id));
          const unique = rawRecs.filter((p) => !existing.has(p.id));
          if (unique.length === 0) {
            setHasMoreRecs(false);
          }
          return [...prev, ...unique];
        });
      } else {
        setBoardRecs(rawRecs);
        setHasMoreRecs(rawRecs.length >= 20);
      }
    } catch (e: any) {
      if (e?.isMaintenance) {
        setIsRecsMaintenance(true);
        toast.info(t('feed.maintenanceTitle'));
      } else {
        toast.error(t('boards.recsError', { error: e.message || 'Unknown' }));
      }
    } finally {
      setBoardRecsLoading(false);
    }
  };

  // Infinite scroll intersection observer: automatically scrolls up to 300 posts
  useEffect(() => {
    if (!recsSentinelRef.current) return;
    if (boardRecsLoading || !hasMoreRecs || boardRecs.length === 0) return;
    if (boardRecs.length >= 300) return; // Strictly halt auto-scroll at 300 posts

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !boardRecsLoading && hasMoreRecs && boardRecs.length < 300) {
          handleGenerateBoardRecs(true);
        }
      },
      { rootMargin: '400px' }
    );

    observer.observe(recsSentinelRef.current);
    return () => observer.disconnect();
  }, [boardRecs.length, boardRecsLoading, hasMoreRecs, selectedBoardId, boardRatings, boardMinScore]);

  const handleAddRecToBoard = async (recPost: PostMetadata) => {
    if (!selectedBoardId || !boardDetail) return;
    const pid = recPost.id ?? (recPost as any).post_id;
    try {
      await TiresiasApi.addPostsToBoard(selectedBoardId, [pid]);
      setAddedRecIds((prev) => new Set([...prev, pid]));
      const newPostItem = {
        ...recPost,
        id: pid,
        added_at: Date.now() / 1000,
      };
      setBoardDetail({
        ...boardDetail,
        posts: [newPostItem, ...boardDetail.posts],
        board: {
          ...boardDetail.board,
          post_count: (boardDetail.board.post_count || boardDetail.posts.length) + 1,
        },
      });
      loadBoards();
      toast.success(t('boards.postAdded', { id: pid }));
    } catch (e: any) {
      toast.error(t('boards.postAddError', { error: e.message || 'Unknown' }));
    }
  };

  const handleRemovePost = async (postId: number) => {
    if (!selectedBoardId || !boardDetail) return;
    try {
      await TiresiasApi.removePostFromBoard(selectedBoardId, postId);
      setBoardDetail({
        ...boardDetail,
        posts: boardDetail.posts.filter((p) => p.id !== postId),
        board: { ...boardDetail.board, post_count: Math.max(0, boardDetail.board.post_count - 1) },
      });
      setAddedRecIds((prev) => {
        const next = new Set(prev);
        next.delete(postId);
        return next;
      });
      loadBoards();
      toast.info(t('boards.postRemoved', { id: postId }));
    } catch (e: any) {
      toast.error(t('boards.postRemoveError', { error: e.message || 'Unknown' }));
    }
  };

  const handleDeleteBoard = async (boardId: string) => {
    if (confirmDeleteId !== boardId) {
      setConfirmDeleteId(boardId);
      setTimeout(() => {
        setConfirmDeleteId((cur) => (cur === boardId ? null : cur));
      }, 5000);
      return;
    }
    setConfirmDeleteId(null);
    try {
      await TiresiasApi.deleteBoard(boardId);
      if (selectedBoardId === boardId) {
        setSelectedBoardId(null);
        setBoardDetail(null);
        if (typeof window !== 'undefined') {
          window.history.pushState(null, '', '/boards');
        }
      }
      loadBoards();
      toast.success(t('boards.deleteSuccess'));
    } catch (e: any) {
      toast.error(t('boards.deleteError', { error: e.message || 'Unknown' }));
    }
  };

  const handleCreateBoard = async (e: Event) => {
    e.preventDefault();
    if (!newTitle.trim()) return;
    try {
      await TiresiasApi.createBoard(newTitle.trim(), newDesc.trim(), undefined, newIsPublic);
      setShowCreateModal(false);
      const createdTitle = newTitle.trim();
      setNewTitle('');
      setNewDesc('');
      setNewIsPublic(false);
      loadBoards();
      toast.success(t('boards.createSuccess', { title: createdTitle }));
    } catch (err: any) {
      toast.error(t('boards.createError', { error: err.message || 'Unknown' }));
    }
  };

  // Listen for board updates from other components (BoardPickerModal, etc.)
  useEffect(() => {
    const onBoardUpdated = (e: any) => {
      const updatedBoardId = e.detail?.boardId;
      loadBoards();
      if (selectedBoardId && String(selectedBoardId) === String(updatedBoardId)) {
        loadBoardDetail(selectedBoardId);
      }
    };
    window.addEventListener('tiresias:board-updated', onBoardUpdated);
    return () => window.removeEventListener('tiresias:board-updated', onBoardUpdated);
  }, [selectedBoardId]);

  // Listen for actions from floating action bar (✕ remove from board, ➕ add to current board)
  useEffect(() => {
    const handleRemoveFromBoard = (e: any) => {
      const pid = e.detail?.postId;
      if (pid) handleRemovePost(pid);
    };
    const handleAddToCurrentBoard = (e: any) => {
      const pid = e.detail?.postId;
      if (pid) {
        const found = boardRecs.find((x) => (x.id ?? (x as any).post_id) === pid);
        if (found) handleAddRecToBoard(found);
      }
    };
    window.addEventListener('tiresias:remove-from-board', handleRemoveFromBoard);
    window.addEventListener('tiresias:add-to-current-board', handleAddToCurrentBoard);
    return () => {
      window.removeEventListener('tiresias:remove-from-board', handleRemoveFromBoard);
      window.removeEventListener('tiresias:add-to-current-board', handleAddToCurrentBoard);
    };
  }, [boardDetail, boardRecs]);

  // Current user ownership
  const currentUserId = config?.userId || config?.authUser?.user_id || 'default_user';
  const isOwner = Boolean(
    !boardDetail?.board.user_id ||
    boardDetail.board.user_id === currentUserId ||
    (config?.authUser?.role ?? 0) >= 100
  );

  // Sorted posts in active board
  const sortedPosts = [...(boardDetail?.posts || [])].sort((a, b) => {
    if (sortBy === 'added_asc') return (a.added_at || 0) - (b.added_at || 0);
    if (sortBy === 'score_desc') return (b.score_val ?? b.score ?? 0) - (a.score_val ?? a.score ?? 0);
    if (sortBy === 'score_asc') return (a.score_val ?? a.score ?? 0) - (b.score_val ?? b.score ?? 0);
    if (sortBy === 'id_desc') return b.id - a.id;
    return (b.added_at || 0) - (a.added_at || 0);
  });

  return (
    <div className="tiresias-page-wrapper">
      <div className="tiresias-header-bar">
        <div className="tiresias-header-title">
          {selectedBoardId && (
            <Button
              variant="secondary"
              size="sm"
              style={{ marginRight: 8, fontSize: 13 }}
              onClick={() => {
                setSelectedBoardId(null);
                setBoardDetail(null);
                if (typeof window !== 'undefined') {
                  window.history.pushState(null, '', '/boards');
                }
              }}
            >
              {t('boards.allBoards')}
            </Button>
          )}
          <span>📂</span>
          <span>{selectedBoardId && boardDetail ? boardDetail.board.title : t('boards.title')}</span>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          {!selectedBoardId && (
            <Button variant="primary" onClick={() => setShowCreateModal(true)}>
              {t('boards.newBoard')}
            </Button>
          )}
        </div>
      </div>

      {/* DETAIL VIEW */}
      {selectedBoardId ? (
        boardDetailLoading || !boardDetail ? (
          <div style={{ textAlign: 'center', padding: 60, color: 'var(--tiresias-gold)', fontSize: 16 }}>
            {t('boards.loadingBoard')}
          </div>
        ) : (
          <div>
            <div style={{ background: 'var(--tiresias-card)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-border)', marginBottom: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
                <div>
                  <h2 style={{ margin: 0, color: 'var(--tiresias-gold)', fontSize: 20 }}>
                    📁 {boardDetail.board.title}
                  </h2>
                  {boardDetail.board.description && (
                    <p style={{ margin: '6px 0', color: 'var(--tiresias-text-muted)', fontSize: 13 }}>
                      {boardDetail.board.description}
                    </p>
                  )}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 6 }}>
                    <span style={{ fontSize: 12, color: 'var(--tiresias-accent)' }}>
                      {t('boards.postsInCollection', { count: boardDetail.board.post_count })}
                    </span>
                    {boardDetail.board.user_id && !isOwner && (
                      <span style={{ fontSize: 12, color: 'var(--tiresias-text-muted)' }}>
                        {t('boards.owner', { owner: boardDetail.board.user_id })}
                      </span>
                    )}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                  {/* Privacy Badge / Toggle Button */}
                  {isOwner ? (
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={updatingPrivacy}
                      onClick={handleTogglePrivacy}
                      title={boardDetail.board.is_public ? t('boards.makePrivateTitle') : t('boards.makePublicTitle')}
                    >
                      <span>{boardDetail.board.is_public ? t('boards.public') : t('boards.private')}</span>
                      <span style={{ fontSize: 10, opacity: 0.7 }}>✎</span>
                    </Button>
                  ) : (
                    <Badge variant={boardDetail.board.is_public ? 'artist' : 'general'}>
                      {boardDetail.board.is_public ? t('boards.public') : t('boards.private')}
                    </Badge>
                  )}

                  {/* Recommendations Button with Algorithm Popover */}
                  <div style={{ position: 'relative', display: 'inline-flex' }}>
                    <Button
                      variant="primary"
                      onClick={() => handleGenerateBoardRecs(false)}
                      disabled={boardRecsLoading}
                    >
                      {boardRecsLoading ? t('boards.searching') : t('boards.recommendations')}
                    </Button>
                    <IconButton
                      icon="⚙️"
                      title={t('boards.boardRecsSettings')}
                      active={showBoardRecsSettings}
                      onClick={() => setShowBoardRecsSettings(!showBoardRecsSettings)}
                      style={{ marginLeft: 4 }}
                    />

                    {showBoardRecsSettings && (
                      <div
                        id="board-algorithm-container"
                        className="tiresias-popover-container active"
                        style={{ top: '100%', right: 0, marginTop: 6, minWidth: 260 }}
                      >
                        <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span>{t('boards.boardParams')}</span>
                          <IconButton
                            icon="✕"
                            title={t('common.close')}
                            size="sm"
                            onClick={() => setShowBoardRecsSettings(false)}
                          />
                        </h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                          <div>
                            <div style={{ fontSize: 12, fontWeight: 'bold', marginBottom: 4, color: 'var(--tiresias-text-primary)' }}>
                              {t('boards.ratingsDefaultAuto')}
                            </div>
                            <div style={{ display: 'flex', gap: 6 }}>
                              {[
                                { key: 's', label: 'Safe', color: 'var(--tiresias-rating-s)' },
                                { key: 'q', label: 'Quest.', color: 'var(--tiresias-rating-q)' },
                                { key: 'e', label: 'Explicit', color: 'var(--tiresias-rating-e)' },
                              ].map(({ key, label, color }) => {
                                const checked = boardRatings.includes(key);
                                return (
                                  <Button
                                    key={key}
                                    size="sm"
                                    variant={checked ? 'primary' : 'secondary'}
                                    style={{
                                      fontSize: 11,
                                      padding: '4px 8px',
                                      background: checked ? color : undefined,
                                      borderColor: checked ? color : undefined,
                                    }}
                                    onClick={() => {
                                      if (checked) {
                                        setBoardRatings(boardRatings.filter((r) => r !== key));
                                      } else {
                                        setBoardRatings([...boardRatings, key]);
                                      }
                                    }}
                                  >
                                    {label}
                                  </Button>
                                );
                              })}
                            </div>
                            <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)', marginTop: 4 }}>
                              {boardRatings.length === 0 ? t('boards.autoContentMatch') : t('boards.selectedRatings', { ratings: boardRatings.join(', ').toUpperCase() })}
                            </div>
                          </div>

                          <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontSize: 12, fontWeight: 'bold', color: 'var(--tiresias-text-primary)' }}>{t('boards.minScore')}</span>
                              <Input
                                type="number"
                                size="sm"
                                style={{ width: 70, textAlign: 'right' }}
                                value={boardMinScore}
                                onChange={(e: any) => setBoardMinScore(parseInt(e.target.value, 10) || 0)}
                              />
                            </div>
                          </div>

                          <Button
                            variant="primary"
                            style={{ width: '100%', fontSize: 12, marginTop: 4 }}
                            onClick={() => {
                              setShowBoardRecsSettings(false);
                              handleGenerateBoardRecs(false);
                            }}
                          >
                            {t('boards.applyAndRecommend')}
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Salient Tags - Sorted descending by post count from left to right */}
              {boardDetail.salient_tags && boardDetail.salient_tags.length > 0 && (
                <div style={{ marginTop: 14 }}>
                  <span style={{ fontSize: 11, color: 'var(--tiresias-accent)', marginRight: 6 }}>
                    {t('boards.motifs')}
                  </span>
                  <div style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 5 }}>
                    {[...boardDetail.salient_tags]
                      .sort((a, b) => (b.post_count || (b as any).count || 0) - (a.post_count || (a as any).count || 0))
                      .slice(0, 15)
                      .map((tagItem) => (
                        <Badge key={tagItem.tag} variant="general">
                          {tagItem.tag} ({tagItem.post_count || (tagItem as any).count || 1})
                        </Badge>
                      ))}
                  </div>
                </div>
              )}
            </div>

            {/* Board Posts Grid Header with Sort Controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <h3 style={{ margin: 0, color: 'var(--tiresias-text-primary)' }}>
                {t('boards.boardPostsHeader', { count: boardDetail.posts.length })}
              </h3>

              {boardDetail.posts.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 12, color: 'var(--tiresias-text-muted)' }}>{t('boards.sort')}</span>
                  <Select
                    size="sm"
                    value={sortBy}
                    onChange={(val) => setSortBy(val as any)}
                    options={[
                      { label: t('boards.sortAddedDesc'), value: 'added_desc' },
                      { label: t('boards.sortAddedAsc'), value: 'added_asc' },
                      { label: t('boards.sortScoreDesc'), value: 'score_desc' },
                      { label: t('boards.sortScoreAsc'), value: 'score_asc' },
                      { label: t('boards.sortIdDesc'), value: 'id_desc' },
                    ]}
                  />
                </div>
              )}
            </div>

            {boardDetail.posts.length === 0 ? (
              <div style={{ textAlign: 'center', padding: 40, color: 'var(--tiresias-text-muted)', background: 'rgba(2, 15, 35, 0.4)', borderRadius: 'var(--tiresias-radius-md)', marginBottom: 28 }}>
                {t('boards.emptyBoard')}
              </div>
            ) : (
              /* Unified e621 Thumbnail Grid for Board Posts */
              <section
                className={`posts-container tiresias-posts-container ${!showDesc ? 'no-stats' : ''}`}
                style={{ '--thumb-image-size': `${cardSize}px`, marginBottom: 32 } as any}
                data-st-contain={imageContain ? 'true' : 'false'}
                data-st-show-desc={showDesc ? 'true' : 'false'}
              >
                {sortedPosts.map((p) => {
                  const pid = p.id ?? (p as any).post_id;
                  const host = typeof window !== 'undefined' ? window.location.host : 'e926.net';
                  const previewUrl = p.preview_url;
                  const scoreVal = p.score_val ?? Math.round((p.score || 0) * 100);
                  return (
                    <article
                      key={pid}
                      className="thumbnail post-thumbnail tiresias-post-thumb"
                      data-id={pid}
                      data-score={scoreVal}
                      data-rating={p.rating || 's'}
                      data-context="board-post"
                      data-board-id={selectedBoardId}
                    >
                      <a href={`https://${host}/posts/${pid}`} target="_blank" rel="noreferrer" className="thm-link">
                        {p.is_video && <div className="tiresias-video-badge">🎬</div>}
                        {previewUrl ? (
                          <img src={previewUrl} alt={`Post #${pid}`} loading="lazy" />
                        ) : (
                          <div className="tiresias-thumb-placeholder">#{pid}</div>
                        )}
                      </a>
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
            )}

            {/* Generated Recommendations Section (Placed at the bottom) */}
            {isRecsMaintenance && (
              <div style={{
                padding: 14,
                background: 'rgba(232, 196, 70, 0.15)',
                border: '1px solid var(--tiresias-gold)',
                borderRadius: 'var(--tiresias-radius-md)',
                marginTop: 20,
                marginBottom: 20,
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

            {boardRecs.length > 0 && (
              <div style={{ marginTop: 24, marginBottom: 28, background: 'var(--tiresias-bg-surface-dark)', padding: 18, borderRadius: 'var(--tiresias-radius-md)', border: '1px solid var(--tiresias-gold)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div>
                    <h3 style={{ margin: 0, color: 'var(--tiresias-gold)', fontSize: 16 }}>
                      {t('boards.recsForBoard', { title: boardDetail.board.title })}
                    </h3>
                    <div style={{ fontSize: 11, color: 'var(--tiresias-text-muted)', marginTop: 2 }}>
                      {t('boards.recsHint')}
                    </div>
                  </div>
                  <IconButton icon="✕" size="sm" onClick={() => setBoardRecs([])} title={t('boards.hideRecs')} />
                </div>

                {/* Unified e621 Thumbnail Grid for Recommendations */}
                <section
                  className={`posts-container tiresias-posts-container ${!showDesc ? 'no-stats' : ''}`}
                  style={{ '--thumb-image-size': `${cardSize}px` } as any}
                  data-st-contain={imageContain ? 'true' : 'false'}
                  data-st-show-desc={showDesc ? 'true' : 'false'}
                >
                  {boardRecs.map((p) => {
                    const pid = p.id ?? (p as any).post_id;
                    const host = typeof window !== 'undefined' ? window.location.host : 'e926.net';
                    const previewUrl = p.preview_url;
                    const isAlreadyInBoard = boardDetail.posts.some((x) => x.id === pid) || addedRecIds.has(pid);
                    const scoreVal = p.score_val ?? Math.round((p.score || 0) * 100);

                    return (
                      <article
                        key={pid}
                        className="thumbnail post-thumbnail tiresias-post-thumb"
                        data-id={pid}
                        data-score={scoreVal}
                        data-rating={p.rating || 's'}
                        data-context={isAlreadyInBoard ? 'feed' : 'board-rec'}
                        data-board-id={selectedBoardId}
                      >
                        <a href={`https://${host}/posts/${pid}`} target="_blank" rel="noreferrer" className="thm-link">
                          {p.is_video && <div className="tiresias-video-badge">🎬</div>}
                          {previewUrl ? (
                            <img src={previewUrl} alt={`Post #${pid}`} loading="lazy" />
                          ) : (
                            <div className="tiresias-thumb-placeholder">#{pid}</div>
                          )}
                        </a>
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

                {/* Infinite Scroll Sentinel for Recommendations - unmounted when reaching 300 posts */}
                {boardRecs.length < 300 && hasMoreRecs && (
                  <div ref={recsSentinelRef} style={{ height: 1, marginTop: 10 }} />
                )}
                {boardRecsLoading && (
                  <div style={{ textAlign: 'center', padding: '16px 0', color: 'var(--tiresias-gold)', fontSize: 13 }}>
                    {t('boards.searchingRecs')}
                  </div>
                )}
                {!boardRecsLoading && boardRecs.length >= 300 && hasMoreRecs && (
                  <div style={{ textAlign: 'center', marginTop: 18 }}>
                    <Button
                      variant="primary"
                      style={{ padding: '8px 24px', fontSize: 14 }}
                      onClick={() => handleGenerateBoardRecs(true)}
                    >
                      {t('boards.loadMoreRecs', { count: boardRecs.length })}
                    </Button>
                  </div>
                )}
                {!boardRecsLoading && !hasMoreRecs && (
                  <div style={{ textAlign: 'center', marginTop: 16, fontSize: 12, color: 'var(--tiresias-text-muted)' }}>
                    {t('boards.allRecsLoaded', { count: boardRecs.length })}
                  </div>
                )}
              </div>
            )}

            {/* Dedicated Danger Zone Card for Board Owner */}
            {isOwner && (
              <div
                style={{
                  marginTop: 36,
                  border: '1px solid rgba(228, 95, 95, 0.4)',
                  borderRadius: 'var(--tiresias-radius-md)',
                  padding: 16,
                  background: 'var(--tiresias-danger-subtle)',
                }}
              >
                <h4 style={{ margin: '0 0 6px 0', color: 'var(--tiresias-danger)', fontSize: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                  {t('boards.dangerZone')}
                </h4>
                <p style={{ margin: '0 0 12px 0', fontSize: 12, color: 'var(--tiresias-text-muted)' }}>
                  {t('boards.dangerWarning')}
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Button
                    variant="danger"
                    onClick={() => handleDeleteBoard(boardDetail.board.id)}
                  >
                    {confirmDeleteId === boardDetail.board.id ? t('boards.confirmDeleteBoard') : t('boards.deleteBoard')}
                  </Button>
                  {confirmDeleteId === boardDetail.board.id && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => setConfirmDeleteId(null)}
                    >
                      {t('boards.cancel')}
                    </Button>
                  )}
                </div>
              </div>
            )}
          </div>
        )
      ) : (
        /* LIST OF ALL BOARDS */
        <div>
          {loading ? (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--tiresias-gold)' }}>
              {t('boards.loadingBoards')}
            </div>
          ) : boards.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 50, color: 'var(--tiresias-text-muted)' }}>
              {t('boards.noBoards')}
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 16 }}>
              {boards.map((b) => {
                const bid = b.id || (b as any).board_id;
                return (
                  <div
                    key={bid}
                    className="tiresias-card-item"
                    style={{ cursor: 'pointer' }}
                    onClick={() => loadBoardDetail(bid)}
                  >
                    <div className="tiresias-thumb-box" style={{ background: 'var(--tiresias-bg-surface-dark)', height: 160 }}>
                      {b.cover_url || b.cover_post_id ? (
                        <img
                          src={b.cover_url || `https://static1.e926.net/data/preview/${b.cover_post_id}.jpg`}
                          alt={b.title}
                          loading="lazy"
                        />
                      ) : (
                        <span style={{ fontSize: 40, opacity: 0.4 }}>📁</span>
                      )}
                    </div>
                    <div className="tiresias-card-meta" style={{ padding: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ fontWeight: 'bold', fontSize: 15, color: 'var(--tiresias-gold)' }}>
                          {b.title}
                        </div>
                        <Badge variant={b.is_public ? 'artist' : 'general'} size="sm">
                          {b.is_public ? t('boards.public') : t('boards.private')}
                        </Badge>
                      </div>
                      {b.description && (
                        <div style={{ fontSize: 12, color: 'var(--tiresias-text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: 3 }}>
                          {b.description}
                        </div>
                      )}
                      <div className="tiresias-meta-row" style={{ marginTop: 8 }}>
                        <span>{t('boards.postsCount', { count: b.post_count })}</span>
                        <span style={{ color: 'var(--tiresias-accent)' }}>{t('boards.openBoard')}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Create Board Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title={t('boards.createModalTitle')}
        icon="📁"
        size="md"
      >
        <form onSubmit={handleCreateBoard}>
          <FormGroup label={t('boards.nameLabel')} required>
            <Input
              type="text"
              placeholder={t('boards.namePlaceholder')}
              value={newTitle}
              onInput={(e) => setNewTitle((e.target as HTMLInputElement).value)}
              required
            />
          </FormGroup>
          <FormGroup label={t('boards.descLabel')}>
            <Input
              type="text"
              placeholder={t('boards.descPlaceholder')}
              value={newDesc}
              onInput={(e) => setNewDesc((e.target as HTMLInputElement).value)}
            />
          </FormGroup>
          <FormGroup label={t('boards.accessLabel')}>
            <SegmentedControl
              options={[
                { label: t('boards.private'), value: 'private' },
                { label: t('boards.public'), value: 'public' },
              ]}
              value={newIsPublic ? 'public' : 'private'}
              onChange={(val) => setNewIsPublic(val === 'public')}
            />
          </FormGroup>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 16 }}>
            <Button variant="secondary" onClick={() => setShowCreateModal(false)}>
              {t('boards.cancelBtn')}
            </Button>
            <Button type="submit" variant="primary" disabled={!newTitle.trim()}>
              {t('boards.createBtn')}
            </Button>
          </div>
        </form>
      </Modal>

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

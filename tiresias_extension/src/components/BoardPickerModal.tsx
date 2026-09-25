import { h } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { TiresiasApi } from '../lib/api';
import { Board, PostMetadata } from '../lib/types';
import { Modal, Button, Input, Badge, toast } from './ui';
import { useTranslation } from '../lib/i18n';

interface Props {
  post: PostMetadata | null;
  onClose: () => void;
  onAdded?: (boardTitle: string) => void;
}

export function BoardPickerModal({ post, onClose, onAdded }: Props) {
  const { t } = useTranslation();
  const [boards, setBoards] = useState<Board[]>([]);
  const [loading, setLoading] = useState(true);
  const [newTitle, setNewTitle] = useState('');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    if (!post) return;
    setLoading(true);
    TiresiasApi.listBoards()
      .then((b) => {
        setBoards(b);
        setLoading(false);
      })
      .catch((e) => {
        console.error('Failed to list boards', e);
        toast.error(t('boards.picker.loadError'));
        setLoading(false);
      });
  }, [post, t]);

  if (!post) return null;

  const handleAddToBoard = async (board: Board) => {
    try {
      await TiresiasApi.addPostsToBoard(board.id, [post.id]);
      toast.success(t('boards.picker.addSuccess', { title: board.title }));
      window.dispatchEvent(new CustomEvent('tiresias:board-updated', { detail: { boardId: board.id, postId: post.id } }));
      if (onAdded) onAdded(board.title);
      onClose();
    } catch (e: any) {
      toast.error(t('boards.picker.addError', { error: e.message || 'Unknown' }));
    }
  };

  const handleCreateBoard = async (e: Event) => {
    e.preventDefault();
    if (!newTitle.trim()) return;
    setCreating(true);
    try {
      const created = await TiresiasApi.createBoard(newTitle.trim(), '', post.id);
      await TiresiasApi.addPostsToBoard(created.id, [post.id]);
      toast.success(t('boards.picker.createSuccess', { title: created.title }));
      window.dispatchEvent(new CustomEvent('tiresias:board-updated', { detail: { boardId: created.id, postId: post.id } }));
      if (onAdded) onAdded(created.title);
      onClose();
    } catch (err: any) {
      toast.error(t('boards.picker.createError', { error: err.message || 'Unknown' }));
      setCreating(false);
    }
  };

  return (
    <Modal
      isOpen={!!post}
      onClose={onClose}
      title={t('boards.picker.modalTitle', { id: post.id })}
      icon="📁"
      size="sm"
    >
      <div>
        {loading ? (
          <div style={{ color: 'var(--tiresias-text-muted)', textAlign: 'center', padding: 20 }}>
            {t('boards.picker.loading')}
          </div>
        ) : (
          <div style={{ maxHeight: '240px', overflowY: 'auto', marginBottom: 16, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {boards.length === 0 ? (
              <div style={{ color: 'var(--tiresias-text-muted)', fontSize: 12, textAlign: 'center', padding: 12 }}>
                {t('boards.picker.empty')}
              </div>
            ) : (
              boards.map((b) => (
                <Button
                  key={b.id}
                  variant="secondary"
                  onClick={() => handleAddToBoard(b)}
                  style={{ justifyContent: 'space-between', width: '100%', padding: '10px 12px' }}
                >
                  <span style={{ fontWeight: 'bold' }}>📁 {b.title}</span>
                  <Badge variant="general">
                    {b.post_count} {t('common.postsCount')}
                  </Badge>
                </Button>
              ))
            )}
          </div>
        )}

        <hr style={{ borderColor: 'var(--tiresias-border)', margin: '14px 0' }} />

        <form onSubmit={handleCreateBoard} style={{ display: 'flex', gap: 8 }}>
          <Input
            type="text"
            placeholder={t('boards.picker.newBoardPlaceholder')}
            value={newTitle}
            onInput={(e) => setNewTitle((e.target as HTMLInputElement).value)}
            disabled={creating}
            style={{ flex: 1 }}
          />
          <Button
            type="submit"
            variant="primary"
            disabled={creating || !newTitle.trim()}
            loading={creating}
            style={{ flexShrink: 0 }}
          >
            {t('boards.picker.createBtn')}
          </Button>
        </form>
      </div>
    </Modal>
  );
}

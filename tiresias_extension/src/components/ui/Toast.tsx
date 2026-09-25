import { h, Fragment } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { ToastEventDetail, ToastDismissDetail, ToastType } from '../../lib/toast';

interface ToastEntry {
  id: string;
  type: ToastType;
  message: string;
  duration: number;
  remainingMs: number;
  timerId: ReturnType<typeof setTimeout> | null;
  lastResumeTime: number;
  isHovered: boolean;
}

export interface ToastProps {
  id: string;
  type: ToastType;
  message: string;
  onDismiss: (id: string) => void;
  onPointerEnter?: () => void;
  onPointerLeave?: () => void;
}

const TYPE_ICONS: Record<ToastType, string> = {
  success: '✓',
  error: '⚠️',
  info: 'ℹ️',
  warning: '⚡',
};

/**
 * Individual Toast Component
 */
export function Toast({ id, type, message, onDismiss, onPointerEnter, onPointerLeave }: ToastProps) {
  const isAlert = type === 'error';

  return (
    <div
      className={`tiresias-floating-toast toast-${type}`}
      role={isAlert ? 'alert' : 'status'}
      aria-live={isAlert ? 'assertive' : 'polite'}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', wordBreak: 'break-word' }}>
        <span style={{ fontSize: '14px', lineHeight: 1, flexShrink: 0 }}>
          {TYPE_ICONS[type] || '•'}
        </span>
        <span style={{ fontSize: '13px', lineHeight: 1.4 }}>{message}</span>
      </div>
      <button
        type="button"
        className="tiresias-modal-close"
        onClick={() => onDismiss(id)}
        aria-label="Dismiss notification"
        title="Dismiss"
        style={{
          cursor: 'pointer',
          padding: '2px 6px',
          marginLeft: '8px',
          background: 'transparent',
          border: 'none',
          color: 'inherit',
          opacity: 0.8,
          fontSize: '13px',
          lineHeight: 1,
          flexShrink: 0,
        }}
      >
        ✕
      </button>
    </div>
  );
}

export interface ToastContainerProps {
  maxToasts?: number;
  className?: string;
}

/**
 * Global Toast Container Component
 * Listens for window-level 'tiresias:toast' events and renders stacked, auto-dismissing toasts.
 */
export function ToastContainer({ maxToasts = 5, className }: ToastContainerProps) {
  const [toasts, setToasts] = useState<ToastEntry[]>([]);
  const toastsRef = useRef<ToastEntry[]>([]);
  toastsRef.current = toasts;

  const dismissToast = (id: string) => {
    setToasts((prev) => {
      const match = prev.find((t) => t.id === id);
      if (match?.timerId) {
        clearTimeout(match.timerId);
      }
      return prev.filter((t) => t.id !== id);
    });
  };

  const scheduleDismiss = (id: string, delay: number) => {
    return setTimeout(() => {
      dismissToast(id);
    }, delay);
  };

  useEffect(() => {
    const handleNewToast = (e: Event) => {
      const customEvent = e as CustomEvent<ToastEventDetail>;
      const detail = customEvent.detail;
      if (!detail || !detail.message) return;

      const duration = detail.duration > 0 ? detail.duration : 2500;
      const timerId = scheduleDismiss(detail.id, duration);

      const newEntry: ToastEntry = {
        id: detail.id,
        type: detail.type || 'info',
        message: detail.message,
        duration,
        remainingMs: duration,
        timerId,
        lastResumeTime: Date.now(),
        isHovered: false,
      };

      setToasts((prev) => {
        // Dismiss if same ID already exists
        const existing = prev.find((t) => t.id === detail.id);
        if (existing?.timerId) {
          clearTimeout(existing.timerId);
        }
        const filtered = prev.filter((t) => t.id !== detail.id);
        const updated = [...filtered, newEntry];
        if (updated.length > maxToasts) {
          const removed = updated.shift();
          if (removed?.timerId) clearTimeout(removed.timerId);
        }
        return updated;
      });
    };

    const handleDismissToast = (e: Event) => {
      const customEvent = e as CustomEvent<ToastDismissDetail>;
      const id = customEvent.detail?.id;
      if (id) {
        dismissToast(id);
      }
    };

    window.addEventListener('tiresias:toast', handleNewToast);
    window.addEventListener('tiresias:toast-dismiss', handleDismissToast);

    return () => {
      window.removeEventListener('tiresias:toast', handleNewToast);
      window.removeEventListener('tiresias:toast-dismiss', handleDismissToast);
      toastsRef.current.forEach((t) => {
        if (t.timerId) clearTimeout(t.timerId);
      });
    };
  }, [maxToasts]);

  const handlePointerEnter = (id: string) => {
    setToasts((prev) =>
      prev.map((t) => {
        if (t.id === id) {
          if (t.timerId) clearTimeout(t.timerId);
          const elapsed = Date.now() - t.lastResumeTime;
          const remainingMs = Math.max(t.remainingMs - elapsed, 1000);
          return {
            ...t,
            timerId: null,
            remainingMs,
            isHovered: true,
          };
        }
        return t;
      })
    );
  };

  const handlePointerLeave = (id: string) => {
    setToasts((prev) =>
      prev.map((t) => {
        if (t.id === id) {
          const remaining = Math.max(t.remainingMs, 1000);
          const timerId = scheduleDismiss(id, remaining);
          return {
            ...t,
            timerId,
            lastResumeTime: Date.now(),
            isHovered: false,
          };
        }
        return t;
      })
    );
  };

  if (toasts.length === 0) return null;

  return (
    <div
      className={`tiresias-toast-container ${className || ''}`}
      role="region"
      aria-label="Notifications"
    >
      {toasts.map((item) => (
        <Toast
          key={item.id}
          id={item.id}
          type={item.type}
          message={item.message}
          onDismiss={dismissToast}
          onPointerEnter={() => handlePointerEnter(item.id)}
          onPointerLeave={() => handlePointerLeave(item.id)}
        />
      ))}
    </div>
  );
}

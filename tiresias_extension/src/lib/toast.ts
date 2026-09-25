/**
 * Tiresias Toast Notification Manager
 *
 * Lightweight, event-driven singleton for broadcasting UI notification messages.
 * Emits 'tiresias:toast' and 'tiresias:toast-dismiss' CustomEvents on the window object.
 */

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastOptions {
  id?: string;
  type?: ToastType;
  message: string;
  duration?: number;
}

export interface ToastEventDetail {
  id: string;
  type: ToastType;
  message: string;
  duration: number;
}

export interface ToastDismissDetail {
  id: string;
}

const DEFAULT_DURATION = 2500;

function createUniqueId(): string {
  return `toast_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function dispatchToast(type: ToastType, message: string, duration = DEFAULT_DURATION, customId?: string): string {
  const id = customId || createUniqueId();
  if (typeof window !== 'undefined') {
    const detail: ToastEventDetail = {
      id,
      type,
      message,
      duration,
    };
    window.dispatchEvent(new CustomEvent('tiresias:toast', { detail }));
  }
  return id;
}

function dismissToast(id: string): void {
  if (typeof window !== 'undefined') {
    const detail: ToastDismissDetail = { id };
    window.dispatchEvent(new CustomEvent('tiresias:toast-dismiss', { detail }));
  }
}

export const toast = {
  /**
   * Display a success notification toast.
   */
  success(message: string, duration = DEFAULT_DURATION, id?: string): string {
    return dispatchToast('success', message, duration, id);
  },

  /**
   * Display an error notification toast.
   */
  error(message: string, duration = 3500, id?: string): string {
    return dispatchToast('error', message, duration, id);
  },

  /**
   * Display an informational notification toast.
   */
  info(message: string, duration = DEFAULT_DURATION, id?: string): string {
    return dispatchToast('info', message, duration, id);
  },

  /**
   * Display a warning notification toast.
   */
  warning(message: string, duration = 3000, id?: string): string {
    return dispatchToast('warning', message, duration, id);
  },

  /**
   * Display a custom toast with fine-grained options.
   */
  custom(options: ToastOptions): string {
    const type = options.type || 'info';
    const duration = options.duration ?? DEFAULT_DURATION;
    return dispatchToast(type, options.message, duration, options.id);
  },

  /**
   * Manually dismiss an active toast by ID.
   */
  dismiss(id: string): void {
    dismissToast(id);
  },
};

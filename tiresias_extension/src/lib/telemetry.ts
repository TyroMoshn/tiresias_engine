import { TiresiasApi } from './api';
import { loadConfig } from './storage';

let initialized = false;
const errorTimestamps: number[] = [];
const lastErrors: Map<string, number> = new Map();
const RATE_LIMIT_MAX = 5;
const RATE_LIMIT_WINDOW_MS = 60_000;
const DEDUP_WINDOW_MS = 10_000;

function isExtensionOrTiresiasRelated(errorMsg: string, stack?: string, filename?: string): boolean {
  const combined = `${errorMsg} ${stack || ''} ${filename || ''}`.toLowerCase();
  // Check if error originates from tiresias components or extension scheme
  if (
    combined.includes('tiresias') ||
    combined.includes('moz-extension:') ||
    combined.includes('chrome-extension:') ||
    combined.includes('entrypoints') ||
    combined.includes('feedpage') ||
    combined.includes('boardspage') ||
    combined.includes('similardrawer') ||
    combined.includes('thumbnailactions') ||
    combined.includes('seentracker')
  ) {
    return true;
  }
  return false;
}

function shouldThrottle(message: string): boolean {
  const now = Date.now();

  // 1. Deduplication window
  const lastTime = lastErrors.get(message);
  if (lastTime && now - lastTime < DEDUP_WINDOW_MS) {
    return true;
  }
  lastErrors.set(message, now);

  // 2. Sliding window rate limit
  while (errorTimestamps.length > 0 && errorTimestamps[0] < now - RATE_LIMIT_WINDOW_MS) {
    errorTimestamps.shift();
  }

  if (errorTimestamps.length >= RATE_LIMIT_MAX) {
    return true;
  }

  errorTimestamps.push(now);
  return false;
}

export function initTelemetry(contextName = 'content'): void {
  if (typeof window === 'undefined' || initialized) return;
  initialized = true;

  window.addEventListener('error', (event) => {
    try {
      const message = String(event.message || event.error || 'Unknown window error');
      const filename = event.filename || '';
      const lineno = event.lineno;
      const colno = event.colno;
      const stack = event.error?.stack;

      if (!isExtensionOrTiresiasRelated(message, stack, filename)) {
        return;
      }

      if (shouldThrottle(message)) {
        return;
      }

      loadConfig()
        .then((cfg) => {
          if (cfg.enableTelemetry === false) return;
          TiresiasApi.reportClientError({
            user_id: cfg.detectedSessionUser || cfg.userId || 'default_user',
            error_type: 'window_error',
            message,
            stack,
            source_file: filename,
            lineno,
            colno,
            metadata: { context: contextName },
          }).catch(() => {});
        })
        .catch(() => {});
    } catch {
      // Ignore
    }
  });

  window.addEventListener('unhandledrejection', (event) => {
    try {
      const reason = event.reason;
      const message = reason instanceof Error ? reason.message : String(reason || 'Unhandled Promise Rejection');
      const stack = reason instanceof Error ? reason.stack : undefined;

      if (!isExtensionOrTiresiasRelated(message, stack)) {
        return;
      }

      if (shouldThrottle(message)) {
        return;
      }

      loadConfig()
        .then((cfg) => {
          if (cfg.enableTelemetry === false) return;
          TiresiasApi.reportClientError({
            user_id: cfg.detectedSessionUser || cfg.userId || 'default_user',
            error_type: 'unhandled_rejection',
            message,
            stack,
            metadata: { context: contextName },
          }).catch(() => {});
        })
        .catch(() => {});
    } catch {
      // Ignore
    }
  });
}

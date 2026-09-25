/**
 * Tiresias i18n & Localization Subsystem
 *
 * Provides type-safe localization, reactive language switching via Preact hooks,
 * cross-context synchronization via chrome.storage.local, and dot-notation interpolation.
 */

import { useState, useEffect } from 'preact/hooks';
import { Language, TranslationKey, TranslationSchema } from './types';
import { ru } from './locales/ru';
import { en } from './locales/en';
import { loadConfig, saveConfig } from '../storage';

export type { Language, TranslationKey, TranslationSchema };

export const locales: Record<Language, TranslationSchema> = { ru, en };

let currentLanguage: Language = 'ru';
let isInitialized = false;

/**
 * Returns the currently active language code ('ru' or 'en').
 */
export function getLanguage(): Language {
  return currentLanguage;
}

/**
 * Initializes language setting from persistent extension storage.
 */
export async function initI18n(): Promise<Language> {
  if (isInitialized) return currentLanguage;
  try {
    const cfg = await loadConfig();
    if (cfg.language && (cfg.language === 'ru' || cfg.language === 'en')) {
      currentLanguage = cfg.language;
    }
  } catch (e) {
    console.warn('[Tiresias i18n] Failed to load language from storage:', e);
  }
  isInitialized = true;
  return currentLanguage;
}

// Automatically initialize asynchronously on module evaluation
initI18n();

/**
 * Sets the active language, saves to extension storage, and broadcasts
 * the change event to all components and windows.
 */
export async function setLanguage(lang: Language): Promise<void> {
  if (currentLanguage === lang) return;
  currentLanguage = lang;

  try {
    await saveConfig({ language: lang });
  } catch (e) {
    console.warn('[Tiresias i18n] Failed to persist language:', e);
  }

  broadcastLanguageChange(lang);
}

/**
 * Broadcasts language change event to current window and listeners.
 */
function broadcastLanguageChange(lang: Language) {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent('tiresias:language-changed', { detail: { language: lang } })
    );
  }
}

/**
 * Retrieves a nested value from an object using a dot-separated path.
 */
function getNestedValue(obj: any, path: string): any {
  if (!obj || typeof obj !== 'object') return undefined;
  const parts = path.split('.');
  let current = obj;
  for (const part of parts) {
    if (current == null || typeof current !== 'object') return undefined;
    current = current[part];
  }
  return current;
}

/**
 * Look up a translation key with dot-notation and optional string interpolation.
 * E.g.: t('common.serverError', { error: 'Timeout' })
 */
export function t(
  key: TranslationKey,
  params?: Record<string, string | number>,
  overrideLang?: Language
): string {
  const lang = overrideLang || currentLanguage;
  const dict = locales[lang] || locales.ru;

  let value = getNestedValue(dict, key);
  if (typeof value !== 'string') {
    // Fallback to alternate locale
    const fallbackDict = lang === 'en' ? locales.ru : locales.en;
    value = getNestedValue(fallbackDict, key);
  }

  if (typeof value !== 'string') {
    return key;
  }

  if (params) {
    return value.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, paramName) => {
      return paramName in params ? String(params[paramName]) : match;
    });
  }

  return value;
}

/**
 * Reactive Preact hook for components to consume translations.
 * Automatically re-renders the component whenever the active language changes.
 */
export function useTranslation() {
  const [lang, setLang] = useState<Language>(currentLanguage);

  useEffect(() => {
    // If module initialized with a stored language different from initial state
    if (currentLanguage !== lang) {
      setLang(currentLanguage);
    }

    const handleLanguageChanged = (e: any) => {
      const nextLang: Language = e.detail?.language || currentLanguage;
      if (nextLang === 'ru' || nextLang === 'en') {
        setLang(nextLang);
      }
    };

    window.addEventListener('tiresias:language-changed', handleLanguageChanged);
    return () => {
      window.removeEventListener('tiresias:language-changed', handleLanguageChanged);
    };
  }, [lang]);

  const translate = (key: TranslationKey, params?: Record<string, string | number>) => {
    return t(key, params, lang);
  };

  return {
    t: translate,
    language: lang,
    setLanguage,
  };
}

// -----------------------------------------------------------------------------
// Cross-context storage synchronization
// -----------------------------------------------------------------------------

function handleStorageChange(newLang: any) {
  if (newLang && (newLang === 'ru' || newLang === 'en') && newLang !== currentLanguage) {
    currentLanguage = newLang;
    broadcastLanguageChange(newLang);
  }
}

// WebExtension storage listener (chrome.storage / browser.storage)
if (typeof chrome !== 'undefined' && chrome.storage?.onChanged) {
  chrome.storage.onChanged.addListener((changes: Record<string, any>, areaName: string) => {
    if (areaName === 'local' && changes['tiresias_settings']) {
      const newLang = (changes['tiresias_settings'].newValue as any)?.language;
      handleStorageChange(newLang);
    }
  });
} else if (typeof browser !== 'undefined' && (browser as any).storage?.onChanged) {
  (browser as any).storage.onChanged.addListener((changes: any, areaName: any) => {
    if (areaName === 'local' && changes['tiresias_settings']) {
      const newLang = (changes['tiresias_settings'].newValue as any)?.language;
      handleStorageChange(newLang);
    }
  });
}

// Fallback window storage event for localStorage synchronization
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (e) => {
    if (e.key === 'tiresias_settings' && e.newValue) {
      try {
        const parsed = JSON.parse(e.newValue);
        handleStorageChange(parsed.language);
      } catch {}
    }
  });

  window.addEventListener('tiresias:language-changed', (e: any) => {
    const nextLang = e.detail?.language;
    if (nextLang && (nextLang === 'ru' || nextLang === 'en')) {
      currentLanguage = nextLang;
    }
  });
}

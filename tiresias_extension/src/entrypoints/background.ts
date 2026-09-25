import { defineBackground } from 'wxt/utils/define-background';
import { TiresiasApi } from '../lib/api';
import { initTelemetry } from '../lib/telemetry';

export default defineBackground(() => {
  initTelemetry('background');
  console.log('[Tiresias] Background worker active');

  const runtime = typeof browser !== 'undefined' && browser.runtime ? browser.runtime : (typeof chrome !== 'undefined' ? chrome.runtime : null);

  // Background proxy to bypass Mixed Content (HTTPS -> HTTP) and page CSP in content scripts
  if (runtime?.onMessage) {
    runtime.onMessage.addListener((message: any, sender: any, sendResponse: (resp: any) => void) => {
      if (message?.type === 'TIRESIAS_FETCH') {
        fetch(message.url, message.options)
          .then(async (resp) => {
            const ct = resp.headers.get('content-type') || '';
            let data: any;
            if (ct.includes('application/json')) {
              try {
                data = await resp.json();
              } catch {
                data = null;
              }
            } else {
              data = await resp.text();
            }
            sendResponse({
              ok: resp.ok,
              status: resp.status,
              statusText: resp.statusText,
              data,
            });
          })
          .catch((err) => {
            sendResponse({
              ok: false,
              status: 0,
              error: err?.message || String(err),
            });
          });
        return true; // Indicates async response
      }
    });
  }

  async function updateBadge() {
    try {
      const res = await TiresiasApi.checkHealth();
      if (typeof chrome !== 'undefined' && chrome.action) {
        if (res.ok) {
          chrome.action.setBadgeText({ text: 'ON' });
          chrome.action.setBadgeBackgroundColor({ color: '#3e9e49' });
        } else {
          chrome.action.setBadgeText({ text: 'OFF' });
          chrome.action.setBadgeBackgroundColor({ color: '#e45f5f' });
        }
      }
    } catch {}
  }

  updateBadge();
  setInterval(updateBadge, 60000);
});


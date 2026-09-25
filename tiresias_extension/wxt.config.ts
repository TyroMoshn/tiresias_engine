import { defineConfig } from 'wxt';

export default defineConfig({
  srcDir: 'src',
  manifestVersion: 3,
  suppressWarnings: {
    firefoxDataCollection: true,
  },
  vite: () => ({
    esbuild: {
      jsx: 'automatic',
      jsxImportSource: 'preact',
    },
    resolve: {
      alias: {
        react: 'preact/compat',
        'react-dom': 'preact/compat',
      },
    },
  }),
  manifest: {
    name: 'Tiresias - e926 / e621 AI Assistant',
    description: 'Personalized neural recommendations and smart boards for e926.net and e621.net',
    version: '1.0.0',
    permissions: ['storage'],
    icons: {
      16: '/icon/16.png',
      32: '/icon/32.png',
      48: '/icon/48.png',
      128: '/icon/128.png',
    },
    action: {
      default_title: 'Tiresias Settings & Status',
      default_icon: {
        16: '/icon/16.png',
        32: '/icon/32.png',
      },
    },
    host_permissions: [
      '*://*.e926.net/*',
      '*://*.e621.net/*',
      'http://localhost:8000/*',
      'http://127.0.0.1:8000/*',
    ],
    optional_host_permissions: [
      '*://*/*',
    ],
    browser_specific_settings: {
      gecko: {
        id: 'tiresias-booru@extension',
        strict_min_version: '128.0',
        update_url: 'https://raw.githubusercontent.com/USER/tiresias/main/updates.json',
      },
    },
  },
});


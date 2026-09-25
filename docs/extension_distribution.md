# Руководство по дистрибуции и подписанию расширения Tiresias

Данное руководство описывает процесс сборки, линтинга, подписания и распространения браузерного расширения **Tiresias** для Mozilla Firefox (Manifest V3), включая систему автономных автообновлений (self-hosted updates) и подписание через AMO (Mozilla Add-ons) API.

---

## 1. Архитектура и модель разрешений (Permissions)

### Модель безопасности Manifest V3
Для соответствия стандартам безопасности Mozilla Add-ons (AMO) и защиты приватности пользователей:
- Широкие разрешения подстановочных знаков (`http://*/*` и `https://*/*`) полностью исключены из `manifest.host_permissions`.
- Фиксированные `host_permissions` строго ограничены целевыми доменами Booru и локальной средой разработки:
  - `*://*.e926.net/*`
  - `*://*.e621.net/*`
  - `http://localhost:8000/*`
  - `http://127.0.0.1:8000/*`

### Динамические разрешения для кастомных серверов и VPS
Для пользователей, подключающих расширение к собственному удалённому серверу, домашнему VPS или локальной сети:
- В `wxt.config.ts` объявлено `optional_host_permissions: ['*://*/*']`.
- При переключении на адрес VPS или кастомного сервера на странице настроек (`SettingsPage.tsx`) расширение извлекает origin сервера (например, `http://remote-vps:8000/*`) и вызывает стандартный WebExtensions API:
  ```typescript
  await browser.permissions.request({ origins: [origin] });
  ```
- Браузер отображает стандартное диалоговое окно запроса доступа исключительно к указанному адресу по прямому действию пользователя. Это гарантирует прохождение аудита AMO и предотвращает ошибки сети (CORS / network access).

---

## 2. Локальная сборка и упаковка

Исходный код расширения находится в каталоге `root/tiresias_extension/`.

### Требования к окружению
- Node.js 20+ (LTS)
- npm 10+

### Основные скрипты `package.json`

| Команда | Действие | Результат / Файл |
|---|---|---|
| `npm run compile` | Проверка типов TypeScript без генерации кода (`tsc --noEmit`) | — |
| `npm run build:firefox` | Сборка релизного Manifest V3 бандла для Firefox | `.output/firefox-mv3/` |
| `npm run zip:firefox` | Формирование zip-архивов расширения и исходного кода | `.output/tiresias-extension-1.0.0-firefox.zip`<br>`.output/tiresias-extension-1.0.0-sources.zip` |
| `npm run lint:firefox` | Сборка и валидация манифеста через официальный `web-ext lint` | Отчёт линтера в терминале |
| `npm run sign:firefox` | Сборка и получение подписанного `.xpi` через API Mozilla AMO | `web-ext-artifacts/*.xpi` |

### Пошаговая локальная сборка

1. Перейдите в каталог расширения:
   ```bash
   cd tiresias_extension
   ```
2. Проверьте типизацию TypeScript:
   ```bash
   npm run compile
   ```
3. Запустите проверку официальным линтером Mozilla:
   ```bash
   npm run lint:firefox
   ```
4. Сформируйте zip-дистрибутив:
   ```bash
   npm run zip:firefox
   ```

---

## 3. Подписание через API Mozilla AMO (Unlisted канал)

Mozilla Firefox требует обязательной криптографической подписи для всех расширений перед их постоянной установкой в релизные версии браузера. Подписание по каналу **Unlisted** (вне каталога) предоставляет валидный цифровой сертификат в формате `.xpi`, не публикуя расширение в публичном каталоге AMO.

### Шаг 1: Получение API ключей AMO
1. Авторизуйтесь на портале [Mozilla Add-on Developer Hub](https://addons.mozilla.org/developers/).
2. Перейдите в раздел [Manage API Keys](https://addons.mozilla.org/developers/addon/api/key/).
3. Сгенерируйте новые учётные данные:
   - **JWT issuer** (`WEB_EXT_API_KEY`): строка вида `user:...`
   - **JWT secret** (`WEB_EXT_API_SECRET`): 64-значная шестнадцатеричная строка (hex)

### Шаг 2: Настройка переменных окружения
Задайте переменные в терминале или CI/CD:

**PowerShell (Windows):**
```powershell
$env:WEB_EXT_API_KEY = "user:12345678:999"
$env:WEB_EXT_API_SECRET = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
```

**Bash / Linux:**
```bash
export WEB_EXT_API_KEY="user:12345678:999"
export WEB_EXT_API_SECRET="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
```

### Шаг 3: Запуск подписания
Выполните команду:
```bash
npm run sign:firefox
```

Команда автоматически:
1. Собирает актуальный билд Firefox MV3 в `.output/firefox-mv3`.
2. Упаковывает и отправляет архив на сервис автоматической проверки и подписи Mozilla (канал `unlisted`).
3. Дожидается завершения автоматической валидации.
4. Скачивает готовый подписанный файл `.xpi` в каталог `tiresias_extension/web-ext-artifacts/`.

---

## 4. Автономные автообновления (Self-Hosted Updates через GitHub Releases)

Tiresias использует официальную спецификацию манифеста обновлений Mozilla для независимых расширений.

### Конфигурация механизма обновлений
В `wxt.config.ts` блок `browser_specific_settings.gecko` указывает:
```typescript
browser_specific_settings: {
  gecko: {
    id: 'tiresias-booru@extension',
    strict_min_version: '128.0',
    update_url: 'https://raw.githubusercontent.com/USER/tiresias/main/updates.json',
  },
}
```

### Структура манифеста обновлений (`updates.json`)
Манифест `root/updates.json` уведомляет Firefox о выходе новых версий:
```json
{
  "addons": {
    "tiresias-booru@extension": {
      "updates": [
        {
          "version": "1.0.0",
          "update_link": "https://github.com/USER/tiresias/releases/download/v1.0.0/tiresias-booru-1.0.0.xpi"
        }
      ]
    }
  }
}
```

### Порядок выпуска нового релиза (Release Workflow)
При публикации обновления:
1. Повысьте версию `"version"` в:
   - `root/tiresias_extension/package.json`
   - `root/tiresias_extension/wxt.config.ts`
2. Подпишите расширение:
   ```bash
   npm run sign:firefox
   ```
3. Создайте тег релиза Git (например, `v1.0.1`) на GitHub.
4. Прикрепите подписанный файл `.xpi` из `web-ext-artifacts/` к GitHub Release под именем `tiresias-booru-1.0.1.xpi`.
5. Обновите файл `root/updates.json` в ветке `main`, указав новую версию и прямую ссылку на скачивание актива из релиза.
6. Firefox в фоновом режиме периодически проверяет `update_url`, автоматически загружает обновленный `.xpi` и применяет его без необходимости переустановки.

---

## 5. Установка и проверка в стандартном релизном Firefox

Поскольку расширение подписано Mozilla AMO по каналу unlisted, оно может быть установлено напрямую в стандартные релизные версии Firefox (Firefox Release 128+ ESR или актуальные версии) без необходимости использовать Firefox Developer Edition, Nightly или менять системные флаги `xpinstall.signatures.required`.

### Инструкция по установке
1. Скачайте подписанный файл `.xpi` (например, `tiresias-booru-1.0.0.xpi`).
2. Запустите Mozilla Firefox.
3. Воспользуйтесь любым из двух способов:
   - **Перетаскивание (Drag & Drop)**: перетащите `.xpi` файл из проводника в любое открытое окно браузера Firefox.
   - **Менеджер дополнений**:
     - Откройте страницу `about:addons`.
     - Нажмите на иконку шестеренки (⚙️) в верхнем правом углу.
     - Выберите пункт **«Установить дополнение из файла...»** (Install Add-on From File...).
     - Выберите скачанный `.xpi` файл.
4. В появившемся системном диалоге Firefox нажмите **«Добавить»** (Add).

### Проверка работоспособности
1. Обратите внимание на иконку Tiresias на панели инструментов Firefox.
2. Бейдж индикатора состояния отображает:
   - 🟢 `ON`: успешное подключение к серверу Tiresias.
   - 🔴 `OFF`: сервер недоступен или соединение отклонено.
3. Кликните по иконке для открытия всплывающего окна (popup) или панели настроек:
   - Проверьте статус подключения в блоке **«Статус сервера»**.
   - При подключении к удаленному VPS введите URL и подтвердите всплывающий системный запрос разрешений браузера.
4. Откройте `https://e926.net` или `https://e621.net`:
   - Убедитесь, что виджеты рекомендаций, кнопки взаимодействия и персональная лента корректно отображаются на миниатюрах постов и страницах каталога.

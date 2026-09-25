# TIRESIAS Serving API (`tiresias_server`)

Высокопроизводительный, легковесный REST API бэкенд на базе **FastAPI** для генерации персональных рекомендаций в стиле Pinterest / Booru (целевая интеграция: **`e926.net`**). Разработан для эффективной работы в любых окружениях: от высокопроизводительных серверов (Dedicated / High-RAM Cloud VPS) до ультра-бюджетных маломощных виртуальных машин (Low-spec VPS / Edge-узлы от 1.5 ГБ ОЗУ и слабых CPU без AVX).

---

## 🏗️ Архитектура движка (4-стадийный конвейер)

Каждый запрос к ленте рекомендаций (`/api/v1/recommend/feed`) обрабатывается синглтоном `Engine` за четыре быстрых этапа со средней задержкой **$< 1$ мс** на производительных серверах и $\sim 20$–$50$ мс на бюджетных VPS:

```mermaid
flowchart TD
    subgraph S1["1. Генерация кандидатов (Retrieval — 4 канала)"]
        A["Channel A: CollabStore (Со-лайки)"] --> CAND["Пул кандидатов"]
        B["Channel B: FAISS SQ8 (Семантика 5.16M + Динамический альфа-микс)"] --> CAND
        C["Channel C: Taste Archetypes (Топ-посты архетипа)"] --> CAND
        D["Channel D: Precomputed Popular (Популярное по рейтингам)"] --> CAND
        FB["Fallback: Случайное сэмплирование mmaps"] -.-> CAND
    end

    subgraph S2["2. Жесткая фильтрация (Hard Filtering)"]
        CAND --> FLT{"Исключение"}
        FLT -->|Дизлайки и Hide| DROP["Отсечено"]
        FLT -->|Пользовательский блэклист тегов| DROP
        FLT -->|Цензурные маски rating_{s,q,e}.roar| DROP
        FLT -->|Типы медиа media_{image,video}.roar| DROP
        FLT -->|not_deleted.roar и запрет SWF| DROP
        FLT -->|Hard ban правил супрессии| DROP
        FLT -->|Порог min_score| DROP
        FLT -->|Прошли фильтр| POOL["Валидный пул"]
    end

    subgraph S3["3. Скоринг и штрафы (Scoring & Suppression)"]
        POOL --> E["Scorer: Вектор + Качество + Коллаб + Теги"]
        E --> SUP["Мягкий штраф за подавленные теги (Suppression Penalty)"]
        SUP --> F{"Ранее просмотрен (Seen)?"}
        F -->|Да| G["Мягкий decay: Score × 0.30"]
        F -->|Нет| H["Полный Score"]
    end

    subgraph S4["4. Freshness Tiering, диверсификация и пагинация"]
        H --> TIER_U["Tier 1: Непросмотренные (Unseen)"]
        G --> TIER_S["Tier 2: Просмотренные (Seen)"]
        TIER_U & TIER_S --> DIV["Защита от однообразия (≤ 2 поста автора подряд)"]
        DIV --> OUT["Итоговая лента FeedResponse"]
    end
```

### Ключевые компоненты `app/core/`:
- **`app/core/engine.py`** (хеш: `fe2dc039`): Главный синглтон движка рекомендаций, координатор 4-стадийного конвейера, адаптивное смешивание архетипов ($\alpha$-decay), интеграция с телеметрией, супрессией, аудитом и сервисным режимом (Maintenance Mode).
- **`app/core/mmaps.py`** (хеш: `86780c00`): Zero-copy доступ к метаданным постов (`post_ids.bin`, `score.bin`, `fav_count.bin`, `w.bin`, `h.bin`, `epoch_day.bin`, `is_deleted.bin`, `rating.bin`, `file_ext.bin`, `post_in_pools_count.bin`) через системный кэш страниц ОС.
- **`app/core/faiss_wrapper.py`** (хеш: `2b8f8d56`): Обертка над 8-битным квантованным индексом `post2vec_sq8.index` (5.16M векторов по 128 измерений). Поддерживает ограничение потоков OpenMP и реконструкцию векторов постов для расчета центроидов досок.
- **`app/core/bitmaps.py`** (хеш: `aa9aa079`): Побитовая фильтрация цензуры, типов медиа и удаленных постов через Roaring Bitmaps (`rating_{s,q,e}.roar`, `media_{image,video}.roar`, `not_deleted.roar`, 256 шардов инвертированного индекса тегов).
- **`app/core/board_engine.py`** (хеш: `cd034529`): Движок тематических Досок (Pinterest-like). Динамический центроид $\vec{c}_B$, расчет Affinity (косинусное сходство с доской), 5 режимов серверной сортировки, TF-IDF Salient Tags и рекомендации "More Like This Board" со 100% hard rejection постов доски.
- **`app/core/collab_store.py`** (хеш: `fcb59dd5`): 64 вкусовых архетипа (`taste_centroids.npy`, `taste_archetypes.parquet`) и граф со-лайков (`post_cofav.parquet`).
- **`app/core/scorer.py`** (хеш: `dabddb60`): Векторизованный многофакторный скоринг, мягкий decay для показов и дедупликация авторов.
- **`app/core/suppression.py`** (`SuppressionManager`): Двухуровневое подавление тегов холодного старта (`initial_suppression.json`): жесткая фильтрация (hard ban) и затухающий штраф со снятием по мере накопления лайков.
- **`app/core/diagnostics.py`** (`DiagnosticsAuditor`): Фоновый и ручной аудит целостности индексов, Mmap-массивов и рантайма.
- **`app/db/database.py`** (`Database`): Локальная SQLite `tiresias_user.db` в режиме `WAL` для мгновенного сохранения пользовательских действий без блокировок.
- **`app/db/telemetry.py`** (`TelemetryDatabase`): Локальная SQLite `tiresias_telemetry.db` в режиме `WAL` для сбора метрик производительности, аудита и клиентских ошибок.

---

## 🚀 Профили нагрузки («Масштабируемость без границ»)

Сервис поддерживает адаптивное масштабирование через переменную окружения `TIRESIAS_PROFILE`:

| Параметр | `performance` (Dedicated / High-Spec VPS) | `eco` (Бюджетный VPS / Ограниченные ресурсы) | Переменная окружения |
|---|---|---|---|
| **Кандидаты FAISS** | 300–500 | **80** | `TIRESIAS_FAISS_TOP_K` |
| **Кандидаты Collab** | 150–300 | **40** | `TIRESIAS_COLLAB_TOP_K` |
| **Кандидаты Popular**| 100–200 | **30** | `TIRESIAS_POPULAR_TOP_K`|
| **Лимит пула скоринга** | 500–1000 | **150** | `TIRESIAS_MAX_CANDIDATES`|
| **Потоки OpenMP FAISS**| Все ядра (Авто) | **2 потока** | `TIRESIAS_FAISS_THREADS` |
| **Лимит выдачи по умолч.**| 30 | **20** | `TIRESIAS_DEFAULT_LIMIT` |
| **Максимальный лимит** | 100 | **50** | `TIRESIAS_MAX_LIMIT` |

---

## 📡 Спецификация REST API

- Интерактивная документация Swagger UI: **`http://localhost:8000/docs`**
- Спецификация OpenAPI 3.1: **`docs/openapi.json`**
- Подробный справочник контрактов: **`docs/api_reference.md`**

### Основные группы эндпоинтов:
1. **Рекомендации (`/api/v1/recommend`)**:
   - `POST /feed` и `GET /feed`: Персонализированная лента постов (с поддержкой `media_types`).
   - `POST /similar` и `GET /similar`: Поиск визуально и семантически похожих постов по `post_id` (с фильтрацией `media_types`).
2. **Доски (`/api/v1/boards`)**:
   - `POST /` и `GET /`: Создание (возвращает `BoardSummary`) и получение списка досок пользователя.
   - `GET /{board_id}`: Просмотр доски с 5 режимами сортировки (`added_at`, `epoch_day`, `score`, `fav_count`, `affinity`), выделением Salient Tags и `BoardCoverage`.
   - `POST /{board_id}/posts` и `DELETE /{board_id}/posts/{post_id}`: Управление постами на доске.
   - `GET /{board_id}/recommend`: Рекомендации "More Like This Board" со 100% исключением сидов (параметр `ratings` через запятую).
   - `GET /{board_id}/export` и `POST /import`: Экспорт и импорт досок в JSON.
3. **Обратная связь (`/api/v1/feedback`)**:
   - `POST /`: Сигналы `like`, `dislike`, `hide`, `bookmark`, а также отмена действий (`undo_like`, `undo_hide`, `remove`).
   - `POST /seen`: Батч просмотренных постов (мягкий decay $\times 0.30$).
4. **Пользовательский профиль (`/api/v1/user`)**:
   - `POST /reset`: Сброс истории взаимодействий (возврат к чистому холодному старту).
   - `POST /archetype/override`: Принудительная фиксация вкусового архетипа и веса его влияния (тестерский режим).
   - `GET /{user_id}/profile`: Детальная статистика профиля, текущий архетип, метрики затухания $\alpha$ и активные правила супрессии.
   - `GET /{user_id}/feedback`: Пагинированная история оценок с обогащенными метаданными из mmap.
5. **Настройки (`/api/v1/settings`)**:
   - Черный список тегов (`/tag_blacklist`).
   - Пользовательские предпочтения (`/preferences`).
6. **Мониторинг, диагностика и обслуживание (`/api/v1/system`)**:
   - `GET /health` и `GET /stats`: Uptime, статус индексов (5.16M векторов), расход RAM RSS, статус диагностики.
   - `GET /diagnostics` и `POST /diagnostics/run`: Просмотр и принудительный запуск аудита целостности.
   - `POST /client-errors`, `GET /client-errors`, `DELETE /client-errors`: Сбор, просмотр и очистка клиентских JS-ошибок.
   - `POST /maintenance/enable`, `POST /maintenance/disable`: Управление сервисным режимом (HTTP 503).
   - `GET /suppression`, `POST /suppression/reload`: Просмотр и горячая перезагрузка правил супрессии без рестарта сервера.
   - `GET /archetypes`, `GET /archetypes/{archetype_id}`: Каталог вкусовых архетипов и просмотр постов кластера.

---

## 🛠️ Развертывание, управление и бенчмаркинг

### 1. Синхронизация боевого датасета на удаленный узел / VPS
Потоковая передача боевых артефактов через SSH без промежуточной буферизации в память:
```cmd
python root\tiresias_server\scripts\sync_data.py --profile serving --target staging
```
Проверка целостности по SHA-256:
```cmd
python root\tiresias_server\scripts\sync_data.py --verify-only --target staging
```

### 2. Управление демоном на удаленном сервере (systemd)
Скрипт управления системным юнитом `systemd --user`:
```cmd
python root\tiresias_server\scripts\manage_laptop_service.py status
python root\tiresias_server\scripts\manage_laptop_service.py start
python root\tiresias_server\scripts\manage_laptop_service.py stop
python root\tiresias_server\scripts\manage_laptop_service.py restart
python root\tiresias_server\scripts\manage_laptop_service.py logs
```

### 3. Нагрузочное стресс-тестирование и поиск утечек памяти
Асинхронный бенчмарк на базе `httpx`:
```cmd
python root\tiresias_server\scripts\stress_bench.py --url http://localhost:8000 --requests 500 --concurrency 10
```
Тест на утечки памяти (2 000 последовательных запросов):
```cmd
python root\tiresias_server\scripts\stress_bench.py --url http://localhost:8000 --leak-test
```

### 4. Развертывание в Docker / Cloud VPS
```bash
cd root/tiresias_server
docker compose up -d --build
```
Для проверки состояния контейнера:
```bash
docker compose logs -f
curl http://localhost:8000/api/v1/system/health
```

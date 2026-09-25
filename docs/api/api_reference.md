# TIRESIAS Serving API — Справочник контрактов

> **Версия API**: 0.3.0
> **Целевая интеграция**: [e926.net](https://e926.net) (безопасное зеркало e621ng)
>
> 📄 **Связанные модули маршрутизации**:
> - `tiresias_server/app/main.py` (хеш: `ab9518b1`, 2026-09-21)
> - `tiresias_server/app/routers/recommend.py` (хеш: `ec9b06a8`, 2026-09-21)
> - `tiresias_server/app/routers/boards.py` (хеш: `e6a7e6c0`, 2026-09-21)
> - `tiresias_server/app/routers/system.py` (хеш: `ff4856a6`, 2026-09-21)
> - `tiresias_server/app/routers/user.py` (хеш: `1c60906b`, 2026-09-22)

## 1. Общие сведения
- **Базовый URL локально**: `http://localhost:8000`
- **Формат данных**: JSON (`application/json`)
- **Интерактивный Swagger UI**: `http://localhost:8000/docs`
- **OpenAPI 3.1 JSON**: [`openapi.json`](./openapi.json)

---

## 2. Эндпоинты по группам

### Группа `RECOMMEND`

#### `POST /api/v1/recommend/feed`
**Описание**: Get Feed Post
Возвращает персонализированную ленту постов с многофакторным скорингом, учетом затухания архетипов и фильтрацией.

**Тело запроса (JSON):**
- Схема: `FeedRequest`
  - `user_id` (`string`, default: `"default_user"`)
  - `ratings` (`List[str]`, default: `["s", "q"]`)
  - `media_types` (`Optional[List[str]]`, default: `["image", "video"]`): допустимые типы медиа (`image`, `video`)
  - `limit` (`integer`, default: 30, ge: 1, le: 100)
  - `cursor` (`integer`, default: 0, ge: 0)
  - `min_score` (`integer`, default: 0)
  - `exclude_tags` (`Optional[List[int]]`, default: None)

**Ответы:**
- **`200`**: Successful Response (`FeedResponse` со списком `FeedItem`, содержащим поля `post_id`, `id`, `score`, `score_val`, `fav_count`, `rating`, `reasons`, `width`, `height`, `file_ext`, `is_video`)
- **`422`**: Validation Error

---

#### `GET /api/v1/recommend/feed`
**Описание**: Get Feed Get
GET version of /feed for easy testing in browser address bar.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `user_id` | `query` | `string` | Нет | Default: "default_user" |
| `limit` | `query` | `integer` | Нет | Default: 30, Min: 1, Max: 100 |
| `cursor` | `query` | `integer` | Нет | Default: 0, Min: 0 |
| `min_score` | `query` | `integer` | Нет | Default: 0 |
| `ratings` | `query` | `array of strings` | Нет | Repeated query param, e.g. `?ratings=s&ratings=q` |
| `media_types` | `query` | `array of strings` | Нет | Repeated query param, e.g. `?media_types=image&media_types=video` |

**Ответы:**
- **`200`**: Successful Response (`FeedResponse`)
- **`422`**: Validation Error

---

#### `POST /api/v1/recommend/similar`
**Описание**: Get Similar Post
Поиск визуально и семантически похожих постов на базе FAISS SQ8 и метаданных.

**Тело запроса (JSON):**
- Схема: `SimilarRequest`
  - `post_id` (`integer`): целевой идентификатор публикации
  - `ratings` (`List[str]`, default: `["s", "q"]`)
  - `media_types` (`Optional[List[str]]`, default: `["image", "video"]`)
  - `limit` (`integer`, default: 20, ge: 1, le: 50)

**Ответы:**
- **`200`**: Successful Response (`SimilarResponse` со списком `SimilarItem`, содержащим `post_id`, `id`, `similarity`, `score`, `fav_count`, `rating`, `width`, `height`, `file_ext`, `is_video`)
- **`422`**: Validation Error

---

#### `GET /api/v1/recommend/similar`
**Описание**: Get Similar Get
GET version of /similar for easy testing in browser address bar.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `post_id` | `query` | `integer` | Да | Target post ID to find similar posts for |
| `limit` | `query` | `integer` | Нет | Default: 20, Min: 1, Max: 50 |
| `ratings` | `query` | `array of strings` | Нет | Repeated query param, e.g. `?ratings=s&ratings=q` |
| `media_types` | `query` | `array of strings` | Нет | Repeated query param, e.g. `?media_types=image` |

**Ответы:**
- **`200`**: Successful Response (`SimilarResponse`)
- **`422`**: Validation Error

---

### Группа `BOARDS`

#### `POST /api/v1/boards`
**Описание**: Create Board
Creates a new board for a user.

**Тело запроса (JSON):**
- Схема: `BoardCreateRequest`

**Ответы:**
- **`201`**: Successful Response (`BoardSummary`: `board_id`, `id`, `user_id`, `name`, `title`, `post_count`, `created_at`, `updated_at`)
- **`422`**: Validation Error

---

#### `GET /api/v1/boards`
**Описание**: List Boards
Lists all boards belonging to a user, ordered by last updated.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `user_id` | `query` | `string` | Нет |  |

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `GET /api/v1/boards/{board_id}`
**Описание**: Get Board
Returns complete board details, salient theme tags, and hydrated posts
sorted according to the requested criteria (including semantic affinity to board centroid).

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `board_id` | `path` | `string` | Да |  |
| `sort_by` | `query` | `string` | Нет | Sort by: 'added_at', 'epoch_day', 'score', 'fav_count', 'affinity' |
| `order` | `query` | `string` | Нет | Sort order: 'desc' or 'asc' |
| `limit` | `query` | `integer` | Нет |  |
| `offset` | `query` | `integer` | Нет |  |

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `PATCH /api/v1/boards/{board_id}`
**Описание**: Update Board
Updates board title, description, or cover image.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `board_id` | `path` | `string` | Да |  |

**Тело запроса (JSON):**
- Схема: `BoardUpdateRequest`

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `DELETE /api/v1/boards/{board_id}`
**Описание**: Delete Board
Deletes a board and cascades removal of all post associations.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `board_id` | `path` | `string` | Да |  |

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `POST /api/v1/boards/{board_id}/posts`
**Описание**: Add Posts To Board
Adds one or more post IDs to the board.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `board_id` | `path` | `string` | Да |  |

**Тело запроса (JSON):**
- Схема: `BoardPostsAddRequest`

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `DELETE /api/v1/boards/{board_id}/posts/{post_id}`
**Описание**: Remove Post From Board
Removes a specific post from the board.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `board_id` | `path` | `string` | Да |  |
| `post_id` | `path` | `integer` | Да |  |

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `POST /api/v1/boards/{board_id}/recommend`
**Описание**: Recommend For Board Post
Generates recommendations matching the collective aesthetic of the board (POST method).
Guarantees hard exclusion of all posts already in the board.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `board_id` | `path` | `string` | Да |  |

**Тело запроса (JSON):**
- Схема: `BoardRecommendRequest`

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `GET /api/v1/boards/{board_id}/recommend`
**Описание**: Recommend For Board Get
Generates recommendations matching the collective aesthetic of the board (GET method).
Convenient for direct browser URL navigation and testing.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `board_id` | `path` | `string` | Да |  |
| `user_id` | `query` | `string` | Нет |  |
| `ratings` | `query` | `string` | Нет | Comma-separated list, e.g. 's,q' |
| `limit` | `query` | `integer` | Нет |  |
| `min_score` | `query` | `integer` | Нет |  |

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `GET /api/v1/boards/{board_id}/export`
**Описание**: Export Board
Exports board metadata and post IDs as a standalone portable JSON.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `board_id` | `path` | `string` | Да |  |

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `POST /api/v1/boards/import`
**Описание**: Import Board
Imports a board from a JSON payload into the specified user's profile.

**Тело запроса (JSON):**
- Схема: `BoardImportRequest`

**Ответы:**
- **`201`**: Successful Response
- **`422`**: Validation Error

---

### Группа `FEEDBACK`

#### `POST /api/v1/feedback`
**Описание**: Submit Feedback
Submits user feedback (like, dislike, hide, skip, save, bookmark, etc.).
Также поддерживает отмену ранее поставленных реакций (`signal_type` или алиас `action`):
`undo_like`, `undo_hide`, `unlike`, `undislike`, `remove`.

**Тело запроса (JSON):**
- Схема: `FeedbackRequest`
  - `user_id` (`string`, default: `"default_user"`)
  - `post_id` (`integer`)
  - `signal_type` (`string`, default: `"like"`): `"like"`, `"dislike"`, `"hide"`, `"skip"`, `"save"`, `"undo_like"`, `"undo_hide"`, `"remove"`

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `POST /api/v1/feedback/seen`
**Описание**: Submit Seen
Submits a batch of post IDs that have been displayed/seen by the user.
Enables soft decay (x0.3) so user sees fresh content without permanent blocking.

**Тело запроса (JSON):**
- Схема: `SeenBatchRequest`

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

### Группа `USER`

#### `POST /api/v1/user/reset`
**Описание**: Reset User History
Очищает историю реакций, просмотров и лайков тегов для указанного пользователя, списка пользователей или всех аккаунтов. Активирует чистый холодный старт.

**Тело запроса (JSON):**
- Схема: `UserResetRequest`
  - `user_id` (`Optional[str]`): ID профиля для сброса
  - `user_ids` (`Optional[List[str]]`): список пользователей для пакетного сброса
  - `reset_all` (`bool`, default: `false`): флаг полной очистки истории всех пользователей

**Ответы:**
- **`200`**: Successful Response (`UserResetResponse`: `success`, `user_id`, `users_affected`, `feedback_deleted`, `seen_deleted`, `tag_likes_deleted`, `message`)
- **`422`**: Validation Error

---

#### `POST /api/v1/user/archetype/override`
**Описание**: Override User Archetype
Устанавливает или снимает ручную фиксацию вкусового архетипа и принудительную силу его влияния для тестирования.

**Тело запроса (JSON):**
- Схема: `ArchetypeOverrideRequest`
  - `user_id` (`Optional[str]`, default: `"default_user"`)
  - `locked_archetype` (`Optional[int]`, диапазон 0..63): ID архетипа или null для авто-режима
  - `forced_weight` (`Optional[float]`, диапазон 0.0..1.0): фиксированный вес влияния или null для адаптивного расчета

**Ответы:**
- **`200`**: Successful Response (`ArchetypeOverrideResponse`: `success`, `user_id`, `taste_archetype_id`, `locked_archetype`, `forced_weight`, `effective_archetype_id`, `taste_coherence`, `archetype_influence`, `message`)
- **`400`**: Bad Request (архетип вне диапазона 0..63)
- **`422`**: Validation Error

---

#### `GET /api/v1/user/{user_id}/profile`
**Описание**: Get User Profile Stats
Возвращает полную статистику профиля: число реакций, лайки тегов, текущий вкусовой архетип, метрику когерентности $R$, динамическую силу влияния $\alpha$ и статус правил супрессии.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `user_id` | `path` | `string` | Да | Идентификатор профиля пользователя |

**Ответы:**
- **`200`**: Successful Response (JSON с полями `user_id`, `likes_count`, `dislikes_count`, `taste_archetype_id`, `taste_coherence`, `archetype_influence`, `suppression`)

---

#### `GET /api/v1/user/{user_id}/feedback`
**Описание**: Get User Feedback History
Возвращает пагинированную историю реакций пользователя с метаданными публикаций из Mmap (`score`, `fav_count`, `rating`, `file_ext`, `width`, `height`, `is_video`).

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `user_id` | `path` | `string` | Да | Идентификатор профиля пользователя |
| `signal_type` | `query` | `string` | Нет | Фильтр по типу реакции (`like`, `dislike`) |
| `limit` | `query` | `integer` | Нет | Default: 50, Min: 1, Max: 200 |
| `offset` | `query` | `integer` | Нет | Default: 0, Min: 0 |

**Ответы:**
- **`200`**: Successful Response (JSON с пагинацией и списком обогащенных записей `items`)

---

### Группа `SETTINGS`

#### `GET /api/v1/settings/tag_blacklist`
**Описание**: Get Tag Blacklist

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `user_id` | `query` | `string` | Нет |  |

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `PUT /api/v1/settings/tag_blacklist`
**Описание**: Add Tag Blacklist

**Тело запроса (JSON):**
- Схема: `TagBlacklistRequest`

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `DELETE /api/v1/settings/tag_blacklist`
**Описание**: Remove Tag Blacklist

**Тело запроса (JSON):**
- Схема: `TagBlacklistRequest`

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `GET /api/v1/settings/preferences`
**Описание**: Get Preferences

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `user_id` | `query` | `string` | Нет |  |

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

#### `PUT /api/v1/settings/preferences`
**Описание**: Update Preferences

**Тело запроса (JSON):**
- Схема: `PreferencesRequest`

**Ответы:**
- **`200`**: Successful Response
- **`422`**: Validation Error

---

### Группа `SYSTEM`

#### `GET /api/v1/system/health`
**Описание**: Get Health
Возвращает статус здоровья, загрузки индексов и расхода памяти.

**Ответы:**
- **`200`**: Successful Response (`HealthResponse`: `status`, `uptime_seconds`, `memory_rss_mb`, `total_posts_indexed`, `faiss_ready`, `faiss_total_vectors`, `collab_archetypes_loaded`, `collab_centroids_loaded`, `database_path`, `maintenance`, `sanity_warnings`, `diagnostics_status`, `telemetry_errors_count`)

---

#### `GET /api/v1/system/stats`
**Описание**: Get Stats
Возвращает расширенные системные метрики и состояние движка.

**Ответы:**
- **`200`**: Successful Response

---

#### `GET /api/v1/system/diagnostics`
**Описание**: Get Diagnostics
Возвращает последний сформированный отчет комплексного аудита целостности индексов, Mmap и рантайма.

**Ответы:**
- **`200`**: Successful Response (JSON)

---

#### `POST /api/v1/system/diagnostics/run`
**Описание**: Run Diagnostics
Принудительно запускает свежий аудит диагностики компонентов сервиса.

**Ответы:**
- **`200`**: Successful Response (JSON)

---

#### `POST /api/v1/system/client-errors`
**Описание**: Report Client Error
Принимает и сохраняет отчеты о JavaScript-ошибках браузерного интерфейса.

**Тело запроса (JSON):**
- Схема: `ClientErrorCreate` (`user_id`, `error_type`, `message`, `stack`, `url`, `source_file`, `lineno`, `colno`, `metadata`)

**Ответы:**
- **`200`**: Successful Response (`{"status": "ok", "error_id": int}`)

---

#### `GET /api/v1/system/client-errors`
**Описание**: Get Client Errors
Возвращает список последних зарегистрированных клиентских ошибок для анализа.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `limit` | `query` | `integer` | Нет | Default: 50, Min: 1, Max: 200 |

**Ответы:**
- **`200`**: Successful Response (`ClientErrorsResponse`: `total_errors`, `errors: List[ClientErrorItem]`)

---

#### `DELETE /api/v1/system/client-errors`
**Описание**: Clear Client Errors
Очищает журнал клиентских ошибок в базе телеметрии.

**Ответы:**
- **`200`**: Successful Response (`{"status": "ok", "cleared_count": int}`)

---

#### `POST /api/v1/system/maintenance/enable`
**Описание**: Enable Maintenance Mode
Переводит сервер в сервисный режим. Запросы к ленте возвращают HTTP 503 Service Unavailable.

**Ответы:**
- **`200`**: Successful Response (`{"status": "ok", "maintenance": true, "message": "Maintenance mode enabled"}`)

---

#### `POST /api/v1/system/maintenance/disable`
**Описание**: Disable Maintenance Mode
Отключает сервисный режим и возобновляет штатную выдачу рекомендаций.

**Ответы:**
- **`200`**: Successful Response (`{"status": "ok", "maintenance": false, "message": "Maintenance mode disabled"}`)

---

#### `GET /api/v1/system/suppression`
**Описание**: Get Suppression Status
Возвращает текущую конфигурацию и правила подавления нежелательных тегов для холодного старта.

**Ответы:**
- **`200`**: Successful Response (JSON)

---

#### `POST /api/v1/system/suppression/reload`
**Описание**: Reload Suppression Rules
Выполняет горячую перезагрузку конфигурационного файла `initial_suppression.json` без перезапуска сервера.

**Ответы:**
- **`200`**: Successful Response (JSON)

---

#### `GET /api/v1/system/archetypes`
**Описание**: List Archetypes
Возвращает обзорный каталог всех 64 вкусовых архетипов с количеством связанных постов.

**Ответы:**
- **`200`**: Successful Response (JSON с полями `total_archetypes`, `archetypes`)

---

#### `GET /api/v1/system/archetypes/{archetype_id}`
**Описание**: Get Archetype Detail
Возвращает детальную выборку постов и распределение рейтингов (`s`, `q`, `e`) для конкретного вкусового архетипа.

**Параметры запроса (Query / Path):**
| Параметр | Расположение | Тип | Обязательный | Описание |
|---|---|---|---|---|
| `archetype_id` | `path` | `integer` | Да | Идентификатор вкусового архетипа (0..63) |
| `limit` | `query` | `integer` | Нет | Default: 100, Min: 1, Max: 500 |

**Ответы:**
- **`200`**: Successful Response (JSON с полями `archetype_id`, `total_posts`, `rating_distribution`, `items`)

---

### Группа `ROOT`

#### `GET /`
**Описание**: Root
Базовый эндпоинт проверки доступности сервиса.

**Ответы:**
- **`200`**: Successful Response

---

# Документация Serving-сервиса (`tiresias_server`)

Полное описание архитектуры, спецификация REST API, описание профилей нагрузки и руководство по развертыванию вынесены в основной файл документации пакета:

👉 **[Документация TIRESIAS Serving API (README.md)](../root/tiresias_server/README.md)**

## Краткая сводка:
- **Назначение**: Легковесный бэкенд на FastAPI для генерации персональной ленты, поиска похожих постов, тематических досок и холодного старта с подавлением нежелательных тегов.
- **Входные артефакты**:
  - `mmaps/*.bin` (`post_ids.bin`, `score.bin`, `fav_count.bin`, `w.bin`, `h.bin`, `epoch_day.bin`, `is_deleted.bin`, `rating.bin`, `file_ext.bin`, `post_in_pools_count.bin`).
  - Системные Roaring-маски: `rating_{s,q,e}.roar`, `media_{image,video}.roar`, `not_deleted.roar`.
  - Инвертированные индексы тегов: `bitmaps/shard_*.roarpack`, `bitmaps/index_*.bin`.
  - Векторные и коллаборативные модели: `features/post2vec_faiss_ids.parquet`, `features/post2vec_sq8.index`, `features/taste_centroids.npy`, `features/taste_archetypes.parquet`, `features/post_cofav.parquet`.
  - Конфигурация супрессии холодного старта: `initial_suppression.json`.
  - BoardEngine также использует `posts_parquet/` и `tags.parquet`.
- **Локальные БД и телеметрия**:
  - `data/tiresias_user.db`: SQLite в WAL-режиме (лайки, дизлайки, блэклист тегов, история показов, переопределение архетипов).
  - `data/tiresias_telemetry.db`: SQLite в WAL-режиме (аудит производительности, системные метрики и журнал клиентских JS-ошибок).
- **Латентность**: $< 1$ мс на запрос на сервере, $\sim 20$–$50$ мс на бюджетном Celeron без AVX.
- **Потребление памяти**: $< 1.0$ ГБ RAM на базе из ~5.16 млн активных индексированных постов.
- **Кросс-платформенность**: Протестирован и стабильно работает на Windows 11 и Linux Lite (Intel Celeron N2840 без AVX).

# Документация TIRESIAS

В этом каталоге собрана полная техническая документация системы рекомендаций Tiresias. Документация разбита по функциональным разделам:

---

## 🏛️ Архитектура и сквозные графы (`architecture/`)
- [`codebase_dependency_graph.md`](architecture/codebase_dependency_graph.md) — Сквозной граф зависимостей всех трёх ролей системы (`build_index`, `tiresias_server`, `tiresias_extension`), правила взаимосвязей и карта инвариантов.
- [`project_overview.md`](architecture/project_overview.md) — Общий обзор концепции проекта, назначение компонентов и стек технологий.
- [`tiresias_server.md`](architecture/tiresias_server.md) — Архитектура серверного демона обслуживания рекомендаций.

---

## 🔌 Серверный API (`api/`)
- [`api_reference.md`](api/api_reference.md) — Полный справочник REST API v0.3.0 для интеграции с Booru-платформами (`/feed`, `/similar`, `/boards`, `/feedback`, `/user`, `/system`).
- [`openapi.json`](api/openapi.json) — Машиночитаемая спецификация OpenAPI 3.1.

---

## 🧩 Браузерное расширение (`extension/`)
- [`extension_distribution.md`](extension/extension_distribution.md) — Руководство по сборке, самохостингу (`updates.json`) и публикации WebExtension Manifest V3 под Firefox и Chrome.

---

## ⚙️ Конвейер индексации данных (`pipeline/`)
Подробный разбор 19 стадий математического пайплайна `build_index`:
- [`main.md`](pipeline/main.md) — Главный оркестратор конвейера и CLI-интерфейс.
- [`config.md`](pipeline/config.md) — Конфигурация путей, гиперпараметров и весов категорий тегов.
- [`io_stage.md`](pipeline/io_stage.md) — Стадия 1: Парсинг сырых дампов `posts.csv` в партиционированный Parquet (DuckDB).
- [`tags_stage.md`](pipeline/tags_stage.md) — Стадии 2–4: Разрешение алиасов, импликаций тегов и шардирование `post_tags`.
- [`stats_stage.md`](pipeline/stats_stage.md) — Стадии 7, 8, 14: Расчёт сглаженного IDF, PPMI матриц и varint-сжатых Top-K списков.
- [`pools_stage.md`](pipeline/pools_stage.md) — Стадии 9–13: Анализ энтропии пулов, серийных комиксов и графа связей.
- [`uploaders_extract.md`](pipeline/uploaders_extract.md) — Стадия 15: Агрегация наборов загрузок по аккаунтам.
- [`tag2vec_stage.md`](pipeline/tag2vec_stage.md) — Стадия 16: Truncated SVD над PPMI-матрицей для получения 128D эмбеддингов тегов.
- [`post2vec_stage.md`](pipeline/post2vec_stage.md) — Стадия 17: Агрегация векторов постов и квантование FAISS SQ8.
- [`collab_stage.md`](pipeline/collab_stage.md) — Стадии 18–19: Граф со-фаворитов (`post_cofav`) и 64 архетипа вкусов (Spherical MiniBatchKMeans).
- [`index_stage.md`](pipeline/index_stage.md) — Стадии 5–6: Сборка zero-copy массивов `data/mmaps/*.bin` и 256 шардов Roaring Bitmaps.
- [`graphs.md`](pipeline/graphs.md) — Реализация алгоритма Тарьяна для поиска циклов в графах синонимов.
- [`utils.md`](pipeline/utils.md) — Вспомогательные утилиты сжатия (LEB128 varint) и файловые хелперы.

---

## 🧪 Автотесты и валидация (`autotests/`)
- [`autotest_index.md`](autotests/autotest_index.md) — Валидация канонической биекции `post_ids.bin` и плотных битмапов.
- [`autotest_post2vec.md`](autotests/autotest_post2vec.md) — Тестирование качества векторного поиска FAISS SQ8.
- [`autotest_tag2vec.md`](autotests/autotest_tag2vec.md) — Валидация семантических расстояний в векторном пространстве тегов.
- [`visualize_post2vec_cpu.md`](autotests/visualize_post2vec_cpu.md) — Визуализация распределения эмбеддингов постов.
- [`thematic_samples.md`](autotests/thematic_samples.md) — Тестовые выборки постов для проверки тематических досок.

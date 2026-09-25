# Модуль `io_stage.py` (Техническая документация)

## 1. Назначение и Архитектурная роль
Модуль `io_stage.py` реализует первый этап ETL-пайплайна (`step_parquet`) — миграцию и нормализацию сырого неструктурированного текстового дампа `posts.csv` (порядка 6 миллионов строк, объем ~5 ГБ) в колоночный, строго типизированный, сжатый формат Apache Parquet с многоуровневым Hive-партиционированием.

## 2. Технологический стек и движок исполнения
Преобразование выполняется без загрузки всего датасета в память Python посредством встроенного аналитического SQL-движка **DuckDB** (`duckdb.connect()`). DuckDB обеспечивает:
- Векторизованное чтение CSV чанками (`read_csv_auto(..., SAMPLE_SIZE=-1, ALL_VARCHAR=1)`).
- Эффективное использование многопоточности процессора (`PRAGMA threads=cfg.workers`).
- Потоковую запись в Parquet с одновременным сжатием Zstandard (ZSTD) и директориальным партиционированием.

## 3. Схема данных и типизация (`_EXPECTED`)
Каждая строка CSV содержит 29 потенциальных полей, которые валидируются и приводятся к каноническим типам. Если колонка отсутствует в заголовке CSV, подставляется безопасный `NULL`- или `default`-литерал:

| Имя колонки | SQL Cast / Выражение | Значение по умолчанию | Физический тип |
| :--- | :--- | :--- | :--- |
| `id` | `CAST(id AS BIGINT)` | `CAST(NULL AS BIGINT)` | `INT64` (Первичный ключ) |
| `uploader_id` | `CAST(uploader_id AS BIGINT)` | `CAST(NULL AS BIGINT)` | `INT64` |
| `created_at` | `CAST(created_at AS TIMESTAMP)`| `CAST(NULL AS TIMESTAMP)`| `TIMESTAMP` (UTC) |
| `md5` | `md5` | `NULL` | `VARCHAR` |
| `source` | `source` | `NULL` | `VARCHAR` |
| `rating` | `rating` | `NULL` | `VARCHAR` (`'s'`, `'q'`, `'e'`) |
| `image_width`, `image_height` | `CAST(... AS INTEGER)` | `CAST(NULL AS INTEGER)` | `INT32` |
| `tag_string` | `tag_string` | `''` | `VARCHAR` |
| `locked_tags` | `locked_tags` | `''` | `VARCHAR` |
| `fav_count` | `CAST(fav_count AS INTEGER)` | `CAST(0 AS INTEGER)` | `INT32` |
| `file_ext` | `file_ext` | `NULL` | `VARCHAR` |
| `parent_id` | `NULLIF(parent_id,'')::BIGINT` | `CAST(NULL AS BIGINT)` | `INT64` |
| `change_seq` | `CAST(change_seq AS BIGINT)` | `CAST(NULL AS BIGINT)` | `INT64` |
| `approver_id` | `NULLIF(approver_id,'')::BIGINT`| `CAST(NULL AS BIGINT)` | `INT64` |
| `file_size` | `CAST(file_size AS BIGINT)` | `CAST(NULL AS BIGINT)` | `INT64` (в байтах) |
| `comment_count` | `CAST(comment_count AS INTEGER)`| `CAST(0 AS INTEGER)` | `INT32` |
| `description` | `description` | `''` | `VARCHAR` |
| `duration` | `NULLIF(duration,'')` | `NULL` | `VARCHAR` |
| `updated_at` | `CAST(updated_at AS TIMESTAMP)`| `CAST(NULL AS TIMESTAMP)`| `TIMESTAMP` |
| `is_deleted` | `(is_deleted='t')` | `CAST(FALSE AS BOOLEAN)` | `BOOLEAN` |
| `is_pending` | `(is_pending='t')` | `CAST(FALSE AS BOOLEAN)` | `BOOLEAN` |
| `is_flagged` | `(is_flagged='t')` | `CAST(FALSE AS BOOLEAN)` | `BOOLEAN` |
| `score`, `up_score`, `down_score` | `CAST(... AS INTEGER)` | `CAST(0 AS INTEGER)` | `INT32` |
| `is_rating_locked`, `is_status_locked`, `is_note_locked` | `(...='t')` | `CAST(FALSE AS BOOLEAN)` | `BOOLEAN` |
| `year` *(синтетическое)* | `strftime(CAST(created_at AS TIMESTAMP), '%Y')` | — | `VARCHAR(4)` |
| `month` *(синтетическое)* | `strftime(CAST(created_at AS TIMESTAMP), '%m')` | — | `VARCHAR(2)` |

## 4. Партиционирование файловой системы
Запись производится с помощью команды `COPY (...) TO ... (FORMAT PARQUET, PARTITION_BY (rating, year, month), COMPRESSION ZSTD)`.

Структура каталогов на диске:
```text
posts_parquet/
  ├─ _SUCCESS
  ├─ rating=s/
  │   ├─ year=2007/
  │   │   ├─ month=01/
  │   │   │   └─ data_0.parquet
  │   │   └─ month=12/
  │   └─ ...
  ├─ rating=q/
  └─ rating=e/
```

### Преимущества партиционирования:
1. **Partition Pruning (отсечение разделов)**: При анализе только безопасного контента (`rating=s`) или определенного временного среза движки Polars/DuckDB физически не читают терабайты данных с диска.
2. **Ограничение размера файла**: Файлы разбиваются на порции по ~10–50 МБ, что идеально ложится в кэш процессора и параллельные потоки чтения.

## 5. Идемпотентность и синхронизация
В каталоге `posts_parquet/` создается сигнальный файл-маркер `_SUCCESS`. При повторном запуске функция `newer_than(sentinel, cfg.csv)` сверяет время изменения маркера и файла `posts.csv`. Если исходник не изменялся и флаг `cfg.force == False`, конвертация мгновенно пропускается.

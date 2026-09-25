#!/usr/bin/env python3
"""
TIRESIAS ENGINE - Documentation Freshness & Hash Checker
Parses Markdown documentation for referenced code files and their hashes,
verifying that documentation accurately reflects the current state of code.
Supports cross-platform normalized hashing (CRLF/LF agnostic).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Matches table row: | `path/to/file.ext` | `hash` | ... |
TABLE_PATTERN = re.compile(
    r'\|\s*`?([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9_]+)`?\s*\|\s*`?([a-fA-F0-9]{7,64})`?\s*\|'
)

# Matches inline pattern: `path/to/file.ext` (hash: `abc12345`...) or (хеш: `abc12345`...)
INLINE_PATTERN = re.compile(
    r'`([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9_]+)`(?:\*\*)?\s*\((?:хеш|hash|sha256):\s*`?([a-fA-F0-9]{7,64})`?'
)

# Matches explicit comment: <!-- doc-hash: path/to/file.ext:hash -->
COMMENT_PATTERN = re.compile(
    r'<!--\s*doc-hash:\s*([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9_]+):([a-fA-F0-9]{7,64})\s*-->'
)


class DocRef(NamedTuple):
    doc_path: Path
    line_number: int
    target_rel_path: str
    documented_hash: str
    raw_match: str


def compute_file_hash(file_path: Path, length: int = 8) -> Optional[str]:
    """Computes cross-platform normalized SHA-256 hash (CRLF/LF agnostic)."""
    if not file_path.is_file():
        return None
    try:
        raw = file_path.read_bytes()
        try:
            text = raw.decode("utf-8")
            raw = text.replace("\r\n", "\n").encode("utf-8")
        except UnicodeDecodeError:
            pass  # Binary file, keep raw bytes
        return hashlib.sha256(raw).hexdigest()[:length]
    except Exception as e:
        return None


def find_all_markdown_files(root: Path) -> List[Path]:
    """Finds all documentation markdown files excluding vendor/cache directories."""
    md_files = []
    ignored = {
        "node_modules", ".git", ".venv", "venv", "env",
        "__pycache__", ".output", "dist", "build", "data"
    }
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place so os.walk never traverses them
        dirnames[:] = [d for d in dirnames if d not in ignored]
        for f in filenames:
            if f.lower().endswith(".md"):
                md_files.append(Path(dirpath) / f)
    return sorted(md_files)


def extract_refs_from_file(md_path: Path) -> List[DocRef]:
    """Extracts all code file hash references from a markdown file."""
    refs = []
    try:
        lines = md_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return refs

    for idx, line in enumerate(lines, start=1):
        # 1. Check comment pattern
        for m in COMMENT_PATTERN.finditer(line):
            refs.append(DocRef(md_path, idx, m.group(1).replace("\\", "/"), m.group(2).lower(), m.group(0)))

        # 2. Check table pattern
        for m in TABLE_PATTERN.finditer(line):
            refs.append(DocRef(md_path, idx, m.group(1).replace("\\", "/"), m.group(2).lower(), m.group(0)))

        # 3. Check inline pattern
        for m in INLINE_PATTERN.finditer(line):
            refs.append(DocRef(md_path, idx, m.group(1).replace("\\", "/"), m.group(2).lower(), m.group(0)))

    return refs


def resolve_target_file(ref_path_str: str, doc_dir: Path) -> Optional[Path]:
    """Resolves target file relative to PROJECT_ROOT or doc_dir."""
    candidates = [
        PROJECT_ROOT / ref_path_str,
        doc_dir / ref_path_str,
        PROJECT_ROOT / "root" / ref_path_str if (PROJECT_ROOT / "root").exists() else None,
    ]
    for cand in candidates:
        if cand and cand.is_file():
            return cand.resolve()
    return None


def check_docs(target_doc: Optional[Path] = None, update_hashes: bool = False) -> int:
    """Verifies or updates documentation file hashes."""
    if target_doc:
        docs = [target_doc]
    else:
        docs = find_all_markdown_files(PROJECT_ROOT)

    total_refs = 0
    ok_count = 0
    changed_count = 0
    missing_count = 0

    print("=" * 68)
    print("        TIRESIAS ENGINE — DOCUMENTATION FRESHNESS CHECK")
    print("=" * 68)

    files_to_update: Dict[Path, str] = {}

    for doc in docs:
        refs = extract_refs_from_file(doc)
        if not refs:
            continue

        rel_doc = doc.relative_to(PROJECT_ROOT)
        print(f"\nDocument: \033[1m{rel_doc}\033[0m ({len(refs)} tracked references)")

        content = doc.read_text(encoding="utf-8") if update_hashes else ""

        for ref in refs:
            total_refs += 1
            target_path = resolve_target_file(ref.target_rel_path, doc.parent)

            if not target_path or not target_path.exists():
                print(f"  \033[91m[MISSING]\033[0m {ref.target_rel_path:<32} (file not found on disk)")
                missing_count += 1
                continue

            current_hash = compute_file_hash(target_path, length=len(ref.documented_hash))

            if current_hash == ref.documented_hash:
                print(f"  \033[92m[OK]\033[0m      {ref.target_rel_path:<32} (hash: {ref.documented_hash})")
                ok_count += 1
            else:
                print(f"  \033[93m[CHANGED]\033[0m {ref.target_rel_path:<32} "
                      f"(doc: {ref.documented_hash} != disk: {current_hash})")
                changed_count += 1

                if update_hashes and current_hash:
                    # Update hash in content
                    old_str = ref.documented_hash
                    # Replace only within this specific reference pattern
                    new_ref_str = ref.raw_match.replace(old_str, current_hash)
                    content = content.replace(ref.raw_match, new_ref_str, 1)

        if update_hashes and doc not in files_to_update and content != doc.read_text(encoding="utf-8"):
            files_to_update[doc] = content

    if update_hashes and files_to_update:
        print("\nUpdating documentation files with fresh hashes...")
        for doc_path, new_content in files_to_update.items():
            doc_path.write_text(new_content, encoding="utf-8")
            print(f"  \033[92m[UPDATED]\033[0m {doc_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "-" * 68)
    print(f"Summary: {ok_count} OK, {changed_count} changed/outdated, {missing_count} missing.")

    if changed_count == 0 and missing_count == 0:
        if total_refs > 0:
            print("\033[92mAll documented files are 100% up to date!\033[0m")
        else:
            print("No hash-tracked file references found in scanned documents.")
        print("-" * 68)
        return 0
    else:
        if update_hashes:
            print("\033[92mDocumentation hashes updated to current codebase state.\033[0m")
            print("-" * 68)
            return 0
        else:
            print("\033[93mWarning: Some code files have changed since documentation was written.\033[0m")
            print("Tip: Run with --update to refresh hashes in documentation files.")
            print("-" * 68)
            return 1


CORE_TRACKED_FILES = [
    # Tools & Orchestration
    ("tools/updater.py", "Оркестратор полного цикла обновления артефактов и обслуживания"),
    ("tools/run_updater.bat", "Переносимый лаунчер обновления (Windows)"),
    ("tools/run_updater.sh", "Переносимый лаунчер обновления (Linux / POSIX)"),
    ("tools/check_docs.py", "Инструмент проверки актуальности кодовой базы и документации"),
    ("tools/check_docs.bat", "Лаунчер проверки свежести документации (Windows)"),
    ("tools/check_docs.sh", "Лаунчер проверки свежести документации (Linux)"),
    ("run_updater.bat", "Корневой forwarder лаунчера обновления (Windows)"),
    ("run_updater.sh", "Корневой forwarder лаунчера обновления (Linux)"),
    ("check_docs.bat", "Корневой forwarder проверки документации (Windows)"),
    ("check_docs.sh", "Корневой forwarder проверки документации (Linux)"),

    # Serving Engine (FastAPI)
    ("tiresias_server/app/main.py", "Точка входа Serving FastAPI сервера"),
    ("tiresias_server/app/core/engine.py", "Главный синглтон движка рекомендаций"),
    ("tiresias_server/app/core/mmaps.py", "Zero-copy доступ к метаданным постов через системный mmap"),
    ("tiresias_server/app/core/faiss_wrapper.py", "Обертка векторного индекса FAISS SQ8"),
    ("tiresias_server/app/core/bitmaps.py", "Побитовая Roaring-фильтрация цензуры и тегов"),
    ("tiresias_server/app/core/board_engine.py", "Движок тематических Досок (центриды, affinity, salient tags)"),
    ("tiresias_server/app/core/scorer.py", "Многофакторный скоринг кандидатов и decay"),
    ("tiresias_server/app/core/collab_store.py", "Хранилище вкусовых архетипов и графа со-лайков"),
    ("tiresias_server/app/routers/recommend.py", "Эндпоинты персональной ленты (/feed, /similar)"),
    ("tiresias_server/app/routers/boards.py", "Эндпоинты тематических досок и коллекций"),
    ("tiresias_server/app/routers/system.py", "Эндпоинты здоровья, профилей и режима техобслуживания"),

    # Build Index Pipeline
    ("build_index/main.py", "Главный оркестратор DAG-конвейера сборки индексов"),
    ("build_index/io_stage.py", "Конвертация сырых CSV в партиционированный Parquet"),
    ("build_index/tag2vec_stage.py", "Обучение эмбеддингов тегов tag2vec через SVD / KNN"),
    ("build_index/post2vec_stage.py", "Агрегация векторов постов и квантование FAISS SQ8"),
    ("build_index/uploaders_extract.py", "Скоростное извлечение статистики авторов через DuckDB"),
]


def dump_manifest_table() -> None:
    """Prints a Markdown table with current hashes of core files for embedding into docs."""
    import datetime
    today = datetime.date.today().isoformat()

    print("| Файл | Хеш (SHA-256) | Дата фиксации | Описание |")
    print("| :--- | :--- | :--- | :--- |")
    for rel_path, desc in CORE_TRACKED_FILES:
        target = PROJECT_ROOT / rel_path
        if target.is_file():
            h = compute_file_hash(target, length=8) or "unknown"
            print(f"| `{rel_path}` | `{h}` | {today} | {desc} |")


def main():
    parser = argparse.ArgumentParser(description="Check or update documentation code references and hashes.")
    parser.add_argument("--file", "-f", type=str, default=None, help="Check a specific markdown file")
    parser.add_argument("--update", "-u", action="store_true", help="Update stale hashes in documentation")
    parser.add_argument("--dump", "-d", action="store_true", help="Print Markdown table of core tracked files and hashes")
    parser.add_argument("--calc", "-c", nargs="+", help="Calculate normalized hash for specified files")

    args = parser.parse_args()

    if args.dump:
        dump_manifest_table()
        sys.exit(0)

    if args.calc:
        for f in args.calc:
            p = Path(f).resolve()
            h = compute_file_hash(p, length=8) if p.is_file() else "NOT_FOUND"
            print(f"{f}: {h}")
        sys.exit(0)

    doc_file = Path(args.file).resolve() if args.file else None
    sys.exit(check_docs(target_doc=doc_file, update_hashes=args.update))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exports OpenAPI JSON specification and generates Markdown API Reference.
Prepares strict contracts for client integrations (Phase 6 browser extensions for e926.net).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
ROOT_DIR = SERVER_DIR.parent
DOCS_DIR = ROOT_DIR / "docs"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def export_openapi() -> int:
    os.environ["TIRESIAS_PROFILE"] = "eco"  # light init

    from app.main import app

    openapi_data = app.openapi()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DOCS_DIR / "openapi.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(openapi_data, f, indent=2, ensure_ascii=False)
    print(f"[PASS] Exported OpenAPI specification: {json_path}")

    # Generate Markdown API Reference
    md_path = DOCS_DIR / "api_reference.md"
    generate_markdown_docs(openapi_data, md_path)
    print(f"[PASS] Generated Markdown API Reference: {md_path}")
    return 0


def generate_markdown_docs(spec: dict, out_path: Path) -> None:
    lines = [
        "# TIRESIAS Serving API — Справочник контрактов",
        "",
        "> **Версия API**: " + spec.get("info", {}).get("version", "0.3.0"),
        "> **Целевая интеграция**: [e926.net](https://e926.net) (безопасное зеркало e621ng)",
        "",
        "## 1. Общие сведения",
        "- **Базовый URL локально**: `http://localhost:8000`",
        "- **Формат данных**: JSON (`application/json`)",
        "- **Интерактивный Swagger UI**: `http://localhost:8000/docs`",
        "- **OpenAPI 3.1 JSON**: [`openapi.json`](./openapi.json)",
        "",
        "---",
        "",
        "## 2. Эндпоинты по группам",
        "",
    ]

    paths = spec.get("paths", {})

    # Group endpoints by tag
    tags_map: dict[str, list[tuple[str, str, dict]]] = {}
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            tags = details.get("tags", ["other"])
            tag = tags[0] if tags else "other"
            tags_map.setdefault(tag, []).append((method.upper(), path, details))

    tag_order = ["recommend", "boards", "feedback", "settings", "system"]
    sorted_tags = sorted(
        tags_map.keys(),
        key=lambda t: tag_order.index(t) if t in tag_order else 99,
    )

    for tag in sorted_tags:
        lines.append(f"### Группа `{tag.upper()}`")
        lines.append("")
        for method, path, details in tags_map[tag]:
            summary = details.get("summary", "")
            description = details.get("description", "").strip()
            lines.append(f"#### `{method} {path}`")
            if summary:
                lines.append(f"**Описание**: {summary}")
            if description and description != summary:
                lines.append(f"{description}")
            lines.append("")

            # Parameters
            params = details.get("parameters", [])
            if params:
                lines.append("**Параметры запроса (Query / Path):**")
                lines.append("| Параметр | Расположение | Тип | Обязательный | Описание |")
                lines.append("|---|---|---|---|---|")
                for p in params:
                    name = p.get("name", "")
                    _in = p.get("in", "query")
                    req = "Да" if p.get("required") else "Нет"
                    schema = p.get("schema", {})
                    p_type = schema.get("type", "string")
                    desc = p.get("description", "").replace("\n", " ")
                    lines.append(f"| `{name}` | `{_in}` | `{p_type}` | {req} | {desc} |")
                lines.append("")

            # Request Body
            req_body = details.get("requestBody", {})
            if req_body:
                lines.append("**Тело запроса (JSON):**")
                content = req_body.get("content", {}).get("application/json", {})
                schema = content.get("schema", {})
                ref = schema.get("$ref", "")
                if ref:
                    schema_name = ref.split("/")[-1]
                    lines.append(f"- Схема: `{schema_name}`")
                lines.append("")

            # Responses
            responses = details.get("responses", {})
            lines.append("**Ответы:**")
            for code, r_info in responses.items():
                r_desc = r_info.get("description", "")
                lines.append(f"- **`{code}`**: {r_desc}")
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(export_openapi())

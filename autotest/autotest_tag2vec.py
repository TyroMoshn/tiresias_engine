#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# --------------------- пересчёт по желанию ---------------------
def run_recompute(root_data: Path, extra_args: List[str]) -> int:
    cmd = [sys.executable, "-m", "build_index.main",
           "--root", str(root_data),
           "--do", "tag2vec"]
    cmd += extra_args
    print("[recompute] running:", " ".join(cmd))
    return subprocess.run(cmd).returncode

# --------------------- загрузка артефактов ---------------------
def load_tag_vectors(features_dir: Path) -> Tuple[pl.DataFrame, Optional[pl.DataFrame], dict]:
    vec = features_dir / "tag2vec.parquet"
    knn = features_dir / "tag2vec_knn.parquet"
    meta = features_dir / "tag2vec_meta.json"

    if not vec.exists():
        raise FileNotFoundError(f"Не найден {vec}")

    vec_df = pl.read_parquet(vec)
    knn_df = pl.read_parquet(knn) if knn.exists() else None
    meta_d = json.loads(meta.read_text()) if meta.exists() else {}
    return vec_df, knn_df, meta_d

def load_tags_table(root_data: Path) -> pl.DataFrame:
    """
    Артефакты, НЕ csv: предпочитаем tags.parquet (считан stats_stage),
    иначе tags_dict.parquet (считан tags_stage). В обоих ожидаем 'tag_id' и 'tag'.
    """
    cand = [root_data / "tags.parquet", root_data / "tags_dict.parquet"]
    for p in cand:
        if p.exists():
            df = pl.read_parquet(p)
            # нормализуем ожидаемые колонки
            cols = {c.lower(): c for c in df.columns}
            tag_id_col = cols.get("tag_id", "tag_id")
            tag_col = cols.get("tag", "tag")
            out = df.rename({tag_id_col: "tag_id", tag_col: "tag"})
            # другие колонки нам не критичны
            return out.select(["tag_id", "tag"])
    raise FileNotFoundError("Не найдены tags.parquet или tags_dict.parquet")

def load_alias_map_if_any(root_data: Path) -> Dict[str, str]:
    """
    Читаем только CSV алиасов (лёгко и быстро), НИЧЕГО не пересчитываем.
    Возвращаем map: alias(lower) -> canon(lower), только статус 'active'.
    """
    path = root_data / "tag_aliases.csv"
    if not path.exists():
        return {}
    df = pl.read_csv(path, infer_schema_length=0, ignore_errors=True)
    cols = {c.lower(): c for c in df.columns}
    need = ("antecedent_name", "consequent_name", "status")
    for n in need:
        if n not in cols:
            return {}
    df = (df.rename({cols["antecedent_name"]: "a",
                     cols["consequent_name"]: "b",
                     cols["status"]: "s"})
            .with_columns(pl.col("a").cast(pl.Utf8),
                          pl.col("b").cast(pl.Utf8),
                          pl.col("s").cast(pl.Utf8))
            .filter(pl.col("s") == "active")
            .select(["a", "b"]))

    # Разворачиваем возможные цепочки alias->alias->canon
    raw = {str(a).lower(): str(b).lower() for a, b in df.iter_rows()}
    canon: Dict[str, str] = {}
    def root(x: str) -> str:
        seen = set()
        while x in raw and x not in seen:
            seen.add(x)
            x = raw[x]
        return x
    for k in list(raw.keys()):
        canon[k] = root(k)
    return canon

# --------------------- поиск тега ---------------------
def resolve_tag_ids(tags_df: pl.DataFrame, name_or_id: str,
                    alias_map: Dict[str, str],
                    suggest: bool = False) -> Tuple[Optional[int], List[str]]:
    """
    Возвращает (tag_id или None, подсказки-строки).
    1) если число — берём как id;
    2) точное совпадение по 'tag';
    3) через алиасы (если есть);
    4) подстрочный поиск для подсказок.
    """
    hints: List[str] = []
    if name_or_id.isdigit():
        return int(name_or_id), hints

    name = name_or_id.strip()
    low = name.lower()

    # точное совпадение по 'tag'
    exact = tags_df.filter(pl.col("tag").str.to_lowercase() == low)
    if exact.height > 0:
        return int(exact["tag_id"][0]), hints

    # через алиасы
    if alias_map:
        canon = alias_map.get(low)
        if canon is not None:
            exact2 = tags_df.filter(pl.col("tag").str.to_lowercase() == canon)
            if exact2.height > 0:
                return int(exact2["tag_id"][0]), hints

    # подсказки по подстроке
    if suggest:
        sub = tags_df.filter(pl.col("tag").str.contains(name, literal=True, strict=False))\
                     .select(["tag_id", "tag"]).head(20)
        if sub.height > 0:
            hints.extend([f"{int(r[0])}\t{str(r[1])}" for r in sub.iter_rows()])

    return None, hints

# --------------------- вектора и knn ---------------------
def vecs_from_polars(df: pl.DataFrame) -> Tuple[np.ndarray, List[int]]:
    if "vec" not in df.columns or "tag_id" not in df.columns:
        raise ValueError("Ожидаю колонки 'tag_id' и 'vec' в tag2vec.parquet")
    ids = [int(x) for x in df["tag_id"].to_list()]
    rows = df["vec"].to_list()
    D = max((len(r) if r is not None else 0) for r in rows) if rows else 0
    M = np.zeros((len(rows), D), dtype=np.float32)
    for i, r in enumerate(rows):
        if r:
            M[i, : len(r)] = np.asarray(r, dtype=np.float32)
    return M, ids

def compute_knn_from_vecs(mat: np.ndarray, tag_ids: List[int], qpos: int, k: int) -> List[Tuple[int, float]]:
    if mat.shape[0] == 0:
        return []
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    M = mat / norms
    sims = (M @ M[qpos:qpos+1].T).reshape(-1)
    sims[qpos] = -np.inf
    k = min(k, M.shape[0] - 1)
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(tag_ids[i], float(sims[i])) for i in idx]

def pretty_print_neighbors(neigh: List[Tuple[int, float]], tags_df: pl.DataFrame, max_rows: int = 100):
    # Соберём быстрый id->name словарь
    id2name = {int(r[0]): str(r[1]) for r in tags_df.select(["tag_id", "tag"]).iter_rows()}
    for i, (tid, sim) in enumerate(neigh[:max_rows], 1):
        nm = id2name.get(tid, str(tid))
        print(f"{i:2d}. tag_id={tid:8d}  name={nm:30s}  sim={sim:.4f}")

# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="Тест tag2vec — показать ближайших соседей для тега (по id или имени).")
    ap.add_argument("--root-data", type=Path, default=Path(r"I:\TIRESIAS_ENGINE\data"))
    ap.add_argument("--tag", type=str, required=True, help="Имя тега (строка) или его ID (число)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--recompute", action="store_true", help="Перед показом заново посчитать tag2vec (только этот шаг)")
    ap.add_argument("--suggest", action="store_true", help="Если имя не найдено, показать подсказки по подстроке")
    # Параметры пересчёта (если --recompute)
    ap.add_argument("--tag2vec-dim", type=int, default=128)
    ap.add_argument("--tag2vec-min-df", type=int, default=200)
    ap.add_argument("--tag2vec-source", choices=["merge", "pmi", "pools"], default="merge")
    ap.add_argument("--tag2vec-pool-alpha", type=float, default=0.5)
    ap.add_argument("--tag2vec-shift", type=float, default=0.0)
    ap.add_argument("--tag2vec-knn-k", type=int, default=32)
    args = ap.parse_args()

    root = args.root_data
    features = root / "features"

    # (0) опционально — пересчёт
    if args.recompute:
        rc = run_recompute(root, [
            "--tag2vec-dim", str(args.tag2vec_dim),
            "--tag2vec-min-df", str(args.tag2vec_min_df),
            "--tag2vec-source", args.tag2vec_source,
            "--tag2vec-pool-alpha", str(args.tag2vec_pool_alpha),
            "--tag2vec-shift", str(args.tag2vec_shift),
            "--tag2vec-knn-k", str(args.tag2vec_knn_k),
        ])
        if rc != 0:
            print("[recompute] завершился с кодом", rc)
            return

    # (1) артефакты
    vec_df, knn_df, meta = load_tag_vectors(features)
    print("Загружено векторов:", vec_df.height)
    if knn_df is not None:
        print("KNN rows:", knn_df.height)
    if meta:
        print("Meta:", json.dumps(meta, ensure_ascii=False))

    tags_df = load_tags_table(root)  # только артефакты
    alias_map = load_alias_map_if_any(root)  # лёгкая CSV для чисто справочного маппинга

    # (2) резолвим тег
    tag_id, hints = resolve_tag_ids(tags_df, args.tag, alias_map, suggest=args.suggest)
    if tag_id is None:
        print(f"Не найдено тегов по имени '{args.tag}'.", "Возможные варианты:" if hints else "")
        for h in hints:
            print("  ", h)
        print("Можно указать явный --tag <id>.")
        return

    print("Используем tag_id =", tag_id)

    # (3) сначала попробуем готовый KNN
    if knn_df is not None:
        dfk = (knn_df
               .filter(pl.col("tag_id") == tag_id)
               .sort("sim", descending=True)  # <= fix: 'descending' вместо 'reverse'
               .head(args.topk))
        rows = [(int(r[0]), float(r[1])) for r in dfk.select(["nn_tag_id", "sim"]).iter_rows()]
        if rows:
            print("Результат (из tag2vec_knn.parquet):")
            pretty_print_neighbors(rows, tags_df)
            return

    # (4) иначе посчитаем по векторам
    M, ids = vecs_from_polars(vec_df)
    if not ids or tag_id not in set(ids):
        print("Выбранный tag_id отсутствует в tag2vec.parquet.")
        return
    id2idx = {tid: i for i, tid in enumerate(ids)}
    neigh = compute_knn_from_vecs(M, ids, qpos=id2idx[tag_id], k=args.topk)
    print("Результат (посчитано по vec):")
    pretty_print_neighbors(neigh, tags_df)

if __name__ == "__main__":
    main()

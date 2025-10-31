from __future__ import annotations
"""
TAG2VEC — офлайн-эмбеддинги тегов из ко-встречаемостей (PMI) и сигналов из pools.
Выход:
  - features/tag2vec.parquet:  [tag_id:int32, vec:list[float32]]
  - features/tag2vec_knn.parquet (опционально): [tag_id:int32, nn_tag_id:int32, sim:float32]
  - features/tag2vec_meta.json: служебные параметры прогонки

Идея:
  1) Собираем неориентированный граф тегов с весами w_ij из:
     - tag_pmi.parquet (PMI>0)
     - tag_co_from_pools.parquet (веса из пулов)
  2) Переводим веса в сдвинутый PMI: X_ij = max(0, log((w_ij * S) / (d_i * d_j)) - shift)
     где d_i = sum_j w_ij, S = sum_{i<j} w_ij.
  3) Строим разреженную матрицу X и делаем усечённое SVD (randomized_svd).
     Эмбеддинг: E = U * sqrt(Sigma). Нормализуем до ||e_i||=1
  4) Сохраняем parquet и (опц.) top-K ближайших соседей по косинусу.

Зависимости: polars, numpy, (желательно) scipy.sparse и sklearn (randomized_svd).
Если scipy/sklearn недоступны, используем fallback на плотный SVD для малых V (<=50k тегов).
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import polars as pl

try:
    import scipy.sparse as sp  # type: ignore
    from scipy.sparse.linalg import svds  # <- добавили
    _HAS_SPARSE = True
except Exception:  # pragma: no cover
    sp = None  # type: ignore
    svds = None  # type: ignore
    _HAS_SPARSE = False

try:
    from sklearn.utils.extmath import randomized_svd  # type: ignore
except Exception:  # pragma: no cover
    randomized_svd = None  # type: ignore

try:
    import faiss  # faiss-cpu 1.12.0
    _HAS_FAISS_FAISS = True
    faiss.omp_set_num_threads(12)
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS_FAISS = False

from .config import Config
from .utils import ensure_dir, newer_than, log

# ----------------------- helpers -----------------------

def _load_tag_universe(cfg: Config, min_df: int, max_tags: int | None) -> pl.DataFrame:
    """Возвращает таблицу тегов, которые будем эмбеддить: [tag_id, df, idf].
    Источник — tags.parquet (если есть, там есть df_local/idf), иначе tags_dict.parquet.
    """
    base = cfg.tags_parquet
    if base.exists():
        df = pl.read_parquet(base).select([
            pl.col("tag_id").cast(pl.Int32),
            pl.coalesce([pl.col("df_local"), pl.col("post_count")]).fill_null(0).alias("df"),
            pl.coalesce([pl.col("idf"), pl.lit(1.0)]).alias("idf"),
        ])
    else:
        df = pl.read_parquet(cfg.root / "tags_dict.parquet").select([
            pl.col("tag_id").cast(pl.Int32),
            pl.col("post_count").fill_null(0).alias("df"),
        ]).with_columns(pl.lit(1.0).alias("idf"))

    if min_df > 0:
        df = df.filter(pl.col("df") >= min_df)
    # ограничим верх — чтобы не взорвать память
    if max_tags is not None and df.height > max_tags:
        df = df.sort("df", descending=True).head(max_tags)
    return df


def _load_edges_from_pmi(cfg: Config, keep: set[int]) -> Dict[Tuple[int, int], float]:
    """Читает tag_pmi.parquet и возвращает веса (симметрично) по положительному PMI."""
    path = cfg.root / "tag_pmi.parquet"
    acc: Dict[Tuple[int, int], float] = defaultdict(float)
    if not path.exists():
        return acc
    df = pl.read_parquet(path).select(["a", "b", "pmi"]).filter(pl.col("pmi") > 0)
    if keep:
        df = df.filter(pl.col("a").is_in(list(keep)) & pl.col("b").is_in(list(keep)))
    for a, b, pmi in df.iter_rows():
        a = int(a); b = int(b)
        if a == b:  # safety
            continue
        if a > b:
            a, b = b, a
        acc[(a, b)] += float(pmi)
    return acc


def _load_edges_from_pools(cfg: Config, keep: set[int]) -> Dict[Tuple[int, int], float]:
    """Читает tag_co_from_pools.parquet и возвращает веса (симметрично)."""
    path = cfg.tag_co_from_pools_parquet
    acc: Dict[Tuple[int, int], float] = defaultdict(float)
    if not path.exists():
        return acc
    df = pl.read_parquet(path).rename({"a_tag_id": "a", "b_tag_id": "b"}).select(["a", "b", "weight"]) \
         .filter(pl.col("weight") > 0)
    if keep:
        df = df.filter(pl.col("a").is_in(list(keep)) & pl.col("b").is_in(list(keep)))
    for a, b, w in df.iter_rows():
        a = int(a); b = int(b)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        acc[(a, b)] += float(w)
    return acc


def _merge_edges(e1: Dict[Tuple[int, int], float], e2: Dict[Tuple[int, int], float], alpha: float) -> Dict[Tuple[int, int], float]:
    """Смешиваем два источника с весами: w = w_pmi + alpha * w_pools."""
    out = defaultdict(float)
    for k, v in e1.items():
        out[k] += v
    for k, v in e2.items():
        out[k] += alpha * v
    return out


def _to_ppmi(edges: Dict[Tuple[int, int], float], tag_ids: List[int], shift: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Пересчитываем в PPMI-значения и отдаём COO-тройки (rows, cols, data) для симметричной матрицы.
    Возвращаем индексы уже как позиционные (0..V-1).
    """
    if not edges:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    id2pos = {int(t): i for i, t in enumerate(tag_ids)}

    # степени вершин и общий вес
    deg = np.zeros(len(tag_ids), dtype=np.float64)
    total_w = 0.0
    for (a, b), w in edges.items():
        ia = id2pos.get(a); ib = id2pos.get(b)
        if ia is None or ib is None:
            continue
        w = float(w)
        deg[ia] += w; deg[ib] += w
        total_w += w

    # избежание деления на ноль
    deg = np.maximum(deg, 1e-12)
    S = max(total_w, 1e-12)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for (a, b), w in edges.items():
        ia = id2pos.get(a); ib = id2pos.get(b)
        if ia is None or ib is None:
            continue
        # PMI сдвиг
        val = math.log((w * S) / (deg[ia] * deg[ib])) - float(shift)
        if val <= 0:
            continue  # PPMI
        # симметрия
        rows.extend([ia, ib])
        cols.extend([ib, ia])
        data.extend([val, val])

    return (np.asarray(rows, dtype=np.int32),
            np.asarray(cols, dtype=np.int32),
            np.asarray(data, dtype=np.float32))


def _embed_sparse(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, V: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Усечённое SVD на разреженной PPMI:
    1) Пытаемся через scipy.sparse.linalg.svds (ARPACK) — быстро и без 'gesdd'.
    2) Если svds падает, используем randomized_svd(n_iter=2) как быстрый fallback.
    Возвращаем (E, S): E.shape=(V,dim), строки E L2-нормированы.
    """
    if rows.size == 0:
        return np.zeros((V, dim), dtype=np.float32), np.zeros((dim,), dtype=np.float32)

    # строим разреженную симметричную матрицу
    if _HAS_SPARSE:
        X = sp.coo_matrix((data.astype(np.float32, copy=False), (rows, cols)),
                          shape=(V, V), dtype=np.float32).tocsr()
        # safety: чуть-чуть «подправим» симметрию (необязательно)
        # X = 0.5 * (X + X.T)

        # Попытка №1: ARPACK
        try:
            U, S, _ = svds(X, k=dim, which="LM", tol=1e-3, maxiter=dim * 8)
            # svds выдаёт S по возрастанию → развернём
            order = np.argsort(S)[::-1]
            S = S[order].astype(np.float32, copy=False)
            U = U[:, order].astype(np.float32, copy=False)
            # полезный лог для отладки
            log("[tag2vec] SVD backend=svds")
        except Exception as e:
            if randomized_svd is None:
                raise
            log(f"[tag2vec] svds failed ({e}); fallback to randomized_svd(n_iter=2)")
            # Попытка №2: randomized_svd — без дорогого малого SVD
            try:
                U, S, _ = randomized_svd(X, n_components=dim, n_iter=2, random_state=0)
            except TypeError:
                U, S, _ = randomized_svd(X, n_components=dim, n_iter=2)
            S = S.astype(np.float32, copy=False)
            U = U.astype(np.float32, copy=False)
    else:
        # жёсткий fallback — плотная матрица (только для маленького V)
        X = np.zeros((V, V), dtype=np.float32)
        X[rows, cols] = data
        X = 0.5 * (X + X.T)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        U, S = U[:, :dim].astype(np.float32, copy=False), S[:dim].astype(np.float32, copy=False)
        log("[tag2vec] SVD backend=dense")

    # как прежде: E = U * sqrt(S), потом L2-норма по строкам
    E = (U * np.sqrt(S[None, :])).astype(np.float32, copy=False)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
    return E, S



def _write_vectors(cfg: Config, tag_ids: List[int], E: np.ndarray) -> None:
    ensure_dir(cfg.features_dir)
    df = pl.DataFrame({
        "tag_id": pl.Series(tag_ids, dtype=pl.Int32),
        "vec": [E[i, :].tolist() for i in range(E.shape[0])],
    })
    out = cfg.features_dir / "tag2vec.parquet"
    df.write_parquet(out, compression="zstd")
    log(f"[tag2vec] saved vectors → {out}")


def _write_knn(cfg: Config, tag_ids: List[int], E: np.ndarray, k: int) -> None:
    """Пишем top-K соседей по косинусу. 
    Предпочитаем FAISS (IndexFlatIP), т.к. E уже L2-нормированы => cosine == dot.
    Пакетный поиск, один раз строим индекс.
    """
    if k <= 0:
        return
    V, D = E.shape
    if V == 0:
        return

    # faiss любит C-contiguous float32
    xb = np.ascontiguousarray(E.astype(np.float32, copy=False))
    use_faiss = _HAS_FAISS_FAISS and bool(getattr(cfg, "tag2vec_use_faiss", True))

    rows: List[int] = []
    cols: List[int] = []
    sims: List[float] = []

    if use_faiss:
        # опционально: управляем потоками FAISS
        try:
            # если в конфиге есть параметр — используем, иначе FAISS сам возьмёт OMP
            threads = int(getattr(cfg, "tag2vec_faiss_threads", 0))
            if threads > 0:
                faiss.omp_set_num_threads(threads)
        except Exception:
            pass

        # exact cosine via inner product
        index = faiss.IndexFlatIP(D)
        index.add(xb)

        # пакетный поиск; берём k+1, чтобы выбросить self-match
        qk = min(k + 1, max(2, k + 1))
        batch = int(getattr(cfg, "tag2vec_knn_batch", 16384))

        for i0 in range(0, V, batch):
            i1 = min(V, i0 + batch)
            Dm, Im = index.search(xb[i0:i1], qk)  # (B, qk)
            # пост-фильтр: выкинуть self и обрезать до k
            for r in range(i1 - i0):
                src = i0 + r
                im = Im[r]
                dm = Dm[r]
                # выбрасываем self-индекс, если присутствует
                keep = im != src
                im = im[keep][:k]
                dm = dm[keep][:k]
                for dst, sim in zip(im, dm):
                    rows.append(src)
                    cols.append(int(dst))
                    sims.append(float(sim))
    else:
        # Fallback: batched numpy (то, что было, но быстрее батч и argpartition)
        batch = 8192
        for i0 in range(0, V, batch):
            i1 = min(V, i0 + batch)
            S = xb[i0:i1] @ xb.T
            # self-сходства → −inf, чтобы не попадали в топ
            for i in range(i0, i1):
                S[i - i0, i] = -np.inf
            idx_part = np.argpartition(-S, kth=min(k, V - 1) - 1, axis=1)[:, :k]
            part_vals = np.take_along_axis(S, idx_part, axis=1)
            order = np.argsort(-part_vals, axis=1)
            top_idx = np.take_along_axis(idx_part, order, axis=1)
            top_vals = np.take_along_axis(part_vals, order, axis=1)
            for r in range(i1 - i0):
                src = i0 + r
                for j in range(top_idx.shape[1]):
                    rows.append(src)
                    cols.append(int(top_idx[r, j]))
                    sims.append(float(top_vals[r, j]))

    df = pl.DataFrame({
        "tag_id": [tag_ids[i] for i in rows],
        "nn_tag_id": [tag_ids[j] for j in cols],
        "sim": sims,
    })
    out = cfg.features_dir / "tag2vec_knn.parquet"
    df.write_parquet(out, compression="zstd")
    log(f"[tag2vec] saved KNN → {out} (faiss={'yes' if use_faiss else 'no'})")


# ----------------------- public step -----------------------

def step_tag2vec(cfg: Config) -> None:
    """Основной шаг построения эмбеддингов тегов."""
    meta_path = cfg.features_dir / "tag2vec_meta.json"
    vec_path = cfg.features_dir / "tag2vec.parquet"

    # флаг актуальности: пересчитываем, если force или входы новее
    inputs = [cfg.root / "tag_pmi.parquet", cfg.tag_co_from_pools_parquet, cfg.tags_parquet, cfg.root / "tags_dict.parquet"]
    if (not cfg.force) and vec_path.exists() and newer_than(vec_path, *[p for p in inputs if p.exists()]):
        log("[tag2vec] already fresh — skip")
        return

    dim = getattr(cfg, "tag2vec_dim", 128)
    min_df = getattr(cfg, "tag2vec_min_df", 100)
    max_tags = getattr(cfg, "tag2vec_max_tags", 200_000)
    source = getattr(cfg, "tag2vec_source", "merge")  # 'pmi' | 'pools' | 'merge'
    alpha = float(getattr(cfg, "tag2vec_pool_alpha", 0.5))
    shift = float(getattr(cfg, "tag2vec_shift", 0.0))
    knn_k = int(getattr(cfg, "tag2vec_knn_k", 32))

    log(f"[tag2vec] params: dim={dim} min_df={min_df} max_tags={max_tags} source={source} alpha={alpha} shift={shift} knn_k={knn_k}")

    # 1) отберём вселенную тегов
    uni = _load_tag_universe(cfg, min_df=min_df, max_tags=max_tags)
    if uni.is_empty():
        log("[tag2vec] universe is empty — nothing to do")
        ensure_dir(cfg.features_dir)
        pl.DataFrame({"tag_id": [], "vec": []}).write_parquet(vec_path, compression="zstd")
        return
    tag_ids: List[int] = [int(x) for x in uni.get_column("tag_id").to_list()]
    keep = set(tag_ids)

    # 2) соберём граф
    if source in ("merge", "pmi"):
        e_pmi = _load_edges_from_pmi(cfg, keep)
    else:
        e_pmi = {}
    if source in ("merge", "pools"):
        e_pools = _load_edges_from_pools(cfg, keep)
    else:
        e_pools = {}

    edges = _merge_edges(e_pmi, e_pools, alpha=alpha)
    n_edges = len(edges)
    log(f"[tag2vec] edges: {n_edges:,} unique pairs")
    if n_edges == 0:
        log("[tag2vec] no edges — saving empty outputs")
        _write_vectors(cfg, tag_ids, np.zeros((len(tag_ids), dim), dtype=np.float32))
        return

    # 3) PPMI
    rows, cols, data = _to_ppmi(edges, tag_ids, shift=shift)
    V = len(tag_ids)
    log(f"[tag2vec] PPMI nnz={data.size:,} for V={V:,}")

    # 4) SVD → эмбеддинги
    E, S = _embed_sparse(rows, cols, data, V=V, dim=dim)

    # 5) Запись
    _write_vectors(cfg, tag_ids, E)

    # 6) KNN (опционально)
    try:
        _write_knn(cfg, tag_ids, E, k=knn_k)
    except Exception as e:
        log(f"[tag2vec] KNN failed: {e}")

    # 7) метаданные
    meta = {
        "dim": int(dim),
        "min_df": int(min_df),
        "max_tags": int(max_tags),
        "source": str(source),
        "pool_alpha": float(alpha),
        "shift": float(shift),
        "knn_k": int(knn_k),
        "vectors": str((cfg.features_dir / "tag2vec.parquet").as_posix()),
        "knn": str((cfg.features_dir / "tag2vec_knn.parquet").as_posix()),
        "sigma": [float(x) for x in S[: min(16, S.shape[0])]],
    }
    ensure_dir(cfg.features_dir)
    (cfg.features_dir / "tag2vec_meta.json").write_text(json.dumps(meta, indent=2))
    log("[tag2vec] done")

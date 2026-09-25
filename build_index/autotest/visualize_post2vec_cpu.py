from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Iterable
from collections import Counter

import numpy as np
import pandas as pd
import faiss
import polars as pl

import umap
import hdbscan

import holoviews as hv
import hvplot.pandas  # registers .hvplot on pandas
hv.extension("bokeh")

from bokeh.models import (
    HoverTool,
    TapTool,
    OpenURL,
    RangeSlider,
    Slider,
    Spinner,
    CustomJSFilter,
    CDSView,
    Checkbox,
    CustomJS,
    Div,
)
from bokeh.plotting import figure, save as bokeh_save
from bokeh.layouts import column, row
from bokeh.models import GlyphRenderer

# ---------------- I/O helpers ----------------

def load_sample_vectors(index_path: Path, sample_size: int | None = 300_000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, int]:
    index = faiss.read_index(str(index_path))
    ntotal = index.ntotal
    dim = index.d
    if ntotal == 0:
        raise RuntimeError("FAISS index is empty")
    if sample_size is None or sample_size > ntotal:
        sample_size = ntotal
    rng = np.random.default_rng(seed)
    row_ids = rng.choice(ntotal, size=sample_size, replace=False).astype("int64")
    row_ids.sort()
    if hasattr(index, "reconstruct_batch"):
        X = index.reconstruct_batch(row_ids)
    else:
        X = np.empty((sample_size, dim), dtype="float32")
        tmp = np.empty((dim,), dtype="float32")
        for i, rid in enumerate(row_ids):
            index.reconstruct(int(rid), tmp)
            X[i] = tmp
    return X.astype("float32"), row_ids, dim


def map_rows_to_post_ids(ids_path: Path, rows: np.ndarray) -> np.ndarray:
    """Map FAISS row indices → post_id using parquet mapping (['row','post_id'])."""
    id_map = pd.read_parquet(ids_path, columns=["row", "post_id"])
    df_rows = pd.DataFrame({"row": rows.astype("int32")})
    sampled = df_rows.merge(id_map, on="row", how="left").sort_values("row")
    post_ids = sampled["post_id"].to_numpy()
    if post_ids.shape[0] != rows.shape[0]:
        raise RuntimeError(f"Row→post_id mapping mismatch: got {post_ids.shape[0]} for {rows.shape[0]} rows")
    return post_ids


def fetch_overlay_metadata_from_artifacts(data_root: Path, post_ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
    """Pull tag_string, fav_count, score for the given post_ids."""
    ids = sorted({int(x) for x in post_ids})
    if not ids:
        return {}

    tags_dict_path = data_root / "tags_dict.parquet"
    tags_lookup = pl.read_parquet(tags_dict_path).select([
        pl.col("tag_id").cast(pl.Int32),
        pl.col("tag").cast(pl.Utf8),
    ])

    pt_lf = pl.scan_parquet(f"{(data_root / 'post_tags_parquet').as_posix()}/**/*.parquet") \
             .select([pl.col("post_id").cast(pl.Int64), pl.col("tag_id").cast(pl.Int32)]) \
             .filter(pl.col("post_id").is_in(ids))

    tag_names = (
        pt_lf.join(tags_lookup.lazy(), on="tag_id", how="left")
             .group_by("post_id")
             .agg(pl.col("tag").drop_nans().drop_nulls().unique().sort().alias("tags"))
             .with_columns(pl.col("tags").list.join(" ").alias("tag_string"))
             .select(["post_id", "tag_string"])
             .collect(engine="streaming")
    )

    posts_lf = pl.scan_parquet(f"{(data_root / 'posts_parquet').as_posix()}/**/*.parquet") \
                 .select([
                     pl.col("id").cast(pl.Int64).alias("post_id"),
                     pl.col("fav_count").cast(pl.Int32),
                     pl.col("score").cast(pl.Int32),
                 ]) \
                 .filter(pl.col("post_id").is_in(ids))

    numbers = posts_lf.collect(engine="streaming")

    meta = numbers.join(tag_names, on="post_id", how="left").with_columns([
        pl.col("tag_string").fill_null(""),
        pl.col("fav_count").fill_null(0),
        pl.col("score").fill_null(0),
    ])

    out: Dict[int, Dict[str, Any]] = {}
    for pid, fav, sc, tstr in meta.select(["post_id", "fav_count", "score", "tag_string"]).iter_rows():
        out[int(pid)] = {"tag_string": tstr or "", "fav_count": int(fav or 0), "score": int(sc or 0)}

    for pid in ids:
        if pid not in out:
            out[pid] = {"tag_string": "", "fav_count": 0, "score": 0}

    return out


# ---------------- ML embedding ----------------

def compute_umap_hdbscan_cpu(X: np.ndarray, min_cluster_size: int = 50, random_state: int = 42) -> pd.DataFrame:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=None,
        n_jobs=-1,
        low_memory=True,
        verbose=True,
    )
    emb = reducer.fit_transform(X)
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=0)
    labels = hdb.fit_predict(emb)
    return pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "cluster": labels})


# ---------------- Plot helpers ----------------

def _shorten(s: str, maxlen: int = 500) -> str:
    if s is None:
        return ""
    s = str(s)
    return (s[:maxlen] + "…") if len(s) > maxlen else s


def _hover_html() -> str:
    return """
    <div>
      <div><b>post_id:</b> @post_id</div>
      <div><b>cluster:</b> @cluster</div>
      <div><b>fav:</b> @fav_count &nbsp;&nbsp; <b>score:</b> @score</div>
      <div><b>cluster top tags:</b></div>
      <div style="max-width: 480px; white-space: normal; word-break: break-word; overflow-wrap: anywhere;">
        @cluster_top_tags
      </div>
      <div><b>tags:</b></div>
      <div style="max-width: 480px; white-space: normal; word-break: break-word; overflow-wrap: anywhere;">
        @tag_string_short
      </div>
    </div>
    """


def _make_bokeh_hooks(url_prefix: str):
    def _setup(plot, element):
        fig = plot.state

        overlay_renderers = []
        for r in fig.renderers:
            ds = getattr(r, "data_source", None)
            data = getattr(ds, "data", None)
            if isinstance(data, dict) and "post_id" in data:
                overlay_renderers.append(r)

        # --- HoverTool
        hover = next((t for t in fig.tools if isinstance(t, HoverTool)), None)
        if hover is None:
            hover = HoverTool()
            fig.add_tools(hover)
        hover.tooltips = _hover_html()
        hover.point_policy = "follow_mouse"
        hover.attachment = "horizontal"
        hover.show_arrow = False
        if overlay_renderers:
            hover.renderers = overlay_renderers

        # --- TapTool
        tap = next((t for t in fig.tools if isinstance(t, TapTool)), None)
        if tap is None:
            tap = TapTool()
            fig.add_tools(tap)
        tap.callback = OpenURL(url=str(url_prefix) + "@post_id")
        if overlay_renderers:
            tap.renderers = overlay_renderers

        try:
            fig.toolbar.active_inspect = hover
            fig.toolbar.active_tap = tap
        except Exception:
            pass

        for r in overlay_renderers:
            try:
                r.level = "glyph"
                r.z = 10
            except Exception:
                pass

    return [_setup]


# ---------------- Build & Save plot ----------------

def _collect_all_figures(root) -> list:
    """Assembling all the Figure from the bokeh object (Figure or Layout)"""
    figs = []
    def _walk(obj):
        if isinstance(obj, figure):
            figs.append(obj)
        for child in getattr(obj, "children", []) or []:
            _walk(child)
    _walk(root)
    return figs or ([root] if hasattr(root, "renderers") else [])



def _postprocess_with_widgets(bokeh_model):
    figs = _collect_all_figures(bokeh_model)
    if not figs:
        return bokeh_model

    fig = figs[0]

    overlay_renderers: list[GlyphRenderer] = []
    bg_renderers: list[GlyphRenderer] = []
    for r in fig.renderers:
        ds = getattr(r, "data_source", None)
        data = getattr(ds, "data", None)
        glyph_type = getattr(getattr(r, "glyph", None), "__class__", type("X",(object,),{})).__name__
        if isinstance(data, dict) and "post_id" in data:
            overlay_renderers.append(r)
        elif glyph_type in ("ImageRGBA", "Image"):
            bg_renderers.append(r)

    if not overlay_renderers:
        return bokeh_model

    ds0 = overlay_renderers[0].data_source
    scores = ds0.data.get("score", [])
    if len(scores) == 0:
        return bokeh_model

    score_min = int(np.nanmin(scores))
    score_max = int(np.nanmax(scores))
    # Normal fallback
    if score_min == score_max:
        score_min = max(0, score_min - 1)
        score_max = score_max + 1

    slider = RangeSlider(title="Score filter", start=score_min, end=score_max, value=(score_min, score_max), step=1)
    score_min_input = Spinner(title="Min score", low=score_min, high=score_max, value=score_min, step=1, width=180)
    score_max_input = Spinner(title="Max score", low=score_min, high=score_max, value=score_max, step=1, width=180)

    slider.js_on_change(
        "value",
        CustomJS(
            args=dict(min_input=score_min_input, max_input=score_max_input),
            code="""
                const [low, high] = cb_obj.value;
                if (min_input.value !== low) {
                    min_input.value = low;
                }
                if (max_input.value !== high) {
                    max_input.value = high;
                }
            """,
        ),
    )

    score_min_input.js_on_change(
        "value",
        CustomJS(
            args=dict(slider=slider),
            code="""
                const raw = Number(cb_obj.value);
                const target = Number.isFinite(raw) ? raw : slider.start;
                const clamped = Math.min(Math.max(target, slider.start), slider.end);
                const [, high] = slider.value;
                slider.value = [Math.min(clamped, high), Math.max(high, clamped)];
            """,
        ),
    )
    score_max_input.js_on_change(
        "value",
        CustomJS(
            args=dict(slider=slider),
            code="""
                const raw = Number(cb_obj.value);
                const target = Number.isFinite(raw) ? raw : slider.end;
                const clamped = Math.min(Math.max(target, slider.start), slider.end);
                const [low] = slider.value;
                slider.value = [Math.min(low, clamped), Math.max(clamped, low)];
            """,
        ),
    )

    # CustomJSFilter
    js_code = """
        const low = slider.value[0];
        const high = slider.value[1];
        // 'source' here is provided automatically by Bokeh for CustomJSFilter
        const score = source.data['score'];
        const N = score.length;
        const mask = new Array(N);
        for (let i = 0; i < N; i++) {
            const s = score[i];
            mask[i] = (s >= low) && (s <= high);
        }
        return mask;
    """

    ds_filters: dict[Any, CustomJSFilter] = {}
    overlay_sources: list[Any] = []
    for r in overlay_renderers:
        ds = r.data_source
        if ds not in ds_filters:
            ds_filters[ds] = CustomJSFilter(args=dict(slider=slider), code=js_code)
            overlay_sources.append(ds)
        r.view = CDSView(filter=ds_filters[ds])

    slider.js_on_change(
        "value",
        CustomJS(
            args=dict(sources=overlay_sources),
            code="""
                for (const src of sources) {
                    src.change.emit();
                }
            """,
        ),
    )

    # Checkbox for BG
    checkbox = Checkbox(label="Show background density", active=True, disabled=(len(bg_renderers) == 0))
    if bg_renderers:
        checkbox.js_on_change(
            "active",
            CustomJS(
                args=dict(bg_renderers=bg_renderers),
                code="""
                    for (const r of bg_renderers) { r.visible = cb_obj.active; }
                """
            )
        )

    glyph0 = getattr(overlay_renderers[0], "glyph", None)
    base_size = getattr(glyph0, "size", 6) if glyph0 else 6
    try:
        base_size = int(base_size)
    except Exception:
        base_size = 6
    size_slider = Slider(title="Dot size", start=max(2, base_size - 6), end=base_size + 12, step=1, value=base_size)
    size_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(renderers=overlay_renderers),
            code="""
                for (const r of renderers) {
                    const g = r.glyph;
                    if (!g) { continue; }
                    g.size = cb_obj.value;
                }
            """,
        ),
    )

    line_alpha_default = getattr(glyph0, "line_alpha", 1.0) if glyph0 else 1.0
    line_width_default = getattr(glyph0, "line_width", 0.7) if glyph0 else 0.7
    try:
        line_alpha_default = float(line_alpha_default)
    except Exception:
        line_alpha_default = 1.0
    try:
        line_width_default = float(line_width_default)
    except Exception:
        line_width_default = 0.7

    outline_slider = Slider(
        title="Outline width",
        start=0,
        end=max(4.0, line_width_default + 2.0),
        step=0.1,
        value=line_width_default,
    )
    outline_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(renderers=overlay_renderers, defaults=dict(line_alpha=line_alpha_default)),
            code="""
                const width = cb_obj.value;
                for (const r of renderers) {
                    const g = r.glyph;
                    if (!g) { continue; }
                    g.line_width = width;
                    g.line_alpha = width > 0 ? defaults.line_alpha : 0;
                }
            """,
        ),
    )

    canvas_width_default = int(getattr(fig, "width", getattr(fig, "plot_width", 1000)) or 1000)
    canvas_height_default = int(getattr(fig, "height", getattr(fig, "plot_height", 780)) or 780)
    canvas_width = Spinner(title="Canvas width", low=400, high=2400, step=20, value=canvas_width_default, width=180)
    canvas_height = Spinner(title="Canvas height", low=300, high=2000, step=20, value=canvas_height_default, width=180)
    canvas_width.js_on_change(
        "value",
        CustomJS(
            args=dict(fig=fig),
            code="""
                const raw = Number(cb_obj.value);
                const width = Math.max(200, Math.floor(Number.isFinite(raw) ? raw : 800));
                fig.width = width;
                fig.plot_width = width;
            """,
        ),
    )
    canvas_height.js_on_change(
        "value",
        CustomJS(
            args=dict(fig=fig),
            code="""
                const raw = Number(cb_obj.value);
                const height = Math.max(200, Math.floor(Number.isFinite(raw) ? raw : 600));
                fig.height = height;
                fig.plot_height = height;
            """,
        ),
    )

    controls_header = Div(text="<b>Display controls</b>")
    slider_inputs = row(score_min_input, score_max_input, sizing_mode="scale_width")
    canvas_header = Div(text="<b>Canvas size</b>")
    canvas_inputs = row(canvas_width, canvas_height, sizing_mode="scale_width")
    controls = column(
        controls_header,
        slider,
        slider_inputs,
        size_slider,
        outline_slider,
        checkbox,
        canvas_header,
        canvas_inputs,
        sizing_mode="stretch_width",
    )

    return column(fig, controls, sizing_mode="stretch_width")


def build_plot_and_save(
    df_all: pd.DataFrame,
    post_ids_all: np.ndarray,
    data_root: Path,
    output_html: Path,
    overlay_frac: float = 0.02,
    url_prefix: str = "https://example.com/postid/",
):
    df = df_all.copy()
    df["post_id"] = post_ids_all.astype("int64")

    # BG (datashader)
    bg = df.hvplot.scatter(
        x="x", y="y", c="cluster",
        datashade=True,
        width=1000, height=780,
        cmap="Category20",
        alpha=0.7,
        title="Post2Vec (CPU) — UMAP + HDBSCAN",
        tools=[]
    )

    if 0 < overlay_frac < 1.0:
        overlay = df.sample(frac=overlay_frac, random_state=42, ignore_index=True)

        overlay_post_ids = overlay["post_id"].unique().tolist()
        meta = fetch_overlay_metadata_from_artifacts(data_root, overlay_post_ids)
        overlay["tag_string"] = overlay["post_id"].map(lambda pid: meta.get(int(pid), {}).get("tag_string", ""))
        overlay["fav_count"] = overlay["post_id"].map(lambda pid: meta.get(int(pid), {}).get("fav_count", 0)).astype("int32")
        overlay["score"] = overlay["post_id"].map(lambda pid: meta.get(int(pid), {}).get("score", 0)).astype("int32")
        overlay["tag_string_short"] = overlay["tag_string"].map(lambda s: _shorten(s, 500))

        cluster_top = {}
        for cl, sub in overlay.groupby("cluster", dropna=False):
            cnt = Counter()
            for s in sub["tag_string"]:
                if s:
                    cnt.update(s.split())
            top = " ".join([t for t, _ in cnt.most_common(12)])
            cluster_top[cl] = _shorten(top, 500)
        overlay["cluster_top_tags"] = overlay["cluster"].map(lambda c: cluster_top.get(c, ""))

        overlay_points = hv.Points(
            overlay,
            kdims=["x", "y"],
            vdims=["cluster", "post_id", "fav_count", "score",
                   "tag_string_short", "cluster_top_tags"],
        ).opts(
            size=6,
            alpha=0.85,
            color="cluster",
            cmap="Category20",
            line_color="black",
            line_alpha=1.0,
            line_width=0.7,
            tools=["hover", "tap", "pan", "wheel_zoom", "reset"],
            active_tools=["tap"],
            hooks=_make_bokeh_hooks(url_prefix),
        )

        plot = bg * overlay_points
    else:
        print("[info] overlay_frac=0 — without interactive background")
        plot = bg

    # Render in Bokeh
    bokeh_model = hv.render(plot, backend="bokeh")

    # Postprocess
    bokeh_layout = _postprocess_with_widgets(bokeh_model)

    print(f"[info] Saving interactive HTML: {output_html}")
    bokeh_save(bokeh_layout, str(output_html))
    print("[ok] Saved:", output_html)


# ---------------- Main ----------------

def main():
    here = Path(__file__).resolve().parent
    default_root = Path(os.environ.get("TIRESIAS_DATA_ROOT", here.parent.parent / "data")).resolve()
    default_features = default_root / "features"

    ap = argparse.ArgumentParser(description="Visualize post2vec space via UMAP and HDBSCAN")
    ap.add_argument("--features-dir", type=Path, default=default_features,
                    help="Folder with post2vec_faiss.index and post2vec_faiss_ids.parquet")
    ap.add_argument("--data-root", type=Path, default=default_root,
                    help="Root with posts_parquet/, post_tags_parquet/, tags_dict.parquet")
    ap.add_argument("--sample-size", type=int, default=300_000)
    ap.add_argument("--min-cluster-size", type=int, default=60)
    ap.add_argument("--overlay-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="post2vec_umap_hdbscan_cpu_with_meta.html")
    ap.add_argument("--url-prefix", type=str, default="https://example.com/postid/")
    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    data_root = Path(args.data_root) if args.data_root else features_dir.parent
    idx_path = features_dir / "post2vec_faiss.index"
    ids_path = features_dir / "post2vec_faiss_ids.parquet"
    output_html = features_dir / args.output

    print("[info] Loading FAISS index:", idx_path)
    X, row_ids, dim = load_sample_vectors(idx_path, sample_size=args.sample_size, seed=args.seed)
    print(f"[info] Sampled {X.shape[0]} vectors, dim={dim}")

    print("[info] Mapping rows → post_id using:", ids_path)
    post_ids = map_rows_to_post_ids(ids_path, row_ids)
    print(f"[info] Mapped {len(post_ids)} rows to post_ids")

    print("[info] Running CPU UMAP + HDBSCAN ...")
    df = compute_umap_hdbscan_cpu(X, min_cluster_size=args.min_cluster_size, random_state=args.seed)
    print("[info] UMAP+HDBSCAN done. Building plots ...")

    build_plot_and_save(df, post_ids, data_root, output_html, overlay_frac=args.overlay_frac, url_prefix=args.url_prefix)
    print("[done] All done.")


if __name__ == "__main__":
    main()

# =============================================================================
# Script: construir_vistas.py
# Descripción:
# Ensambla las vistas de datos para UI (visualización) y Motor de pedidos a
# partir del dataset unificado de sustitutos (9.5), el multiproveedor y, de
# forma opcional, catálogo enriquecido y métricas de stock/consumo.
#
# Flujo del pipeline:
# 1) Cargar datasets de entrada (unificado, multiproveedor y opcionales).
# 2) Identificar proveedor preferente por producto.
# 3) Rankear sustitutos internos y externos (aplicando umbral de score).
# 4) Construir vistas:
#    - UI: ui_products, ui_substitutes
#    - Motor: engine_substitution_rules, engine_supplier_preferred,
#             engine_replenishment_view
# 5) (Opcional) Exportar vistas a parquet/csv si se solicita por CLI.
#
# Inputs (por defecto):
#   - data/processed/substitutes_unified.csv
#   - data/clean/supplier_catalog_multi.csv
#   - (opcional) data/processed/catalog_items_enriquecido.csv
#   - (opcional) data/processed/stock_positions.csv
#   - (opcional) data/processed/consumo_agg.csv
#
# Outputs (opcionales; solo si se pasan --out_ui / --out_engine):
#   - <out_ui>/ui_products.parquet
#   - <out_ui>/ui_substitutes.parquet
#   - <out_engine>/engine_substitution_rules.parquet
#   - <out_engine>/engine_supplier_preferred.parquet
#   - <out_engine>/engine_replenishment_view.parquet
#
# Dependencias:
#   - pandas, numpy
#
# Uso CLI (ejemplo):
#   python scripts/export/construir_vistas.py \
#     --unificado data/processed/substitutes_unified.csv \
#     --multi data/clean/supplier_catalog_multi.csv \
#     --catalogo data/processed/catalog_items_enriquecido.csv \
#     --stock data/processed/stock_positions.csv \
#     --consumo data/processed/consumo_agg.csv \
#     --min_score 0.70 \
#     --out_ui data/exports/ui \
#     --out_engine data/exports/engine
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]   # scripts/export -> proyecto raíz
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia encabezados: BOM, espacios, tipado str."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)  # BOM
        .str.strip()
    )
    return df

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Devuelve el nombre real (case-insensitive) para cualquiera de los candidatos,
    tolerando BOM/espacios.
    """
    normal = {c.lower().replace("\ufeff","").strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("\ufeff","").strip()
        if key in normal:
            return normal[key]
    raise KeyError(f"No se encontró ninguna de {candidates} en columnas={list(df.columns)}")

def _preferente_por_heuristica(df: pd.DataFrame) -> pd.DataFrame:
    """Marca como preferente la mejor fila por heurística (prioridad, precio, LT, disponibilidad)."""
    g = df.copy()
    for c in ["prioridad","precio","lead_time","disponibilidad"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.sort_values(
        by=["prioridad","precio","lead_time","disponibilidad"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    g["__is_pref"] = False
    if len(g) > 0:
        g.iloc[0, g.columns.get_loc("__is_pref")] = True
    return g

def _build_preferente(df_multi: pd.DataFrame) -> pd.DataFrame:
    """Devuelve DF con proveedor preferente por Product_ID, robusto a variantes de encabezado."""
    df_multi = _clean_cols(df_multi)
    pid = _find_col(df_multi, ["Product_ID","product_id","ProductId","id_producto"])

    cols_l = {c.lower(): c for c in df_multi.columns}
    if "__is_pref" in cols_l:
        mark = cols_l["__is_pref"]
        pref = df_multi[df_multi[mark] == True].copy()
    elif "is_preferred" in cols_l:
        mark = cols_l["is_preferred"]
        pref = df_multi[df_multi[mark] == True].copy()
    else:
        marked = df_multi.groupby(pid, group_keys=False).apply(_preferente_por_heuristica)
        pref = marked[marked["__is_pref"]].copy()

    pref = pref.rename(columns={
        "supplier_id":"preferred_supplier_id",
        "precio":"preferred_price",
        "lead_time":"preferred_lead_time",
        "disponibilidad":"preferred_disponibilidad",
        "prioridad":"preferred_prioridad",
        "lead_time_bucket":"preferred_lead_time_bucket",
        pid: "Product_ID",
    })
    keep = [
        "Product_ID","preferred_supplier_id","preferred_price","preferred_lead_time",
        "preferred_disponibilidad","preferred_prioridad","preferred_lead_time_bucket",
    ]
    for k in keep:
        if k not in pref.columns:
            pref[k] = np.nan
    return pref[keep].drop_duplicates()

def _rank_internos(df_int: pd.DataFrame) -> pd.DataFrame:
    """Ranking determinista para internos (mejor primero)."""
    df = df_int.copy()
    for c in ["prioridad","precio","lead_time","disponibilidad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(
        by=["Product_ID","prioridad","precio","lead_time","disponibilidad"],
        ascending=[True, True, True, True, False],
        kind="mergesort",
    )
    df["rank"] = df.groupby("Product_ID").cumcount() + 1
    return df

def _rank_externos(df_ext: pd.DataFrame, min_score: float) -> pd.DataFrame:
    """Ranking para externos por score (filtrando por umbral)."""
    df = df_ext.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["score"].notna() & (df["score"] >= float(min_score))]
    df = df.sort_values(["Product_ID","score"], ascending=[True, False], kind="mergesort")
    df["rank"] = df.groupby("Product_ID").cumcount() + 1
    return df

def _read_if_exists(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p and p.exists() else None

def _to_parquet_or_csv(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dirs(out_path.parent)
    suf = out_path.suffix.lower()
    if suf == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suf == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)

# ==== 3. LÓGICA PRINCIPAL =====================================================
def construir_vistas(
    *,
    path_unificado: str | Path,
    path_multi: str | Path,
    path_catalogo: str | Path | None = None,
    path_stock: str | Path | None = None,
    path_consumo: str | Path | None = None,
    min_score: float = 0.70,
) -> dict[str, pd.DataFrame]:
    """
    Devuelve un dict con:
      - ui_substitutes, ui_products
      - engine_substitution_rules, engine_supplier_preferred, engine_replenishment_view
    No escribe a disco.
    """
    # Carga + limpieza de columnas
    base = _clean_cols(pd.read_csv(path_unificado))
    df_multi = _clean_cols(pd.read_csv(path_multi))
    log.info("unificado cols: %s", list(base.columns))
    log.info("multi cols: %s", list(df_multi.columns))

    df_catalogo = _read_if_exists(Path(path_catalogo)) if path_catalogo else None
    if df_catalogo is not None:
        df_catalogo = _clean_cols(df_catalogo)
        log.info("catalogo cols: %s", list(df_catalogo.columns))
    df_stock = _read_if_exists(Path(path_stock)) if path_stock else None
    if df_stock is not None:
        df_stock = _clean_cols(df_stock)
        log.info("stock cols: %s", list(df_stock.columns))
    df_cons = _read_if_exists(Path(path_consumo)) if path_consumo else None
    if df_cons is not None:
        df_cons = _clean_cols(df_cons)
        log.info("consumo cols: %s", list(df_cons.columns))

    # Normalizar columnas esperadas en base
    needed = ["Product_ID","tipo","Substitute_Product_ID","Substitute_Supplier_ID",
              "score","precio","disponibilidad","lead_time","prioridad","lead_time_bucket"]
    for c in needed:
        if c not in base.columns:
            base[c] = np.nan

    # Preferente
    df_pref = _build_preferente(df_multi)

    # Ranking por tipo
    internos = base[base["tipo"]=="interno"].copy()
    externos = base[base["tipo"]=="externo"].copy()
    internos_rank = _rank_internos(internos)
    externos_rank = _rank_externos(externos, min_score)

    # UI · substitutes
    ui_substitutes = pd.concat([internos_rank, externos_rank], ignore_index=True)
    ui_substitutes = ui_substitutes.sort_values(["Product_ID","tipo","rank"]).reset_index(drop=True)

    # UI · products (contadores + preferente + catálogo + stock/consumo + top externo)
    cnt = ui_substitutes.groupby(["Product_ID","tipo"]).size().unstack(fill_value=0)
    if "interno" not in cnt.columns: cnt["interno"] = 0
    if "externo" not in cnt.columns: cnt["externo"] = 0
    cnt = cnt.rename(columns={"interno":"subs_internos_count","externo":"subs_externos_count"}).reset_index()

    top_ext = (externos_rank.sort_values(["Product_ID","rank"])
               .groupby("Product_ID").first().reset_index()
               .rename(columns={"Substitute_Product_ID":"top_externo_id","score":"top_externo_score"}))[
                   ["Product_ID","top_externo_id","top_externo_score"]
               ]

    ui_products = cnt.merge(df_pref, on="Product_ID", how="left")

    # Catálogo enriquecido
    if df_catalogo is not None:
        tmp = df_catalogo.copy()
        pid_col = _find_col(tmp, ["Product_ID","product_id","id_producto"])
        tmp = tmp.rename(columns={pid_col:"Product_ID"})
        for k in ["name","category","uom","pack_size"]:
            if k not in tmp.columns: tmp[k] = np.nan
        ui_products = ui_products.merge(tmp[["Product_ID","name","category","uom","pack_size"]], on="Product_ID", how="left")

    # Stock
    if df_stock is not None:
        tmp = df_stock.copy()
        pid_col = _find_col(tmp, ["Product_ID","product_id","id_producto"])
        tmp = tmp.rename(columns={pid_col:"Product_ID"})
        if "on_hand" not in tmp.columns: tmp["on_hand"] = np.nan
        ui_products = ui_products.merge(tmp[["Product_ID","on_hand"]], on="Product_ID", how="left")

    # Consumo
    if df_cons is not None:
        tmp = df_cons.copy()
        pid_col = _find_col(tmp, ["Product_ID","product_id","id_producto"])
        tmp = tmp.rename(columns={pid_col:"Product_ID"})
        for k in ["consumo_30d","consumo_90d","consumo_diario","dias_cobertura"]:
            if k not in tmp.columns: tmp[k] = np.nan
        ui_products = ui_products.merge(tmp[["Product_ID","consumo_30d","consumo_90d","consumo_diario","dias_cobertura"]], on="Product_ID", how="left")

    cols_products = [
        "Product_ID","name","category","uom","pack_size",
        "preferred_supplier_id","preferred_price","preferred_lead_time",
        "preferred_disponibilidad","preferred_prioridad","preferred_lead_time_bucket",
        "subs_internos_count","subs_externos_count","top_externo_id","top_externo_score",
    ]
    for opt in ["on_hand","consumo_30d","consumo_90d","consumo_diario","dias_cobertura"]:
        if opt in ui_products.columns:
            cols_products.append(opt)
    ui_products = ui_products[[c for c in cols_products if c in ui_products.columns]]

    # Engine · rules
    engine_substitution_rules = ui_substitutes.rename(columns={
        "Substitute_Product_ID":"alt_product_id",
        "Substitute_Supplier_ID":"alt_supplier_id",
    })[
        ["Product_ID","tipo","alt_product_id","alt_supplier_id","score","precio","lead_time","disponibilidad","prioridad","rank"]
    ].reset_index(drop=True)
    engine_substitution_rules["alt_type"] = engine_substitution_rules["tipo"].map({"interno":"internal","externo":"external"})
    engine_substitution_rules = engine_substitution_rules.drop(columns=["tipo"])

    # Engine · preferred
    engine_supplier_preferred = df_pref.rename(columns={
        "Product_ID":"product_id",
        "preferred_price":"precio",
        "preferred_lead_time":"lead_time",
    })[
        ["product_id","preferred_supplier_id","precio","lead_time","preferred_prioridad","preferred_disponibilidad","preferred_lead_time_bucket"]
    ]

    # Engine · replenishment
    subs_count = ui_substitutes.groupby("Product_ID").size().rename("substitutes_n").reset_index()
    engine_replenishment_view = df_pref.merge(subs_count, on="Product_ID", how="left").fillna({"substitutes_n":0})
    if df_stock is not None:
        tmp = df_stock.copy()
        pid_col = _find_col(tmp, ["Product_ID","product_id","id_producto"])
        tmp = tmp.rename(columns={pid_col:"Product_ID"})
        if "on_hand" not in tmp.columns: tmp["on_hand"] = np.nan
        engine_replenishment_view = engine_replenishment_view.merge(tmp[["Product_ID","on_hand"]], on="Product_ID", how="left")
    if df_cons is not None:
        tmp = df_cons.copy()
        pid_col = _find_col(tmp, ["Product_ID","product_id","id_producto"])
        tmp = tmp.rename(columns={pid_col:"Product_ID"})
        for k in ["consumo_diario","dias_cobertura"]:
            if k not in tmp.columns: tmp[k] = np.nan
        engine_replenishment_view = engine_replenishment_view.merge(tmp[["Product_ID","consumo_diario","dias_cobertura"]], on="Product_ID", how="left")

    return {
        "ui_substitutes": ui_substitutes,
        "ui_products": ui_products,
        "engine_substitution_rules": engine_substitution_rules,
        "engine_supplier_preferred": engine_supplier_preferred,
        "engine_replenishment_view": engine_replenishment_view,
    }

# ==== 4. EXPORTACIÓN / I-O OPCIONAL ==========================================
def exportar_vista(df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    ensure_dirs(out_path.parent)
    _to_parquet_or_csv(df, out_path)
    return out_path

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construye vistas para UI y Motor a partir del unificado de sustitutos.")
    p.add_argument("--unificado", type=str, default=str(PROCESSED_DIR / "substitutes_unified.csv"),
                   help="Ruta del CSV unificado de sustitutos.")
    p.add_argument("--multi", type=str, default=str(CLEAN_DIR / "supplier_catalog_multi.csv"),
                   help="Ruta del CSV multiproveedor.")
    p.add_argument("--catalogo", type=str, default=str(PROCESSED_DIR / "catalog_items_enriquecido.csv"),
                   help="(Opcional) Catálogo con nombre/categoría/uom/pack.")
    p.add_argument("--stock", type=str, default=str(PROCESSED_DIR / "stock_positions.csv"),
                   help="(Opcional) Métricas de stock.")
    p.add_argument("--consumo", type=str, default=str(PROCESSED_DIR / "consumo_agg.csv"),
                   help="(Opcional) Métricas de consumo.")
    p.add_argument("--min_score", type=float, default=0.70,
                   help="Umbral mínimo de score para externos.")
    p.add_argument("--out_ui", type=str, default=None,
                   help="(Opcional) Directorio o fichero base para exportar vistas UI.")
    p.add_argument("--out_engine", type=str, default=None,
                   help="(Opcional) Directorio o fichero base para exportar vistas Engine.")
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    views = construir_vistas(
        path_unificado=args.unificado,
        path_multi=args.multi,
        path_catalogo=args.catalogo,
        path_stock=args.stock,
        path_consumo=args.consumo,
        min_score=args.min_score,
    )

    # Export opcional
    if args.out_ui:
        out_ui = Path(args.out_ui)
        out_dir = out_ui.parent if out_ui.suffix else out_ui
        log.info("Exportando vistas UI en: %s", out_dir)
        exportar_vista(views["ui_products"], out_dir / "ui_products.parquet")
        exportar_vista(views["ui_substitutes"], out_dir / "ui_substitutes.parquet")

    if args.out_engine:
        out_eng = Path(args.out_engine)
        out_dir = out_eng.parent if out_eng.suffix else out_eng
        log.info("Exportando vistas Engine en: %s", out_dir)
        exportar_vista(views["engine_substitution_rules"], out_dir / "engine_substitution_rules.parquet")
        exportar_vista(views["engine_supplier_preferred"], out_dir / "engine_supplier_preferred.parquet")
        exportar_vista(views["engine_replenishment_view"], out_dir / "engine_replenishment_view.parquet")

    log.info("Vistas construidas correctamente.")

if __name__ == "__main__":
    main()

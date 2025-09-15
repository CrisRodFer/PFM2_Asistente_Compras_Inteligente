# =============================================================================
# Script: proveedor_preferente.py
# Descripción:
#   FASE 9.3 · Proveedor preferente y alternativas internas
#
#   Entradas:
#     - data/processed/catalog_items.parquet
#     - data/clean/supplier_catalog.csv (o .parquet/.xlsx)
#         columnas mínimas: Product_ID, supplier_id, precio, disponibilidad, lead_time
#         opcionales: prioridad, lead_time_bucket, supplier_sku
#
#   Reglas preferente (orden):
#     1) disponibilidad DESC
#     2) lead_time ASC
#     3) precio ASC
#     4) prioridad ASC (si existe)
#
#   Score alternativas internas (respecto al preferente):
#     score = 0.5*rel_disp + 0.25*rel_lt + 0.25*rel_px
#       rel_disp = min(1, disp_alt / disp_pref)
#       rel_lt   = min(1, lt_pref / lt_alt)
#       rel_px   = min(1, px_pref / px_alt)
#
#   Salidas:
#     - data/processed/supplier_catalog_clean.(parquet/csv)
#     - data/processed/preferred_supplier.(parquet/csv)
#     - data/processed/substitutes_internal.(parquet/csv)
#     - reports/fase9_3_qc.csv + fase9_3_resumen.txt
#
# Dependencias: pandas, numpy, pyarrow, (openpyxl si XLSX)
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG RUTAS =========================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"
REPORTS_DIR = ROOT_DIR / "reports"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def _normalize_product_id_series(s: pd.Series) -> pd.Series:
    """Normaliza Product_ID: 1000.0/'1000 '/001000 -> '1000' (string o NaN)."""
    s_num = pd.to_numeric(s, errors="coerce")
    out = s.astype("string")
    m = s_num.notna()
    out.loc[m] = s_num.loc[m].astype("Int64").astype(str)
    out = out.str.strip()
    out = out.where(~out.isin(["", "nan", "None"]), pd.NA)
    return out

def _read_catalog_items(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe catalog_items: {path}")
    df = pd.read_parquet(path)
    if "Product_ID" not in df.columns:
        raise ValueError("catalog_items debe contener 'Product_ID'.")
    df["Product_ID"] = _normalize_product_id_series(df["Product_ID"])
    return df

def _read_supplier_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe supplier_catalog: {path}")
    if path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    required = ["Product_ID", "supplier_id", "precio", "disponibilidad", "lead_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en supplier_catalog: {missing}")
    df["Product_ID"] = _normalize_product_id_series(df["Product_ID"])
    df["supplier_id"] = df["supplier_id"].astype("string").str.strip()
    for c in ["precio", "disponibilidad", "lead_time", "prioridad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["Product_ID", "supplier_id"], keep="first")
    return df

def _pick_preferred(group: pd.DataFrame) -> pd.Series:
    order_cols = ["disponibilidad", "lead_time", "precio"]
    ascending = [False, True, True]
    if "prioridad" in group.columns:
        order_cols.append("prioridad")
        ascending.append(True)
    # mergesort para estabilidad reproducible
    ordered = group.sort_values(order_cols, ascending=ascending, kind="mergesort")
    return ordered.iloc[0]

def _build_preferred_and_internal(sup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (preferred_supplier, substitutes_internal)."""
    # --- columnas base a conservar en preferente (incl. opcionales)
    cols_base = ["Product_ID", "supplier_id", "precio", "disponibilidad", "lead_time"]
    if "prioridad" in sup.columns:
        cols_base.append("prioridad")
    if "lead_time_bucket" in sup.columns:
        cols_base.append("lead_time_bucket")

    # --- preferente por producto
    pref = (
        sup.groupby("Product_ID", as_index=False)
           .apply(_pick_preferred)
           .reset_index(drop=True)[cols_base]
           .rename(columns={
               "supplier_id": "supplier_id_pref",
               "precio": "precio_pref",
               "disponibilidad": "disp_pref",
               "lead_time": "lead_time_pref",
               "prioridad": "prioridad_pref" if "prioridad" in sup.columns else None,
               "lead_time_bucket": "lead_time_bucket_pref" if "lead_time_bucket" in sup.columns else None
           })
    )

    # --- join para calcular alternativas
    join_cols = ["Product_ID", "supplier_id_pref", "precio_pref", "disp_pref", "lead_time_pref"]
    if "prioridad" in sup.columns:
        join_cols.append("prioridad_pref")
    if "lead_time_bucket" in sup.columns:
        join_cols.append("lead_time_bucket_pref")

    joined = sup.merge(pref[join_cols], on="Product_ID", how="left")

    # --- alternativos ≠ preferente
    alt = joined[joined["supplier_id"] != joined["supplier_id_pref"]].copy()
    if alt.empty:
        subs_internal = pd.DataFrame(columns=[
            "Product_ID", "supplier_id", "tipo", "priority", "score_sustitucion",
            "precio", "disponibilidad", "lead_time"
        ])
        return pref, subs_internal

    # --- score componentes
    rel_disp = (alt["disponibilidad"] / alt["disp_pref"].replace(0, np.nan)).clip(upper=1).fillna(0)
    rel_lt   = (alt["lead_time_pref"] / alt["lead_time"].replace(0, np.nan)).clip(upper=1).fillna(0)
    rel_px   = (alt["precio_pref"] / alt["precio"].replace(0, np.nan)).clip(upper=1).fillna(0)
    alt["score_sustitucion"] = (0.5*rel_disp + 0.25*rel_lt + 0.25*rel_px).clip(0, 1)

    # --- prioridad alternativas
    alt = alt.sort_values(
        by=["Product_ID", "disponibilidad", "lead_time", "precio"],
        ascending=[True, False, True, True],
        kind="mergesort"
    )
    alt["priority"] = alt.groupby("Product_ID").cumcount() + 1

    subs_internal = alt.assign(tipo="supplier_alt")[
        ["Product_ID", "supplier_id", "tipo", "priority", "score_sustitucion", "precio", "disponibilidad", "lead_time"]
    ].reset_index(drop=True)

    return pref, subs_internal

def _qc_report(cat: pd.DataFrame, sup: pd.DataFrame, pref: pd.DataFrame, subs_internal: pd.DataFrame) -> pd.DataFrame:
    alerts = []
    items_cat  = set(cat["Product_ID"].dropna().unique())
    items_sup  = set(sup["Product_ID"].dropna().unique())
    items_pref = set(pref["Product_ID"].dropna().unique())

    # sin proveedor
    for it in sorted(items_cat - items_sup):
        alerts.append({"tipo": "WARN", "Product_ID": it, "detalle": "Producto sin proveedor en supplier_catalog"})

    # multiproveedor
    multi_counts = sup.groupby("Product_ID")["supplier_id"].nunique()
    for it, n in multi_counts[multi_counts > 1].items():
        alerts.append({"tipo": "INFO", "Product_ID": it, "detalle": f"{int(n)} proveedores (se genera ranking interno)"})

    # multiproveedor esperado pero sin alternativas
    con_2omas = set(multi_counts[multi_counts > 1].index.tolist())
    con_alt   = set(subs_internal["Product_ID"].unique().tolist())
    for it in sorted(con_2omas - con_alt):
        alerts.append({"tipo": "WARN", "Product_ID": it, "detalle": "Multiproveedor sin substitutes_internal (revisar datos)"})

    # sin preferente (raro)
    for it in sorted(items_cat - items_pref):
        alerts.append({"tipo": "WARN", "Product_ID": it, "detalle": "Sin proveedor preferente asignado"})

    return pd.DataFrame(alerts)

# ==== 3. PIPELINE =============================================================
def ejecutar(catalog_items_path: Path, supplier_catalog_path: Path):
    log.info("== FASE 9.3 | Proveedor preferente y alternativas internas ==")

    cat = _read_catalog_items(catalog_items_path)
    sup_raw = _read_supplier_catalog(supplier_catalog_path)

    # filtra a productos del catálogo canónico
    sup = sup_raw[sup_raw["Product_ID"].isin(set(cat["Product_ID"].dropna()))].copy()

    pref, subs_internal = _build_preferred_and_internal(sup)
    rep = _qc_report(cat, sup, pref, subs_internal)

    # resumen
    n_items_cat = cat["Product_ID"].nunique()
    n_items_sup = sup["Product_ID"].nunique()
    n_pref      = pref["Product_ID"].nunique() if not pref.empty else 0
    n_multi     = sup.groupby("Product_ID")["supplier_id"].nunique().gt(1).sum()
    n_con_alt   = subs_internal["Product_ID"].nunique() if not subs_internal.empty else 0
    n_warn      = len(rep[rep["tipo"]=="WARN"]) if not rep.empty else 0

    resumen = (
        "FASE 9.3 · Resumen proveedor preferente y alternativas internas\n"
        f"- Productos en catálogo canónico: {n_items_cat}\n"
        f"- Productos con proveedor (supplier_catalog): {n_items_sup}\n"
        f"- Productos con proveedor preferente asignado: {n_pref}\n"
        f"- Productos con >1 proveedor: {n_multi}\n"
        f"- Productos con alternativas internas generadas: {n_con_alt}\n"
        f"- Incidencias (WARN) en QC: {n_warn}\n"
    )
    print("\n" + resumen)
    return sup, pref, subs_internal, rep, resumen

# ==== 4. EXPORT ===============================================================
def exportar(sup: pd.DataFrame, pref: pd.DataFrame, subs_internal: pd.DataFrame, rep: pd.DataFrame, resumen: str) -> None:
    ensure_dirs(PROCESSED_DIR, REPORTS_DIR)
    # supplier_catalog_clean
    sup.to_parquet(PROCESSED_DIR / "supplier_catalog_clean.parquet", index=False)
    sup.to_csv(PROCESSED_DIR / "supplier_catalog_clean.csv", index=False, encoding="utf-8-sig")
    # preferred_supplier
    pref.to_parquet(PROCESSED_DIR / "preferred_supplier.parquet", index=False)
    pref.to_csv(PROCESSED_DIR / "preferred_supplier.csv", index=False, encoding="utf-8-sig")
    # substitutes_internal
    subs_internal.to_parquet(PROCESSED_DIR / "substitutes_internal.parquet", index=False)
    subs_internal.to_csv(PROCESSED_DIR / "substitutes_internal.csv", index=False, encoding="utf-8-sig")
    # QC + resumen
    rep.to_csv(REPORTS_DIR / "fase9_3_qc.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "fase9_3_resumen.txt").write_text(resumen, encoding="utf-8")
    log.info("Exportados supplier_catalog_clean / preferred_supplier / substitutes_internal y reportes.")

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE 9.3 · Proveedor preferente y alternativas internas.")
    p.add_argument("--catalog_items", type=str, default=str(PROCESSED_DIR / "catalog_items.parquet"),
                   help="Ruta a catalog_items.parquet (salida 9.1).")
    p.add_argument("--supplier_catalog", type=str, default=str(CLEAN_DIR / "supplier_catalog.csv"),
                   help="Ruta a supplier_catalog (csv/parquet/xlsx).")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    try:
        sup, pref, subs_internal, rep, resumen = ejecutar(Path(args.catalog_items), Path(args.supplier_catalog))
        exportar(sup, pref, subs_internal, rep, resumen)
        log.info("FASE 9.3 completada.")
    except Exception as e:
        log.exception("Error en FASE 9.3: %s", e)
        raise

if __name__ == "__main__":
    main()


# =============================================================================
# Script: consolidar_outliers_fase6.py
# FASE 6.3 — Resultados consolidados y decisiones finales
#
# Qué hace:
#   - Une los resultados de 6.1 (validación complementaria) y 6.2 (DBSCAN)
#     a nivel (product_id, year) y resuelve duplicidades con prioridad:
#        top_venta > mixto > pico_aislado
#   - Añade la decisión final estandarizada:
#        top_venta -> sin_cambio
#        mixto     -> alerta_pendiente
#        pico_aislado -> suavizado_a015
#   - Calcula métricas globales de cobertura:
#        % productos afectados, % registros y % ventas (sobre días candidatos)
#   - Exporta:
#        - reports/outliers/outliers_resumen.csv (tabla consolidada producto-año)
#        - reports/outliers/outliers_resumen_metricas.csv (métricas)
#
# Entradas (por defecto):
#   - data/processed/demanda_all_adjusted_postnoise.parquet
#   - reports/outliers/outliers_candidatos_nuevos_productos.csv   (FASE 6.1)
#   - reports/outliers/outliers_dbscan_productos.csv               (FASE 6.2)
#   - (opcionales, para métricas por días)
#     reports/outliers/outliers_candidatos_nuevos_dias.csv
#     reports/outliers/outliers_dbscan_dias.csv
#
# Uso:
#   python scripts/analysis/consolidar_outliers_fase6.py
#   # o con rutas personalizadas:
#   python scripts/analysis/consolidar_outliers_fase6.py \
#       --root "C:\ruta\al\repo"
#
# Dependencias: pandas numpy pyarrow
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(Path(__file__).stem)

# ----------------------- Utilidades ------------------------------------------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _to_dt(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce")

def _to_str(x: pd.Series) -> pd.Series:
    # robusto frente a NaNs y tipos mixtos
    return x.astype("string").fillna(pd.NA)

def _to_int(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype("Int64")

# ----------------------- Carga -----------------------------------------------
def cargar_demanda(parquet_path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(parquet_path)
    df = _std_cols(df)
    if "date" not in df.columns or "product_id" not in df.columns:
        raise KeyError("El parquet debe contener 'date' y 'product_id'.")

    # normalización de tipos
    df["date"] = _to_dt(df["date"])
    df["product_id"] = _to_str(df["product_id"])
    df["year"] = df["date"].dt.year.astype("Int64")

    # detectar columna de cantidad
    if "sales_quantity" in df.columns:
        qty = "sales_quantity"
    elif "demand_day" in df.columns:
        qty = "demand_day"
    elif "demand_day_priceadj" in df.columns:
        qty = "demand_day_priceadj"
    else:
        raise KeyError("No se encontró columna de cantidad (sales_quantity/demand_day/demand_day_priceadj).")

    return df, qty

def cargar_fuentes(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    f61 = report_dir / "outliers_candidatos_nuevos_productos.csv"
    f62 = report_dir / "outliers_dbscan_productos.csv"
    if not f61.exists() or not f62.exists():
        raise FileNotFoundError(f"Faltan entradas: {f61} (6.1) o {f62} (6.2).")

    df61 = _std_cols(pd.read_csv(f61))
    df62 = _std_cols(pd.read_csv(f62))
    for d in (df61, df62):
        d["product_id"] = _to_str(d["product_id"])
        d["year"] = _to_int(d["year"])

    df61["origen"] = "val_comp"
    df62["origen"] = "dbscan"
    return df61, df62

def cargar_dias(report_dir: Path) -> pd.DataFrame:
    """Une días candidatos de 6.1 y 6.2 (para métricas)."""
    d1 = report_dir / "outliers_candidatos_nuevos_dias.csv"
    d2 = report_dir / "outliers_dbscan_dias.csv"
    frames = []
    for fp in (d1, d2):
        if fp.exists():
            d = _std_cols(pd.read_csv(fp, parse_dates=["date"]))
            if "year" not in d.columns:
                d["year"] = d["date"].dt.year
            d["product_id"] = _to_str(d["product_id"])
            d["date"] = _to_dt(d["date"])
            frames.append(d[["product_id", "date"]].drop_duplicates())
    if not frames:
        return pd.DataFrame(columns=["product_id", "date"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    # asegurar tipos correctos
    out["product_id"] = _to_str(out["product_id"])
    out["date"] = _to_dt(out["date"])
    return out

# ----------------------- Consolidación ---------------------------------------
def consolidar_producto_anio(df61: pd.DataFrame, df62: pd.DataFrame) -> pd.DataFrame:
    cols_keep = ["product_id", "year", "tipo_outlier", "decision_aplicada", "origen"]
    base = pd.concat([df61[cols_keep], df62[cols_keep]], ignore_index=True)

    # tipos homogéneos
    base["product_id"] = _to_str(base["product_id"])
    base["year"] = _to_int(base["year"])

    prioridad = {"top_venta": 3, "mixto": 2, "pico_aislado": 1}
    base["prio"] = base["tipo_outlier"].map(prioridad).fillna(0).astype(int)
    base = base.sort_values(["product_id", "year", "prio"], ascending=[True, True, False])
    res = base.drop_duplicates(subset=["product_id", "year"], keep="first").drop(columns=["prio"])

    decision_map = {"top_venta": "sin_cambio", "mixto": "alerta_pendiente", "pico_aislado": "suavizado_a015"}
    res["decision_final"] = res["tipo_outlier"].map(decision_map)
    return res

# ----------------------- Métricas --------------------------------------------
def calcular_metricas(
    df_dem: pd.DataFrame, qty_col: str, df_resumen: pd.DataFrame, dias_union: pd.DataFrame
) -> pd.DataFrame:
    # Coerción de tipos para claves
    df_resumen = df_resumen.copy()
    df_resumen["product_id"] = _to_str(df_resumen["product_id"])
    df_resumen["year"] = _to_int(df_resumen["year"])

    n_prod_total = df_dem["product_id"].nunique()
    n_prod_out = df_resumen["product_id"].nunique()
    pct_prod = 100 * n_prod_out / n_prod_total if n_prod_total else 0

    n_reg_total = len(df_dem)
    n_reg_is_out = int(((df_dem["is_outlier"] == 1) if "is_outlier" in df_dem.columns else 0).sum())

    # Métricas basadas en unión de días candidatos (si existen)
    if not dias_union.empty:
        dem_key = df_dem[["product_id", "date", qty_col]].copy()
        dem_key["product_id"] = _to_str(dem_key["product_id"])
        dem_key["date"] = _to_dt(dem_key["date"])

        dias_union = dias_union.copy()
        dias_union["product_id"] = _to_str(dias_union["product_id"])
        dias_union["date"] = _to_dt(dias_union["date"])

        affected = dem_key.merge(dias_union, on=["product_id", "date"], how="inner")
        n_reg_candidatos = len(affected)
        pct_reg_candidatos = 100 * n_reg_candidatos / n_reg_total if n_reg_total else 0
        total_qty = dem_key[qty_col].sum()
        pct_ventas_candidatos = 100 * affected[qty_col].sum() / total_qty if total_qty else 0
    else:
        n_reg_candidatos = 0
        pct_reg_candidatos = 0.0
        pct_ventas_candidatos = 0.0

    met = pd.DataFrame({
        "productos_totales": [n_prod_total],
        "productos_con_outlier": [n_prod_out],
        "pct_productos_con_outlier": [round(pct_prod, 2)],
        "registros_totales": [n_reg_total],
        "registros_is_outlier_1": [n_reg_is_out],
        "registros_dias_candidatos": [n_reg_candidatos],
        "pct_registros_dias_candidatos": [round(pct_reg_candidatos, 2)],
        "pct_ventas_dias_candidatos": [round(pct_ventas_candidatos, 2)],
    })
    return met

# ----------------------- Orquestación ----------------------------------------
def ejecutar(root_dir: Path) -> tuple[Path, Path]:
    data_dir = root_dir / "data" / "processed"
    report_dir = root_dir / "reports" / "outliers"
    report_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "demanda_all_adjusted_postnoise.parquet"
    log.info("Leyendo demanda: %s", parquet_path)
    df_dem, qty_col = cargar_demanda(parquet_path)

    log.info("Cargando fuentes 6.1 y 6.2 desde: %s", report_dir)
    df61, df62 = cargar_fuentes(report_dir)

    log.info("Consolidando producto-año…")
    df_resumen = consolidar_producto_anio(df61, df62)

    log.info("Calculando métricas de cobertura…")
    dias_union = cargar_dias(report_dir)
    met = calcular_metricas(df_dem, qty_col, df_resumen, dias_union)

    out_csv = report_dir / "outliers_resumen.csv"
    out_met = report_dir / "outliers_resumen_metricas.csv"
    df_resumen.to_csv(out_csv, index=False)
    met.to_csv(out_met, index=False)

    log.info("Guardados:\n- %s\n- %s", out_csv, out_met)
    log.info("Resumen por tipo_outlier (producto-año):\n%s",
             df_resumen["tipo_outlier"].value_counts().to_string())
    return out_csv, out_met

# ----------------------- CLI --------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE 6.3 — Consolidado de outliers y métricas")
    p.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[2]),
                   help="Raíz del proyecto (carpeta que contiene data/ y reports/).")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    try:
        out_csv, out_met = ejecutar(Path(args.root))
        print("\nOK. Archivos generados:")
        print(" -", out_csv)
        print(" -", out_met)
    except Exception as e:
        log.exception("Error consolidando outliers (Fase 6.3): %s", e)
        raise

if __name__ == "__main__":
    main()

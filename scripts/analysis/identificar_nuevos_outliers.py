
# =============================================================================
# Script: identificar_nuevos_outliers.py
# Descripción:
#   Parte 6.1 de la FASE 6 (Análisis de outliers).
#   Valida si han surgido nuevos candidatos a outlier entre los registros con
#   is_outlier = 0 en el dataset final ajustado. La detección es intra-producto
#   y se basa en criterios robustos y parametrizables:
#     - z-score robusto (MAD) > z_thresh
#     - valor > P99 del producto
#     - valor > P95 del producto y > mediana + k * MAD
#   Tras detectar días candidatos, se agrupa a nivel producto para clasificar
#   automáticamente cada producto en:
#     - top_venta    (percentil 99 por ventas anuales o top-N)
#     - pico_aislado (proporción de días candidatos < frac_aislado)
#     - mixto        (resto)
#
# Flujo del pipeline:
#   0) Config + utilidades
#   1) Carga y normalización de columnas
#   2) Filtro is_outlier = 0
#   3) Detección de días candidatos (intra-producto)
#   4) Agregación por producto y clasificación automática
#   5) Exportación de tablas a reports/outliers/
#
# Input:
#   - data/processed/demanda_all_adjusted_postnoise.parquet
#     Columnas esperadas (nombres flexibles, se normalizan):
#       Product_ID | product_id
#       Date | date
#       Demand_Day / Sales_Quantity | demand_day | sales_quantity
#       Cluster | cluster (opcional)
#       is_outlier
#
# Output:
#   - reports/outliers/outliers_candidatos_nuevos_dias.csv
#       (nivel día: product_id, date, valor, flags activas)
#   - reports/outliers/outliers_candidatos_nuevos_productos.csv
#       (nivel producto-año: clasificación, decisión recomendada)
#
# Dependencias:
#   - pandas, numpy, pyarrow (para leer parquet)
#
# Instalación rápida:
#   pip install pandas numpy pyarrow
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
# Ajustado: el script está en scripts/analysis/, la raíz está dos niveles arriba
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
OUT_OUTLIERS_DIR = REPORTS_DIR / "outliers"

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
        Path(d).mkdir(parents=True, exist_ok=True)

def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isclose(mad, 0.0):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    ren = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=ren)
    qty_col = (
        "sales_quantity" if "sales_quantity" in df.columns
        else "demand_day" if "demand_day" in df.columns
        else "demand_day_priceadj"
    )
    out = pd.DataFrame({
        "product_id": df["product_id"],
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "sales_quantity": pd.to_numeric(df[qty_col], errors="coerce"),
        "is_outlier": df["is_outlier"].astype(int),
    })
    if "cluster" in df.columns:
        out["cluster"] = df["cluster"]
    out["year"] = out["date"].dt.year
    out = out.dropna(subset=["date", "sales_quantity"])
    out = out[out["sales_quantity"] >= 0]
    return out

def detectar_candidatos_intra_producto(
    df: pd.DataFrame,
    z_thresh: float = 3.5,
    k_mad: float = 4.0,
    p95: float = 95.0,
    p99: float = 99.0,
) -> pd.DataFrame:
    def _flags(g: pd.DataFrame) -> pd.DataFrame:
        x = g["sales_quantity"]
        z = robust_zscore(x)
        p95_val = np.percentile(x, p95)
        p99_val = np.percentile(x, p99)
        med = x.median()
        mad = (x - med).abs().median()
        flag_z = z.abs() > z_thresh
        flag_p99 = x > p99_val
        if mad == 0 or np.isclose(mad, 0.0):
            flag_p95kmad = x > p99_val
        else:
            flag_p95kmad = (x > p95_val) & (x > (med + k_mad * mad))
        out = pd.DataFrame({
            "flag_z": flag_z,
            "flag_p99": flag_p99,
            "flag_p95kmad": flag_p95kmad,
        }, index=g.index)
        out["flag_any"] = out.any(axis=1)
        return out

    df = df.sort_values(["product_id", "year", "date"]).copy()
    flags = df.groupby(["product_id", "year"], group_keys=False).apply(_flags)
    return pd.concat([df.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)

def clasificar_producto_auto(
    df_flags: pd.DataFrame,
    frac_aislado: float = 0.05,
    top_percentil: float = 99.0,
    top_n: int | None = None,
) -> pd.DataFrame:
    yearly = (df_flags.groupby(["product_id", "year"], as_index=False)["sales_quantity"]
              .sum()
              .rename(columns={"sales_quantity": "year_total"}))
    yearly["is_top"] = False
    for y, sub in yearly.groupby("year"):
        umbral = np.percentile(sub["year_total"], top_percentil)
        yearly.loc[sub.index[sub["year_total"] >= umbral], "is_top"] = True
        if top_n is not None and len(sub) > top_n:
            extra = sub.nlargest(top_n, "year_total").index
            yearly.loc[extra, "is_top"] = True
    prop = (df_flags.groupby(["product_id", "year"])["flag_any"]
            .mean()
            .reset_index(name="prop_dias_candidatos"))
    base = yearly.merge(prop, on=["product_id", "year"], how="left")
    base["tipo_outlier"] = np.where(
        base["is_top"], "top_venta",
        np.where(base["prop_dias_candidatos"] < frac_aislado, "pico_aislado", "mixto")
    )
    decision_map = {
        "top_venta": "sin_cambio",
        "pico_aislado": "suavizado_a015",
        "mixto": "alerta_pendiente",
    }
    base["decision_aplicada"] = base["tipo_outlier"].map(decision_map)
    return base

# ==== 3. LÓGICA PRINCIPAL =====================================================
def ejecutar_validacion_nuevos(
    in_parquet: Path,
    z_thresh: float = 3.5,
    k_mad: float = 4.0,
    p95: float = 95.0,
    p99: float = 99.0,
    frac_aislado: float = 0.05,
    top_percentil: float = 99.0,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Leyendo parquet: %s", in_parquet)
    df_raw = pd.read_parquet(in_parquet)
    df = normalizar_columnas(df_raw)
    df0 = df[df["is_outlier"] == 0].copy()
    if df0.empty:
        log.warning("No hay registros con is_outlier = 0.")
        return pd.DataFrame(), pd.DataFrame()
    log.info("Detectando días candidatos…")
    fl = detectar_candidatos_intra_producto(df0, z_thresh, k_mad, p95, p99)
    df_dias = fl[fl["flag_any"]].copy()
    df_dias = df_dias[[
        "product_id", "year", "date", "sales_quantity",
        "flag_z", "flag_p99", "flag_p95kmad"
    ]]
    log.info("Clasificando productos…")
    df_prod = clasificar_producto_auto(fl, frac_aislado, top_percentil, top_n)
    return df_dias, df_prod

# ==== 4. EXPORTACIÓN / I/O ====================================================
def exportar_resultados(df_dias: pd.DataFrame, df_prod: pd.DataFrame) -> tuple[Path, Path]:
    ensure_dirs(OUT_OUTLIERS_DIR)
    out_dias = OUT_OUTLIERS_DIR / "outliers_candidatos_nuevos_dias.csv"
    out_prod = OUT_OUTLIERS_DIR / "outliers_candidatos_nuevos_productos.csv"
    df_dias.to_csv(out_dias, index=False)
    df_prod.to_csv(out_prod, index=False)
    return out_dias, out_prod

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE 6.1 - Validación complementaria de nuevos outliers.")
    p.add_argument("--in-parquet", type=str,
                   default=str(PROCESSED_DIR / "demanda_all_adjusted_postnoise.parquet"),
                   help="Ruta al parquet final ajustado.")
    p.add_argument("--z-thresh", type=float, default=3.5)
    p.add_argument("--k-mad", type=float, default=4.0)
    p.add_argument("--p95", type=float, default=95.0)
    p.add_argument("--p99", type=float, default=99.0)
    p.add_argument("--frac-aislado", type=float, default=0.05)
    p.add_argument("--top-percentil", type=float, default=99.0)
    p.add_argument("--top-n", type=int, default=None)
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    try:
        df_dias, df_prod = ejecutar_validacion_nuevos(
            in_parquet=Path(args.in_parquet),
            z_thresh=args.z_thresh,
            k_mad=args.k_mad,
            p95=args.p95,
            p99=args.p99,
            frac_aislado=args.frac_aislado,
            top_percentil=args.top_percentil,
            top_n=args.top_n,
        )
        out_dias, out_prod = exportar_resultados(df_dias, df_prod)
        log.info("Escritos:\n- %s\n- %s", out_dias, out_prod)
    except Exception as e:
        log.exception("Fallo en FASE 6.1: %s", e)
        raise

if __name__ == "__main__":
    main()
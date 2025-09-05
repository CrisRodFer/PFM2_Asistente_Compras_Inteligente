
# =============================================================================
# Script: clasificar_outliers_dbscan.py
# Descripción:
#   FASE 6.2 – Clasificación de outliers detectados por DBSCAN.
#
#   Reutiliza la lógica de 6.1 (validación complementaria), pero aquí se
#   trabaja con los registros marcados previamente como outliers globales:
#       is_outlier = 1
#
#   Pasos:
#     1) Carga parquet final (demanda_all_adjusted_postnoise.parquet)
#     2) Normaliza columnas y filtra is_outlier = 1
#     3) Detecta días "candidatos" intra-producto con reglas robustas:
#          - |z(MAD)| > z_thresh
#          - valor > P99
#          - valor > P95 y > mediana + k*MAD
#     4) Agrega por (product_id, year) y clasifica:
#          top_venta / pico_aislado / mixto
#        (top_venta = percentil 99 o top-N anual)
#        (pico_aislado = prop_dias_candidatos < frac_aislado)
#     5) Exporta CSVs:
#          reports/outliers/outliers_dbscan_dias.csv
#          reports/outliers/outliers_dbscan_productos.csv
#          reports/outliers/outliers_dbscan_productos_consolidado.csv
#
#   NOTA: Este script solo clasifica. El cruce con calendario real se hará
#         a continuación, en un script análogo al usado en 6.1.
#
# Entradas:
#   - data/processed/demanda_all_adjusted_postnoise.parquet
#
# Dependencias:
#   pip install pandas numpy pyarrow
# =============================================================================

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd

# ==== 0. CONFIG RUTAS =========================================================
# El archivo está en: scripts/analysis/ -> la raíz del repo está dos niveles arriba
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
OUT_DIR = REPORTS_DIR / "outliers"

# ==== 1. LOGGING ==============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def robust_zscore(x: pd.Series) -> pd.Series:
    """z-score robusto basado en MAD. Si MAD≈0 devuelve 0s para evitar infinitos."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isclose(mad, 0.0):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad

def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Homogeneiza nombres y devuelve columnas clave:
      product_id, date, sales_quantity, is_outlier, (cluster opcional), year
    """
    df = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})

    # cantidad
    if "sales_quantity" in df.columns:
        qty = "sales_quantity"
    elif "demand_day" in df.columns:
        qty = "demand_day"
    elif "demand_day_priceadj" in df.columns:
        qty = "demand_day_priceadj"
    else:
        raise KeyError("No se encontró columna de cantidad (sales_quantity/demand_day/demand_day_priceadj).")

    if "date" not in df.columns:
        raise KeyError("No se encontró la columna 'date'.")
    if "product_id" not in df.columns:
        raise KeyError("No se encontró la columna 'product_id'.")
    if "is_outlier" not in df.columns:
        raise KeyError("No se encontró la columna 'is_outlier'.")

    out = pd.DataFrame({
        "product_id": df["product_id"],
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "sales_quantity": pd.to_numeric(df[qty], errors="coerce"),
        "is_outlier": pd.to_numeric(df["is_outlier"], errors="coerce").fillna(0).astype(int),
    })
    if "cluster" in df.columns:
        out["cluster"] = df["cluster"]
    out["year"] = out["date"].dt.year
    out = out.dropna(subset=["date", "sales_quantity"])
    out = out[out["sales_quantity"] >= 0]
    return out

def detectar_flags_intra(df0: pd.DataFrame,
                         z_thresh: float = 3.5,
                         k_mad: float = 4.0,
                         p95: float = 95.0,
                         p99: float = 99.0) -> pd.DataFrame:
    """
    Aplica reglas robustas por (product_id, year) y añade flags:
      flag_z, flag_p99, flag_p95kmad, flag_any
    """
    def _flags(g: pd.DataFrame) -> pd.DataFrame:
        x = g["sales_quantity"]
        z = robust_zscore(x)
        p95v = np.percentile(x, p95)
        p99v = np.percentile(x, p99)
        med = x.median()
        mad = (x - med).abs().median()

        flag_z = z.abs() > z_thresh
        flag_p99 = x > p99v
        flag_p95kmad = (x > p95v) & (x > (med + k_mad * mad)) if mad != 0 else (x > p99v)

        out = pd.DataFrame({
            "flag_z": flag_z,
            "flag_p99": flag_p99,
            "flag_p95kmad": flag_p95kmad,
        }, index=g.index)
        out["flag_any"] = out.any(axis=1)
        return out

    df0 = df0.sort_values(["product_id", "year", "date"]).copy()
    flags = df0.groupby(["product_id", "year"], group_keys=False).apply(_flags)
    return pd.concat([df0.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)

def clasificar_por_producto(df_flags: pd.DataFrame,
                            frac_aislado: float = 0.05,
                            top_percentil: float = 99.0,
                            top_n: int | None = None) -> pd.DataFrame:
    """
    Clasifica (product_id, year):
      - top_venta: total anual ≥ percentil 'top_percentil' o en top-N.
      - pico_aislado: prop_dias_candidatos < frac_aislado
      - mixto: resto
    Devuelve columnas: year_total, is_top, prop_dias_candidatos, n_dias_candidatos, n_dias, tipo_outlier, decision_aplicada.
    """
    yearly = (df_flags.groupby(["product_id", "year"], as_index=False)["sales_quantity"]
              .sum().rename(columns={"sales_quantity": "year_total"}))
    yearly["is_top"] = False
    for y, sub in yearly.groupby("year"):
        thr = np.percentile(sub["year_total"], top_percentil)
        yearly.loc[sub.index[sub["year_total"] >= thr], "is_top"] = True
        if top_n and len(sub) > top_n:
            yearly.loc[sub.nlargest(top_n, "year_total").index, "is_top"] = True

    agg = df_flags.groupby(["product_id", "year"]).agg(
        prop_dias_candidatos=("flag_any", "mean"),
        n_dias_candidatos=("flag_any", "sum"),
        n_dias=("flag_any", "size"),
    ).reset_index()

    base = yearly.merge(agg, on=["product_id", "year"], how="left")
    base["tipo_outlier"] = np.where(
        base["is_top"], "top_venta",
        np.where((base["prop_dias_candidatos"].fillna(0) < frac_aislado), "pico_aislado", "mixto")
    )
    decision_map = {"top_venta": "sin_cambio", "pico_aislado": "suavizado_a015", "mixto": "alerta_pendiente"}
    base["decision_aplicada"] = base["tipo_outlier"].map(decision_map)
    return base

def consolidar_por_producto(df_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidación 1 fila por product_id:
      prioridad: top_venta > mixto > pico_aislado (guardamos años implicados).
    """
    prio = pd.CategoricalDtype(categories=["pico_aislado", "mixto", "top_venta"], ordered=True)
    tmp = df_yearly.copy()
    tmp["tipo_outlier"] = tmp["tipo_outlier"].astype(prio)

    agg = tmp.groupby("product_id").agg(
        tipo_outlier=("tipo_outlier", "max"),
        years=("year", lambda s: ",".join(map(str, sorted(set(s))))),
        n_years=("year", "nunique"),
        year_total_sum=("year_total", "sum"),
        n_dias_candidatos_sum=("n_dias_candidatos", "sum"),
        n_dias_sum=("n_dias", "sum"),
    ).reset_index()

    decision_map = {"top_venta": "sin_cambio", "pico_aislado": "suavizado_a015", "mixto": "alerta_pendiente"}
    agg["decision_aplicada"] = agg["tipo_outlier"].astype(str).map(decision_map)
    return agg

# ==== 3. EJECUCIÓN ============================================================
def ejecutar(in_parquet: Path,
             z_thresh: float = 3.5,
             k_mad: float = 4.0,
             p95: float = 95.0,
             p99: float = 99.0,
             frac_aislado: float = 0.05,
             top_percentil: float = 99.0,
             top_n: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta la clasificación de outliers DBSCAN (is_outlier=1).
    Devuelve:
      - df_dias (nivel día con flags)
      - df_prod (nivel producto-año clasificado)
      - df_conso (consolidado 1 fila por producto)
    """
    log.info("Leyendo parquet: %s", in_parquet)
    df_raw = pd.read_parquet(in_parquet)
    df = normalizar(df_raw)

    # Solo los marcados por DBSCAN (outliers estructurales)
    df1 = df[df["is_outlier"] == 1].copy()
    if df1.empty:
        log.warning("No hay registros con is_outlier = 1.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    log.info("Detectando días candidatos intra-producto…")
    flags = detectar_flags_intra(df1, z_thresh=z_thresh, k_mad=k_mad, p95=p95, p99=p99)

    df_dias = flags[flags["flag_any"]].copy()[[
        "product_id", "year", "date", "sales_quantity", "flag_z", "flag_p99", "flag_p95kmad"
    ]]

    log.info("Clasificando productos (top_venta / pico_aislado / mixto)…")
    df_prod = clasificar_por_producto(flags, frac_aislado=frac_aislado,
                                      top_percentil=top_percentil, top_n=top_n)

    # Duplicados reales (product_id, year) -> nos quedamos la primera
    if df_prod.duplicated(subset=["product_id", "year"]).any():
        log.warning("Se detectaron duplicados por (product_id, year). Se conservan primeras ocurrencias.")
        df_prod = df_prod.drop_duplicates(subset=["product_id", "year"], keep="first")

    df_conso = consolidar_por_producto(df_prod)

    return df_dias, df_prod, df_conso

# ==== 4. EXPORT ===============================================================
def exportar(df_dias: pd.DataFrame, df_prod: pd.DataFrame, df_conso: pd.DataFrame) -> tuple[Path, Path, Path]:
    ensure_dirs(OUT_DIR)
    p1 = OUT_DIR / "outliers_dbscan_dias.csv"
    p2 = OUT_DIR / "outliers_dbscan_productos.csv"
    p3 = OUT_DIR / "outliers_dbscan_productos_consolidado.csv"
    df_dias.to_csv(p1, index=False)
    df_prod.to_csv(p2, index=False)
    df_conso.to_csv(p3, index=False)
    log.info("Guardados:\n- %s\n- %s\n- %s", p1, p2, p3)
    return p1, p2, p3

# ==== 5. CLI ==================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE 6.2 - Clasificación de outliers DBSCAN (is_outlier=1)")
    p.add_argument("--in-parquet", type=str, default=str(PROCESSED_DIR / "demanda_all_adjusted_postnoise.parquet"),
                   help="Ruta al parquet final ajustado.")
    p.add_argument("--z-thresh", type=float, default=3.5, help="Umbral |z(MAD)| para días candidatos.")
    p.add_argument("--k-mad", type=float, default=4.0, help="k en regla P95 + k*MAD.")
    p.add_argument("--p95", type=float, default=95.0, help="Percentil intra-producto (P95).")
    p.add_argument("--p99", type=float, default=99.0, help="Percentil intra-producto fuerte (P99).")
    p.add_argument("--frac-aislado", type=float, default=0.05, help="Umbral de proporción para 'pico_aislado'.")
    p.add_argument("--top-percentil", type=float, default=99.0, help="Percentil anual para 'top_venta'.")
    p.add_argument("--top-n", type=int, default=None, help="Top-N anual alternativo para 'top_venta'.")
    return p.parse_args()

def main() -> None:
    a = parse_args()
    try:
        df_dias, df_prod, df_conso = ejecutar(
            in_parquet=Path(a.in_parquet),
            z_thresh=a.z_thresh, k_mad=a.k_mad, p95=a.p95, p99=a.p99,
            frac_aislado=a.frac_aislado, top_percentil=a.top_percentil, top_n=a.top_n,
        )
        exportar(df_dias, df_prod, df_conso)
        if not df_prod.empty:
            log.info("Resumen por tipo (producto-año):\n%s", df_prod["tipo_outlier"].value_counts().to_string())
            log.info("Productos únicos (consolidado): %d", df_conso["product_id"].nunique())
    except Exception as e:
        log.exception("Fallo en FASE 6.2 (DBSCAN): %s", e)
        raise

if __name__ == "__main__":
    main()

# =============================================================================
# Script: generar_subset_modelado.py
# Objetivo:
#   Crear el subset final de modelado añadiendo dos columnas al parquet base:
#     - tipo_outlier_year        (top_venta | pico_aislado | mixto | none)
#     - decision_outlier_year    (sin_cambio | suavizado_a015 | alerta_pendiente | none)
#
# Notas:
#   - NO modifica la columna is_outlier original (se mantiene como "origen DBSCAN").
#   - Une por (product_id, year) usando la tabla consolidada: reports/outliers/outliers_resumen.csv
#   - Guarda el resultado en: data/processed/subset_modelado.parquet
#
# Uso:
#   python scripts/analysis/generar_subset_modelado.py
#   # o con rutas personalizadas:
#   python scripts/analysis/generar_subset_modelado.py \
#       --root "C:\ruta\al\repo"
#
# Dependencias: pandas, pyarrow
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(Path(__file__).stem)

# ---------- Helpers ----------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _to_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna(pd.NA)

def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

# ---------- Core ----------
def ejecutar(root_dir: Path) -> Path:
    data_dir = root_dir / "data" / "processed"
    reports_dir = root_dir / "reports" / "outliers"
    base_parquet = data_dir / "demanda_all_adjusted_postnoise.parquet"
    resumen_csv  = reports_dir / "outliers_resumen.csv"
    out_parquet  = data_dir / "subset_modelado.parquet"

    if not base_parquet.exists():
        raise FileNotFoundError(f"No existe el parquet base: {base_parquet}")
    if not resumen_csv.exists():
        raise FileNotFoundError(f"No existe el consolidado: {resumen_csv}")

    # 1) Cargar base de demanda
    log.info("Leyendo parquet base: %s", base_parquet)
    df = pd.read_parquet(base_parquet)
    df = _std_cols(df)

    # normalizar claves
    if "date" not in df.columns or "product_id" not in df.columns:
        raise KeyError("El parquet debe contener 'date' y 'product_id'.")
    df["date"] = _to_dt(df["date"])
    df["product_id"] = _to_str(df["product_id"])
    df["year"] = df["date"].dt.year.astype("Int64")

    # 2) Cargar consolidado de outliers (producto-año)
    log.info("Leyendo consolidado: %s", resumen_csv)
    res = pd.read_csv(resumen_csv)
    res = _std_cols(res)
    need = {"product_id", "year", "tipo_outlier", "decision_final"}
    if not need.issubset(res.columns):
        faltan = need - set(res.columns)
        raise KeyError(f"Faltan columnas en outliers_resumen.csv: {faltan}")

    res["product_id"] = _to_str(res["product_id"])
    res["year"] = _to_int(res["year"])

    # 3) Merge por (product_id, year)
    log.info("Haciendo merge por (product_id, year)…")
    df_merge = df.merge(
        res[["product_id", "year", "tipo_outlier", "decision_final"]],
        on=["product_id", "year"], how="left"
    )

    # 4) Renombrar y rellenar 'none' donde no aplica
    df_merge = df_merge.rename(columns={
        "tipo_outlier": "tipo_outlier_year",
        "decision_final": "decision_outlier_year"
    })
    df_merge["tipo_outlier_year"] = df_merge["tipo_outlier_year"].fillna("none")
    df_merge["decision_outlier_year"] = df_merge["decision_outlier_year"].fillna("none")

    # 5) Guardar subset final
    log.info("Guardando subset final: %s", out_parquet)
    df_merge.to_parquet(out_parquet, index=False)

    # 6) Pequeño resumen por consola
    n_total = len(df_merge)
    n_none = int((df_merge["tipo_outlier_year"] == "none").sum())
    n_tag  = n_total - n_none
    log.info("Filas totales: %s | Filas etiquetadas: %s | Sin etiqueta: %s", n_total, n_tag, n_none)

    return out_parquet

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generar subset final de modelado con etiquetas de outliers (año).")
    p.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[2]),
                   help="Raíz del proyecto (carpeta que contiene data/ y reports/).")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    try:
        out_path = ejecutar(Path(args.root))
        print("\nOK. Subset de modelado generado en:")
        print(out_path)
    except Exception as e:
        log.exception("Error generando subset de modelado: %s", e)
        raise

if __name__ == "__main__":
    main()

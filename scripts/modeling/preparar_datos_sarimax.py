
# scripts/modelado/preparar_datos_sarimax.py
# =============================================================================
# Descripción:
#   Aplica la utilidad `prepare_cluster_data()` para generar los datasets
#   por clúster (train/val/test) listos para SARIMAX (y otros modelos),
#   y los exporta a disco.
#
# Flujo del pipeline:
#   1) Leer dataset base (parquet).
#   2) Preparación por clúster con exógenas (estandarización + lags).
#   3) Exportar datasets por clúster: train, val, test.
#   4) Exportar resumen (catálogo de series) y agregación completa.
#
# Input:
#   - data/processed/dataset_modelado_ready.parquet
#
# Outputs:
#   - data/processed/modelado/sarimax/cluster_{id}/train.csv
#   - data/processed/modelado/sarimax/cluster_{id}/val.csv
#   - data/processed/modelado/sarimax/cluster_{id}/test.csv
#   - reports/modelado/sarimax/catalogo_series.csv
#   - data/processed/modelado/sarimax/agg_full.parquet  (agregación clúster–día)
#
# Dependencias:
#   - pandas, numpy, pyarrow (para parquet)
#   - scikit-learn (StandardScaler)
#
# Instalación rápida:
#   pip install pandas numpy pyarrow scikit-learn
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_BASE = PROCESSED_DIR / "modelado" / "sarimax"
REPORTS_DIR = ROOT_DIR / "reports" / "modelado" / "sarimax"
for d in (OUT_BASE, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ==== 1. IMPORTS + LOGGING ====================================================
# ==== bootstrap import path ====
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# ==============================

import argparse
import logging
import pandas as pd
import numpy as np

from scripts.utils.preprocesamiento import prepare_cluster_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def export_cluster_splits(prepared: dict, out_base: Path) -> pd.DataFrame:
    """
    Exporta train/val/test por clúster y devuelve un catálogo resumen.
    """
    rows = []
    for cl, splits in prepared.items():
        cl_dir = out_base / f"cluster_{cl}"
        cl_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "val", "test"):
            df_split = splits[split_name].reset_index()
            out_csv = cl_dir / f"{split_name}.csv"
            df_split.to_csv(out_csv, index=False)

            rows.append({
                "cluster_id": cl,
                "split": split_name,
                "n_rows": len(df_split),
                "date_min": df_split["date"].min(),
                "date_max": df_split["date"].max(),
                "path": str(out_csv.relative_to(ROOT_DIR))
            })

    catalog = pd.DataFrame(rows).sort_values(["cluster_id","split"])
    return catalog

def compute_agg_full(df: pd.DataFrame, exog_vars: list[str]) -> pd.DataFrame:
    """
    Crea y devuelve la agregación clúster–día (suma objetivo, media exógenas).
    """
    agg = (
        df.groupby(["cluster_id","date"])
          .agg(
              sales_quantity=("sales_quantity","sum"),
              **{col:(col,"mean") for col in exog_vars}
          )
          .reset_index()
          .sort_values(["cluster_id","date"])
    )
    return agg

# ==== 3. LÓGICA PRINCIPAL =====================================================
def main(inp: Path, standardize: bool, add_lags: bool, lags: tuple[int, ...]) -> None:
    log.info("Leyendo dataset base: %s", inp)
    df = pd.read_parquet(inp)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["cluster_id","date"])

    exog_vars = [
        "price_factor_effective",
        "m_agosto_nonprice",
        "m_competition",
        "m_inflation",
        "m_promo",
    ]

    # Agregación completa clúster–día (útil para inspección y auditoría)
    agg_full = compute_agg_full(df, exog_vars)
    agg_path = OUT_BASE / "agg_full.parquet"
    agg_full.to_parquet(agg_path, index=False)
    log.info("Agregación clúster–día guardada en: %s", agg_path)

    # Preparación por clúster con util de preprocesamiento
    prepared = prepare_cluster_data(
        df,
        target="sales_quantity",
        exog_vars=exog_vars,
        standardize=standardize,
        add_lags=add_lags,
        lag_days=lags
    )

    # Exportar splits por clúster
    catalog = export_cluster_splits(prepared, OUT_BASE)
    cat_path = REPORTS_DIR / "catalogo_series.csv"
    catalog.to_csv(cat_path, index=False)
    log.info("Catálogo de series escrito en: %s", cat_path)

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preparación de datos para SARIMAX por clúster.")
    p.add_argument("--inp", type=str,
                   default=str(PROCESSED_DIR / "dataset_modelado_ready.parquet"),
                   help="Ruta de entrada (parquet).")
    p.add_argument("--no-standardize", action="store_true",
                   help="Desactiva estandarización de exógenas.")
    p.add_argument("--no-lags", action="store_true",
                   help="Desactiva generación de lags de exógenas.")
    p.add_argument("--lags", type=int, nargs="*", default=[1,7],
                   help="Lista de lags a generar para exógenas.")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(
        inp=Path(args.inp),
        standardize=not args.no_standardize,
        add_lags=not args.no_lags,
        lags=tuple(args.lags),
    )

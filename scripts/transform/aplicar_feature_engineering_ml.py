
# scripts/transform/aplicar_feature_engineering_ml.py
# =============================================================================
# Descripción:
#   Aplica la función `add_time_features` al dataset base para enriquecerlo con
#   variables de tipo temporal (lags, medias móviles, calendario). 
#   Este paso es esencial para preparar los datos de entrada a modelos de ML.
#
# Flujo del script:
#   1. Carga el dataset base `dataset_modelado_ready.parquet`.
#   2. Define las variables exógenas a enriquecer (EXOG).
#   3. Aplica la función `add_time_features` con lags, ventanas móviles y calendario.
#   4. Guarda el dataset enriquecido como `dataset_ml_ready.parquet`.
#
# Entradas:
#   - data/processed/dataset_modelado_ready.parquet
#
# Salidas:
#   - data/processed/dataset_ml_ready.parquet
#
# Dependencias:
#   - scripts/utils/feature_engineering.py (función `add_time_features`)
#   - pandas, pathlib
#
# Ejemplo de ejecución:
#   python scripts/transform/aplicar_feature_engineering_ml.py
#
# Requisitos:
#   pip install pandas pyarrow
# =============================================================================

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import sys

# --- Fix para poder ejecutar el script directamente con "python scripts/..." ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# -----------------------------------------------------------------------------

# Importa la utilidad
from scripts.utils.feature_engineering import add_time_features

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

INP = PROCESSED_DIR / "dataset_modelado_ready.parquet"   # entrada base
OUT = PROCESSED_DIR / "dataset_ml_ready.parquet"         # salida enriquecida

# Variables exógenas sobre las que crear lags/rolling
EXOG_COLS = [
    "price_factor_effective",
    "m_agosto_nonprice",
    "m_competition",
    "m_inflation",
    "m_promo",
]

def run(lags=(1, 7, 14), roll_windows=(7, 28), add_cal=False, drop_na=False):
    assert INP.exists(), f"No existe el parquet de entrada: {INP}"
    df = pd.read_parquet(INP)

    # Recomendado: agrupar por cluster; si quieres más granularidad, añade "product_id"
    df_feat = add_time_features(
        df,
        date_col="date",
        group_cols=("cluster_id",),      # o ("cluster_id", "product_id")
        feature_cols=EXOG_COLS,
        lags=lags,
        roll_windows=roll_windows,
        roll_min_frac=0.6,
        add_calendar=add_cal,
        drop_initial_na=True  # recomendado para ML,
    )

    df_feat.to_parquet(OUT, index=False)
    print(f"[OK] Dataset ML enriquecido guardado en: {OUT}")
    print(f"     Filas: {len(df_feat):,} | Columnas: {df_feat.shape[1]}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generar lags y rolling para ML.")
    p.add_argument("--lags", type=int, nargs="*", default=[1, 7, 14],
                   help="Lista de lags, p.ej.: --lags 1 7 14")
    p.add_argument("--rolling", type=int, nargs="*", default=[7, 28],
                   help="Ventanas de medias móviles, p.ej.: --rolling 7 28")
    p.add_argument("--calendar", action="store_true", help="Añade variables de calendario")
    p.add_argument("--drop-na", action="store_true", help="Eliminar filas iniciales con NaN de lags/rolling")
    args = p.parse_args()

    run(lags=tuple(args.lags), roll_windows=tuple(args.rolling),
        add_cal=args.calendar, drop_na=args.drop_na)

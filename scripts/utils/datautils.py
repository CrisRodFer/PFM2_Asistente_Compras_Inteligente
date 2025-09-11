# scripts/utils/datautils.py
# =============================================================================
# Descripción:
#   Utilidades de carga y particionado temporal del dataset ML.
#   Asegura consistencia del split (train ≤ 2023, val = 2024) y tipado de fecha.
#
# Entradas:
#   - path (str/Path): ruta al parquet enriquecido (dataset_ml_ready.parquet)
#
# Salidas:
#   - DataFrame completo con date en datetime64[ns]
#   - Tuplas (train_df, val_df) para modelado temporal
#
# Dependencias:
#   - pandas, pathlib
#
# Ejemplo de uso:
#   df = load_ml_dataset("data/processed/dataset_ml_ready.parquet")
#   train, val = temporal_split(df)
# =============================================================================

from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_ml_dataset(path) -> pd.DataFrame:
    """
    Carga el parquet con el dataset enriquecido para ML
    y garantiza tipo datetime en la columna 'date'.
    """
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError("El dataset no contiene la columna 'date'.")
    df["date"] = pd.to_datetime(df["date"])
    return df


def temporal_split(df: pd.DataFrame, date_col: str = "date"):
    """
    Divide el dataset en:
      - train: filas con año <= 2023
      - val  : filas con año == 2024
    """
    train = df[df[date_col].dt.year <= 2023].copy()
    val   = df[df[date_col].dt.year == 2024].copy()
    return train, val

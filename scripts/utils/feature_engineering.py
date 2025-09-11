
# =============================================================================
# Script: feature_engineering.py
# Descripción:
#   Aplica ingeniería de variables sobre el dataset de modelado, generando
#   *lags* y otras variables auxiliares para enriquecer la información
#   disponible antes del entrenamiento de modelos de Machine Learning.
#
# Flujo del pipeline:
#   1) Carga dataset base en formato Parquet.
#   2) Genera variables lag (ej. t-1, t-7) para capturar dependencias temporales.
#   3) Añade columnas derivadas asegurando alineación con la variable objetivo.
#   4) Devuelve el dataset enriquecido, listo para validación o modelado.
#
# Input:
#   - data/processed/dataset_modelado_ready.parquet
#
# Output:
#   - (no genera salida directa; devuelve un DataFrame para reutilización).
#
# Dependencias:
#   - pandas
#   - numpy
#
# Instalación rápida:
#   pip install pandas numpy
# =============================================================================

from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    group_cols: Sequence[str] = ("cluster_id",),   # añade "product_id" si quieres granularidad por producto
    feature_cols: Sequence[str],
    lags: Iterable[int] = (1, 7, 14),
    roll_windows: Iterable[int] = (7, 28),
    roll_min_frac: float = 0.6,                   # min_periods = int(window * roll_min_frac)
    add_calendar: bool = False,                   # añade dow, month, week si True
    drop_initial_na: bool = False                 # si True, elimina filas incompletas tras generar features
) -> pd.DataFrame:
    """
    Añade lags y medias móviles para las columnas dadas, agrupando por group_cols y ordenando por date_col.
    Evita leakage usando shift(1) antes de rolling.

    Devuelve un nuevo DataFrame (no modifica el original).
    """
    if not set(feature_cols).issubset(df.columns):
        missing = sorted(set(feature_cols) - set(df.columns))
        raise ValueError(f"Faltan columnas en df: {missing}")

    # Copia y orden temporal/agrupación
    df = df.copy()
    df = df.sort_values(list(group_cols) + [date_col])

    # Garantizar tipo datetime
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])

    # LAGS
    for col in feature_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(list(group_cols), sort=False)[col].shift(lag)

    # ROLLING (media) sin leakage: usamos shift(1) antes del rolling
    for col in feature_cols:
        for win in roll_windows:
            s = (
                df.groupby(list(group_cols), sort=False)[col]
                  .apply(lambda g: g.shift(1).rolling(
                      window=win,
                      min_periods=max(1, int(np.ceil(win * roll_min_frac)))
                  ).mean())
                  .reset_index(level=list(range(len(group_cols))), drop=True)
            )
            df[f"{col}_ma{win}"] = s

    # (Opcional) calendario
    if add_calendar:
        dt = df[date_col].dt
        df["dow"]   = dt.dayofweek.astype("int8")   # 0=Lunes
        df["month"] = dt.month.astype("int8")
        df["week"]  = dt.isocalendar().week.astype("int16")

    # (Opcional) eliminar filas con NaN creados por lags/rolling iniciales
    if drop_initial_na:
        # solo NaN provenientes de nuevas columnas
        new_cols = [c for c in df.columns if any(c.endswith(f"_lag{l}") for l in lags)
                    or any(c.endswith(f"_ma{w}") for w in roll_windows)]
        df = df.dropna(subset=new_cols)

    return df

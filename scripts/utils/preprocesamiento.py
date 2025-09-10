
# =============================================================================
# Script: preprocesamiento.py
# Descripción:
#   Utilidades de preprocesamiento para modelado temporal por clúster.
#   Incluye la función `prepare_cluster_data()` que:
#     - Agrega demanda y exógenas a nivel clúster–día.
#     - Divide en train (2022–2023), validación (2024) y test (2025).
#     - (Opcional) Estandariza exógenas por clúster (z-score).
#     - (Opcional) Genera lags de exógenas (p. ej., t−1 y t−7).
#
# Flujo del pipeline (de la función principal):
#   1) Limpieza básica y orden temporal por clúster/fecha.
#   2) Agregación clúster–día:
#        - sales_quantity: suma
#        - exógenas: media
#   3) Estandarización (opcional) y generación de lags (opcional).
#   4) Split temporal: train (2022–2023), val (2024), test (>=2025).
#
# Input esperado:
#   - DataFrame con columnas: date (datetime), cluster_id, sales_quantity
#     y las exógenas a utilizar (p. ej., price_factor_effective, m_*).
#
# Output de la función:
#   - Dict por clúster con DataFrames "train", "val", "test".
#   - No escribe ficheros (las exportaciones se realizan desde scripts externos).
#
# Dependencias:
#   - pandas
#   - scikit-learn (para StandardScaler)
#
# Instalación rápida:
#   pip install pandas scikit-learn
# =============================================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_cluster_data(df, target="sales_quantity",
                         exog_vars=None,
                         cluster_col="cluster_id",
                         date_col="date",
                         standardize=True,
                         add_lags=True,
                         lag_days=(1,7)):
    """
    Prepara los datos para modelado temporal por clúster:
    - Agrupa por clúster y fecha (suma de target, media de exógenas).
    - Divide en train (2022–2023), validación (2024) y test (2025).
    - Estandariza exógenas por clúster (opcional).
    - Añade lags de exógenas (opcional).
    
    Parámetros
    ----------
    df : pd.DataFrame
        Dataset con columnas de fecha, clúster, target y exógenas.
    target : str
        Nombre de la variable objetivo (ej. 'sales_quantity').
    exog_vars : list
        Lista de columnas exógenas a incluir.
    cluster_col : str
        Columna de clúster.
    date_col : str
        Columna de fechas.
    standardize : bool
        Si True, aplica estandarización z-score a exógenas por clúster.
    add_lags : bool
        Si True, añade lags a exógenas.
    lag_days : tuple
        Días de lag a generar (ej. (1,7)).

    Returns
    -------
    dict
        Diccionario con claves {cluster_id: {"train": df, "val": df, "test": df}}
    """
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([cluster_col, date_col])

    # Agregación por clúster–día
    agg = (
        df.groupby([cluster_col, date_col])
          .agg(
              **{target: (target, "sum")},
              **{col: (col, "mean") for col in exog_vars}
          )
          .reset_index()
    )

    results = {}

    for cl in sorted(agg[cluster_col].unique()):
        dcl = agg[agg[cluster_col] == cl].copy().set_index(date_col)

        # Escalado
        if standardize:
            scaler = StandardScaler()
            dcl[exog_vars] = scaler.fit_transform(dcl[exog_vars].fillna(0.0))

        # Lags
        if add_lags:
            for lag in lag_days:
                for c in exog_vars:
                    dcl[f"{c}_lag{lag}"] = dcl[c].shift(lag)
            dcl = dcl.fillna(0.0)

        # División temporal
        train = dcl[dcl.index.year <= 2023]
        val   = dcl[dcl.index.year == 2024]
        test  = dcl[dcl.index.year >= 2025]

        results[cl] = {"train": train, "val": val, "test": test}

    return results

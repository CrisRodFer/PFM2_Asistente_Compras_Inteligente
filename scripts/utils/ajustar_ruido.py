# =============================================================================
# Script: ajuste_ruido.py
# Descripción:
# Utilidades para medir y ajustar el “ruido” (variabilidad añadida) de una
# serie sintetizada (`Demand_Final_Nois ed`) respecto a la base ajustada
# (`Demand_Final`). Define funciones reutilizables y, de forma opcional,
# una CLI sencilla para operar sobre un parquet.
#
# Flujo del pipeline (típico):
# 1) Cargar dataset procesado (parquet).
# 2) Calcular métricas de ruido por clúster (media Final/Noised y % extra).
# 3) Proponer factores k_c por clúster: k_c = min(1, TGT / Δc).
# 4) Ajustar N^adj = F + (N - F) * k_c y guardar/retornar.
#
# Input (CLI opcional):
#   - data/processed/demanda_all_adjusted.parquet
#
# Output (CLI opcional):
#   - data/processed/demanda_all_adjusted_postnoise.parquet
#
# Dependencias:
#   - pandas
#   - numpy
#
# Instalación rápida:
#   pip install pandas numpy pyarrow
# =============================================================================

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def calcular_factores_kc_por_cluster(
    df: pd.DataFrame,
    col_cluster: str = "Cluster",
    col_final: str = "Demand_Final",
    col_noised: str = "Demand_Final_Noiseds",
    objetivo_ruido: float = 0.22,
    eps: float = 1e-9,
) -> Tuple[pd.DataFrame, Dict[float, float]]:
    """
    Calcula, para cada clúster, el factor k_c que reduce el exceso de ruido
    hacia el objetivo indicado.

    Definiciones:
        ruido_pct = (mean_noised - mean_final) / (mean_final + eps)
        k_c = min(1, objetivo_ruido / (ruido_pct + eps))

    Parámetros
    ----------
    df : DataFrame
        Datos con columnas de clúster, demanda final y demanda noised.
    col_cluster : str
        Nombre de la columna de clúster.
    col_final : str
        Nombre de la columna de la serie ajustada final (sin ruido extra).
    col_noised : str
        Nombre de la columna con ruido añadido.
    objetivo_ruido : float
        Porcentaje objetivo de ruido medio por clúster (p.ej., 0.22 ≈ 22%).
    eps : float
        Pequeña constante numérica para evitar divisiones por cero.

    Devuelve
    --------
    g : DataFrame
        Tabla por clúster con medias y ruido_pct.
    k_map : dict
        Diccionario {cluster -> k_c} con el factor de escala del ruido.
    """
    g = (
        df.groupby(col_cluster)[[col_final, col_noised]]
        .mean(numeric_only=True)
        .rename(columns={col_final: "mean_final", col_noised: "mean_noised"})
    )
    g["ruido_pct"] = (g["mean_noised"] - g["mean_final"]) / (g["mean_final"] + eps)

    k = (objetivo_ruido / (g["ruido_pct"] + eps)).clip(upper=1.0)
    # limpiar inf/NaN y llenar con 1 (no ajustar si no hay ruido medible)
    k = k.replace([np.inf, -np.inf], 1.0).fillna(1.0)

    g["k_c"] = k
    k_map = k.to_dict()
    return g, k_map


def aplicar_ajuste_ruido_por_cluster(
    df: pd.DataFrame,
    k_map: Dict[float, float],
    col_cluster: str = "Cluster",
    col_final: str = "Demand_Final",
    col_noised: str = "Demand_Final_Noiseds",
    col_salida: str = "Demand_Final_Noiseds_adj",
    piso: float | None = 0.0,
) -> pd.DataFrame:
    """
    Aplica el ajuste SOLO al componente de ruido: N_adj = F + (N - F) * k_c.

    Parámetros
    ----------
    df : DataFrame
        Datos de entrada.
    k_map : dict
        Mapa {cluster -> k_c} calculado con `calcular_factores_kc_por_cluster`.
    col_cluster, col_final, col_noised : str
        Nombres de columnas.
    col_salida : str
        Nombre de la columna de salida con la serie noised ajustada.
    piso : float or None
        Si no es None, fuerza un mínimo para la serie ajustada.

    Devuelve
    --------
    DataFrame
        Copia del DataFrame con la nueva columna `col_salida`.
    """
    out = df.copy()
    delta = out[col_noised] - out[col_final]
    k_vec = out[col_cluster].map(k_map).astype(float).fillna(1.0)

    out[col_salida] = out[col_final] + delta * k_vec
    if piso is not None:
        out[col_salida] = out[col_salida].clip(lower=piso)
    return out




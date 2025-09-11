# scripts/utils/metrics.py
# =============================================================================
# Descripción:
#   Métricas personalizadas para evaluación de modelos de regresión en el PFM2.
#   Incluye SMAPE, WAPE y MAE, reutilizables en SARIMAX, ML y otros módulos.
#
# Entradas:
#   - y_true: array-like con valores reales
#   - y_pred: array-like con predicciones de un modelo
#
# Salidas:
#   - float con el valor de la métrica
#
# Dependencias:
#   - numpy
#
# Ejemplo de uso:
#   from scripts.utils.metrics import smape, wape, mae
#   score = smape(y_true, y_pred)
# =============================================================================

from __future__ import annotations
import numpy as np


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (en %).
    Menor es mejor.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return (200.0 / len(y_true)) * np.sum(
        np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )


def wape(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    Weighted Absolute Percentage Error (proporción 0-1).
    Menor es mejor.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + eps)


def mae(y_true, y_pred) -> float:
    """
    Mean Absolute Error.
    Menor es mejor.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_pred - y_true))

# scripts/utils/model_registry.py
# =============================================================================
# Descripción:
#   Registro de modelos soportados (pipelines) y sus grids de hiperparámetros.
#   Separa la definición de modelos de la lógica de entrenamiento/evaluación.
#
# Modelos:
#   - linear  : Regresión Lineal
#   - ridge   : Ridge Regression
#   - lasso   : Lasso Regression
#   - rf      : RandomForestRegressor
#   - xgb     : XGBRegressor (requiere xgboost instalado)
#
# Salidas:
#   - MODELS: dict con {name: {"pipeline_builder": fn, "param_grids": {"small": {...}, "full": {...}}}}
#
# Dependencias:
#   - scikit-learn, xgboost (opcional)
#
# Ejemplo de uso:
#   from scripts.utils.model_registry import MODELS
#   entry = MODELS["ridge"]; pipe = entry["pipeline_builder"](num_cols, cat_cols)
#   grid  = entry["param_grids"]["small"]
# =============================================================================

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def _preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def _pipe_linear(num_cols, cat_cols):
    return Pipeline([
        ("pre", _preprocessor(num_cols, cat_cols)),
        ("mdl", LinearRegression())
    ])


def _pipe_ridge(num_cols, cat_cols):
    return Pipeline([
        ("pre", _preprocessor(num_cols, cat_cols)),
        ("mdl", Ridge(random_state=42))
    ])


def _pipe_lasso(num_cols, cat_cols):
    return Pipeline([
        ("pre", _preprocessor(num_cols, cat_cols)),
        ("mdl", Lasso(random_state=42, max_iter=10000))
    ])


def _pipe_rf(num_cols, cat_cols):
    # RF no necesita escalado; pero mantenemos el mismo ColumnTransformer para one-hot
    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", ct),
        ("mdl", RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        ))
    ])


def _pipe_xgb(num_cols, cat_cols):
    # XGB admite num + one-hot; sin scaler
    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", ct),
        ("mdl", XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        ))
    ]) if _HAS_XGB else None


MODELS = {
    "linear": {
        "pipeline_builder": _pipe_linear,
        "param_grids": {
            "small": {"mdl__fit_intercept": [True, False]},
            "full":  {"mdl__fit_intercept": [True, False]}
        },
    },
    "ridge": {
        "pipeline_builder": _pipe_ridge,
        "param_grids": {
            "small": {"mdl__alpha": [0.1, 1.0, 10.0]},
            "full":  {"mdl__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
    },
    "lasso": {
        "pipeline_builder": _pipe_lasso,
        "param_grids": {
            "small": {"mdl__alpha": [0.001, 0.01, 0.1]},
            "full":  {"mdl__alpha": [0.0005, 0.001, 0.005, 0.01, 0.1, 1.0]},
        },
    },
    "rf": {
        "pipeline_builder": _pipe_rf,
        "param_grids": {
            "small": {"mdl__n_estimators": [300], "mdl__max_depth": [None, 10]},
            "full":  {"mdl__n_estimators": [300, 600], "mdl__max_depth": [None, 8, 12]},
        },
    },
}

if _HAS_XGB:
    MODELS["xgb"] = {
        "pipeline_builder": _pipe_xgb,
        "param_grids": {
            "small": {"mdl__n_estimators": [400, 600], "mdl__max_depth": [4, 6], "mdl__learning_rate": [0.05, 0.1]},
            "full":  {"mdl__n_estimators": [400, 600, 800], "mdl__max_depth": [4, 6, 8],
                      "mdl__learning_rate": [0.03, 0.05, 0.1], "mdl__subsample": [0.8, 1.0]},
        },
    }

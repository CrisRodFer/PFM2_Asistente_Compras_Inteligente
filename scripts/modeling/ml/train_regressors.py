# scripts/modeling/ml/train_regressors.py
# =============================================================================
# Entrenamiento de modelos de regresión por CLÚSTER (ML)
# - Lee el dataset enriquecido para ML (parquet con lags/rolling/calendar).
# - Entrena 4 modelos por cada cluster_id: Linear/RidgeCV, RandomForest,
#   XGBoost y HistGradientBoosting.
# - Evalúa en validación 2024 con MAE, WAPE, sMAPE.
# - Guarda métricas y predicciones por clúster y modelo.
#
# Salidas:
#   outputs/modeling/ml/metrics_val_2024_ml.csv
#   outputs/modeling/ml/preds_val_2024_ml.csv
#   reports/modeling/ml/winners_por_cluster_ml.csv
#
# Requisitos:
#   pip install pandas numpy scikit-learn xgboost pyarrow
#
# Notas:
# - Se entrena por clúster (un modelo por cluster_id).
# - El target es 'sales_quantity'.
# - Se excluyen columnas no-numéricas irrelevantes y 'cluster_id' del set de
#   features porque estamos entrenando por clúster.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

# --------------------------- rutas y logging ----------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data" / "processed"
OUT_DIR = ROOT_DIR / "outputs" / "modeling" / "ml"
REPORTS_DIR = ROOT_DIR / "reports" / "modeling" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ------------------------------- métricas -------------------------------------
def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return (200.0 / len(y_true)) * np.sum(np.abs(y_pred - y_true) / denom)

def wape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + eps)

def mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)

# ----------------------------- modelos base -----------------------------------
def get_models() -> dict[str, object]:
    """Devuelve un diccionario de modelos ML con hiperparámetros razonables."""
    models: dict[str, object] = {
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            tree_method="hist", objective="reg:squarederror",
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=None, learning_rate=0.06, max_iter=400, random_state=42
        ),
    }
    return models

# ----------------------------- utilidades -------------------------------------
EXCLUDE_NON_FEATURES = {
    "date", "sales_quantity", "product_id", "is_outlier",
    "tipo_outlier_year", "decision_outlier_year", "cluster_id",
}

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Selecciona columnas numéricas útiles (excluye target y no-features)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in EXCLUDE_NON_FEATURES]
    return feats

def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide por tiempo: train (<=2023) y val (=2024)."""
    train = df[df["date"].dt.year <= 2023].copy()
    val   = df[df["date"].dt.year == 2024].copy()
    return train, val

# ------------------------------- runner ---------------------------------------
def run(inp_parquet: Path) -> None:
    assert inp_parquet.exists(), f"No encuentro el parquet: {inp_parquet}"
    df = pd.read_parquet(inp_parquet)
    df["date"] = pd.to_datetime(df["date"])

    clusters = sorted(df["cluster_id"].unique().tolist())
    log.info("Clusters detectados: %s", clusters)

    models = get_models()

    metrics_rows: list[dict] = []
    preds_rows: list[pd.DataFrame] = []

    for cl in clusters:
        dcl = df[df["cluster_id"] == cl].copy()
        if dcl.empty:
            continue

        train, val = split_train_val(dcl)
        if train.empty or val.empty:
            log.warning("Cluster %s sin datos suficientes en train o val.", cl)
            continue

        feat_cols = get_feature_cols(dcl)
        X_tr, y_tr = train[feat_cols], train["sales_quantity"]
        X_va, y_va = val[feat_cols],   val["sales_quantity"]

        log.info("Cluster %s | feats=%d | train=%d | val=%d",
                 cl, len(feat_cols), len(train), len(val))

        for name, model in models.items():
            try:
                mdl = model
                mdl.fit(X_tr, y_tr)
                yhat = mdl.predict(X_va)

                row = {
                    "cluster": cl,
                    "model": name,
                    "mae": mae(y_va, yhat),
                    "wape": wape(y_va, yhat),
                    "smape": smape(y_va, yhat),
                }
                metrics_rows.append(row)

                preds_rows.append(pd.DataFrame({
                    "date": val["date"].values,
                    "cluster": cl,
                    "model": name,
                    "y_true": y_va.values,
                    "y_pred": yhat,
                }))
            except Exception as ex:
                log.exception("Fallo entrenando %s en cluster %s: %s", name, cl, ex)

    # ------------------- guardar resultados -------------------
    if not metrics_rows:
        log.error("No se generaron métricas. Revisa los datos/modelos.")
        return

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["cluster", "smape"])
    preds_df = pd.concat(preds_rows, ignore_index=True).sort_values(["cluster", "date", "model"])

    metrics_path = OUT_DIR / "metrics_val_2024_ml.csv"
    preds_path   = OUT_DIR / "preds_val_2024_ml.csv"
    winners_path = REPORTS_DIR / "winners_por_cluster_ml.csv"

    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    winners = metrics_df.loc[metrics_df.groupby("cluster")["smape"].idxmin()].reset_index(drop=True)
    winners.to_csv(winners_path, index=False)

    log.info("Métricas: %s", metrics_path)
    log.info("Predicciones: %s", preds_path)
    log.info("Ganadores por clúster: %s", winners_path)

# -------------------------------- CLI -----------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento ML por clúster (val=2024).")
    p.add_argument(
        "--inp",
        type=str,
        default=str(DATA_DIR / "dataset_ml_ready.parquet"),
        help="Parquet enriquecido con features para ML.",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run(Path(args.inp))

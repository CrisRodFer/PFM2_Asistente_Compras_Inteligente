# =============================================================================
# Script: feature_importance.py
# Descripción:
# Calcula la importancia de variables por clúster para modelos de ML
# (RandomForest y XGBoost) usando el dataset enriquecido preparado para ML.
#
# Flujo del pipeline:
# 1) Lectura del dataset enriquecido (Parquet).
# 2) Selección automática de features (numéricas, excluyendo target/derivados).
# 3) Split temporal coherente (train ≤ 2023, val = 2024).
# 4) Entrenamiento de modelos (RF y/o XGB) por clúster.
# 5) Cálculo de importancias nativas (%).
# 6) Exportación en formato largo, resúmenes globales y barplots (Top-k).
#
# Input:
#   - data/processed/dataset_ml_ready.parquet
#
# Output:
#   - outputs/modeling/ml/feature_importance/fi_long_{MODEL}.csv
#   - outputs/modeling/ml/feature_importance/fi_summary_{MODEL}.csv
#   - outputs/modeling/ml/feature_importance/fi_top{K}_{MODEL}.csv
#   - outputs/modeling/ml/feature_importance/{model}_top{K}_cluster{c}.png
#   - outputs/modeling/ml/feature_importance/{model}_top{K}_summary.png
#
# Dependencias:
#   - pandas
#   - numpy
#   - scikit-learn
#   - xgboost (opcional)
#   - matplotlib
#
# Instalación rápida:
#   pip install pandas numpy scikit-learn xgboost matplotlib
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ==== 0. CONFIG (RUTAS BASE) ==================================================
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT_DIR / "outputs" / "modeling" / "ml" / "feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA = PROCESSED_DIR / "dataset_ml_ready.parquet"
TARGET = "sales_quantity"
DATE_COL = "date"
CLUSTER_COL = "cluster_id"

EXCLUDE_COLS = {
    TARGET, DATE_COL, CLUSTER_COL, "product_id",
    "demand_day_priceadj", "demand_adjust"
}

# ==== 1. IMPORTS + LOGGING ====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

HAS_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False
    warnings.warn("xgboost no está instalado: se omitirá XGBRegressor.")

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def _auto_feature_cols(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in EXCLUDE_COLS]

def _split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(DATE_COL)
    train = df[df[DATE_COL] < "2024-01-01"].copy()
    val = df[(df[DATE_COL] >= "2024-01-01") & (df[DATE_COL] <= "2024-12-31")].copy()
    return train, val

def _fit_rf(X, y, random_state=42) -> RandomForestRegressor:
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                               n_jobs=-1, random_state=random_state)
    rf.fit(X, y)
    return rf

def _fit_xgb(X, y, random_state=42) -> "XGBRegressor":
    xgb = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, objective="reg:squarederror",
        random_state=random_state, n_jobs=-1, tree_method="hist",
    )
    xgb.fit(X, y)
    return xgb

def _importances_rf(model: RandomForestRegressor, feature_names: list[str]) -> pd.Series:
    imp = pd.Series(model.feature_importances_, index=feature_names)
    return (imp / imp.sum() * 100).sort_values(ascending=False)

def _importances_xgb(model: "XGBRegressor", feature_names: list[str]) -> pd.Series:
    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    mapping = {f"f{i}": name for i, name in enumerate(feature_names)}
    s = pd.Series({mapping.get(k, k): v for k, v in gain_dict.items()}, dtype=float)
    s = s.reindex(feature_names).fillna(0.0)
    return (s / s.sum() * 100 if s.sum() > 0 else s).sort_values(ascending=False)

def _plot_topk(imp: pd.Series, topk: int, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    top = imp.head(topk)[::-1]
    plt.figure(figsize=(8, max(3, 0.35 * len(top))))
    plt.barh(top.index, top.values)
    plt.title(title)
    plt.xlabel("Importancia (%)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def _plot_summary(mean_imp: pd.DataFrame, topk: int, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    top = mean_imp.head(topk).iloc[::-1]
    plt.figure(figsize=(8, max(3, 0.35 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.title(title)
    plt.xlabel("Importancia media (%)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

# ==== 3. LÓGICA PRINCIPAL =====================================================
def compute_feature_importance(dataset_path: Path, models: list[str], topk: int, plot: bool) -> None:
    df = pd.read_parquet(dataset_path)
    feature_cols = _auto_feature_cols(df)
    clusters = sorted(pd.unique(df[CLUSTER_COL]))
    log.info("Detectados %d clústeres | %d features", len(clusters), len(feature_cols))

    long_rf, long_xgb = [], []

    for cl in clusters:
        dcl = df[df[CLUSTER_COL] == cl].copy()
        train, _ = _split_train_val(dcl)
        X, y = train[feature_cols].values, train[TARGET].values

        if "rf" in models:
            rf = _fit_rf(X, y)
            imp = _importances_rf(rf, feature_cols)
            long_rf.append(pd.DataFrame({"cluster": cl, "model": "RandomForest",
                                         "feature": imp.index, "importance": imp.values}))
            if plot:
                _plot_topk(imp, topk, f"RF Top {topk} (cluster {cl})",
                           OUT_DIR / f"rf_top{topk}_cluster{cl}.png")

        if "xgb" in models and HAS_XGB:
            xb = _fit_xgb(X, y)
            impx = _importances_xgb(xb, feature_cols)
            long_xgb.append(pd.DataFrame({"cluster": cl, "model": "XGBRegressor",
                                          "feature": impx.index, "importance": impx.values}))
            if plot:
                _plot_topk(impx, topk, f"XGB Top {topk} (cluster {cl})",
                           OUT_DIR / f"xgb_top{topk}_cluster{cl}.png")

    if long_rf:
        df_lr = pd.concat(long_rf, ignore_index=True)
        df_lr.to_csv(OUT_DIR / "fi_long_RF.csv", index=False)
        _save_summary(df_lr, "RF", topk, plot)

    if long_xgb:
        df_lx = pd.concat(long_xgb, ignore_index=True)
        df_lx.to_csv(OUT_DIR / "fi_long_XGB.csv", index=False)
        _save_summary(df_lx, "XGB", topk, plot)

def _save_summary(df_long: pd.DataFrame, tag: str, topk: int, plot: bool) -> None:
    mean_imp = df_long.groupby("feature", as_index=False)["importance"].mean()
    mean_imp = mean_imp.sort_values("importance", ascending=False)
    mean_imp.to_csv(OUT_DIR / f"fi_summary_{tag}.csv", index=False)
    mean_imp.head(topk).to_csv(OUT_DIR / f"fi_top{topk}_{tag}.csv", index=False)
    if plot:
        _plot_summary(mean_imp, topk, f"{tag} Top {topk} (media global)",
                      OUT_DIR / f"{tag.lower()}_top{topk}_summary.png")

# ==== 4. EXPORTACIÓN / I/O OPCIONAL ========================================== #
# (ya integrado en compute + _save_summary)

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Importancia de features por clúster (RF/XGB).")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--models", nargs="+", default=["rf", "xgb"], choices=["rf", "xgb"])
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    compute_feature_importance(Path(args.dataset), args.models, args.topk, args.plot)

if __name__ == "__main__":
    main()

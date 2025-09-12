# =============================================================================
# Script: backtesting.py
# Descripción:
# Backtesting por clúster con dos cortes temporales (rolling-origin reducido):
#   - Split A: Train 2022 → Validación 2023
#   - Split B: Train 2022–2023 → Validación 2024
#
# Modelos evaluados:
#   - Naive (baseline): modo 'last' (constante) o 'seasonal' (periodo s)
#   - Random Forest (RF) como modelo recomendado
#   - SARIMAX (opcional): orden fijo con exógenas acotadas (si statsmodels está disponible)
#
# Flujo del pipeline:
# 1) Carga del dataset enriquecido (parquet)
# 2) Selección de features (numéricas, excl. target/fechas/ids/derivados del target)
# 3) Backtest por clúster en dos splits temporales
# 4) Cálculo de métricas (MAE, WAPE, sMAPE) en validación
# 5) Exportación de métricas por clúster y resumen global
#
# Input:
#   - data/processed/dataset_ml_ready.parquet
#
# Output:
#   - reports/backtests/metrics_all.csv      (tabla larga: model,cluster,split,MAE,WAPE,sMAPE)
#   - reports/backtests/summary/metrics_global.csv (promedios por modelo y split)
#
# Dependencias:
#   - pandas
#   - numpy
#   - scikit-learn
#   - statsmodels (opcional para SARIMAX)
#
# Instalación rápida:
#   pip install pandas numpy scikit-learn statsmodels
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]  # ajustado a tu arq. de proyecto
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "backtests"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# SARIMAX (opcional)
HAS_STATS = True
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    HAS_STATS = False
    warnings.warn("statsmodels no está instalado: se omitirá SARIMAX.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# Campos del dataset
TARGET = "sales_quantity"
DATE_COL = "date"
CLUSTER_COL = "cluster_id"

# Exclusiones (no pueden ser features)
EXCLUDE_COLS = {
    TARGET, DATE_COL, CLUSTER_COL, "product_id",
    "demand_day_priceadj", "demand_adjust"
}

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def auto_feature_cols(df: pd.DataFrame) -> list[str]:
    """Devuelve numéricas válidas como features (excluyendo target/fecha/ids/derivados)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat = [c for c in num_cols if c not in EXCLUDE_COLS]
    if not feat:
        raise ValueError("No se encontraron columnas numéricas para usar como features.")
    return feat

def pick_exog_for_sarimax(df: pd.DataFrame, feature_cols: list[str], max_vars: int = 8) -> list[str]:
    """
    Selección conservadora de exógenas para SARIMAX (opcional, acotada):
    Preferimos señales de Precio/Promoción/Competencia si existen.
    """
    order = []
    # Prioridades por patrón
    prio = [
        "price_factor_effective", "price_virtual", "price_", "precio_",
        "promo", "competition", "nonprice", "inflation", "agosto"
    ]
    used = set()
    for pat in prio:
        for c in feature_cols:
            if c in used: 
                continue
            if pat in c:
                order.append(c); used.add(c)
            if len(order) >= max_vars:
                return order
    # Rellenar con otras si queda hueco
    for c in feature_cols:
        if c not in used:
            order.append(c)
            if len(order) >= max_vars:
                break
    return order

def split_train_val(df: pd.DataFrame, train_end: str, val_year: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal: train ≤ train_end ; val = val_year completo."""
    df = df.sort_values(DATE_COL)
    train = df[df[DATE_COL] <= train_end].copy()
    val = df[(df[DATE_COL] >= f"{val_year}-01-01") & (df[DATE_COL] <= f"{val_year}-12-31")].copy()
    return train, val

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred))) if y_true.size else np.nan
    # WAPE = sum(|e|)/sum(|y|) * 100
    denom = np.sum(np.abs(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom != 0 else np.nan
    # sMAPE robusto
    denom_sm = (np.abs(y_true) + np.abs(y_pred))
    mask = denom_sm != 0
    smape = float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom_sm[mask]) * 100) if mask.any() else np.nan
    return {"MAE": mae, "WAPE_%": wape, "sMAPE_%": smape}

# ==== 3. MODELOS ==============================================================

def naive_predict(train_df: pd.DataFrame, val_df: pd.DataFrame, mode: str = "last", seasonal_period: int = 7) -> np.ndarray:
    """
    Naive sin fugas:
    - last: usa el último valor de train como nivel constante para toda la validación.
    - seasonal: usa el valor de train desplazado 'seasonal_period' días atrás;
                si no hay matching en train, cae a 'last'.
    """
    if val_df.empty or train_df.empty:
        return np.array([])
    y_tr = train_df[TARGET].values
    if mode == "seasonal":
        # Mapear por fecha: y_hat[t] = y[t - s] usando SOLO train
        last_level = y_tr[-1]
        tr = train_df.set_index(DATE_COL)[TARGET]
        preds = []
        for d in val_df[DATE_COL]:
            prev = d - pd.Timedelta(days=seasonal_period)
            pred = tr.get(prev, np.nan)
            if np.isnan(pred):
                pred = last_level
            preds.append(pred)
        return np.asarray(preds, dtype=float)
    else:
        # last (constante del último día de train)
        return np.full(shape=len(val_df), fill_value=y_tr[-1], dtype=float)

def fit_predict_rf(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if val_df.empty or train_df.empty:
        return np.array([])
    X_tr, y_tr = train_df[feature_cols].values, train_df[TARGET].values
    X_va = val_df[feature_cols].values
    rf = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42
    )
    rf.fit(X_tr, y_tr)
    return rf.predict(X_va)

def fit_predict_sarimax(train_df: pd.DataFrame, val_df: pd.DataFrame, exog_cols: list[str],
                        order=(1,0,1), seasonal_order=(1,1,1,7)) -> np.ndarray:
    """SARIMAX robusto: alinea y limpia NaNs en y/exog, y usa forecast(steps)."""
    if not HAS_STATS or val_df.empty or train_df.empty:
        return np.array([])

    # --- Construir series e exógenas con índice de fecha (solo para alinear)
    tr_idx = pd.to_datetime(train_df[DATE_COL])
    va_idx = pd.to_datetime(val_df[DATE_COL])

    y_tr = pd.Series(train_df[TARGET].values, index=tr_idx, dtype=float)
    exog_tr = train_df[exog_cols].astype(float)
    exog_tr.index = tr_idx

    exog_va = val_df[exog_cols].astype(float)
    exog_va.index = va_idx

    # --- Limpieza/alineación:
    # 1) Si target train tiene NaN, elimino esas filas en y y exog
    mask_tr = ~y_tr.isna()
    y_tr = y_tr[mask_tr]
    exog_tr = exog_tr.loc[mask_tr]

    # 2) Relleno posibles NaNs restantes en exógenas (ffill/bfill)
    exog_tr = exog_tr.ffill().bfill()
    exog_va = exog_va.ffill().bfill()

    # 3) Verificación de shapes coherentes
    if len(exog_tr) != len(y_tr):
        # recorto por intersección de índices
        common = exog_tr.index.intersection(y_tr.index)
        y_tr = y_tr.loc[common]
        exog_tr = exog_tr.loc[common]

    # 4) Asegurar mismas columnas en train/val
    exog_va = exog_va.reindex(columns=exog_tr.columns).ffill().bfill()

    try:
        model = SARIMAX(
            y_tr,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog_tr,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)

        # Forecast por pasos usando EXOG de validación
        steps = len(val_df)
        fc = res.forecast(steps=steps, exog=exog_va.iloc[:steps])
        return np.asarray(fc, dtype=float)

    except Exception as e:
        log.warning("SARIMAX fallo (%s). Se devuelve vector vacío.", e)
        return np.array([])

# ==== 4. LÓGICA PRINCIPAL =====================================================
def run_backtest(
    dataset_path: Path,
    models: list[str],
    naive_mode: str = "last",
    seasonal_period: int = 7,
    sarimax_order: tuple = (1,0,1),
    sarimax_seasonal: tuple = (1,1,1,7),
    out_dir: Path = REPORTS_DIR,
) -> None:
    ensure_dirs(out_dir, out_dir / "summary")
    df = pd.read_parquet(dataset_path)
    # Checks
    for col in (DATE_COL, CLUSTER_COL, TARGET):
        if col not in df.columns:
            raise ValueError(f"Dataset incompleto: falta {col}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    feature_cols = auto_feature_cols(df)
    clusters = sorted(df[CLUSTER_COL].unique())

    # Exógenas para SARIMAX (acotadas)
    sarimax_exog = pick_exog_for_sarimax(df, feature_cols, max_vars=8)

    # Splits
    splits = {
        "A": {"train_end": "2022-12-31", "val_year": "2023"},
        "B": {"train_end": "2023-12-31", "val_year": "2024"},
    }

    rows = []  # acumulamos métricas (long format)

    for cl in clusters:
        dcl = df[df[CLUSTER_COL] == cl].copy()
        dcl = dcl.sort_values(DATE_COL).reset_index(drop=True)
        dcl.index = np.arange(len(dcl))  # para SARIMAX (índice entero consecutivo)

        for split_name, cfg in splits.items():
            train, val = split_train_val(dcl, cfg["train_end"], cfg["val_year"])
            if train.empty or val.empty:
                log.info("Cluster %s split %s sin datos de train/val. Saltando.", cl, split_name)
                continue

            # --- Naive
            if "naive" in models:
                yhat = naive_predict(train, val, mode=naive_mode, seasonal_period=seasonal_period)
                m = metrics(val[TARGET].values, yhat)
                rows.append({"model":"naive", "cluster":cl, "split":split_name, **m})

            # --- Random Forest
            if "rf" in models:
                yhat = fit_predict_rf(train, val, feature_cols)
                m = metrics(val[TARGET].values, yhat)
                rows.append({"model":"rf", "cluster":cl, "split":split_name, **m})

            # --- SARIMAX (opcional)
            if "sarimax" in models and HAS_STATS:
                yhat = fit_predict_sarimax(train, val, exog_cols=sarimax_exog, order=sarimax_order, seasonal_order=sarimax_seasonal)
                if yhat.size:
                    m = metrics(val[TARGET].values, yhat)
                    rows.append({"model":"sarimax", "cluster":cl, "split":split_name, **m})
                else:
                    log.info("Cluster %s split %s: SARIMAX sin predicción (omitido).", cl, split_name)

    # --- Guardar métricas (long) y resumen
    if not rows:
        log.warning("No se generaron métricas. Revisa dataset/splits.")
        return

    metrics_all = pd.DataFrame(rows)
    ensure_dirs(out_dir, out_dir / "summary")
    out_all = out_dir / "metrics_all.csv"
    metrics_all.to_csv(out_all, index=False)
    log.info("Métricas escritas en: %s", out_all)

    summary = (
        metrics_all
        .groupby(["model", "split"], as_index=False)[["MAE","WAPE_%","sMAPE_%"]]
        .mean()
        .sort_values(["split","WAPE_%"])
    )
    summary_out = out_dir / "summary" / "metrics_global.csv"
    summary.to_csv(summary_out, index=False)
    log.info("Resumen global escrito en: %s", summary_out)

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtesting por clúster (dos splits A/B).")
    p.add_argument("--dataset", type=str, default=str(PROCESSED_DIR / "dataset_ml_ready.parquet"),
                   help="Ruta al dataset parquet.")
    p.add_argument("--models", nargs="+", default=["naive","rf","sarimax"],
                   choices=["naive","rf","sarimax"],
                   help="Modelos a evaluar.")
    p.add_argument("--naive-mode", type=str, default="last", choices=["last","seasonal"],
                   help="Modo del Naive baseline.")
    p.add_argument("--seasonal-period", type=int, default=7,
                   help="Periodo estacional para naive seasonal y SARIMAX (s).")
    p.add_argument("--sarimax-order", type=int, nargs=3, default=(1,0,1),
                   metavar=("p","d","q"), help="Orden (p,d,q) para SARIMAX.")
    p.add_argument("--sarimax-seasonal", type=int, nargs=4, default=(1,1,1,7),
                   metavar=("P","D","Q","s"), help="Orden estacional (P,D,Q,s).")
    p.add_argument("--outdir", type=str, default=str(REPORTS_DIR),
                   help="Carpeta de salida para métricas.")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    if "sarimax" in args.models and not HAS_STATS:
        warnings.warn("Se pidió SARIMAX pero statsmodels no está disponible. Se omitirá.")
        args.models = [m for m in args.models if m != "sarimax"]
    try:
        run_backtest(
            dataset_path=Path(args.dataset),
            models=args.models,
            naive_mode=args.naive_mode,
            seasonal_period=int(args.seasonal_period),
            sarimax_order=tuple(args.sarimax_order),
            sarimax_seasonal=tuple(args.sarimax_seasonal),
            out_dir=Path(args.outdir),
        )
    except Exception as e:
        log.exception("Fallo en el backtesting: %s", e)
        raise

if __name__ == "__main__":
    main()

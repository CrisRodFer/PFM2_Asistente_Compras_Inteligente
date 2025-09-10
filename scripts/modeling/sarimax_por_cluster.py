# scripts/modelado/sarimax_por_cluster.py
# =============================================================================
# Descripción:
#   Entrena SARIMAX por clúster con exógenas (precio + factores externos),
#   compara contra baselines y guarda métricas/predicciones.
#
# Flujo:
#   0) CONFIG RUTAS
#   1) IMPORTS + LOGGING
#   2) UTILIDADES (métricas, split, creación de exógenas, grid SARIMAX)
#   3) LÓGICA PRINCIPAL (loop por clúster)
#   4) EXPORTACIÓN (csv con métricas, predicciones y modelo_ganador_por_cluster)
#   5) CLI / MAIN
#
# Input:
#   data/processed/dataset_modelado_ready.parquet
# Output:
#   outputs/modelado/sarimax/*.csv
#   reports/modelado/tabla_modelo_ganador_por_cluster.csv
#
# Dependencias:
#   pip install pandas numpy statsmodels pyarrow
# =============================================================================

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

# 0. CONFIG
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "modelado" / "sarimax"
REPORTS_DIR = ROOT_DIR / "reports" / "modelado"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. IMPORTS + LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 2. UTILIDADES
def smape(y_true, y_pred, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return (200.0/len(y_true)) * np.sum(np.abs(y_pred - y_true) / denom)

def wape(y_true, y_pred, eps=1e-8):
    return np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + eps)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def seasonal_naive_forecast(y_train, y_val, season=365):
    # replica el valor de hace 'season' pasos; si no alcanza, usa naive simple
    y_hat = np.zeros_like(y_val, dtype=float)
    for i in range(len(y_val)):
        idx = len(y_train) + i - season
        if idx >= 0:
            y_hat[i] = y_train[idx]
        else:
            y_hat[i] = y_train[-1]
    return y_hat

def holt_winters_baseline_notebook_like(y_train, len_val):
    # placeholder sencillo (se asume que ya tienes métricas de ETS en notebook)
    # aquí devolvemos naive estacional como proxy si no quieres re-entrenar ETS en script
    return seasonal_naive_forecast(y_train, np.zeros(len_val), season=365)

def time_split(df, date_col="date"):
    df = df.sort_values(date_col)
    train = df[(df[date_col].dt.year <= 2023)]
    val   = df[(df[date_col].dt.year == 2024)]
    return train, val

def build_exog(df, use_lags=True, lag_days=(1,7)):
    exog_cols = ["price_factor_effective","m_agosto_nonprice","m_competition","m_inflation","m_promo"]
    exog = df[exog_cols].copy()
    if use_lags:
        for lag in lag_days:
            for c in exog_cols:
                exog[f"{c}_lag{lag}"] = exog[c].shift(lag)
    # estandarizar por serie
    scaler = StandardScaler()
    exog_scaled = pd.DataFrame(
        scaler.fit_transform(exog.fillna(method="ffill").fillna(0.0)),
        index=exog.index,
        columns=exog.columns
    )
    return exog_scaled

def fit_sarimax_grid(y_train, y_val, exog_train, exog_val, seasonal_period=365,
                     orders=((1,0,1),(1,1,1),(2,0,1)),
                     sorders=((0,1,1),(1,1,0),(1,1,1))):
    best = {"smape": np.inf}
    for (p,d,q) in orders:
        for (P,D,Q) in sorders:
            try:
                model = SARIMAX(
                    y_train, exog=exog_train,
                    order=(p,d,q), seasonal_order=(P,D,Q,seasonal_period),
                    enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False)
                pred = model.predict(start=y_train.index[-1]+1, end=y_val.index[-1], exog=exog_val)
                cur = {
                    "order": (p,d,q),
                    "sorder": (P,D,Q),
                    "mae": mae(y_val.values, pred.values),
                    "wape": wape(y_val.values, pred.values),
                    "smape": smape(y_val.values, pred.values),
                    "model": model,
                    "pred_val": pred
                }
                if cur["smape"] < best["smape"]:
                    best = cur
            except Exception as e:
                logging.debug(f"Falló {(p,d,q)}x{(P,D,Q)}: {e}")
                continue
    return best

# 3. LÓGICA PRINCIPAL
def run(train_exog_lags=True):
    path = PROCESSED_DIR / "dataset_modelado_ready.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["cluster_id","date"])

    results = []
    preds_2024 = []
    winners = []

    for cl in sorted(df["cluster_id"].unique()):
        logging.info(f"==> Cluster {cl}")
        dcl = df[df["cluster_id"]==cl].copy()

        # Agregación por clúster-día (media de exógenas; suma de demanda)
        agg = (dcl
               .groupby("date")
               .agg(
                   sales_quantity=("sales_quantity","sum"),
                   price_factor_effective=("price_factor_effective","mean"),
                   m_agosto_nonprice=("m_agosto_nonprice","mean"),
                   m_competition=("m_competition","mean"),
                   m_inflation=("m_inflation","mean"),
                   m_promo=("m_promo","mean"),
               )
               .sort_index())

        # Split temporal
        train, val = time_split(agg, "date")
        y_train = train["sales_quantity"]
        y_val   = val["sales_quantity"]

        # Exógenas
        exog_all = build_exog(agg, use_lags=train_exog_lags)
        exog_train = exog_all.loc[y_train.index]
        exog_val   = exog_all.loc[y_val.index]

        # Baselines
        sn_val = seasonal_naive_forecast(y_train.values, y_val.values, season=365)
        sn_metrics = {
            "cluster": cl, "model":"SeasonalNaive",
            "mae": mae(y_val.values, sn_val),
            "wape": wape(y_val.values, sn_val),
            "smape": smape(y_val.values, sn_val)
        }

        # Holt-Winters (proxy o ya calculado en notebook)
        hw_val = holt_winters_baseline_notebook_like(y_train.values, len(y_val))
        hw_metrics = {
            "cluster": cl, "model":"HoltWinters_proxy",
            "mae": mae(y_val.values, hw_val),
            "wape": wape(y_val.values, hw_val),
            "smape": smape(y_val.values, hw_val)
        }

        # SARIMAX (grid corto)
        best = fit_sarimax_grid(y_train, y_val, exog_train, exog_val)
        sar_metrics = {
            "cluster": cl, "model": f"SARIMAX{best['order']}x{best['sorder']}",
            "mae": best["mae"], "wape": best["wape"], "smape": best["smape"]
        }

        # Reunir métricas
        results.extend([sn_metrics, hw_metrics, sar_metrics])

        # Ganador por clúster
        cmp_df = pd.DataFrame([sn_metrics, hw_metrics, sar_metrics]).sort_values("smape")
        winner = cmp_df.iloc[0].to_dict()
        winners.append(winner)

        # Guardar predicciones de validación 2024
        preds_2024.append(
            pd.DataFrame({
                "date": y_val.index,
                "cluster_id": cl,
                "y_true": y_val.values,
                "yhat_sarimax": best["pred_val"].values,
                "yhat_seasonal_naive": sn_val,
                "yhat_holtwinters_proxy": hw_val,
            })
        )

        logging.info(f"Cluster {cl} ganador: {winner['model']} (sMAPE={winner['smape']:.3f})")

    # 4. EXPORTACIÓN
    metrics_df = pd.DataFrame(results)
    preds_val_df = pd.concat(preds_2024).sort_values(["cluster_id","date"])
    winners_df = pd.DataFrame(winners)

    metrics_df.to_csv(OUTPUTS_DIR / "metrics_val_2024_sarimax_baselines.csv", index=False)
    preds_val_df.to_csv(OUTPUTS_DIR / "preds_val_2024_por_cluster.csv", index=False)
    winners_df.to_csv(REPORTS_DIR / "modelo_ganador_por_cluster.csv", index=False)

# 5. CLI / MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_lags", action="store_true", help="No usar lags en exógenas")
    args = parser.parse_args()
    run(train_exog_lags=not args.no_lags)

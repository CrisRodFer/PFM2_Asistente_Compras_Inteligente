# scripts/modeling/sarimax_por_cluster.py
# =============================================================================
# Descripción:
#   Entrena SARIMAX por clúster con exógenas (precio + factores externos),
#   compara contra baselines y guarda métricas/predicciones (validación 2024).
#
# Entradas (dos modos):
#   A) Splits preparados en:
#        data/processed/modelado/sarimax/cluster_{id}/{train,val}.csv
#   B) --from-parquet: agrega desde el parquet base (inp) clúster–día
#
# Salidas:
#   outputs/modeling/sarimax/metrics_val_2024.csv
#   outputs/modeling/sarimax/preds_val_2024.csv
#   reports/modeling/modelo_ganador_por_cluster.csv
#
# Dependencias:
#   pip install pandas numpy statsmodels scikit-learn pyarrow
# =============================================================================

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler

# ---------------------------- Config & rutas ---------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# ⚠️ Ruta corregida: "modelado" en lugar de "modeling"
BASE_PREPARED_DIR = PROCESSED_DIR / "modelado" / "sarimax"

OUT_DIR = ROOT_DIR / "outputs" / "modeling" / "sarimax"
REPORTS_DIR = ROOT_DIR / "reports" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(Path(__file__).stem)

EXOG_BASE = [
    "price_factor_effective",
    "m_agosto_nonprice",
    "m_competition",
    "m_inflation",
    "m_promo",
]

# ------------------------------- Métricas ------------------------------------
def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return (200.0/len(y_true)) * np.sum(np.abs(y_pred - y_true) /
                                        (np.abs(y_true) + np.abs(y_pred) + eps))

def wape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + eps)

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_pred - y_true))

# ---------------------------- Utilidades modelo ------------------------------
def seasonal_naive_forecast(y_train: np.ndarray, y_val_len: int, season: int) -> np.ndarray:
    """Repite el último bloque estacional de y_train hasta cubrir la validación."""
    last_season = np.asarray(y_train[-season:], dtype=float)
    reps = int(np.ceil(y_val_len / float(season)))
    return np.tile(last_season, reps)[:y_val_len]

def build_exog_with_lags(df_exog: pd.DataFrame,
                         use_lags: bool,
                         lag_days: tuple[int, ...]) -> pd.DataFrame:
    exog = df_exog.copy()
    if use_lags:
        for lag in lag_days:
            for c in df_exog.columns:
                exog[f"{c}_lag{lag}"] = df_exog[c].shift(lag)
    exog = exog.ffill().fillna(0.0)  # evita FutureWarning
    scaler = StandardScaler()
    exog_scaled = pd.DataFrame(scaler.fit_transform(exog), index=exog.index, columns=exog.columns)
    return exog_scaled

def fit_sarimax_grid(y_train: pd.Series, y_val: pd.Series,
                     exog_train: pd.DataFrame, exog_val: pd.DataFrame,
                     seasonal_period: int,
                     orders=((1,0,1),(1,1,1),(0,1,1)),
                     sorders=((0,1,1),(1,1,0),(1,1,1))) -> dict:
    """Devuelve dict con 'order','sorder','mae','wape','smape','pred_val','model'
       o bien {'failed': True} si todo falla (incluida la variante sin exógenas)."""
    best = {"smape": np.inf}

    # 1) Intento con exógenas
    for (p,d,q) in orders:
        for (P,D,Q) in sorders:
            try:
                model = SARIMAX(y_train, exog=exog_train,
                                order=(p,d,q), seasonal_order=(P,D,Q, seasonal_period),
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                pred = model.predict(start=y_val.index[0], end=y_val.index[-1], exog=exog_val)
                cur = {"order":(p,d,q),"sorder":(P,D,Q),
                       "mae":mae(y_val,pred),"wape":wape(y_val,pred),"smape":smape(y_val,pred),
                       "pred_val":pred,"model":model}
                if cur["smape"] < best["smape"]:
                    best = cur
            except Exception:
                continue
    if "order" in best:
        return best

    # 2) Fallback: SARIMA sin exógenas
    for (p,d,q) in orders:
        for (P,D,Q) in sorders:
            try:
                model = SARIMAX(y_train,
                                order=(p,d,q), seasonal_order=(P,D,Q, seasonal_period),
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                pred = model.predict(start=y_val.index[0], end=y_val.index[-1])
                cur = {"order":(p,d,q),"sorder":(P,D,Q),
                       "mae":mae(y_val,pred),"wape":wape(y_val,pred),"smape":smape(y_val,pred),
                       "pred_val":pred,"model":model,"no_exog":True}
                if cur["smape"] < best.get("smape", np.inf):
                    best = cur
            except Exception:
                continue
    if "order" in best:
        return best

    return {"failed": True}

def aggregate_from_parquet(df: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    dcl = df[df["cluster_id"] == cluster_id].copy()
    agg = (dcl.groupby("date")
           .agg(sales_quantity=("sales_quantity","sum"),
                **{c:(c,"mean") for c in EXOG_BASE})
           .sort_index())
    return agg

# --------------------------------- Runner ------------------------------------
def run(inp: Path,
        seasonal: int = 365,
        use_prepared: bool | None = None,   # None => autodetect
        use_lags: bool = True,
        lag_days: tuple[int, ...] = (1,7)) -> None:

    prepared_exists = any(BASE_PREPARED_DIR.glob("cluster_*"))

    # Autodetección si no se indicó explícitamente
    if use_prepared is None:
        use_prepared = prepared_exists

    log.info("Fuente de datos: %s",
             "splits preparados" if use_prepared else "parquet (agregación en vuelo)")
    log.info("Inicio SARIMAX | seasonal=%s | lags=%s", seasonal, (lag_days if use_lags else "no"))

    prepared = {}
    if use_prepared:
        clusters = sorted(int(p.name.split("_")[-1]) for p in BASE_PREPARED_DIR.glob("cluster_*") if p.is_dir())
        if not clusters:
            raise FileNotFoundError("No hay splits preparados en data/processed/modelado/sarimax/")
        for cl in clusters:
            cl_dir = BASE_PREPARED_DIR / f"cluster_{cl}"
            tr = pd.read_csv(cl_dir / "train.csv", parse_dates=["date"]).set_index("date")
            va = pd.read_csv(cl_dir / "val.csv",   parse_dates=["date"]).set_index("date")
            prepared[cl] = (tr, va)
        df_base = None
    else:
        assert inp.exists(), f"No encuentro el parquet base: {inp}"
        df_base = pd.read_parquet(inp)
        df_base["date"] = pd.to_datetime(df_base["date"])
        df_base = df_base.sort_values(["cluster_id","date"])
        clusters = sorted(df_base["cluster_id"].unique().tolist())

    results: list[dict] = []
    preds_val_list: list[pd.DataFrame] = []

    for cl in clusters:
        log.info("==> Cluster %s", cl)

        if use_prepared:
            train, val = prepared[cl]
        else:
            agg = aggregate_from_parquet(df_base, cl)
            train = agg[agg.index.year <= 2023]
            val   = agg[agg.index.year == 2024]

        # target
        y_train = train["sales_quantity"].asfreq('D')
        y_val   = val["sales_quantity"].asfreq('D')

        # exógenas
        exog_train = build_exog_with_lags(train[EXOG_BASE], use_lags, lag_days).asfreq('D').loc[y_train.index]
        exog_val   = build_exog_with_lags(val[EXOG_BASE],   use_lags, lag_days).asfreq('D').loc[y_val.index]

        # Baselines
        yhat_sn = seasonal_naive_forecast(y_train.values, len(y_val), season=seasonal)
        try:
            hw = ExponentialSmoothing(
                y_train, trend="add", seasonal="add",
                seasonal_periods=seasonal, initialization_method="estimated"
            ).fit(optimized=True)
            yhat_hw = hw.forecast(len(y_val)).values
        except Exception:
            yhat_hw = yhat_sn

        # SARIMAX (grid con fallbacks)
        best = fit_sarimax_grid(y_train, y_val, exog_train, exog_val, seasonal_period=seasonal)

        if best.get("failed", False):
            model_name = "SARIMAX_failed→SeasonalNaive"
            yhat_sarimax = yhat_sn
            sarimax_mae   = mae(y_val, yhat_sarimax)
            sarimax_wape  = wape(y_val, yhat_sarimax)
            sarimax_smape = smape(y_val, yhat_sarimax)
        else:
            model_name = f"SARIMAX{best['order']}x{best['sorder']}" + ("_noExog" if best.get("no_exog") else "")
            yhat_sarimax = np.asarray(best["pred_val"])
            sarimax_mae, sarimax_wape, sarimax_smape = best["mae"], best["wape"], best["smape"]

        rows = [
            {"cluster": cl, "model": "SeasonalNaive",
             "mae": mae(y_val,yhat_sn),"wape":wape(y_val,yhat_sn),"smape":smape(y_val,yhat_sn)},
            {"cluster": cl, "model": "HoltWinters",
             "mae": mae(y_val,yhat_hw),"wape":wape(y_val,yhat_hw),"smape":smape(y_val,yhat_hw)},
            {"cluster": cl, "model": model_name,
             "mae": sarimax_mae,"wape": sarimax_wape,"smape": sarimax_smape},
        ]
        results.extend(rows)

        preds_val_list.append(pd.DataFrame({
            "date": y_val.index,"cluster": cl,"y_true": y_val.values,
            "yhat_sn": yhat_sn,"yhat_hw": yhat_hw,"yhat_sarimax": yhat_sarimax
        }))

        # -------- Guardado incremental por clúster --------
        metrics_partial = pd.DataFrame(results).sort_values(["cluster","smape"])
        metrics_partial.to_csv(OUT_DIR / "metrics_val_2024_partial.csv", index=False)

        preds_partial = pd.concat(preds_val_list).sort_values(["cluster","date"])
        preds_partial.to_csv(OUT_DIR / "preds_val_2024_partial.csv", index=False)

        log.info("Cluster %s ganador provisional (sMAPE): %.3f",
                 cl, min(r["smape"] for r in rows))

    # ---------------------------- Export final ----------------------------
    metrics_df = pd.DataFrame(results).sort_values(["cluster","smape"])
    winners_df = metrics_df.loc[metrics_df.groupby("cluster")["smape"].idxmin()].reset_index(drop=True)
    preds_val_df = pd.concat(preds_val_list).sort_values(["cluster","date"])

    metrics_df.to_csv(OUT_DIR / "metrics_val_2024.csv", index=False)
    preds_val_df.to_csv(OUT_DIR / "preds_val_2024.csv", index=False)
    winners_df.to_csv(REPORTS_DIR / "modelo_ganador_por_cluster.csv", index=False)

    log.info("Métricas guardadas en: %s", OUT_DIR / "metrics_val_2024.csv")
    log.info("Predicciones val guardadas en: %s", OUT_DIR / "preds_val_2024.csv")
    log.info("Ganadores por clúster guardados en: %s", REPORTS_DIR / "modelo_ganador_por_cluster.csv")

# --------------------------------- CLI ---------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento SARIMAX por clúster con exógenas.")
    p.add_argument("--inp", type=str,
                   default=str(PROCESSED_DIR / "dataset_modelado_ready.parquet"),
                   help="Ruta del parquet base (usado solo si --from-parquet).")
    p.add_argument("--from-parquet", action="store_true",
                   help="Fuerza leer/agregar desde parquet (más lento). Por defecto usa splits preparados si existen.")
    p.add_argument("--seasonal", type=int, default=365, help="Periodo estacional (por defecto 365).")
    p.add_argument("--no-lags", action="store_true", help="Desactiva lags en exógenas.")
    p.add_argument("--lags", type=int, nargs="*", default=[1,7], help="Lista de lags para exógenas (por defecto 1 7).")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    # Autodetección por defecto; si se pasa --from-parquet, usamos parquet
    use_prepared = (None if not args.from_parquet else False)

    run(inp=Path(args.inp),
        seasonal=args.seasonal,
        use_prepared=use_prepared,
        use_lags=not args.no_lags,
        lag_days=tuple(args.lags))

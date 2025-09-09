# =============================================================================
# Script: holt_winters_ets.py
# Descripción:
# Implementa el baseline Holt-Winters (ETS: Error, Trend, Seasonality)
# para predicción de la demanda diaria por clúster en el proyecto PFM2.
# Este baseline combina nivel, tendencia y estacionalidad, ponderando más
# las observaciones recientes para capturar cambios graduales.
#
# Flujo del pipeline:
# 1) Lectura del dataset procesado de modelado
#    (data/processed/dataset_modelado_ready.parquet).
# 2) Validación de columnas mínimas: date, sales_quantity, cluster_id (o alias).
# 3) Agregado de demanda a nivel (date, cluster_id).
# 4) Ajuste de un modelo ETS (Holt-Winters) por clúster usando 2022–2023.
# 5) Generación de predicciones:
#    - Validación (2024): con métricas (MAE, WAPE, sMAPE).
#    - Test (2025): con o sin métricas (según disponibilidad de y_true).
# 6) Exportación de resultados (predicciones y métricas) a disco.
#
# Input:
#   - data/processed/dataset_modelado_ready.parquet
#     Columnas requeridas:
#       • date (datetime64)
#       • sales_quantity (float/int)
#       • cluster_id (categoría/str) o alias equivalente
#
# Output:
#   - data/processed/preds/baselines/holt_winters/preds_val.parquet
#   - data/processed/preds/baselines/holt_winters/preds_test.parquet
#   - reports/baselines/holt_winters/metrics_validation.csv
#   - reports/baselines/holt_winters/metrics_test.csv (si existe y_true en 2025)
#
# Dependencias:
#   - pandas
#   - numpy
#   - statsmodels
#
# Instalación rápida:
#   pip install pandas numpy statsmodels
# =============================================================================

from pathlib import Path
import argparse
import logging
import sys
import time
import numpy as np
import pandas as pd
import warnings

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except Exception as e:
    raise ImportError("Falta 'statsmodels'. Instálalo con: pip install statsmodels") from e


# ---------------- Rutas base del proyecto (robustas para script y notebook) ----------------

def _guess_root_from_cwd() -> Path:
    """Sube por los padres desde el CWD hasta encontrar 'data/processed'. Si no, devuelve CWD."""
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "data" / "processed").exists():
            return p
    return cwd

try:
    ROOT_DIR = Path(__file__).resolve().parents[2]  # ejecutado como script dentro del repo
except NameError:
    ROOT_DIR = _guess_root_from_cwd()               # ejecutado desde notebook / entorno interactivo

DATA_DIR       = ROOT_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
REPORTS_DIR    = ROOT_DIR / "reports"
OUTPUTS_DIR    = PROCESSED_DIR / "preds" / "baselines" / "holt_winters"

DEFAULT_INPUT  = (PROCESSED_DIR / "dataset_modelado_ready.parquet").resolve()

# ---------------- Configuración de logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------- Utilidades y métricas ----------------

def ensure_dirs(save_outputs: bool):
    if save_outputs:
        (REPORTS_DIR / "baselines" / "holt_winters").mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def check_columns(df: pd.DataFrame, date_col: str, target_col: str, cluster_col: str):
    missing = [c for c in [date_col, target_col, cluster_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        raise TypeError(f"La columna '{date_col}' debe ser datetime64. Tipado actual: {df[date_col].dtype}")

def aggregate_cluster_daily(df: pd.DataFrame, date_col: str, cluster_col: str, target_col: str) -> pd.DataFrame:
    g = (
        df[[date_col, cluster_col, target_col]]
        .groupby([date_col, cluster_col], as_index=False, sort=False)[target_col]
        .sum()
    )
    if g[cluster_col].dtype == "object":
        g[cluster_col] = g[cluster_col].astype("category")
    return g

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(np.mean(2.0 * num / den))

def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / np.clip(np.sum(np.abs(y_true)), eps, None))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _interpolate_feb29(df: pd.DataFrame, date_col: str, y_col: str) -> pd.DataFrame:
    """
    Si existe 02-29 en la serie (p. ej., año bisiesto) y el valor está NaN,
    se interpola con la media de 02-28 y 03-01 cuando ambos existan.
    """
    df = df.sort_values(date_col).copy()
    mmdd = df[date_col].dt.strftime("%m-%d")
    if not (mmdd == "02-29").any():
        return df
    idx_29 = mmdd[mmdd == "02-29"].index   # <-- corregido
    for i in idx_29:
        d = df.loc[i, date_col]
        d28 = d.replace(month=2, day=28)
        d01 = d.replace(month=3, day=1)
        v28 = df.loc[df[date_col] == d28, y_col]
        v01 = df.loc[df[date_col] == d01, y_col]
        if len(v28) and len(v01) and pd.isna(df.loc[i, y_col]):
            df.loc[i, y_col] = 0.5 * float(v28.iloc[0]) + 0.5 * float(v01.iloc[0])
    return df

def _resolve_input_path(input_arg: Path) -> Path:
    """
    Devuelve una ruta existente al parquet:
    - Si es absoluta y existe -> OK.
    - Si es relativa -> prueba ROOT_DIR, CWD y padres de ROOT_DIR (en ese orden).
    - Si no aparece -> busca por nombre dentro del repo y, si no, lanza error claro.
    """
    cand = Path(input_arg)

    if cand.is_absolute() and cand.exists():
        return cand

    tried = []
    bases = [ROOT_DIR, Path.cwd(), *ROOT_DIR.parents[:3]]
    for base in bases:
        p = (base / cand).resolve()
        tried.append(p)
        if p.exists():
            return p

    # Búsqueda por nombre
    try:
        hits = list(ROOT_DIR.rglob(cand.name))
        for p in hits:
            if p.is_file():
                logging.warning(f"Input no encontrado en rutas esperadas; usando hallazgo: {p}")
                return p
    except Exception:
        pass

    msg = " | ".join(str(x) for x in tried)
    raise FileNotFoundError(
        f"No se encontró el dataset de entrada.\n"
        f"Argumento recibido: {input_arg}\n"
        f"Rutas intentadas: {msg}\n"
        f"Sugerencia: ejecuta con --input \"{DEFAULT_INPUT}\""
    )


# ---------------- ETS (Holt-Winters) por clúster ----------------

# Hiperparámetros por defecto (Opción 1: sin tendencia / sin amortiguado)
ETS_TREND = None          # antes: "add"
ETS_SEASONAL = "add"      # 'add' | 'mul' | None (aditiva es robusta con ceros)
ETS_DAMPED = False        # antes: True
SEASONAL_PERIODS = 365    # anual aprox. en datos diarios

def _fit_predict_ets(series: pd.Series,
                     start_pred: pd.Timestamp,
                     end_pred: pd.Timestamp,
                     seasonal_periods: int,
                     trend,
                     seasonal,
                     damped: bool) -> pd.DataFrame:
    """
    Ajusta ETS sobre 'series' (índice diario) y predice del start_pred al end_pred (ambos inclusive).
    """
    # Asegurar frecuencia diaria continua en entrenamiento (relleno con 0 donde falte)
    s = series.asfreq("D").fillna(0.0)

    # statsmodels requiere al menos ~ 2*seasonal_periods para modelos con estacionalidad
    if seasonal and len(s) < 2 * seasonal_periods:
        raise ValueError(f"Serie demasiado corta para ETS estacional (len={len(s)}).")

    model = ExponentialSmoothing(
        s,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
        initialization_method="estimated"
    )

    # Reducir sesgo sistemático
    with warnings.catch_warnings():
        warnings.simplefilter("once", ConvergenceWarning)
        fitted = model.fit(optimized=True, remove_bias=True)

    pred_index = pd.date_range(start=start_pred, end=end_pred, freq="D")
    yhat = fitted.predict(start=pred_index[0], end=pred_index[-1])
    return pd.DataFrame({"date": pred_index, "y_pred": np.asarray(yhat, dtype=float)})

def _predict_split_for_cluster(g: pd.DataFrame,
                               date_col: str,
                               target_col: str,
                               year_from: int,
                               year_to: int,
                               seasonal_periods: int,
                               trend,
                               seasonal,
                               damped: bool) -> pd.DataFrame:
    """
    Ajusta ETS con datos de [year_from, year_to-1] y predice year_to completo.
    """
    train_mask = (g[date_col].dt.year >= year_from) & (g[date_col].dt.year <= (year_to - 1))
    train = g.loc[train_mask, [date_col, target_col]].copy()
    if train.empty:
        raise ValueError(f"Sin datos de entrenamiento para años {year_from}-{year_to-1}")

    series = train.set_index(date_col)[target_col].sort_index()
    start_pred = pd.Timestamp(year=year_to, month=1, day=1)
    end_pred   = pd.Timestamp(year=year_to, month=12, day=31)

    preds = _fit_predict_ets(
        series,
        start_pred=start_pred,
        end_pred=end_pred,
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
        damped=damped
    )
    return preds

def holt_winters_predict_cluster_level(agg_df: pd.DataFrame,
                                       date_col: str,
                                       target_col: str,
                                       cluster_col: str,
                                       train_years: str,
                                       year_target: int,
                                       seasonal_periods: int,
                                       trend,
                                       seasonal,
                                       damped: bool) -> pd.DataFrame:
    """
    Entrena ETS con los años 'train_years' y predice 'year_target' por clúster.
    Devuelve: date, <cluster_col>, y_true (si existe), y_pred, split_year
    """
    years = [int(y) for y in train_years.split(",")]
    y_from = min(years)

    rows = []
    for cl, g in agg_df.groupby(cluster_col, sort=False):
        g = g[[date_col, target_col]].copy()

        preds = _predict_split_for_cluster(
            g, date_col, target_col, y_from, year_target,
            seasonal_periods=seasonal_periods,
            trend=trend, seasonal=seasonal, damped=damped
        )
        preds[cluster_col] = cl  # usar SIEMPRE cluster_col

        # incorporar y_true si existe
        truth = agg_df[(agg_df[cluster_col] == cl) & (agg_df[date_col].dt.year == year_target)][[date_col, target_col]]
        truth = truth.rename(columns={date_col: "date", target_col: "y_true"})
        merged = preds.merge(truth, on="date", how="left")

        # Interpolar posible 29-feb en y_pred si y_true existe (mantener comparabilidad)
        merged = _interpolate_feb29(merged, "date", "y_pred")

        merged["split_year"] = year_target
        rows.append(merged[["date", cluster_col, "y_true", "y_pred", "split_year"]])

    out = pd.concat(rows, ignore_index=True).sort_values(["date", cluster_col])
    return out


def compute_metrics_by_cluster(preds: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    rows = []
    for cl, g in preds.groupby(cluster_col, sort=False):
        g_valid = g.dropna(subset=["y_true"])
        if g_valid.empty:
            continue
        y_true = g_valid["y_true"].to_numpy(dtype=float)
        y_pred = g_valid["y_pred"].to_numpy(dtype=float)
        rows.append({
            cluster_col: cl, "n_days": len(g_valid),
            "MAE": mae(y_true, y_pred), "WAPE": wape(y_true, y_pred), "sMAPE": smape(y_true, y_pred),
        })
    cols = [cluster_col, "n_days", "MAE", "WAPE", "sMAPE"]
    return pd.DataFrame(rows, columns=cols).sort_values(cluster_col) if rows else pd.DataFrame(columns=cols)


# ---------------- Runner ----------------

def run(input_path: Path,
        date_col: str,
        target_col: str,
        cluster_col: str,
        train_years: str,
        val_year: int,
        test_year: int,
        seasonal_periods: int = SEASONAL_PERIODS,
        trend = ETS_TREND,
        seasonal = ETS_SEASONAL,
        damped: bool = ETS_DAMPED,
        allow_missing_test: bool = True,
        save_outputs: bool = True):

    t_all = time.time()
    ensure_dirs(save_outputs=save_outputs)

    input_path = _resolve_input_path(Path(input_path))
    logging.info(f"Leyendo dataset: {input_path}")
    df = pd.read_parquet(input_path)

    # Tipado de fecha
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        logging.info(f"Parseando columna de fecha '{date_col}' a datetime.")
        df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")

    # Columna de clúster (tolerante a alias)
    if cluster_col not in df.columns:
        aliases = ["cluster_id", "cluster", "Cluster", "cluster_label", "clustername", "cluster_name"]
        found = next((c for c in aliases if c in df.columns), None)
        if found is not None:
            logging.warning(f"Columna '{cluster_col}' no encontrada. Usando alias: '{found}'")
            cluster_col = found
        else:
            raise ValueError(f"No se encontró la columna de clúster. Probados alias: {aliases}")

    check_columns(df, date_col, target_col, cluster_col)

    years_present = sorted(df[date_col].dt.year.unique().tolist())
    logging.info(f"Años presentes en el dataset: {years_present}")

    # Agregado cluster-día
    t0 = time.time()
    agg_df = aggregate_cluster_daily(df, date_col=date_col, cluster_col=cluster_col, target_col=target_col)
    logging.info(f"Agregado cluster-día en {time.time()-t0:.2f}s (rows={len(agg_df):,}, "
                 f"clusters={agg_df[cluster_col].nunique()})")

    # -------- Validación --------
    t1 = time.time()
    preds_val = holt_winters_predict_cluster_level(
        agg_df=agg_df, date_col=date_col, target_col=target_col, cluster_col=cluster_col,
        train_years=train_years, year_target=int(val_year),
        seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal, damped=damped
    ).assign(split="validation")
    logging.info(f"ETS validación {val_year}: listo en {time.time()-t1:.2f}s (rows={len(preds_val):,})")

    metrics_val = compute_metrics_by_cluster(preds_val, cluster_col=cluster_col)
    if not metrics_val.empty:
        gv = preds_val.dropna(subset=["y_true"])
        global_val = pd.DataFrame([{
            cluster_col: "__GLOBAL__", "n_days": gv.shape[0],
            "MAE": mae(gv["y_true"].to_numpy(), gv["y_pred"].to_numpy()),
            "WAPE": wape(gv["y_true"].to_numpy(), gv["y_pred"].to_numpy()),
            "sMAPE": smape(gv["y_true"].to_numpy(), gv["y_pred"].to_numpy()),
        }])
        metrics_val = pd.concat([metrics_val, global_val], ignore_index=True)

    # -------- Test --------
    has_test_truth = int(test_year) in years_present
    if has_test_truth:
        t2 = time.time()
        preds_test = holt_winters_predict_cluster_level(
            agg_df=agg_df, date_col=date_col, target_col=target_col, cluster_col=cluster_col,
            train_years=train_years, year_target=int(test_year),
            seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal, damped=damped
        ).assign(split="test")
        logging.info(f"ETS test {test_year} (con y_true): listo en {time.time()-t2:.2f}s (rows={len(preds_test):,})")
        metrics_test = compute_metrics_by_cluster(preds_test, cluster_col=cluster_col)
        if not metrics_test.empty:
            gt = preds_test.dropna(subset=["y_true"])
            global_test = pd.DataFrame([{
                cluster_col: "__GLOBAL__", "n_days": gt.shape[0],
                "MAE": mae(gt["y_true"].to_numpy(), gt["y_pred"].to_numpy()),
                "WAPE": wape(gt["y_true"].to_numpy(), gt["y_pred"].to_numpy()),
                "sMAPE": smape(gt["y_true"].to_numpy(), gt["y_pred"].to_numpy()),
            }])
            metrics_test = pd.concat([metrics_test, global_test], ignore_index=True)
    else:
        if not allow_missing_test:
            raise ValueError("No hay y_true de test y 'allow_missing_test' está desactivado.")
        t2 = time.time()
        preds_test = holt_winters_predict_cluster_level(
            agg_df=agg_df, date_col=date_col, target_col=target_col, cluster_col=cluster_col,
            train_years=train_years, year_target=int(test_year),
            seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal, damped=damped
        ).assign(split="test")
        preds_test["y_true"] = np.nan
        metrics_test = pd.DataFrame()
        logging.warning("Predicciones de test generadas SIN y_true. Métricas de test se omiten.")
        logging.info(f"ETS test {test_year} (sin y_true): listo en {time.time()-t2:.2f}s (rows={len(preds_test):,})")

    # -------- Exportación --------
    if save_outputs:
        reports_dir = REPORTS_DIR / "baselines" / "holt_winters"
        reports_dir.mkdir(parents=True, exist_ok=True)
        preds_val_path    = OUTPUTS_DIR / "preds_val.parquet"
        preds_test_path   = OUTPUTS_DIR / "preds_test.parquet"
        metrics_val_path  = reports_dir / "metrics_validation.csv"
        metrics_test_path = reports_dir / "metrics_test.csv"

        preds_val.to_parquet(preds_val_path, index=False)
        preds_test.to_parquet(preds_test_path, index=False)
        metrics_val.to_csv(metrics_val_path, index=False)
        if not metrics_test.empty:
            metrics_test.to_csv(metrics_test_path, index=False)
        else:
            logging.warning("No se exportan métricas de TEST: no hay y_true.")

        logging.info(f"Preds VAL → {preds_val_path}")
        logging.info(f"Preds TEST → {preds_test_path}")
        logging.info(f"Métricas VAL → {metrics_val_path}")
        if not metrics_test.empty:
            logging.info(f"Métricas TEST → {metrics_test_path}")
    else:
        logging.info("save_outputs=False → No se guardan ficheros en disco.")

    logging.info(f"OK Holt-Winters ETS. Tiempo total: {time.time()-t_all:.2f}s")


# ---------------- CLI (tolerante) ----------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Baseline Holt-Winters (ETS) por clúster (agregado cluster-día).")

    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--date-col", type=str, default="date")
    p.add_argument("--target-col", type=str, default="sales_quantity")
    p.add_argument("--cluster-col", type=str, default="cluster_id",
                   help="Nombre de la columna de clúster (por defecto 'cluster_id').")

    p.add_argument("--train-years", type=str, default="2022,2023")
    p.add_argument("--val-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)

    p.add_argument("--seasonal-periods", type=int, default=SEASONAL_PERIODS)
    p.add_argument("--trend", type=str, default="none", choices=["add", "mul", "none"],
                   help="Tendencia ETS (por defecto 'none' acorde a Opción 1).")
    p.add_argument("--seasonal", type=str, default=ETS_SEASONAL, choices=["add", "mul", "none"])
    p.add_argument("--damped", action="store_true", default=False,
                   help="Activa tendencia amortiguada. Por defecto desactivado (Opción 1).")

    p.add_argument("--allow-missing-test", action="store_true", default=True)
    p.add_argument("--save-outputs", dest="save_outputs", action="store_true", default=True)
    p.add_argument("--no-save", dest="save_outputs", action="store_false", help="No guardar outputs en disco")

    args, _ = p.parse_known_args(argv)

    # Normalizar 'none' -> None para trend/seasonal
    args.trend    = None if (isinstance(args.trend, str) and args.trend.lower() == "none") else args.trend
    args.seasonal = None if (isinstance(args.seasonal, str) and args.seasonal.lower() == "none") else args.seasonal
    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        run(
            input_path=Path(args.input),
            date_col=args.date_col,
            target_col=args.target_col,
            cluster_col=args.cluster_col,
            train_years=args.train_years,
            val_year=args.val_year,
            test_year=args.test_year,
            seasonal_periods=args.seasonal_periods,
            trend=args.trend,
            seasonal=args.seasonal,
            damped=args.damped,
            allow_missing_test=args.allow_missing_test,
            save_outputs=args.save_outputs,
        )
    except Exception as e:
        logging.exception(f"Error en holt_winters_ets.py: {e}")
        sys.exit(1)


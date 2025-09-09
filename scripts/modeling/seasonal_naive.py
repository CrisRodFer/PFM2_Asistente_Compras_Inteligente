# ==================================================================================================
# Script: seasonal_naive.py
#
# Propósito
# ---------
# Baseline **Seasonal Naive** por clúster (cluster_id) para demanda diaria.
# Predice que la demanda de un día es igual a la del **mismo MM-DD** del año anterior
# dentro del **mismo clúster**.
#
# Justificación
# -------------
# - Es un benchmark simple y estable para comparar modelos más complejos.
# - Captura la estacionalidad diaria sin sobreajustar.
# - Minimiza coste computacional y memoria (agregación por clúster-día).
#
# Flujo del pipeline
# ------------------
# 0) Carga, validación y tipados del dataset.
# 1) **Agregación** a nivel [date, cluster_id] sumando `sales_quantity` (evita joins many-to-many).
# 2) **Validación 2024** usando 2023 como referencia -> métricas por clúster + global (MAE, WAPE, sMAPE).
# 3) **Test 2025** usando 2024 como referencia.
#    - Si no existen `y_true` 2025 (lo normal, es nuestro horizonte): se generan predicciones y
#      **no** se calculan métricas de test (`--allow-missing-test` activado por defecto).
# 4) **29/02**:
#    - Si el año destino es bisiesto, se garantiza un valor para "02-29" en el mapeo MM-DD del año
#      origen por interpolación entre "02-28" y "03-01" (sin crear fechas inválidas).
#    - Si no es bisiesto, "02-29" se omite en la construcción del calendario destino.
# 5) Export:
#    - Predicciones: `data/processed/preds/baselines/seasonal_naive/`
#    - Métricas:     `reports/baselines/seasonal_naive/`
#
# Entradas mínimas
# ----------------
# - Parquet: `data/processed/dataset_modelado_ready.parquet`
# - Columnas: `date` (datetime64), `sales_quantity` (numérico), `cluster_id` (categoría/id)
#
# Salidas
# -------
# - `data/processed/preds/baselines/seasonal_naive/preds_val.parquet`
# - `data/processed/preds/baselines/seasonal_naive/preds_test.parquet`
# - `reports/baselines/seasonal_naive/metrics_validation.csv`
# - `reports/baselines/seasonal_naive/metrics_test.csv` (solo si hay y_true)
#
# Uso
# ---
# Terminal:
#   python scripts/modeling/seasonal_naive.py \
#       --input data/processed/dataset_modelado_ready.parquet \
#       --date-col date --target-col sales_quantity --cluster-col cluster_id \
#       --val-year 2024 --test-year 2025 --leap-fill interp
#
# Notebook (comportamiento idéntico):
#   !python scripts/modeling/seasonal_naive.py \
#       --input data/processed/dataset_modelado_ready.parquet \
#       --date-col date --target-col sales_quantity --cluster-col cluster_id \
#       --val-year 2024 --test-year 2025 --leap-fill interp
#
# Dependencias
# ------------
#   pip install pandas pyarrow numpy
#
# Nota importante
# ---------------
# Es normal no tener `y_true` de 2025: es el horizonte que se quiere predecir. El script genera
# las predicciones de test y omite sus métricas (comportamiento esperado y documentado).
# ==================================================================================================

from pathlib import Path
import argparse, sys, logging, time
import numpy as np
import pandas as pd

# ---------------- Rutas base del proyecto (robustas para script y notebook) ----------------

def _guess_root_from_cwd() -> Path:
    """
    Sube por los padres desde el CWD hasta encontrar una carpeta que contenga 'data/processed'.
    Si no la encuentra, devuelve el CWD actual.
    """
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "data" / "processed").exists():
            return p
    return cwd

try:
    # Cuando se ejecuta como script
    ROOT_DIR = Path(__file__).resolve().parents[2]
except NameError:
    # Cuando se ejecuta desde notebook (no hay __file__)
    ROOT_DIR = _guess_root_from_cwd()

DATA_DIR       = ROOT_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
REPORTS_DIR    = ROOT_DIR / "reports"
OUTPUTS_DIR    = PROCESSED_DIR / "preds" / "baselines" / "seasonal_naive"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------- Utilidades ----------------

def ensure_dirs():
    (REPORTS_DIR / "baselines" / "seasonal_naive").mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def check_columns(df: pd.DataFrame, date_col: str, target_col: str, cluster_col: str):
    missing = [c for c in [date_col, target_col, cluster_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        raise TypeError(f"La columna '{date_col}' debe ser datetime64. Tipado actual: {df[date_col].dtype}")

def add_month_day(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    return df.assign(_mmdd=df[date_col].dt.strftime("%m-%d"))

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    num = np.abs(y_true - y_pred); den = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return np.mean(2.0 * num / den)

def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return np.sum(np.abs(y_true - y_pred)) / np.clip(np.sum(np.abs(y_true)), eps, None)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def augment_prev_with_feb29(prev_map: pd.DataFrame, target_col: str, cluster_col: str) -> pd.DataFrame:
    rows = []
    for cl, g in prev_map.groupby(cluster_col, sort=False):
        if not (g["_mmdd"] == "02-29").any():
            v28 = g.loc[g["_mmdd"] == "02-28", target_col].mean()
            v01 = g.loc[g["_mmdd"] == "03-01", target_col].mean()
            if pd.notna(v28) and pd.notna(v01):
                rows.append({cluster_col: cl, "_mmdd": "02-29", target_col: 0.5 * (v28 + v01)})
            elif pd.notna(v28):
                rows.append({cluster_col: cl, "_mmdd": "02-29", target_col: float(v28)})
            elif pd.notna(v01):
                rows.append({cluster_col: cl, "_mmdd": "02-29", target_col: float(v01)})
    if rows:
        prev_map = pd.concat([prev_map, pd.DataFrame(rows)], ignore_index=True)
    return prev_map

def aggregate_cluster_daily(df: pd.DataFrame, date_col: str, cluster_col: str, target_col: str) -> pd.DataFrame:
    g = (
        df[[date_col, cluster_col, target_col]]
        .groupby([date_col, cluster_col], as_index=False, sort=False)[target_col]
        .sum()
    )
    if g[cluster_col].dtype == "object":
        g[cluster_col] = g[cluster_col].astype("category")
    return g

def build_prev_map(prev_df: pd.DataFrame, date_col: str, cluster_col: str, target_col: str,
                   ensure_feb29: bool) -> pd.DataFrame:
    tmp = add_month_day(prev_df, date_col)[[cluster_col, "_mmdd", target_col]].copy()
    tmp = tmp.groupby([cluster_col, "_mmdd"], as_index=False, sort=False)[target_col].mean()
    if ensure_feb29:
        tmp = augment_prev_with_feb29(tmp, target_col=target_col, cluster_col=cluster_col)
    return tmp.rename(columns={target_col: "y_prev"})

def seasonal_naive_predict_cluster_level(
    agg_df: pd.DataFrame, year_target: int, date_col: str, target_col: str,
    cluster_col: str, leap_fill: str = "interp",
) -> pd.DataFrame:
    t0 = time.time()
    prev_year = year_target - 1
    is_target_leap = pd.Timestamp(year=year_target, month=12, day=31).is_leap_year

    df_prev = agg_df[agg_df[date_col].dt.year == prev_year].copy()
    df_tgt  = agg_df[agg_df[date_col].dt.year == year_target].copy()

    prev_map = build_prev_map(
        prev_df=df_prev, date_col=date_col, cluster_col=cluster_col, target_col=target_col,
        ensure_feb29=(is_target_leap and leap_fill == "interp"),
    )

    df_tgt = add_month_day(df_tgt, date_col)
    merged = (
        df_tgt[[date_col, cluster_col, "_mmdd", target_col]]
        .merge(prev_map, on=[cluster_col, "_mmdd"], how="left")
        .rename(columns={target_col: "y_true"})
    )
    merged["y_pred"] = merged["y_prev"]
    merged = merged.drop(columns=["_mmdd", "y_prev"], errors="ignore").rename(columns={date_col: "date"})
    merged["split_year"] = year_target
    merged = merged[["date", cluster_col, "y_true", "y_pred", "split_year"]].sort_values(["date", cluster_col])

    logging.info(f"SN predict {year_target}: agregado + mapeo en {time.time()-t0:.2f}s "
                 f"(rows={len(merged):,}, clusters={merged[cluster_col].nunique()})")
    return merged

def seasonal_naive_forecast_without_truth_cluster_level(
    agg_df: pd.DataFrame, year_target: int, date_col: str, target_col: str, cluster_col: str,
) -> pd.DataFrame:
    """
    Predicciones para year_target SIN y_true a partir del año anterior, en nivel CLÚSTER.
    Construcción segura de fecha destino (evita errores con 29/02 y colisión de nombres).
    """
    prev_year = year_target - 1
    df_prev = agg_df[agg_df[date_col].dt.year == prev_year].copy()
    if df_prev.empty:
        raise ValueError(f"No hay datos del año anterior ({prev_year}) para predecir {year_target}.")

    tmp = df_prev[[date_col, cluster_col, target_col]].copy()
    tmp["_mmdd"] = tmp[date_col].dt.strftime("%m-%d")

    is_target_leap = pd.Timestamp(year=year_target, month=12, day=31).is_leap_year
    if not is_target_leap:
        tmp = tmp.loc[tmp["_mmdd"] != "02-29"].copy()

    # construir fecha destino en columna temporal y renombrar al final
    tmp["date_target"] = pd.to_datetime(str(year_target) + "-" + tmp["_mmdd"], errors="coerce")
    tmp = tmp.dropna(subset=["date_target"]).copy()

    tmp = (
        tmp.drop(columns=[date_col, "_mmdd"])
           .rename(columns={"date_target": "date", target_col: "y_pred"})
    )
    tmp["y_true"] = np.nan
    tmp["split_year"] = year_target
    tmp = tmp[["date", cluster_col, "y_true", "y_pred", "split_year"]].sort_values(["date", cluster_col])
    return tmp

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

def run(input_path: Path, date_col: str, target_col: str, cluster_col: str,
        train_years: str, val_year: int, test_year: int,
        leap_fill: str = "interp", allow_missing_test: bool = True):
    t_all = time.time()
    ensure_dirs()

    # Normalizar la ruta de entrada respecto al ROOT_DIR si es relativa (fix notebook)
    input_path = Path(input_path)
    if not input_path.is_absolute():
        input_path = (ROOT_DIR / input_path).resolve()

    logging.info(f"Leyendo dataset: {input_path}")
    df = pd.read_parquet(input_path)

    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        logging.info(f"Parseando columna de fecha '{date_col}' a datetime.")
        df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")

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

    t0 = time.time()
    agg_df = aggregate_cluster_daily(df, date_col=date_col, cluster_col=cluster_col, target_col=target_col)
    logging.info(f"Agregado cluster-día en {time.time()-t0:.2f}s (rows={len(agg_df):,}, "
                 f"clusters={agg_df[cluster_col].nunique()})")

    needed_train_val = set([*(int(y) for y in train_years.split(",")), int(val_year)])
    years_present_agg = set(agg_df["date"].dt.year.unique().tolist())
    if not needed_train_val.issubset(years_present_agg):
        raise ValueError(f"Faltan años requeridos para train/val {sorted(list(needed_train_val))} en el agregado.")

    has_test_truth = int(test_year) in years_present_agg
    if not has_test_truth and not allow_missing_test:
        raise ValueError("No hay datos reales de test y --allow-missing-test no está activo.")

    preds_val = seasonal_naive_predict_cluster_level(
        agg_df=agg_df, year_target=int(val_year),
        date_col=date_col, target_col=target_col, cluster_col=cluster_col, leap_fill=leap_fill
    ).assign(split="validation")

    if has_test_truth:
        preds_test = seasonal_naive_predict_cluster_level(
            agg_df=agg_df, year_target=int(test_year),
            date_col=date_col, target_col=target_col, cluster_col=cluster_col, leap_fill=leap_fill
        ).assign(split="test")
    else:
        preds_test = seasonal_naive_forecast_without_truth_cluster_level(
            agg_df=agg_df, year_target=int(test_year),
            date_col=date_col, target_col=target_col, cluster_col=cluster_col
        ).assign(split="test")
        logging.warning("Predicciones de test generadas SIN y_true. Métricas de test se omiten.")

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

    if has_test_truth:
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
        metrics_test = pd.DataFrame()

    reports_dir = REPORTS_DIR / "baselines" / "seasonal_naive"
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

    logging.info(f"OK Seasonal-Naive. Tiempo total: {time.time()-t_all:.2f}s")
    logging.info(f"Preds VAL → {preds_val_path}")
    logging.info(f"Preds TEST → {preds_test_path}")
    logging.info(f"Métricas VAL → {metrics_val_path}")
    if not metrics_test.empty:
        logging.info(f"Métricas TEST → {metrics_test_path}")

# ---------------- CLI (tolerante a notebook) ----------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Baseline Seasonal Naive por clúster (agregado a cluster-día).")
    p.add_argument("--input", type=str, default=str(PROCESSED_DIR / "dataset_modelado_ready.parquet"))
    p.add_argument("--date-col", type=str, default="date")
    p.add_argument("--target-col", type=str, default="sales_quantity")
    p.add_argument("--cluster-col", type=str, default="cluster_id",
                   help="Nombre de la columna de clúster (por defecto 'cluster_id').")
    p.add_argument("--train-years", type=str, default="2022,2023",
                   help="Años de entrenamiento (verificación de presencia).")
    p.add_argument("--val-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--leap-fill", type=str, choices=["interp", "drop"], default="interp",
                   help="Tratamiento de '02-29' en mapeo MM-DD cuando el destino es bisiesto.")
    p.add_argument("--allow-missing-test", action="store_true", default=True,
                   help="Permite generar predicciones de test aunque el parquet no tenga y_true (omite métricas).")
    # Ignorar argumentos extra que añade Jupyter (p.ej., --f=...)
    args, _ = p.parse_known_args(argv)
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
            leap_fill=args.leap_fill,
            allow_missing_test=args.allow_missing_test,
        )
    except Exception as e:
        logging.exception(f"Error en seasonal_naive.py: {e}")
        sys.exit(1)

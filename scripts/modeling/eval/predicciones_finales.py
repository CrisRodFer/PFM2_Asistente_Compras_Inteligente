# =============================================================================
# Script: predicciones_finales.py
# Autor:  Equipo PFM2
# Descripción:
#   Entrena Random Forest por clúster con todo el histórico disponible (2022–2024)
#   y genera las predicciones diarias para 2025 (enero–diciembre).
#
#   Si el dataset ML listo no contiene filas en 2025, el script construye
#   automáticamente una MATRIZ FUTURA de features en modo "status-quo":
#     - Repite el último valor disponible por id (cluster / product_id si existe).
#     - Recalcula columnas de calendario (dow, month, weekofyear, is_weekend) si están.
#     - Pone a 0 las columnas de promo (si las hay).
#
# Flujo:
#   1) Cargar dataset enriquecido (parquet) y seleccionar features válidas.
#   2) Split: train ≤ 2024-12-31; horizonte 2025-01-01..2025-12-31.
#   3) (Pre-check) Garantizar unicidad por clave adecuada en el horizonte.
#   4) Entrenar RF por clúster con 2022–2024 y predecir 2025.
#   5) Validación estructural de predicciones (columnas, rango, nulos, duplicados).
#   6) Exportar:
#        - data/processed/predicciones_2025.parquet  (predicciones diarias)
#        - reports/predicciones/summary_2025.csv     (resumen por clúster y total)
#
# Entradas:
#   - data/processed/dataset_ml_ready.parquet
#
# Salidas:
#   - data/processed/predicciones_2025.parquet
#   - reports/predicciones/summary_2025.csv
#
# Notas:
#   - Este script no recalcula métricas; la validación de rendimiento se realizó en 8.5.
#   - El objetivo aquí es el "final fit" y la generación de predicciones consumibles.
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]  # .../PFM2_Asistente_Compras_Inteligente
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "predicciones"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROCESSED_DIR / "dataset_ml_ready.parquet"
OUT_PARQUET  = PROCESSED_DIR / "predicciones_2025.parquet"
OUT_SUMMARY  = REPORTS_DIR / "summary_2025.csv"

# Columnas clave del dataset
TARGET      = "sales_quantity"
DATE_COL    = "date"
CLUSTER_COL = "cluster_id"

# Exclusiones (no pueden ser features)
EXCLUDE_COLS = {
    TARGET, DATE_COL, CLUSTER_COL, "product_id",
    "demand_day_priceadj", "demand_adjust"
}

# Horizonte definitivo
PRED_START = "2025-01-01"
PRED_END   = "2025-12-31"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def auto_feature_cols(df: pd.DataFrame) -> list[str]:
    """Numéricas válidas como features (excluye target/fecha/ids/derivados)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in EXCLUDE_COLS]
    if not feats:
        raise ValueError("No se encontraron columnas numéricas para usar como features.")
    return feats

def _recompute_calendar_cols(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Recalcular columnas de calendario SOLO si existen en el DF."""
    d = pd.to_datetime(df[date_col])
    if "dow" in df.columns:
        df["dow"] = d.dt.dayofweek
    if "month" in df.columns:
        df["month"] = d.dt.month
    if "weekofyear" in df.columns:
        df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    if "is_weekend" in df.columns:
        df["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    return df

def _make_future_matrix_status_quo(
    hist: pd.DataFrame, feature_cols: list[str], start: str, end: str
) -> pd.DataFrame:
    """Construye matriz futura copiando último valor por id y recalculando calendario."""
    dates = pd.date_range(start, end, freq="D")
    id_cols = [CLUSTER_COL] + (["product_id"] if "product_id" in hist.columns else [])

    # Última fila por id antes del horizonte
    last_by_id = (
        hist.sort_values(DATE_COL)
            .groupby(id_cols, as_index=False)
            .tail(1)
            .reset_index(drop=True)
    )

    # Repetir por fechas
    rep = last_by_id.loc[last_by_id.index.repeat(len(dates))].reset_index(drop=True)
    rep[DATE_COL] = np.tile(dates.values, len(last_by_id))

    # Mantener solo columnas necesarias para predicción
    cols_needed = set([DATE_COL] + id_cols + feature_cols)
    rep = rep[[c for c in rep.columns if c in cols_needed]].copy()

    # Poner a 0 columnas de promo si existen
    promo_cols = [c for c in feature_cols if "promo" in c.lower()]
    for c in promo_cols:
        if c in rep.columns:
            rep[c] = 0.0

    # Recalcular calendario si está presente
    rep = _recompute_calendar_cols(rep, DATE_COL)
    return rep

def _horizon_key_cols(df: pd.DataFrame) -> list[str]:
    """Clave correcta para unicidad en horizonte/predicciones."""
    return [CLUSTER_COL, "product_id", DATE_COL] if "product_id" in df.columns else [CLUSTER_COL, DATE_COL]

def validate_output_schema(df: pd.DataFrame) -> None:
    """Validación estructural mínima del dataset de predicciones."""
    required = {DATE_COL, CLUSTER_COL, "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predicciones sin columnas requeridas: {missing}")

    if df[DATE_COL].isna().any():
        raise ValueError("Existen fechas nulas en las predicciones.")
    if df["y_pred"].isna().any():
        raise ValueError("Existen predicciones nulas en 'y_pred'.")

    # Cobertura temporal
    s, e = pd.to_datetime(PRED_START), pd.to_datetime(PRED_END)
    if df[DATE_COL].min() < s or df[DATE_COL].max() > e:
        raise ValueError("Rango temporal fuera del [PRED_START, PRED_END].")

    # Duplicados: clave depende de si hay product_id
    key_cols = _horizon_key_cols(df)
    dup = df.duplicated(subset=key_cols).sum()
    if dup:
        raise ValueError(f"Hay {dup} duplicados en la clave {key_cols} del resultado.")

def summarize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen por clúster y global (suma de y_pred)."""
    grp = (df.groupby(CLUSTER_COL, as_index=False)["y_pred"]
             .sum()
             .rename(columns={"y_pred": "forecast_sum"}))
    total = pd.DataFrame([{CLUSTER_COL: "TOTAL", "forecast_sum": grp["forecast_sum"].sum()}])
    return pd.concat([grp, total], ignore_index=True)

# ==== 3. LÓGICA PRINCIPAL =====================================================
def run_final_predictions(
    dataset_path: Path = DATASET_PATH,
    out_parquet: Path = OUT_PARQUET,
    out_summary: Path = OUT_SUMMARY,
    pred_start: str = PRED_START,
    pred_end: str = PRED_END,
) -> None:
    ensure_dirs(out_parquet.parent, out_summary.parent)

    log.info("Leyendo dataset: %s", dataset_path)
    df = pd.read_parquet(dataset_path)
    for c in (DATE_COL, CLUSTER_COL, TARGET):
        if c not in df.columns:
            raise ValueError(f"Dataset incompleto: falta {c}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Train y horizonte nominal
    train = df[df[DATE_COL] <= "2024-12-31"].copy()
    horizon = df[(df[DATE_COL] >= pred_start) & (df[DATE_COL] <= pred_end)].copy()

    feature_cols = auto_feature_cols(df)

    # Si no hay filas en 2025, construimos matriz futura status-quo
    if horizon.empty:
        log.warning("No hay filas en el horizonte %s..%s. Se generará matriz futura status-quo.",
                    pred_start, pred_end)
        hist_until_2024 = df[df[DATE_COL] <= "2024-12-31"].copy()
        horizon = _make_future_matrix_status_quo(hist_until_2024, feature_cols, pred_start, pred_end)

    # --- Pre-check de duplicados reales en el horizonte (clave correcta)
    key_cols = _horizon_key_cols(horizon)
    dup_h = horizon.duplicated(subset=key_cols).sum()
    if dup_h:
        log.warning("Horizonte contiene %d duplicados por clave %s. Se aplicará drop_duplicates(keep='last').",
                    dup_h, key_cols)
        horizon = horizon.drop_duplicates(subset=key_cols, keep="last").copy()

    clusters = sorted(df[CLUSTER_COL].unique())
    log.info("Clusters detectados: %s | Nº features: %d", clusters, len(feature_cols))

    rows = []

    for cl in clusters:
        tr = train[train[CLUSTER_COL] == cl]
        hv = horizon[horizon[CLUSTER_COL] == cl]

        if tr.empty or hv.empty:
            log.warning("Cluster %s sin datos de train o de horizonte. Saltando.", cl)
            continue

        X_tr = tr[feature_cols].values
        y_tr = tr[TARGET].values
        X_hv = hv[feature_cols].values

        rf = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_tr, y_tr)
        y_hat = rf.predict(X_hv)

        pred_dict = {
            DATE_COL: pd.to_datetime(hv[DATE_COL].values),
            CLUSTER_COL: cl,
            "y_pred": y_hat
        }
        if "product_id" in hv.columns:
            pred_dict["product_id"] = hv["product_id"].values

        rows.append(pd.DataFrame(pred_dict))

    if not rows:
        raise RuntimeError("No se generaron predicciones para ningún clúster.")

    preds = pd.concat(rows, ignore_index=True)

    # Validación estructural
    validate_output_schema(preds)

    # Exportar
    preds.to_parquet(out_parquet, index=False)
    log.info("Predicciones escritas en: %s", out_parquet)

    # Resumen por clúster y total
    summary = summarize_predictions(preds)
    summary.to_csv(out_summary, index=False)
    log.info("Resumen escrito en: %s", out_summary)

# ==== 4. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predicciones finales 2025 (RF por clúster).")
    p.add_argument("--dataset", type=str, default=str(DATASET_PATH),
                   help="Ruta al parquet del dataset ML listo.")
    p.add_argument("--out-parquet", type=str, default=str(OUT_PARQUET),
                   help="Ruta de salida (parquet) para predicciones 2025.")
    p.add_argument("--out-summary", type=str, default=str(OUT_SUMMARY),
                   help="Ruta de salida (CSV) para resumen por clúster.")
    p.add_argument("--start", type=str, default=PRED_START, help="Fecha inicio horizonte (YYYY-MM-DD).")
    p.add_argument("--end", type=str, default=PRED_END, help="Fecha fin horizonte (YYYY-MM-DD).")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    try:
        run_final_predictions(
            dataset_path=Path(args.dataset),
            out_parquet=Path(args.out_parquet),
            out_summary=Path(args.out_summary),
            pred_start=args.start,
            pred_end=args.end,
        )
    except Exception as e:
        log.exception("Fallo en predicciones finales: %s", e)
        raise

if __name__ == "__main__":
    main()



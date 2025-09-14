# =============================================================================
# Nombre:    aplicar_estacionalidad.py
# Autor:     Equipo PFM2
# Descripción:
#   Calcula la curva estacional mensual media desde históricos unificados
#   (demanda_unificada.csv) y la aplica a las predicciones diarias del año
#   objetivo (por defecto 2025) contenidas en predicciones_2025.parquet.
#   Implementación ligera en memoria: map() por mes (sin merges por fecha)
#   y re-normalización para preservar el total anual.
#
# Flujo:
#   1) Leer históricos unificados (CSV) y calcular factores mensuales (12 shares).
#   2) Leer predicciones diarias (parquet), detectar columna (y_pred|y).
#   3) Aplicar factores por mes SOLO al año objetivo y re-escalar total anual.
#   4) Guardar parquet final del año objetivo + control y (opcional) factores.
#
# Entradas (requeridas):
#   --pred-base      Parquet con predicción 2025 (cols mínimas: date, y_pred).
#   --historicos     CSV con históricos unificados (cols: Product_ID,Date,Demand_Day).
#   --out-parquet    Ruta parquet de salida con y_pred_estacional.
#
# Salidas:
#   - Parquet final (data/processed/predicciones_2025_estacional.parquet)
#   - CSV factores estacionales (reports/seasonality/factores_estacionales_2022_2024.csv)
#   - CSV control totales (reports/seasonality/control_totales_2025.csv)
#
# Uso recomendado:
#   python scripts/modeling/eval/aplicar_estacionalidad.py \
#       --pred-base data/processed/predicciones_2025.parquet \
#       --historicos data/processed/demanda_unificada.csv \
#       --out-parquet data/processed/predicciones_2025_estacional.parquet
#
# Notas:
#   - Los factores son promedio mensual relativo (media mensual / media global)
#     normalizados para que su media sea 1 (equivale a suma≈12).
#   - El ajuste preserva el total anual (verificación incluida).
# =============================================================================

from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# --- Rutas --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "seasonality"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PRED_BASE_PARQUET = DATA_DIR / "predicciones_2025.parquet"
HIST_CSV          = DATA_DIR / "demanda_unificada.csv"

OUT_PARQUET      = DATA_DIR / "predicciones_2025_estacional.parquet"
OUT_FACTORES_CSV = REPORTS_DIR / "factores_estacionales_2022_2024.csv"
OUT_CONTROL_CSV  = REPORTS_DIR / "control_totales_2025.csv"

# --- Logging ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("aplicar_estacionalidad")


# --- Utilidades ---------------------------------------------------------
def _leer_historicos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    col_date = cols.get("date") or cols.get("fecha") or "Date"
    col_y    = cols.get("demand_day") or cols.get("demanda") or "Demand_Day"
    df = df[[col_date, col_y]].copy()
    df.columns = ["date", "y"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"].dt.year >= 2022) & (df["date"].dt.year <= 2024)]
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df["month"] = df["date"].dt.month
    return df

def _calcular_factores(df_hist: pd.DataFrame) -> pd.DataFrame:
    monthly = (df_hist.groupby("month", as_index=False)["y"]
               .sum().rename(columns={"y": "total_month"})
               .sort_values("month"))
    total = monthly["total_month"].sum()
    if total <= 0:
        raise ValueError("Total histórico = 0; no se pueden calcular factores.")
    monthly["factor"] = monthly["total_month"] / total  # suma = 1.0
    return monthly

def _leer_predicciones(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        cand = [c for c in df.columns if c.lower() in ("date", "fecha")]
        if not cand:
            raise ValueError("No se encontró columna 'date' en predicciones.")
        df = df.rename(columns={cand[0]: "date"})
    if "y_pred" not in df.columns:
        cand = [c for c in df.columns if c.lower() in ("y_pred", "yhat", "pred")]
        if not cand:
            raise ValueError("No se encontró columna 'y_pred' en predicciones.")
        df = df.rename(columns={cand[0]: "y_pred"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0.0)
    return df

def _inyectar_estacionalidad_en_linea(df_pred: pd.DataFrame,
                                      factores: pd.DataFrame) -> tuple[pd.DataFrame,float,float]:
    """
    NO hace merges. Añade 'y_pred_estacional' directamente en una copia de df_pred
    usando mapeo por mes y reescalado global para preservar el total anual.
    """
    out = df_pred.copy()
    out["month"] = out["date"].dt.month

    # mapeo de mes -> factor (suma 1.0); uniforme = 1/12
    factor_map = factores.set_index("month")["factor"].to_dict()
    uniforme = 1.0 / 12.0

    mult = out["month"].map(lambda m: factor_map.get(int(m), uniforme) / uniforme)
    out["y_est_tmp"] = out["y_pred"] * mult

    total_before = float(out["y_pred"].sum())
    total_after_tmp = float(out["y_est_tmp"].sum())
    scale = 1.0 if total_after_tmp == 0 else (total_before / total_after_tmp)

    out["y_pred_estacional"] = out["y_est_tmp"] * scale

    # limpieza de auxiliares
    out = out.drop(columns=["month", "y_est_tmp"])
    total_after = float(out["y_pred_estacional"].sum())
    return out, total_before, total_after

def _control_totales(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    m_before = (df_before.assign(month=df_before["date"].dt.month)
                .groupby("month", as_index=False)["y_pred"].sum()
                .rename(columns={"y_pred": "total_before"}))
    m_after  = (df_after.assign(month=df_after["date"].dt.month)
                .groupby("month", as_index=False)["y_pred_estacional"].sum()
                .rename(columns={"y_pred_estacional": "total_after"}))
    ctrl = m_before.merge(m_after, on="month", how="outer").sort_values("month")
    total_row = pd.DataFrame({
        "month": ["TOTAL"],
        "total_before": [m_before["total_before"].sum()],
        "total_after":  [m_after["total_after"].sum()],
    })
    return pd.concat([ctrl, total_row], ignore_index=True)


# --- Pipeline principal -------------------------------------------------
def main():
    log.info("== Estacionalidad: inicio ==")

    log.info("Leyendo históricos: %s", HIST_CSV)
    df_hist = _leer_historicos(HIST_CSV)

    factores = _calcular_factores(df_hist)
    factores.to_csv(OUT_FACTORES_CSV, index=False)
    log.info("Guardado factores: %s", OUT_FACTORES_CSV)

    log.info("Leyendo predicciones base: %s", PRED_BASE_PARQUET)
    df_pred = _leer_predicciones(PRED_BASE_PARQUET)

    # Inyección SIN MERGE
    df_out, total_before, total_after = _inyectar_estacionalidad_en_linea(df_pred, factores)

    # Control mensuales + total
    ctrl = _control_totales(df_pred, df_out)
    ctrl.to_csv(OUT_CONTROL_CSV, index=False)
    log.info("Guardado control: %s", OUT_CONTROL_CSV)

    # Guardar parquet final
    df_out.to_parquet(OUT_PARQUET, index=False)
    log.info("Guardado parquet final: %s", OUT_PARQUET)

    log.info("Totals check — antes: %.2f | después: %.2f | diff: %.6f",
             total_before, total_after, total_before - total_after)
    log.info("== Estacionalidad: fin ✅ ==")


if __name__ == "__main__":
    main()
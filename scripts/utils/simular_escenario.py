
# =============================================================================
# Script: simular_escenario.py
# Descripción:
#   Utilidad genérica para construir escenarios "optimista" y "pesimista"
#   a partir de un baseline estacionalizado, usando métricas de backtesting
#   (WAPE medio por clúster).
#
# Flujo del pipeline:
# 1) Carga baseline y métricas de backtesting
# 2) Cálculo de factores por clúster según escenario
# 3) Aplicación de factores al baseline
# 4) Generación de controles (totales globales y por clúster)
#
# Input:
#   - Baseline: parquet/csv/xlsx con columnas mínimas [date, cluster, forecast_sum]
#   - Métricas: CSV con columnas [cluster, WAPE_%] (subapartado 8.5)
#
# Output:
#   - DataFrame ajustado con forecast_sum modificado
#   - CSVs de control si se llama a generar_controles()
#
# Dependencias:
#   - pandas
#   - numpy
#
# Instalación rápida:
#   pip install pandas numpy pyarrow
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]   # utils está dos niveles por debajo
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ==== 1. IMPORTS + LOGGING ====================================================
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    """Crea directorios si no existen."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def _ensure_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Levanta error si faltan columnas requeridas."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] faltan columnas requeridas: {missing}")

# ==== 3. LÓGICA PRINCIPAL =====================================================
def load_baseline(path: Path, date_col: str = "date",
                  cluster_col: str = "cluster", yhat_col: str = "forecast_sum") -> pd.DataFrame:
    """Carga el baseline desde parquet/csv/xlsx y valida columnas mínimas."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Extensión no soportada: {path.suffix}")
    _ensure_columns(df, [date_col, cluster_col, yhat_col], "baseline")
    return df

def load_metrics(path: Path, cluster_col: str = "cluster",
                 wape_col: str = "WAPE_%", agg: str = "mean") -> pd.DataFrame:
    """Carga métricas de backtesting y devuelve WAPE agregado por clúster."""
    df = pd.read_csv(path)
    _ensure_columns(df, [cluster_col, wape_col], "metrics")
    grouped = df.groupby(cluster_col, as_index=False)[wape_col].agg(agg)
    return grouped

def calcular_factores(metrics_by_cluster: pd.DataFrame, escenario: str,
                      cluster_col: str = "cluster", wape_col: str = "WAPE_%",
                      min_factor: float = 0.0) -> pd.DataFrame:
    """
    Construye factores por clúster:
    - optimista: 1 + WAPE_%/100
    - pesimista: 1 - WAPE_%/100 (mínimo = min_factor)
    """
    _ensure_columns(metrics_by_cluster, [cluster_col, wape_col], "metrics_by_cluster")
    df = metrics_by_cluster.copy()

    if escenario == "optimista":
        df["factor"] = 1 + (df[wape_col] / 100)
    elif escenario == "pesimista":
        df["factor"] = 1 - (df[wape_col] / 100)
        df["factor"] = df["factor"].clip(lower=min_factor)
    else:
        raise ValueError("Escenario debe ser 'optimista' o 'pesimista'.")

    return df[[cluster_col, "factor"]]

def aplicar_factores(baseline: pd.DataFrame, factors: pd.DataFrame,
                     on: list[str] = ["cluster"], yhat_col: str = "forecast_sum") -> pd.DataFrame:
    """Aplica factores multiplicativos al baseline y devuelve escenario ajustado."""
    out = baseline.merge(factors, on=on, how="left", validate="m:1")
    if "factor" not in out.columns:
        raise ValueError("No se encontró la columna 'factor' tras unir factores.")
    out[yhat_col] = out[yhat_col] * out["factor"]
    return out.drop(columns="factor")

def validate_structure(df: pd.DataFrame, date_col: str = "date",
                       cluster_col: str = "cluster", yhat_col: str = "forecast_sum",
                       expect_year: int | None = None) -> None:
    """Valida estructura, NaN, negativos y año esperado opcional."""
    _ensure_columns(df, [date_col, cluster_col, yhat_col], "escenario")
    if df[yhat_col].isna().any():
        raise ValueError("Existen NaN en forecast_sum tras el ajuste.")
    if (df[yhat_col] < 0).any():
        raise ValueError("Se han generado valores negativos.")
    if expect_year is not None:
        years = set(pd.to_datetime(df[date_col]).dt.year.unique())
        if years != {expect_year}:
            raise ValueError(f"Año inesperado en fechas: {years}")

# ==== 4. EXPORTACIÓN / I/O OPCIONAL ==========================================
def generar_controles(df_base: pd.DataFrame, df_scen: pd.DataFrame,
                      out_dir: Path, escenario: str,
                      cluster_col: str = "cluster", yhat_col: str = "forecast_sum") -> None:
    """Genera CSVs de control (totales globales y por clúster)."""
    ensure_dirs(out_dir)

    # Global
    total_base = float(df_base[yhat_col].sum())
    total_scen = float(df_scen[yhat_col].sum())
    pd.DataFrame({
        "escenario": [escenario],
        "total_base": [total_base],
        "total_scen": [total_scen],
        "variacion_%": [(total_scen - total_base) / max(total_base, 1e-9) * 100]
    }).to_csv(out_dir / f"control_totales_{escenario}.csv", index=False)

    # Por clúster
    base_c = df_base.groupby(cluster_col)[yhat_col].sum().reset_index(name="total_base")
    scen_c = df_scen.groupby(cluster_col)[yhat_col].sum().reset_index(name="total_scen")
    merged = base_c.merge(scen_c, on=cluster_col, how="outer").fillna(0)
    merged["variacion_%"] = (merged["total_scen"] - merged["total_base"]) / merged["total_base"].replace(0, np.nan) * 100
    merged.to_csv(out_dir / f"control_por_cluster_{escenario}.csv", index=False)

# ==== 5. CLI / MAIN ===========================================================
# Este módulo es una utilidad genérica, no ejecuta nada por sí mismo.

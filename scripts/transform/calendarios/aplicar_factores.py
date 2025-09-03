# =============================================================================
# Script: aplicar_factores.py
# Descripción:
#   Lee demanda_price_adjusted.parquet y calendar_externos.parquet,
#   aplica multiplicadores por factor (no precio) respetando ventanas/scope,
#   genera demanda final (sin y con ruido) y trazabilidad.
#
# Flujo:
#   1) Cargar base (Fase 4): Product_ID, Date, Cluster, Demand_Day, ...
#   2) Cargar y expandir calendar_externos a granularidad diaria
#   3) Construir columnas M_* por fila (producto/día) según scope
#   4) Demand_Final = Demand_Day * ∏ M_k | Factors_Applied
#   5) Añadir ruido lognormal media-preservada -> Demand_Final_Noised
#   6) Exportar parquet final + tabla de chequeos
#
# Input:
#   data/processed/demanda_price_adjusted.parquet
#   outputs/tables/calendar_externos.parquet
#
# Output:
#   data/processed/demanda_all_adjusted.parquet
#   outputs/tables/calendar_total.parquet (expandido diario)
#
# Dependencias: pandas, numpy, pyarrow
# =============================================================================

from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# ==== 0. CONFIG (RUTAS BASE) =================================================
try:
    ROOT_DIR = Path(__file__).resolve().parents[2]
except NameError:
    ROOT_DIR = Path(r"C:\...\PFM2_Asistente_Compras_Inteligente")

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "tables"

BASE_PARQUET = PROCESSED_DIR / "demanda_price_adjusted.parquet"
CALENDAR_PARQUET = OUTPUTS_DIR / "calendar_externos.parquet"
OUT_PARQUET = PROCESSED_DIR / "demanda_all_adjusted.parquet"
CAL_TOTAL_PARQUET = OUTPUTS_DIR / "calendar_total.parquet"

# Ruido (puedes ajustar por cluster)
SIGMA_BY_CLUSTER = {"default": 0.05, "0": 0.04, "1": 0.05, "2": 0.06, "3": 0.05}

# ==== 1. LOGGING ==============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("aplicar_factores")

# ==== 2. UTILIDADES ==========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs: d.mkdir(parents=True, exist_ok=True)

def _read_parquet(path: Path, cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=cols)

def expand_calendar_daily(cal: pd.DataFrame) -> pd.DataFrame:
    """
    Expande [start,end] a una fila por día; mantiene factor_type, lift y scope.
    """
    rows = []
    for _, r in cal.iterrows():
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            rows.append({"Date": d.date(), **{k: r[k] for k in cal.columns if k not in ("start","end")}})
    return pd.DataFrame(rows)

def merge_scope(df: pd.DataFrame, cal_daily: pd.DataFrame, factor: str) -> pd.Series:
    """
    Devuelve una Serie con el multiplicador para 'factor' por fila (producto/día).
    Scope admitido: global (1), cluster (usa df['Cluster']), product_id (usa df['Product_ID']).
    """
    sub = cal_daily[cal_daily["factor_type"] == factor]
    if sub.empty:
        return pd.Series(1.0, index=df.index, name=f"M_{factor}")
    # por simplicidad inicial: solo global (scope vacío). Extensible a cluster/product.
    m = pd.merge(
        df[["Date"]].assign(_ix=np.arange(len(df))),
        sub[["Date","lift","scope_type","scope_values"]],
        on="Date", how="left"
    )
    s = m["lift"].fillna(1.0).astype(float)
    s.index = df.index
    s.name = f"M_{factor}"
    return s

def combine_factors(df: pd.DataFrame, m_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """
    Calcula Demand_Final y Factors_Applied a partir de Demand_Day y m_cols.
    """
    mult = df[m_cols].prod(axis=1)
    demand_final = df["Demand_Day"] * mult
    applied = df[m_cols].apply(lambda r: "|".join(sorted([c.replace("M_","") for c,v in r.items() if v != 1.0])) or "None", axis=1)
    return demand_final, applied

def apply_noise_mean_preserving(x: pd.Series, sigma: pd.Series, seed: int = 42) -> pd.Series:
    """
    Ruido lognormal media-preservada: mu = -0.5*sigma^2
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=len(x))
    mu = -0.5 * (sigma ** 2)
    m_noise = np.exp(mu + sigma * eps)
    return np.ceil(np.maximum(0.0, x * m_noise))

def sigma_by_cluster(series_cluster: pd.Series) -> pd.Series:
    return series_cluster.astype(str).map(SIGMA_BY_CLUSTER).fillna(SIGMA_BY_CLUSTER["default"]).astype(float)

# ==== 3. LÓGICA PRINCIPAL ====================================================
def run() -> pd.DataFrame:
    log.info("Cargando base F4: %s", BASE_PARQUET)
    base = _read_parquet(BASE_PARQUET)
    # Normalizaciones de columnas esperadas
    ren = {c: "Demand_Day" for c in base.columns if str(c).lower() in ("demand_day","sales_quantity")}
    if ren: base = base.rename(columns=ren)
    base["Date"] = pd.to_datetime(base["Date"]).dt.date

    log.info("Cargando calendario externos: %s", CALENDAR_PARQUET)
    cal = _read_parquet(CALENDAR_PARQUET)
    cal["start"] = pd.to_datetime(cal["start"]).dt.date
    cal["end"]   = pd.to_datetime(cal["end"]).dt.date
    cal_daily = expand_calendar_daily(cal)
    cal_daily.to_parquet(CAL_TOTAL_PARQUET, index=False)

    # === construir multiplicadores M_* ===
    factors = sorted(cal["factor_type"].unique().tolist())
    for f in factors:
        base[f"M_{f}"] = merge_scope(base, cal_daily, f)

    # Si quieres columnas con nombres "amables":
    rename_map = {
        "M_inflacion_cpi": "M_inflation",
        "M_marketing_push": "M_promo",
        "M_season_extra": "M_seasonExtra",
        "M_agosto_nonprice": "M_agosto_nonprice",
        "M_competition_high": "M_competition",
    }
    base = base.rename(columns={k:v for k,v in rename_map.items() if k in base.columns})

    m_cols = [c for c in base.columns if c.startswith("M_")]
    for c in ["M_inflation","M_promo","M_seasonExtra","M_competition","M_agosto_nonprice","M_segments"]:
        if c not in base.columns:
            base[c] = 1.0  # asegurar presencia

    # Demanda final (sin ruido)
    m_cols_final = ["M_inflation","M_promo","M_seasonExtra","M_competition","M_agosto_nonprice","M_segments"]
    base["Demand_Final"], base["Factors_Applied"] = combine_factors(base, m_cols_final)

    # Ruido al final (también fuera de ventanas)
    cluster_col = "Cluster" if "Cluster" in base.columns else None
    sigmas = sigma_by_cluster(base[cluster_col]) if cluster_col else pd.Series(SIGMA_BY_CLUSTER["default"], index=base.index)
    base["Demand_Final_Noised"] = apply_noise_mean_preserving(base["Demand_Final"], sigmas, seed=2025)

    return base

# ==== 4. EXPORTACIÓN / I/O ====================================================
def save(df: pd.DataFrame) -> None:
    ensure_dirs(PROCESSED_DIR, OUTPUTS_DIR)
    df.to_parquet(OUT_PARQUET, index=False)

# ==== 5. CLI / MAIN ==========================================================
def main():
    df = run()
    save(df)
    logging.info("Demanda ajustada exportada a: %s", OUT_PARQUET)

if __name__ == "__main__":
    main()

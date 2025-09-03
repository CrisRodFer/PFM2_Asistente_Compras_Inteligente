# =============================================================================
# Script: aplicar_factores.py
# Descripción:
#   Lee demanda_price_adjusted.parquet y calendar_externos_completo.parquet,
#   aplica multiplicadores por factor respetando ventanas/scope,
#   genera demanda final (sin y con ruido) y trazabilidad.
#
# Entradas:
#   data/processed/demanda_price_adjusted.parquet
#   outputs/tables/calendar_externos_completo.parquet
#
# Salidas:
#   data/processed/demanda_all_adjusted.parquet
#   outputs/tables/calendar_total_externos.parquet
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
    ROOT_DIR = Path(r"C:\Users\crisr\Desktop\Máster Data Science & IA\PROYECTO\PFM2_Asistente_Compras_Inteligente")

DATA_DIR       = ROOT_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
OUTPUTS_DIR    = ROOT_DIR / "outputs" / "tables"

BASE_PARQUET         = PROCESSED_DIR / "demanda_price_adjusted.parquet"
CALENDAR_PARQUET     = OUTPUTS_DIR / "calendar_externos_completo.parquet"  # ✅ enriquecido
OUT_PARQUET          = PROCESSED_DIR / "demanda_all_adjusted.parquet"
CAL_TOTAL_PARQUET    = OUTPUTS_DIR / "calendar_total_externos.parquet"

SIGMA_BY_CLUSTER = {"default": 0.05, "0": 0.04, "1": 0.05, "2": 0.06, "3": 0.05}

# ==== 1. LOGGING ==============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("aplicar_factores")

# ==== 2. UTILIDADES ==========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def _read_parquet(path: Path, cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=cols)

def expand_calendar_daily(cal: pd.DataFrame) -> pd.DataFrame:
    if cal.empty:
        return pd.DataFrame(columns=["Date","factor_type","lift","scope_type","scope_values"])
    rows = []
    for _, r in cal.iterrows():
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            rows.append({
                "Date": d.normalize(),
                "factor_type": r["factor_type"],
                "lift": float(r["lift"]),
                "scope_type": r.get("scope_type", "global"),
                "scope_values": r.get("scope_values", "")
            })
    return pd.DataFrame(rows)

def merge_scope(df: pd.DataFrame, cal_daily: pd.DataFrame, factor: str) -> pd.Series:
    sub = cal_daily[cal_daily["factor_type"] == factor].copy()
    if sub.empty:
        return pd.Series(1.0, index=df.index, name=f"M_{factor}")
    sub = sub.groupby("Date", as_index=False)["lift"].max()
    m = pd.merge(
        df[["Date"]].assign(_ix=np.arange(len(df))),
        sub[["Date","lift"]],
        on="Date", how="left"
    )
    s = m["lift"].fillna(1.0).astype(float)
    s.index = df.index
    s.name = f"M_{factor}"
    return s

def combine_factors(df: pd.DataFrame, m_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    mult = df[m_cols].prod(axis=1)
    demand_final = df["Demand_Day"] * mult
    applied = df[m_cols].apply(
        lambda r: "|".join(sorted([c.replace("M_","") for c, v in r.items() if v != 1.0])) or "None",
        axis=1
    )
    return demand_final, applied

def apply_noise_mean_preserving(x: pd.Series, sigma: pd.Series, seed: int = 2025) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=len(x))
    mu = -0.5 * (sigma ** 2)
    m_noise = np.exp(mu + sigma * eps)
    return np.ceil(np.maximum(0.0, x * m_noise))

def sigma_by_cluster(series_cluster: pd.Series) -> pd.Series:
    return series_cluster.astype(str).map(SIGMA_BY_CLUSTER).fillna(SIGMA_BY_CLUSTER["default"]).astype(float)

# ==== 3. LÓGICA PRINCIPAL ====================================================
def run() -> pd.DataFrame:
    log.info("📂 Cargando base F4...")
    base = _read_parquet(BASE_PARQUET)
    base["Date"] = pd.to_datetime(base["Date"]).dt.normalize()

    if "Demand_Day" not in base.columns:
        for c in base.columns:
            if str(c).lower() in ("demand_day","sales_quantity"):
                base = base.rename(columns={c:"Demand_Day"})
                break

    log.info("📂 Cargando calendario enriquecido...")
    cal = _read_parquet(CALENDAR_PARQUET)
    cal["start"] = pd.to_datetime(cal["start"]).dt.normalize()
    cal["end"]   = pd.to_datetime(cal["end"]).dt.normalize()
    cal_daily = expand_calendar_daily(cal)
    cal_daily.to_parquet(CAL_TOTAL_PARQUET, index=False)
    log.info("📅 Calendario expandido y guardado.")

    log.info("⚙️  Aplicando factores...")
    factors = sorted(cal["factor_type"].unique().tolist())
    for f in factors:
        base[f"M_{f}"] = merge_scope(base, cal_daily, f)

    rename_map = {
        "M_inflacion_cpi":   "M_inflation",
        "M_marketing_push":  "M_promo",
        "M_season_extra":    "M_seasonExtra",
        "M_agosto_nonprice": "M_agosto_nonprice",
        "M_competition_high":"M_competition",
    }
    base.rename(columns={k:v for k,v in rename_map.items() if k in base.columns}, inplace=True)

    m_cols_final = ["M_inflation","M_promo","M_seasonExtra","M_competition","M_agosto_nonprice","M_segments"]
    for c in m_cols_final:
        if c not in base.columns:
            base[c] = 1.0

    log.info("🧮 Calculando demanda final (sin ruido)...")
    base["Demand_Final"], base["Factors_Applied"] = combine_factors(base, m_cols_final)

    log.info("🎲 Añadiendo ruido media-preservada...")
    cluster_col = "Cluster" if "Cluster" in base.columns else None
    sigmas = sigma_by_cluster(base[cluster_col]) if cluster_col else pd.Series(SIGMA_BY_CLUSTER["default"], index=base.index)
    base["Demand_Final_Noised"] = apply_noise_mean_preserving(base["Demand_Final"], sigmas, seed=2025)

    log.info("✅ Demanda final calculada.")
    return base

# ==== 4. EXPORTACIÓN =========================================================
def save(df: pd.DataFrame) -> None:
    ensure_dirs(PROCESSED_DIR, OUTPUTS_DIR)
    df.to_parquet(OUT_PARQUET, index=False)
    log.info(f"💾 Archivo guardado correctamente en: {OUT_PARQUET}")

# ==== 5. CLI / MAIN ==========================================================
def main():
    log.info("🚀 Iniciando pipeline de aplicación de factores...")
    df = run()
    save(df)
    log.info("🎉 Proceso completado con éxito.")

if __name__ == "__main__":
    main()


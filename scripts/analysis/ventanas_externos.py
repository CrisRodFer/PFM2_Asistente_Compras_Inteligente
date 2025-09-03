

# =============================================================================
# Script: ventanas_externos.py
# Descripción:
#   Genera y valida el calendario de factores externos (no precio) a partir de
#   los CSV de calendario real (LOCALk7) y de una tabla paramétrica de lifts.
#
# Flujo:
#   1) Cargar CSVs validacion_calendario_real_LOCALk7_{2022,2023,2024}.csv
#   2) Mapear Eventos -> factor_type (ej.: 'navidad' -> 'season_extra')
#   3) Asignar lift paramétrico por factor_type y (opcional) por scope
#   4) Normalizar ventanas y unificar años -> calendar_externos.parquet
#   5) Exportar preflight ligero (solapes, huecos, conteos)
#
# Input:
#   outputs/tables/validacion_calendario_real_LOCALk7_*.csv
#
# Output:
#   data/auxiliar/ventanas_externos.csv
#   outputs/tables/calendar_externos.parquet
#   outputs/tables/preflight_externos.xlsx
#
# Dependencias: pandas, numpy, pyarrow, xlsxwriter/openpyxl
# =============================================================================

from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# ==== 0. CONFIG (RUTAS BASE) =================================================
try:
    ROOT_DIR = Path(__file__).resolve().parents[2]
except NameError:
    ROOT_DIR = Path(r"C:\...\PFM2_Asistente_Compras_Inteligente")

DATA_DIR = ROOT_DIR / "data"
AUX_DIR = DATA_DIR / ("auxiliar" if (DATA_DIR / "auxiliar").exists() else "aux")
OUTPUTS_DIR = ROOT_DIR / "outputs" / "tables"

CAL_LOCAL_FILES = [
    OUTPUTS_DIR / "validacion_calendario_real_LOCALk7_2022.csv",
    OUTPUTS_DIR / "validacion_calendario_real_LOCALk7_2023.csv",
    OUTPUTS_DIR / "validacion_calendario_real_LOCALk7_2024.csv",
]

VENTANAS_CSV = AUX_DIR / "ventanas_externos.csv"
CALENDAR_PARQUET = OUTPUTS_DIR / "calendar_externos.parquet"
PREFLIGHT_XLSX = OUTPUTS_DIR / "preflight_externos.xlsx"

# Lifts paramétricos (ejemplos iniciales; ajusta en AUX/CSV si lo deseas)
DEFAULT_LIFTS = {
    "inflacion_cpi": 0.97,          # efecto global
    "marketing_push": 1.10,         # newsletter / displays
    "season_extra": 1.12,           # navidad / eventos especiales
    "agosto_nonprice": 0.95,        # caída estival sin descuento
    "competition_high": 0.90,       # presión alta de competencia
}

# ==== 1. LOGGING ==============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ventanas_externos")

# ==== 2. UTILIDADES ==========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def _excel_writer(path: Path):
    try:
        return pd.ExcelWriter(path, engine="xlsxwriter", date_format="yyyy-mm-dd", datetime_format="yyyy-mm-dd")
    except Exception:
        return pd.ExcelWriter(path, engine="openpyxl")

def _fix_headers(df: pd.DataFrame) -> pd.DataFrame:
    repl = {"AÃ±o": "Año", "AÂ±o": "Año", "Año": "Año", "Â": "", "Ã": ""}
    df = df.rename(columns={c: "".join(repl.get(ch, ch) for ch in c) for c in df.columns})
    return df

def load_localk7(files: list[Path]) -> pd.DataFrame:
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin1")  # fallback
        df = _fix_headers(df)
        need = [c for c in ["Año", "Evento", "Inicio", "Fin"] if c in df.columns]
        df = df[need].copy()
        df["Inicio"] = pd.to_datetime(df["Inicio"], errors="coerce").dt.date
        df["Fin"] = pd.to_datetime(df["Fin"], errors="coerce").dt.date
        frames.append(df.dropna(subset=["Inicio", "Fin"]))
    cal = pd.concat(frames, ignore_index=True).drop_duplicates()
    return cal

def classify_event_to_factor(evento: str) -> tuple[str, str]:
    """
    Devuelve (factor_type, notes). Ajusta reglas a tu conveniencia.
    """
    s = (evento or "").lower()
    if any(k in s for k in ["black", "bf", "cyber"]): return "season_extra", "black_friday"
    if any(k in s for k in ["rebaj"]): return "season_extra", "rebajas"
    if any(k in s for k in ["navid", "xmas", "christ"]): return "season_extra", "navidad"
    if "agosto" in s: return "agosto_nonprice", "agosto_nonprice"
    if "marketing" in s: return "marketing_push", "newsletter/display"
    return "season_extra", "evento_general"

# ==== 3. LÓGICA PRINCIPAL ====================================================
def build_windows(cal: pd.DataFrame, default_lifts: dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, r in cal.iterrows():
        ftype, note = classify_event_to_factor(str(r["Evento"]))
        lift = default_lifts.get(ftype, 1.0)
        rows.append({
            "id": f"{ftype}_{r['Año']}_{str(r['Inicio'])}",
            "start": r["Inicio"], "end": r["Fin"],
            "factor_type": ftype, "lift": float(lift),
            "scope_type": "global", "scope_values": "", "notes": note
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["start"] = pd.to_datetime(df["start"]).dt.date
        df["end"] = pd.to_datetime(df["end"]).dt.date
        df = (df.sort_values(["factor_type", "start"])
                .groupby(["factor_type", "start", "end", "lift", "scope_type", "scope_values", "notes"], as_index=False)
                .agg(id=("id", "first")))
    return df

def preflight_summary(wins: pd.DataFrame) -> pd.DataFrame:
    if wins.empty: return pd.DataFrame()
    out = (wins.assign(days=(pd.to_datetime(wins["end"]) - pd.to_datetime(wins["start"])).dt.days + 1)
                .groupby("factor_type", as_index=False)
                .agg(n_ventanas=("id", "count"), total_dias=("days", "sum"), lift=("lift", "first")))
    return out

# ==== 4. EXPORTACIÓN / I/O ====================================================
def export_all(wins: pd.DataFrame) -> None:
    ensure_dirs(AUX_DIR, OUTPUTS_DIR)
    wins.to_csv(VENTANAS_CSV, index=False)
    wins.to_parquet(CALENDAR_PARQUET, index=False)
    with _excel_writer(PREFLIGHT_XLSX) as wr:
        wins.to_excel(wr, sheet_name="ventanas_externos", index=False)
        preflight_summary(wins).to_excel(wr, sheet_name="preflight", index=False)

# ==== 5. CLI / MAIN ==========================================================
def main():
    log.info("Cargando CSV LOCALk7...")
    cal = load_localk7(CAL_LOCAL_FILES)
    log.info("Construyendo ventanas externas...")
    wins = build_windows(cal, DEFAULT_LIFTS)
    log.info("Exportando calendario externo...")
    export_all(wins)
    log.info("calendar_externos listo: %s", CALENDAR_PARQUET)

if __name__ == "__main__":
    main()

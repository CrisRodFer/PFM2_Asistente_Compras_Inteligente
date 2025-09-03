
# =============================================================================
# Script: enriquecer_ventanas_externos.py
# Descripción:
#   - Carga ventanas_externos.csv (salida de ventanas_externos.py)
#   - Consolida solapes intra-factor (un solo lift por día/factor; se toma el máx)
#   - Añade factores faltantes de forma paramétrica:
#       * Inflación (mensual)
#       * Marketing push (varias campañas al año)
#       * Competencia alta (periodos anuales)
#   - Reconstruye ventanas continuas y exporta calendario enriquecido
#
# Entradas:
#   data/auxiliar/ventanas_externos.csv
#
# Salidas:
#   data/auxiliar/ventanas_externos_completo.csv
#   outputs/tables/calendar_externos_completo.parquet
#   outputs/tables/preflight_externos_completo.xlsx
#
# Dependencias: pandas, numpy, pyarrow, xlsxwriter/openpyxl
# =============================================================================

from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ==== 0) CONFIG (rutas) ======================================================
PROJECT_ROOT = Path(r"C:\Users\crisr\Desktop\Máster Data Science & IA\PROYECTO\PFM2_Asistente_Compras_Inteligente")
AUX_DIR      = PROJECT_ROOT / "data" / "auxiliar"
OUT_DIR      = PROJECT_ROOT / "outputs" / "tables"

IN_VENTANAS              = AUX_DIR / "ventanas_externos.csv"
OUT_VENTANAS_COMPLETO    = AUX_DIR / "ventanas_externos_completo.csv"
OUT_CALENDAR_COMPLETO    = OUT_DIR / "calendar_externos_completo.parquet"
OUT_PREFLIGHT_COMPLETO   = OUT_DIR / "preflight_externos_completo.xlsx"

AUX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 0b) CONFIG (lifts y reglas paramétricas) ===============================
LIFT_INFLACION   = 0.97
LIFT_MARKETING   = 1.10
LIFT_COMPETENCIA = 0.90

# Campañas de marketing por año (mes, día_inicio, día_fin)
MARKETING_WINDOWS = [(4, 10, 20), (9, 10, 20)]  # abril y septiembre aprox

# Periodos de competencia por año (mes, día_inicio, día_fin)
COMPETENCIA_WINDOWS = [(6, 1, 14)]  # primera quincena de junio

# ==== 1) LOGGING =============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("enriquecer_externos")


# ==== 2) UTILS ===============================================================
def _excel_writer(path: Path):
    try:
        return pd.ExcelWriter(path, engine="xlsxwriter", date_format="yyyy-mm-dd", datetime_format="yyyy-mm-dd")
    except Exception:
        return pd.ExcelWriter(path, engine="openpyxl")

def expand_windows_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Expande [start,end] a granularidad diaria."""
    rows = []
    for _, r in df.iterrows():
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            rows.append({"Date": d.normalize(), "factor_type": r["factor_type"], "lift": float(r["lift"])})
    return pd.DataFrame(rows)

def compress_daily_to_windows(daily: pd.DataFrame) -> pd.DataFrame:
    """Convierte granularidad diaria -> ventanas continuas por (factor_type, lift)."""
    if daily.empty:
        return pd.DataFrame(columns=["id","start","end","factor_type","lift","scope_type","scope_values","notes"])
    out_rows = []
    for ftype, g in daily.sort_values(["factor_type","Date"]).groupby(["factor_type","lift"]):
        factor, lift = ftype
        dates = g["Date"].dt.date.tolist()
        if not dates: continue
        # recorrer run-length de fechas consecutivas
        start = prev = dates[0]
        for cur in dates[1:]:
            if (cur - prev) != timedelta(days=1):
                out_rows.append({"factor_type": factor, "lift": float(lift), "start": start, "end": prev})
                start = cur
            prev = cur
        out_rows.append({"factor_type": factor, "lift": float(lift), "start": start, "end": prev})
    wins = pd.DataFrame(out_rows)
    if wins.empty:
        return pd.DataFrame(columns=["id","start","end","factor_type","lift","scope_type","scope_values","notes"])
    wins["id"] = (
        wins["factor_type"].astype(str)
        + "_"
        + wins["start"].astype(str)
    )
    wins["scope_type"] = "global"
    wins["scope_values"] = ""
    wins["notes"] = ""
    return wins[["id","start","end","factor_type","lift","scope_type","scope_values","notes"]]

def month_start_end(d: date) -> tuple[date, date]:
    start = date(d.year, d.month, 1)
    if d.month == 12:
        end = date(d.year, 12, 31)
    else:
        end = date(d.year, d.month + 1, 1) - timedelta(days=1)
    return start, end

def generate_monthly_windows(start_date: date, end_date: date, factor_type: str, lift: float) -> pd.DataFrame:
    months = []
    cur = date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        ms, me = month_start_end(cur)
        ms = max(ms, start_date)
        me = min(me, end_date)
        months.append({"id": f"{factor_type}_{ms}", "start": ms, "end": me,
                       "factor_type": factor_type, "lift": float(lift),
                       "scope_type":"global","scope_values":"","notes":"monthly"})
        # siguiente mes
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return pd.DataFrame(months)

def generate_yearly_fixed_windows(start_date: date, end_date: date, tuples_mdy: list[tuple[int,int,int]], factor_type: str, lift: float, notes: str) -> pd.DataFrame:
    """tuples_mdy: [(mes, dia_inicio, dia_fin), ...]"""
    rows = []
    for y in range(start_date.year, end_date.year + 1):
        for (m, d1, d2) in tuples_mdy:
            try:
                s = date(y, m, d1)
                e = date(y, m, d2)
            except ValueError:
                continue
            if e < start_date or s > end_date: 
                continue
            s = max(s, start_date)
            e = min(e, end_date)
            rows.append({"id": f"{factor_type}_{s}", "start": s, "end": e,
                         "factor_type": factor_type, "lift": float(lift),
                         "scope_type":"global","scope_values":"","notes":notes})
    return pd.DataFrame(rows)


# ==== 3) MAIN ================================================================
def main():
    # 3.1 Leer ventanas originales
    if not IN_VENTANAS.exists():
        raise FileNotFoundError(f"No existe {IN_VENTANAS}. Genera primero ventanas_externos.csv.")
    try:
        base = pd.read_csv(IN_VENTANAS, parse_dates=["start","end"])
    except UnicodeDecodeError:
        base = pd.read_csv(IN_VENTANAS, encoding="latin1", parse_dates=["start","end"])
    base["start"] = pd.to_datetime(base["start"]).dt.normalize()
    base["end"]   = pd.to_datetime(base["end"]).dt.normalize()

    log.info("Cargado ventanas_externos.csv -> %s filas, factores: %s",
             base.shape[0], ", ".join(sorted(base["factor_type"].unique())))

    # 3.2 Expandir a diario y CONSOLIDAR intra-factor por día (máx lift)
    daily = expand_windows_daily(base)
    if daily.empty:
        raise ValueError("No hay ventanas para expandir.")
    consolidated = (daily.groupby(["Date","factor_type"], as_index=False)
                         .agg(lift=("lift","max")))

    # 3.3 Definir rango temporal global (para generar paramétricos)
    start_date = consolidated["Date"].min().date()
    end_date   = consolidated["Date"].max().date()
    log.info("Rango detectado: %s -> %s", start_date, end_date)

    # 3.4 Generar ventanas PARAMÉTRICAS (inflación, marketing, competencia)
    infl = generate_monthly_windows(start_date, end_date, "inflacion_cpi", LIFT_INFLACION)
    mkt  = generate_yearly_fixed_windows(start_date, end_date, MARKETING_WINDOWS, "marketing_push", LIFT_MARKETING, "campaign")
    comp = generate_yearly_fixed_windows(start_date, end_date, COMPETENCIA_WINDOWS, "competition_high", LIFT_COMPETENCIA, "competitor_pressure")

    # 3.5 Unir todo en diario y volver a CONSOLIDAR (máx lift por día/factor)
    add_dfs = []
    for df in [infl, mkt, comp]:
        if not df.empty:
            add_dfs.append(expand_windows_daily(df))
    if add_dfs:
        daily_full = pd.concat([consolidated.rename(columns={"Date":"Date"}), *add_dfs], ignore_index=True)
    else:
        daily_full = consolidated.copy()
    daily_full = (daily_full.groupby(["Date","factor_type"], as_index=False)
                           .agg(lift=("lift","max")))

    # 3.6 Comprimir a ventanas continuas
    wins_full = compress_daily_to_windows(daily_full)

    # 3.7 Exportar
    wins_full.to_csv(OUT_VENTANAS_COMPLETO, index=False)
    wins_full.to_parquet(OUT_CALENDAR_COMPLETO, index=False)
    with _excel_writer(OUT_PREFLIGHT_COMPLETO) as wr:
        wins_full.to_excel(wr, sheet_name="ventanas_externos_completo", index=False)
        (wins_full.assign(days=(pd.to_datetime(wins_full["end"]) - pd.to_datetime(wins_full["start"])).dt.days + 1)
                 .groupby("factor_type", as_index=False)
                 .agg(n_ventanas=("id","count"), total_dias=("days","sum"), lift=("lift","first"))
        ).to_excel(wr, sheet_name="preflight", index=False)

    log.info("OK -> %s | %s | %s", OUT_VENTANAS_COMPLETO, OUT_CALENDAR_COMPLETO, OUT_PREFLIGHT_COMPLETO)
    log.info("Resumen por factor:\n%s", wins_full["factor_type"].value_counts())

if __name__ == "__main__":
    main()

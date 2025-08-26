# =============================================================================
# Script: validacion_calendario_real.py
# Versión: Iteración 4 (final) — igual a Notebook
# Descripción:
#   - Baseline LOCAL ±k días (con fallback mensual)
#   - Ventanas corridas ±shift (mejor alineación de picos/vales)
#
# Entradas:
#   data/processed/demanda_diaria_YYYY.csv  (Date, Product, Demand_Day)
#
# Salidas (unificadas con el Notebook):
#   outputs/tables/validacion_calendario_real_SHIFT_localk{k}_s{shift}_{YYYY}.csv
#   outputs/tables/validacion_calendario_real_kpis_SHIFT_k{k}_s{shift}.csv
#   outputs/figures/evolucion_2024_con_eventos_SHIFT_k{k}_s{shift}.png
# =============================================================================

from __future__ import annotations

# ==== 0) Rutas base ===========================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]      # project-root/
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"              # inputs
OUTPUTS_DIR = ROOT_DIR / "outputs"                  # outputs (Notebook + script)

# ==== 1) Imports + logging ====================================================
import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validacion_calendario_real")

# ==== 2) Constantes de dominio ===============================================
EVENTO_ESPERADO = {
    "Rebajas Invierno":"pico", "Rebajas Verano":"pico", "San Valentín":"pico",
    "Black Friday":"pico", "Cyber Monday":"pico", "Prime Day":"pico",
    "Navidad":"pico", "Vuelta al cole":"pico",
    "Agosto":"valle",
    "Semana Santa":"mixto", "Festivo Nacional":"mixto",
}
THR_PICO = +0.05
THR_VALLE = -0.05
MIN_OBS_BASE = 3  # mín. días para baseline

# ==== 3) Utilidades de rutas ==================================================
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

# ==== 4) Utilidades de fechas =================================================
def easter_sunday(y: int) -> date:
    a = y % 19; b = y // 100; c = y % 100
    d = b // 4; e = b % 4; f = (b + 8) // 25
    g = (b - f + 1) // 3; h = (19*a + b - d - g + 15) % 30
    i = c // 4; k = c % 4; l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = 1 + ((h + l - 7*m + 114) % 31)
    return date(y, month, day)

def nth_weekday_of_month(y:int, m:int, weekday:int, n:int)->date:
    d = date(y, m, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    d += timedelta(days=7*(n-1))
    return d

def black_friday(y:int)->date:
    return nth_weekday_of_month(y, 11, weekday=4, n=4)

def cyber_monday(y:int)->date:
    return black_friday(y) + timedelta(days=3)

def prime_day_pair(y:int)->tuple[date, date]:
    tue = nth_weekday_of_month(y, 7, weekday=1, n=2)
    return tue, tue + timedelta(days=1)

# ==== 5) Calendario real ======================================================
def calendario_real(y:int) -> pd.DataFrame:
    E = easter_sunday(y)
    jueves_santo = E - timedelta(days=3)
    viernes_santo = E - timedelta(days=2)
    bf = black_friday(y); cm = cyber_monday(y)
    pd1, pd2 = prime_day_pair(y)

    eventos = []
    def add(name, start, end=None):
        if end is None: end = start
        for d in pd.date_range(start, end, freq="D").date:
            eventos.append((name, d))

    # Festivos nacionales (no regionales)
    for mm, dd in [(1,1),(1,6),(5,1),(8,15),(10,12),(11,1),(12,6),(12,8),(12,25)]:
        add("Festivo Nacional", date(y, mm, dd))

    # Semana Santa
    add("Semana Santa", jueves_santo); add("Semana Santa", viernes_santo)

    # Campañas ecommerce
    add("Rebajas Invierno", date(y,1,7), date(y,1,31))
    add("Rebajas Verano",  date(y,7,1), date(y,7,31))
    add("San Valentín",    date(y,2,14))
    add("Black Friday",    bf); add("Cyber Monday", cm)
    add("Prime Day",       pd1); add("Prime Day", pd2)
    add("Navidad",         date(y,12,20), date(y,12,31))
    add("Vuelta al cole",  date(y,9,1), date(y,9,15))
    add("Agosto",          date(y,8,1), date(y,8,31))

    cal = pd.DataFrame(eventos, columns=["Evento","Date"]).drop_duplicates()
    cal["Date"] = pd.to_datetime(cal["Date"])
    cal["Year"] = y
    return cal

# ==== 6) Carga ================================================================
def load_total(inp_dir: Path, y:int)->pd.DataFrame:
    f = inp_dir / f"demanda_diaria_{y}.csv"
    df = pd.read_csv(f, parse_dates=["Date"])
    df = df.groupby("Date", as_index=False)["Demand_Day"].sum().rename(
        columns={"Demand_Day":"Demand_Total"}
    )
    df["Year"] = y
    df["Month"] = df["Date"].dt.month
    return df.sort_values("Date")

# ==== 7) Baselines y métricas ================================================
def baseline_local(df: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp, k:int) -> float | None:
    left = d0 - pd.Timedelta(days=k)
    right = d1 + pd.Timedelta(days=k)
    m_window = (df["Date"] >= left) & (df["Date"] <= right)
    m_event  = (df["Date"] >= d0) & (df["Date"] <= d1)
    vals = df.loc[m_window & (~m_event), "Demand_Total"]
    if vals.shape[0] >= MIN_OBS_BASE:
        return float(vals.mean())
    return None

def baseline_mensual_excluyendo(df: pd.DataFrame, d_series: pd.Series) -> float | None:
    meses = d_series.dt.month.unique()
    df_mes = df[df["Month"].isin(meses)]
    vals = df_mes[~df_mes["Date"].isin(d_series)]["Demand_Total"]
    return float(vals.mean()) if vals.shape[0] >= MIN_OBS_BASE else None

def uplift_evento(df: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp, k:int) -> tuple[float | None, float | None]:
    dr = pd.date_range(d0, d1, freq="D")
    media_ev = df[df["Date"].isin(dr)]["Demand_Total"].mean()
    base = baseline_local(df, d0, d1, k=k)
    if base is None:
        base = baseline_mensual_excluyendo(df, pd.Series(dr))
    return (None if pd.isna(media_ev) else float(media_ev)), (None if base is None else float(base))

# ==== 8) Evaluación con ventanas ±shift ======================================
def evaluar_eventos_shift(df: pd.DataFrame, cal: pd.DataFrame, k_local:int, shift_max:int, year:int)->pd.DataFrame:
    rows = []
    for ev, g in cal.groupby("Evento"):
        d_series = pd.to_datetime(g["Date"].sort_values())
        d0_orig, d1_orig = d_series.min(), d_series.max()
        esperado = EVENTO_ESPERADO.get(ev, "mixto")

        best = {"shift": 0, "uplift": np.nan, "media_ev": None, "base": None,
                "inicio": d0_orig, "fin": d1_orig}

        for s in range(-shift_max, shift_max+1):
            d0 = d0_orig + pd.Timedelta(days=s)
            d1 = d1_orig + pd.Timedelta(days=s)
            media_ev, base = uplift_evento(df, d0, d1, k=k_local)
            uplift = (media_ev - base) / base if (media_ev is not None and base is not None and base>0) else np.nan
            if np.isnan(uplift):
                continue
            if esperado == "pico":
                cond = (np.isnan(best["uplift"]) or uplift > best["uplift"])
            elif esperado == "valle":
                cond = (np.isnan(best["uplift"]) or uplift < best["uplift"])
            else:
                cond = (np.isnan(best["uplift"]) or abs(uplift) > abs(best["uplift"]))
            if cond:
                best.update({"shift": s, "uplift": uplift, "media_ev": media_ev, "base": base,
                             "inicio": d0, "fin": d1})

        uplift_pct = None if np.isnan(best["uplift"]) else round(100*best["uplift"], 2)
        pasa = None
        if not np.isnan(best["uplift"]):
            if esperado == "pico":
                pasa = (best["uplift"] >= THR_PICO)
            elif esperado == "valle":
                pasa = (best["uplift"] <= THR_VALLE)

        rows.append({
            "Año": year,
            "Evento": ev,
            "Esperado": esperado,
            "Shift_dias": int(best["shift"]),
            "Inicio_shift": best["inicio"].date(),
            "Fin_shift": best["fin"].date(),
            "MediaEvento": None if best["media_ev"] is None else round(best["media_ev"], 2),
            "MediaBase_Local": None if best["base"] is None else round(best["base"], 2),
            "Uplift%": uplift_pct,
            "Pasa": (bool(pasa) if pasa is not None else None),
            "k_local": k_local,
            "shift_max": shift_max
        })

    return pd.DataFrame(rows).sort_values(["Año","Evento"]).reset_index(drop=True)

# ==== 9) KPI ==================================================================
def kpi_por_anio(valid: pd.DataFrame)->pd.DataFrame:
    kpi = (valid[valid["Esperado"].isin(["pico","valle"])]
           .groupby("Año")["Pasa"]
           .apply(lambda s: round(100*float((s==True).mean()), 1))
           .rename("%Eventos_OK").reset_index())
    return kpi

# ==== 10) Main ================================================================
def run(years:list[int], k_local:int, shift:int, inp_dir:Path, out_dir:Path, fig_2024:bool=True)->pd.DataFrame:
    out_tables = out_dir / "tables"
    out_figs   = out_dir / "figures"
    ensure_dirs(out_tables, out_figs)

    todos = []
    for y in years:
        df = load_total(inp_dir, y)
        cal = calendario_real(y)
        res = evaluar_eventos_shift(df, cal, k_local=k_local, shift_max=shift, year=y)
        out_y = out_tables / f"validacion_calendario_real_SHIFT_localk{k_local}_s{shift}_{y}.csv"
        res.to_csv(out_y, index=False)
        log.info("Guardado: %s", out_y)
        todos.append(res)

    valid = pd.concat(todos, ignore_index=True)
    kpi_df = kpi_por_anio(valid)
    kpi_df.rename(columns={"%Eventos_OK": f"%Eventos_OK_SHIFT(k{k_local},±{shift})"}, inplace=True)
    kpi_path = out_tables / f"validacion_calendario_real_kpis_SHIFT_k{k_local}_s{shift}.csv"
    kpi_df.to_csv(kpi_path, index=False)
    log.info("KPI guardado: %s", kpi_path)

    # Figura 2024 (opcional)
    if fig_2024 and (2024 in years):
        try:
            df24 = load_total(inp_dir, 2024)
            sel = valid[valid["Año"]==2024]
            fig, ax = plt.subplots(figsize=(14, 4.5))
            ax.plot(df24["Date"], df24["Demand_Total"], lw=1.2, label="2024")
            for _, r in sel.iterrows():
                if r["Evento"] in {"Rebajas Verano","Navidad","Agosto","Vuelta al cole"}:
                    s = pd.to_datetime(r["Inicio_shift"]); e = pd.to_datetime(r["Fin_shift"])
                    ax.axvspan(s, e, alpha=0.08, color="tab:orange")
            ax.set_title(f"Evolución diaria 2024 con ventanas desplazadas (k={k_local}, ±{shift})")
            ax.set_ylabel("Demanda total diaria"); ax.grid(True, alpha=.2)
            plt.tight_layout()
            out_fig = out_figs / f"evolucion_2024_con_eventos_SHIFT_k{k_local}_s{shift}.png"
            plt.savefig(out_fig, dpi=150); plt.close()
            log.info("Figura guardada: %s", out_fig)
        except Exception as e:
            log.warning("No se generó figura 2024: %s", e)

    return kpi_df

def parse_args()->argparse.Namespace:
    p = argparse.ArgumentParser(description="Validación con calendario real (LOCAL ±k + ventanas ±shift)")
    p.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--shift", type=int, default=3)
    p.add_argument("--in", dest="inp", type=str, default=str(PROCESSED_DIR))
    p.add_argument("--out", dest="out", type=str, default=str(OUTPUTS_DIR))
    p.add_argument("--no-fig-2024", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inp_dir = Path(args.inp)
    out_dir = Path(args.out)
    ensure_dirs(out_dir)
    log.info("Entrada: %s", inp_dir)
    log.info("Salida:  %s", out_dir)
    kpi = run(args.years, args.k, args.shift, inp_dir, out_dir, fig_2024=not args.no_fig_2024)
    log.info("\nResumen KPI:\n%s", kpi.to_string(index=False))



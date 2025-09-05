# =============================================================================
# Script: clasificar_picos_aislados.py
# Descripción:
#   FASE 6.1 (extensión): Cruza los días candidatos de productos clasificados
#   como "pico_aislado" con el CALENDARIO REAL (2022-2024) para separar:
#     - picos justificados por evento (rebajas, BF, Navidad, etc.) -> mantener
#     - picos no justificados -> candidatos a suavizado/capping
#
# Entradas por defecto:
#   - reports/outliers/outliers_candidatos_nuevos_dias.csv
#   - reports/outliers/outliers_candidatos_nuevos_productos.csv
#   - outputs/tables/validacion_calendario_real_LOCALk7_20*.csv  (se busca recursivo)
#
# Salidas:
#   - reports/outliers/outliers_picos_aislados_justificados.csv
#   - reports/outliers/outliers_picos_aislados_no_justificados.csv
#   - reports/outliers/outliers_picos_aislados_resumen_calendario.csv
#
# Uso:
#   python scripts/analysis/clasificar_picos_aislados.py
#   # o indicando carpeta/patrón:
#   python scripts/analysis/clasificar_picos_aislados.py --calendar-dir "outputs/tables" --calendar-pattern "validacion_calendario_real_LOCALk7_20*.csv"
# =============================================================================

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd

# ==== 0. CONFIG (RUTAS) =======================================================
ROOT_DIR = Path(__file__).resolve().parents[2]              # raíz del repo
REPORTS_DIR = ROOT_DIR / "reports" / "outliers"
DATA_DIR = ROOT_DIR / "data"

# Por defecto, los calendarios están en outputs/tables (según tu proyecto)
DEFAULT_CALENDAR_DIR = ROOT_DIR / "outputs" / "tables"
DEFAULT_CALENDAR_PATTERN = "validacion_calendario_real_LOCALk7_20*.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(Path(__file__).stem)

# ==== 1. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip() for c in df.columns})

def _read_outliers(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dias_fp = report_dir / "outliers_candidatos_nuevos_dias.csv"
    prod_fp = report_dir / "outliers_candidatos_nuevos_productos.csv"
    if not dias_fp.exists() or not prod_fp.exists():
        raise FileNotFoundError(f"No encuentro {dias_fp} o {prod_fp}. Ejecuta antes 6.1 (identificar_nuevos_outliers.py).")
    df_dias = pd.read_csv(dias_fp, parse_dates=["date"])
    df_prod = pd.read_csv(prod_fp)
    df_dias.columns = [c.strip().lower() for c in df_dias.columns]
    df_prod.columns = [c.strip().lower() for c in df_prod.columns]
    if "year" not in df_dias.columns:
        df_dias["year"] = pd.to_datetime(df_dias["date"]).dt.year
    return df_dias, df_prod

def _read_calendars(calendar_dir: Path, calendar_pattern: str) -> pd.DataFrame:
    """Carga y concatena calendarios anuales desde calendar_dir (búsqueda recursiva por pattern)."""
    paths = list(calendar_dir.rglob(calendar_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No se encontraron calendarios con patrón '{calendar_pattern}' debajo de: {calendar_dir}"
        )
    frames = []
    for p in sorted(paths):
        dfc = pd.read_csv(p)
        dfc = _norm(dfc).rename(columns={"Año": "year", "Evento": "evento", "Inicio": "inicio", "Fin": "fin"})
        dfc["inicio"] = pd.to_datetime(dfc["inicio"], errors="coerce")
        dfc["fin"] = pd.to_datetime(dfc["fin"], errors="coerce")
        dfc["year"] = pd.to_numeric(dfc["year"], errors="coerce").astype("Int64")
        dfc["origen_fichero"] = p.as_posix()
        frames.append(dfc.dropna(subset=["inicio", "fin"]))
    cal = pd.concat(frames, ignore_index=True)
    return cal

def expand_calendar_rows(cal: pd.DataFrame) -> pd.DataFrame:
    """Convierte cada [inicio, fin] en fechas diarias (date, year, evento)."""
    out_list = []
    for _, r in cal.iterrows():
        dates = pd.date_range(start=r["inicio"].date(), end=r["fin"].date(), freq="D")
        if len(dates):
            out_list.append(pd.DataFrame({"date": dates, "year": r["year"], "evento": r["evento"]}))
    return pd.concat(out_list, ignore_index=True) if out_list else pd.DataFrame(columns=["date", "year", "evento"])

def merge_outliers_calendar(df_dias_pi: pd.DataFrame, cal_days: pd.DataFrame) -> pd.DataFrame:
    """Cruza días candidatos (pico_aislado) con calendario por (date, year)."""
    m = df_dias_pi.merge(cal_days, how="left", on=["date", "year"])
    evento_agg = (
        m.groupby(["product_id", "year", "date"], as_index=False)["evento"]
         .apply(lambda s: "; ".join(sorted(set([e for e in s.dropna().astype(str) if e.strip()]))))
    )
    evento_agg.rename(columns={"evento": "evento_calendario"}, inplace=True)
    base = df_dias_pi.merge(evento_agg, on=["product_id", "year", "date"], how="left")
    base["justificado"] = base["evento_calendario"].fillna("").str.len() > 0
    return base

# ==== 2. EJECUCIÓN ============================================================
def ejecutar(
    report_dir: Path = REPORTS_DIR,
    calendar_dir: Path = DEFAULT_CALENDAR_DIR,
    calendar_pattern: str = DEFAULT_CALENDAR_PATTERN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1) Outliers (días y productos)
    df_dias, df_prod = _read_outliers(report_dir)

    # 2) Solo productos-año 'pico_aislado'
    pi = df_prod[df_prod["tipo_outlier"].str.lower() == "pico_aislado"].copy()
    if pi.empty:
        log.warning("No hay productos clasificados como 'pico_aislado'.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_dias_pi = df_dias.merge(pi[["product_id", "year"]].drop_duplicates(), on=["product_id", "year"], how="inner")
    if df_dias_pi.empty:
        log.warning("No hay días candidatos asociados a 'pico_aislado'.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 3) Calendarios
    cal = _read_calendars(calendar_dir, calendar_pattern)
    cal_days = expand_calendar_rows(cal)

    # 4) Cruce
    df_pi_all = merge_outliers_calendar(df_dias_pi, cal_days)

    # 5) Split y resumen
    df_just = df_pi_all[df_pi_all["justificado"]].copy().sort_values(["year", "product_id", "date"])
    df_noju = df_pi_all[~df_pi_all["justificado"]].copy().sort_values(["year", "product_id", "date"])
    resumen_evento = (
        df_pi_all.assign(evento=lambda d: d["evento_calendario"].fillna("∅"))
                 .groupby(["year", "justificado", "evento"], dropna=False)
                 .size()
                 .reset_index(name="conteo")
                 .sort_values(["year", "justificado", "conteo"], ascending=[True, False, False])
    )

    # 6) Export
    ensure_dirs(report_dir)
    (report_dir / "outliers_picos_aislados_justificados.csv").write_text("")  # touch seguro Windows (por si permisos)
    (report_dir / "outliers_picos_aislados_no_justificados.csv").write_text("")
    (report_dir / "outliers_picos_aislados_resumen_calendario.csv").write_text("")
    df_just.to_csv(report_dir / "outliers_picos_aislados_justificados.csv", index=False)
    df_noju.to_csv(report_dir / "outliers_picos_aislados_no_justificados.csv", index=False)
    resumen_evento.to_csv(report_dir / "outliers_picos_aislados_resumen_calendario.csv", index=False)

    log.info("Guardados en %s", report_dir)
    return df_pi_all, df_just, df_noju

# ==== 3. CLI ==================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE 6.1 – Cruce de picos aislados con calendario real")
    p.add_argument("--report-dir", type=str, default=str(REPORTS_DIR), help="Carpeta reports/outliers")
    p.add_argument("--calendar-dir", type=str, default=str(DEFAULT_CALENDAR_DIR), help="Carpeta donde están los CSV de calendario")
    p.add_argument("--calendar-pattern", type=str, default=DEFAULT_CALENDAR_PATTERN, help="Patrón glob para los CSV de calendario")
    return p.parse_args()

def main() -> None:
    a = parse_args()
    try:
        ejecutar(report_dir=Path(a.report_dir), calendar_dir=Path(a.calendar_dir), calendar_pattern=a.calendar_pattern)
    except Exception as e:
        log.exception("Error en clasificar_picos_aislados: %s", e)
        raise

if __name__ == "__main__":
    main()

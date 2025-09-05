# =============================================================================
# Script: clasificar_picos_aislados_dbscan.py
# Descripción:
#   FASE 6.2 (extensión): Cruza los días candidatos de productos clasificados
#   como "pico_aislado" dentro de los OUTLIERS DBSCAN (is_outlier=1) con el
#   CALENDARIO REAL (2022-2024), para separar:
#     - picos justificados por evento (rebajas, BF, Navidad, etc.) -> mantener
#     - picos no justificados -> candidatos a suavizado/capping
#
# Entradas por defecto:
#   - reports/outliers/outliers_dbscan_dias.csv
#   - reports/outliers/outliers_dbscan_productos.csv
#   - outputs/tables/validacion_calendario_real_LOCALk7_20*.csv  (búsqueda recursiva)
#
# Salidas:
#   - reports/outliers/outliers_dbscan_picos_justificados.csv
#   - reports/outliers/outliers_dbscan_picos_no_justificados.csv
#   - reports/outliers/outliers_dbscan_picos_resumen_calendario.csv
#
# Uso:
#   python scripts/analysis/cruzar_dbscan_picos_con_calendario.py
#   # o indicando carpeta/patrón de calendarios:
#   python scripts/analysis/cruzar_dbscan_picos_con_calendario.py ^
#     --calendar-dir "outputs/tables" ^
#     --calendar-pattern "validacion_calendario_real_LOCALk7_20*.csv"
#
# Dependencias:
#   pip install pandas numpy
# =============================================================================

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import pandas as pd

# ==== 0. CONFIG (RUTAS) =======================================================
# Este archivo está pensado para ubicarse en scripts/analysis/
ROOT_DIR = Path(__file__).resolve().parents[2]  # raíz del repo
REPORTS_DIR = ROOT_DIR / "reports" / "outliers"

# Por defecto, los calendarios están en outputs/tables (según tu proyecto)
DEFAULT_CALENDAR_DIR = ROOT_DIR / "outputs" / "tables"
DEFAULT_CALENDAR_PATTERN = "validacion_calendario_real_LOCALk7_20*.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(Path(__file__).stem)

# ==== 1. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df

def _read_dbscan_outliers(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee los ficheros generados por la clasificación de DBSCAN:
      - outliers_dbscan_dias.csv  (nivel día, con flags)
      - outliers_dbscan_productos.csv  (nivel producto-año, con tipo_outlier)
    """
    dias_fp = report_dir / "outliers_dbscan_dias.csv"
    prod_fp = report_dir / "outliers_dbscan_productos.csv"
    if not dias_fp.exists() or not prod_fp.exists():
        raise FileNotFoundError(
            f"No encuentro {dias_fp} o {prod_fp}. Ejecuta antes el script de 6.2 (clasificar_outliers_dbscan.py)."
        )
    df_dias = pd.read_csv(dias_fp, parse_dates=["date"])
    df_prod = pd.read_csv(prod_fp)
    df_dias = _norm_cols(df_dias)
    df_prod = _norm_cols(df_prod)
    # asegurar tipos básicos
    if "year" not in df_dias.columns:
        df_dias["year"] = pd.to_datetime(df_dias["date"]).dt.year
    return df_dias, df_prod

def _read_calendars(calendar_dir: Path, calendar_pattern: str) -> pd.DataFrame:
    """
    Carga y concatena calendarios anuales desde calendar_dir (búsqueda recursiva por pattern).
    Se esperan columnas: 'Año','Evento','Inicio','Fin' (otras columnas se preservan si existen).
    """
    paths = list(calendar_dir.rglob(calendar_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No se encontraron calendarios con patrón '{calendar_pattern}' debajo de: {calendar_dir}"
        )
    frames = []
    for p in sorted(paths):
        dfc = pd.read_csv(p)
        dfc = _norm_cols(dfc)
        # Renombrar a estándar
        m = {"Año": "year", "Evento": "evento", "Inicio": "inicio", "Fin": "fin"}
        for k, v in m.items():
            if k not in dfc.columns:
                raise KeyError(f"El calendario '{p.name}' no contiene la columna requerida '{k}'.")
        dfc = dfc.rename(columns=m)
        # Tipos
        dfc["inicio"] = pd.to_datetime(dfc["inicio"], errors="coerce")
        dfc["fin"] = pd.to_datetime(dfc["fin"], errors="coerce")
        dfc["year"] = pd.to_numeric(dfc["year"], errors="coerce").astype("Int64")
        dfc["origen_fichero"] = p.as_posix()
        dfc = dfc.dropna(subset=["inicio", "fin"])
        frames.append(dfc)
    cal = pd.concat(frames, ignore_index=True)
    return cal

def expand_calendar_rows(cal: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte cada intervalo [inicio, fin] en fechas diarias individuales.
    Devuelve columnas: date, year, evento
    """
    out_list = []
    for _, r in cal.iterrows():
        rng = pd.date_range(start=r["inicio"].date(), end=r["fin"].date(), freq="D")
        if len(rng):
            out_list.append(pd.DataFrame({"date": rng, "year": r["year"], "evento": r["evento"]}))
    if not out_list:
        return pd.DataFrame(columns=["date", "year", "evento"])
    return pd.concat(out_list, ignore_index=True)

def merge_outliers_calendar(df_dias_pi: pd.DataFrame, cal_days: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza días candidatos (solo pico_aislado) con calendario por (date, year).
    Agrega posibles múltiples eventos por fecha.
    """
    m = df_dias_pi.merge(cal_days, how="left", on=["date", "year"])
    evento_agg = (
        m.groupby(["product_id", "year", "date"], as_index=False)["evento"]
         .apply(lambda s: "; ".join(sorted(set([e for e in s.dropna().astype(str) if e.strip()]))))
    ).rename(columns={"evento": "evento_calendario"})
    base = df_dias_pi.merge(evento_agg, on=["product_id", "year", "date"], how="left")
    base["justificado"] = base["evento_calendario"].fillna("").str.len() > 0
    return base

# ==== 2. EJECUCIÓN ============================================================
def ejecutar(report_dir: Path,
             calendar_dir: Path,
             calendar_pattern: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el cruce de picos aislados (DBSCAN) con calendario real.
    Retorna:
      - df_all: días candidatos de productos 'pico_aislado' con etiqueta justificado
      - df_just: subset justificado
      - df_noju: subset no justificado
    """
    # 1) Cargar outliers (DBSCAN)
    df_dias, df_prod = _read_dbscan_outliers(report_dir)

    # 2) Filtrar SOLO productos-año tipo 'pico_aislado'
    if "tipo_outlier" not in df_prod.columns:
        raise KeyError("El archivo de productos DBSCAN no tiene 'tipo_outlier'.")
    pi = df_prod[df_prod["tipo_outlier"].str.lower() == "pico_aislado"].copy()
    if pi.empty:
        log.warning("No hay productos clasificados como 'pico_aislado' en DBSCAN. Nada que cruzar.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    key_cols = ["product_id", "year"]
    df_dias_pi = df_dias.merge(pi[key_cols].drop_duplicates(), on=key_cols, how="inner")
    if df_dias_pi.empty:
        log.warning("No hay días candidatos asociados a productos 'pico_aislado' (DBSCAN).")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 3) Cargar y expandir calendarios
    cal = _read_calendars(calendar_dir, calendar_pattern)
    cal_days = expand_calendar_rows(cal)

    # 4) Cruce por fecha
    df_all = merge_outliers_calendar(df_dias_pi, cal_days)
    df_just = df_all[df_all["justificado"]].copy().sort_values(["year", "product_id", "date"])
    df_noju = df_all[~df_all["justificado"]].copy().sort_values(["year", "product_id", "date"])

    # 5) Resumen por año y evento
    resumen_evento = (
        df_all.assign(evento=lambda d: d["evento_calendario"].fillna("∅"))
              .groupby(["year", "justificado", "evento"], dropna=False)
              .size()
              .reset_index(name="conteo")
              .sort_values(["year", "justificado", "conteo"], ascending=[True, False, False])
    )

    # 6) Export
    ensure_dirs(report_dir)
    (report_dir / "outliers_dbscan_picos_justificados.csv").write_text("")  # touch seguro
    (report_dir / "outliers_dbscan_picos_no_justificados.csv").write_text("")
    (report_dir / "outliers_dbscan_picos_resumen_calendario.csv").write_text("")
    df_just.to_csv(report_dir / "outliers_dbscan_picos_justificados.csv", index=False)
    df_noju.to_csv(report_dir / "outliers_dbscan_picos_no_justificados.csv", index=False)
    resumen_evento.to_csv(report_dir / "outliers_dbscan_picos_resumen_calendario.csv", index=False)

    log.info("Guardados en %s", report_dir)
    return df_all, df_just, df_noju

# ==== 3. CLI ==================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE 6.2 – Cruce de picos aislados (DBSCAN) con calendario real")
    p.add_argument("--report-dir", type=str, default=str(REPORTS_DIR),
                   help="Carpeta reports/outliers con outliers_dbscan_*.csv")
    p.add_argument("--calendar-dir", type=str, default=str(DEFAULT_CALENDAR_DIR),
                   help="Carpeta donde están los CSV de calendario (por defecto outputs/tables)")
    p.add_argument("--calendar-pattern", type=str, default=DEFAULT_CALENDAR_PATTERN,
                   help="Patrón glob para localizar los CSV de calendario")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    try:
        ejecutar(report_dir=Path(args.report_dir),
                 calendar_dir=Path(args.calendar_dir),
                 calendar_pattern=args.calendar_pattern)
    except Exception as e:
        log.exception("Error en cruce DBSCAN picos con calendario: %s", e)
        raise

if __name__ == "__main__":
    main()

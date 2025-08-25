# scripts/transform/desagregar_demanda_2024.py
# =============================================================================
# Desagrega Historico(s)_Ventas_2024.csv aplicando el calendario estacional 2024.
# Guarda la salida en data/processed/demanda_diaria_2024.csv
# Incluye validaciones: calendario, filas esperadas, duplicados, NaN/negativos y masa.
# =============================================================================

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------- Bootstrapping robusto (funciona en subcarpetas) ----------
def find_project_root(start: Path, marker="scripts", max_up: int = 8) -> Path:
    p = start.resolve()
    for _ in range(max_up):
        if (p / marker).is_dir():
            return p
        p = p.parent
    raise RuntimeError(f"No se encontró la carpeta '{marker}' hacia arriba.")

ROOT = find_project_root(Path(__file__).parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.desagregar_demanda_anual import desagregar_demanda_anual
from scripts.utils.validar_calendario_estacional import validar_calendario_estacional

# ---------- Parámetros ----------
YEAR = 2024
DATA_DIR     = ROOT / "data" / "clean"
PROC_DIR     = ROOT / "data" / "processed"
CATALOG_PATH = DATA_DIR / "Catalogo_Productos_Limpio.xlsx"  # ajusta si procede

HIST_CANDIDATES = [
    DATA_DIR / f"Historico_Ventas_{YEAR}.csv",
    DATA_DIR / f"Historicos_Ventas_{YEAR}.csv",
]
CAL_PATH = ROOT / "outputs" / f"calendario_estacional_{YEAR}.csv"
OUT_PATH = PROC_DIR / f"demanda_diaria_{YEAR}.csv"

# ---------- Utilidades ----------
def _pick_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró el histórico 2024. Probados:\n  - " + "\n  - ".join(map(str, paths)))

def _normalize_bool(x) -> bool:
    if isinstance(x, bool): 
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    if s in {"true","verdadero","sí","si","1","y","yes"}:  return True
    if s in {"false","falso","no","0","n"}:                return False
    return False

def _canon_id(s: pd.Series) -> pd.Series:
    """Convierte IDs a string canónica (sin espacios ni sufijo '.0')."""
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
    )

def _read_universe_from_catalog(path: Path) -> pd.DataFrame:
    # Cols candidatas por si cambian nombres
    prod_cols = ["Product_ID","product_id","PRODUCT_ID"]
    nov_cols  = ["novedades","Novedades","is_new","IsNew","EsNovedad"]

    try:
        cat = pd.read_excel(path)  # requiere openpyxl
    except Exception as e:
        raise RuntimeError(f"No se pudo leer el catálogo '{path}': {e}")

    prod_col = next((c for c in prod_cols if c in cat.columns), None)
    nov_col  = next((c for c in nov_cols  if c in cat.columns), None)
    if prod_col is None or nov_col is None:
        raise KeyError(f"En el catálogo deben existir columnas de ID ({prod_cols}) y 'novedades' ({nov_cols}).")

    mask_no_new = ~cat[nov_col].map(_normalize_bool)
    ids = (cat.loc[mask_no_new, prod_col].dropna())
    universo = pd.DataFrame({"Product_ID": _canon_id(ids).unique()})
    universo = universo.dropna().astype({"Product_ID": "string"}).sort_values("Product_ID").reset_index(drop=True)
    return universo

def _report_checks(df_daily: pd.DataFrame,
                   n_products_expected: int,
                   cal_df: pd.DataFrame,
                   *,
                   id_col="Product_ID", date_col="Date", out_col="Demand_Day",
                   totales_full: pd.DataFrame | None = None,
                   year_col="Year", qty_name="Sales Quantity") -> None:
    """QA visible: filas esperadas, duplicados, NaN/negativos, (opcional) masa."""
    days_in_year = cal_df[date_col].dt.date.nunique()
    expected = n_products_expected * days_in_year
    actual   = len(df_daily)
    print(f"   - Filas esperadas: {expected:,} | obtenidas: {actual:,}")
    if actual != expected:
        raise AssertionError("El nº de filas no coincide con productos×días.")

    dupes = int(df_daily.duplicated(subset=[id_col, date_col]).sum())
    nan_dd = int(df_daily[out_col].isna().sum())
    neg_dd = int((df_daily[out_col] < 0).sum())
    print(f"   - Duplicados (Product_ID, Date): {dupes}")
    print(f"   - NaN en {out_col}: {nan_dd} | Negativos: {neg_dd}")
    if dupes or nan_dd or neg_dd:
        raise AssertionError("Se detectaron duplicados o valores inválidos en la salida.")

    if totales_full is not None:
        agg_daily = (df_daily.assign(Year=df_daily[date_col].dt.year)
                            .groupby([id_col, "Year"], as_index=False)[out_col].sum()
                            .rename(columns={out_col: "Total_Diario"}))
        comp = (totales_full.rename(columns={qty_name: "Total_Year"})
                         .merge(agg_daily, on=[id_col, "Year"], how="left")
                         .fillna(0))
        max_err = float((comp["Total_Year"] - comp["Total_Diario"]).abs().max())
        print(f"   - Conservación de masa (máx abs err): {max_err:.3e}")
        if max_err > 1e-9:
            raise AssertionError(f"Conservación de masa fallida (máx err={max_err:.3e}).")

# ---------- Main ----------
def main():
    print(f"\n🧩 DESAGREGACIÓN DEMANDA · {YEAR}")

    hist_path = _pick_existing(HIST_CANDIDATES)
    if not CAL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el calendario: {CAL_PATH}")
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"No se encontró el catálogo: {CATALOG_PATH}")

    print(f"• Histórico  : {hist_path}")
    print(f"• Calendario : {CAL_PATH}")
    print(f"• Catálogo   : {CATALOG_PATH}")

    # 1) Validar calendario
    print("\n🔎 Validando calendario…")
    cal_res = validar_calendario_estacional(str(CAL_PATH), year=YEAR, verbose=True)
    if not cal_res["ok"]:
        print("❌ Calendario inválido. Abortando.")
        raise SystemExit(1)
    print("✅ Calendario OK.")

    # 2) Leer histórico y agregar totales por (producto, año)
    hist = pd.read_csv(hist_path)
    hist["Product_ID"] = _canon_id(hist["Product_ID"]).astype("string")   # ← normaliza ID
    hist["Date"] = pd.to_datetime(hist["Date"], errors="raise")
    hist["Year"] = hist["Date"].dt.year.astype(int)

    totales = (
        hist.groupby(["Product_ID", "Year"], as_index=False)["Sales Quantity"].sum()
    )
    totales["Product_ID"] = _canon_id(totales["Product_ID"]).astype("string")
    totales["Year"] = totales["Year"].astype(int)

    # 3) Universo desde catálogo: NO-NOVEDADES
    universo = _read_universe_from_catalog(CATALOG_PATH)   # Product_ID en string
    n_universo = len(universo)
    print(f"📚 Universo (no-novedades) = {n_universo:,} productos")

    # 4) Completar totales 2024 con 0 para ausentes
    grid = universo.assign(Year=int(YEAR))  # Year int
    totales_full = grid.merge(
        totales, on=["Product_ID", "Year"], how="left", validate="one_to_one"
    )
    totales_full["Sales Quantity"] = pd.to_numeric(
        totales_full["Sales Quantity"], errors="coerce"
    ).fillna(0.0)

    added_zeros = int((totales_full["Sales Quantity"] == 0).sum() - (totales["Sales Quantity"] == 0).sum())
    print(f"🔧 Productos añadidos con 0 ventas en {YEAR}: {added_zeros:,}")

    # 5) Leer calendario
    cal_df = pd.read_csv(CAL_PATH, parse_dates=["Date"])

    # 6) Desagregar usando los totales completados
    print("\n⚙️  Desagregando…")
    df_daily = desagregar_demanda_anual(
        df_demanda_anual=totales_full,
        calendario_estacional=cal_df,
        id_col="Product_ID",
        year_col="Year",
        qty_col="Sales Quantity",
        date_col="Date",
        weight_col="Peso Normalizado",
        out_col="Demand_Day",
        tol=1e-9,
        check_mass=True,
    )

    # 7) QA visible
    print("\n✅ Validaciones de salida:")
    _report_checks(df_daily, n_products_expected=n_universo, cal_df=cal_df,
                   id_col="Product_ID", date_col="Date", out_col="Demand_Day",
                   totales_full=totales_full, year_col="Year", qty_name="Sales Quantity")

    # 8) Guardar
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(OUT_PATH, index=False)
    print(f"\n💾 Guardado: {OUT_PATH}")
    print(f"📊 Resumen: filas={len(df_daily):,} | productos={df_daily['Product_ID'].nunique():,}")

if __name__ == "__main__":
    main()
# scripts/transform/desagregar_demanda_2022.py
# =============================================================================
# Desagrega Historico_Ventas_2022.csv aplicando el calendario estacional 2022.
# Guarda la salida en data/processed/demanda_diaria_2022.csv
# Incluye validaciones visibles: calendario, filas esperadas, duplicados, NaN/negativos y masa.
# =============================================================================

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- Bootstrapping: ejecutar directamente con "Run Python File" en VSCode
ROOT = Path(__file__).resolve().parents[2]  # .../scripts/transform -> raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.desagregar_demanda_anual import desagregar_demanda_anual
from scripts.utils.validar_calendario_estacional import validar_calendario_estacional

YEAR = 2022
DATA_DIR = ROOT / "data" / "clean"
PROC_DIR = ROOT / "data" / "processed"

# Histórico: acepta ambas variantes de nombre
HIST_CANDIDATES = [
    DATA_DIR / f"Historico_Ventas_{YEAR}.csv",
    DATA_DIR / f"Historicos_Ventas_{YEAR}.csv",
]

# Calendario: usa dónde lo tengas ahora mismo (ajusta si lo mueves a data/interim)
CAL_PATH = ROOT / "outputs" / f"calendario_estacional_{YEAR}.csv"
OUT_PATH = PROC_DIR / f"demanda_diaria_{YEAR}.csv"

def _pick_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró el histórico 2022. Probados:\n  - " + "\n  - ".join(map(str, paths)))

def _report_checks(df_daily: pd.DataFrame,
                   totales: pd.DataFrame,
                   cal_df: pd.DataFrame,
                   *,
                   id_col="Product_ID", date_col="Date", year_col="Year",
                   qty_name="Sales Quantity", out_col="Demand_Day") -> None:
    """QA visible: filas esperadas, duplicados, NaN/negativos, masa."""
    # días por año en calendario y nº de productos por año en los totales
    days_per_year = (cal_df.assign(Year=cal_df[date_col].dt.year)
                           .groupby("Year", as_index=False)[date_col].nunique()
                           .set_index("Year")[date_col].to_dict())
    prods_per_year = (totales.groupby(year_col)[id_col].nunique().to_dict())

    expected = sum(prods_per_year.get(y, 0) * days_per_year.get(y, 0) for y in prods_per_year.keys())
    actual = len(df_daily)
    print(f"   - Filas esperadas: {expected:,} | obtenidas: {actual:,}")
    if actual != expected:
        raise AssertionError("El nº de filas no coincide con productos×días (revisa totales/calendario).")

    # Duplicados y calidad de Demand_Day
    dupes = int(df_daily.duplicated(subset=[id_col, date_col]).sum())
    nan_dd = int(df_daily[out_col].isna().sum())
    neg_dd = int((df_daily[out_col] < 0).sum())
    print(f"   - Duplicados (Product_ID, Date): {dupes}")
    print(f"   - NaN en {out_col}: {nan_dd} | Negativos: {neg_dd}")
    if dupes or nan_dd or neg_dd:
        raise AssertionError("Se detectaron duplicados o valores inválidos en la salida.")

    # Conservación de masa (además del check interno de la función)
    agg_daily = (df_daily.assign(Year=df_daily[date_col].dt.year)
                        .groupby([id_col, "Year"], as_index=False)[out_col].sum()
                        .rename(columns={out_col: "Total_Diario"}))
    comp = (totales.rename(columns={qty_name: "Total_Year"})
                   .merge(agg_daily, on=[id_col, "Year"], how="left").fillna(0))
    comp["AbsErr"] = (comp["Total_Year"] - comp["Total_Diario"]).abs()
    max_err = float(comp["AbsErr"].max())
    print(f"   - Conservación de masa (máx abs err): {max_err:.3e}")
    if max_err > 1e-9:
        raise AssertionError(f"Conservación de masa fallida (máx err={max_err:.3e}).")

def main():
    print(f"\n🧩 DESAGREGACIÓN DEMANDA · {YEAR}")
    hist_path = _pick_existing(HIST_CANDIDATES)
    if not CAL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el calendario: {CAL_PATH}")

    print(f"• Histórico  : {hist_path}")
    print(f"• Calendario : {CAL_PATH}")

    # 1) Validar calendario (estructura + 365/366 + suma de pesos≈1)
    print("\n🔎 Validando calendario…")
    cal_res = validar_calendario_estacional(str(CAL_PATH), year=YEAR, verbose=True)
    if not cal_res["ok"]:
        print("❌ Calendario inválido. Abortando.")
        raise SystemExit(1)
    print("✅ Calendario OK.")

    # 2) Leer histórico y agregar totales por (producto, año)
    hist = pd.read_csv(hist_path)
    hist["Date"] = pd.to_datetime(hist["Date"], errors="raise")
    hist["Year"] = hist["Date"].dt.year
    totales = (hist.groupby(["Product_ID", "Year"], as_index=False)["Sales Quantity"].sum())
    n_products = totales["Product_ID"].nunique()
    

    EXPECTED_PRODUCTS = 8999  # no-novedades del catálogo
    actual = totales["Product_ID"].nunique()
    assert actual == EXPECTED_PRODUCTS, (
    f"Se esperaban {EXPECTED_PRODUCTS} productos; hay {actual}. "
    "Revisa histórico y/o catálogo."
    )
    print(f"📦 Totales {YEAR}: productos={actual:,}")

    # 3) Leer calendario
    cal_df = pd.read_csv(CAL_PATH, parse_dates=["Date"])

    # 4) Desagregar
    print("\n⚙️  Desagregando…")
    df_daily = desagregar_demanda_anual(
        df_demanda_anual=totales,
        calendario_estacional=cal_df,
        id_col="Product_ID",
        year_col="Year",
        qty_col="Sales Quantity",   # ← mapeo directo a tu columna real
        date_col="Date",
        weight_col="Peso Normalizado",
        out_col="Demand_Day",
        tol=1e-9,
        check_mass=True,
    )

    # 5) QA visible
    print("\n✅ Validaciones de salida:")
    _report_checks(df_daily, totales, cal_df,
                   id_col="Product_ID", date_col="Date", year_col="Year",
                   qty_name="Sales Quantity", out_col="Demand_Day")

    # 6) Guardar en data/processed
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(OUT_PATH, index=False)
    print(f"\n💾 Guardado: {OUT_PATH}")
    print(f"📊 Resumen: filas={len(df_daily):,} | productos={df_daily['Product_ID'].nunique():,}")

if __name__ == "__main__":
    main()


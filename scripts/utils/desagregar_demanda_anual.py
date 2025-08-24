# =============================================================================
# Script: desagregar_demanda_anual.py
# Descripción:
#   Función genérica (pura) para desagregar demanda anual por producto a demanda
#   diaria aplicando un calendario estacional con pesos normalizados.
#   No realiza I/O. Devuelve un DataFrame con la desagregación.
#
# Flujo
#   1) Validaciones de columnas de entrada.
#   2) Preparación del calendario (Date→datetime, extracción de Year).
#   3) Comprobación de normalización de pesos por año (suma≈1).
#   4) Join por año (producto × días), cálculo de 'Demand_Day'.
#   5) (Opcional) Chequeo de conservación de masa por (producto, año).
#
# Inputs
#   - df_demanda_anual: DataFrame con columnas [id_col, year_col, qty_col]
#       (p.ej. ['Product_ID', 'Year', 'Demand']).
#   - calendario_estacional: DataFrame con [date_col, weight_col]
#       (p.ej. ['Date', 'Peso Normalizado']) y al menos un año.
#
# outputs:
#   - DataFrame con columnas [id_col, date_col, out_col] = demanda diaria.
#
# Notas:
#   - Soporta múltiples años en el calendario. Exige pesos normalizados por año.
#   - Los nombres de columnas son configurables para facilitar la integración.
# =============================================================================

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

def desagregar_demanda_anual(
    df_demanda_anual: pd.DataFrame,
    calendario_estacional: pd.DataFrame,
    *,
    id_col: str = "Product_ID",
    year_col: str = "Year",
    qty_col: str = "Sales Quantity",              
    date_col: str = "Date",
    weight_col: str = "Peso Normalizado",
    out_col: str = "Demand_Day",
    tol: float = 1e-9,
    check_mass: bool = True,
) -> pd.DataFrame:
    """
    Desagrega totales anuales por producto a nivel diario aplicando pesos del calendario.

    Parameters
    ----------
    df_demanda_anual : pd.DataFrame
        Debe contener, al menos, columnas [id_col, year_col, qty_col].
    calendario_estacional : pd.DataFrame
        Debe contener [date_col, weight_col]; puede incluir varios años.
    id_col, year_col, qty_col, date_col, weight_col, out_col : str
        Nombres de columnas. Por defecto: 'Product_ID', 'Year', 'Demand', 'Date',
        'Peso Normalizado', 'Demand_Day'.
    tol : float
        Tolerancia para la suma de pesos por año (y para comprobaciones numéricas).
    check_mass : bool
        Si True, comprueba conservación de masa por (producto, año).

    Returns
    -------
    pd.DataFrame
        Columnas [id_col, date_col, out_col], ordenado por [id_col, date_col].
    """
    # 1) Validaciones de columnas en la demanda anual
    required_in = {id_col, year_col, qty_col}
    missing = required_in - set(df_demanda_anual.columns)
    if missing:
        raise ValueError(f"Faltan columnas en df_demanda_anual: {sorted(missing)}")

    # 2) Preparación del calendario
    cal = calendario_estacional.copy()
    if date_col not in cal.columns or weight_col not in cal.columns:
        raise ValueError(f"calendario_estacional debe contener '{date_col}' y '{weight_col}'")
    cal[date_col] = pd.to_datetime(cal[date_col], errors="raise")
    cal["__Year"] = cal[date_col].dt.year

    # 3) Comprobar normalización de pesos por año (suma≈1)
    sums: Dict[int, float] = cal.groupby("__Year")[weight_col].sum().to_dict()
    bad = {y: s for y, s in sums.items() if not np.isclose(s, 1.0, atol=tol, rtol=0.0)}
    if bad:
        raise ValueError(f"Calendario no normalizado por año (suma!=1 dentro de tolerancia): {bad}")

    # Aviso si hay años en demanda sin calendario
    years_demand = set(df_demanda_anual[year_col].unique().tolist())
    years_cal = set(cal["__Year"].unique().tolist())
    missing_years = sorted(years_demand - years_cal)
    if missing_years:
        # No abortar: dejaremos que el merge deje fuera esos años, pero informamos.
        print(f"⚠️  Falta calendario para los años: {missing_years}. Esos registros no se desagregarán.")

    # 4) Join por año (producto × días) y cálculo de demanda diaria
    merged = df_demanda_anual.merge(
        cal[[date_col, "__Year", weight_col]],
        left_on=year_col, right_on="__Year",
        how="inner", validate="many_to_many"
    )
    merged[out_col] = merged[qty_col] * merged[weight_col]

    out = merged[[id_col, date_col, out_col]].sort_values([id_col, date_col], kind="mergesort").reset_index(drop=True)

    # 5) Conservación de masa por (producto, año)
    if check_mass:
        # reconstruimos el año a partir de la fecha para comparar apples-to-apples
        check = merged.assign(__Year_from_date=merged[date_col].dt.year) \
                      .groupby([id_col, "__Year_from_date"], as_index=False)[out_col].sum()
        check = check.rename(columns={"__Year_from_date": year_col, out_col: "sum_day"})
        target = df_demanda_anual[[id_col, year_col, qty_col]].rename(columns={qty_col: "sum_year"})
        comp = check.merge(target, on=[id_col, year_col], how="right")  # right para ver si falta algo
        comp["abs_err"] = (comp["sum_day"].fillna(0) - comp["sum_year"]).abs()
        max_err = comp["abs_err"].max()
        if not np.isfinite(max_err):
            raise AssertionError("No se pudo verificar la conservación de masa (posibles años sin calendario).")
        if max_err > tol:
            raise AssertionError(f"Conservación de masa fallida. Máx. error={max_err:.3e}")

    return out

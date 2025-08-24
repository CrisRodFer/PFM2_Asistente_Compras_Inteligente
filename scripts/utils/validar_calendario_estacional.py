# =============================================================================
# Script: validar_calendario_estacional.py
# Descripción:
#   Función genérica para validar un calendario estacional diario.
#   NO realiza I/O ni escribe a disco: recibe un DataFrame o la ruta a un CSV,
#   ejecuta comprobaciones estructurales y devuelve un dict con el resultado.
#
#   Checks principales:
#     - Columnas mínimas: 'Date', 'Peso Normalizado'
#     - Fechas parseables y pertenecientes a un único año (opcionalmente forzado)
#     - Longitud esperada (365/366), sin duplicados y sin huecos (continuidad)
#     - Coherencia bisiesto ↔ presencia de 29/02
#     - Calidad de pesos: sin NaN/Inf/negativos y suma ≈ 1.0 (con tolerancia)
#     - (Opcional) Coherencia de 'LeapNote' si existe
#
# FLujo de la función:
#   1) Carga (CSV o DataFrame)
#   2) Validaciones estructurales (fechas, longitudes, duplicados, continuidad)
#   3) Validaciones de peso (suma y dominio)
#   4) Resumen y retorno (ok/errors/warnings/summary)
#
# Inputs:
#   - calendar_obj: ruta CSV (str|Path) o pandas.DataFrame
#   - year: int opcional (si None, se infiere del propio calendario)
#   - tol: tolerancia absoluta para la suma de pesos (default 1e-9)
#   - verbose: imprime un resumen legible si True
#
# Outputs:
#   - dict con:
#       ok: bool
#       errors: list[str]
#       warnings: list[str]
#       summary: dict (métricas y columnas presentes)
#
# dependencias:
#   - pandas, numpy
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Dict, Any
import calendar as _cal
import numpy as np
import pandas as pd


# =============================================================================
# 1. UTILIDADES
# =============================================================================

def _is_leap(year: int) -> bool:
    """True si el año es bisiesto."""
    return _cal.isleap(year)


# =============================================================================
# 2. FUNCIÓN GENÉRICA (PURA)
# =============================================================================

def validar_calendario_estacional(
    calendar_obj: Union[str, Path, pd.DataFrame],
    year: Optional[int] = None,
    tol: float = 1e-9,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Valida un calendario estacional diario.

    Parameters
    ----------
    calendar_obj : str | Path | pd.DataFrame
        Ruta a CSV o DataFrame con columnas (mínimo) ['Date', 'Peso Normalizado'].
    year : int | None
        Año a validar. Si None, se infiere y se exige que todas las fechas
        pertenezcan a un único año.
    tol : float
        Tolerancia absoluta para la suma de 'Peso Normalizado'.
    verbose : bool
        Si True, imprime un resumen; si False, no imprime.

    Returns
    -------
    Dict[str, Any]
        { 'ok': bool, 'errors': list[str], 'warnings': list[str], 'summary': dict, 'source': str }
    """
    # 0) Carga
    if isinstance(calendar_obj, (str, Path)):
        df = pd.read_csv(calendar_obj)
        source = str(calendar_obj)
    elif isinstance(calendar_obj, pd.DataFrame):
        df = calendar_obj.copy()
        source = "<DataFrame>"
    else:
        raise TypeError("calendar_obj debe ser una ruta a CSV o un pandas.DataFrame")

    out: Dict[str, Any] = {"source": source, "ok": False, "errors": [], "warnings": [], "summary": {}}

    # 1) Columnas mínimas
    required = {"Date", "Peso Normalizado"}
    missing = required - set(df.columns)
    if missing:
        out["errors"].append(f"Faltan columnas obligatorias: {sorted(missing)}")
        return out

    # 2) Parseo de fecha y consistencia del año
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    except Exception as e:
        out["errors"].append(f"La columna 'Date' no es convertible a datetime: {e}")
        return out

    years = df["Date"].dt.year.unique()
    if year is None:
        if len(years) != 1:
            out["errors"].append(f"El calendario contiene varios años: {years.tolist()}")
            return out
        year = int(years[0])
    else:
        if (df["Date"].dt.year != year).any():
            out["errors"].append(f"Hay fechas fuera del año {year}. Años detectados: {years.tolist()}")
            return out

    # 3) Longitud, duplicados y continuidad
    expected_rows = 366 if _is_leap(year) else 365
    n_rows = len(df)
    if n_rows != expected_rows:
        out["errors"].append(f"Número de filas={n_rows}; se esperaban {expected_rows} para {year}.")

    dupes = int(df["Date"].duplicated().sum())
    if dupes > 0:
        out["errors"].append(f"Se han detectado {dupes} fechas duplicadas.")

    full_range = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    missing_dates = full_range.difference(df["Date"])
    if len(missing_dates) > 0:
        out["errors"].append(f"Faltan {len(missing_dates)} fechas dentro del año {year}.")

    extras = df.loc[~df["Date"].isin(full_range), "Date"]
    if len(extras) > 0:
        out["errors"].append(f"Existen {len(extras)} fechas fuera del rango anual.")

    # 4) Coherencia bisiesto ↔ 29/02
    has_feb29 = ((df["Date"].dt.month == 2) & (df["Date"].dt.day == 29)).any()
    if _is_leap(year) and not has_feb29:
        out["errors"].append("El año es bisiesto pero no aparece el 29/02.")
    if (not _is_leap(year)) and has_feb29:
        out["errors"].append("El año NO es bisiesto y aparece el 29/02.")

    # 5) Calidad de pesos y suma ≈ 1.0
    try:
        w = df["Peso Normalizado"].astype(float)
    except Exception as e:
        out["errors"].append(f"'Peso Normalizado' no es convertible a float: {e}")
        return out

    if w.isna().any():
        out["errors"].append("Hay valores NaN en 'Peso Normalizado'.")
    if np.isinf(w).any():
        out["errors"].append("Hay valores ±Inf en 'Peso Normalizado'.")
    if (w < 0).any():
        out["errors"].append("Hay valores negativos en 'Peso Normalizado'.")

    sum_w = float(w.sum())
    if not np.isclose(sum_w, 1.0, atol=tol, rtol=0.0):
        out["errors"].append(f"La suma de 'Peso Normalizado' es {sum_w:.12f} (esperado 1.0±{tol}).")

    # 6) LeapNote (si existe)
    if "LeapNote" in df.columns:
        ln = str(df["LeapNote"].iloc[0])
        if _is_leap(year) and ln not in {"explicit", "redistribute"}:
            out["warnings"].append(f"LeapNote poco informativo para año bisiesto: '{ln}'.")
        if (not _is_leap(year)) and ln not in {"n/a", "", "None"}:
            out["warnings"].append(f"LeapNote inesperado para año no bisiesto: '{ln}'.")

    # 7) Resumen y salida
    out["summary"] = {
        "year": year,
        "rows": int(n_rows),
        "expected_rows": int(expected_rows),
        "sum_weights": sum_w,
        "has_feb29": bool(has_feb29),
        "missing_dates": int(len(missing_dates)),
        "duplicates": int(dupes),
        "columns": list(df.columns),
    }

    out["ok"] = (len(out["errors"]) == 0)

    if verbose:
        if out["ok"]:
            print(f"✅ Calendario {year} válido | filas={n_rows} | suma={sum_w:.12f} | 29/02={has_feb29}")
            if out["warnings"]:
                print("⚠️  Avisos:", *[f"- {w}" for w in out["warnings"]], sep="\n")
        else:
            print(f"❌ Calendario {year} inválido:")
            for e in out["errors"]:
                print(f"- {e}")
            if out["warnings"]:
                print("⚠️  Avisos:", *[f"- {w}" for w in out["warnings"]], sep="\n")

    return out


# =============================================================================
# 3. CLI / MAIN (opcional)
#    Permite ejecutar la validación desde terminal o VSCode "como módulo".
#    Ejemplo:
#      python -m scripts.utils.validar_calendario_estacional --path outputs/calendario_estacional_2024.csv --year 2024
# =============================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Valida un calendario estacional CSV")
    p.add_argument("--path", required=True, help="Ruta al CSV del calendario")
    p.add_argument("--year", type=int, default=None, help="Año esperado (opcional)")
    p.add_argument("--tol", type=float, default=1e-9, help="Tolerancia para suma de pesos")
    p.add_argument("--quiet", action="store_true", help="No imprimir resumen (verbose=False)")

    args = p.parse_args()
    res = validar_calendario_estacional(args.path, year=args.year, tol=args.tol, verbose=not args.quiet)
    # Exit code útil para CI: 0 OK / 1 KO
    raise SystemExit(0 if res["ok"] else 1)
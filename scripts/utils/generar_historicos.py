# =============================================================================
# Script: generar_historicos.py
# Descripción:
# Módulo base SIN efectos secundarios. Construye el histórico diario para un año
# objetivo a partir de una previsión limpia y devuelve los DataFrames + métricas.
# Opcionalmente expone una función de exportación que guarda:
#   - Histórico en CSV y Parquet bajo el directorio indicado (p.ej., data/clean/)
#   - Reporte de huecos SIEMPRE en data/reports/
#
# Flujo del pipeline (capa base, sin I/O):
# 1) Cargar previsión y validar columnas
# 2) Normalizar y controlar duplicados (pre-merge)
# 3) Re-etiquetar fechas al año objetivo (manejo seguro de 29/02)
# 4) Construir calendario completo por Product_ID (365/366)
# 5) Integrar previsión en calendario (NaN en días sin observación)
# 6) Validar que no hay duplicados post-merge y que el calendario es coherente
# 7) Generar reporte de huecos y devolver (historico_df, gaps_df, metrics)
#
# Input (lo proporciona quien llama):
#   - data/clean/Prevision_Demanda_2025_Limpia.xlsx (o el que indiques)
#
# Output (solo si se llama a exportación):
#   - <out_dir>/Historico_Ventas_<YEAR>.csv
#   - <out_dir>/Historico_Ventas_<YEAR>.parquet
#   - data/reports/reporte_huecos_historico_<YEAR>.csv
#
# Dependencias:
#   - pandas
#   - pyarrow   (para Parquet si usas exportación)
#   - openpyxl  (si la previsión está en Excel)
#
# Instalación rápida:
#   pip install pandas pyarrow openpyxl
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

DATE_COL_DEFAULT = "Date"
ID_COL_DEFAULT   = "Product_ID"
QTY_COL_DEFAULT  = "Sales Quantity"


# =============================================================================
# 1. UTILIDADES
# =============================================================================

def ensure_dirs(*dirs: Path) -> None:
    """Crea directorios (y padres) si no existen."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def is_leap_year(year: int) -> bool:
    """Devuelve True si `year` es bisiesto (regla gregoriana)."""
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def load_forecast(path: Path, *, date_col: str, id_col: str, qty_col: str) -> pd.DataFrame:
    """
    Lee la previsión desde Excel y valida columnas mínimas.
    (La normalización y control de duplicados se hace en `prepare_forecast`).
    """
    df = pd.read_excel(path)
    missing = [c for c in (id_col, date_col, qty_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    return df.copy()


def prepare_forecast(df: pd.DataFrame, *, date_col: str, id_col: str, qty_col: str) -> pd.DataFrame:
    """
    Normaliza tipos, limpia duplicados pre-merge y devuelve una copia lista.
    - Fecha: a datetime normalizado al día (evita horas residuales).
    - Product_ID: string para consistencia de clave.
    - Sales Quantity: numérico (coerce → NaN si no convertible).
    - Duplicados (Product_ID, Date): conserva el último registro.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[id_col] = out[id_col].astype(str)
    out[qty_col] = pd.to_numeric(out[qty_col], errors="coerce")
    if out.duplicated([id_col, date_col], keep=False).any():
        ndup = int(out.duplicated([id_col, date_col], keep=False).sum())
        print(f"⚠️ Detectados {ndup} registros duplicados en [{id_col}, {date_col}] en la previsión → se conserva el último.")
        out = out.drop_duplicates([id_col, date_col], keep="last")
    return out


def relabel_year_safe(d: pd.Timestamp, target_year: int) -> pd.Timestamp:
    """
    Re-etiqueta el año manteniendo mes/día cuando sea válido.
    Si 29/02 no existe en el año destino, mapea a 28/02.
    Para otros casos límite, ajusta al último día del mes.
    """
    try:
        return d.replace(year=target_year)
    except ValueError:
        if d.month == 2 and d.day == 29:
            return pd.Timestamp(year=target_year, month=2, day=28)
        last_day = (pd.Timestamp(year=target_year, month=d.month, day=1) + pd.offsets.MonthEnd(1)).day
        return pd.Timestamp(year=target_year, month=d.month, day=min(d.day, last_day))


def relabel_year(df: pd.DataFrame, *, date_col: str, target_year: int) -> pd.DataFrame:
    """Aplica `relabel_year_safe` sobre la columna de fecha."""
    out = df.copy()
    out[date_col] = out[date_col].apply(lambda dt: relabel_year_safe(dt, target_year))
    return out


def build_full_calendar(product_ids: pd.Series, *, year: int, date_col: str, id_col: str) -> pd.DataFrame:
    """
    Construye calendario completo (365/366) por `Product_ID`, ordenando IDs para
    salidas estables (útil en diffs/tests).
    """
    all_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    pid_sorted = pd.Index(product_ids.astype(str).unique()).sort_values()
    cal = pd.MultiIndex.from_product([pid_sorted, all_dates], names=[id_col, date_col]).to_frame(index=False)
    return cal


def integrate_forecast(calendar_df: pd.DataFrame, relabeled_df: pd.DataFrame,
                       *, date_col: str, id_col: str, qty_col: str) -> pd.DataFrame:
    """Left-join de previsión sobre calendario completo; preserva NaN en días sin observación."""
    return calendar_df.merge(
        relabeled_df[[id_col, date_col, qty_col]],
        how="left", on=[id_col, date_col]
    )


def assert_no_duplicate_keys(df: pd.DataFrame, *, id_col: str, date_col: str, where: str = "") -> None:
    """
    Asegura que no haya duplicados en la clave tras un paso crítico (p.ej., post-merge).
    Lanza AssertionError con muestra de ejemplos si los hay.
    """
    dups = df.duplicated([id_col, date_col])
    if dups.any():
        n = int(dups.sum())
        sample = df.loc[dups, [id_col, date_col]].head(10).to_dict("records")
        loc = f" ({where})" if where else ""
        raise AssertionError(f"Se han encontrado {n} duplicados en la clave [{id_col}, {date_col}]{loc}. Ejemplos: {sample}")


def gap_report(df: pd.DataFrame, *, year: int, date_col: str, id_col: str, qty_col: str) -> pd.DataFrame:
    """
    Reporte de huecos por producto:
      - total_gaps: nº de días con NaN
      - total_days: días esperados (365/366)
      - Columnas adicionales para fechas especiales: YYYY-12-31 y, si aplica, YYYY-02-29.
    """
    tmp = df.copy()
    tmp["_is_gap"] = tmp[qty_col].isna()

    gaps_by_product = (
        tmp.groupby(id_col)["_is_gap"]
           .agg(total_gaps="sum", total_days="count")
           .reset_index()
    )

    special_dates = [pd.Timestamp(year=year, month=12, day=31)]
    if is_leap_year(year):
        special_dates.append(pd.Timestamp(year=year, month=2, day=29))

    check = tmp.loc[tmp[date_col].isin(special_dates), [id_col, date_col, qty_col]].copy()
    check["is_gap"] = check[qty_col].isna()

    if check.empty:
        return gaps_by_product

    special = (
        check.assign(date_str=check[date_col].dt.strftime("%Y-%m-%d"))
             .pivot_table(index=id_col, columns="date_str", values="is_gap", aggfunc="first")
             .reset_index()
    )
    return gaps_by_product.merge(special, on=id_col, how="left")


def sanity_checks(df: pd.DataFrame, *, year: int, date_col: str, id_col: str) -> None:
    """Verifica que cada producto tenga 365/366 fechas únicas según el año objetivo."""
    expected_days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").size
    counts = df.groupby(id_col)[date_col].nunique()
    if (counts != expected_days).any():
        bad = counts[counts != expected_days]
        raise AssertionError(f"Productos sin {expected_days} días en calendario: {list(bad.index)[:10]}…")


def summarize(df: pd.DataFrame, *, qty_col: str, id_col: str) -> Dict[str, Any]:
    """Métricas básicas del dataset integrado."""
    n_filas = len(df)
    return {
        "n_filas": n_filas,
        "n_productos": df[id_col].nunique(),
        "pct_huecos": float(df[qty_col].isna().mean() if n_filas else 0.0)
    }


# =============================================================================
# 2. API REUTILIZABLE (SIN I/O)
# =============================================================================

def generar_historico_df(
    input_path: str | Path,
    year: int,
    *,
    date_col: str = DATE_COL_DEFAULT,
    id_col: str = ID_COL_DEFAULT,
    qty_col: str = QTY_COL_DEFAULT,
    skip_checks: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Devuelve (historico_df, gaps_df, metrics) sin escribir en disco.
    Realiza: carga, normalización, re-etiquetado, calendario, merge, validaciones y métricas.
    """
    year = int(year)

    # 1) Carga + normalización + re-etiquetado
    _df_raw = load_forecast(Path(input_path), date_col=date_col, id_col=id_col, qty_col=qty_col)
    df_forecast = prepare_forecast(_df_raw, date_col=date_col, id_col=id_col, qty_col=qty_col)
    df_relabeled = relabel_year(df_forecast, date_col=date_col, target_year=year)

    # 2) Calendario + integración
    calendar_df = build_full_calendar(df_relabeled[id_col], year=year, date_col=date_col, id_col=id_col)
    historico = integrate_forecast(calendar_df, df_relabeled, date_col=date_col, id_col=id_col, qty_col=qty_col)

    # 3) Validación clave post-merge + reporte de huecos
    assert_no_duplicate_keys(historico, id_col=id_col, date_col=date_col, where="post-merge")
    gaps = gap_report(historico, year=year, date_col=date_col, id_col=id_col, qty_col=qty_col)

    # 4) Checks opcionales de calendario
    if not skip_checks:
        sanity_checks(historico, year=year, date_col=date_col, id_col=id_col)

    # 5) Métricas
    metrics = summarize(historico, qty_col=qty_col, id_col=id_col)
    dec31_col = f"{year}-12-31"
    metrics.update({
        "leap_year": is_leap_year(year),
        "expected_days": pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").size,
        "all_missing_dec31": bool(gaps[dec31_col].fillna(False).all()) if dec31_col in gaps.columns else False
    })
    return historico, gaps, metrics


# =============================================================================
# 3. EXPORTACIÓN OPCIONAL (I/O EXPLÍCITO)
# =============================================================================

def exportar_historico(
    historico: pd.DataFrame,
    gaps: pd.DataFrame,
    *,
    out_dir: str | Path,
    year: int,
    date_col: str = DATE_COL_DEFAULT,
    id_col: str = ID_COL_DEFAULT,
    qty_col: str = QTY_COL_DEFAULT,
    csv_compressed: bool = False
) -> Dict[str, Path]:
    """
    Exporta CSV/Parquet bajo `out_dir` y el reporte de huecos en `data/reports/`.

    Returns
    -------
    Dict[str, Path]: {"parquet": Path, "csv": Path, "gaps": Path}
    """
    out_dir = Path(out_dir)
    reports_dir = Path("data/reports")  # Ubicación fija para reportes
    ensure_dirs(out_dir, reports_dir)

    # Orden y limpieza de columnas antes de exportar
    df = historico[[id_col, date_col, qty_col] + [c for c in historico.columns if c not in (id_col, date_col, qty_col)]]
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    out_parquet = out_dir / f"Historico_Ventas_{year}.parquet"
    out_csv     = out_dir / f"Historico_Ventas_{year}.csv"
    out_gaps    = reports_dir / f"reporte_huecos_historico_{year}.csv"

    # Histórico
    df.to_parquet(out_parquet, index=False)
    if csv_compressed:
        out_csv = out_csv.with_suffix(".csv.gz")
        df.to_csv(out_csv, index=False, compression="gzip")
    else:
        df.to_csv(out_csv, index=False)

    # Reporte de huecos
    gaps.to_csv(out_gaps, index=False)

    return {"parquet": out_parquet, "csv": out_csv, "gaps": out_gaps}

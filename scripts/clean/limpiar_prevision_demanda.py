
# ============================================================
# Script: limpiar_prevision_demanda.py
# Descripción:
# Este script toma como entrada el archivo con la previsión corregida (`Prevision_demanda_2025_corregida.xlsx`)
# y realiza una limpieza de registros incompletos, errores de fechas o cantidades, 
# dejando el dataset listo para su uso como histórico simulado para 2024.
#
# Flujo del pipeline:
#   1. Cargar archivo desde data/raw/
#   2. Validar columnas clave y tipos básicos
#   3. Normalizar fechas y filtrar año objetivo (2025)
#   4. Eliminar negativos en 'Sales Quantity'
#   5. Detectar y resolver duplicados por (Product_ID, Date) con media
#   6. Auditoría rápida y exportación a Excel y Parquet en data/clean/

#
# Input:  data/raw/Prevision_demanda_2025_corregida.xlsx
# Output: 
#   - data/clean/Prevision_Demanda_2025_Limpia.xlsx
#   - data/clean/Prevision_Demanda_2025_Limpia.parquet
# ============================================================

from __future__ import annotations

# --- Bootstrap para que el script encuentre la raíz del proyecto (que contiene 'src') ---
import sys
from pathlib import Path

_here = Path(__file__).resolve()
p = _here
for _ in range(6):  # sube como mucho 6 niveles
    if (p / "src").exists():
        sys.path.insert(0, str(p))     # añade la carpeta raíz al sys.path
        break
    p = p.parent
# -----------------------------------------------------------------------------------------

from typing import List
import pandas as pd

# Rutas del proyecto (absolutas y portables)
from src.config import DATA_RAW, DATA_CLEAN
# Logger opcional (si lo tienes en src/logging_conf)
try:
    from src.logging_conf import get_logger
    logger = get_logger("fase1.prevision2025")
except Exception:  # pragma: no cover
    logger = None

# -----------------------------
# 0. Utilidad de salida única
# -----------------------------
def say(msg: str) -> None:
    """
    Emite un mensaje usando el logger si está disponible; si no, por print().
    Evita duplicidad de mensajes en consola.
    """
    if logger:
        logger.info(msg)
    else:
        print(msg)


# -----------------------------
# 1. Utilidades de E/S
# -----------------------------
def cargar_datos(nombre_archivo: str) -> pd.DataFrame:
    """
    Carga el Excel desde data/raw con engine 'openpyxl'.

    Parameters
    ----------
    nombre_archivo : str
        Nombre del archivo a cargar desde DATA_RAW.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en DATA_RAW.
    """
    path_archivo = DATA_RAW / nombre_archivo
    if not path_archivo.exists():
        raise FileNotFoundError(f"No encuentro el archivo: {path_archivo}")

    say(f"Leyendo: {path_archivo}")
    return pd.read_excel(path_archivo, engine="openpyxl")  # robusto para .xlsx


def exportar_archivo(df: pd.DataFrame, base_nombre_salida: str) -> None:
    """
    Exporta el DataFrame a Excel y Parquet en data/clean.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a exportar.
    base_nombre_salida : str
        Nombre base sin extensión (se generan .xlsx y .parquet).
    """
    out_base = DATA_CLEAN / base_nombre_salida
    out_base.parent.mkdir(parents=True, exist_ok=True)

    xlsx_path = out_base.with_suffix(".xlsx")
    pq_path = out_base.with_suffix(".parquet")

    df.to_excel(xlsx_path, index=False)
    df.to_parquet(pq_path, index=False)  # requiere pyarrow

    say(f"📁 Exportado: {xlsx_path}  |  {pq_path}")


# -----------------------------
# 2. Limpieza y validaciones
# -----------------------------
def check_schema(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Valida que el DataFrame contiene las columnas requeridas y
    avisa si 'Date' no está en datetime.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        say(f"⚠️ Faltan columnas requeridas: {missing}")

    if "Date" in df.columns and not str(df["Date"].dtype).startswith("datetime"):
        say("⚠️ 'Date' no es datetime; se normalizará con pd.to_datetime.")


def normalizar_y_filtrar_anio(df: pd.DataFrame, anio_objetivo: int) -> pd.DataFrame:
    """
    Normaliza 'Date' a datetime (nivel día) y filtra por año objetivo.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df[df["Date"].dt.year == anio_objetivo]
    return df


def eliminar_nulos(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Elimina filas con nulos en las columnas indicadas."""
    return df.dropna(subset=cols).copy()

def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tipos suaves recomendados para ahorrar sorpresas y memoria.
    """
    df = df.copy()
    if "Product_ID" in df.columns:
        df["Product_ID"] = df["Product_ID"].astype("string")
    if "Sales Quantity" in df.columns:
        df["Sales Quantity"] = pd.to_numeric(df["Sales Quantity"], errors="coerce")
    return df


def clip_negativos(df: pd.DataFrame, col: str = "Sales Quantity") -> pd.DataFrame:
    """
    Reemplaza negativos por 0 (clip) en la columna indicada.
    """
    df = df.copy()
    if col in df.columns:
        n_bad = (df[col] < 0).sum()
        if n_bad:
            say(f"{n_bad} valores negativos en '{col}' → clip a 0")
        df[col] = df[col].clip(lower=0)
    return df


def reportar_duplicados(df: pd.DataFrame) -> int:
    """
    Reporta duplicados por la combinación (Product_ID, Date).
    """
    dups = int(df.duplicated(["Product_ID", "Date"]).sum())
    say(f"🔎 Duplicados (Product_ID+Date): {dups}")
    return dups


def resolver_duplicados(
    df: pd.DataFrame,
    keys=("Product_ID", "Date"),
    qty_col="Sales Quantity",
    strategy: str = "mean",  # "mean" | "sum" | "max" | "first"
) -> pd.DataFrame:
    """
    Consolida duplicados por (Product_ID, Date) según estrategia.
      - mean: media de 'Sales Quantity' (por defecto)
      - sum:  suma
      - max:  máximo
      - first: conserva la primera fila
    Otras columnas se conservan con 'first'.

    Parameters
    ----------
    df : pd.DataFrame
    keys : tuple
        Claves para agrupar (por defecto, ('Product_ID', 'Date')).
    qty_col : str
        Columna de cantidad a consolidar.
    strategy : str
        Estrategia de consolidación: 'mean' | 'sum' | 'max' | 'first'.

    Returns
    -------
    pd.DataFrame
        DataFrame sin duplicados por keys.
    """
    if not df.duplicated(list(keys)).any():
        return df

    say(f"♻️ Resolviendo duplicados por {keys} con estrategia='{strategy}'")

    other = [c for c in df.columns if c not in (*keys, qty_col)]
    agg = {}

    if strategy == "sum":
        agg[qty_col] = "sum"
    elif strategy == "max":
        agg[qty_col] = "max"
    elif strategy == "mean":
        agg[qty_col] = "mean"
    else:
        agg[qty_col] = "first"

    for c in other:
        agg[c] = "first"

    out = df.groupby(list(keys), as_index=False, sort=False).agg(agg)
    return out



# -----------------------------
# 3. Auditoría
# -----------------------------
def audit_quick(df: pd.DataFrame, name: str) -> None:
    """
    Auditoría ligera del dataset: tamaño, rango de fechas, nº productos,
    duplicados clave y suma total de ventas.
    """
    filas = len(df)
    fecha_min = df["Date"].min() if "Date" in df.columns else None
    fecha_max = df["Date"].max() if "Date" in df.columns else None
    productos = df["Product_ID"].nunique() if "Product_ID" in df.columns else None
    dups = int(df.duplicated(["Product_ID", "Date"]).sum()) \
        if set(["Product_ID", "Date"]).issubset(df.columns) else None
    total = float(df["Sales Quantity"].sum()) if "Sales Quantity" in df.columns else None

    say(
        f"\n== Auditoría: {name} ==\n"
        f"Filas: {filas}\n"
        f"Rango fechas: {fecha_min} → {fecha_max}\n"
        f"Productos únicos: {productos}\n"
        f"Duplicados ID+Date: {dups}\n"
        f"Ventas totales: {total}\n"
    )


# -----------------------------
# 4. Pipeline principal
# -----------------------------
def main() -> None:
    """
    Ejecuta el pipeline de limpieza de la previsión 2025:
    carga, validaciones ligeras, normalización, filtro de año,
    clip de negativos, resolución de duplicados por (Product_ID, Date)
    con media, auditoría y exportación.
    """
    nombre_entrada = "Prevision_demanda_2025_corregida.xlsx"
    nombre_salida_base = "Prevision_Demanda_2025_Limpia"
    anio_objetivo = 2025

    say("🚀 Iniciando limpieza de previsión 2025...")

    # 1) Cargar
    df = cargar_datos(nombre_entrada)

    # 2) Validaciones ligeras (no intrusivas)
    columnas_requeridas = ["Product_ID", "Date", "Sales Quantity", "Demand Trend"]
    check_schema(df, columnas_requeridas)

    # 3) Normalizar fechas y filtrar año objetivo
    df = normalizar_y_filtrar_anio(df, anio_objetivo)

    # 4) Eliminar nulos en columnas clave (tras normalizar)
    df = eliminar_nulos(df, ["Product_ID", "Date", "Sales Quantity"])

    # 4.1) Tipos recomendados (string + numérico)
    df = enforce_types(df)


    # 5) Clip de negativos (mantienes todos los registros, pero sin valores < 0)
    df = clip_negativos(df, "Sales Quantity")

    # 6) Duplicados clave (Product_ID + Date) → resolver con 'mean'
    dups = reportar_duplicados(df)
    if dups:
        df = resolver_duplicados(df, strategy="mean")

    # 7) Auditoría rápida
    audit_quick(df, "1.1_prevision_2025")

    # 8) Exportar Excel + Parquet en data/clean
    exportar_archivo(df, nombre_salida_base)

    say("✅ Proceso completado.")




# -----------------------------
# 5. Ejecución directa
# -----------------------------
if __name__ == "__main__":
    # Dependencias recomendadas:
    #   pip install openpyxl pyarrow
    main()



    
# Auditoría opcional: comprobar si falta el 31/12/2025
from datetime import date
def report_missing_day(df, year: int, day: int = 31, month: int = 12):
    target = pd.Timestamp(date(year, month, day))
    have = set(df.loc[df["Date"] == target, "Product_ID"].unique())
    allp = set(df["Product_ID"].unique())
    missing = sorted(allp - have)
    say(f"📌 Productos sin {target.date()}: {len(missing)}")
    if missing:
        out = DATA_CLEAN / f"missing_{year}-{month:02d}-{day:02d}_prevision2025.csv"
        pd.DataFrame({"Product_ID": missing}).to_csv(out, index=False)
        say(f"↳ Listado exportado a: {out}")

df = pd.read_excel(DATA_CLEAN / "Prevision_Demanda_2025_Limpia.xlsx")
report_missing_day(df, 2025, 31, 12)
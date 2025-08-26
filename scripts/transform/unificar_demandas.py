# =============================================================================
# Script: unificar_demandas.py
# FASE 3 · Subapartado 3.1 — Unificación de demandas (2022–2024)
#
# Descripción:
#   Une los archivos de demanda desagregada (2022, 2023 y 2024) en un único
#   archivo consolidado y ejecuta validaciones clave de integridad.
#
# Inputs esperados (por defecto):
#   data/processed/demanda_diaria_2022.csv
#   data/processed/demanda_diaria_2023.csv
#   data/processed/demanda_diaria_2024.csv
#
# Output:
#   data/processed/demanda_unificada.csv
#   reports/validation/validacion_unificacion.csv
#
# Dependencias:
#   pip install pandas
# =============================================================================

from pathlib import Path
import argparse
import sys
import pandas as pd

# 0. CONFIG
ROOT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"

DEFAULT_INPUTS = [
    PROCESSED_DIR / "demanda_diaria_2022.csv",
    PROCESSED_DIR / "demanda_diaria_2023.csv",
    PROCESSED_DIR / "demanda_diaria_2024.csv",
]
DEFAULT_OUTPUT = PROCESSED_DIR / "demanda_unificada.csv"
DEFAULT_VALIDATION = VALIDATION_DIR / "validacion_unificacion.csv"

# 1. UTILIDADES
REQUIRED_COLS = ("Product_ID", "Date", "Demand_Day")

def _err(msg: str, strict: bool):
    if strict:
        raise ValueError(msg)
    else:
        print(f"[ADVERTENCIA] {msg}", file=sys.stderr)

def validar_columnas(df: pd.DataFrame, strict: bool = True):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

def cargar_csv(path: Path, strict: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_csv(path)
    validar_columnas(df, strict=strict)
    return df

def normalizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Date a datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
    # Product_ID como string
    df["Product_ID"] = df["Product_ID"].astype(str)
    # Demand_Day numérica
    df["Demand_Day"] = pd.to_numeric(df["Demand_Day"], errors="coerce")
    return df

def unificar_demandas(paths: list[Path], strict: bool = True) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = cargar_csv(p, strict=strict)
        df = normalizar_tipos(df)
        df["_input_file"] = p.name
        frames.append(df)
    df_u = pd.concat(frames, ignore_index=True)
    df_u = df_u.sort_values(["Product_ID", "Date"]).reset_index(drop=True)
    return df_u

def validar_dataset_unificado(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Devuelve un dataframe-resumen por año con métricas clave y emite advertencias/errores."""
    issues = []

    # A. NaNs y negativos
    n_nan_date = df["Date"].isna().sum()
    n_nan_prod = df["Product_ID"].isna().sum()
    n_nan_sales = df["Demand_Day"].isna().sum()
    if n_nan_date or n_nan_prod or n_nan_sales:
        _err(f"NaNs detectados -> Date: {n_nan_date}, Product_ID: {n_nan_prod}, Demand_Day: {n_nan_sales}", strict)

    n_neg = (df["Demand_Day"] < 0).sum()
    if n_neg:
        _err(f"Demandas negativas detectadas: {n_neg}", strict)

    # B. Duplicados exactos
    n_dups = df.duplicated(subset=["Product_ID", "Date"]).sum()
    if n_dups:
        _err(f"Duplicados exactos (Product_ID, Date): {n_dups}", strict)

    # C. Rango de fechas y métricas por año
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        _err("La columna 'Date' no es datetime tras la normalización.", True)

    df["Year"] = df["Date"].dt.year
    resumen = (
        df.groupby("Year")
          .agg(
              registros=("Date", "count"),
              productos_unicos=("Product_ID", "nunique"),
              demanda_total=("Demand_Day", "sum"),
              fecha_min=("Date", "min"),
              fecha_max=("Date", "max"),
          )
          .reset_index()
          .sort_values("Year")
    )

    return resumen

def exportar_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

# 2. CLI / MAIN
def parse_args():
    p = argparse.ArgumentParser(description="Unificar demandas desagregadas 2022–2024 y validar integridad.")
    p.add_argument("--in-2022", default=str(DEFAULT_INPUTS[0]), help="CSV demanda diaria 2022")
    p.add_argument("--in-2023", default=str(DEFAULT_INPUTS[1]), help="CSV demanda diaria 2023")
    p.add_argument("--in-2024", default=str(DEFAULT_INPUTS[2]), help="CSV demanda diaria 2024")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Ruta CSV unificado de salida")
    p.add_argument("--val-out", default=str(DEFAULT_VALIDATION), help="Ruta CSV resumen validación")
    p.add_argument("--strict", action="store_true", help="Si se activa, las advertencias lanzan error")
    return p.parse_args()

def main():
    args = parse_args()
    paths = [Path(args.in_2022), Path(args.in_2023), Path(args.in_2024)]
    df_u = unificar_demandas(paths, strict=args.strict)

    # Validaciones y resumen
    resumen = validar_dataset_unificado(df_u, strict=args.strict)

    # Exportar
    out_path = exportar_csv(df_u, Path(args.out))
    val_path = exportar_csv(resumen, Path(args.val_out))

    print(f"[OK] Demanda unificada -> {out_path} (filas: {len(df_u)})")
    print(f"[OK] Resumen validación -> {val_path}")
    print(resumen.to_string(index=False))

if __name__ == "__main__":
    main()

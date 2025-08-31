
# =============================================================================
# Script: filtrar_outliers.py
# Descripción: Filtra el archivo `demanda_con_outliers.csv` para obtener
#              únicamente las filas correspondientes a productos marcados
#              como outliers (is_outlier=1).
# Flujo:
#   1. Cargar `demanda_con_outliers.csv` desde data/processed
#   2. Filtrar filas con is_outlier = 1
#   3. Guardar el resultado como `outliers.csv` en data/processed
#   4. Exportar un reporte con estadísticas básicas en reports
# Input:  data/processed/demanda_con_outliers.csv
# Output: data/processed/outliers.csv
#         Archivo `data/processed/outliers.csv`
#         reports/outliers_report.txt
# Dependencias: pandas
# =============================================================================

from pathlib import Path
import pandas as pd

# --- Paths base ---
try:
    ROOT_DIR = Path(__file__).resolve().parents[2]
except NameError:
    ROOT_DIR = Path.cwd()
    if ROOT_DIR.name.lower() == "notebooks":
        ROOT_DIR = ROOT_DIR.parent

DATA_DIR    = ROOT_DIR / "data"
PROC_DIR    = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# --- Archivos ---
IN_CSV      = PROC_DIR / "demanda_con_outliers.csv"
OUT_CSV     = PROC_DIR / "outliers.csv"
OUT_REPORT  = REPORTS_DIR / "outliers_report.txt"

IN_CSV       = PROC_DIR / "demanda_con_outliers.csv"
OUT_CSV      = PROC_DIR / "outliers.csv"
OUT_PARQUET  = PROC_DIR / "outliers.parquet"
OUT_REPORT   = REPORTS_DIR / "outliers_report.txt"


def _read_csv(path: Path) -> pd.DataFrame:
    """Lectura segura del CSV de demanda (con dtypes y fechas)."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    return pd.read_csv(
        path,
        dtype={"Product_ID": str},
        parse_dates=["Date"],
        dayfirst=False,
        low_memory=False,
    )


def filtrar_outliers(in_path: Path, out_csv: Path, out_parquet: Path, report_path: Path) -> None:
    """
    Filtra un dataset de demanda para extraer únicamente los productos outliers.

    Args:
        in_path (Path): Ruta del CSV de entrada que contiene `is_outlier`.
        out_csv (Path): Ruta donde guardar el CSV de outliers.
        out_parquet (Path): Ruta donde guardar el Parquet de outliers.
        report_path (Path): Ruta donde guardar el reporte de estadísticas.
    """
    print(f"[INFO] Leyendo demanda con outliers: {in_path}")
    df = _read_csv(in_path)

    if "is_outlier" not in df.columns:
        raise KeyError("El archivo no contiene la columna `is_outlier`.")

    # Asegurar tipo entero para evitar sorpresas (p.ej., '1'/'0' como strings)
    df["is_outlier"] = pd.to_numeric(df["is_outlier"], errors="coerce").fillna(0).astype("int8")

    # --- Filtrado ---
    df_out = df.loc[df["is_outlier"].eq(1)].copy()

    # --- Guardado (CSV + Parquet) ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(out_csv, index=False)
    # Para parquet usamos pyarrow por ser más rápido/compatible
    df_out.to_parquet(out_parquet, index=False, engine="pyarrow")

    # --- Reporte ---
    n_total          = len(df)
    n_out            = len(df_out)
    n_products_total = df["Product_ID"].nunique()
    n_products_out   = df_out["Product_ID"].nunique()
    fechas           = (df_out["Date"].min(), df_out["Date"].max()) if not df_out.empty else (None, None)

    report_lines = [
        "== REPORTE DE FILTRADO DE OUTLIERS ==",
        f"Entrada : {in_path}",
        f"Salida  : {out_csv.name}  y  {out_parquet.name}",
        "",
        f"Filas totales en demanda           : {n_total:,}",
        f"Productos únicos en demanda        : {n_products_total:,}",
        f"Filas marcadas como outlier (==1)  : {n_out:,}",
        f"Productos únicos con outliers      : {n_products_out:,}",
        f"Rango temporal outliers            : {fechas[0]} ➜ {fechas[1]}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print("[OK] Proceso finalizado correctamente.")


if __name__ == "__main__":
    filtrar_outliers(IN_CSV, OUT_CSV, OUT_PARQUET, OUT_REPORT)
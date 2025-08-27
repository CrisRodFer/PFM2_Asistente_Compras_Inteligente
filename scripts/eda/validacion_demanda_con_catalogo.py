# =============================================================================
# Script: validacion_demanda_con_catalogo.py
# FASE 3 · EDA — Validación de demanda enriquecida y filtrada (solo activos)
#
# Objetivo:
#   Validar el dataset 'demanda_con_catalogo.csv' verificando:
#     - Columnas obligatorias presentes y tipos básicos.
#     - Estado_Producto == "Activo" en todos los registros.
#     - Ausencia de duplicados (Product_ID, Date).
#     - Ausencia de NaN y de valores negativos en Demand_Day.
#     - Fechas dentro del rango [2022-01-01, 2024-12-31].
#     - Cobertura diaria por Product_ID y Año (días esperados vs. observados).
#     - Consistencia de Categoria por Product_ID (una única categoría).
#
# Input:
#   data/processed/demanda_con_catalogo.csv
#
# Outputs (reports/validation/):
#   - validacion_demanda_con_catalogo_resumen.csv
#   - validacion_dup_product_date.csv
#   - validacion_valores_invalidos.csv
#   - validacion_cobertura_diaria.csv
#   - validacion_categoria_inconsistente.csv
#   - validacion_activo_invalido.csv
#
# Uso:
#   - Terminal:  python scripts/eda/validacion_demanda_con_catalogo.py
#   - Notebook:  main([])  # usa valores por defecto
# =============================================================================

from pathlib import Path
import argparse
import sys
import pandas as pd
from datetime import date

# 0) RUTAS BASE (ajuste notebook/terminal)
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    here = Path.cwd()
    ROOT_DIR = here if (here / "data" / "processed").exists() else here.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"

INPUT_PATH = PROCESSED_DIR / "demanda_con_catalogo.csv"
OUT_RESUMEN = VALIDATION_DIR / "validacion_demanda_con_catalogo_resumen.csv"
OUT_DUPS = VALIDATION_DIR / "validacion_dup_product_date.csv"
OUT_INVALID = VALIDATION_DIR / "validacion_valores_invalidos.csv"
OUT_COVERAGE = VALIDATION_DIR / "validacion_cobertura_diaria.csv"
OUT_CAT_INCONS = VALIDATION_DIR / "validacion_categoria_inconsistente.csv"
OUT_ACTIVO_INV = VALIDATION_DIR / "validacion_activo_invalido.csv"

REQUIRED_COLS = {"Product_ID", "Date", "Demand_Day", "Categoria", "Estado_Producto"}

# Rango esperado de fechas del histórico
START_DATE = pd.Timestamp("2022-01-01")
END_DATE   = pd.Timestamp("2024-12-31")

# 1) UTILIDADES
def exportar(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

def cargar_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    df = pd.read_csv(path)
    faltan = REQUIRED_COLS - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {faltan}")
    # Tipos
    df["Product_ID"] = df["Product_ID"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Demand_Day"] = pd.to_numeric(df["Demand_Day"], errors="coerce")
    return df

def chequear_activos(df: pd.DataFrame) -> pd.DataFrame:
    mask_inv = ~df["Estado_Producto"].astype(str).str.strip().str.lower().eq("activo")
    return df.loc[mask_inv].copy()

def encontrar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    dmask = df.duplicated(subset=["Product_ID", "Date"], keep=False)
    return df.loc[dmask].sort_values(["Product_ID", "Date"]).copy()

def encontrar_invalidos(df: pd.DataFrame) -> pd.DataFrame:
    mask_nan = df["Date"].isna() | df["Demand_Day"].isna() | df["Product_ID"].isna()
    mask_neg = df["Demand_Day"] < 0
    mask_out = (df["Date"] < START_DATE) | (df["Date"] > END_DATE)
    bad = df.loc[mask_nan | mask_neg | mask_out].copy()
    # etiqueta de tipo de problema
    bad["motivo"] = ""
    bad.loc[df["Date"].isna(), "motivo"] += "Date NaN; "
    bad.loc[df["Demand_Day"].isna(), "motivo"] += "Demand_Day NaN; "
    bad.loc[df["Product_ID"].isna(), "motivo"] += "Product_ID NaN; "
    bad.loc[df["Demand_Day"] < 0, "motivo"] += "Demand_Day < 0; "
    bad.loc[(df["Date"] < START_DATE) | (df["Date"] > END_DATE), "motivo"] += "Date fuera de rango; "
    return bad

def dias_en_anio(y: int) -> int:
    return 366 if pd.Timestamp(year=y, month=12, day=31).dayofyear == 366 else 365

def cobertura_por_producto_anio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (Product_ID, Year) calcula:
      - days_observed: nº de días con registros
      - days_expected: 365/366
      - missing_days: diferencia
      - tiene_gaps: missing_days > 0
    No asume actividad “real” todo el año; simplemente reporta gaps vs calendario.
    """
    tmp = df.copy()
    tmp["Year"] = tmp["Date"].dt.year
    counts = tmp.groupby(["Product_ID", "Year"])["Date"].nunique().reset_index(name="days_observed")
    counts["days_expected"] = counts["Year"].apply(dias_en_anio)
    counts["missing_days"] = counts["days_expected"] - counts["days_observed"]
    counts["tiene_gaps"] = counts["missing_days"] > 0
    return counts.sort_values(["Product_ID", "Year"])

def categoria_inconsistente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta Product_ID con más de una categoría asignada.
    """
    g = df.groupby("Product_ID")["Categoria"].nunique().reset_index(name="n_categorias")
    inc = g[g["n_categorias"] > 1]["Product_ID"]
    if inc.empty:
        return pd.DataFrame(columns=["Product_ID", "categorias_detectadas"])
    cats = (df[df["Product_ID"].isin(inc)]
            .groupby("Product_ID")["Categoria"]
            .apply(lambda s: sorted(s.dropna().unique().tolist()))
            .reset_index(name="categorias_detectadas"))
    return cats

# 2) CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validación integral de demanda_con_catalogo.csv")
    p.add_argument("--in", dest="inp", default=str(INPUT_PATH), help="CSV de entrada (demanda_con_catalogo)")
    p.add_argument("--out-resumen", default=str(OUT_RESUMEN))
    p.add_argument("--out-dups", default=str(OUT_DUPS))
    p.add_argument("--out-invalid", default=str(OUT_INVALID))
    p.add_argument("--out-coverage", default=str(OUT_COVERAGE))
    p.add_argument("--out-cat-incons", default=str(OUT_CAT_INCONS))
    p.add_argument("--out-activo-inv", default=str(OUT_ACTIVO_INV))
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    df = cargar_dataset(Path(args.inp))

    # Chequeos
    df_act_inv = chequear_activos(df)
    df_dups = encontrar_duplicados(df)
    df_bad  = encontrar_invalidos(df)
    df_cov  = cobertura_por_producto_anio(df)
    df_cat_inc = categoria_inconsistente(df)

    # Resumen global
    resumen = pd.DataFrame({
        "filas_total": [len(df)],
        "productos_total": [df["Product_ID"].nunique()],
        "fechas_min": [df["Date"].min()],
        "fechas_max": [df["Date"].max()],
        "n_duplicados_product_date": [len(df_dups)],
        "n_valores_invalidos": [len(df_bad)],
        "n_productos_categoria_inconsistente": [len(df_cat_inc)],
        "n_registros_estado_no_activo": [len(df_act_inv)],
        "pct_productos_con_gaps": [
            100.0 * (df_cov.groupby("Product_ID")["tiene_gaps"].max().sum() / df_cov["Product_ID"].nunique())
            if not df_cov.empty else 0.0
        ],
    })

    # Exportar
    exportar(resumen, Path(args.out_resumen))
    if len(df_dups): exportar(df_dups, Path(args.out_dups))
    if len(df_bad): exportar(df_bad, Path(args.out_invalid))
    if len(df_cov): exportar(df_cov, Path(args.out_coverage))
    if len(df_cat_inc): exportar(df_cat_inc, Path(args.out_cat_incons))
    if len(df_act_inv): exportar(df_act_inv, Path(args.out_activo_inv))

    print("[OK] Validación completada.")
    print(f" - Resumen -> {args.out_resumen}")
    if len(df_dups): print(f" - Duplicados -> {args.out_dups} ({len(df_dups)})")
    if len(df_bad): print(f" - Valores inválidos -> {args.out_invalid} ({len(df_bad)})")
    if len(df_cov): print(f" - Cobertura diaria -> {args.out_coverage} ({len(df_cov)})")
    if len(df_cat_inc): print(f" - Categoría inconsistente -> {args.out_cat_incons} ({len(df_cat_inc)})")
    if len(df_act_inv): print(f" - Registros no Activo -> {args.out_activo_inv} ({len(df_act_inv)})")

    try:
        from IPython.display import display  # noqa
        display(resumen)
    except Exception:
        print(resumen.to_string(index=False))

    return resumen, {"dups": df_dups, "invalid": df_bad, "coverage": df_cov,
                     "cat_incons": df_cat_inc, "activo_inv": df_act_inv}

# 3) ENTRYPOINT
if __name__ == "__main__":
    if any("ipykernel" in arg for arg in sys.argv):
        main([])    
    else:
        main()      
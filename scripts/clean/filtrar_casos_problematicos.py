
# =============================================================================
# Script: filtrar_casos_problematicos.py
# Ubicación sugerida: scripts/processing/filtrar_casos_problematicos.py
#
# Objetivo:
#   Limpiar 'demanda_con_catalogo.csv' de casos problemáticos antes de PCA/clustering:
#     - Duplicados (Product_ID, Date)
#     - NaN/negativos/fuera de rango en Date/Demand_Day
#     - Estado_Producto ≠ "Activo" (por seguridad)
#     - Categoria nula
#     - Productos con demanda total == 0 (2022–2024)
#     - Productos con cobertura insuficiente (umbral configurable)
#     - (Opcional) Outliers obvios por IQR, por producto
#
# Entradas:
#   data/processed/demanda_con_catalogo.csv
#
# Salidas:
#   data/processed/demanda_filtrada.csv
#   reports/validation/fcp_*.csv (múltiples reportes)
#
# Uso:
#   - Terminal:  python scripts/processing/filtrar_casos_problematicos.py
#   - Notebook:  main([])
#
# Dependencias: pandas
# =============================================================================

from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np

# ---------- RUTAS BASE (notebook/terminal) ----------
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
OUTPUT_PATH = PROCESSED_DIR / "demanda_filtrada.csv"

OUT_RESUMEN     = VALIDATION_DIR / "fcp_resumen.csv"
OUT_DUPS        = VALIDATION_DIR / "fcp_duplicados.csv"
OUT_INVALID     = VALIDATION_DIR / "fcp_invalidos.csv"
OUT_ZERO_DEMAND = VALIDATION_DIR / "fcp_zero_demand.csv"
OUT_LOW_COVER   = VALIDATION_DIR / "fcp_low_coverage.csv"
OUT_CAT_NULL    = VALIDATION_DIR / "fcp_categoria_nula.csv"
OUT_NO_ACTIVO   = VALIDATION_DIR / "fcp_no_activo.csv"
OUT_RULES_PER_ID= VALIDATION_DIR / "fcp_reglas_aplicadas_por_producto.csv"

REQUIRED_COLS = {"Product_ID", "Date", "Demand_Day", "Categoria", "Estado_Producto"}

# ---------- UTILIDADES ----------
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
    df["Product_ID"] = df["Product_ID"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Demand_Day"] = pd.to_numeric(df["Demand_Day"], errors="coerce")
    return df

def dias_en_anio(y: int) -> int:
    # 366 si año bisiesto
    return 366 if pd.Timestamp(year=y, month=12, day=31).dayofyear == 366 else 365

def cobertura_global(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, scope: str = "overall") -> pd.DataFrame:
    """
    Calcula cobertura de días por Product_ID.
    scope = "overall": usa días observados totales / días esperados totales (2022–2024)
    scope = "per_year": requiere que cada año cumpla el umbral (se reporta el mínimo por año)
    """
    tmp = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    tmp["Year"] = tmp["Date"].dt.year

    # Observados por producto y año
    obs = tmp.groupby(["Product_ID", "Year"])["Date"].nunique().reset_index(name="days_observed")
    # Esperados por año
    obs["days_expected"] = obs["Year"].apply(dias_en_anio)

    if scope == "per_year":
        # cobertura mínima entre años presentes
        cov = (obs.assign(cov=obs["days_observed"] / obs["days_expected"])
                  .groupby("Product_ID")["cov"].min()
                  .reset_index(name="coverage"))
        cov["coverage_scope"] = "per_year_min"
        return cov

    # scope overall
    total_obs = obs.groupby("Product_ID")["days_observed"].sum()
    total_exp = obs.groupby("Product_ID")["days_expected"].sum()
    cov = (total_obs / total_exp).reset_index(name="coverage")
    cov["coverage_scope"] = "overall"
    return cov

def etiquetar_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Devuelve True si el valor es outlier por IQR (Q1-1.5*IQR, Q3+1.5*IQR).
    Si la serie es casi constante (IQR=0), no marca outliers.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(False, index=series.index)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

# ---------- CLI ----------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Filtrado de casos problemáticos en demanda_con_catalogo.csv")
    p.add_argument("--in", dest="inp", default=str(INPUT_PATH), help="CSV de entrada (demanda_con_catalogo)")
    p.add_argument("--out", default=str(OUTPUT_PATH), help="CSV de salida (demanda_filtrada)")
    p.add_argument("--out-resumen", default=str(OUT_RESUMEN))
    p.add_argument("--start", default="2022-01-01", help="Fecha inicio rango esperado")
    p.add_argument("--end", default="2024-12-31", help="Fecha fin rango esperado")
    p.add_argument("--min-coverage", type=float, default=0.50, help="Cobertura mínima requerida (0–1)")
    p.add_argument("--coverage-scope", choices=["overall","per_year"], default="overall",
                   help="Cómo calcular cobertura mínima (global 2022–2024 o mínimo por año)")
    p.add_argument("--remove-outliers", action="store_true", help="Elimina outliers por IQR por Product_ID (opcional)")
    return p.parse_args(argv)

# ---------- MAIN ----------
def main(argv=None):
    args = parse_args(argv)

    start = pd.Timestamp(args.start)
    end   = pd.Timestamp(args.end)

    df = cargar_dataset(Path(args.inp))

    # Track de reglas aplicadas por Product_ID
    reglas = {}  # product_id -> set(reglas)

    # 1) Seguridad: Estado_Producto
    no_activo = df[~df["Estado_Producto"].astype(str).str.strip().str.lower().eq("activo")].copy()
    if not no_activo.empty:
        for pid in no_activo["Product_ID"].unique():
            reglas.setdefault(pid, set()).add("estado_no_activo")
    df = df[df["Estado_Producto"].astype(str).str.strip().str.lower().eq("activo")]

    # 2) Invalidos: NaN/negativos/fuera de rango
    mask_nan = df["Date"].isna() | df["Demand_Day"].isna() | df["Product_ID"].isna()
    mask_neg = df["Demand_Day"] < 0
    mask_out = (df["Date"] < start) | (df["Date"] > end)
    invalid = df.loc[mask_nan | mask_neg | mask_out].copy()
    if not invalid.empty:
        for pid in invalid["Product_ID"].unique():
            reglas.setdefault(pid, set()).add("valores_invalidos")
    df = df.loc[~(mask_nan | mask_neg | mask_out)].copy()

    # 3) Categoria nula
    cat_null = df[df["Categoria"].isna()].copy()
    if not cat_null.empty:
        for pid in cat_null["Product_ID"].unique():
            reglas.setdefault(pid, set()).add("categoria_nula")
    df = df[~df["Categoria"].isna()].copy()

    # 4) Duplicados (Product_ID, Date)
    dups_mask = df.duplicated(subset=["Product_ID", "Date"], keep=False)
    dups = df.loc[dups_mask].copy()
    if not dups.empty:
        for pid in dups["Product_ID"].unique():
            reglas.setdefault(pid, set()).add("duplicados_product_date")
    # eliminamos duplicados dejando el primero
    df = df.drop_duplicates(subset=["Product_ID", "Date"], keep="first").copy()

    # 5) Productos con demanda total == 0
    total_by_id = df.groupby("Product_ID")["Demand_Day"].sum()
    zero_ids = total_by_id[total_by_id == 0].index.tolist()
    zero_df = df[df["Product_ID"].isin(zero_ids)].copy()
    if zero_ids:
        for pid in zero_ids:
            reglas.setdefault(pid, set()).add("demanda_total_cero")
    df = df[~df["Product_ID"].isin(zero_ids)].copy()

    # 6) Cobertura insuficiente
    cov = cobertura_global(df, start, end, scope=args.coverage_scope)
    low_cov_ids = cov.loc[cov["coverage"] < args.min_coverage, "Product_ID"].tolist()
    low_cov_df = df[df["Product_ID"].isin(low_cov_ids)].copy()
    if low_cov_ids:
        for pid in low_cov_ids:
            reglas.setdefault(pid, set()).add(f"cobertura_baja_{args.coverage_scope}")
    df = df[~df["Product_ID"].isin(low_cov_ids)].copy()

    # 7) (Opcional) Outliers por IQR por producto
    removed_outliers = pd.DataFrame(columns=df.columns)
    if args.remove_outliers:
        out_mask_list = []
        for pid, g in df.groupby("Product_ID"):
            m = etiquetar_outliers_iqr(g["Demand_Day"])
            if m.any():
                reglas.setdefault(pid, set()).add("outliers_iqr")
            idx = g.index[m]
            out_mask_list.append(idx)
        if out_mask_list:
            idx_all = out_mask_list[0].union_many(out_mask_list[1:]) if len(out_mask_list) > 1 else out_mask_list[0]
            removed_outliers = df.loc[idx_all].copy()
            df = df.drop(index=idx_all)

    # ---------- RESÚMENES Y EXPORT ----------
    # Trazabilidad por producto-reglas
    rules_rows = []
    for pid, rset in reglas.items():
        rules_rows.append({"Product_ID": pid, "reglas": ",".join(sorted(rset))})
    rules_df = pd.DataFrame(rules_rows).sort_values("Product_ID") if rules_rows else pd.DataFrame(columns=["Product_ID","reglas"])

    resumen = pd.DataFrame({
        "filas_finales": [len(df)],
        "productos_finales": [df["Product_ID"].nunique()],
        "start_date": [start],
        "end_date": [end],
        "min_coverage": [args.min_coverage],
        "coverage_scope": [args.coverage_scope],
        "duplicados_eliminados": [int(len(dups)/2) if len(dups) else 0],  # aprox. parejas
        "invalidos_eliminados": [len(invalid)],
        "productos_demanda_cero_eliminados": [len(zero_ids)],
        "productos_baja_cobertura_eliminados": [len(low_cov_ids)],
        "outliers_eliminados": [len(removed_outliers)],
        "registros_entrada_estimada": [None],  # puedes completar si quieres leyendo len antes de filtrar
    })

    exportar(df, Path(args.out))
    if not dups.empty:        exportar(dups, OUT_DUPS)
    if not invalid.empty:     exportar(invalid, OUT_INVALID)
    if not zero_df.empty:     exportar(zero_df, OUT_ZERO_DEMAND)
    if not low_cov_df.empty:  exportar(low_cov_df, OUT_LOW_COVER)
    if not cat_null.empty:    exportar(cat_null, OUT_CAT_NULL)
    if not no_activo.empty:   exportar(no_activo, OUT_NO_ACTIVO)
    if not rules_df.empty:    exportar(rules_df, OUT_RULES_PER_ID)
    exportar(resumen, OUT_RESUMEN)

    print(f"[OK] Demanda filtrada -> {args.out} (filas: {len(df)}, productos: {df['Product_ID'].nunique()})")
    print(f"[OK] Resumen -> {OUT_RESUMEN}")
    if len(dups):        print(f" - Duplicados -> {OUT_DUPS} ({len(dups)})")
    if len(invalid):     print(f" - Inválidos -> {OUT_INVALID} ({len(invalid)})")
    if len(zero_df):     print(f" - Demanda total cero -> {OUT_ZERO_DEMAND} ({len(zero_df['Product_ID'].unique())} productos)")
    if len(low_cov_df):  print(f" - Baja cobertura -> {OUT_LOW_COVER} ({len(low_cov_df['Product_ID'].unique())} productos)")
    if len(cat_null):    print(f" - Categoria nula -> {OUT_CAT_NULL} ({len(cat_null)})")
    if len(no_activo):   print(f" - Estado ≠ Activo -> {OUT_NO_ACTIVO} ({len(no_activo)})")
    if not rules_df.empty:
        print(f" - Reglas por producto -> {OUT_RULES_PER_ID}")

    try:
        from IPython.display import display  # noqa
        display(resumen)
    except Exception:
        print(resumen.to_string(index=False))

    return df, resumen

# ---------- ENTRYPOINT ----------
if __name__ == "__main__":
    if any("ipykernel" in arg for arg in sys.argv):
        main([])  
    else:
        main()     

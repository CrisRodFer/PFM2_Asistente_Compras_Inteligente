
# =============================================================================
# Script: cruzar_catalogo.py
# FASE 3 · Subapartado 3.2 — Cruce con catálogo y asociación de categorías
#
# Descripción (paso a paso):
#   1) Carga catálogo limpio desde data/clean/Catalogo_Productos_Limpio.xlsx
#   2) Filtra catálogo a Estado_Producto == "Activo" (case-insensitive/espacios)
#   3) Une (INNER JOIN) los activos con la demanda unificada
#   4) Valida:
#       - Columnas obligatorias en demanda y catálogo
#       - Todos los registros resultantes están "Activos"
#       - No hay duplicados (Product_ID, Date)
#       - Rango de fechas se mantiene (min y max iguales al de entrada)
#       - Nº de productos de demanda sin correspondencia en catálogo activo
#       - Categoría no nula en la salida
#   5) Exporta:
#       - data/processed/demanda_con_catalogo.csv
#       - reports/validation/resumen_cruce_catalogo.csv
#       - (opcional) reports/validation/productos_sin_match_catalogo.csv
#
# Uso:
#   - Terminal:  python scripts/fase3/cruzar_catalogo.py
#   - Notebook:  main([])   # usa valores por defecto
#
# Entradas por defecto:
#   - data/processed/demanda_unificada.csv
#   - data/clean/Catalogo_Productos_Limpio.xlsx  (primera hoja)
#
# Columnas esperadas:
#   DEMANDA  -> Product_ID, Date, Demand_Day
#   CATALOGO -> Product_ID, Categoria, Estado_Producto
# =============================================================================

from pathlib import Path
import argparse, sys, re
import pandas as pd

# --- rutas (notebook/terminal) ---
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    nb_dir = Path.cwd()
    ROOT_DIR = nb_dir if (nb_dir / "data" / "processed").exists() else nb_dir.parent

DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"

DEFAULT_DEMANDA  = PROCESSED_DIR / "demanda_unificada.csv"
DEFAULT_CATALOGO = CLEAN_DIR / "Catalogo_Productos_Limpio.xlsx"
DEFAULT_OUTPUT   = PROCESSED_DIR / "demanda_con_catalogo.csv"
DEFAULT_RESUMEN  = VALIDATION_DIR / "resumen_cruce_catalogo.csv"
DEFAULT_SINMATCH = VALIDATION_DIR / "productos_sin_match_catalogo.csv"

REQ_DEMANDA  = {"Product_ID","Date","Demand_Day"}
REQ_CATALOGO = {"Product_ID","Categoria","Estado_Producto"}

# --- util: quitar acentos / normalizar nombres de columnas ---
_ACENTOS = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
def norm_colname(c: str) -> str:
    s = str(c).translate(_ACENTOS).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

COL_ALIASES = {
    "product_id": ["product_id","producto_id","id_producto","sku","id","codigo","codigo_producto"],
    "categoria": ["categoria","categoria_producto","category","familia"],
    "estado_producto": ["estado_producto","estado","status","estadoitem","estado_articulo"],
}

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm = {c: norm_colname(c) for c in df.columns}
    df = df.rename(columns=norm)
    rename_map = {}
    for canon, aliases in COL_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = canon
                break
    df = df.rename(columns=rename_map)
    return df

# --- leer demanda ---
def leer_demanda(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"No se encontró demanda: {path}")
    df = pd.read_csv(path)
    faltan = REQ_DEMANDA - set(df.columns)
    if faltan: raise ValueError(f"Demanda sin columnas requeridas: {faltan}")
    df["Product_ID"] = df["Product_ID"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any(): raise ValueError("Fechas no parseables en 'Date'.")
    df["Key_Norm"] = normalize_key(df["Product_ID"])
    return df

# --- leer catálogo con autodetección de hoja/encabezado ---
def leer_catalogo(path: Path, sheet=None) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"No se encontró catálogo: {path}")
    xls = pd.ExcelFile(path)
    sheets = [sheet] if sheet is not None else xls.sheet_names
    for sh in sheets:
        for header in [0,1,2,3]:
            try:
                df = pd.read_excel(path, sheet_name=sh, header=header)
                df = map_columns(df)
                if {"product_id","categoria","estado_producto"}.issubset(df.columns):
                    # normalizaciones mínimas
                    df["product_id"] = df["product_id"].astype(str)
                    df["estado_producto"] = df["estado_producto"].astype(str)
                    return df.rename(columns={
                        "product_id":"Product_ID","categoria":"Categoria","estado_producto":"Estado_Producto"
                    }), sh, header
            except Exception:
                continue
    raise ValueError("No se localizaron columnas requeridas en ninguna hoja (buscadas: Product_ID, Categoria, Estado_Producto).")

# --- normalizar claves para join ---
_SEP_PATTERN = re.compile(r"[\s\-\_./]+")
def normalize_key(series_or_str):
    def _one(x):
        s = str(x).strip().upper()
        s = _SEP_PATTERN.sub("", s)
        return s.lstrip("0") if s.isdigit() else s
    if isinstance(series_or_str, pd.Series):
        return series_or_str.map(_one)
    return _one(series_or_str)

# --- lógica principal ---
def filtrar_activos(df_cat: pd.DataFrame):
    raw_counts = df_cat["Estado_Producto"].str.strip().str.lower().value_counts(dropna=False)
    act_mask = df_cat["Estado_Producto"].str.strip().str.lower().eq("activo")
    df_act = df_cat[act_mask].copy()
    return df_act, int((~act_mask).sum()), raw_counts

def cruzar(df_dem, df_cat_act):
    df_cat_act["Key_Norm"] = normalize_key(df_cat_act["Product_ID"])
    left = df_dem.copy()
    out = left.merge(df_cat_act[["Key_Norm","Categoria","Estado_Producto"]], on="Key_Norm", how="inner")
    # mantenemos Product_ID de demanda
    return out

def diagnostico_keys(df_dem, df_cat_act):
    d = set(df_dem["Key_Norm"].unique())
    c = set(normalize_key(df_cat_act["Product_ID"]).unique())
    inter = d & c
    return {
        "dem_keys": len(d),
        "cat_keys": len(c),
        "inter": len(inter),
        "solo_dem_ej": list(d - c)[:10],
        "solo_cat_ej": list(c - d)[:10],
    }

def validar_salida(df_in, df_out, n_inactivos):
    if not df_out["Estado_Producto"].str.strip().str.lower().eq("activo").all():
        raise ValueError("Hay registros con Estado_Producto != 'Activo' en la salida.")
    n_dups = df_out.duplicated(subset=["Product_ID","Date"]).sum()
    if n_dups: raise ValueError(f"Duplicados (Product_ID,Date) tras cruce: {n_dups}")
    in_min, in_max = df_in["Date"].min(), df_in["Date"].max()
    out_min, out_max = df_out["Date"].min(), df_out["Date"].max()
    resumen = pd.DataFrame({
        "registros_entrada":[len(df_in)],
        "registros_salida":[len(df_out)],
        "productos_entrada":[df_in["Product_ID"].nunique()],
        "productos_salida":[df_out["Product_ID"].nunique()],
        "inactivos_eliminados_cat":[n_inactivos],
        "fecha_min_entrada":[in_min], "fecha_max_entrada":[in_max],
        "fecha_min_salida":[out_min], "fecha_max_salida":[out_max],
        "rango_fechas_igual":[bool(in_min==out_min and in_max==out_max)],
        "filas_con_categoria_nula":[int(df_out["Categoria"].isna().sum())],
    })
    return resumen

# --- CLI ---
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cruzar demanda con catálogo (detecta hoja/encabezado y normaliza claves).")
    p.add_argument("--demanda",  default=str(DEFAULT_DEMANDA))
    p.add_argument("--catalogo", default=str(DEFAULT_CATALOGO))
    p.add_argument("--sheet",    default=None, help="Nombre/índice de hoja (opcional). Si no, autodetecta.")
    p.add_argument("--out",      default=str(DEFAULT_OUTPUT))
    p.add_argument("--val-out",  default=str(DEFAULT_RESUMEN))
    p.add_argument("--out-sinmatch", default=str(DEFAULT_SINMATCH))
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    df_dem = leer_demanda(Path(args.demanda))
    df_cat, hoja, header = leer_catalogo(Path(args.catalogo), sheet=args.sheet)
    print(f"[INFO] Catálogo leído de hoja='{hoja}' con header={header}. Columnas: {list(df_cat.columns)}")

    df_act, n_inactivos, dist_estado = filtrar_activos(df_cat)
    print("[INFO] Conteo Estado_Producto en catálogo:\n", dist_estado.to_string())

    diag = diagnostico_keys(df_dem, df_act)
    print(f"[DIAG] claves demanda={diag['dem_keys']}, claves catálogo activos={diag['cat_keys']}, intersección={diag['inter']}")
    if diag["inter"] == 0:
        print("[ALERTA] Intersección de claves = 0. Ejemplos solo en DEM:", diag["solo_dem_ej"])
        print("[ALERTA] Ejemplos solo en CAT:", diag["solo_cat_ej"])

    df_out = cruzar(df_dem, df_act)

    resumen = validar_salida(df_dem, df_out, n_inactivos)

    # export
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(DEFAULT_OUTPUT, index=False)
    resumen.to_csv(DEFAULT_RESUMEN, index=False)

    # claves sin match (útiles para revisar el porqué)
    solo_dem = sorted(set(df_dem["Key_Norm"].unique()) - set(normalize_key(df_act["Product_ID"]).unique()))
    if solo_dem:
        pd.DataFrame({"Key_Norm": solo_dem}).to_csv(DEFAULT_SINMATCH, index=False)

    print(f"[OK] Inactivos eliminados: {n_inactivos}")
    print(f"[OK] Salida filas: {len(df_out)} | Productos: {df_out['Product_ID'].nunique()}")
    print(f"[OK] -> {DEFAULT_OUTPUT}")
    print(f"[OK] Resumen -> {DEFAULT_RESUMEN}")
    try:
        from IPython.display import display
        display(resumen)
    except Exception:
        print(resumen.to_string(index=False))
    return df_out, resumen

if __name__ == "__main__":
    if any("ipykernel" in arg for arg in sys.argv):
        main([])
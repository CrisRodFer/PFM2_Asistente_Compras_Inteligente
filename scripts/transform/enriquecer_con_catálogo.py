# =============================================================================
# Script: enriquecer_con_catalogo.py
# FASE 3 · 3.2.A — Enriquecimiento de demanda con catálogo (sin filtrar)
#
# Flujo:
#   1) Cargar data/processed/demanda_unificada.csv
#   2) Cargar data/clean/Catalogo_Productos_Limpio.xlsx
#   3) Normalizar claves y hacer LEFT JOIN por Product_ID_norm
#   4) Añadir Categoria y Estado_Producto a cada fila de demanda
#   5) Exportar data/processed/demanda_enriquecida.csv
#   6) Guardar reports/validation/resumen_enriquecimiento.csv
#
# Uso:
#   python scripts/fase3/enriquecer_con_catalogo.py
#   python scripts/fase3/enriquecer_con_catalogo.py --debug
#
# Dependencias: pandas, openpyxl
# =============================================================================

from pathlib import Path
import argparse, sys, re
import pandas as pd

# ---------------- RUTAS BASE (notebook/terminal) ----------------
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    here = Path.cwd()
    ROOT_DIR = here if (here / "data" / "processed").exists() else here.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"
REPORTS_DIR = ROOT_DIR / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"

DEMANDA_IN  = PROCESSED_DIR / "demanda_unificada.csv"
CATALOGO_IN = CLEAN_DIR / "Catalogo_Productos_Limpio.xlsx"
ENRIQ_OUT   = PROCESSED_DIR / "demanda_enriquecida.csv"
RESUMEN_OUT = VALIDATION_DIR / "resumen_enriquecimiento.csv"

REQ_DEM = {"Product_ID", "Date", "Demand_Day"}
REQ_CAT = {"Product_ID", "Categoria", "Estado_Producto"}

SEP_PAT = re.compile(r"[\s\-\_./]+")

# ----------------- UTILIDADES -----------------
def norm_id_series(s: pd.Series) -> pd.Series:
    """str -> strip -> quitar '.0' -> upper -> quitar separadores."""
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    s = s.str.upper().str.replace(SEP_PAT, "", regex=True)
    return s

def cargar_demanda(path: Path, debug=False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] No existe demanda: {path}")
    df = pd.read_csv(path)
    faltan = REQ_DEM - set(df.columns)
    if faltan:
        raise ValueError(f"[ERROR] Demanda sin columnas requeridas: {faltan}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("[ERROR] Fechas no parseables en 'Date' de demanda.")

    df["Product_ID"] = df["Product_ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["Product_ID_norm"] = norm_id_series(df["Product_ID"])

    if debug:
        print(f"[DEBUG] Demanda: filas={len(df)}, productos={df['Product_ID'].nunique()}")
        print("[DEBUG] Demanda.head():")
        print(df.head(3).to_string(index=False))
    return df

def autodetectar_catalogo(path: Path, debug=False) -> tuple[pd.DataFrame, str, int]:
    """
    Intenta leer el Excel probando varias hojas y headers,
    normaliza nombres, valida columnas requeridas y devuelve (df, hoja, header).
    """
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] No existe catálogo: {path}")
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] No se puede abrir el Excel. Asegúrate de tener 'openpyxl' instalado.\n{e}"
        )

    def norm_col(c: str) -> str:
        # quita espacios visibles y no visibles
        return str(c).replace("\u00A0", " ").strip()

    for sh in xls.sheet_names:
        for header in [0, 1, 2, 3]:
            try:
                df = pd.read_excel(path, sheet_name=sh, header=header, engine="openpyxl")
                df.columns = [norm_col(c) for c in df.columns]
                if REQ_CAT.issubset(set(df.columns)):
                    if debug:
                        print(f"[DEBUG] Catálogo leído -> hoja='{sh}', header={header}")
                        print("[DEBUG] Columnas catálogo:", list(df.columns))
                    return df, sh, header
            except Exception:
                continue
    raise ValueError(
        "[ERROR] No se localizaron columnas requeridas en el catálogo "
        f"(buscadas: {sorted(REQ_CAT)}). Revisa hoja/encabezado."
    )

def preparar_catalogo(df_cat: pd.DataFrame, debug=False) -> pd.DataFrame:
    df = df_cat.copy()
    # tipados y limpieza básica
    df["Product_ID"] = df["Product_ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["Estado_Producto"] = df["Estado_Producto"].astype(str)
    df["Product_ID_norm"] = norm_id_series(df["Product_ID"])
    if debug:
        print(f"[DEBUG] Catálogo: filas={len(df)}, productos={df['Product_ID'].nunique()}")
        print("[DEBUG] Catálogo.head():")
        print(df[["Product_ID","Categoria","Estado_Producto"]].head(5).to_string(index=False))
    return df

def exportar_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

# ----------------- CLI -----------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Enriquecer demanda con catálogo (añadir Categoria y Estado_Producto).")
    p.add_argument("--demanda", default=str(DEMANDA_IN))
    p.add_argument("--catalogo", default=str(CATALOGO_IN))
    p.add_argument("--sheet", default=None, help="Hoja del Excel (opcional). Si no, autodetecta.")
    p.add_argument("--out", default=str(ENRIQ_OUT))
    p.add_argument("--val-out", default=str(RESUMEN_OUT))
    p.add_argument("--debug", action="store_true", help="Imprime diagnósticos detallados")
    return p.parse_args(argv)

# ----------------- MAIN -----------------
def main(argv=None):
    args = parse_args(argv)

    # 1) Cargar demanda
    dem = cargar_demanda(Path(args.demanda), debug=args.debug)

    # 2) Cargar catálogo (autodetección si no se indica hoja)
    if args.sheet is None:
        cat_raw, hoja, header = autodetectar_catalogo(Path(args.catalogo), debug=args.debug)
    else:
        # si la hoja está indicada, intenta leer directo con header=0
        cat_raw = pd.read_excel(Path(args.catalogo), sheet_name=args.sheet, header=0, engine="openpyxl")
        cat_raw.columns = [str(c).strip() for c in cat_raw.columns]
        faltan = REQ_CAT - set(cat_raw.columns)
        if faltan:
            raise ValueError(f"[ERROR] La hoja '{args.sheet}' no contiene columnas {sorted(REQ_CAT)}")
        hoja, header = str(args.sheet), 0
        if args.debug:
            print(f"[DEBUG] Catálogo leído -> hoja='{hoja}', header={header}")
            print("[DEBUG] Columnas catálogo:", list(cat_raw.columns))

    cat = preparar_catalogo(cat_raw, debug=args.debug)

    # 3) Diagnóstico de claves
    inter = len(set(dem["Product_ID_norm"].unique()) & set(cat["Product_ID_norm"].unique()))
    if args.debug:
        print(f"[DEBUG] claves demanda: {dem['Product_ID_norm'].nunique()} | "
              f"claves catálogo: {cat['Product_ID_norm'].nunique()} | intersección: {inter}")

    # 4) LEFT JOIN por clave normalizada (conservando Product_ID original de demanda)
    joined = dem.merge(
        cat[["Product_ID_norm", "Categoria", "Estado_Producto"]],
        on="Product_ID_norm", how="left"
    ).drop(columns=["Product_ID_norm"])

    # 5) Validación y resumen
    resumen = pd.DataFrame({
        "registros_entrada":        [len(dem)],
        "productos_entrada":        [dem["Product_ID"].nunique()],
        "registros_enriquecidos":   [len(joined)],
        "productos_enriquecidos":   [joined["Product_ID"].nunique()],
        "filas_sin_categoria":      [int(joined["Categoria"].isna().sum())],
        "filas_sin_estado":         [int(joined["Estado_Producto"].isna().sum())],
        "interseccion_claves":      [inter],
        "hoja_excel_usada":         [hoja],
        "header_index_usado":       [header],
    })

    # 6) Export
    out_path = exportar_csv(joined, Path(args.out))
    val_path = exportar_csv(resumen, Path(args.val_out))

    print(f"[OK] Demanda enriquecida -> {out_path} ({len(joined)} filas)")
    print(f"[OK] Resumen -> {val_path}")
    print(resumen.to_string(index=False))

    # Sugerencia si algo quedó vacío
    if len(joined) and (joined["Categoria"].isna().all() or joined["Estado_Producto"].isna().all()):
        print("\n[AVISO] El join se ejecutó pero las columnas del catálogo quedaron vacías.")
        print("        Revisa que las columnas del Excel estén exactamente como: Product_ID, Categoria, Estado_Producto,")
        print("        y que la hoja/encabezado detectados sean correctos. Ejecuta con --debug para más detalle.")
    return joined, resumen

# ----------------- ENTRYPOINT -----------------
if __name__ == "__main__":
    if any("ipykernel" in arg for arg in sys.argv):
        main([])  
    else:
        main()    


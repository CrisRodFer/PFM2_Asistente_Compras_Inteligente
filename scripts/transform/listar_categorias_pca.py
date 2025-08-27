# =============================================================================
# Script: listar_categorias_pca.py
# Ubicación sugerida: scripts/eda/listar_categorias_pca.py
#
# Objetivo:
#   Leer el output de PCA por categorías (categoria_features.csv) y generar:
#     - Listado único de categorías (con y sin normalización de acentos)
#     - Resumen con nº total de categorías y presencia de "Otros"
#
# Entradas:
#   data/processed/pca/categoria_features.csv
#
# Salidas:
#   reports/validation/categorias_finales.csv
#   reports/validation/categorias_finales_resumen.csv
#
# Uso:
#   - Terminal:  python scripts/eda/listar_categorias_pca.py
#   - Notebook:  main([])  # usa valores por defecto
#
# Parámetros (CLI):
#   --in            Ruta al csv de features por categoría
#   --out-list      CSV con el listado de categorías
#   --out-resumen   CSV con el resumen (conteo)
#   --normalizar    Si se pasa, añade columna 'Categoria_Limpia' sin acentos
# =============================================================================

from pathlib import Path
import argparse
import sys
import pandas as pd

try:
    from unidecode import unidecode  # opcional si --normalizar
except Exception:
    unidecode = None

# ---------- RUTAS BASE (Notebook/Terminal) ----------
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    here = Path.cwd()
    ROOT_DIR = here if (here / "data" / "processed").exists() else here.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PCA_DIR = PROCESSED_DIR / "pca"
REPORTS_DIR = ROOT_DIR / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"

FEATURES_IN = PCA_DIR / "categoria_features.csv"
OUT_LIST = VALIDATION_DIR / "categorias_finales.csv"
OUT_RESUMEN = VALIDATION_DIR / "categorias_finales_resumen.csv"

REQUIRED_COLS = {"Categoria"}

# ---------- UTILIDADES ----------
def exportar(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

def cargar_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    df = pd.read_csv(path)
    faltan = REQUIRED_COLS - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en {path.name}: {faltan}")
    return df

# ---------- CLI ----------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Listar categorías resultantes del PCA")
    p.add_argument("--in", dest="inp", default=str(FEATURES_IN), help="Ruta a categoria_features.csv")
    p.add_argument("--out-list", default=str(OUT_LIST), help="CSV con listado de categorías")
    p.add_argument("--out-resumen", default=str(OUT_RESUMEN), help="CSV con resumen de categorías")
    p.add_argument("--normalizar", action="store_true", help="Añade columna 'Categoria_Limpia' sin acentos")
    return p.parse_args(argv)

# ---------- MAIN ----------
def main(argv=None):
    args = parse_args(argv)

    df = cargar_features(Path(args.inp))

    # Listado único
    cats = pd.DataFrame(sorted(df["Categoria"].astype(str).unique()), columns=["Categoria"])

    if args.normalizar:
        if unidecode is None:
            print("[ADVERTENCIA] 'unidecode' no está instalado. Instala 'unidecode' o ejecuta sin --normalizar.")
        else:
            cats["Categoria_Limpia"] = cats["Categoria"].apply(lambda x: unidecode(x).strip())

    # Resumen
    n_total = len(cats)
    tiene_otros = bool((cats["Categoria"].str.strip().str.lower() == "otros").any())
    resumen = pd.DataFrame({
        "n_categorias": [n_total],
        "incluye_otros": [tiene_otros]
    })

    # Exportar
    out_list_path = exportar(cats, Path(args.out_list))
    out_res_path = exportar(resumen, Path(args.out_resumen))

    # Log
    print(f"[OK] Listado de categorías -> {out_list_path} (n = {n_total})")
    print(f"[OK] Resumen -> {out_res_path}")
    try:
        from IPython.display import display  # noqa
        display(resumen)
        display(cats.head(20))
    except Exception:
        print(resumen.to_string(index=False))
        print("\nEjemplo de categorías:")
        print(cats.head(20).to_string(index=False))

    return cats, resumen

# ---------- ENTRYPOINT ----------
if __name__ == "__main__":
    if any("ipykernel" in arg for arg in sys.argv):
        main([])   # Notebook
    else:
        main()     # Terminal

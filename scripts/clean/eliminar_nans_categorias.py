# ============================================================================
# Script: eliminar_nans_categorias.py
# Descripción:
#   Elimina filas con NaN en una columna categórica (por defecto, 'Categoria_reducida')
#   del dataset enriquecido. Pensado como paso posterior a enriquecer_demanda_filtrada.py.
#
# Uso (terminal):
#   python scripts/clean/eliminar_nans_categorias.py
#   python scripts/clean/eliminar_nans_categorias.py --in data/processed/otro.csv --out data/processed/sin_nans.csv --col Categoria_reducida
#
# Uso (notebook):
#   %run scripts/clean/eliminar_nans_categorias.py
#
# Salida:
#   - CSV limpio sin NaNs en la columna indicada.
# ============================================================================

from pathlib import Path
import sys, argparse, logging
import pandas as pd

# -------------------------- Raíz del proyecto -------------------------------
def _detect_root_when_no_file():
    """En notebook, sube hasta encontrar una carpeta 'data'."""
    here = Path().resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir():
            return p
    return here

if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]  # .../PFM2_Asistente_Compras_Inteligente
else:
    ROOT_DIR = _detect_root_when_no_file()

DATA_DIR      = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- Logging ------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("elim_nans")

# --- Parche Jupyter: elimina el --f=... del kernel para argparse ------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# ----------------------------------------------------------------------------

# --------------------------------- Core -------------------------------------
def limpiar_archivo(in_path: Path, out_path: Path, col_objetivo: str = "Categoria_reducida"):
    """Carga el CSV, elimina filas con NaN en 'col_objetivo' y guarda el resultado."""
    in_path  = Path(in_path)
    out_path = Path(out_path)

    logger.info(f"Cargando: {in_path}")
    df = pd.read_csv(in_path)

    if col_objetivo not in df.columns:
        raise KeyError(f"No se encontró la columna '{col_objetivo}' en el archivo de entrada.")

    # Métricas iniciales
    n_total = len(df)
    n_prods = df["Product_ID"].nunique() if "Product_ID" in df.columns else None
    n_nans  = int(df[col_objetivo].isna().sum())

    # Filtrar
    df_limpio = df.dropna(subset=[col_objetivo]).copy()
    n_final = len(df_limpio)

    # Guardar
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_limpio.to_csv(out_path, index=False)

    # Resumen
    logger.info("=== LIMPIEZA NaNs EN '%s' ===", col_objetivo)
    logger.info("Filas originales: %d", n_total)
    logger.info("NaNs eliminados : %d (%.2f%%)", n_nans, (n_nans / n_total * 100) if n_total else 0)
    logger.info("Filas finales   : %d", n_final)
    if n_prods is not None:
        logger.info("Productos únicos (entrada): %d", n_prods)
        logger.info("Productos únicos (salida) : %d", df_limpio["Product_ID"].nunique())
    # Sanity quick stats si existe precio
    if "precio_medio" in df_limpio.columns:
        desc = df_limpio["precio_medio"].describe()
        logger.info("Stats precio_medio:\n%s", desc)
    logger.info(f"Guardado: {out_path}")

    return df_limpio

# --------------------------------- CLI --------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Elimina filas con NaN en una columna dada.")
    p.add_argument("--in",  dest="in_file",  type=str, default=str(PROCESSED_DIR / "demanda_filtrada_enriquecida.csv"),
                   help="CSV de entrada (enriquecido).")
    p.add_argument("--out", dest="out_file", type=str, default=str(PROCESSED_DIR / "demanda_filtrada_enriquecida_sin_nans.csv"),
                   help="CSV de salida (limpio).")
    p.add_argument("--col", dest="col_obj",  type=str, default="Categoria_reducida",
                   help="Columna sobre la que eliminar NaNs (por defecto 'Categoria_reducida').")

    # En notebook ignoramos argumentos del kernel si no vienen argv
    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []

    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> in=%s | out=%s | col=%s", args.in_file, args.out_file, args.col_obj)
    return args

def main():
    args = parse_args()
    try:
        limpiar_archivo(Path(args.in_file), Path(args.out_file), args.col_obj)
        logger.info("Proceso finalizado correctamente.")
    except Exception as e:
        logger.exception(f"Error limpiando NaNs: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

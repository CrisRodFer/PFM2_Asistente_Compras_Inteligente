

# ============================================================================
# Script: eliminar_productos_sin_pcs.py
# Descripción:
#   Elimina de productos_features.csv los Product_ID que aparecen en
#   reports/productos_sin_pcs.csv (productos sin PC1–PC3 tras el cruce con el PCA).
#   Genera un archivo limpio para clustering y valida el resultado.
#
# Uso:
#   python scripts/clean/eliminar_productos_sin_pcs.py
#   python scripts/clean/eliminar_productos_sin_pcs.py \
#       --features data/processed/productos_features.csv \
#       --sin-pcs  reports/productos_sin_pcs.csv \
#       --out      data/processed/productos_features_clean.csv
# ============================================================================

from pathlib import Path
import sys, argparse, logging
import pandas as pd

# --------------------------- Root (notebook-safe) ---------------------------
def _detect_root_when_no_file():
    here = Path().resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir():
            return p
    return here

if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    ROOT_DIR = _detect_root_when_no_file()

DATA_DIR      = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR   = ROOT_DIR / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- Logging ------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("elim_sin_pcs")

# --- Parche Jupyter: elimina --f=... del kernel para argparse ----------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# -----------------------------------------------------------------------------

# --------------------------------- Core --------------------------------------
def limpiar_features(features_path: Path, sinpcs_path: Path, out_path: Path):
    features_path = Path(features_path)
    sinpcs_path   = Path(sinpcs_path)
    out_path      = Path(out_path)

    # 1) Cargar
    logger.info(f"Cargando features: {features_path}")
    feats = pd.read_csv(features_path)

    logger.info(f"Cargando listado sin PCs: {sinpcs_path}")
    sin_pcs = pd.read_csv(sinpcs_path)

    # 2) Comprobaciones básicas
    if "Product_ID" not in feats.columns:
        raise KeyError("productos_features.csv debe contener la columna 'Product_ID'.")
    if "Product_ID" not in sin_pcs.columns:
        raise KeyError("productos_sin_pcs.csv debe contener la columna 'Product_ID'.")

    # 3) Validaciones previas
    n_inicial = feats["Product_ID"].nunique()
    dup_prod  = feats["Product_ID"].duplicated().sum()
    if dup_prod > 0:
        logger.warning("Se han detectado %d duplicados de Product_ID en features.", dup_prod)

    ids_eliminar = set(sin_pcs["Product_ID"].dropna().astype(int).unique().tolist())
    n_eliminar   = len(ids_eliminar)

    # IDs a eliminar que no están en features (para información)
    ids_no_en_features = ids_eliminar - set(feats["Product_ID"].unique())
    if ids_no_en_features:
        logger.warning("Hay %d IDs en 'sin_pcs' que no están en features. Se ignorarán.",
                       len(ids_no_en_features))

    # 4) Filtrado
    mask_keep = ~feats["Product_ID"].isin(ids_eliminar)
    feats_clean = feats.loc[mask_keep].copy()

    # 5) Validaciones posteriores
    n_final = feats_clean["Product_ID"].nunique()
    n_elim_real = n_inicial - n_final

    logger.info("=== VALIDACIONES ===")
    logger.info("Productos únicos (entrada): %d", n_inicial)
    logger.info("Productos a eliminar (listado): %d", n_eliminar)
    logger.info("Productos eliminados (real)   : %d", n_elim_real)
    logger.info("Productos únicos (salida)    : %d", n_final)

    # NaNs en columnas clave tras limpieza
    cols_clave = [c for c in ["d_total","d_media","d_std","cv","p95","mediana","precio_medio","PC1","PC2","PC3"]
                  if c in feats_clean.columns]
    nan_report = {c: int(feats_clean[c].isna().sum()) for c in cols_clave}
    logger.info("NaNs por columna (salida): %s", nan_report)

    # 6) Guardar resultados
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats_clean.to_csv(out_path, index=False)
    logger.info(f"Guardado features limpio: {out_path} (filas={len(feats_clean)}, cols={feats_clean.shape[1]})")

    # 7) Guardar reporte de eliminados aplicados
    removed_ids = sorted(list(set(feats["Product_ID"].unique()) - set(feats_clean["Product_ID"].unique())))
    reporte_path = REPORTS_DIR / "productos_eliminados_sin_pcs_aplicados.csv"
    pd.DataFrame({"Product_ID": removed_ids}).to_csv(reporte_path, index=False)
    logger.info(f"Guardado reporte de eliminados aplicados: {reporte_path}")

    return feats_clean

# ----------------------------------- CLI ------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Elimina de productos_features los Product_ID listados como 'sin PCs'.")
    p.add_argument("--features", type=str, default=str(PROCESSED_DIR / "productos_features.csv"))
    p.add_argument("--sin-pcs",  type=str, default=str(REPORTS_DIR / "productos_sin_pcs.csv"))
    p.add_argument("--out",      type=str, default=str(PROCESSED_DIR / "productos_features_clean.csv"))

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []
    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> features=%s | sin_pcs=%s | out=%s", args.features, args.sin_pcs, args.out)
    return args

def main():
    args = parse_args()
    try:
        limpiar_features(Path(args.features), Path(args.sin_pcs), Path(args.out))
        logger.info("Limpieza finalizada correctamente.")
    except Exception as e:
        logging.exception(f"Error limpiando features: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

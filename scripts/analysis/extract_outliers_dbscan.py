# ============================================================================
# Script: extract_outliers_dbscan.py
# Descripción:
#   Extrae los productos marcados como ruido por DBSCAN (label = -1) desde
#   data/processed/productos_clusters_dbscan.csv y genera:
#     - reports/outliers_dbscan.csv  -> listado de productos outlier
#     - reports/outliers_dbscan_resumen.csv -> resumen (conteos y %)
#   Opcionalmente enriquece el listado con métricas de features si se aporta
#   un fichero de features (clean o norm) para facilitar el análisis posterior.
#
# Uso:
#   python scripts/analysis/extract_outliers_dbscan.py
#   python scripts/analysis/extract_outliers_dbscan.py \
#       --in data/processed/productos_clusters_dbscan.csv \
#       --features data/processed/productos_features_clean.csv \
#       --out reports/outliers_dbscan.csv
# ============================================================================

from pathlib import Path
import sys
import argparse
import logging
import pandas as pd

# --------------------------- Raíz del proyecto ------------------------------
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

# --------------------------------- Logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_outliers_dbscan")

# --- Parche Jupyter: elimina --f=... del kernel para argparse ---------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# ----------------------------------------------------------------------------

# ------------------------------- Funciones ----------------------------------
def _check_cols(df: pd.DataFrame, required: list[str], where: str):
    faltan = [c for c in required if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en {where}: {faltan}")

def extraer_outliers_dbscan(
    clusters_path: Path,
    out_path: Path,
    features_path: Path | None = None,
    resumen_path: Path | None = None,
):
    clusters_path = Path(clusters_path)
    out_path = Path(out_path)
    if resumen_path is None:
        resumen_path = REPORTS_DIR / "outliers_dbscan_resumen.csv"

    logger.info(f"Cargando clusters DBSCAN: {clusters_path}")
    dfc = pd.read_csv(clusters_path)

    _check_cols(dfc, ["Product_ID", "Cluster_DBSCAN"], "clusters DBSCAN")

    # Conteos globales
    n_total = len(dfc)
    n_noise = int((dfc["Cluster_DBSCAN"] == -1).sum())
    pct_noise = (n_noise / n_total) if n_total > 0 else 0.0
    logger.info(f"Productos totales: {n_total} | Outliers (label=-1): {n_noise} ({pct_noise:.2%})")

    # Filtro de outliers
    outliers = dfc.loc[dfc["Cluster_DBSCAN"] == -1, ["Product_ID", "Cluster_DBSCAN"]].copy()

    # Enriquecimiento opcional con métricas de features
    if features_path:
        features_path = Path(features_path)
        logger.info(f"Enriqueciendo con features: {features_path}")
        dff = pd.read_csv(features_path)
        if "Product_ID" not in dff.columns:
            raise KeyError("El fichero de features debe contener 'Product_ID'.")

        # Selección de columnas populares si existen (no hace falta que estén todas)
        cand_cols = [
            "d_total", "d_media", "d_std", "cv", "p95", "mediana",
            "precio_medio", "PC1", "PC2", "PC3", "Categoria", "Categoria_reducida"
        ]
        keep_cols = ["Product_ID"] + [c for c in cand_cols if c in dff.columns]
        dff = dff[keep_cols].copy()

        outliers = outliers.merge(dff, on="Product_ID", how="left")

    # Guardado
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outliers.to_csv(out_path, index=False)
    logger.info(f"Guardado listado de outliers: {out_path} (filas={len(outliers)})")

    # Resumen
    df_res = pd.DataFrame([{
        "n_total": n_total,
        "n_outliers": n_noise,
        "pct_outliers": pct_noise,
        "source_clusters": str(clusters_path),
        "source_features": str(features_path) if features_path else "",
        "output": str(out_path),
    }])
    df_res.to_csv(resumen_path, index=False)
    logger.info(f"Guardado resumen: {resumen_path}")

    return {
        "n_total": n_total,
        "n_outliers": n_noise,
        "pct_outliers": pct_noise,
        "paths": {"outliers": str(out_path), "resumen": str(resumen_path)},
    }

# ------------------------------------ CLI -----------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extrae productos outlier (ruido) detectados por DBSCAN.")
    p.add_argument("--in",
                   dest="inp",
                   type=str,
                   default=str(PROCESSED_DIR / "productos_clusters_dbscan.csv"),
                   help="CSV con 'Product_ID' y 'Cluster_DBSCAN'")
    p.add_argument("--features",
                   dest="features",
                   type=str,
                   default="",  # opcional
                   help="CSV de features para enriquecer (clean o norm). Opcional.")
    p.add_argument("--out",
                   dest="outp",
                   type=str,
                   default=str(REPORTS_DIR / "outliers_dbscan.csv"),
                   help="Ruta de salida para el listado de outliers")
    p.add_argument("--resumen",
                   dest="resumen",
                   type=str,
                   default=str(REPORTS_DIR / "outliers_dbscan_resumen.csv"),
                   help="Ruta de salida para el resumen")
    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []
    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> in=%s | features=%s | out=%s | resumen=%s",
                args.inp, args.features, args.outp, args.resumen)
    return args

def main():
    args = parse_args()
    try:
        info = extraer_outliers_dbscan(
            clusters_path=Path(args.inp),
            out_path=Path(args.outp),
            features_path=Path(args.features) if args.features else None,
            resumen_path=Path(args.resumen),
        )
        logger.info("Extracción finalizada. Outliers: %s (%.2f%%) | Rutas: %s",
                    info["n_outliers"], info["pct_outliers"]*100, info["paths"])
    except Exception as e:
        logging.exception(f"Error extrayendo outliers DBSCAN: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

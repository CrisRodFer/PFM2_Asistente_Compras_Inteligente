# ============================================================================
# Script: clustering_productos_gmm.py
# Descripción:
#   Aplica Gaussian Mixture Models (GMM) sobre productos_features_norm.csv.
#   - Explora k en un rango (por defecto 3..10) y diferentes tipos de covarianza.
#   - Calcula métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz).
#   - Selecciona k y cov_type por mejor Silhouette (o se puede forzar).
#   - Exporta dataset con asignaciones y métricas por k.
#
# Entradas (por defecto):
#   - data/processed/productos_features_norm.csv
#
# Salidas:
#   - data/processed/productos_clusters_gmm.csv
#   - reports/silhouette_vs_k_gmm.csv
#   - reports/davies_bouldin_vs_k_gmm.csv
#   - reports/calinski_harabasz_vs_k_gmm.csv
#
# Uso:
#   python scripts/modeling/clustering_productos_gmm.py
#   python scripts/modeling/clustering_productos_gmm.py \
#       --in data/processed/productos_features_norm.csv \
#       --out data/processed/productos_clusters_gmm.csv \
#       --k-min 3 --k-max 10 --force-k 5 --cov-type full
# ============================================================================

from pathlib import Path
import sys, argparse, logging
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

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
logger = logging.getLogger("clustering_gmm")

# --- Parche Jupyter: elimina --f=... del kernel para argparse ---------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# ----------------------------------------------------------------------------

NUM_COLS = [
    "d_total", "d_media", "d_std", "cv", "p95", "mediana",
    "precio_medio", "PC1", "PC2", "PC3"
]

def _check_columns(df: pd.DataFrame, cols: list[str]):
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en el dataset de entrada: {faltan}")

def explorar_y_clusterizar(in_path: Path,
                           out_path: Path,
                           k_min: int = 3,
                           k_max: int = 10,
                           force_k: int | None = None,
                           cov_type: str | None = None,
                           sil_sample: int = 7000,
                           random_state: int = 42,
                           max_iter: int = 500):

    # 1) Cargar dataset
    in_path  = Path(in_path)
    out_path = Path(out_path)
    logger.info(f"Cargando features normalizadas: {in_path}")
    df = pd.read_csv(in_path)

    if "Product_ID" not in df.columns:
        raise KeyError("Se requiere columna 'Product_ID' en el dataset de entrada.")
    _check_columns(df, NUM_COLS)

    X = df[NUM_COLS].astype(float).values
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    # 2) Definir configuraciones
    ks = list(range(max(2, k_min), max(k_min, k_max) + 1))
    cov_types = [cov_type] if cov_type else ["full", "diag", "tied", "spherical"]

    res_sil, res_dbi, res_ch = [], [], []

    # 3) Exploración
    for k in ks:
        for ct in cov_types:
            logger.info(f"[Exploración] GMM con k={k}, cov_type={ct}")
            gmm = GaussianMixture(
                n_components=k, covariance_type=ct,
                random_state=random_state, max_iter=max_iter
            )
            labels = gmm.fit_predict(X)

            # Silhouette (muestreo si N muy grande)
            if n > sil_sample:
                idx = rng.choice(n, size=sil_sample, replace=False)
                X_s, y_s = X[idx], labels[idx]
            else:
                X_s, y_s = X, labels

            try:
                sil = float(silhouette_score(X_s, y_s, metric="euclidean"))
            except Exception as e:
                logger.warning(f"Silhouette falló (k={k}, cov={ct}): {e}")
                sil = np.nan

            try:
                dbi = float(davies_bouldin_score(X, labels))
            except Exception:
                dbi = np.nan
            try:
                ch = float(calinski_harabasz_score(X, labels))
            except Exception:
                ch = np.nan

            res_sil.append({"k": k, "cov_type": ct, "silhouette": sil})
            res_dbi.append({"k": k, "cov_type": ct, "davies_bouldin": dbi})
            res_ch.append({"k": k, "cov_type": ct, "calinski_harabasz": ch})

    df_sil = pd.DataFrame(res_sil)
    df_dbi = pd.DataFrame(res_dbi)
    df_ch  = pd.DataFrame(res_ch)

    path_sil = REPORTS_DIR / "silhouette_vs_k_gmm.csv"
    path_dbi = REPORTS_DIR / "davies_bouldin_vs_k_gmm.csv"
    path_ch  = REPORTS_DIR / "calinski_harabasz_vs_k_gmm.csv"
    df_sil.to_csv(path_sil, index=False)
    df_dbi.to_csv(path_dbi, index=False)
    df_ch.to_csv(path_ch, index=False)
    logger.info(f"Guardado: {path_sil}")
    logger.info(f"Guardado: {path_dbi}")
    logger.info(f"Guardado: {path_ch}")

    # 4) Selección de mejor modelo
    if force_k is not None:
        best_k = int(force_k)
        best_cov = cov_type if cov_type else "full"
        logger.info(f"Usando k forzado por CLI: k={best_k}, cov={best_cov}")
    else:
        df_sil_valid = df_sil.dropna(subset=["silhouette"])
        if df_sil_valid.empty:
            best_k, best_cov = int(np.median(ks)), "full"
            logger.warning(f"No hay silhouette válido; usando k={best_k}, cov_type={best_cov}")
        else:
            row = df_sil_valid.loc[df_sil_valid["silhouette"].idxmax()]
            best_k, best_cov = int(row["k"]), row["cov_type"]
            logger.info(f"Selección automática: k={best_k}, cov_type={best_cov}, silhouette={row['silhouette']:.4f}")

    # 5) Modelo final
    gmm_final = GaussianMixture(
        n_components=best_k, covariance_type=best_cov,
        random_state=random_state, max_iter=max_iter
    )
    labels_final = gmm_final.fit_predict(X)

    # Métricas finales
    try:
        sil_final = float(silhouette_score(X if n <= sil_sample else X[rng.choice(n, sil_sample, replace=False)],
                                           labels_final if n <= sil_sample else labels_final[rng.choice(n, sil_sample, replace=False)],
                                           metric="euclidean"))
    except Exception:
        sil_final = np.nan
    try:
        dbi_final = float(davies_bouldin_score(X, labels_final))
    except Exception:
        dbi_final = np.nan
    try:
        ch_final = float(calinski_harabasz_score(X, labels_final))
    except Exception:
        ch_final = np.nan

    _, counts = np.unique(labels_final, return_counts=True)
    dist_sizes = {int(i): int(c) for i, c in enumerate(counts)}

    logger.info("=== VALIDACIÓN CLUSTERING (GMM) ===")
    logger.info(f"k final: {best_k} | cov_type: {best_cov}")
    logger.info(f"Silhouette final       : {sil_final:.4f}" if not np.isnan(sil_final) else "Silhouette final: NaN")
    logger.info(f"Davies-Bouldin final  : {dbi_final:.4f}" if not np.isnan(dbi_final) else "Davies-Bouldin final: NaN")
    logger.info(f"Calinski-Harabasz final: {ch_final:.2f}" if not np.isnan(ch_final) else "Calinski-Harabasz final: NaN")
    logger.info(f"Tamaños de cluster     : {dist_sizes} (min={counts.min()})")

    # 6) Export
    df_out = df.copy()
    df_out["Cluster_GMM"] = labels_final
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Guardado dataset con clusters (GMM): {out_path} (filas={len(df_out)})")

    return {
        "k_final": best_k,
        "cov_type": best_cov,
        "silhouette_final": sil_final,
        "davies_bouldin_final": dbi_final,
        "calinski_harabasz_final": ch_final,
        "sizes": dist_sizes,
        "paths": {
            "clusters": str(out_path),
            "silhouette_vs_k": str(path_sil),
            "davies_bouldin_vs_k": str(path_dbi),
            "calinski_harabasz_vs_k": str(path_ch),
        },
    }

# ------------------------------------ CLI -----------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Clustering de productos con Gaussian Mixture Models (GMM).")
    p.add_argument("--in",       dest="inp",   type=str, default=str(PROCESSED_DIR / "productos_features_norm.csv"))
    p.add_argument("--out",      dest="outp",  type=str, default=str(PROCESSED_DIR / "productos_clusters_gmm.csv"))
    p.add_argument("--k-min",    dest="kmin",  type=int, default=3)
    p.add_argument("--k-max",    dest="kmax",  type=int, default=10)
    p.add_argument("--force-k",  dest="kforce", type=int, default=None)
    p.add_argument("--cov-type", dest="covtype", type=str, default=None,
                   choices=["full","diag","tied","spherical"],
                   help="Forzar tipo de covarianza; si no se especifica, se prueban todas.")
    p.add_argument("--sil-sample", dest="silsample", type=int, default=7000)
    p.add_argument("--seed",     dest="seed",   type=int, default=42)
    p.add_argument("--max-iter", dest="maxiter", type=int, default=500)

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []

    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> in=%s | out=%s | k=[%d..%d] | force_k=%s | cov_type=%s",
                args.inp, args.outp, args.kmin, args.kmax, str(args.kforce), str(args.covtype))
    return args

def main():
    args = parse_args()
    try:
        info = explorar_y_clusterizar(
            in_path=Path(args.inp),
            out_path=Path(args.outp),
            k_min=args.kmin,
            k_max=args.kmax,
            force_k=args.kforce,
            cov_type=args.covtype,
            sil_sample=args.silsample,
            random_state=args.seed,
            max_iter=args.maxiter,
        )
        logger.info("Proceso finalizado. k_final=%s | cov_type=%s | silhouette=%s | dbi=%s | ch=%s",
                    info["k_final"], info["cov_type"], info["silhouette_final"], info["davies_bouldin_final"], info["calinski_harabasz_final"])
        logger.info("Rutas: %s", info["paths"])
    except Exception as e:
        logging.exception(f"Error en clustering GMM: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

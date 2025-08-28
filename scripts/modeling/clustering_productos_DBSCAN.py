# ============================================================================
# Script: clustering_productos_dbscan.py
# Descripción:
#   Aplica DBSCAN sobre productos_features_norm.csv.
#   - Explora combinaciones de (eps, min_samples).
#   - Calcula métricas (n_clusters, %ruido, silhouette cuando aplica).
#   - Selecciona la mejor combinación por silhouette (tie-break: menos ruido,
#     más clusters) y ajusta el modelo final.
#   - Exporta dataset con labels y un CSV de exploración.
#
# Entradas (por defecto):
#   - data/processed/productos_features_norm.csv
#
# Salidas:
#   - data/processed/productos_clusters_dbscan.csv
#   - reports/dbscan_exploracion.csv
#
# Uso:
#   python scripts/modeling/clustering_productos_dbscan.py
#   python scripts/modeling/clustering_productos_dbscan.py \
#       --in data/processed/productos_features_norm.csv \
#       --out data/processed/productos_clusters_dbscan.csv \
#       --eps 0.4 0.5 0.7 --min-samples 5 10 15
# ============================================================================

from pathlib import Path
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

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
logger = logging.getLogger("clustering_dbscan")

# --- Parche Jupyter: elimina --f=... del kernel para argparse ---------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# ----------------------------------------------------------------------------

# Variables numéricas esperadas en productos_features_norm.csv
NUM_COLS = [
    "d_total", "d_media", "d_std", "cv", "p95", "mediana",
    "precio_medio", "PC1", "PC2", "PC3"
]

def _check_columns(df: pd.DataFrame, cols: list[str]):
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en el dataset de entrada: {faltan}")

# --------------------------- Lógica principal --------------------------------
def explorar_y_clusterizar_dbscan(in_path: Path,
                                  out_path: Path,
                                  eps_list: list[float],
                                  min_samples_list: list[int],
                                  metric: str = "euclidean",
                                  n_jobs: int = -1):
    """
    Explora combinaciones de (eps, min_samples), evalúa y ajusta el modelo final.
    """
    in_path  = Path(in_path)
    out_path = Path(out_path)

    logger.info(f"Cargando features normalizadas: {in_path}")
    df = pd.read_csv(in_path)

    if "Product_ID" not in df.columns:
        raise KeyError("Se requiere columna 'Product_ID' en el dataset de entrada.")
    _check_columns(df, NUM_COLS)

    X = df[NUM_COLS].astype(float).values
    n = X.shape[0]

    resultados = []
    # Exploración de grid
    for eps in eps_list:
        for ms in min_samples_list:
            model = DBSCAN(eps=eps, min_samples=ms, metric=metric, n_jobs=n_jobs)
            labels = model.fit_predict(X)

            # Métricas
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            ruido_pct = n_noise / n if n > 0 else 0.0

            if n_clusters > 1:
                try:
                    sil = float(silhouette_score(X, labels, metric=metric))
                except Exception as e:
                    logger.warning(f"Silhouette falló para eps={eps}, min_samples={ms}: {e}")
                    sil = np.nan
            else:
                sil = np.nan

            resultados.append({
                "eps": eps,
                "min_samples": ms,
                "clusters": n_clusters,
                "ruido": n_noise,
                "ruido_pct": ruido_pct,
                "silhouette": sil
            })

            logger.info(f"[Exploración] eps={eps:.3f}, min_samples={ms} "
                        f"=> clusters={n_clusters}, ruido={n_noise} ({ruido_pct:.2%}), "
                        f"silhouette={sil if not np.isnan(sil) else 'NaN'}")

    df_expl = pd.DataFrame(resultados)
    expl_path = REPORTS_DIR / "dbscan_exploracion.csv"
    df_expl.to_csv(expl_path, index=False)
    logger.info(f"Guardado reporte de exploración: {expl_path}")

    # Selección del mejor set de hiperparámetros
    # Regla: mayor silhouette; empate -> menor ruido_pct; empate -> mayor nº clusters
    df_val = df_expl.copy()
    df_val["silhouette_filled"] = df_val["silhouette"].fillna(-np.inf)
    best_idx = (
        df_val.sort_values(
            by=["silhouette_filled", "ruido_pct", "clusters"],
            ascending=[False, True, True]
        ).index[0]
    )
    best_row = df_expl.loc[best_idx]
    best_eps = float(best_row["eps"])
    best_ms  = int(best_row["min_samples"])
    logger.info(f"Selección final -> eps={best_eps}, min_samples={best_ms} | "
                f"silhouette={best_row['silhouette']}, clusters={best_row['clusters']}, ruido_pct={best_row['ruido_pct']:.2%}")

    # Modelo final
    final_model = DBSCAN(eps=best_eps, min_samples=best_ms, metric=metric, n_jobs=n_jobs)
    labels_final = final_model.fit_predict(X)

    n_clusters_final = len(set(labels_final)) - (1 if -1 in labels_final else 0)
    n_noise_final = int((labels_final == -1).sum())
    ruido_pct_final = n_noise_final / n if n > 0 else 0.0

    if n_clusters_final > 1:
        try:
            sil_final = float(silhouette_score(X, labels_final, metric=metric))
        except Exception:
            sil_final = np.nan
    else:
        sil_final = np.nan

    logger.info("=== VALIDACIÓN CLUSTERING (DBSCAN) ===")
    logger.info(f"eps={best_eps}, min_samples={best_ms}")
    logger.info(f"Clusters finales: {n_clusters_final}")
    logger.info(f"Ruido final: {n_noise_final} ({ruido_pct_final:.2%})")
    logger.info(f"Silhouette final: {sil_final if not np.isnan(sil_final) else 'NaN'}")

    # Export
    df_out = df.copy()
    df_out["Cluster_DBSCAN"] = labels_final
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Guardado dataset con clusters (DBSCAN): {out_path} (filas={len(df_out)})")

    return {
        "eps": best_eps,
        "min_samples": best_ms,
        "silhouette_final": sil_final,
        "clusters_final": n_clusters_final,
        "ruido_final": n_noise_final,
        "ruido_pct_final": ruido_pct_final,
        "paths": {
            "clusters": str(out_path),
            "exploracion": str(expl_path),
        },
    }

# ------------------------------------ CLI -----------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Clustering de productos con DBSCAN.")
    p.add_argument("--in",  dest="inp",  type=str, default=str(PROCESSED_DIR / "productos_features_norm.csv"))
    p.add_argument("--out", dest="outp", type=str, default=str(PROCESSED_DIR / "productos_clusters_dbscan.csv"))

    p.add_argument("--eps", nargs="+", type=float, default=[0.4, 0.5, 0.7, 1.0],
                   help="Lista de eps a explorar (ej: --eps 0.4 0.5 0.7)")
    p.add_argument("--min-samples", nargs="+", type=int, default=[5, 10, 15],
                   help="Lista de min_samples a explorar (ej: --min-samples 5 10 15)")

    p.add_argument("--metric", type=str, default="euclidean",
                   choices=["euclidean", "manhattan", "chebyshev", "minkowski"],
                   help="Métrica de distancia")
    p.add_argument("--n-jobs", type=int, default=-1)

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []

    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> in=%s | out=%s | eps=%s | min_samples=%s | metric=%s",
                args.inp, args.outp, args.eps, args.min_samples, args.metric)
    return args

def main():
    args = parse_args()
    try:
        info = explorar_y_clusterizar_dbscan(
            in_path=Path(args.inp),
            out_path=Path(args.outp),
            eps_list=list(args.eps),
            min_samples_list=list(args.min_samples),
            metric=args.metric,
            n_jobs=args.n_jobs,
        )
        logger.info("Proceso finalizado. eps=%s | min_samples=%s | silhouette=%s | clusters=%s | ruido=%s (%.2f%%)",
                    info["eps"], info["min_samples"], info["silhouette_final"],
                    info["clusters_final"], info["ruido_final"], info["ruido_pct_final"]*100)
        logger.info("Rutas: %s", info["paths"])
    except Exception as e:
        logging.exception(f"Error en clustering DBSCAN: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# ============================================================================
# Script: clustering_productos_agglomerative.py
# Descripción:
#   Clustering jerárquico aglomerativo sobre productos_features_norm.csv.
#   - Explora k en un rango (por defecto 3..10) y calcula Silhouette, Davies-Bouldin y Calinski-Harabasz.
#   - Selecciona k por máximo Silhouette (o se puede forzar por CLI).
#   - Ajusta el modelo final y asigna cluster por producto.
#
# Entradas (por defecto):
#   - data/processed/productos_features_norm.csv
#
# Salidas:
#   - data/processed/productos_clusters_agglom.csv
#   - reports/silhouette_vs_k_agglom.csv
#   - reports/davies_bouldin_vs_k_agglom.csv
#   - reports/calinski_harabasz_vs_k_agglom.csv
#
# Uso:
#   python scripts/modeling/clustering_productos_agglomerative.py
#   python scripts/modeling/clustering_productos_agglomerative.py \
#       --in data/processed/productos_features_norm.csv \
#       --out data/processed/productos_clusters_agglom.csv \
#       --k-min 3 --k-max 10 --force-k 5 --linkage ward
# ============================================================================

from pathlib import Path
import sys, argparse, logging
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
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
logger = logging.getLogger("clustering_agglomerative")

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

def _fit_predict_agglomerative(X: np.ndarray, k: int, linkage: str, metric: str):
    # Nota: linkage='ward' requiere metric='euclidean' en sklearn.
    if linkage == "ward" and metric != "euclidean":
        metric = "euclidean"
    model = AgglomerativeClustering(
        n_clusters=k, linkage=linkage, metric=metric
    )
    labels = model.fit_predict(X)
    return labels

def explorar_y_clusterizar(in_path: Path,
                           out_path: Path,
                           k_min: int = 3,
                           k_max: int = 10,
                           force_k: int | None = None,
                           linkage: str = "ward",
                           metric: str = "euclidean",
                           sil_sample: int = 7000,
                           random_state: int = 42):
    # 1) Cargar
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

    # 2) Explorar k
    ks = list(range(max(2, k_min), max(k_min, k_max) + 1))
    res_sil, res_dbi, res_ch = [], [], []

    for k in ks:
        logger.info(f"[Exploración] Agglomerative (linkage={linkage}, metric={metric}) con k={k}")
        labels = _fit_predict_agglomerative(X, k, linkage, metric)

        # Silhouette (muestreo si N muy grande)
        if n > sil_sample:
            idx = rng.choice(n, size=sil_sample, replace=False)
            X_s = X[idx]; y_s = labels[idx]
        else:
            X_s, y_s = X, labels

        try:
            sil = float(silhouette_score(X_s, y_s, metric="euclidean"))
        except Exception as e:
            logger.warning(f"Silhouette falló para k={k}: {e}")
            sil = np.nan

        # Davies-Bouldin (↓ mejor) y Calinski-Harabasz (↑ mejor) con todo X (si posible)
        try:
            dbi = float(davies_bouldin_score(X, labels))
        except Exception as e:
            logger.warning(f"Davies-Bouldin falló para k={k}: {e}")
            dbi = np.nan
        try:
            ch = float(calinski_harabasz_score(X, labels))
        except Exception as e:
            logger.warning(f"Calinski-Harabasz falló para k={k}: {e}")
            ch = np.nan

        res_sil.append({"k": k, "silhouette": sil})
        res_dbi.append({"k": k, "davies_bouldin": dbi})
        res_ch.append({"k": k, "calinski_harabasz": ch})

    df_sil = pd.DataFrame(res_sil)
    df_dbi = pd.DataFrame(res_dbi)
    df_ch  = pd.DataFrame(res_ch)

    path_sil = REPORTS_DIR / "silhouette_vs_k_agglom.csv"
    path_dbi = REPORTS_DIR / "davies_bouldin_vs_k_agglom.csv"
    path_ch  = REPORTS_DIR / "calinski_harabasz_vs_k_agglom.csv"
    df_sil.to_csv(path_sil, index=False)
    df_dbi.to_csv(path_dbi, index=False)
    df_ch.to_csv(path_ch, index=False)
    logger.info(f"Guardado: {path_sil}")
    logger.info(f"Guardado: {path_dbi}")
    logger.info(f"Guardado: {path_ch}")

    # 3) Selección de k
    if force_k is not None:
        best_k = int(force_k)
        logger.info(f"Usando k forzado por CLI: k={best_k}")
    else:
        df_sil_valid = df_sil.dropna(subset=["silhouette"])
        if df_sil_valid.empty:
            best_k = int(np.median(ks))
            logger.warning(f"No hay silhouette válido; usando k={best_k} (mediana del rango).")
        else:
            max_sil = df_sil_valid["silhouette"].max()
            candidatos = df_sil_valid.loc[df_sil_valid["silhouette"] == max_sil, "k"].tolist()
            best_k = min(candidatos)
            logger.info(f"Selección automática por silhouette: k={best_k} (silhouette={max_sil:.4f})")

    # 4) Modelo final con best_k
    labels_final = _fit_predict_agglomerative(X, best_k, linkage, metric)

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

    # Distribución tamaños
    _, counts = np.unique(labels_final, return_counts=True)
    dist_sizes = {int(i): int(c) for i, c in enumerate(counts)}
    logger.info("=== VALIDACIÓN CLUSTERING (AGGLOMERATIVE) ===")
    logger.info(f"k final: {best_k} | linkage: {linkage} | metric: {metric}")
    logger.info(f"Silhouette final       : {sil_final:.4f}" if not np.isnan(sil_final) else "Silhouette final: NaN")
    logger.info(f"Davies-Bouldin final  : {dbi_final:.4f}" if not np.isnan(dbi_final) else "Davies-Bouldin final: NaN")
    logger.info(f"Calinski-Harabasz final: {ch_final:.2f}" if not np.isnan(ch_final) else "Calinski-Harabasz final: NaN")
    logger.info(f"Tamaños de cluster     : {dist_sizes} (min={counts.min()})")

    # 5) Export asignaciones
    df_out = df.copy()
    df_out["Cluster_Agglo"] = labels_final
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Guardado dataset con clusters (agglomerative): {out_path} (filas={len(df_out)})")

    return {
        "k_final": best_k,
        "linkage": linkage,
        "metric": metric,
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
    p = argparse.ArgumentParser(description="Clustering aglomerativo con exploración de k y validación interna.")
    p.add_argument("--in",       dest="inp",   type=str, default=str(PROCESSED_DIR / "productos_features_norm.csv"))
    p.add_argument("--out",      dest="outp",  type=str, default=str(PROCESSED_DIR / "productos_clusters_agglom.csv"))
    p.add_argument("--k-min",    dest="kmin",  type=int, default=3)
    p.add_argument("--k-max",    dest="kmax",  type=int, default=10)
    p.add_argument("--force-k",  dest="kforce", type=int, default=None)
    p.add_argument("--linkage",  dest="linkage", type=str, default="ward", choices=["ward","average","complete","single"])
    p.add_argument("--metric",   dest="metric",  type=str, default="euclidean",
                   help="Distancia para enlaces != ward. Con ward se forzará 'euclidean'.")
    p.add_argument("--sil-sample", dest="silsample", type=int, default=7000)
    p.add_argument("--seed",     dest="seed",   type=int, default=42)

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []

    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> in=%s | out=%s | k=[%d..%d] | force_k=%s | linkage=%s | metric=%s | sil_sample=%d",
                args.inp, args.outp, args.kmin, args.kmax, str(args.kforce), args.linkage, args.metric, args.silsample)
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
            linkage=args.linkage,
            metric=args.metric,
            sil_sample=args.silsample,
            random_state=args.seed,
        )
        logger.info("Proceso finalizado. k_final=%s | silhouette=%s | dbi=%s | ch=%s",
                    info["k_final"], info["silhouette_final"], info["davies_bouldin_final"], info["calinski_harabasz_final"])
        logger.info("Rutas: %s", info["paths"])
    except Exception as e:
        logging.exception(f"Error en clustering aglomerativo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

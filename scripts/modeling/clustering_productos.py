
# ============================================================================
# Script: clustering_productos.py
# Descripción:
#   Aplica clustering K-Means sobre productos_features_norm.csv.
#   - Explora k en un rango (por defecto 3..10), calcula Inercia (codo) y Silhouette.
#   - Selecciona k (máximo silhouette) salvo que se fuerce por CLI.
#   - Ajusta el modelo final y asigna cluster por producto.
#
# Entradas (por defecto):
#   - data/processed/productos_features_norm.csv
#
# Salidas:
#   - data/processed/productos_clusters.csv
#   - reports/inercia_vs_k.csv
#   - reports/silhouette_vs_k.csv
#
# Uso:
#   python scripts/modeling/clustering_productos.py
#   python scripts/modeling/clustering_productos.py \
#       --in data/processed/productos_features_norm.csv \
#       --out data/processed/productos_clusters.csv \
#       --k-min 3 --k-max 10 --force-k 6 --sil-sample 5000
# ============================================================================

from pathlib import Path
import sys, argparse, logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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
logger = logging.getLogger("clustering_productos")

# --- Parche Jupyter: elimina --f=... del kernel para argparse ---------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# ----------------------------------------------------------------------------

# ------------------------------- Utilidades ---------------------------------
NUM_COLS = [
    "d_total", "d_media", "d_std", "cv", "p95", "mediana",
    "precio_medio", "PC1", "PC2", "PC3"
]

def _check_columns(df: pd.DataFrame, cols: list[str]):
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en el dataset de entrada: {faltan}")

def _sample_for_silhouette(X: np.ndarray, max_n: int, random_state: int = 42):
    n = X.shape[0]
    if n <= max_n:
        return X, np.arange(n)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_n, replace=False)
    return X[idx], idx

# ---------------------------------- Core ------------------------------------
def explorar_y_clusterizar(in_path: Path,
                           out_path: Path,
                           k_min: int = 3,
                           k_max: int = 10,
                           force_k: int | None = None,
                           sil_sample: int = 5000,
                           random_state: int = 42,
                           n_init: int = 20):

    # 1) Cargar dataset normalizado
    in_path  = Path(in_path)
    out_path = Path(out_path)
    logger.info(f"Cargando features normalizadas: {in_path}")
    df = pd.read_csv(in_path)

    if "Product_ID" not in df.columns:
        raise KeyError("Se requiere columna 'Product_ID' en el dataset de entrada.")
    _check_columns(df, NUM_COLS)

    X = df[NUM_COLS].astype(float).values
    n, d = X.shape
    logger.info(f"Dimensiones: n={n}, d={d}")

    # 2) Explorar rango de k (si no se fuerza)
    ks = list(range(max(2, k_min), max(k_min, k_max) + 1))
    res_inercia = []
    res_sil = []

    for k in ks:
        logger.info(f"[Exploración] Ajustando KMeans con k={k} ...")
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        res_inercia.append({"k": k, "inercia": inertia})

        # Silhouette (requiere k>=2, ya garantizado) — muestreo opcional por eficiencia
        X_sil, idx_sil = _sample_for_silhouette(X, max_n=sil_sample, random_state=random_state)
        lab_sil = labels[idx_sil]
        try:
            sil = float(silhouette_score(X_sil, lab_sil, metric="euclidean"))
        except Exception as e:
            logger.warning(f"Silhouette falló para k={k}: {e}")
            sil = np.nan
        res_sil.append({"k": k, "silhouette": sil})

    df_inercia = pd.DataFrame(res_inercia)
    df_sil = pd.DataFrame(res_sil)

    # Guardar reportes
    path_inercia = REPORTS_DIR / "inercia_vs_k.csv"
    path_sil = REPORTS_DIR / "silhouette_vs_k.csv"
    df_inercia.to_csv(path_inercia, index=False)
    df_sil.to_csv(path_sil, index=False)
    logger.info(f"Guardado: {path_inercia}")
    logger.info(f"Guardado: {path_sil}")

    # 3) Selección de k
    if force_k is not None:
        best_k = int(force_k)
        logger.info(f"Usando k forzado por CLI: k={best_k}")
    else:
        # Elegir k por máximo silhouette (ignorando NaN); si empate, el menor k
        df_sil_valid = df_sil.dropna(subset=["silhouette"])
        if df_sil_valid.empty:
            # fallback: si no hay silhouette válido, usar punto medio del rango
            best_k = int(np.median(ks))
            logger.warning(f"No se pudo calcular silhouette; usando k={best_k} (mediana del rango).")
        else:
            max_sil = df_sil_valid["silhouette"].max()
            candidatos = df_sil_valid.loc[df_sil_valid["silhouette"] == max_sil, "k"].tolist()
            best_k = min(candidatos)
            logger.info(f"Selección automática por silhouette: k={best_k} (silhouette={max_sil:.4f})")

    # 4) Modelo final con best_k
    logger.info(f"Ajustando modelo final KMeans con k={best_k} ...")
    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=n_init)
    labels_final = km_final.fit_predict(X)

    # Validación silhouette final (completo o muestreado si es muy grande)
    X_sil_final, idx_sil_final = _sample_for_silhouette(X, max_n=sil_sample, random_state=random_state)
    lab_sil_final = labels_final[idx_sil_final]
    try:
        sil_final = float(silhouette_score(X_sil_final, lab_sil_final, metric="euclidean"))
    except Exception as e:
        logger.warning(f"Silhouette final falló para k={best_k}: {e}")
        sil_final = np.nan

    # 5) Distribución de tamaños de cluster
    _, counts = np.unique(labels_final, return_counts=True)
    dist_sizes = {int(i): int(c) for i, c in enumerate(counts)}
    min_size = counts.min()
    logger.info("=== VALIDACIÓN CLUSTERING ===")
    logger.info(f"k final: {best_k}")
    logger.info(f"Silhouette (final): {sil_final:.4f}" if not np.isnan(sil_final) else "Silhouette (final): NaN")
    logger.info(f"Tamaños de cluster: {dist_sizes} (min={min_size})")

    # 6) Export asignaciones
    df_clusters = df.copy()
    df_clusters["Cluster"] = labels_final
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clusters.to_csv(out_path, index=False)
    logger.info(f"Guardado dataset con clusters: {out_path} (filas={len(df_clusters)}, cols={df_clusters.shape[1]})")

    # 7) Devolver info clave
    return {
        "k_final": best_k,
        "silhouette_final": sil_final,
        "sizes": dist_sizes,
        "paths": {
            "clusters": str(out_path),
            "inercia_vs_k": str(path_inercia),
            "silhouette_vs_k": str(path_sil),
        },
    }

# ------------------------------------ CLI -----------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Clustering de productos (K-Means) con exploración de k e informes.")
    p.add_argument("--in",       dest="inp",  type=str, default=str(PROCESSED_DIR / "productos_features_norm.csv"))
    p.add_argument("--out",      dest="outp", type=str, default=str(PROCESSED_DIR / "productos_clusters.csv"))
    p.add_argument("--k-min",    dest="kmin", type=int, default=3)
    p.add_argument("--k-max",    dest="kmax", type=int, default=10)
    p.add_argument("--force-k",  dest="kforce", type=int, default=None, help="Forzar k concreto. Si se indica, salta la selección automática.")
    p.add_argument("--sil-sample", dest="silsample", type=int, default=5000,
                   help="Máximo de observaciones para calcular silhouette (muestreo aleatorio si N>valor).")
    p.add_argument("--seed",     dest="seed", type=int, default=42)
    p.add_argument("--n-init",   dest="ninit", type=int, default=20)

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []

    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> in=%s | out=%s | k=[%d..%d] | force_k=%s | sil_sample=%d | seed=%d | n_init=%d",
                args.inp, args.outp, args.kmin, args.kmax, str(args.kforce), args.silsample, args.seed, args.ninit)
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
            sil_sample=args.silsample,
            random_state=args.seed,
            n_init=args.ninit
        )
        logger.info("Proceso finalizado. k_final=%s | silhouette_final=%s", info["k_final"], info["silhouette_final"])
        logger.info("Rutas: %s", info["paths"])
    except Exception as e:
        logging.exception(f"Error en clustering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

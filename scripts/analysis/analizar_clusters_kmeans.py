# ============================================================================
# Script: analizar_clusters_kmeans.py
#
# Descripción:
#   Genera un análisis de interpretación para los clusters obtenidos con
#   K-Means sobre el dataset de productos. Calcula estadísticos por cluster
#   (demanda total/media/variabilidad, p95, mediana, precio y PCs) y, si es
#   posible, la distribución de categorías por cluster.
#
#
#
# Entradas:
#   --in        (default: data/processed/productos_clusters.csv)
#                Dataset con asignación de Cluster por Product_ID y features.
#   --lookup    (default: data/processed/demanda_filtrada_enriquecida_sin_nans.csv)
#                Fichero de referencia para traer la categoría si falta.
#
# Salidas:
#   reports/clusters_kmeans_stats.csv
#     → tabla de estadísticos por cluster (n_productos, d_total/d_media/d_std,
#       cv, p95, mediana, precio_medio, PC1, PC2, PC3; según disponibilidad).
#
#   reports/clusters_kmeans_categorias.csv   [si hay columna de categoría]
#     → distribución de categorías × cluster.
#
# Validaciones:
#   - Comprueba columnas mínimas: Product_ID y Cluster.
#   - Avisa si faltan features numéricas esperadas (no bloquea).
#   - Loguea tamaños de cluster y cobertura de categoría al anexar.
#
# Uso (ejemplo):
#   python scripts/analysis/analizar_clusters_kmeans.py \
#       --in data/processed/productos_clusters.csv \
#       --lookup data/processed/demanda_filtrada_enriquecida_sin_nans.csv
#
# Resultado esperado:
#   - Visión clara de tamaños y perfiles de cluster.
#   - Foto de coherencia por categorías (si aplica).
#   - Insumo para la decisión final del método de clustering y para negocio.
# =============================================================================

from pathlib import Path
import sys
import argparse
import logging
import pandas as pd
import numpy as np

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analizar_clusters_kmeans")

# Parche Jupyter
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]

NUM_COLS = [
    "d_total", "d_media", "d_std", "cv", "p95", "mediana",
    "precio_medio", "PC1", "PC2", "PC3"
]
CAT_COL_CANDIDATES = ["Categoria_reducida", "Categoria"]

def _check_required(df: pd.DataFrame):
    missing = [c for c in ["Product_ID", "Cluster"] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas obligatorias en el dataset de entrada: {missing}")
    num_missing = [c for c in NUM_COLS if c not in df.columns]
    if num_missing:
        logger.warning("No se encontraron algunas columnas numéricas esperadas: %s", num_missing)

def _pick_category_column(df: pd.DataFrame) -> str | None:
    for c in CAT_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _agg_stats(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {"Product_ID": "nunique"}
    for c in NUM_COLS:
        if c in df.columns:
            agg_dict[c] = ["mean", "median", "std", "min", "max"]
    stats = df.groupby("Cluster").agg(agg_dict)

    # aplanado robusto
    new_cols = []
    for c in stats.columns:
        if isinstance(c, tuple):
            flat = "_".join([x for x in c if x])
            flat = flat.replace("Product_ID_", "n_productos_")
            new_cols.append(flat)
        else:
            new_cols.append(str(c))
    stats.columns = new_cols
    stats = stats.reset_index()

    for col in stats.columns:
        if col != "Cluster" and stats[col].dtype.kind in "fc":
            stats[col] = stats[col].astype(float).round(4)
    return stats

def _cats_table(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    tab = (df.groupby([cat_col, "Cluster"])
             .size()
             .reset_index(name="count"))
    pivot = tab.pivot(index=cat_col, columns="Cluster", values="count").fillna(0).astype(int)
    pivot = pivot.sort_index().reset_index()
    return pivot

def _attach_category_from_lookup(df: pd.DataFrame, lookup_path: Path) -> tuple[pd.DataFrame, str, float]:
    """
    Si el df no tiene categoria, intenta anexarla desde lookup (demanda enriquecida).
    Usa la moda por Product_ID. Devuelve (df_mod, nombre_col_usada, cobertura_pct).
    """
    lookup_path = Path(lookup_path)
    if not lookup_path.exists():
        logger.warning("Lookup de categoría no encontrado: %s", lookup_path)
        return df, "", 0.0

    lk = pd.read_csv(lookup_path)
    cat_col = None
    for c in CAT_COL_CANDIDATES:
        if c in lk.columns:
            cat_col = c
            break
    if cat_col is None:
        logger.warning("El lookup no tiene columnas de categoría (%s).", CAT_COL_CANDIDATES)
        return df, "", 0.0

    # moda por Product_ID
    lk = lk[["Product_ID", cat_col]].dropna()
    # si un product_id tiene varias, quedarse con la moda
    lk = (lk
          .groupby("Product_ID")[cat_col]
          .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
          .reset_index())

    before_missing = df["Product_ID"].nunique()
    df = df.merge(lk, on="Product_ID", how="left")
    covered = df[cat_col].notna().sum()
    coverage_pct = round(100 * covered / len(df), 2)

    logger.info("Categoría anexada desde lookup '%s' -> cobertura filas: %.2f%%", cat_col, coverage_pct)
    return df, cat_col, coverage_pct

def analizar_clusters_kmeans(in_path: Path,
                             out_stats: Path,
                             out_cats: Path | None = None,
                             lookup_path: Path | None = None):
    in_path  = Path(in_path)
    out_stats = Path(out_stats)
    out_cats = Path(out_cats) if out_cats else (REPORTS_DIR / "clusters_kmeans_categorias.csv")

    logger.info(f"Cargando dataset con clusters K-Means: {in_path}")
    df = pd.read_csv(in_path)
    _check_required(df)

    # Si no hay categoría, intentar adjuntarla desde lookup
    cat_col = _pick_category_column(df)
    if cat_col is None and lookup_path:
        df, cat_col, coverage = _attach_category_from_lookup(df, lookup_path)

    n_total = len(df)
    sizes = df["Cluster"].value_counts().sort_index().to_dict()
    logger.info("Productos totales: %d", n_total)
    logger.info("Tamaños de cluster: %s (min=%s, max=%s)",
                sizes, min(sizes.values()), max(sizes.values()))

    stats = _agg_stats(df)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_stats, index=False)
    logger.info(f"Guardado descriptivos por cluster: {out_stats}")

    if cat_col:
        cats = _cats_table(df, cat_col)
        out_cats.parent.mkdir(parents=True, exist_ok=True)
        cats.to_csv(out_cats, index=False)
        logger.info(f"Guardado distribución categorías × cluster: {out_cats} (columna usada: {cat_col})")
    else:
        logger.info("No se encontró columna de categoría (ni en dataset ni en lookup). Se omite distribución por categorías.")

    return {
        "n_total": n_total,
        "sizes": sizes,
        "paths": {
            "stats": str(out_stats),
            "categorias": str(out_cats) if cat_col else "",
        },
        "category_col_used": cat_col or "",
    }

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Análisis descriptivo de clusters K-Means (interpretación de negocio).")
    p.add_argument("--in",        dest="inp",      type=str, default=str(PROCESSED_DIR / "productos_clusters.csv"))
    p.add_argument("--out-stats", dest="outstats", type=str, default=str(REPORTS_DIR / "clusters_kmeans_stats.csv"))
    p.add_argument("--out-cats",  dest="outcats",  type=str, default=str(REPORTS_DIR / "clusters_kmeans_categorias.csv"))
    p.add_argument("--lookup",    dest="lookup",   type=str, default=str(PROCESSED_DIR / "demanda_filtrada_enriquecida_sin_nans.csv"))

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []
    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> in=%s | out_stats=%s | out_cats=%s | lookup=%s",
                args.inp, args.outstats, args.outcats, args.lookup)
    return args

def main():
    args = parse_args()
    try:
        info = analizar_clusters_kmeans(
            in_path=Path(args.inp),
            out_stats=Path(args.outstats),
            out_cats=Path(args.outcats),
            lookup_path=Path(args.lookup) if args.lookup else None,
        )
        logger.info("Análisis finalizado. Tamaños: %s | Rutas: %s", info["sizes"], info["paths"])
        if info["category_col_used"]:
            logger.info("Categoría utilizada para distribución: %s", info["category_col_used"])
    except Exception as e:
        logging.exception(f"Error analizando clusters K-Means: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# ============================================================================
# Script: analizar_clusters_gmm.py
#
# Descripción:
#   Este script genera un análisis de interpretación para los clusters
#   obtenidos con GMM a partir del archivo `data/processed/productos_clusters_gmm.csv`.
#   Calcula métricas descriptivas por cluster (demanda, variabilidad, precio, PCs)
#   y revisa la distribución de categorías. El objetivo es evaluar la
#   coherencia e interpretabilidad de los clusters y facilitar la decisión
#   final sobre el método de clustering a emplear.
#
# Flujo del pipeline:
#   1. Carga dataset con asignación de cluster (columna `Cluster`).
#   2. Valida existencia de columnas clave y de features numéricas esperadas.
#   3. Calcula descriptivos por cluster (totales y estadísticos).
#   4. Si no existe columna de categoría, la anexa automáticamente desde
#      `demanda_filtrada_enriquecida_sin_nans.csv` usando la moda por `Product_ID`.
#   5. Genera dos reportes:
#        - reports/clusters_gmm_stats.csv      → descriptivos por cluster
#        - reports/clusters_gmm_categorias.csv → distribución categorías × cluster
#
# Inputs esperados:
#   - data/processed/productos_clusters_gmm.csv
#   - data/processed/demanda_filtrada_enriquecida_sin_nans.csv (para lookup categoría)
#
# Outputs generados:
#   - reports/clusters_gmm_stats.csv
#   - reports/clusters_gmm_categorias.csv
#
# Dependencias:
#   pip install pandas numpy
#
# =============================================================================

from pathlib import Path
import argparse
import sys
import logging
import pandas as pd
import numpy as np

# ----------------------------- Logging ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analizar_clusters_gmm")

# -------------------------- Utilidades ---------------------------------------
POSSIBLE_CLUSTER_COLS = ["Cluster_GMM", "Cluster", "cluster"]
POSSIBLE_CATEGORY_COLS = ["Categoria_reducida", "Categoria", "categoria_reducida", "categoria"]

NUM_FEATURES_CANDIDATES = [
    # demanda agregada / precio / PCs (cuando estén)
    "d_total", "d_total_mean", "d_total_median", "d_total_std", "d_total_min", "d_total_max",
    "d_media", "d_media_mean", "d_media_median", "d_media_std",
    "cv", "p95", "mediana",
    "precio_medio",
    "PC1", "PC2", "PC3",
]

def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _attach_category_if_missing(df: pd.DataFrame, lookup_path: Path) -> tuple[pd.DataFrame, float, str]:
    """
    Si el DF no trae columna de categoría, intenta anexarla por Product_ID usando lookup.
    Devuelve (df_out, coverage, used_colname)
    """
    cat_col = _find_first_col(df, POSSIBLE_CATEGORY_COLS)
    if cat_col:
        return df, 1.0, cat_col  # ya trae categoría

    if "Product_ID" not in df.columns:
        logger.warning("No existe columna 'Product_ID' para poder anexar categoría desde lookup.")
        return df, 0.0, ""

    if not lookup_path.exists():
        logger.warning(f"Lookup no disponible: {lookup_path}")
        return df, 0.0, ""

    logger.info(f"Anexando categoría desde lookup: {lookup_path}")
    lk = pd.read_csv(lookup_path)
    lk_cat_col = _find_first_col(lk, POSSIBLE_CATEGORY_COLS)
    if not lk_cat_col:
        logger.warning("El lookup no contiene columna de categoría.")
        return df, 0.0, ""

    use_cols = ["Product_ID", lk_cat_col]
    lk = lk[use_cols].drop_duplicates()
    before = len(df)
    df_out = df.merge(lk, how="left", on="Product_ID")
    coverage = df_out[lk_cat_col].notna().mean()
    logger.info(f"Cobertura categoría tras lookup: {coverage:.2%}  (filas={before})")
    return df_out, float(coverage), lk_cat_col

def _agg_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptivos por cluster sobre todas las numéricas disponibles de NUM_FEATURES_CANDIDATES.
    Siempre incluye n_productos_nunique (por Product_ID si existe).
    """
    num_cols = [c for c in NUM_FEATURES_CANDIDATES if c in df.columns]
    agg_map = {}
    for c in num_cols:
        agg_map[c] = ["mean", "median", "std", "min", "max"]

    # n_productos por cluster
    if "Product_ID" in df.columns:
        base = df.groupby("Cluster", as_index=False).agg(n_productos_nunique=("Product_ID", "nunique"))
    else:
        base = df.groupby("Cluster", as_index=False).size().rename(columns={"size": "n_productos_nunique"})

    if agg_map:
        stats = (
            df.groupby("Cluster")
              .agg(agg_map)
        )
        # aplanar nombres multi-índice
        stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
        stats = stats.reset_index()
        out = base.merge(stats, how="left", on="Cluster")
    else:
        logger.warning("No hay métricas numéricas disponibles para calcular descriptivos; se devuelve solo conteos.")
        out = base.copy()

    # Ordenar por Cluster (numérico si aplica)
    try:
        out = out.sort_values("Cluster")
    except Exception:
        pass
    return out

def _cats_distribution(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """
    Devuelve tabla pivot de categorías × cluster con conteos.
    """
    tabla = (
        df.pivot_table(index=cat_col, columns="Cluster", values="Product_ID",
                       aggfunc="nunique", fill_value=0, observed=True)
        .reset_index()
    )
    return tabla

# -------------------------- Lógica principal ----------------------------------
def analizar_clusters_gmm(
    in_path: Path,
    out_stats: Path | None = None,
    out_cats: Path | None = None,
    lookup_path: Path | None = None,
) -> dict:
    logger.info(f"Cargando dataset con clusters GMM: {in_path}")
    df = pd.read_csv(in_path)

    # 1) Resolver columna de cluster
    cluster_col = _find_first_col(df, POSSIBLE_CLUSTER_COLS)
    if cluster_col is None:
        raise KeyError(f"Falta columna de cluster. No se encontró ninguna de {POSSIBLE_CLUSTER_COLS} en {in_path}")
    if cluster_col != "Cluster":
        df = df.rename(columns={cluster_col: "Cluster"})
        logger.info(f"Columna de cluster detectada '{cluster_col}' → renombrada a 'Cluster'.")

    # 2) Validaciones suaves
    if "Product_ID" not in df.columns:
        logger.warning("No existe columna 'Product_ID'; algunos descriptores serán aproximados.")
    logger.info(f"Productos totales: {len(df)}")

    # 3) Anexar categoría si no existe
    cat_col = _find_first_col(df, POSSIBLE_CATEGORY_COLS)
    coverage = None
    if not cat_col and lookup_path is not None:
        df, coverage, cat_col = _attach_category_if_missing(df, lookup_path)

    # 4) Descriptivos y categorías
    sizes = df["Cluster"].value_counts().sort_index().to_dict()
    logger.info(f"Tamaños de cluster: {sizes}")

    stats = _agg_stats(df)

    if out_stats:
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(out_stats, index=False)
        logger.info(f"Guardado descriptivos por cluster: {out_stats}")

    cats_path = None
    if cat_col:
        cats = _cats_distribution(df, cat_col)
        if out_cats:
            out_cats.parent.mkdir(parents=True, exist_ok=True)
            cats.to_csv(out_cats, index=False)
            cats_path = out_cats
            logger.info(f"Guardada distribución categorías × cluster: {out_cats}")
        else:
            logger.info("Se calculó distribución por categorías, pero no se indicó --outcats; no se guardó.")
    else:
        logger.info("No se dispone de columna de categoría; se omite tabla categoría × cluster.")

    logger.info(
        "Análisis finalizado. Tamaños: %s | Rutas: {'stats': '%s', 'cats': '%s'}",
        sizes, str(out_stats) if out_stats else "", str(cats_path) if cats_path else ""
    )
    if coverage is not None:
        logger.info(f"Categoría anexada desde lookup con cobertura: {coverage:.2%}")

    return {
        "sizes": sizes,
        "stats_path": str(out_stats) if out_stats else "",
        "cats_path": str(cats_path) if cats_path else "",
        "category_coverage": coverage,
    }

# ----------------------------- CLI / MAIN ------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Analiza clusters GMM y genera descriptivos y distribución por categoría.")
    # Paths
    p.add_argument("--inp", type=str, default="data/processed/productos_clusters_gmm.csv",
                   help="Ruta del CSV con los clusters GMM (debe incluir Product_ID y Cluster_GMM/Cluster).")
    p.add_argument("--outstats", type=str, default="reports/clusters_gmm_stats.csv",
                   help="Ruta de salida para los descriptivos por cluster.")
    p.add_argument("--outcats", type=str, default="reports/clusters_gmm_categorias.csv",
                   help="Ruta de salida para la tabla categorías × cluster.")
    p.add_argument("--lookup", type=str, default="data/processed/demanda_filtrada_enriquecida_sin_nans.csv",
                   help="Lookup para anexar categoría si no está en el dataset (por Product_ID).")

    # Compatibilidad Notebook: si no pasan argv y estamos en Jupyter, ignorar args del kernel
    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []

    args, _ = p.parse_known_args(argv)
    return args

def main(argv=None):
    args = _parse_args(argv)
    info = analizar_clusters_gmm(
        in_path=Path(args.inp),
        out_stats=Path(args.outstats) if args.outstats else None,
        out_cats=Path(args.outcats) if args.outcats else None,
        lookup_path=Path(args.lookup) if args.lookup else None,
    )
    logger.info("Proceso finalizado: %s", info)

if __name__ == "__main__":
    main()
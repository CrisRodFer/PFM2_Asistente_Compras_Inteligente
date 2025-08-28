# ============================================================================
# Script: analizar_clusters_gmm.py
#
# Descripción:
# Caracteriza e interpreta los clusters generados con GMM a nivel producto.
#
#
#
# Entradas:
#   -- Entrada principal: data/processed/productos_clusters_gmm.csv
#              
#   -- Lookup de categorías: data/processed/demanda_filtrada_enriquecida_sin_nans.csv
#      para anexar 'Categoria_reducida' por Product_ID (moda).   
#
# Salidas:
#   reports/clusters_gmm_stats.csv
#
#   reports/clusters_gmm_categorias.csv  (si existe columna de categoría)
#

#
# Uso (ejemplo):
#      python scripts/analysis/analizar_clusters_gmm.py \
#        --in data/processed/productos_clusters_gmm.csv \
#        --out-stats reports/clusters_gmm_stats.csv \
#       --out-cats reports/clusters_gmm_categorias.csv
# =============================================================================
from pathlib import Path
import argparse
import logging
import sys
import numpy as np
import pandas as pd

# ----------------------------- 0. CONFIG -----------------------------------
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    ROOT_DIR = Path().resolve()

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LOOKUP_CAT = PROCESSED_DIR / "demanda_filtrada_enriquecida_sin_nans.csv"  # para anexar Categoria_reducida

# ----------------------------- 1. LOGGING ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analizar_clusters_gmm")

# ----------------------------- 2. UTILS ------------------------------------
NUMERIC_CANDIDATES = [
    "d_total", "d_media", "d_std", "cv", "p95", "mediana",
    "precio_medio", "PC1", "PC2", "PC3"
]
CAT_COL_CANDIDATES = ["Categoria_reducida", "Categoria"]

def _safe_numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return cols

def _attach_category_if_needed(df: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    """Anexa 'Categoria_reducida' desde lookup por moda de Product_ID si df no tiene categoría."""
    for c in CAT_COL_CANDIDATES:
        if c in df.columns:
            return df, 1.0, c  # ya viene, cobertura total

    if not LOOKUP_CAT.exists():
        logger.warning("No hay lookup de categorías. No se anexará categoría.")
        return df, 0.0, ""

    logger.info(f"Anexando categorías desde lookup: {LOOKUP_CAT}")
    lu = pd.read_csv(LOOKUP_CAT, low_memory=False)

    cat_col = next((c for c in CAT_COL_CANDIDATES if c in lu.columns), None)
    if cat_col is None or "Product_ID" not in lu.columns:
        logger.warning("Lookup sin columnas esperadas (categoria/Product_ID). Se omite.")
        return df, 0.0, ""

    moda = (
        lu.dropna(subset=[cat_col])
          .groupby("Product_ID")[cat_col]
          .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
          .rename("Categoria_reducida")
          .reset_index()
    )
    df = df.merge(moda, on="Product_ID", how="left")
    cobertura = 1.0 - df["Categoria_reducida"].isna().mean()
    logger.info(f"Cobertura de categoría anexada: {cobertura:.2%}")
    return df, cobertura, "Categoria_reducida"

def _agg_stats(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = _safe_numeric_cols(df)
    if not num_cols:
        raise ValueError("No se encontraron columnas numéricas para agregar.")

    aggs = {c: ["mean", "median", "std", "min", "max"] for c in num_cols}
    out = df.groupby("Cluster").agg(aggs).sort_index()
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]  # aplanar MultiIndex
    out.insert(0, "n_productos_nunique", df.groupby("Cluster")["Product_ID"].nunique())
    out = out.reset_index()
    return out

def _cats_table(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    return (
        df.groupby([cat_col, "Cluster"])["Product_ID"]
          .nunique()
          .unstack(fill_value=0)
          .reset_index()
          .rename_axis(None, axis=1)
    )

# ----------------------------- 3. CORE --------------------------------------
def analizar_clusters_gmm(in_path: Path,
                          out_stats: Path | None,
                          out_cats: Path | None) -> dict:
    logger.info(f"Cargando dataset con clusters GMM: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)

    for col in ["Product_ID", "Cluster"]:
        if col not in df.columns:
            raise KeyError(f"Falta columna obligatoria '{col}' en {in_path}")

    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").astype("Int64")
    logger.info(f"Productos totales: {df['Product_ID'].nunique()}")

    # Anexar categoría si no existe
    df, cobertura, cat_col = _attach_category_if_needed(df)

    sizes = df.groupby("Cluster")["Product_ID"].nunique().to_dict()
    logger.info(f"Tamaños de cluster: {sizes} (min={min(sizes.values())}, max={max(sizes.values())})")

    stats = _agg_stats(df)
    stats_path = REPORTS_DIR / "clusters_gmm_stats.csv" if out_stats is None else Path(out_stats)
    stats.to_csv(stats_path, index=False)
    logger.info(f"Guardado descriptivos por cluster: {stats_path}")

    cats_path = ""
    if cat_col:
        cats = _cats_table(df, cat_col)
        cats_path_obj = REPORTS_DIR / "clusters_gmm_categorias.csv" if out_cats is None else Path(out_cats)
        cats.to_csv(cats_path_obj, index=False)
        cats_path = str(cats_path_obj)
        logger.info(f"Guardado distribución categorías × cluster: {cats_path}")
        logger.info(f"Categoría utilizada para distribución: {cat_col}")
    else:
        logger.info("No se encontró columna de categoría. Se omite distribución por categorías.")

    return {
        "sizes": sizes,
        "coverage_categoria": cobertura,
        "stats_path": str(stats_path),
        "cats_path": cats_path,
        "cat_col": cat_col,
    }

# ----------------------------- 4. CLI ---------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Análisis de clusters GMM a nivel producto.")
    p.add_argument("--in", dest="inp", type=str,
                   default=str(PROCESSED_DIR / "productos_clusters_gmm.csv"),
                   help="Ruta del CSV con Product_ID y Cluster (GMM).")
    p.add_argument("--out-stats", dest="outstats", type=str, default="",
                   help="Ruta de salida para clusters_gmm_stats.csv (opcional).")
    p.add_argument("--out-cats", dest="outcats", type=str, default="",
                   help="Ruta de salida para clusters_gmm_categorias.csv (opcional).")

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []

    args, _ = p.parse_known_args(argv)
    return args

def main():
    args = parse_args()
    info = analizar_clusters_gmm(
        in_path=Path(args.inp),
        out_stats=Path(args.outstats) if args.outstats else None,
        out_cats=Path(args.outcats) if args.outcats else None,
    )
    logging.info(
        "Análisis finalizado. Tamaños: %s | Cobertura categoría: %.2f | Rutas: %s",
        info["sizes"], info["coverage_categoria"],
        {"stats": info["stats_path"], "cats": info["cats_path"]}
    )

if __name__ == "__main__":
    main()
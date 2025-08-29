
# =============================================================================
# Script: asignar_clusters_a_demanda.py
# Descripción:
#   Enriquece la demanda desagregada con el identificador de cluster (K-Means)
#   y elimina las filas de productos que no tengan cluster asignado.
#
# Flujo del pipeline:
#   1) Carga la demanda desagregada final de Fase 3
#      (demanda_filtrada_enriquecida_sin_nans.csv).
#   2) Carga el lookup oficial de clusters (productos_clusters.csv).
#   3) Estandariza claves y valida columnas.
#   4) Merge (LEFT) demanda × clusters por Product_ID.
#   5) Filtra filas con Cluster nulo (sin cluster) y (opcional) etiqueta outliers.
#   6) Exporta:
#        - data/processed/demanda_con_clusters.csv
#        - reports/productos_sin_cluster_en_demanda.csv (diagnóstico)
#
# Inputs (por defecto):
#   - data/processed/demanda_filtrada_enriquecida_sin_nans.csv
#   - data/processed/productos_clusters.csv
#   - (opcional) data/processed/outliers_dbscan.csv  -> marca is_outlier=1
#
# Outputs (por defecto):
#   - data/processed/demanda_con_clusters.csv
#   - reports/productos_sin_cluster_en_demanda.csv
#
# Dependencias:
#   - pandas
#   - numpy
#
# Ejemplos de uso:
#   python scripts/transform/asignar_clusters_a_demanda.py
#   python scripts/transform/asignar_clusters_a_demanda.py \
#       --demanda data/processed/demanda_filtrada_enriquecida_sin_nans.csv \
#       --clusters data/processed/productos_clusters.csv \
#       --out data/processed/demanda_con_clusters.csv \
#       --out-removed reports/productos_sin_cluster_en_demanda.csv \
#       --outliers data/processed/outliers_dbscan.csv
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np

# Este script vive en scripts/transform -> subimos 2 niveles para llegar a la raíz
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# Rutas por defecto
DEF_DEMANDA = PROCESSED_DIR / "demanda_filtrada_enriquecida_sin_nans.csv"
DEF_CLUSTERS = PROCESSED_DIR / "productos_clusters.csv"
DEF_OUT = PROCESSED_DIR / "demanda_con_clusters.csv"
DEF_OUT_REMOVED = REPORTS_DIR / "productos_sin_cluster_en_demanda.csv"

# ==== 1. LOGGING ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("asignar_clusters_a_demanda")


# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    """Crea directorios si no existen."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _standardize_product_id(df: pd.DataFrame, col: str = "Product_ID") -> pd.DataFrame:
    """
    Estandariza Product_ID a string para evitar mismatches en merge.
    """
    if col not in df.columns:
        raise KeyError(f"Falta columna obligatoria '{col}'.")
    df[col] = df[col].astype(str).str.strip()
    return df


def _detect_cluster_col(df: pd.DataFrame) -> str:
    """
    Detecta una columna de cluster razonable y devuelve su nombre.
    Acepta variantes comunes y normaliza luego a 'Cluster'.
    """
    candidates = ["Cluster", "cluster", "Cluster_KMeans", "Cluster_GMM", "ClusterId", "label"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No se encontró columna de cluster en el lookup. Revisar nombres (p. ej. {candidates}).")


def _load_outliers_set(path: Path | None) -> set[str]:
    """
    Carga un conjunto de Product_ID marcados como outliers desde un CSV (opcional).
    Solo requiere la columna Product_ID.
    """
    if path is None:
        return set()
    if not Path(path).exists():
        log.warning("Ruta de outliers no existe: %s (se omite)", path)
        return set()
    df = pd.read_csv(path, low_memory=False)
    if "Product_ID" not in df.columns:
        log.warning("El CSV de outliers no trae 'Product_ID'. Se omite etiquetado de outliers.")
        return set()
    s = set(df["Product_ID"].astype(str).str.strip().unique().tolist())
    log.info("Outliers cargados: %d Product_ID", len(s))
    return s


# ==== 3. LÓGICA PRINCIPAL =====================================================
def asignar_clusters(
    demanda_path: Path,
    clusters_path: Path,
    out_path: Path,
    out_removed_path: Path,
    outliers_path: Path | None = None,
) -> dict:
    """
    Asigna el Cluster a la demanda desagregada y filtra filas sin cluster.

    Parámetros
    ----------
    demanda_path : Path
        CSV con demanda desagregada final (Fase 3), ya limpia de NaNs.
    clusters_path : Path
        CSV con el lookup oficial de clusters por Product_ID (K-Means k=4).
    out_path : Path
        Ruta de salida para la demanda con clusters asignados y filtrada.
    out_removed_path : Path
        Ruta de salida para el reporte de productos sin cluster encontrados en demanda.
    outliers_path : Path | None
        (Opcional) CSV con Product_ID de outliers (DBSCAN) para marcar is_outlier=1.

    Retorna
    -------
    dict
        Resumen con métricas de ejecución (filas antes/después, % drop, etc.).
    """
    # --- Cargar entradas
    log.info("Cargando demanda: %s", demanda_path)
    dem = pd.read_csv(demanda_path, low_memory=False)

    log.info("Cargando clusters: %s", clusters_path)
    clu = pd.read_csv(clusters_path, low_memory=False)

    # --- Estandarizar claves y columnas
    dem = _standardize_product_id(dem, "Product_ID")
    clu = _standardize_product_id(clu, "Product_ID")

    cluster_col = _detect_cluster_col(clu)
    if cluster_col != "Cluster":
        clu = clu.rename(columns={cluster_col: "Cluster"})
        log.info("Columna de cluster '%s' renombrada a 'Cluster'.", cluster_col)

    # Validaciones
    if "Cluster" not in clu.columns:
        raise KeyError("El lookup de clusters debe tener columna 'Cluster'.")

    # Asegurar unicidad por Product_ID en el lookup
    clu = clu[["Product_ID", "Cluster"]].drop_duplicates()
    dup_cnt = clu["Product_ID"].value_counts().gt(1).sum()
    if dup_cnt > 0:
        log.warning("Se detectaron %d Product_ID con múltiples clusters en el lookup; se conservará el primero.", dup_cnt)
        clu = clu.drop_duplicates(subset=["Product_ID"], keep="first")

    # --- Merge demanda × clusters
    n_rows_before = len(dem)
    n_products_before = dem["Product_ID"].nunique()

    dem_merged = dem.merge(clu, on="Product_ID", how="left")

    # --- Reporte de filas/productos sin cluster
    mask_no_cluster = dem_merged["Cluster"].isna()
    removed = dem_merged.loc[mask_no_cluster, ["Product_ID"]].copy()
    if not removed.empty:
        removed["reason"] = "sin_cluster"
        # Consolidar a nivel producto
        removed_products = (
            removed.groupby("Product_ID", as_index=False)
                   .agg(n_rows=("Product_ID", "size"),
                        reason=("reason", "first"))
        )
        ensure_dirs(out_removed_path.parent)
        removed_products.to_csv(out_removed_path, index=False)
        log.info("Guardado reporte de productos sin cluster: %s (n=%d)", out_removed_path, len(removed_products))
    else:
        log.info("No se encontraron filas sin cluster en la demanda.")

    # --- Filtrado: quedarnos solo con filas con cluster asignado
    dem_ok = dem_merged.loc[~mask_no_cluster].copy()

    # --- (Opcional) Etiquetado de outliers
    outliers_set = _load_outliers_set(outliers_path)
    if outliers_set:
        dem_ok["is_outlier"] = dem_ok["Product_ID"].isin(outliers_set).astype(int)
        log.info("Marcados is_outlier=1 para %d filas.", int(dem_ok["is_outlier"].sum()))
    else:
        dem_ok["is_outlier"] = 0

    # --- Exportar
    ensure_dirs(out_path.parent)
    dem_ok.to_csv(out_path, index=False)

    # --- Métricas de salida
    n_rows_after = len(dem_ok)
    n_products_after = dem_ok["Product_ID"].nunique()
    n_rows_dropped = int(mask_no_cluster.sum())
    pct_rows_dropped = (n_rows_dropped / max(1, n_rows_before)) * 100.0
    n_products_removed = removed["Product_ID"].nunique() if not removed.empty else 0

    log.info("Filas antes: %d | después: %d | drop: %d (%.2f%%)",
             n_rows_before, n_rows_after, n_rows_dropped, pct_rows_dropped)
    log.info("Productos antes: %d | después: %d | productos eliminados por 'sin_cluster': %d",
             n_products_before, n_products_after, n_products_removed)

    # Distribución por cluster (rápido)
    cluster_sizes = dem_ok.groupby("Cluster")["Product_ID"].nunique().to_dict()
    log.info("Productos por Cluster (nunique): %s", cluster_sizes)

    return {
        "rows_before": n_rows_before,
        "rows_after": n_rows_after,
        "rows_dropped": n_rows_dropped,
        "pct_rows_dropped": pct_rows_dropped,
        "products_before": n_products_before,
        "products_after": n_products_after,
        "products_removed": n_products_removed,
        "cluster_product_sizes": cluster_sizes,
        "out_path": str(out_path),
        "out_removed_path": str(out_removed_path),
    }


# ==== 4. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Asigna Cluster a demanda desagregada y elimina filas sin cluster."
    )
    p.add_argument("--demanda", type=str, default=str(DEF_DEMANDA),
                   help="CSV de demanda desagregada final (Fase 3).")
    p.add_argument("--clusters", type=str, default=str(DEF_CLUSTERS),
                   help="CSV lookup de clusters por Product_ID (K-Means k=4).")
    p.add_argument("--out", type=str, default=str(DEF_OUT),
                   help="CSV de salida con demanda + Cluster (filtrada).")
    p.add_argument("--out-removed", type=str, default=str(DEF_OUT_REMOVED),
                   help="CSV reporte de productos sin cluster encontrados en demanda.")
    p.add_argument("--outliers", type=str, default="",
                   help="(Opcional) CSV con outliers (DBSCAN) para marcar is_outlier=1 (requiere Product_ID).")
    # Compat. Notebook: ignorar args del kernel si procede
    import sys as _sys
    if ("ipykernel" in _sys.modules) or ("IPython" in _sys.modules):
        # no hacer nada especial, parse_known más simple
        pass
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()
    demanda_path = Path(args.demanda)
    clusters_path = Path(args.clusters)
    out_path = Path(args.out)
    out_removed_path = Path(args.out_removed)
    outliers_path = Path(args.outliers) if args.outliers else None

    try:
        info = asignar_clusters(
            demanda_path=demanda_path,
            clusters_path=clusters_path,
            out_path=out_path,
            out_removed_path=out_removed_path,
            outliers_path=outliers_path,
        )
        log.info("Proceso finalizado. Resumen: %s", info)
    except Exception as e:
        log.exception("Error asignando clusters a demanda: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

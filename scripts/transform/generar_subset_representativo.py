# =============================================================================
# Script: generar_subset_representativo.py
# Descripción:
#   Selecciona un subset representativo de productos (~30% del catálogo)
#   a partir del archivo de demanda con clusters asignados.
#
# Flujo:
#   1. Carga el dataset con demanda y clusters.
#   2. Calcula métricas por producto (demanda total y variabilidad).
#   3. Selecciona productos aplicando criterios:
#        - Mantener todos los clusters (0, 1, 2, 3).
#        - Reducir el cluster 2 (mayoritario) de forma proporcional.
#        - Mantener casi íntegros clusters minoritarios.
#        - Priorizar productos con mayor demanda o mayor variabilidad.
#        - No eliminar outliers: se conservan siempre como variable de control.
#   4. Exporta el subset en CSV y Parquet y genera reportes de validación.
#
# Input:
#   data/processed/demanda_con_clusters.csv
#
# Output:
#   data/processed/demanda_subset.csv
#   data/processed/demanda_subset.parquet
#   reports/subset_productos.csv
#   reports/subset_resumen_validacion.txt
#
# Dependencias:
#   pip install pandas numpy
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import logging

# --- Rutas base ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# --- Archivos ---
INPUT_FILE   = PROC_DIR / "demanda_con_clusters.csv"
OUT_DEMANDA  = PROC_DIR / "demanda_subset.csv"
OUT_DEMANDA_PARQUET = PROC_DIR / "demanda_subset.parquet"
OUT_PRODS    = REPORTS_DIR / "subset_productos.csv"
OUT_RESUMEN  = REPORTS_DIR / "subset_resumen_validacion.txt"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("generar_subset_representativo")


def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula demanda total y coeficiente de variación por producto.
    """
    metrics = (
        df.groupby(["Product_ID", "Cluster", "is_outlier"])
          .agg(demanda_total=("Demand_Day", "sum"),
               cv=("Demand_Day", lambda x: x.std() / x.mean() if x.mean() > 0 else 0))
          .reset_index()
    )
    return metrics


def seleccionar_productos(metrics: pd.DataFrame, frac: float = 0.3) -> pd.DataFrame:
    """
    Selecciona productos siguiendo criterios:
    - Mantener todos los clusters.
    - Reducir proporcionalmente cluster 2.
    - Priorizar productos con mayor demanda o mayor variabilidad.
    - Mantener siempre los outliers.
    """
    productos = []

    total_productos = metrics["Product_ID"].nunique()
    objetivo = int(total_productos * frac)
    logger.info(f"Productos totales: {total_productos} | Objetivo subset: {objetivo} ({int(frac*100)}%)")

    # --- Outliers: se conservan completos ---
    outliers = metrics[metrics["is_outlier"] == 1]
    productos.append(outliers)

    # --- Procesar cada cluster ---
    for c in sorted(metrics["Cluster"].unique()):
        cluster_data = metrics[(metrics["Cluster"] == c) & (metrics["is_outlier"] == 0)]
        n_cluster = cluster_data["Product_ID"].nunique()

        if c == 2:
            # Cluster mayoritario → reducción proporcional
            n_select = int(n_cluster * frac)
            seleccion = cluster_data.nlargest(n_select, ["demanda_total", "cv"])
            logger.info(f"Cluster {c}: {n_cluster} productos → seleccionados {n_select}")
        else:
            # Clusters minoritarios → mantener completos
            seleccion = cluster_data
            logger.info(f"Cluster {c}: {n_cluster} productos → mantenidos completos")

        productos.append(seleccion)

    subset = pd.concat(productos, ignore_index=True)
    return subset


def exportar_resultados(df: pd.DataFrame, demanda: pd.DataFrame):
    """
    Exporta los archivos de salida:
    - demanda_subset.csv y demanda_subset.parquet con las filas correspondientes a los productos seleccionados.
    - subset_productos.csv con la lista de productos seleccionados.
    - subset_resumen_validacion.txt con métricas de control.
    """
    productos_subset = df["Product_ID"].unique()
    demanda_subset = demanda[demanda["Product_ID"].isin(productos_subset)]

    # Guardar en CSV y Parquet
    demanda_subset.to_csv(OUT_DEMANDA, index=False)
    demanda_subset.to_parquet(OUT_DEMANDA_PARQUET, index=False)
    df.to_csv(OUT_PRODS, index=False)

    with open(OUT_RESUMEN, "w", encoding="utf-8") as f:
        f.write(f"Productos totales (base): {demanda['Product_ID'].nunique()}\n")
        f.write(f"Objetivo subset (≈30%): {len(productos_subset)}\n")
        f.write(f"Productos en subset: {len(productos_subset)}\n\n")

        resumen = df.groupby("Cluster")["Product_ID"].nunique()
        f.write("Distribución por cluster:\n")
        f.write(resumen.to_string())
        f.write("\n")

    logger.info("Subset generado correctamente.")


def main():
    # Cargar dataset
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Leyendo: {INPUT_FILE}")

    # Calcular métricas
    metrics = calcular_metricas(df)

    # Seleccionar productos
    subset = seleccionar_productos(metrics, frac=0.3)

    # Exportar resultados
    exportar_resultados(subset, df)


if __name__ == "__main__":
    main()

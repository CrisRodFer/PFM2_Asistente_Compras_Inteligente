# ============================================================================
# Script: unir_outliers_con_demanda.py
# Descripción: Une el archivo de outliers detectados (outliers_dbscan.csv)
#              con el archivo de demanda desagregada enriquecida
#              (demanda_filtrada_enriquecida_sin_nans.csv).
#
# Flujo:
#   1. Cargar dataset de demanda completa.
#   2. Cargar lista de outliers.
#   3. Unir ambos datasets (añadir columna is_outlier).
#   4. Exportar dataset combinado + reporte de resumen.
#
# Input esperado:
#   - data/processed/demanda_filtrada_enriquecida_sin_nans.csv
#   - reports/outliers_dbscan.csv
#
# Output generado:
#   - data/processed/demanda_con_outliers.csv
#   - reports/unir_outliers_con_demanda_report.txt
#
# Dependencias:
#   pip install pandas
# ============================================================================

from pathlib import Path
import pandas as pd
import logging

# --- Rutas base ---
ROOT_DIR = Path(__file__).resolve().parents[2]  # .../PFM2_Asistente_Compras_Inteligente/
DATA_DIR = ROOT_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# Entradas
DEMANDA_FILE = PROC_DIR / "demanda_filtrada_enriquecida_sin_nans.csv"
OUTLIERS_FILE = REPORTS_DIR / "outliers_dbscan.csv"

# Salidas
OUT_CSV = PROC_DIR / "demanda_con_outliers.csv"
OUT_REPORT = REPORTS_DIR / "unir_outliers_con_demanda_report.txt"

# --- Configuración logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("unir_outliers_con_demanda")


def cargar_datos(demanda_file: Path, outliers_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cargar datasets de demanda y outliers.
    """
    demanda = pd.read_csv(demanda_file)
    outliers = pd.read_csv(outliers_file)

    return demanda, outliers


def unir_datasets(demanda: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    """
    Une la demanda con los outliers, creando la columna `is_outlier`.
    """
    demanda["is_outlier"] = 0
    demanda.loc[demanda["Product_ID"].isin(outliers["Product_ID"]), "is_outlier"] = 1
    return demanda


def generar_reporte(df: pd.DataFrame, outliers: pd.DataFrame, out_path: Path):
    """
    Genera un reporte de resumen tras la unión.
    """
    total_productos = df["Product_ID"].nunique()
    total_outliers = outliers["Product_ID"].nunique()
    filas_outliers = df["is_outlier"].sum()

    resumen = [
        "=== REPORTE UNIÓN OUTLIERS + DEMANDA ===",
        f"Productos totales en demanda: {total_productos}",
        f"Productos outliers detectados: {total_outliers}",
        f"Filas marcadas como outlier en demanda: {filas_outliers}",
        "",
        f"Archivo combinado guardado en: {OUT_CSV}"
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(resumen))

    logger.info("Reporte generado: %s", out_path)


def main():
    logger.info("Cargando archivos...")
    demanda, outliers = cargar_datos(DEMANDA_FILE, OUTLIERS_FILE)

    logger.info("Unificando demanda con outliers...")
    df = unir_datasets(demanda, outliers)

    logger.info("Guardando dataset unificado en %s", OUT_CSV)
    df.to_csv(OUT_CSV, index=False)

    generar_reporte(df, outliers, OUT_REPORT)
    logger.info("Proceso finalizado correctamente.")


if __name__ == "__main__":
    main()

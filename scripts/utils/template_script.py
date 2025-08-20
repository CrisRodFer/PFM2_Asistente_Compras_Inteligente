# =============================================================================
# Script: NOMBRE_DEL_SCRIPT.py
# Descripción:
# Breve descripción de lo que hace este script. Adaptar según necesidad.
#
# Flujo del pipeline (ejemplo, modificar según corresponda):
# 1) Paso 1
# 2) Paso 2
# 3) Paso 3
#
# Input (ejemplo):
#   - data/raw/ejemplo.csv
#
# Output (ejemplo):
#   - data/clean/ejemplo.csv
#
# Dependencias:
#   - pandas
#   - numpy
#
# Instalación rápida:
#   pip install pandas numpy
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]   # ajusta a tu estructura si hace falta
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    """Crea directorios si no existen."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def ejemplo_utilidad(x: int) -> int:
    """Ejemplo de utilidad (modificar/eliminar)."""
    return x

# ==== 3. LÓGICA PRINCIPAL =====================================================
def funcion_principal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la lógica principal del script sobre el DataFrame de entrada.
    Ajustar/expandir según el caso.
    """
    # TODO: implementar la transformación real
    return df

# ==== 4. EXPORTACIÓN / I/O OPCIONAL ==========================================
def exportar_resultados(df: pd.DataFrame, out_path: Path) -> Path:
    """
    Exporta un DataFrame y devuelve la ruta escrita (útil para logs/tests).
    """
    out_path = Path(out_path)
    ensure_dirs(out_path.parent)
    df.to_csv(out_path, index=False)
    return out_path

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plantilla de script para el proyecto.")
    p.add_argument("--in", dest="inp", type=str, default=str(RAW_DIR / "ejemplo.csv"),
                   help="Ruta de entrada (CSV).")
    p.add_argument("--out", dest="out", type=str, default=str(CLEAN_DIR / "ejemplo.csv"),
                   help="Ruta de salida (CSV).")
    # añadir más flags si procede (e.g., --sep, --chunksize, --param-x, etc.)
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    inp = Path(args.inp)
    out = Path(args.out)

    try:
        log.info("Leyendo: %s", inp)
        df = pd.read_csv(inp)

        log.info("Procesando datos…")
        df_proc = funcion_principal(df)

        written = exportar_resultados(df_proc, out)
        log.info("Salida escrita en: %s", written)

    except Exception as e:
        log.exception("Fallo en la ejecución: %s", e)
        raise

if __name__ == "__main__":
    main()


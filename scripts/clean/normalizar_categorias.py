# ============================================================================
# Script: normalizar_categorias.py
# Descripción:
#   Normaliza la columna 'Categoria' en demanda_filtrada.csv para eliminar
#   problemas de codificación y asegurar consistencia en todo el pipeline.
#
#   Transformaciones aplicadas:
#     - Minúsculas
#     - Eliminación de acentos/caracteres Unicode (NFKD)
#     - Colapso de espacios múltiples en uno
#     - Eliminación de caracteres extraños
#
#   Guarda un nuevo archivo data/processed/demanda_filtrada_norm.csv
#
# Flujo:
#   1) Cargar data/processed/demanda_filtrada.csv
#   2) Normalizar columna 'Categoria'
#   3) Guardar dataset como demanda_filtrada_norm.csv
#   4) Validaciones: nº de filas, nº de productos, nº de categorías únicas
#
# Inputs:
#   - data/processed/demanda_filtrada.csv
#
# Outputs:
#   - data/processed/demanda_filtrada_norm.csv
#
# Dependencias:
#   - pandas, numpy
#
# Instalación rápida:
#   pip install pandas numpy
#
# Ejemplos de uso:
#   python scripts/clean/normalizar_categorias.py
# ============================================================================
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np
import logging
import unicodedata
import re

# ----------------------------- 0. CONFIG -----------------------------------
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    ROOT_DIR = Path().resolve()

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# -------------------------- 1. LOGGING -------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("normalizar_categorias")

# ------------------------ 2. UTILIDADES ------------------------------------
def normalize_text(x: str) -> str:
    """
    Normaliza texto para categorías:
      - minúsculas
      - elimina acentos
      - colapsa espacios
      - elimina caracteres no alfanum/espacio básicos
    """
    if pd.isna(x):
        return x
    x = str(x).strip().lower()
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8", "ignore")
    x = re.sub(r"\s+", " ", x)
    return x.strip()

# ----------------------- 3. LÓGICA PRINCIPAL -------------------------------
def normalizar_categorias(demanda_path: Path, out_path: Path):
    logger.info(f"Cargando demanda: {demanda_path}")
    df = pd.read_csv(demanda_path)

    if "Categoria" not in df.columns:
        raise KeyError("El dataset debe contener la columna 'Categoria'.")

    # Copia de seguridad
    df["Categoria_original"] = df["Categoria"]

    # Normalización
    df["Categoria"] = df["Categoria"].apply(normalize_text)

    # Guardar
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Archivo normalizado guardado en: {out_path}")

    # ---------------- VALIDACIONES ----------------
    logger.info("=== VALIDACIONES ===")
    logger.info(f"Filas: {len(df)}")
    if "Product_ID" in df.columns:
        logger.info(f"Productos únicos: {df['Product_ID'].nunique()}")
    logger.info(f"Categorías únicas antes: {df['Categoria_original'].nunique()}")
    logger.info(f"Categorías únicas después: {df['Categoria'].nunique()}")
    logger.info("Ejemplo de categorías normalizadas:")
    logger.info(df[["Categoria_original", "Categoria"]].drop_duplicates().head(10))

    return df

# ------------------------- 4. CLI / MAIN -----------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Normaliza las categorías en demanda_filtrada.csv")
    p.add_argument("--demanda", type=str, default=str(PROCESSED_DIR / "demanda_filtrada.csv"))
    p.add_argument("--out", type=str, default=str(PROCESSED_DIR / "demanda_filtrada_norm.csv"))
    return p.parse_args()

def main():
    args = parse_args()
    try:
        normalizar_categorias(
            demanda_path=Path(args.demanda),
            out_path=Path(args.out)
        )
    except Exception as e:
        logger.exception(f"Error normalizando categorías: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

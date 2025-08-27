
# ============================================================================
# Script: listar_categorias_desalineadas.py
# Descripción:
#   Diagnóstico de desalineaciones de 'Categoria' entre:
#     - data/processed/demanda_filtrada.csv
#     - data/processed/pca/categoria_scores.csv
#   Normaliza texto (acentos, mayúsculas, espacios) y lista:
#     1) Categorías de DEMANDA que no matchean con PCA (con sugerencias fuzzy)
#     2) Categorías en PCA que no aparecen en DEMANDA
#   Exporta reportes en /reports:
#     - categorias_sin_match_demanda_vs_pca.csv
#     - categorias_no_usadas_en_demanda.csv
#     - plantilla_mapping_categorias.csv  (para fijar el mapeo definitivo)
#
# Flujo:
#   1) Cargar, normalizar categorías de ambos orígenes
#   2) Calcular diferencias de conjuntos + métricas de ocurrencia
#   3) Generar sugerencias con difflib (ratio de similitud) para cada no-match
#   4) Exportar CSVs y loguear resumen
#
# Entradas (por defecto):
#   - data/processed/demanda_filtrada.csv   [cols: Product_ID, Categoria, ...]
#   - data/processed/pca/categoria_scores.csv [cols: Categoria, PC1..]
#
# Salidas:
#   - reports/categorias_sin_match_demanda_vs_pca.csv
#   - reports/categorias_no_usadas_en_demanda.csv
#   - reports/plantilla_mapping_categorias.csv
#
# Dependencias: pandas, numpy (stdlib: difflib, unicodedata, re)
# Instalación:  pip install pandas numpy
#
# Ejemplos:
#   python scripts/diagnostics/listar_categorias_desalineadas.py
#   python scripts/diagnostics/listar_categorias_desalineadas.py \
#       --demanda data/processed/demanda_filtrada.csv \
#       --pca-scores data/processed/pca/categoria_scores.csv
# ============================================================================

# ------------------------------- 0. CONFIG ----------------------------------
from pathlib import Path
import argparse
import sys

if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    ROOT_DIR = Path().resolve()

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PCA_DIR = PROCESSED_DIR / "pca"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------- 1. IMPORTS + LOGGING ----------------------------
import pandas as pd
import numpy as np
import logging
import difflib
import unicodedata
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("diag_categorias")

# ------------------------------- 2. UTILIDADES ------------------------------
def normalize_text(s: pd.Series) -> pd.Series:
    """
    Normaliza texto para matching robusto:
      - minúsculas
      - quita acentos (NFKD)
      - colapsa espacios
      - elimina caracteres no alfanum/espacio
    """
    def _clean(x):
        if pd.isna(x):
            return x
        x = str(x).strip().lower()
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8", "ignore")
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"[^a-z0-9\s\-/&]", "", x)  # permite -, /, &, dígitos
        x = x.strip()
        return x
    return s.apply(_clean)


def fuzzy_suggest(target: str, candidates: list[str], n=1):
    """
    Sugerencia fuzzy con difflib.
    Devuelve (sugerido, score) o ("", 0.0) si no hay candidato.
    """
    if not candidates:
        return "", 0.0
    # get_close_matches ya usa SequenceMatcher under the hood
    matches = difflib.get_close_matches(target, candidates, n=n, cutoff=0.0)
    if not matches:
        return "", 0.0
    best = matches[0]
    score = difflib.SequenceMatcher(None, target, best).ratio()
    return best, float(score)

# ---------------------------- 3. LÓGICA PRINCIPAL ---------------------------
def construir_reportes(demanda_path: Path, pca_scores_path: Path):
    # -- Cargar
    logger.info(f"Cargando demanda: {demanda_path}")
    dem = pd.read_csv(demanda_path) if demanda_path.suffix.lower() == ".csv" else pd.read_excel(demanda_path)
    logger.info(f"Cargando scores PCA: {pca_scores_path}")
    pcs = pd.read_csv(pca_scores_path) if pca_scores_path.suffix.lower() == ".csv" else pd.read_excel(pca_scores_path)

    if "Categoria" not in dem.columns:
        raise KeyError("demanda_filtrada.csv debe contener la columna 'Categoria'.")
    if "Categoria" not in pcs.columns:
        raise KeyError("categoria_scores.csv debe contener la columna 'Categoria'.")

    # -- Normalizar
    dem["Categoria_norm"] = normalize_text(dem["Categoria"])
    pcs["Categoria_norm"]  = normalize_text(pcs["Categoria"])

    # -- Métricas de ocurrencia en DEMANDA
    #   nº filas y nº productos distintos por categoría
    if "Product_ID" in dem.columns:
        dem_stats = (dem.groupby(["Categoria", "Categoria_norm"])
                       .agg(n_filas=("Categoria", "size"),
                            n_productos=("Product_ID", "nunique"))
                       .reset_index())
    else:
        dem_stats = (dem.groupby(["Categoria", "Categoria_norm"])
                       .agg(n_filas=("Categoria", "size"))
                       .reset_index())
        dem_stats["n_productos"] = np.nan

    # -- Conjuntos
    set_dem = set(dem_stats["Categoria_norm"].dropna().unique().tolist())
    set_pca = set(pcs["Categoria_norm"].dropna().unique().tolist())

    # 3.1 DEMANDA sin match en PCA (con sugerencias fuzzy)
    sin_match = sorted(list(set_dem - set_pca))
    pcs_list = sorted(list(set_pca))
    rows = []
    for cat_norm in sin_match:
        sug, score = fuzzy_suggest(cat_norm, pcs_list, n=1)
        fila = dem_stats.loc[dem_stats["Categoria_norm"] == cat_norm].iloc[0]
        rows.append({
            "categoria_original_demanda": fila["Categoria"],
            "categoria_norm_demanda": cat_norm,
            "n_filas_demanda": int(fila["n_filas"]),
            "n_productos_demanda": int(fila["n_productos"]) if not pd.isna(fila["n_productos"]) else "",
            "sugerencia_pca_norm": sug,
            "sugerencia_score": round(score, 4),
        })
    df_no_match = pd.DataFrame(rows).sort_values(["sugerencia_score", "categoria_norm_demanda"], ascending=[False, True])

    # 3.2 Categorías PCA no usadas en DEMANDA
    pca_no_usadas = sorted(list(set_pca - set_dem))
    df_pca_no_usadas = (pcs.loc[pcs["Categoria_norm"].isin(pca_no_usadas), ["Categoria", "Categoria_norm"]]
                          .drop_duplicates()
                          .sort_values("Categoria_norm"))

    # 3.3 Plantilla de mapping
    plantilla = df_no_match[["categoria_norm_demanda", "sugerencia_pca_norm", "sugerencia_score"]].copy()
    plantilla = plantilla.rename(columns={"sugerencia_pca_norm": "categoria_norm_pca_sugerida",
                                          "sugerencia_score": "score"})
    plantilla["usar_sugerencia"] = np.where(plantilla["score"] >= 0.75, "SI", "REVISAR")
    plantilla["comentario"] = ""

    # -- Exportar
    out1 = REPORTS_DIR / "categorias_sin_match_demanda_vs_pca.csv"
    out2 = REPORTS_DIR / "categorias_no_usadas_en_demanda.csv"
    out3 = REPORTS_DIR / "plantilla_mapping_categorias.csv"

    df_no_match.to_csv(out1, index=False)
    df_pca_no_usadas.to_csv(out2, index=False)
    plantilla.to_csv(out3, index=False)

    logger.info(f"Guardado: {out1}")
    logger.info(f"Guardado: {out2}")
    logger.info(f"Guardado: {out3}")

    # -- Resumen
    logger.info("Resumen:")
    logger.info("  - Total categorias DEMANDA: %d", len(set_dem))
    logger.info("  - Total categorias PCA:     %d", len(set_pca))
    logger.info("  - DEMANDA sin match en PCA: %d", len(sin_match))
    logger.info("  - PCA no usadas en DEMANDA: %d", len(pca_no_usadas))

    return df_no_match, df_pca_no_usadas, plantilla

# -------------------------------- 4. CLI / MAIN -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Lista categorías desalineadas entre demanda y PCA (con sugerencias).")
    p.add_argument("--demanda", type=str, default=str(PROCESSED_DIR / "demanda_filtrada.csv"),
                   help="Ruta a demanda_filtrada.csv")
    p.add_argument("--pca-scores", type=str, default=str(PCA_DIR / "categoria_scores.csv"),
                   help="Ruta a categoria_scores.csv")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        construir_reportes(
            demanda_path=Path(args.demanda),
            pca_scores_path=Path(args.pca_scores),
        )
        logging.info("Diagnóstico finalizado correctamente.")
    except Exception as e:
        logging.exception(f"Error en diagnóstico de categorías: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
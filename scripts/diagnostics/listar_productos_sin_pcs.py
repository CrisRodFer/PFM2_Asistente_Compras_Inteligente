# ============================================================================
# Script: listar_productos_sin_pcs.py
# Descripción:
#   Identifica los productos que no recibieron PC1–PC3 (por falta de match de
#   su Categoria_reducida en categoria_scores.csv). Genera:
#     - Un listado detallado de productos sin PCs.
#     - Un resumen por categoría (nº productos y demanda total).
#
# Entradas:
#   - data/processed/demanda_filtrada_enriquecida_sin_nans.csv
#   - data/processed/pca/categoria_scores.csv
#
# Salidas (en reports/):
#   - productos_sin_pcs.csv
#   - resumen_sin_pcs_por_categoria.csv
#
# Uso:
#   python scripts/diagnostics/listar_productos_sin_pcs.py
#   python scripts/diagnostics/listar_productos_sin_pcs.py \
#       --demanda data/processed/demanda_filtrada_enriquecida_sin_nans.csv \
#       --pca-scores data/processed/pca/categoria_scores.csv \
#       --outdir reports
# ============================================================================

from pathlib import Path
import sys, argparse, logging
import pandas as pd
import numpy as np
import re, unicodedata

# ---------------------------- Root (notebook-safe) --------------------------
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

DATA_DIR       = ROOT_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
PCA_DIR        = PROCESSED_DIR / "pca"
REPORTS_DIR    = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- Logging ------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("diag_sin_pcs")

# --- parche Jupyter: elimina --f=... ----------------------------------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# ----------------------------------------------------------------------------

def normalize_text(x: str) -> str:
    if pd.isna(x): return x
    x = str(x).strip().lower()
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8", "ignore")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _pick_demand_col(df: pd.DataFrame) -> str:
    for c in ["Demand_Day","demand_day","Demanda","demanda"]:
        if c in df.columns: return c
    raise KeyError("No se encontró columna de demanda (p.ej. 'Demand_Day').")

def main(demanda_path: Path, pca_scores_path: Path, outdir: Path):
    demanda_path  = Path(demanda_path)
    pca_scores_path = Path(pca_scores_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cargando demanda: {demanda_path}")
    dem = pd.read_csv(demanda_path)

    logger.info(f"Cargando scores PCA: {pca_scores_path}")
    pcs = pd.read_csv(pca_scores_path) if pca_scores_path.suffix.lower()==".csv" else pd.read_excel(pca_scores_path)

    # columnas esenciales
    if "Product_ID" not in dem.columns:
        raise KeyError("Se requiere 'Product_ID' en demanda.")
    if "Categoria_reducida" not in dem.columns and "Categoria" not in dem.columns:
        raise KeyError("Se requiere 'Categoria_reducida' (o 'Categoria').")
    dem_col = _pick_demand_col(dem)

    cat_dem_col = "Categoria_reducida" if "Categoria_reducida" in dem.columns else "Categoria"
    cat_pca_col = "Categoria" if "Categoria" in pcs.columns else next(
        (c for c in pcs.columns if c.lower() in ["categoria","categoría","category"]), None
    )
    if cat_pca_col is None:
        raise KeyError("categoria_scores debe contener columna 'Categoria'.")

    # normalizar categorías para el join
    dem["_cat_norm_join"] = dem[cat_dem_col].apply(normalize_text)
    pcs["_cat_norm_join"] = pcs[cat_pca_col].apply(normalize_text)

    # set de categorías con PCs
    set_pca = set(pcs["_cat_norm_join"].dropna().unique())

    # categoría reducida por producto (primera ocurrencia) y demanda total
    demanda_por_prod = (dem.groupby("Product_ID")[dem_col]
                          .sum()
                          .rename("d_total")
                          .reset_index())
    cat_por_prod = (dem.groupby("Product_ID")
                      .agg(
                          Categoria_original=("Categoria", "first") if "Categoria" in dem.columns else ("Product_ID","size"),
                          Categoria_reducida=("Categoria_reducida","first") if "Categoria_reducida" in dem.columns else ("Categoria","first"),
                          _cat_norm_join=("_cat_norm_join","first"),
                          precio_medio=("precio_medio","mean") if "precio_medio" in dem.columns else ("Product_ID","size"),
                      )
                      .reset_index())

    # merge demanda + categorias
    prod = demanda_por_prod.merge(cat_por_prod, on="Product_ID", how="left")

    # marcar sin PCs
    prod["sin_pcs"] = ~prod["_cat_norm_join"].isin(set_pca)

    sin_pcs = prod.loc[prod["sin_pcs"]].copy()

    n_total_prods = prod["Product_ID"].nunique()
    n_sin_pcs     = sin_pcs["Product_ID"].nunique()
    pct_sin_pcs   = n_sin_pcs / n_total_prods * 100 if n_total_prods else 0

    d_total_global = prod["d_total"].sum()
    d_total_sin    = sin_pcs["d_total"].sum()
    pct_demand     = d_total_sin / d_total_global * 100 if d_total_global else 0

    logger.info("=== RESUMEN ===")
    logger.info(f"Productos totales  : {n_total_prods}")
    logger.info(f"Sin PCs (productos): {n_sin_pcs} ({pct_sin_pcs:.2f}%)")
    logger.info(f"Demanda total (global): {d_total_global:,.0f}")
    logger.info(f"Demanda sin PCs       : {d_total_sin:,.0f} ({pct_demand:.2f}%)")

    # --- Salida 1: listado detallado
    cols_detalle = ["Product_ID","Categoria_original","Categoria_reducida","d_total","precio_medio"]
    cols_detalle = [c for c in cols_detalle if c in sin_pcs.columns]
    detalle_path = outdir / "productos_sin_pcs.csv"
    sin_pcs.sort_values("d_total", ascending=False)[cols_detalle].to_csv(detalle_path, index=False)
    logger.info(f"Guardado listado: {detalle_path}")

    # --- Salida 2: resumen por categoría
    resumen = (sin_pcs
               .groupby(["Categoria_reducida"], dropna=False)
               .agg(n_productos=("Product_ID","nunique"),
                    d_total=("d_total","sum"))
               .reset_index()
               .sort_values(["d_total","n_productos"], ascending=[False, False]))
    resumen["pct_productos"] = resumen["n_productos"] / n_total_prods * 100
    resumen["pct_demanda"]   = resumen["d_total"] / d_total_global * 100

    resumen_path = outdir / "resumen_sin_pcs_por_categoria.csv"
    resumen.to_csv(resumen_path, index=False)
    logger.info(f"Guardado resumen: {resumen_path}")

    # top categorías por impacto
    logger.info("Top categorías sin PCs por demanda:")
    logger.info("\n%s", resumen.head(10).to_string(index=False))

    logger.info("Diagnóstico finalizado correctamente.")
    return sin_pcs, resumen

# ----------------------------------- CLI ------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lista productos sin PC1–PC3 (sin match en categoria_scores).")
    p.add_argument("--demanda", type=str, default=str(PROCESSED_DIR / "demanda_filtrada_enriquecida_sin_nans.csv"))
    p.add_argument("--pca-scores", type=str, default=str(PCA_DIR / "categoria_scores.csv"))
    p.add_argument("--outdir", type=str, default=str(REPORTS_DIR))

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []

    args, _ = p.parse_known_args(argv)
    return args

if __name__ == "__main__":
    try:
        args = parse_args()
        main(Path(args.demanda), Path(args.pca_scores), Path(args.outdir))
    except Exception as e:
        logging.exception(f"Error en diagnóstico: {e}")
        sys.exit(1)

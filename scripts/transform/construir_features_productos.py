# ============================================================================
# Script: construir_features_productos.py
# Descripción:
#   Construye la matriz de características por producto para el clustering.
#   Agrega métricas de demanda por Product_ID, cruza el precio medio e inyecta
#   PC1, PC2 y PC3 de la categoría (desde categoria_scores.csv).
#
# Entradas (por defecto):
#   - data/processed/demanda_filtrada_enriquecida_sin_nans.csv
#   - data/processed/pca/categoria_scores.csv
#
# Salida:
#   - data/processed/productos_features.csv
#
# Uso:
#   python scripts/transform/construir_features_productos.py
#   python scripts/transform/construir_features_productos.py \
#       --demanda data/processed/otra.csv \
#       --pca-scores data/processed/pca/categoria_scores.csv \
#       --out data/processed/productos_features.csv
# ============================================================================

from pathlib import Path
import sys, argparse, logging
import pandas as pd
import numpy as np
import unicodedata, re

# --------------------------- Raíz del proyecto ------------------------------
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
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- Logging ------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("features_productos")

# --- Parche Jupyter: elimina el --f=... del kernel para argparse ------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# ----------------------------------------------------------------------------

# ------------------------------- Utilidades ---------------------------------
def normalize_text(x: str) -> str:
    if pd.isna(x): return x
    x = str(x).strip().lower()
    x = unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("utf-8","ignore")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _pick_demand_col(df: pd.DataFrame) -> str:
    candidates = ["Demand_Day", "demand_day", "Demanda", "demanda", "qty", "quantity"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("No se encontró columna de demanda (p.ej. 'Demand_Day').")

def _pick_price_col(df: pd.DataFrame) -> str:
    candidates = ["precio_medio", "Precio_medio", "precio", "price", "Price"]
    for c in candidates:
        if c in df.columns:
            return c
    # no es obligatorio si luego se cruza desde otra tabla; aquí lo esperamos ya presente
    raise KeyError("No se encontró columna de precio (p.ej. 'precio_medio').")

def _pick_category_reduced(df: pd.DataFrame) -> str:
    candidates = ["Categoria_reducida", "categoria_reducida", "Categoria_norm", "Categoria"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("No se encontró columna de categoría reducida / normalizada.")

# --------------------------------- Core -------------------------------------
def construir_features(demanda_path: Path, pca_scores_path: Path, out_path: Path):
    demanda_path = Path(demanda_path)
    pca_scores_path = Path(pca_scores_path)
    out_path = Path(out_path)

    # 1) Cargar demanda enriquecida y limpia
    logger.info(f"Cargando demanda: {demanda_path}")
    dem = pd.read_csv(demanda_path)

    # columnas clave
    if "Product_ID" not in dem.columns:
        raise KeyError("El dataset debe tener la columna 'Product_ID'.")
    dem_col = _pick_demand_col(dem)
    price_col = _pick_price_col(dem)
    cat_col_dem = _pick_category_reduced(dem)

    # asegurar tipos
    dem[dem_col] = _ensure_numeric(dem[dem_col])
    dem[price_col] = _ensure_numeric(dem[price_col])

    # 2) Agregaciones de demanda por producto
    logger.info("Agregando métricas de demanda por Product_ID...")
    agg = dem.groupby("Product_ID")[dem_col].agg(
        d_total = "sum",
        d_media = "mean",
        d_std   = "std",
        p95     = lambda s: np.nanpercentile(s.dropna().values, 95) if s.notna().any() else np.nan,
        mediana = "median",
        n_obs   = "count",
    ).reset_index()

    # coeficiente de variación
    agg["cv"] = agg["d_std"] / agg["d_media"]
    # ordenar columnas
    agg = agg[["Product_ID","d_total","d_media","d_std","cv","p95","mediana","n_obs"]]

    # 3) Precio medio por producto (por seguridad, re-agregamos)
    logger.info("Calculando precio medio por Product_ID...")
    precio_prod = (dem.groupby("Product_ID", as_index=False)[price_col]
                     .mean()
                     .rename(columns={price_col: "precio_medio"}))

    # 4) PCs por categoría reducida
    logger.info(f"Cargando scores PCA: {pca_scores_path}")
    pcs = pd.read_csv(pca_scores_path) if pca_scores_path.suffix.lower()==".csv" else pd.read_excel(pca_scores_path)

    # detectar columna 'Categoria' en pcs
    cat_col_pca = "Categoria" if "Categoria" in pcs.columns else next(
        (c for c in pcs.columns if c.lower() in ["categoria","categoría","category"]), None
    )
    if cat_col_pca is None:
        raise KeyError("categoria_scores.csv debe tener columna 'Categoria'.")

    # normalizar nombres de categoría en ambos lados
    dem["_cat_norm_join"] = dem[cat_col_dem].apply(normalize_text)
    pcs["_cat_norm_join"] = pcs[cat_col_pca].apply(normalize_text)

    # asegurar PCs disponibles
    pc_cols = [c for c in ["PC1","PC2","PC3"] if c in pcs.columns]
    if len(pc_cols) < 1:
        raise KeyError("No se encontraron columnas PC1/PC2/PC3 en categoria_scores.csv.")
    # mantener solo columnas necesarias para el join
    pcs_slim = pcs[["_cat_norm_join", *pc_cols]].drop_duplicates("_cat_norm_join")

    # PCs por producto: 1) primera categoría reducida observada por producto
    cat_por_prod = (dem.groupby("Product_ID")["_cat_norm_join"]
                      .first()
                      .reset_index())
    prod_pcs = cat_por_prod.merge(pcs_slim, on="_cat_norm_join", how="left").drop(columns=["_cat_norm_join"])

    # 5) Unir todo: métricas + precio + PCs
    logger.info("Combinando agregaciones, precio y PCs...")
    feats = (agg.merge(precio_prod, on="Product_ID", how="left")
                .merge(prod_pcs,   on="Product_ID", how="left"))

    # 6) Validaciones
    n_products_in  = dem["Product_ID"].nunique()
    n_products_out = feats["Product_ID"].nunique()
    logger.info("=== VALIDACIONES ===")
    logger.info(f"Productos únicos (entrada): {n_products_in}")
    logger.info(f"Productos únicos (salida) : {n_products_out}")
    # NaNs en columnas clave
    cols_check = ["d_total","d_media","d_std","cv","p95","mediana","precio_medio", *pc_cols]
    nan_report = {c: int(feats[c].isna().sum()) for c in cols_check if c in feats.columns}
    logger.info(f"NaNs por columna (claves): {nan_report}")
    # describe rápido
    logger.info("Stats d_total/d_media/precio_medio:\n%s",
                feats[["d_total","d_media","precio_medio"]].describe())

    # 7) Guardar
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ordenar columnas finales
    final_cols = ["Product_ID","d_total","d_media","d_std","cv","p95","mediana","precio_medio", *pc_cols]
    feats = feats[final_cols]
    feats.to_csv(out_path, index=False)
    logger.info(f"Guardado: {out_path} (filas={len(feats)}, cols={len(feats.columns)})")

    return feats

# --------------------------------- CLI --------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Construye productos_features.csv para clustering.")
    p.add_argument("--demanda",    type=str, default=str(PROCESSED_DIR / "demanda_filtrada_enriquecida_sin_nans.csv"))
    p.add_argument("--pca-scores", type=str, default=str(PCA_DIR / "categoria_scores.csv"))
    p.add_argument("--out",        type=str, default=str(PROCESSED_DIR / "productos_features.csv"))

    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []
    args, _ = p.parse_known_args(argv)

    logger.info("ARGS -> demanda=%s | pca=%s | out=%s", args.demanda, args.pca_scores, args.out)
    return args

def main():
    args = parse_args()
    try:
        construir_features(Path(args.demanda), Path(args.pca_scores), Path(args.out))
        logger.info("Construcción de features finalizada correctamente.")
    except Exception as e:
        logger.exception(f"Error construyendo features: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

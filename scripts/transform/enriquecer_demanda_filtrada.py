# ============================================================================
# Script: enriquecer_demanda_filtrada.py
# Descripción:
#   Enriquecimiento de la demanda normalizada con:
#     (1) precio medio por Product_ID
#     (2) categoria_reducida según PCA (match directo o mapping manual)
#
# NOTA:
#   Esta versión NO fuerza que categoria_reducida pertenezca al set oficial del PCA.
#   (puede dar 23 categorías si el mapping incluye una etiqueta no-PCA)
#
# Entradas por defecto:
#   - data/processed/demanda_filtrada_norm.csv
#   - data/raw/Historico_Ventas_2023_Corregido.xlsx
#   - data/processed/pca/categoria_scores.csv
#   - reports/plantilla_mapping_categorias.csv
#
# Salida:
#   - data/processed/demanda_filtrada_enriquecida.csv
# ============================================================================

from pathlib import Path
import sys, argparse, logging
import pandas as pd
import numpy as np
import unicodedata, re

# -------------------------- raíz del proyecto -------------------------------
def pick_root_with_file(script_dir: Path, rel_file: str) -> Path:
    """Elige la primera raíz que tenga 'data/' y el fichero relativo `rel_file`."""
    candidates = []
    if len(script_dir.parents) >= 2:
        candidates.append(script_dir.parents[2])      # <root> si estamos en scripts/transform
    candidates += [p for p in script_dir.parents]
    candidates.append(Path.cwd().resolve())
    for base in candidates:
        if (base / "data").is_dir() and (base / rel_file).exists():
            return base
    for base in candidates:
        if (base / "data").is_dir():
            return base
    return script_dir

_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
_REL_DEFAULT_IN = Path("data/processed/demanda_filtrada_norm.csv")

ROOT_DIR = pick_root_with_file(_SCRIPT_DIR, str(_REL_DEFAULT_IN))
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PCA_DIR = PROCESSED_DIR / "pca"
REPORTS_DIR = ROOT_DIR / "reports"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("enriquecer_demanda")
logger.info(f"ROOT_DIR detectado: {ROOT_DIR}")
logger.info(f"Demanda por defecto: {PROCESSED_DIR / 'demanda_filtrada_norm.csv'}")

# ------------------------------ utilidades ----------------------------------
def normalize_text(x: str) -> str:
    if pd.isna(x): return x
    x = str(x).strip().lower()
    x = unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("utf-8","ignore")
    x = re.sub(r"\s+"," ", x)
    return x.strip()

def to_float_safe(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"[.\s]", "", regex=True)  # quita puntos de miles / espacios
              .str.replace(",", ".", regex=False)     # coma -> punto
              .replace({"": np.nan}).astype(float))

def read_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

# ----------------------------- core -----------------------------------------
def enriquecer_demanda(demanda_path: Path, precios_path: Path, pca_scores_path: Path,
                       mapping_path: Path, out_path: Path):

    # 1) Demanda normalizada
    logger.info(f"Cargando demanda: {demanda_path}")
    df = read_any(demanda_path)
    if "Product_ID" not in df.columns or "Categoria" not in df.columns:
        raise KeyError("El dataset debe contener 'Product_ID' y 'Categoria'.")
    df["Categoria"] = df["Categoria"].apply(normalize_text)

    # 2) Precio medio por producto
    logger.info(f"Cargando precios: {precios_path}")
    dfp = read_any(precios_path)
    if "Product_ID" not in dfp.columns:
        pid = next((c for c in dfp.columns if c.lower() in ["product_id","producto_id","id_producto"]), None)
        if pid: dfp = dfp.rename(columns={pid:"Product_ID"})
        else: raise KeyError("El archivo de precios debe traer 'Product_ID'.")
    price_col = next((c for c in ["Price","price","Precio","precio","precio_medio"] if c in dfp.columns), None)
    if price_col is None:
        raise KeyError("No se encontró columna Price/Precio en el archivo de precios.")
    if not pd.api.types.is_numeric_dtype(dfp[price_col]):
        dfp[price_col] = to_float_safe(dfp[price_col])

    precio_med = (dfp.groupby("Product_ID", as_index=False)[price_col]
                    .mean().rename(columns={price_col:"precio_medio"}))
    df = df.merge(precio_med, on="Product_ID", how="left")

    # 3) Categorías reducidas del PCA (solo para conocer el set; NO se fuerza)
    logger.info(f"Cargando categorías reducidas (PCA): {pca_scores_path}")
    pcs = read_any(pca_scores_path)
    cat_col = "Categoria" if "Categoria" in pcs.columns else next(
        (c for c in pcs.columns if c.lower() in ["categoria","categoría","category"]), None
    )
    if cat_col is None:
        raise KeyError("categoria_scores.csv debe tener columna 'Categoria'.")
    pcs["Categoria_norm_pca"] = pcs[cat_col].apply(normalize_text)
    valid_cats = set(pcs["Categoria_norm_pca"].dropna().unique())

    # 4) Mapping manual (NO se valida contra valid_cats aquí)
    logger.info(f"Cargando mapping: {mapping_path}")
    map_df = read_any(mapping_path)
    req = {"categoria_norm_demanda","categoria_norm_pca_sugerida"}
    if not req.issubset(map_df.columns):
        raise KeyError("El mapping debe tener columnas 'categoria_norm_demanda' y 'categoria_norm_pca_sugerida'.")
    map_df["categoria_norm_demanda"] = map_df["categoria_norm_demanda"].apply(normalize_text)
    map_df["categoria_norm_pca_sugerida"] = map_df["categoria_norm_pca_sugerida"].apply(normalize_text)
    mapping_dict = dict(zip(map_df["categoria_norm_demanda"], map_df["categoria_norm_pca_sugerida"]))

    # 5) Asignación de Categoria_reducida (match directo -> mapping). SIN filtro posterior.
    def asignar(cat: str):
        if pd.isna(cat): return np.nan
        if cat in valid_cats:        # ya coincide con alguna reducida del PCA
            return cat
        return mapping_dict.get(cat, np.nan)

    df["Categoria_reducida"] = df["Categoria"].apply(asignar)

    # 6) Export
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Guardado: {out_path} ({len(df)} filas, {df['Product_ID'].nunique()} productos)")
    return df, valid_cats

# ------------------------------- CLI ----------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Enriquece demanda (precio + categoria reducida PCA).")
    p.add_argument("--demanda",    type=str, default=str(PROCESSED_DIR / "demanda_filtrada_norm.csv"))
    p.add_argument("--precios",    type=str, default=str(RAW_DIR / "Historico_Ventas_2023_Corregido.xlsx"))
    p.add_argument("--pca-scores", type=str, default=str(PCA_DIR / "categoria_scores.csv"))
    p.add_argument("--mapping",    type=str, default=str(REPORTS_DIR / "plantilla_mapping_categorias.csv"))
    p.add_argument("--out",        type=str, default=str(PROCESSED_DIR / "demanda_filtrada_enriquecida.csv"))

    # Compatibilidad Notebook: ignorar '--f=...' de Jupyter
    if argv is None:
        in_nb = ("ipykernel" in sys.modules) or ("IPython" in sys.modules)
        argv = [] if in_nb else None
    args, _ = p.parse_known_args(argv)

    logger.info(
        "ARGS RESUELTOS -> demanda=%s | precios=%s | pca=%s | mapping=%s | out=%s",
        args.demanda, args.precios, args.pca_scores, args.mapping, args.out
    )
    return args

def main():
    args = parse_args()
    try:
        df_out, valid_cats = enriquecer_demanda(
            demanda_path=Path(args.demanda),
            precios_path=Path(args.precios),
            pca_scores_path=Path(args.pca_scores),
            mapping_path=Path(args.mapping),
            out_path=Path(args.out),
        )
        # -------- VALIDACIONES --------
        logger.info("=== VALIDACIÓN ===")
        df_in = read_any(args.demanda)
        logger.info(f"Filas in/out: {len(df_in)} -> {len(df_out)}")
        logger.info(f"Productos únicos in/out: {df_in['Product_ID'].nunique()} -> {df_out['Product_ID'].nunique()}")
        pct_precio = df_out.groupby("Product_ID")["precio_medio"].first().notna().mean()*100
        logger.info(f"Productos con precio: {pct_precio:.2f}%")
        logger.info(f"Stats precio_medio:\n{df_out['precio_medio'].describe()}")

        usadas = df_out["Categoria_reducida"].dropna().unique()
        logger.info(f"Categorías reducidas únicas (usadas): {len(usadas)} (esperado ≈ {len(valid_cats)})")
        n_nan = int(df_out["Categoria_reducida"].isna().sum())
        logger.info(f"NaNs en Categoria_reducida: {n_nan}")

        logger.info(df_out[["Product_ID","Categoria","Categoria_reducida","precio_medio"]].sample(5, random_state=42))
        logger.info("=== VALIDACIÓN FINALIZADA ===")
    except Exception as e:
        logger.exception(f"Error enriqueciendo demanda: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


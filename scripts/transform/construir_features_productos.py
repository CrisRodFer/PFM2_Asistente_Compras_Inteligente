# ============================================================================
# Script: construir_features_productos.py
# Descripción:
#   Construye el dataset de features a nivel producto para el clustering (3.5).
#   Usa demanda diaria filtrada (Product_ID, Date, Demand_Day, Categoria,
#   Estado_Producto), agrega métricas por producto, inyecta PCs por categoría
#   y cruza precio medio desde un histórico/tabla de precios.
#
# Inputs:
#   - data/processed/demanda_filtrada.csv
#   - data/processed/pca/categoria_scores.csv  (Categoria, PC1..PCk)
#   - data/raw/Historico_Ventas_2023_Corregido.xlsx  (Product_ID|product_id, Price)
#
# Outputs:
#   - data/processed/productos_features.csv
#   - (opcional) data/processed/productos_features_scaled.csv  [--scale]
#
# Dependencias: pandas, numpy, scikit-learn, openpyxl
# Instalación:  pip install pandas numpy scikit-learn openpyxl
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
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"
PCA_DIR = PROCESSED_DIR / "pca"   # <- ruta correcta para categoria_scores.csv

# -------------------------- 1. IMPORTS + LOGGING ----------------------------
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import unicodedata
import re


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("features_productos")

# ------------------------------- 2. UTILIDADES ------------------------------
def infer_quantity_col(df: pd.DataFrame) -> str:
    """Localiza la columna de cantidad de demanda (incluye 'Demand_Day')."""
    candidates = [
        "Demand_Day", "demand_day",
        "sales_quantity", "Sales Quantity",
        "demand", "Demand",
        "quantity"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "No se encontró columna de cantidad. Esperadas: "
        "Demand_Day / Sales Quantity / Demand / quantity"
    )

def read_table_any(path: Path) -> pd.DataFrame:
    """Lee CSV o Excel según extensión; error claro si no existe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def to_float_safe(s: pd.Series) -> pd.Series:
    """Convierte a float tolerando separadores de miles y coma decimal."""
    return (
        s.astype(str)
         .str.replace(r"[.\s]", "", regex=True)  # quita puntos/miles
         .str.replace(",", ".", regex=False)      # coma -> punto
         .replace({"": np.nan})
         .astype(float)
    )

def standardize_df(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """Estandariza numéricas excepto exclude_cols."""
    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        logger.warning("No hay columnas numéricas para estandarizar.")
        return df.copy()
    scaler = StandardScaler()
    out = df.copy()
    out[num_cols] = scaler.fit_transform(out[num_cols])
    return out


def normalize_categoria(s: pd.Series) -> pd.Series:
    """
    Normaliza texto de categorías:
      - Pasa a minúsculas
      - Quita espacios extra
      - Quita acentos/caracteres Unicode especiales
    """
    def _clean(txt):
        if pd.isna(txt):
            return txt
        # minusculas
        txt = str(txt).strip().lower()
        # normalización unicode (quita acentos)
        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("utf-8", "ignore")
        # espacios múltiples -> uno
        txt = re.sub(r"\s+", " ", txt)
        return txt
    
    return s.apply(_clean)

# ---------------------------- 3. LÓGICA PRINCIPAL ---------------------------
def construir_features(demanda_path: Path, precios_path: Path, pca_scores_path: Path) -> pd.DataFrame:
    """Construye el dataframe de features a nivel producto con tus encabezados."""
    # -- 3.1 Demanda
    logger.info(f"Cargando demanda filtrada: {demanda_path}")
    df = read_table_any(demanda_path)

    if "Product_ID" not in df.columns:
        pid_guess = next((c for c in df.columns if c.lower() in ["product_id", "id_producto", "producto_id"]), None)
        if pid_guess:
            df = df.rename(columns={pid_guess: "Product_ID"})
        else:
            raise KeyError("No se encontró 'Product_ID' en demanda_filtrada.csv")

    if "Categoria" not in df.columns:
        raise KeyError("Se esperaba columna 'Categoria' en demanda_filtrada.csv")

    if "Estado_Producto" in df.columns:
        before = len(df)
        df = df[df["Estado_Producto"].astype(str).str.lower().eq("activo")].copy()
        logger.info(f"Demanda: filtrado activos {before} -> {len(df)}")

    qty_col = infer_quantity_col(df)
    logger.info(f"Columna de demanda detectada: {qty_col}")

    df["Product_ID"] = pd.to_numeric(df["Product_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Product_ID"]).copy()
    df["Product_ID"] = df["Product_ID"].astype(int)
    df["Categoria"] = normalize_categoria(df["Categoria"])

    grp = df.groupby("Product_ID")[qty_col]
    agg = pd.DataFrame({
        "d_total": grp.sum(),
        "d_media": grp.mean(),
        "d_std": grp.std(ddof=0),
        "p95": grp.quantile(0.95),
        "mediana": grp.median(),
        "n_obs": grp.count(),
    }).reset_index()
    agg["cv"] = np.where(agg["d_media"] > 0, agg["d_std"] / agg["d_media"], 0.0)

    cat_from_demand = (
        df.groupby("Product_ID")["Categoria"]
          .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
          .reset_index()
    )
    features = agg.merge(cat_from_demand, on="Product_ID", how="left")

    # -- 3.2 Precios
    logger.info(f"Cargando archivo de precios: {precios_path}")
    dfp = read_table_any(precios_path)

    if "Product_ID" not in dfp.columns:
        pid_guess = next((c for c in dfp.columns if c.lower() in ["product_id", "id_producto", "producto_id"]), None)
        if pid_guess:
            dfp = dfp.rename(columns={pid_guess: "Product_ID"})
        else:
            raise KeyError("No se encontró 'Product_ID' en archivo de precios")

    price_col = next((c for c in ["Price", "price", "Precio", "precio", "precio_medio"] if c in dfp.columns), None)
    if price_col:
        if not pd.api.types.is_numeric_dtype(dfp[price_col]):
            dfp[price_col] = to_float_safe(dfp[price_col])
        precio_med = (
            dfp.groupby("Product_ID", as_index=False)[price_col]
               .mean()
               .rename(columns={price_col: "precio_medio"})
        )
        features = features.merge(precio_med, on="Product_ID", how="left")
    else:
        logger.warning("No se encontró columna de precio. Se continuará sin precio.")

    # -- 3.3 PCA por categoría
    logger.info(f"Cargando scores PCA: {pca_scores_path}")
    pcs = read_table_any(pca_scores_path)

    if "Categoria" not in pcs.columns:
        cat_guess = next((c for c in pcs.columns if c.lower() in ["categoria", "category", "categoría"]), None)
        if cat_guess:
            pcs = pcs.rename(columns={cat_guess: "Categoria"})
        else:
            raise KeyError("No se encontró 'Categoria' en categoria_scores")

    pcs["Categoria"] = normalize_categoria(pcs["Categoria"])
    pc_cols = [c for c in pcs.columns if c.upper().startswith("PC")]
    if not pc_cols:
        raise KeyError("No se detectaron columnas PC* en categoria_scores")

    features = features.merge(pcs[["Categoria"] + pc_cols], on="Categoria", how="left")

    ordered = ["Product_ID", "Categoria", "d_total", "d_media", "d_std", "cv", "p95", "mediana", "n_obs"]
    if "precio_medio" in features.columns:
        ordered.append("precio_medio")
    ordered += pc_cols
    ordered = [c for c in ordered if c in features.columns]
    features = features[ordered].copy()

    for c in features.columns:
        if c not in ["Product_ID", "Categoria"]:
            features[c] = pd.to_numeric(features[c], errors="coerce")

    if features["Categoria"].isna().any():
        logger.warning("Existen productos sin Categoria tras el ensamblado.")
    for c in pc_cols:
        if features[c].isna().any():
            logger.warning(f"NaNs en {c}: posibles nombres de categoría no coinciden.")

    logger.info(f"Features construidas: {features.shape[0]} filas, {features.shape[1]} columnas.")
    return features

# ----------------------------- 4. EXPORTACIÓN -------------------------------
def exportar_features(df: pd.DataFrame, out_path: Path, scale: bool = False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Guardado: {out_path}")

    if scale:
        exclude = ["Product_ID", "Categoria"]
        df_scaled = standardize_df(df, exclude_cols=exclude)
        scaled_path = out_path.with_name(out_path.stem + "_scaled.csv")
        df_scaled.to_csv(scaled_path, index=False)
        logger.info(f"Guardado (escalado): {scaled_path}")

# -------------------------------- 5. CLI / MAIN -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Construye productos_features.csv para clustering (3.5).")
    p.add_argument("--demanda", type=str, default=str(PROCESSED_DIR / "demanda_filtrada.csv"),
                   help="Ruta a demanda filtrada (CSV/XLSX).")
    p.add_argument("--precios", type=str, default=str(RAW_DIR / "Historico_Ventas_2023_Corregido.xlsx"),
                   help="Ruta a archivo de precios (CSV/XLSX).")
    p.add_argument("--pca-scores", type=str, default=str(PCA_DIR / "categoria_scores.csv"),
                   help="Ruta a scores de PCA por categoría (CSV/XLSX).")
    p.add_argument("--out", type=str, default=str(PROCESSED_DIR / "productos_features.csv"),
                   help="Ruta de salida.")
    p.add_argument("--scale", action="store_true", help="Exportar también versión estandarizada (_scaled.csv).")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        feats = construir_features(
            demanda_path=Path(args.demanda),
            precios_path=Path(args.precios),
            pca_scores_path=Path(args.pca_scores),
        )
        exportar_features(feats, out_path=Path(args.out), scale=args.scale)

        pc_cols = [c for c in feats.columns if c.upper().startswith("PC")]
        logger.info("RESUMEN | cols: %s", ", ".join(feats.columns.astype(str).tolist()))
        logger.info(
            "d_total percentiles: p5=%.2f | p50=%.2f | p95=%.2f | max=%.2f",
            np.nanpercentile(feats["d_total"], 5),
            np.nanpercentile(feats["d_total"], 50),
            np.nanpercentile(feats["d_total"], 95),
            np.nanmax(feats["d_total"]),
        )
        if "precio_medio" in feats.columns:
            logger.info("precio_medio: mean=%.2f | std=%.2f",
                        np.nanmean(feats["precio_medio"]),
                        np.nanstd(feats["precio_medio"]))
        if pc_cols:
            logger.info(f"PCs detectadas: {', '.join(pc_cols)}")

        logger.info("¡productos_features.csv generado con éxito!")
    except Exception as e:
        logger.exception(f"Error generando features de productos: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


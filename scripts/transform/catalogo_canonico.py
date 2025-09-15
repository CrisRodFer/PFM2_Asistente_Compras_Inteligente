# =============================================================================
# Script: catalogo_canonico.py
# Descripción:
#   FASE 9.1 · Construcción del catálogo canónico de productos.
#
#   Reglas:
#     1) Mantener SOLO productos con Product_ID en catálogo Y presentes en demanda.
#     2) Descartar productos con Product_ID en catálogo pero SIN demanda.
#     3) Catalogar como "novedades" los productos sin Product_ID en catálogo.
#
# Flujo del pipeline:
#   1) Cargar catálogo y demanda
#   2) Normalizar Product_ID (soluciona 1000.0 vs 1000)
#   3) Intersección, descartes y novedades
#   4) (Opcional) Enriquecer con metadatos de demanda (cluster/flags/precio)
#   5) Exportar parquet + csv + reportes
#
# Input:
#   - data/clean/Catalogo_Productos_Limpio.xlsx
#   - data/processed/demanda_subset.csv
#
# Output:
#   - data/processed/catalog_items.parquet / .csv
#   - data/processed/novedades.parquet / .csv
#   - reports/fase9_1_descartados.csv
#   - reports/fase9_1_resumen.txt
#
# Dependencias:
#   - pandas, numpy, openpyxl, pyarrow
#   pip install pandas numpy openpyxl pyarrow
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports" if (DATA_DIR / "reports").exists() else ROOT_DIR / "reports"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import pandas as pd
import numpy as np
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def _normalize_product_id_series(s: pd.Series) -> pd.Series:
    """
    Normaliza Product_ID para que 1000.0, '1000 ', '001000' -> '1000'.
    - Intenta convertir a numérico; si procede, castea a Int64 y luego a string.
    - Limpia espacios y valores vacíos.
    """
    s_num = pd.to_numeric(s, errors="coerce")  # números válidos -> float
    out = s.astype("string")                   # baseline
    mask_num = s_num.notna()
    out.loc[mask_num] = s_num.loc[mask_num].astype("Int64").astype(str)  # sin .0
    out = out.str.strip()
    out = out.where(~out.isin(["", "nan", "None"]), pd.NA)
    return out

def _safe_mode(x):
    cnt = Counter([v for v in x if pd.notna(v)])
    return cnt.most_common(1)[0][0] if cnt else np.nan

def read_catalog(path_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(path_xlsx)
    if "Product_ID" not in df.columns:
        raise ValueError("El catálogo debe contener la columna 'Product_ID'.")
    df["Product_ID"] = _normalize_product_id_series(df["Product_ID"])
    return df

def read_demanda_ids(path_csv: Path) -> pd.Series:
    df_ids = pd.read_csv(path_csv, usecols=["Product_ID"])
    df_ids["Product_ID"] = _normalize_product_id_series(df_ids["Product_ID"])
    return df_ids["Product_ID"].dropna()

def build_meta_from_demanda(path_csv: Path, ids_keep: set[str]) -> pd.DataFrame:
    """
    Construye metadatos por Product_ID si existen columnas útiles.
    Agrega por:
      - cluster/Cluster/cluster_id -> modo
      - is_outlier/is_top/top_ventas/flag_outlier -> max (OR)
      - precio_medio/price_mean -> mean
    """
    df = pd.read_csv(path_csv)
    if "Product_ID" not in df.columns:
        return pd.DataFrame(columns=["Product_ID"])
    df["Product_ID"] = _normalize_product_id_series(df["Product_ID"])
    df = df[df["Product_ID"].isin(ids_keep)]

    agg = {}
    for c in ["cluster", "Cluster", "cluster_id"]:
        if c in df.columns:
            agg[c] = _safe_mode
            break
    for c in ["is_outlier", "is_top", "top_ventas", "flag_outlier"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            agg[c] = "max"
    for c in ["precio_medio", "price_mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            agg[c] = "mean"

    if not agg:
        return pd.DataFrame(columns=["Product_ID"])
    return df.groupby("Product_ID", as_index=False).agg(agg)

# ==== 3. LÓGICA PRINCIPAL =====================================================
def construir_catalogo(catalog_xlsx: Path, demanda_csv: Path):
    log.info("Construyendo catálogo canónico…")
    cat = read_catalog(catalog_xlsx)
    n_total = len(cat)

    cat_with_id = cat[cat["Product_ID"].notna()].copy()
    cat_no_id = cat[cat["Product_ID"].isna()].copy()

    ids_demanda = set(read_demanda_ids(demanda_csv).unique())
    n_ids_dem = len(ids_demanda)

    mask_keep = cat_with_id["Product_ID"].isin(ids_demanda)
    catalog_final = cat_with_id[mask_keep].copy()
    descartados = cat_with_id[~mask_keep].copy()
    descartados.insert(0, "__motivo__", "En catálogo con Product_ID pero sin demanda")
    novedades = cat_no_id.copy()

    # Enriquecimiento (opcional)
    try:
        meta = build_meta_from_demanda(demanda_csv, set(catalog_final["Product_ID"].unique()))
        if not meta.empty:
            catalog_final = catalog_final.merge(meta, on="Product_ID", how="left")
            log.info("Catálogo enriquecido con metadatos de demanda.")
    except Exception as e:
        log.warning("No se pudieron añadir metadatos de demanda: %s", e)

    resumen = (
        "FASE 9.1 · Resumen consolidación catálogo\n"
        f"- Total catálogo original: {n_total}\n"
        f"- Con Product_ID: {len(cat_with_id)}\n"
        f"- Sin Product_ID (novedades): {len(novedades)}\n"
        f"- Product_ID distintos en demanda: {n_ids_dem}\n"
        f"- Catálogo canónico final: {len(catalog_final)}\n"
        f"- Descartados: {len(descartados)}\n"
    )

    # Mostrar tanto en logs como en salida directa de terminal
    
    print("\n" + resumen)
    log.info("Resumen: final=%d, novedades=%d, descartados=%d",
         len(catalog_final), len(novedades), len(descartados))
    
    return catalog_final, novedades, descartados, resumen

# ==== 4. EXPORTACIÓN / I/O OPCIONAL ==========================================
def exportar(catalog_final: pd.DataFrame, novedades: pd.DataFrame, descartados: pd.DataFrame, resumen: str) -> None:
    ensure_dirs(PROCESSED_DIR, REPORTS_DIR)
    catalog_final.to_parquet(PROCESSED_DIR / "catalog_items.parquet", index=False)
    catalog_final.to_csv(PROCESSED_DIR / "catalog_items.csv", index=False, encoding="utf-8-sig")
    novedades.to_parquet(PROCESSED_DIR / "novedades.parquet", index=False)
    novedades.to_csv(PROCESSED_DIR / "novedades.csv", index=False, encoding="utf-8-sig")
    descartados.to_csv(REPORTS_DIR / "fase9_1_descartados.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "fase9_1_resumen.txt").write_text(resumen, encoding="utf-8")
    log.info("Exportado catalog_items / novedades (parquet+csv) y reportes.")

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE 9.1 · Construcción catálogo canónico (normaliza Product_ID).")
    p.add_argument("--catalog_xlsx", type=str,
                   default=r"C:\Users\crisr\Desktop\Máster Data Science & IA\PROYECTO\PFM2_Asistente_Compras_Inteligente\data\clean\Catalogo_Productos_Limpio.xlsx")
    p.add_argument("--demanda_csv", type=str,
                   default=r"C:\Users\crisr\Desktop\Máster Data Science & IA\PROYECTO\PFM2_Asistente_Compras_Inteligente\data\processed\demanda_subset.csv")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    try:
        catalog_final, novedades, descartados, resumen = construir_catalogo(
            Path(args.catalog_xlsx), Path(args.demanda_csv)
        )
        exportar(catalog_final, novedades, descartados, resumen)
        log.info("Proceso completado.")
    except Exception as e:
        log.exception("Error en catalogo_canonico: %s", e)
        raise

if __name__ == "__main__":
    main()

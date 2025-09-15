# =============================================================================
# Script: construir_catalogo_proveedores.py
# Descripción:
#   Genera data/clean/supplier_catalog.csv a partir de data/processed/catalog_items.csv
#   - supplier_id = columna 'Proveedor'
#   - disponibilidad = suma de 'Stock Real' por (Product_ID, Proveedor)
#   - precio = media de precio si existe (si no, NaN)
#   - lead_time por proveedor (rango aleatorio reproducible: 2-4, 5-7, 10-15 días)
#   - prioridad = 1 (placeholder)
#
# Entradas (por defecto):
#   - data/processed/catalog_items.csv
#
# Salidas: 
#   - data/clean/supplier_catalog.csv
#
# Dependencias: pandas, numpy
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def normalize_product_id(s: pd.Series) -> pd.Series:
    """Normaliza Product_ID: 1000.0 / '1000 ' / '001000' -> '1000' (string o NaN)."""
    s_num = pd.to_numeric(s, errors="coerce")
    out = s.astype("string")
    m = s_num.notna()
    out.loc[m] = s_num.loc[m].astype("Int64").astype(str)
    out = out.str.strip()
    out = out.where(~out.isin(["", "nan", "None"]), pd.NA)
    return out

def detect_first(colnames: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in colnames:
            return c
    return None

def assign_lead_time_per_supplier(supplier_ids: pd.Series,
                                  rng: np.random.Generator,
                                  probs=(0.50, 0.35, 0.15)) -> pd.DataFrame:
    """
    Asigna a CADA proveedor un bucket y un lead_time entero dentro del rango.
    probs: (prob_2_4, prob_5_7, prob_10_15)
    """
    buckets = {
        "2-4":  (2, 4),
        "5-7":  (5, 7),
        "10-15": (10, 15),
    }
    names = list(buckets.keys())
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()  # normaliza por si acaso

    choices = rng.choice(names, size=len(supplier_ids), p=probs)
    lt_vals = []
    for ch in choices:
        lo, hi = buckets[ch]
        lt_vals.append(int(rng.integers(lo, hi + 1)))

    return pd.DataFrame({
        "supplier_id": supplier_ids.values.astype("string"),
        "lead_time_bucket": choices,
        "lead_time": lt_vals,
    })

# ==== 3. LÓGICA PRINCIPAL =====================================================
def construir_supplier_catalog(
    catalog_items_csv: Path,
    out_csv: Path,
    seed: int = 42,
    probs=(0.50, 0.35, 0.15),
) -> pd.DataFrame:
    log.info("Cargando catalog_items: %s", catalog_items_csv)
    if not catalog_items_csv.exists():
        raise FileNotFoundError(f"No existe catalog_items.csv: {catalog_items_csv}")

    df = pd.read_csv(catalog_items_csv, dtype=str, low_memory=False)

    # columnas necesarias
    if "Product_ID" not in df.columns or "Proveedor" not in df.columns:
        raise ValueError("Se requieren columnas 'Product_ID' y 'Proveedor' en catalog_items.csv")

    # normalizaciones básicas
    df["Product_ID"] = normalize_product_id(df["Product_ID"])
    df["Proveedor"] = df["Proveedor"].astype("string").str.strip()

    # detectar columnas de stock y precio (opcionales)
    col_stock = detect_first(
        df.columns.tolist(),
        ["Stock Real", "Stock_Real", "stock_real", "Stock"]
    )
    col_precio = detect_first(
        df.columns.tolist(),
        ["precio_medio", "Precio Medio", "precio", "Precio"]
    )

    if col_stock is None:
        log.warning("No se encontró columna de stock; se usará 0 como disponibilidad.")
        df["__stock_tmp__"] = 0.0
        col_stock = "__stock_tmp__"
    else:
        df[col_stock] = pd.to_numeric(df[col_stock], errors="coerce").fillna(0)

    if col_precio is not None:
        df[col_precio] = pd.to_numeric(df[col_precio], errors="coerce")

    # agregación por (Product_ID, Proveedor)
    agg = {col_stock: "sum"}
    if col_precio is not None:
        agg[col_precio] = "mean"

    sup = (
        df[["Product_ID", "Proveedor", col_stock] + ([col_precio] if col_precio else [])]
          .dropna(subset=["Product_ID", "Proveedor"])
          .groupby(["Product_ID", "Proveedor"], as_index=False)
          .agg(agg)
          .rename(columns={
              "Proveedor": "supplier_id",
              col_stock: "disponibilidad",
              **({col_precio: "precio"} if col_precio else {})
          })
    )

    # lead_time por proveedor (reproducible)
    rng = np.random.default_rng(seed)
    suppliers = pd.DataFrame({"supplier_id": sup["supplier_id"].drop_duplicates()})
    lead_map = assign_lead_time_per_supplier(suppliers["supplier_id"], rng, probs=probs)
    sup = sup.merge(lead_map, on="supplier_id", how="left")

    # completar columnas y tipos
    if "precio" not in sup.columns:
        sup["precio"] = np.nan
    sup["prioridad"] = 1

    # ordenar columnas
    sup = sup[
        ["Product_ID", "supplier_id", "precio", "disponibilidad", "lead_time", "prioridad", "lead_time_bucket"]
    ]

    # tipos numéricos
    for c in ["precio", "disponibilidad", "lead_time", "prioridad"]:
        sup[c] = pd.to_numeric(sup[c], errors="coerce")

    # exportar
    ensure_dirs(CLEAN_DIR)
    sup.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # resumen en consola
    resumen = (
        "Supplier catalog generado\n"
        f"- Origen: {catalog_items_csv}\n"
        f"- Destino: {out_csv}\n"
        f"- Filas (Product_ID x proveedor): {len(sup):,}\n"
        f"- Productos únicos: {sup['Product_ID'].nunique():,}\n"
        f"- Proveedores únicos: {sup['supplier_id'].nunique():,}\n"
        f"- % filas con precio: {sup['precio'].notna().mean()*100:.1f}%\n"
    )
    print("\n" + resumen)

    # distribución de buckets (ayuda diagnóstico)
    dist = sup["lead_time_bucket"].value_counts(normalize=True).mul(100).round(1)
    print("Distribución lead_time_bucket (%):")
    print(dist.to_string())

    return sup

# ==== 4. EXPORTACIÓN / I/O OPCIONAL ==========================================
# (No hay export extra; ya se escribe supplier_catalog.csv)

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construye data/clean/supplier_catalog.csv desde catalog_items.csv")
    p.add_argument("--catalog_items_csv", type=str, default=str(PROCESSED_DIR / "catalog_items.csv"),
                   help="Ruta a data/processed/catalog_items.csv")
    p.add_argument("--out_csv", type=str, default=str(CLEAN_DIR / "supplier_catalog.csv"),
                   help="Ruta de salida data/clean/supplier_catalog.csv")
    p.add_argument("--seed", type=int, default=42, help="Semilla para lead time reproducible")
    p.add_argument("--probs", type=float, nargs=3, default=(0.50, 0.35, 0.15),
                   help="Probabilidades para buckets de lead time: 2-4, 5-7, 10-15")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    try:
        construir_supplier_catalog(
            Path(args.catalog_items_csv),
            Path(args.out_csv),
            seed=args.seed,
            probs=tuple(args.probs),
        )
        log.info("Supplier catalog construido correctamente.")
    except Exception as e:
        log.exception("Error construyendo supplier_catalog: %s", e)
        raise

if __name__ == "__main__":
    main()

# =============================================================================
# Script: unificar_sustitutos.py
# Descripción:
# Unifica sustitutos internos (multiproveedor: alternativas de proveedor para un
# mismo producto) y externos (productos similares por categoría) en un catálogo
# único que servirá para reglas de fallback en compras y stock.
#
# Flujo del pipeline:
# 1) Leer internos (data/clean/supplier_catalog_multi.csv) y externos
#    (data/clean/substitutes.csv).
# 2) Internos: detectar proveedor preferente por heurística y tomar el resto como
#    alternativas internas.
# 3) Externos: mapear product_id → sustituto_id (+score).
# 4) Unificar en un esquema superset y exportar a data/processed/substitutes_unified.csv.
#
# Input (por defecto):
#   - data/clean/supplier_catalog_multi.csv
#   - data/clean/substitutes.csv
#
# Output:
#   - data/processed/substitutes_unified.csv
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
ROOT_DIR = Path(__file__).resolve().parents[2]  # scripts/transform -> proyecto raíz
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
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Devuelve el nombre real de una columna buscando por candidatos (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"No se encontró ninguna de las columnas {candidates} en {list(df.columns)}")

# --- Internos (multiproveedor) ------------------------------------------------
def _marcar_preferente(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Marca un proveedor preferente por producto usando la heurística:
    prioridad ↑ (menor primero), precio ↑, lead_time ↑, disponibilidad ↓ (tie-break).
    El resto del grupo se considera alternativa interna.
    """
    g = grp.copy()
    # Asegura tipos numéricos cuando existan
    for c in ["prioridad", "precio", "lead_time", "disponibilidad"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.sort_values(
        by=["prioridad", "precio", "lead_time", "disponibilidad"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    g["__is_preferred"] = False
    if len(g) > 0:
        g.iloc[0, g.columns.get_loc("__is_preferred")] = True
    return g

def normalizar_internos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye sustitutos internos a partir de un catálogo multiproveedor con columnas:
    ['Product_ID','supplier_id','precio','disponibilidad','lead_time','prioridad','lead_time_bucket'].
    Devuelve un DataFrame con columnas superset:
      Product_ID, tipo, Substitute_Product_ID (NaN), Substitute_Supplier_ID,
      score (NaN), precio, disponibilidad, lead_time, prioridad, lead_time_bucket
    Solo se devuelven los proveedores NO preferentes de cada producto.
    """
    req = ["Product_ID", "supplier_id"]
    for r in req:
        if r not in df.columns:
            raise KeyError(f"Falta la columna requerida '{r}' en internos: {list(df.columns)}")

    marked = df.groupby("Product_ID", group_keys=False).apply(_marcar_preferente)
    alts = marked[~marked["__is_preferred"]].copy()

    if alts.empty:
        log.warning("No se han encontrado alternativas internas (multiproveedor con un único proveedor por producto).")

    out = alts.rename(columns={"supplier_id": "Substitute_Supplier_ID"})
    cols_keep = [
        "Product_ID",
        "Substitute_Supplier_ID",
        "precio",
        "disponibilidad",
        "lead_time",
        "prioridad",
        "lead_time_bucket",
    ]
    for c in cols_keep:
        if c not in out.columns:
            out[c] = np.nan

    out = out[cols_keep]
    out["tipo"] = "interno"
    out["Substitute_Product_ID"] = np.nan
    out["score"] = np.nan

    # Tipos
    out["Product_ID"] = out["Product_ID"].astype(str)
    return out[
        [
            "Product_ID",
            "tipo",
            "Substitute_Product_ID",
            "Substitute_Supplier_ID",
            "score",
            "precio",
            "disponibilidad",
            "lead_time",
            "prioridad",
            "lead_time_bucket",
        ]
    ]

# --- Externos (producto ↔ producto) ------------------------------------------
def normalizar_externos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza catálogo de sustitutos externos con columnas:
      product_id, sustituto_id, score
    Devuelve el mismo esquema superset que internos (rellenando con NaN lo que no aplique).
    """
    pid_a = _find_col(df, ["product_id", "Product_ID", "id_producto", "id_A"])
    pid_b = _find_col(df, ["sustituto_id", "Substitute_ID", "id_sustituto", "id_B"])

    out = df.rename(columns={pid_a: "Product_ID", pid_b: "Substitute_Product_ID"}).copy()
    if "score" not in out.columns:
        out["score"] = np.nan

    # Relleno de columnas no aplicables
    out["Substitute_Supplier_ID"] = np.nan
    for c in ["precio", "disponibilidad", "lead_time", "prioridad", "lead_time_bucket"]:
        out[c] = np.nan
    out["tipo"] = "externo"

    # Limpieza y tipos
    out = out.dropna(subset=["Product_ID", "Substitute_Product_ID"])
    out["Product_ID"] = out["Product_ID"].astype(str)
    out["Substitute_Product_ID"] = out["Substitute_Product_ID"].astype(str)
    out = out[out["Product_ID"] != out["Substitute_Product_ID"]]

    return out[
        [
            "Product_ID",
            "tipo",
            "Substitute_Product_ID",
            "Substitute_Supplier_ID",
            "score",
            "precio",
            "disponibilidad",
            "lead_time",
            "prioridad",
            "lead_time_bucket",
        ]
    ]

# ==== 3. LÓGICA PRINCIPAL =====================================================
def unificar_sustitutos(internos: pd.DataFrame, externos: pd.DataFrame) -> pd.DataFrame:
    df_all = pd.concat([internos, externos], ignore_index=True)

    # Elimina duplicados exactos y auto-sustituciones (por si entraran en externos)
    df_all = df_all[df_all["Product_ID"] != df_all["Substitute_Product_ID"]]
    df_all = df_all.drop_duplicates()

    # Orden consistente
    ordered_cols = [
        "Product_ID",
        "tipo",
        "Substitute_Product_ID",
        "Substitute_Supplier_ID",
        "score",
        "precio",
        "disponibilidad",
        "lead_time",
        "prioridad",
        "lead_time_bucket",
    ]
    for c in ordered_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan
    df_all = df_all[ordered_cols]

    return df_all.reset_index(drop=True)

# ==== 4. EXPORTACIÓN / I/O OPCIONAL ==========================================
def exportar_resultados(df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    ensure_dirs(out_path.parent)
    df.to_csv(out_path, index=False)
    return out_path

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unificación de sustitutos internos y externos.")
    p.add_argument(
        "--internos",
        type=str,
        default=str(CLEAN_DIR / "supplier_catalog_multi.csv"),
        help="Ruta de entrada para multiproveedor (CSV en data/clean).",
    )
    p.add_argument(
        "--externos",
        type=str,
        default=str(CLEAN_DIR / "substitutes.csv"),
        help="Ruta de entrada para sustitutos externos (CSV en data/clean).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(PROCESSED_DIR / "substitutes_unified.csv"),
        help="Ruta de salida del catálogo unificado (CSV en data/processed).",
    )
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    internos_path = Path(args.internos)
    externos_path = Path(args.externos)
    out_path = Path(args.out)

    try:
        if not internos_path.exists():
            raise FileNotFoundError(f"No existe el archivo de internos: {internos_path}")
        if not externos_path.exists():
            raise FileNotFoundError(f"No existe el archivo de externos: {externos_path}")

        log.info("Leyendo internos: %s", internos_path)
        df_internos = pd.read_csv(internos_path)

        log.info("Leyendo externos: %s", externos_path)
        df_externos = pd.read_csv(externos_path)

        log.info("Normalizando catálogos…")
        df_internos_norm = normalizar_internos(df_internos)
        df_externos_norm = normalizar_externos(df_externos)

        log.info("Unificando sustitutos…")
        df_unificado = unificar_sustitutos(df_internos_norm, df_externos_norm)

        written = exportar_resultados(df_unificado, out_path)
        log.info("Catálogo unificado escrito en: %s", written)

    except Exception as e:
        log.exception("Fallo en la ejecución: %s", e)
        raise

if __name__ == "__main__":
    main()

# =============================================================================
# Script: preparacion_dataset_modelado.py
# Descripción:
#   Prepara el dataset para el modelado:
#   - Renombra __cluster__ -> cluster_id y demand_final_noised -> sales_quantity
#   - Elimina columnas duplicadas/redundantes: cluster, __product_id__, demand_final_noiseds_adj (si existe)
#   - Normaliza tipos básicos (date, product_id, cluster_id)
#   - Controla duplicados por (product_id, date)
#   - Selecciona/features recomendadas (intersección segura)
#   - Guarda data/processed/dataset_modelado_ready.parquet (por defecto)
#
# Uso (CLI):
#   python scripts/transform/preparacion_dataset_modelado.py
#   python scripts/transform/preparacion_dataset_modelado.py \
#       --in "data/processed/subset_modelado.parquet" \
#       --out "data/processed/dataset_modelado_ready.parquet"
#
# Uso (notebook):
#   from scripts.transform.preparacion_dataset_modelado import run_prep
#   out_path = run_prep()  # o run_prep(inp=..., outp=...)
#   out_path
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import pandas as pd

# ---------- Helper: localizar raíz del repo (busca data/processed hacia arriba)
def find_repo_root(start: Path | None = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for parent in (p, *p.parents):
        if (parent / "data" / "processed").exists():
            return parent
    return p  # fallback

# ---------- Entorno (sirve para script y notebook)
if "__file__" in globals():
    _start = Path(__file__).resolve().parent
    LOGGER_NAME = Path(__file__).stem
else:
    _start = Path.cwd()
    LOGGER_NAME = "notebook.preparacion_dataset_modelado"

ROOT_DIR = find_repo_root(_start)
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ---------- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(LOGGER_NAME)

# ---------- Config “suave”: columnas a eliminar/renombrar/usar si existen
RENAME_MAP = {
    "__cluster__": "cluster_id",
    "demand_final_noised": "sales_quantity",
}
DROP_CANDIDATES = [
    "cluster",               # duplicado: nos quedamos con __cluster__ -> cluster_id
    "__product_id__",        # duplicado de product_id
    "demand_final_noiseds_adj",  # columna auxiliar que no aporta
]
# Features recomendadas (se usará la intersección para evitar KeyError)
FEATURES_RECOMENDADAS = [
    # ids & fecha (estos los forzamos aparte)
    # target -> sales_quantity (tras renombrado)
    "precio_medio",
    "price_virtual",
    "price_factor_effective",
    "demand_day_priceadj",
    # factores externos
    "m_agosto_nonprice",
    "m_competition",
    "m_inflation",
    "m_promo",
    # trazabilidad/outliers (opcionales, según uso como exógenas)
    "is_outlier",
    "tipo_outlier_year",
    "decision_outlier_year",
]

# ---------- Núcleo ------------------------------------------------------------
def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la preparación para modelado y devuelve el DataFrame listo."""
    df = df.copy()

    # 1) Renombrados (solo si existen)
    cols_a_renombrar = {c: n for c, n in RENAME_MAP.items() if c in df.columns}
    df = df.rename(columns=cols_a_renombrar)

    # Validaciones mínimas
    required = {"product_id", "date", "cluster_id", "sales_quantity"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas tras renombrado: {missing}")

    # 2) Eliminar columnas redundantes si existen
    to_drop = [c for c in DROP_CANDIDATES if c in df.columns]
    if to_drop:
        log.info("Eliminando columnas redundantes: %s", to_drop)
        df = df.drop(columns=to_drop)

    # 3) Normalizar tipos
    df["date"] = pd.to_datetime(df["date"])
    df["product_id"] = df["product_id"].astype(str)
    # cluster_id como int (permitiendo nulos si los hubiera por seguridad)
    if df["cluster_id"].isna().any():
        df["cluster_id"] = df["cluster_id"].astype("Int64")
    else:
        df["cluster_id"] = df["cluster_id"].astype(int)

    # 4) Control de duplicados por (product_id, date)
    dups = df.duplicated(["product_id", "date"])
    n_dup = int(dups.sum())
    if n_dup > 0:
        log.warning("Se detectaron %s duplicados (product_id, date). Se conservará el primero.", n_dup)
        df = df.loc[~dups].copy()

    # 5) Verificación de nulos en target
    n_null_target = int(df["sales_quantity"].isna().sum())
    if n_null_target > 0:
        log.warning("Se encontraron %s nulos en sales_quantity. Filtrando filas nulas.", n_null_target)
        df = df.loc[df["sales_quantity"].notna()].copy()

    # 6) Selección de columnas finales (intersección segura)
    keep_base = ["product_id", "date", "cluster_id", "sales_quantity"]
    keep_feats = [c for c in FEATURES_RECOMENDADAS if c in df.columns]
    cols_finales = keep_base + keep_feats
    df = df[cols_finales].sort_values(["product_id", "date"]).reset_index(drop=True)

    return df

# ---------- CLI ---------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preparación del dataset para modelado.")
    p.add_argument("--in",  dest="inp",  type=str, default=str(PROCESSED_DIR / "subset_modelado.parquet"),
                   help="Ruta de entrada (PARQUET).")
    p.add_argument("--out", dest="outp", type=str, default=str(PROCESSED_DIR / "dataset_modelado_ready.parquet"),
                   help="Ruta de salida (PARQUET).")
    # Ignora flags de Jupyter si corre dentro de notebook
    args, _ = p.parse_known_args()
    return args

def run_prep(inp: str | Path = None, outp: str | Path = None) -> str:
    """Atajo para usar desde notebook o como función."""
    inp_path = Path(inp) if inp else (PROCESSED_DIR / "subset_modelado.parquet")
    out_path = Path(outp) if outp else (PROCESSED_DIR / "dataset_modelado_ready.parquet")

    log.info("Leyendo: %s", inp_path)
    df = pd.read_parquet(inp_path)

    log.info("Preparando dataset…")
    df_ready = prepare_dataset(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_ready.to_parquet(out_path, index=False)
    log.info("Guardado dataset listo para modelado en: %s", out_path)

    return str(out_path)

def main() -> None:
    args = _parse_args()
    run_prep(args.inp, args.outp)

if __name__ == "__main__":
    main()

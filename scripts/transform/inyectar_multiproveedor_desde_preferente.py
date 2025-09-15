# =============================================================================
# Script: inyectar_multiproveedor_desde_preferente.py
# Descripción:
# Aplicación de la herramienta `asignar_multiproveedor` para **inyectar**
# proveedores alternativos (escenarios de **multiproveedor**) a partir del
# catálogo con proveedor preferente. Este script **sí** realiza E/S y expone
# parámetros por CLI; genera un `supplier_catalog_multi.csv` listo para la FASE 9.3.
#
# Flujo del pipeline:
# 1) Leer `preferred_supplier.(csv|parquet)` (salida FASE 9.3 o equivalente).
# 2) Construir catálogo **base** (1 proveedor por Product_ID) con
#    `build_base_from_preferred`.
# 3) (Opcional) Leer `supplier_catalog.csv` como **universo** de proveedores para
#    escoger supplier_id “reales” y su lead_time típico.
# 4) Generar alternativos:
#    4.1) **Automático** con `auto_generate_alternatives` (garantiza supplier ≠ preferente;
#         varía precio, disponibilidad —con probabilidad de 0 para alertas— y lead_time).
#    4.2) **Manual** (si se pasa CSV) validando con `validate_manual` y concatenando.
# 5) Unir **base + alternativos** (auto + manual), eliminar duplicados (Product_ID, supplier_id)
#    y **exportar** como `supplier_catalog_multi.csv`.
# 6) Mostrar **resumen** (filas, nº de productos multiproveedor antes/después, ejemplos).
#
# Input:
#   - data/processed/preferred_supplier.(csv|parquet)
#   - (opcional) data/clean/supplier_catalog.(csv|parquet) como universo de proveedores
#   - (opcional) data/clean/alternativos_manual.csv con columnas:
#       Product_ID, supplier_id, precio, disponibilidad, lead_time, prioridad
#
# Output:
#   - data/clean/supplier_catalog_multi.csv  (Catálogo listo para FASE 9.3)
#
# Dependencias:
#   - pandas
#   - numpy
#   - (opcional) pyarrow (si lees/escribes parquet)
#
# Instalación rápida:
#   pip install pandas numpy pyarrow
#
# Ejemplos de uso:
#   # 50 productos con alternativo (automático), usando universo real
#   python scripts/transform/inyectar_multiproveedor_desde_preferente.py ^
#     --k 50 ^
#     --supplier_universe data/clean/supplier_catalog.csv
#
#   # 5% de productos + lista manual adicional
#   python scripts/transform/inyectar_multiproveedor_desde_preferente.py ^
#     --pct 0.05 ^
#     --manual data/clean/alternativos_manual.csv
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
import importlib.util

# --------------------- Rutas base ---------------------
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"

# ----------------- Import herramienta por RUTA -----------------
TOOL_PATH = ROOT_DIR / "scripts" / "utils" / "asignar_multiproveedor.py"
if not TOOL_PATH.exists():
    raise FileNotFoundError(f"No se encontró la herramienta en: {TOOL_PATH}")

spec = importlib.util.spec_from_file_location("asignar_multiproveedor", TOOL_PATH)
tool = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
assert spec and spec.loader
spec.loader.exec_module(tool)                 # type: ignore[attr-defined]

build_base_from_preferred  = tool.build_base_from_preferred
validate_manual            = tool.validate_manual
auto_generate_alternatives = tool.auto_generate_alternatives

# --------------------- Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# --------------------- IO helpers --------------------
def read_preferred(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe preferred_supplier: {path}")
    return pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)

def read_universe(path: Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p, dtype=str, low_memory=False)
    for c in ["precio", "disponibilidad", "lead_time", "prioridad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "supplier_id" in df.columns:
        df["supplier_id"] = df["supplier_id"].astype("string").str.strip()
    return df

# --------------------- Args --------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inyecta multiproveedor desde preferred_supplier usando la herramienta genérica."
    )
    p.add_argument("--preferred", type=str, default=str(PROCESSED_DIR / "preferred_supplier.csv"))
    p.add_argument("--supplier_universe", type=str, default=str(CLEAN_DIR / "supplier_catalog.csv"),
                   help="(Opcional) universo de proveedores para elegir supplier_id reales y su lead_time típico.")
    p.add_argument("--manual", type=str, default="",
                   help="(Opcional) CSV con alternativos manuales: Product_ID,supplier_id,precio,disponibilidad,lead_time,prioridad")
    p.add_argument("--out", type=str, default=str(CLEAN_DIR / "supplier_catalog_multi.csv"))

    g = p.add_mutually_exclusive_group()
    g.add_argument("--k", type=int, default=50, help="Nº de productos a los que añadir alternativo (modo AUTO).")
    g.add_argument("--pct", type=float, default=None, help="Porcentaje 0-1 de productos (modo AUTO).")

    p.add_argument("--prob-zero", type=float, default=0.15)
    p.add_argument("--stock-min", type=float, default=0.6)
    p.add_argument("--stock-max", type=float, default=1.2)
    p.add_argument("--price-min", type=float, default=0.9)
    p.add_argument("--price-max", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=23)

    p.add_argument("--dry-run", action="store_true", help="No escribe salida, solo muestra diagnóstico.")
    p.add_argument("--debug",   action="store_true", help="Muestra conteos y ejemplos adicionales.")
    return p.parse_args()

# --------------------- Fallback generator -------------
def _fallback_generate_from_base(
    base_df: pd.DataFrame,
    universe_df: pd.DataFrame | None,
    *,
    k: int | None,
    pct: float | None,
    prob_zero: float,
    stock_min: float,
    stock_max: float,
    price_min: float,
    price_max: float,
    seed: int,
) -> pd.DataFrame:
    """
    Si auto_generate_alternatives devuelve 0 filas (p.ej., sin intersección),
    generamos alternativos directamente desde base_df.
    """
    rng = np.random.default_rng(seed)
    base = base_df.copy()
    base["Product_ID"]  = base["Product_ID"].astype("string").str.strip()
    base["supplier_id"] = base["supplier_id"].astype("string").str.strip()

    # Universo de suppliers para elegir IDs “reales”
    if universe_df is not None and "supplier_id" in universe_df.columns:
        pool = universe_df["supplier_id"].dropna().astype("string").str.strip().unique().tolist()
        lead_map = {}
        if "lead_time" in universe_df.columns:
            uni = universe_df.dropna(subset=["supplier_id"]).copy()
            uni["lead_time"] = pd.to_numeric(uni.get("lead_time"), errors="coerce")
            sup_lead = uni.sort_values("lead_time").drop_duplicates(subset=["supplier_id"], keep="first")
            lead_map = dict(zip(sup_lead["supplier_id"], sup_lead["lead_time"]))
    else:
        pool = base["supplier_id"].unique().tolist()
        lead_map = {}

    prods = base["Product_ID"].unique().tolist()
    if pct is not None:
        n = max(1, int(len(prods) * pct))
    elif k is not None:
        n = min(max(k, 1), len(prods))
    else:
        n = min(40, len(prods))

    picks = rng.choice(np.array(prods, dtype=object), size=n, replace=False)

    out = []
    for pid in picks:
        row = base.loc[base["Product_ID"] == pid].iloc[0]
        supplier_pref = row["supplier_id"]
        px_pref = float(row["precio"])
        disp_pref = float(row["disponibilidad"])

        minus_pref = [s for s in pool if str(s) != str(supplier_pref)]
        if not minus_pref:
            minus_pref = [str(supplier_pref) + "_ALT"]
        supplier_alt = str(rng.choice(minus_pref))

        px_alt = px_pref * float(rng.uniform(price_min, price_max))
        if rng.uniform() < prob_zero:
            disp_alt = 0.0
        else:
            disp_alt = disp_pref * float(rng.uniform(stock_min, stock_max))

        if supplier_alt in lead_map and not pd.isna(lead_map.get(supplier_alt, np.nan)):
            lt_alt = int(max(1, round(float(lead_map[supplier_alt]))))
        else:
            bucket = rng.choice(["2-4", "5-7", "10-15"], p=[0.5, 0.35, 0.15])
            lo, hi = (2, 4) if bucket == "2-4" else (5, 7) if bucket == "5-7" else (10, 15)
            lt_alt = int(rng.integers(lo, hi + 1))

        out.append({
            "Product_ID": str(pid),
            "supplier_id": supplier_alt,
            "precio": px_alt,
            "disponibilidad": disp_alt,
            "lead_time": lt_alt,
            "prioridad": 1,
            "lead_time_bucket": "2-4" if lt_alt <= 4 else ("5-7" if lt_alt <= 7 else "10-15"),
        })

    return pd.DataFrame(out)

# --------------------- Main --------------------------
def main() -> None:
    args = _parse_args()
    preferred_path = Path(args.preferred)
    universe_path  = Path(args.supplier_universe) if args.supplier_universe else None
    out_path       = Path(args.out)

    pref_df     = read_preferred(preferred_path)
    base_df     = build_base_from_preferred(pref_df)
    universe_df = read_universe(universe_path)

    # Diagnóstico previo
    if args.debug:
        inter = set(base_df["Product_ID"].astype("string")) & set(pref_df["Product_ID"].astype("string"))
        log.info("DEBUG | base_df: %s filas, preferred: %s filas, intersección Product_ID: %s",
                 len(base_df), len(pref_df), len(inter))

    # AUTO (herramienta)
    alts_auto = auto_generate_alternatives(
        base=base_df, pref=pref_df, universe=universe_df,
        k=args.k, pct=args.pct,
        prob_zero=args.prob_zero,
        stock_min=args.stock_min, stock_max=args.stock_max,
        price_min=args.price_min, price_max=args.price_max,
        seed=args.seed,
    )

    if args.debug:
        log.info("DEBUG | Alternativos AUTO generados: %s", len(alts_auto))

    # Fallback si no hay alternativos
    if alts_auto.empty:
        log.warning("No se generaron alternativos en AUTO (0 filas). Activando Fallback.")
        alts_auto = _fallback_generate_from_base(
            base_df=base_df, universe_df=universe_df,
            k=args.k, pct=args.pct,
            prob_zero=args.prob_zero,
            stock_min=args.stock_min, stock_max=args.stock_max,
            price_min=args.price_min, price_max=args.price_max,
            seed=args.seed,
        )
        log.info("Fallback generó %s alternativos.", len(alts_auto))

    # MANUAL (opcional)
    if args.manual:
        manual_path = Path(args.manual)
        if manual_path.exists():
            manual_df = validate_manual(pd.read_csv(manual_path))
            pref_map = pref_df.set_index("Product_ID")["supplier_id_pref"].astype("string")
            bad = manual_df["Product_ID"].isin(pref_map.index) & (
                manual_df["supplier_id"] == manual_df["Product_ID"].map(pref_map)
            )
            manual_df = manual_df.loc[~bad].copy()
        else:
            log.warning("No se encontró CSV manual: %s", manual_path)
            manual_df = pd.DataFrame(columns=["Product_ID","supplier_id","precio","disponibilidad","lead_time","prioridad","lead_time_bucket"])
    else:
        manual_df = pd.DataFrame(columns=["Product_ID","supplier_id","precio","disponibilidad","lead_time","prioridad","lead_time_bucket"])

    # Unión
    sup_multi = pd.concat([base_df, alts_auto, manual_df], ignore_index=True)\
                  .drop_duplicates(subset=["Product_ID","supplier_id"], keep="first")

    # Diagnóstico final
    prev_multi = base_df.groupby("Product_ID")["supplier_id"].nunique().gt(1).sum()
    post_multi = sup_multi.groupby("Product_ID")["supplier_id"].nunique().gt(1).sum()

    print("\nResumen inyección multiproveedor")
    print(f"- Preferente                : {preferred_path}")
    print(f"- Universo proveedores (opt): {universe_path if universe_path else '(no usado)'}")
    print(f"- Alternativos AUTO/Fallback: {len(alts_auto):,}")
    print(f"- Alternativos MANUAL       : {len(manual_df):,}")
    print(f"- Filas base                : {len(base_df):,}")
    print(f"- Filas totales             : {len(sup_multi):,}")
    print(f"- Productos multiproveedor antes : {prev_multi}")
    print(f"- Productos multiproveedor después: {post_multi}")
    if args.debug and not alts_auto.empty:
        print(">> Muestras de alternativos:\n", alts_auto.head(5))

    # Escritura
    if args.dry_run:
        log.info("Dry-run: no se escribe archivo (--dry-run activo).")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sup_multi.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info("Escrito: %s  (filas=%s)", out_path, len(sup_multi))

if __name__ == "__main__":
    main()
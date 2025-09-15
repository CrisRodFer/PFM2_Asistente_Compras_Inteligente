
# =============================================================================
# Script: asignar_multiproveedor.py
# Descripción:
# Herramienta **GENÉRICA y reutilizable** (sin efectos secundarios) para generar
# proveedores alternativos (escenarios de **multiproveedor**) a partir de
# DataFrames en memoria. No lee ni escribe ficheros, no expone CLI: ofrece
# funciones puras para que otros scripts/notebooks la invoquen.
#
# Flujo del pipeline (uso típico desde otro script/notebook):
# 1) Cargar datasets necesarios (p. ej., preferred_supplier y/o supplier_catalog).
# 2) Construir catálogo base desde preferred_supplier con `build_base_from_preferred`.
# 3) Generar alternativos:
#    3.1) **Automático** con `auto_generate_alternatives` (supplier ≠ preferente).
#    3.2) **Manual** validando una lista con `validate_manual`.
# 4) Unir base + alternativos y continuar el flujo (p. ej., FASE 9.3).
#
# Input (en memoria):
#   - preferred_supplier_df (DataFrame) con columnas:
#       Product_ID, supplier_id_pref, precio_pref, disp_pref, lead_time_pref
#       (opcional) lead_time_bucket_pref
#   - (opcional) supplier_universe_df (DataFrame) para elegir supplier_id “reales”
#       y su lead_time típico.
#   - (opcional) lista manual (DataFrame) con columnas:
#       Product_ID, supplier_id, precio, disponibilidad, lead_time, prioridad
#
# Output (en memoria):
#   - base_df = catálogo base con columnas:
#       Product_ID, supplier_id, precio, disponibilidad, lead_time, prioridad, lead_time_bucket
#   - alts_auto_df = alternativos **automáticos** (mismo esquema que base_df)
#   - alts_manual_df = alternativos **manuales** validados (mismo esquema)
#
# Dependencias:
#   - pandas
#   - numpy
#
# Instalación rápida:
#   pip install pandas numpy
# =============================================================================


from __future__ import annotations
from pathlib import Path
import argparse
import logging
from typing import Iterable, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
log = logging.getLogger("asignar_multiproveedor")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# ---------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------
def _norm_id(s: pd.Series) -> pd.Series:
    """Normaliza IDs: quita espacios, convierte numéricos a str sin .0."""
    s = s.astype("string")
    as_num = pd.to_numeric(s, errors="coerce")
    s.loc[as_num.notna()] = as_num.loc[as_num.notna()].astype("Int64").astype(str)
    return s.str.strip()

def _bucket_from_lt(x: float | int | None) -> str | pd.NA:
    if pd.isna(x): 
        return pd.NA
    x = int(x)
    if 2 <= x <= 4:
        return "2-4"
    if 5 <= x <= 7:
        return "5-7"
    return "10-15"

def _away_from_one(rng: np.random.Generator, low: float, high: float, eps: float) -> float:
    """
    Genera valores en [low, 1-eps] U [1+eps, high].
    Evita factores pegados a 1 para que superen la validación.
    """
    lo1, hi1 = low, max(low, 1 - eps)
    lo2, hi2 = min(1 + eps, high), high
    # si el rango alrededor de 1 es demasiado estrecho, ensancha un poco
    if hi1 <= lo1 and hi2 <= lo2:
        lo1, hi1, lo2, hi2 = low, 0.97, 1.03, high
    if rng.uniform() < 0.5 and hi1 > lo1:
        return float(rng.uniform(lo1, hi1))
    return float(rng.uniform(lo2, hi2))

# ---------------------------------------------------------------------
# 1) Construcción base a partir de preferred_supplier
# ---------------------------------------------------------------------
def build_base_from_preferred(preferred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el preferred_supplier en catálogo base (1 fila por producto-proveedor preferente).
    Columnas esperadas en preferred_df:
        Product_ID, supplier_id_pref, precio_pref, disp_pref, lead_time_pref
    Devuelve DataFrame con:
        Product_ID, supplier_id, precio, disponibilidad, lead_time, prioridad, lead_time_bucket
    """
    df = preferred_df.copy()
    req = {"Product_ID", "supplier_id_pref", "precio_pref", "disp_pref", "lead_time_pref"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Faltan columnas en preferred_supplier: {sorted(miss)}")

    df["Product_ID"] = _norm_id(df["Product_ID"])
    df["supplier_id"] = df["supplier_id_pref"].astype("string").str.strip()
    df["precio"] = pd.to_numeric(df["precio_pref"], errors="coerce")
    df["disponibilidad"] = pd.to_numeric(df["disp_pref"], errors="coerce")
    df["lead_time"] = pd.to_numeric(df["lead_time_pref"], errors="coerce").fillna(7).astype(int)
    df["prioridad"] = 1
    df["lead_time_bucket"] = df["lead_time"].map(_bucket_from_lt)

    cols = ["Product_ID","supplier_id","precio","disponibilidad","lead_time","prioridad","lead_time_bucket"]
    base = df[cols].copy()
    return base

# ---------------------------------------------------------------------
# 2) Validación de alternativos manuales
# ---------------------------------------------------------------------
def validate_manual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida/normaliza un CSV manual con columnas:
        Product_ID, supplier_id, precio, disponibilidad, lead_time, prioridad
    Rellena lead_time_bucket y normaliza tipos.
    """
    need = {"Product_ID","supplier_id","precio","disponibilidad","lead_time","prioridad"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"El CSV manual debe contener columnas: {sorted(need)}. Faltan: {sorted(miss)}")

    out = df.copy()
    out["Product_ID"] = _norm_id(out["Product_ID"])
    out["supplier_id"] = out["supplier_id"].astype("string").str.strip()
    out["precio"] = pd.to_numeric(out["precio"], errors="coerce")
    out["disponibilidad"] = pd.to_numeric(out["disponibilidad"], errors="coerce").clip(lower=0)
    out["lead_time"] = pd.to_numeric(out["lead_time"], errors="coerce").fillna(7).astype(int).clip(lower=1)
    out["prioridad"] = pd.to_numeric(out["prioridad"], errors="coerce").fillna(1).astype(int)
    out["lead_time_bucket"] = out["lead_time"].map(_bucket_from_lt)
    return out

# ---------------------------------------------------------------------
# 3) Generador AUTO de alternativos (núcleo reutilizable)
# ---------------------------------------------------------------------
def auto_generate_alternatives(
    *,
    base: pd.DataFrame,
    pref: Optional[pd.DataFrame] = None,
    universe: Optional[pd.DataFrame] = None,
    k: Optional[int] = None,
    pct: Optional[float] = None,
    prob_zero: float = 0.15,
    stock_min: float = 0.6,
    stock_max: float = 1.2,
    price_min: float = 0.9,
    price_max: float = 1.1,
    seed: int = 23,
    eps_price: float = 0.01,
    eps_stock: float = 0.02,
) -> pd.DataFrame:
    """
    Genera proveedores alternativos para un subconjunto de productos.
    - supplier_id del alternativo SIEMPRE distinto del preferente.
    - precio/disponibilidad se alejan de 1 (±eps) para pasar validaciones.
    - lead_time derivado del universo si existe; si no, bucket aleatorio.

    base: catálogo base con columnas:
        Product_ID, supplier_id, precio, disponibilidad, lead_time
        (suele ser el preferente)
    pref: preferred_supplier para mapear el supplier preferente (opcional pero recomendado)
        requiere columnas: Product_ID, supplier_id_pref, precio_pref, disp_pref, lead_time_pref
    universe: universo de proveedores para tomar supplier_id reales y su lead_time típico (opcional)

    Devuelve DF con columnas:
        Product_ID, supplier_id, precio, disponibilidad, lead_time, prioridad, lead_time_bucket
        (SOLO las filas de alternativos)
    """
    rng = np.random.default_rng(seed)

    # Normalizaciones mínimas
    b = base.copy()
    b["Product_ID"] = _norm_id(b["Product_ID"])
    b["supplier_id"] = b["supplier_id"].astype("string").str.strip()
    b["precio"] = pd.to_numeric(b["precio"], errors="coerce")
    b["disponibilidad"] = pd.to_numeric(b["disponibilidad"], errors="coerce")
    b["lead_time"] = pd.to_numeric(b["lead_time"], errors="coerce").fillna(7).astype(int)

    # Mapeo preferente
    if pref is not None and "supplier_id_pref" in pref.columns:
        p = pref.copy()
        p["Product_ID"] = _norm_id(p["Product_ID"])
        pref_map = p.set_index("Product_ID")["supplier_id_pref"].astype("string").str.strip()
    else:
        # Si no nos lo pasan, usamos el supplier del base como "preferente"
        pref_map = b.set_index("Product_ID")["supplier_id"].astype("string")

    # Universo para escoger suppliers reales y lead_time típico
    if universe is not None and "supplier_id" in universe.columns:
        uni = universe.copy()
        uni["supplier_id"] = uni["supplier_id"].astype("string").str.strip()
        pool = uni["supplier_id"].dropna().unique().tolist()
        # lead_time típico por supplier (min de los disponibles)
        if "lead_time" in uni.columns:
            uni["lead_time"] = pd.to_numeric(uni["lead_time"], errors="coerce")
            sup_lead = uni.sort_values("lead_time").drop_duplicates(subset=["supplier_id"], keep="first")
            lead_map = dict(zip(sup_lead["supplier_id"], sup_lead["lead_time"]))
        else:
            lead_map = {}
    else:
        pool = b["supplier_id"].dropna().unique().tolist()
        lead_map = {}

    # Candidatos (por defecto, todos los Product_ID del base)
    products = b["Product_ID"].dropna().unique().tolist()
    if pct is not None:
        n = max(1, int(len(products) * float(pct)))
    elif k is not None:
        n = min(max(int(k), 1), len(products))
    else:
        n = min(40, len(products))

    if n == 0:
        return pd.DataFrame(columns=["Product_ID","supplier_id","precio","disponibilidad","lead_time","prioridad","lead_time_bucket"])

    picks = rng.choice(np.array(products, dtype=object), size=n, replace=False)

    rows = []
    for pid in picks:
        # datos del preferente/base
        row_b = b.loc[b["Product_ID"] == pid].iloc[0]
        supplier_pref = str(pref_map.get(pid, row_b["supplier_id"]))
        px_pref = float(row_b["precio"]) if not pd.isna(row_b["precio"]) else 0.0
        disp_pref = float(row_b["disponibilidad"]) if not pd.isna(row_b["disponibilidad"]) else 0.0

        # elegir supplier alternativo != preferente
        minus_pref = [s for s in pool if str(s) != supplier_pref]
        if not minus_pref:
            minus_pref = [supplier_pref + "_ALT"]
        supplier_alt = str(rng.choice(minus_pref))

        # precio y stock alejados de 1
        px_factor = _away_from_one(rng, price_min, price_max, eps=eps_price)
        px_alt = px_pref * px_factor

        if rng.uniform() < prob_zero:
            disp_alt = 0.0
        else:
            st_factor = _away_from_one(rng, stock_min, stock_max, eps=eps_stock)
            disp_alt = disp_pref * st_factor

        # redondeo de disponibilidad y garantía de diferencia
        disp_alt = float(np.round(disp_alt, 0))
        if disp_pref > 0 and abs(disp_alt - disp_pref) < 1e-9:
            # mueve ±1 unidad respetando no-negatividad
            disp_alt = max(0.0, disp_alt + (1.0 if rng.uniform() < 0.5 else -1.0))

        # lead time del universo si existe; si no, aleatorio por bucket
        if supplier_alt in lead_map and not pd.isna(lead_map.get(supplier_alt, np.nan)):
            lt_alt = int(max(1, round(float(lead_map[supplier_alt]))))
        else:
            bucket = rng.choice(["2-4","5-7","10-15"], p=[0.5,0.35,0.15])
            lo, hi = (2,4) if bucket=="2-4" else (5,7) if bucket=="5-7" else (10,15)
            lt_alt = int(rng.integers(lo, hi+1))

        rows.append({
            "Product_ID": str(pid),
            "supplier_id": supplier_alt,
            "precio": float(px_alt),
            "disponibilidad": float(disp_alt),
            "lead_time": int(lt_alt),
            "prioridad": 1,
            "lead_time_bucket": "2-4" if lt_alt <= 4 else ("5-7" if lt_alt <= 7 else "10-15"),
        })

    alts = pd.DataFrame(rows)
    # evitar duplicados por si el pool devuelve suppliers repetidos
    alts = alts.drop_duplicates(subset=["Product_ID","supplier_id"], keep="first").reset_index(drop=True)
    return alts

# ---------------------------------------------------------------------
# 4) CLI opcional (simulación sobre supplier_catalog.csv)
# ---------------------------------------------------------------------
def _read_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet",".pq") else pd.read_csv(path, dtype=str, low_memory=False)
    # tipado
    df["Product_ID"] = _norm_id(df["Product_ID"])
    for c_in, c_out in [("supplier_id","supplier_id"), ("precio","precio"), ("disponibilidad","disponibilidad"), ("lead_time","lead_time")]:
        if c_in in df.columns:
            if c_out in ["precio","disponibilidad","lead_time"]:
                df[c_out] = pd.to_numeric(df[c_in], errors="coerce")
            else:
                df[c_out] = df[c_in].astype("string").str.strip()
    if "prioridad" not in df.columns:
        df["prioridad"] = 1
    if "lead_time_bucket" not in df.columns:
        df["lead_time_bucket"] = df["lead_time"].map(_bucket_from_lt)
    # columnas mínimas
    cols = ["Product_ID","supplier_id","precio","disponibilidad","lead_time","prioridad","lead_time_bucket"]
    return df[[c for c in cols if c in df.columns]]

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Herramienta auxiliar: asigna proveedores alternativos sintéticos."
    )
    p.add_argument("--in", dest="inp", type=str, default=str(Path("data/clean/supplier_catalog.csv")))
    p.add_argument("--out", type=str, default="")  # si vacío => dry-run
    g = p.add_mutually_exclusive_group()
    g.add_argument("--k", type=int, default=10, help="Número de productos a modificar.")
    g.add_argument("--pct", type=float, default=None, help="Porcentaje (0-1) de productos.")
    p.add_argument("--price-min", type=float, default=0.9)
    p.add_argument("--price-max", type=float, default=1.1)
    p.add_argument("--stock-min", type=float, default=0.6)
    p.add_argument("--stock-max", type=float, default=1.2)
    p.add_argument("--prob-zero", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--apply", action="store_true", help="Escribe --out. Si no se pasa, es dry-run (no guarda).")
    return p.parse_args()

def _cli_main() -> None:
    args = _parse_cli()
    inp = Path(args.inp)
    out = Path(args.out) if args.out else None

    base = _read_catalog(inp)
    # usamos la propia base como universo simple (solo suppliers y lead_time)
    uni = base[["supplier_id","lead_time"]].drop_duplicates().copy()

    alts = auto_generate_alternatives(
        base=base,
        pref=None,
        universe=uni,
        k=args.k,
        pct=args.pct,
        prob_zero=args.prob_zero,
        stock_min=args.stock_min,
        stock_max=args.stock_max,
        price_min=args.price_min,
        price_max=args.price_max,
        seed=args.seed,
    )

    multi = pd.concat([base, alts], ignore_index=True)\
              .drop_duplicates(subset=["Product_ID","supplier_id"], keep="first")

    before = base.groupby("Product_ID")["supplier_id"].nunique().gt(1).sum()
    after  = multi.groupby("Product_ID")["supplier_id"].nunique().gt(1).sum()

    print("\nResumen simulación multiproveedor")
    print(f"- Entrada             : {inp}")
    print(f"- Filas entrada       : {len(base):,}")
    print(f"- Alternativos añadidos: {len(alts):,}")
    print(f"- Filas salida        : {len(multi):,}")
    print(f"- Productos multiproveedor (antes) : {before}")
    print(f"- Productos multiproveedor (después): {after}")

    if args.apply and out:
        out.parent.mkdir(parents=True, exist_ok=True)
        multi.to_csv(out, index=False, encoding="utf-8-sig")
        log.info("Guardado catálogo simulado en: %s", out)
    else:
        log.info("Dry-run: no se escribió ningún archivo. Usa --apply y --out para guardar.")

# ---------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    _cli_main()
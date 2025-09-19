
# ============================================================
# Script: generar_pedidos_clientes.py
# Descripción:
#   Genera pedidos de cliente sintéticos para probar el ciclo
#   de inventario (Fase 10). Produce un CSV con columnas:
#     - date (YYYY-MM-DD)
#     - item_id (int)
#     - qty (int, >0)
#
# Flujo pipeline:
#   0) Config rutas (Pathlib)
#   1) Imports + logging
#   2) Utilidades (muestreos, validaciones)
#   3) Lógica principal (generación por día)
#   4) Exportación / I/O (append o overwrite)
#   5) CLI / main (argumentos)
#
# Input esperado:
#   - Fuente de IDs: inventario (data/clean/Inventario.csv) o catálogo
#     (data/processed/catalog_items_enriquecido.csv). Se prioriza inventario
#     (para no pedir SKUs inexistentes) y si no existe se usa catálogo.
#
# Output:
#   - data/raw/customer_orders_AE.csv (por defecto) o ruta personalizada.
#
# Dependencias:
#   - Python 3.10+, pandas, numpy
#   - pip quick: pip install pandas numpy
# ============================================================

from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
import argparse
import logging
import random

import numpy as np
import pandas as pd

# ----- Rutas del proyecto -----
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"

CAT_ENR = PROCESSED_DIR / "catalog_items_enriquecido.csv"
INV_CSV = CLEAN_DIR / "Inventario.csv"
DEFAULT_OUT = RAW_DIR / "customer_orders_AE.csv"

# ----- Logging -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gen_pedidos")


# -------------------- Utilidades --------------------
def read_csv_smart(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, sep=None, engine="python", encoding="utf-8")


def load_ids_and_weights(outlier_weight: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (ids, probs) para muestreo ponderado.
    Fuente preferente: catalog_items_enriquecido.csv (Product_ID, is_outlier).
    Fallback: Inventario.csv (item_id o Product_ID) con probs uniformes.
    """
    cat = read_csv_smart(CAT_ENR)
    if cat is not None and not cat.empty:
        low = {c.lower(): c for c in cat.columns}
        pid_col = low.get("product_id")
        out_col = low.get("is_outlier")
        if not pid_col:
            raise ValueError("El catálogo enriquecido no tiene columna Product_ID.")
        ids = pd.to_numeric(cat[pid_col], errors="coerce").dropna().astype(int).values
        if out_col:
            out_flags = cat[out_col].fillna(0).astype(int).values
            w = np.where(out_flags == 1, float(outlier_weight), 1.0).astype(float)
        else:
            w = np.ones_like(ids, dtype=float)
        w = np.clip(w, 1e-9, None)
        probs = w / w.sum()
        log.info("IDs desde catalog_items_enriquecido.csv (%d). Peso outliers=%s", ids.size, outlier_weight)
        return ids, probs

    inv = read_csv_smart(INV_CSV)
    if inv is not None and not inv.empty:
        low = {c.lower(): c for c in inv.columns}
        col = low.get("item_id") or low.get("product_id")
        if not col:
            raise ValueError("Inventario.csv no tiene item_id/Product_ID.")
        ids = pd.to_numeric(inv[col], errors="coerce").dropna().astype(int).values
        probs = np.ones_like(ids, dtype=float) / max(1, ids.size)
        log.info("IDs desde Inventario.csv (%d). Probabilidades uniformes.", ids.size)
        return ids, probs

    raise FileNotFoundError("Sin fuente de IDs (ni catálogo enriquecido ni inventario).")


def rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 12345)


def sample_orders_for_day(
    the_date: date,
    ids: np.ndarray,
    probs: np.ndarray,
    n_orders: int,
    items_per_order: tuple[int, int],
    qty_range: tuple[int, int],
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    min_items, max_items = items_per_order
    qmin, qmax = qty_range
    for _ in range(n_orders):
        k = int(rng.integers(min_items, max_items + 1))
        k = max(1, min(k, ids.size))
        # Muestreo ponderado sin reposición dentro del pedido
        chosen = rng.choice(ids, size=k, replace=False, p=probs)
        q_lines = rng.integers(qmin, qmax + 1, size=k)
        rows.extend({"date": the_date.isoformat(), "item_id": int(pid), "qty": int(q)} for pid, q in zip(chosen, q_lines))
    return rows


# -------------------- API principal --------------------
def generate_orders(
    start: date,
    end: date,
    orders_per_day: int | tuple[int, int] = 80,
    items_per_order: tuple[int, int] = (1, 3),
    qty_range: tuple[int, int] = (1, 3),
    seed: int | None = 42,
    outlier_weight: int = 5,
) -> pd.DataFrame:
    """
    Genera pedidos sintéticos en el rango [start, end] (ambos inclusive).
    - orders_per_day: entero fijo o rango (min,max)
    - items_per_order: rango de ítems por pedido
    - qty_range: rango de cantidades por línea
    - outlier_weight: multiplicador de prob para is_outlier=1
    """
    ids, probs = load_ids_and_weights(outlier_weight=outlier_weight)
    rng = rng_from_seed(seed)

    rows: list[dict] = []
    cur = start
    while cur <= end:
        n_day = rng.integers(orders_per_day[0], orders_per_day[1] + 1) if isinstance(orders_per_day, tuple) else int(orders_per_day)
        rows.extend(
            sample_orders_for_day(
                the_date=cur,
                ids=ids,
                probs=probs,
                n_orders=int(n_day),
                items_per_order=items_per_order,
                qty_range=qty_range,
                rng=rng,
            )
        )
        cur += timedelta(days=1)

    df = pd.DataFrame(rows, columns=["date", "item_id", "qty"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").astype("Int64")
    df = df[df["qty"] > 0].copy()
    return df


def save_orders(df: pd.DataFrame, out_path: Path, mode: str = "append") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"append", "overwrite"}:
        raise ValueError("mode debe ser 'append' u 'overwrite'")
    if mode == "overwrite" or not out_path.exists():
        df.to_csv(out_path, index=False)
    else:
        base = read_csv_smart(out_path)
        if base is None or base.empty:
            df.to_csv(out_path, index=False)
        else:
            pd.concat([base, df], ignore_index=True).to_csv(out_path, index=False)
    return out_path


# -------------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera pedidos sintéticos ponderando top ventas (is_outlier=1).")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--orders-per-day", type=str, default="80", help="Entero o rango 'min,max' (p.ej. 50,100)")
    p.add_argument("--items-per-order", type=str, default="1,3", help="Rango 'min,max' de ítems por pedido")
    p.add_argument("--qty-range", type=str, default="1,3", help="Rango 'min,max' para qty por línea")
    p.add_argument("--seed", type=int, default=42, help="Semilla RNG")
    p.add_argument("--outlier-weight", type=int, default=5, help="Peso para is_outlier=1 (≥1)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Ruta CSV de salida")
    p.add_argument("--mode", type=str, choices=["append", "overwrite"], default="append", help="append/overwrite")
    return p.parse_args()


def _parse_pair(txt: str) -> tuple[int, int]:
    parts = [int(x) for x in txt.split(",")]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        a, b = parts
        return (min(a, b), max(a, b))
    raise ValueError(f"Formato inválido: {txt!r}")


def main():
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    opd = _parse_pair(args.orders_per_day) if "," in args.orders_per_day else (int(args.orders_per_day), int(args.orders_per_day))
    items = _parse_pair(args.items_per_order)
    qty = _parse_pair(args.qty_range)

    log.info("Generando: %s→%s | opd=%s | items/ped=%s | qty=%s | outlier_w=%d | seed=%s",
             start, end, opd, items, qty, args.outlier_weight, args.seed)

    df = generate_orders(
        start=start,
        end=end,
        orders_per_day=opd,
        items_per_order=items,
        qty_range=qty,
        seed=args.seed,
        outlier_weight=int(args.outlier_weight),
    )
    out = save_orders(df, Path(args.out), mode=args.mode)
    log.info("Escrito %d líneas en %s", len(df), out)


if __name__ == "__main__":
    main()
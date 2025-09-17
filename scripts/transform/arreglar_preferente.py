# =============================================================================
# Script: arreglar_preferente.py
# Descripción:
#   Repara el catálogo multiproveedor para asegurar EXACTAMENTE un preferente (prioridad=1) por product_id.
#   Normaliza cabeceras, tipa columnas clave y aplica un desempate determinista:
#       menor prioridad -> menor precio -> menor lead_time -> mayor disponibilidad.
#   Por defecto, APLICA los cambios y SOBRESCRIBE el fichero de entrada salvo que se indique lo contrario.
#
# Uso (CLI):
#   # Sobrescribe el fichero de entrada (comportamiento por defecto):
#   python scripts/transform/arreglar_preferente.py --in data/clean/supplier_catalog_multi.csv
#
#   # Guardar en otro fichero:
#   python scripts/transform/arreglar_preferente.py --in data/clean/supplier_catalog_multi.csv \
#       --out data/clean/supplier_catalog_multi_fixed.csv
#
#   # Dry-run (no escribe nada):
#   python scripts/transform/arreglar_preferente.py --in data/clean/supplier_catalog_multi.csv --no-apply
#
#   # Con copia de seguridad .bak antes de sobrescribir:
#   python scripts/transform/arreglar_preferente.py --in data/clean/supplier_catalog_multi.csv --backup
#
# Entradas:
#   - CSV/Parquet con cabeceras equivalentes a:
#       product_id, supplier_id, precio, lead_time, disponibilidad, prioridad
#
# Salidas:
#   - CSV/Parquet con prioridades normalizadas (en --out o sobrescribiendo --in)
# =============================================================================

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import unicodedata


# ---------- Logging ----------
def setup_logging(verb: int) -> None:
    level = logging.WARNING if verb == 0 else logging.INFO if verb == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------- Normalización de cabeceras ----------
def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    s = s.replace("/", " ").replace("-", " ").replace(".", " ").replace(",", " ")
    s = "_".join(t for t in s.split() if t)
    return s


HEADER_MAP: Dict[str, str] = {
    # product_id
    "product_id": "product_id",
    "productid": "product_id",
    "id_producto": "product_id",
    "producto_id": "product_id",
    "id": "product_id",
    # supplier_id
    "supplier_id": "supplier_id",
    "supplierid": "supplier_id",
    "proveedor_id": "supplier_id",
    "id_proveedor": "supplier_id",
    # precio
    "precio": "precio",
    "price": "precio",
    "precio_unitario": "precio",
    "precio_medio": "precio",  # si no hay 'precio', lo usaremos
    # lead_time
    "lead_time": "lead_time",
    "leadtime": "lead_time",
    "tiempo_entrega": "lead_time",
    # disponibilidad
    "disponibilidad": "disponibilidad",
    "stock_real": "disponibilidad",
    "stock": "disponibilidad",
    # prioridad
    "prioridad": "prioridad",
    "priority": "prioridad",
    "preferente": "prioridad",
}

REQUIRED_COLS = ["product_id", "supplier_id", "prioridad"]
NUMERIC_COLS = ["precio", "lead_time", "disponibilidad", "prioridad"]


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for orig in df.columns:
        slug = _slugify(str(orig))
        rename[orig] = HEADER_MAP.get(slug, slug)
    return df.rename(columns=rename)


def ensure_required_and_types(suppliers: pd.DataFrame) -> pd.DataFrame:
    # Crear faltantes
    for col in REQUIRED_COLS:
        if col not in suppliers.columns:
            suppliers[col] = np.nan

    # Derivar precio si falta
    if "precio" not in suppliers.columns and "precio_medio" in suppliers.columns:
        suppliers["precio"] = suppliers["precio_medio"]

    # Tipado numérico
    for c in NUMERIC_COLS:
        if c in suppliers.columns:
            suppliers[c] = pd.to_numeric(suppliers[c], errors="coerce")

    # Ids como string
    suppliers["product_id"] = suppliers["product_id"].astype(str)
    suppliers["supplier_id"] = suppliers["supplier_id"].astype(str)

    # Si 'prioridad' es todo NaN, arrancamos a 2
    if suppliers["prioridad"].isna().all():
        suppliers["prioridad"] = 2

    return suppliers


# ---------- Reparación por grupo ----------
def _fix_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    if (g["prioridad"] == 1).sum() == 1:
        return g

    by: List[str] = []
    asc: List[bool] = []

    if "prioridad" in g.columns:
        by.append("prioridad");        asc.append(True)   # menor primero (NaN al final)
    if "precio" in g.columns:
        by.append("precio");           asc.append(True)   # menor primero
    if "lead_time" in g.columns:
        by.append("lead_time");        asc.append(True)   # menor primero
    if "disponibilidad" in g.columns:
        by.append("disponibilidad");   asc.append(False)  # mayor primero

    if by:
        g_sorted = g.sort_values(by=by, ascending=asc, na_position="last")
        idx_pref = g_sorted.index[0]
    else:
        idx_pref = g.index[0]

    g["prioridad"] = 2
    g.loc[idx_pref, "prioridad"] = 1
    return g


# ---------- I/O ----------
def read_any(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, dtype=str, low_memory=False)


def write_any(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Normaliza prioridades para asegurar un preferente único por product_id."
    )
    ap.add_argument("--in", dest="inp", default="data/clean/supplier_catalog_multi.csv",
                    help="Ruta de entrada (CSV o Parquet).")
    ap.add_argument("--out", dest="out", default=None,
                    help="Ruta de salida (si se omite y apply=True, se sobrescribe la entrada).")
    # AHORA aplicamos por defecto. Usa --no-apply para dry-run.
    ap.add_argument("--no-apply", dest="apply", action="store_false",
                    help="Dry-run: no escribe salida (por defecto se aplica).")
    ap.add_argument("--backup", action="store_true",
                    help="Si se sobrescribe la entrada, crea copia .bak antes.")
    ap.add_argument("-v", "--verbose", action="count", default=1,
                    help="Verboso (repite para más detalle).")
    ap.add_argument("--show", type=int, default=20,
                    help="Muestra por pantalla hasta N productos corregidos (por defecto 20).")
    args = ap.parse_args()
    setup_logging(args.verbose)

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(f"No existe el fichero de entrada: {inp}")

    logging.info(f"Leyendo entrada: {inp}")
    suppliers = read_any(inp)

    suppliers = normalize_headers(suppliers)
    suppliers = ensure_required_and_types(suppliers)

    num_pref_before = (
        suppliers.groupby("product_id")["prioridad"]
        .apply(lambda s: (pd.to_numeric(s, errors="coerce") == 1).sum())
    )
    bad_before = num_pref_before[num_pref_before != 1]
    logging.info(f"Productos con preferente inválido ANTES: {int((bad_before != 1).sum())} (total grupos={len(num_pref_before)})")

    logging.info("Arreglando prioridades por producto...")
    suppliers_fixed = suppliers.groupby("product_id", group_keys=False).apply(_fix_group)

    num_pref_after = (
        suppliers_fixed.groupby("product_id")["prioridad"]
        .apply(lambda s: (pd.to_numeric(s, errors="coerce") == 1).sum())
    )
    bad_after = num_pref_after[num_pref_after != 1]
    ok = bad_after.empty
    logging.info(f"Productos con preferente inválido DESPUÉS: {0 if ok else int((bad_after != 1).sum())}")

    changed_ids = sorted(set(bad_before.index.tolist()))
    if changed_ids:
        logging.info(f"Muestra de productos corregidos (hasta {args.show}):")
        muestra = suppliers_fixed[suppliers_fixed["product_id"].isin(changed_ids)]
        cols_show = [c for c in ["product_id", "supplier_id", "precio", "lead_time", "disponibilidad", "prioridad"] if c in muestra.columns]
        with pd.option_context("display.max_rows", args.show, "display.max_colwidth", 80):
            print(
                muestra[cols_show]
                .sort_values(["product_id", "prioridad", "precio", "lead_time"], ascending=[True, True, True, True])
                .head(args.show)
            )

    total_products = suppliers_fixed["product_id"].nunique()
    logging.info(f"Total de productos: {total_products}")

    # Escritura: por defecto APPLY=True → sobreescribir entrada si no hay --out
    if args.apply:
        out_path = Path(args.out) if args.out else inp
        if args.out is None and args.backup:
            bak = out_path.with_suffix(out_path.suffix + ".bak")
            shutil.copy2(out_path, bak)
            logging.info(f"Backup creado: {bak}")
        logging.info(f"Escribiendo salida en: {out_path}")
        write_any(suppliers_fixed, out_path)
        logging.info("Escritura completada.")
    else:
        logging.info("Dry-run (por --no-apply): no se ha escrito ningún fichero.")


if __name__ == "__main__":
    main()
# scripts: movimientos_stock.py
# ======================================================
# Procesador de movimientos de inventario
#
# Descripción:
#   Consume pedidos de clientes y actualiza el inventario,
#   dejando trazabilidad (ledger), etiquetas (rotura/bajo stock),
#   sugerencias de compra y órdenes por proveedor.
#   Incluye integración opcional de sustitutos.
#
# Flujo:
#   1. Carga inventario y pedidos
#   2. Aplica consumos → ledger
#   3. Recalcula ROP, flags y sugerencias
#   4. Genera órdenes y alertas
#   5. Exporta salidas a processed/fase10_stock
#
# Inputs esperados:
#   - data/clean/Inventario.csv
#   - data/customer_orders_AE.csv
#   - data/supplier_catalog_demo.csv
#   - data/processed/substitutes_unified.csv
#
# Outputs:
#   - data/processed/fase10_stock/inventory_updated.csv
#   - data/processed/fase10_stock/ledger_movimientos.csv
#   - data/processed/fase10_stock/alerts.csv
#   - data/processed/fase10_stock/sugerencias_compra.csv
#   - data/processed/fase10_stock/ordenes_compra.csv
#   - data/processed/fase10_stock/ordenes_compra_lineas.csv
#   - data/processed/fase10_stock/sugerencias_sustitutos.csv
#
# Dependencias:
#   pip install pandas numpy
# ======================================================
# scripts/operativa/movimientos_stock.py
# ======================================================
# Procesador de movimientos de inventario (robusto)
# ======================================================
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# ------------------------- Rutas
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA       = PROJECT_ROOT / "data"
CLEAN      = DATA / "clean"
RAW        = DATA / "raw"
PROCESSED  = DATA / "processed"
OUTDIR     = PROCESSED / "fase10_stock"
OUTDIR.mkdir(parents=True, exist_ok=True)

INV_FILE         = CLEAN / "Inventario.csv"
ORDERS_FILE      = RAW   / "customer_orders_AE.csv"
SUPPLIER_FILE    = RAW   / "supplier_catalog_demo.csv"
SUBSTITUTES_FILE = PROCESSED / "substitutes_unified.csv"

# ------------------------- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("movimientos_stock")

# ------------------------- Utils
def ceil_to_multiple(x: int, mult: int) -> int:
    if pd.isna(mult) or mult in (0, 1) or x <= 0:
        return int(max(0, x))
    return int(np.ceil(x / mult) * mult)

def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Lector robusto:
      1) intenta tal cual;
      2) autodetecta separador (sep=None, engine='python', encoding utf-8);
      3) prueba latin-1 si fallase lo anterior.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero: {path}")
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=kwargs.get("encoding", "utf-8"))
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")

# ------------------------- Lecturas normalizadas
def read_inventory_csv(path: Path) -> pd.DataFrame:
    log.info("Cargando inventario: %s", path)
    df = _safe_read_csv(path, sep=";", decimal=",")
    rename_map = {
        "Product_ID": "item_id", "product_id": "item_id", "Item_ID": "item_id",
        "Stock Real": "stock_actual", "Stock": "stock_actual", "stock": "stock_actual",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    if "item_id" not in df.columns:
        raise ValueError(f"Inventario sin columna item_id. Columnas: {list(df.columns)}")
    if "stock_actual" not in df.columns:
        raise ValueError(f"Inventario sin columna stock_actual. Columnas: {list(df.columns)}")

    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64").dropna().astype(int)
    df["stock_actual"] = pd.to_numeric(df["stock_actual"], errors="coerce").fillna(0).astype(int)

    # opcionales
    for col in ["supplier_name", "item_name", "category"]:
        if col not in df.columns:
            df[col] = np.nan
    if "ROP" not in df.columns:
        df["ROP"] = 0.0
    return df

def read_orders_csv(orders_path: Path) -> pd.DataFrame:
    """
    Devuelve ['date','Product_ID','qty'] aceptando:
      - 'date'|'fecha' en formatos ISO (con o sin hora, con 'T' o espacio),
      - 'product_id' o 'item_id' -> 'Product_ID',
      - 'qty' o 'quantity' -> 'qty'.
    Conversión de fechas **sin** formato fijo (evita el “unconverted data remains”).
    """
    if not orders_path or not Path(orders_path).exists():
        return pd.DataFrame(columns=["date", "Product_ID", "qty"])

    df = _safe_read_csv(orders_path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "Product_ID", "qty"])

    low = {c.lower(): c for c in df.columns}
    date_col = low.get("date") or low.get("fecha")
    pid_col  = low.get("product_id") or low.get("item_id")
    qty_col  = low.get("qty") or low.get("quantity")
    if not (date_col and pid_col and qty_col):
        return pd.DataFrame(columns=["date", "Product_ID", "qty"])

    df = df.rename(columns={date_col: "date", pid_col: "Product_ID", qty_col: "qty"})

    # --- fechas robustas ---
    d = df["date"].astype(str).str.strip().str.replace("T", " ", regex=False)
    # “2025-09-18 00:00:00” -> “2025-09-18”
    d = d.str.split().str[0].str.slice(0, 10)
    df["date"] = pd.to_datetime(d, errors="coerce")        # <- sin 'format'
    df["Product_ID"] = pd.to_numeric(df["Product_ID"], errors="coerce").astype("Int64")
    df["qty"]        = pd.to_numeric(df["qty"],        errors="coerce").astype("Int64")

    df = df.dropna(subset=["date", "Product_ID", "qty"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", "Product_ID", "qty"]]

def read_catalog_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.info("Catálogo no encontrado: %s (se continúa sin catálogo).", path)
        return pd.DataFrame()
    log.info("Cargando catálogo proveedores: %s", path)
    df = _safe_read_csv(path)
    rename_map = {"Product_ID": "Product_ID", "product_id": "item_id",
                  "supplier": "supplier_name", "proveedor": "supplier_name"}
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    if "Product_ID" in df.columns:
        df["Product_ID"] = pd.to_numeric(df["Product_ID"], errors="coerce").astype("Int64").dropna().astype(int)
    for col in ["moq", "multiplo", "precio", "lead_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "multiplo" in df.columns:
        df["multiplo"] = df["multiplo"].fillna(1).replace(0, 1).astype(int)
    return df

def read_substitutes_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.info("Sustitutos no encontrados: %s (se continúa sin sustitutos).", path)
        return pd.DataFrame()
    log.info("Cargando sustitutos: %s", path)
    df = _safe_read_csv(path)
    rename_map = {"Product_ID": "item_id", "product_id": "item_id", "item id": "item_id", "id_item": "item_id"}
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    if "item_id" not in df.columns:
        raise ValueError(f"No se encontró 'item_id' en {path}. Columnas: {list(df.columns)}")
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64").dropna().astype(int)
    if "score" in df.columns:
        df = df.sort_values(["item_id", "score"], ascending=[True, False])
    return df

# ------------------------- Proceso principal
def run(
    inventario: Path = INV_FILE,
    orders_path: Path = ORDERS_FILE,
    supplier_catalog: Path = SUPPLIER_FILE,
    substitutes: Path = SUBSTITUTES_FILE,
    outdir: Path = OUTDIR,
):
    log.info("== INICIO MOVIMIENTOS ==")
    outdir.mkdir(parents=True, exist_ok=True)

    inv  = read_inventory_csv(inventario)
    ords = read_orders_csv(orders_path)   # <- fechas robustas aquí
    cat  = read_catalog_csv(supplier_catalog)
    subs = read_substitutes_csv(substitutes)

    ledger_rows, alerts_rows = [], []

    # lookup proveedor
    cat_lookup = None
    if not cat.empty and ("item_id" in cat.columns or "Product_ID" in cat.columns):
        base = cat.copy()
        if "Product_ID" in base.columns and "item_id" not in base.columns:
            base = base.rename(columns={"Product_ID": "item_id"})
        cols = ["item_id"] + [c for c in ["supplier_name", "precio", "moq", "multiplo", "lead_time"] if c in base.columns]
        cat_lookup = base[cols].copy()

    items_set = set(inv["item_id"])
    for _, r in ords.iterrows():
        item = int(r["Product_ID"]); qty = int(r["qty"]); ts = r["date"]

        if item not in items_set:
            alerts_rows.append({"timestamp": datetime.now(), "severidad": "WARN",
                                "mensaje": f"Pedido recibido para item desconocido {item}",
                                "item_id": item, "supplier_id": None})
            continue

        idx = inv.index[inv["item_id"] == item][0]
        stock_prev = int(inv.at[idx, "stock_actual"])
        stock_new  = stock_prev - qty
        inv.at[idx, "stock_actual"] = stock_new

        ledger_rows.append({
            "timestamp": ts, "item_id": item, "tipo": "venta",
            "qty": -qty, "stock_resultante": stock_new, "info": np.nan
        })

        if stock_new <= 0:
            sup = None
            if cat_lookup is not None:
                row_sup = cat_lookup.loc[cat_lookup["item_id"] == item]
                if not row_sup.empty and "supplier_name" in row_sup.columns:
                    sup = row_sup.iloc[0]["supplier_name"]
            alerts_rows.append({
                "timestamp": datetime.now(), "severidad": "CRIT",
                "mensaje": f"Rotura detectada tras pedido (item {item})",
                "item_id": item, "supplier_id": sup
            })

    inv["flag_rotura"] = inv["stock_actual"] <= 0
    inv["flag_bajo"]   = inv["stock_actual"] < inv.get("ROP", 0)

    cands = inv[(inv["flag_rotura"]) | (inv["flag_bajo"])].copy()
    cands["qty_sugerida"] = (-cands["stock_actual"]).clip(lower=0).astype(int)

    if cat_lookup is not None:
        cands = cands.merge(cat_lookup, on="item_id", how="left")
        if "moq" not in cands.columns: cands["moq"] = 0
        if "multiplo" not in cands.columns: cands["multiplo"] = 1
        cands["qty_sugerida"] = cands[["qty_sugerida", "moq"]].max(axis=1).astype(int)
        cands["qty_sugerida"] = cands.apply(lambda x: ceil_to_multiple(x["qty_sugerida"], x["multiplo"]), axis=1)
        if "precio" in cands.columns:
            cands["importe"] = (cands["qty_sugerida"] * cands["precio"]).round(2)
        else:
            cands["importe"] = np.nan
    else:
        cands["supplier_name"] = np.nan
        cands["moq"] = 0
        cands["multiplo"] = 1
        cands["importe"] = np.nan

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "supplier_name" not in cands.columns:
        cands["supplier_name"] = "SIN_PROVEEDOR"

    cands["motivo"] = np.where(cands["flag_rotura"], "rotura", "bajo_stock")
    cands["order_id"] = "PO_" + cands["supplier_name"].fillna("NA").astype(str).str.replace(r"\W+", "", regex=True) + "_" + ts_str

    oc_lines_cols = ["order_id", "item_id", "motivo", "qty_sugerida", "precio", "importe", "supplier_name"]
    oc_lines = cands[[c for c in oc_lines_cols if c in cands.columns]].copy()
    oc_head = (oc_lines.groupby(["order_id","supplier_name"], dropna=False)
                      .agg(num_lineas=("item_id","count"),
                           importe_total=("importe","sum"))
                      .reset_index())

    subs_out = pd.DataFrame()
    if not subs.empty:
        rot_ids = set(inv.loc[inv["flag_rotura"], "item_id"])
        sub_filt = subs[subs["item_id"].isin(rot_ids)].copy()
        if "score" in sub_filt.columns:
            sub_filt = sub_filt.sort_values(["item_id","score"], ascending=[True, False])
        subs_out = sub_filt.groupby("item_id").head(3)
        sin_subs = rot_ids - set(subs_out["item_id"])
        for iid in list(sin_subs)[:10000]:
            alerts_rows.append({
                "timestamp": datetime.now(), "severidad": "CRIT",
                "mensaje": f"Sin sustitutos disponibles para item {iid}",
                "item_id": iid, "supplier_id": np.nan
            })

    df_ledger = pd.DataFrame(ledger_rows)
    df_alerts = pd.DataFrame(alerts_rows)

    inv.to_csv(OUTDIR / "inventory_updated.csv", index=False)
    df_ledger.to_csv(OUTDIR / "ledger_movimientos.csv", index=False)
    cands.rename(columns={"precio":"precio_unit"}, inplace=True)
    cands[["item_id","qty_sugerida","moq","multiplo","importe","supplier_name","motivo"]].to_csv(OUTDIR / "sugerencias_compra.csv", index=False)
    oc_head.to_csv(OUTDIR / "ordenes_compra.csv", index=False)
    oc_lines.to_csv(OUTDIR / "ordenes_compra_lineas.csv", index=False)
    subs_out.to_csv(OUTDIR / "sugerencias_sustitutos.csv", index=False)
    df_alerts.to_csv(OUTDIR / "alerts.csv", index=False)

    log.info("== FIN MOVIMIENTOS ==")
    log.info("Exportado en: %s", OUTDIR)

if __name__ == "__main__":
    run(
        inventario=INV_FILE,
        orders_path=ORDERS_FILE,
        supplier_catalog=SUPPLIER_FILE,
        substitutes=SUBSTITUTES_FILE,
        outdir=OUTDIR,
    )

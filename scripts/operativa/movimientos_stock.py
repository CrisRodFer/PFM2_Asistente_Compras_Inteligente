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
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# -------------------------
# 0) RUTAS DEL PROYECTO
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../PFM2_Asistente_Compras_Inteligente
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

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("movimientos_stock")

# -------------------------
# Utils
# -------------------------
def ceil_to_multiple(x: int, mult: int) -> int:
    if pd.isna(mult) or mult in (0, 1) or x <= 0:
        return int(max(0, x))
    return int(np.ceil(x / mult) * mult)

def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero: {path}")
    return pd.read_csv(path, **kwargs)

# -------------------------
# Lecturas normalizadas
# -------------------------
def read_inventory_csv(path: Path) -> pd.DataFrame:
    """Lee Inventario.csv (separador ';') y normaliza a: item_id, stock_actual, supplier_name, item_name, category."""
    log.info("Cargando inventario: %s", path)
    df = _safe_read_csv(path, sep=";", decimal=",")
    rename_map = {
        "Product_ID": "item_id",
        "product_id": "item_id",
        "Item_ID": "item_id",
        "Stock Real": "stock_actual",
        "stock": "stock_actual",
        "Stock": "stock_actual",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    if "item_id" not in df.columns:
        raise ValueError(f"Inventario sin columna item_id. Columnas: {list(df.columns)}")
    if "stock_actual" not in df.columns:
        raise ValueError(f"Inventario sin columna stock_actual. Columnas: {list(df.columns)}")

    df["item_id"] = df["item_id"].astype(int)
    df["stock_actual"] = pd.to_numeric(df["stock_actual"], errors="coerce").fillna(0).astype(int)

    # Asegura columnas opcionales
    for col in ["supplier_name", "item_name", "category"]:
        if col not in df.columns:
            df[col] = np.nan

    # ROP placeholder si no existe (se calcula en futuras fases)
    if "ROP" not in df.columns:
        df["ROP"] = 0.0

    return df

def read_orders_csv(path: Path) -> pd.DataFrame:
    """Lee pedidos A–E con columnas: order_id, date, item_id, qty."""
    log.info("Cargando pedidos: %s", path)
    df = _safe_read_csv(path)
    rename_map = {
        "item id": "item_id",
        "Product_ID": "item_id",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    required = {"date", "item_id", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pedidos sin columnas {missing}. Columnas: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df["item_id"] = df["item_id"].astype(int)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("date")

def read_catalog_csv(path: Path) -> pd.DataFrame:
    """Catálogo proveedor (si existe). Normaliza item_id y columnas moq/multiplo/precio/lead_time si están."""
    if not path.exists():
        log.info("Catálogo no encontrado: %s (se continúa sin catálogo).", path)
        return pd.DataFrame()
    log.info("Cargando catálogo proveedores: %s", path)
    df = pd.read_csv(path)
    rename_map = {
        "Product_ID": "item_id",
        "product_id": "item_id",
        "supplier": "supplier_name",
        "proveedor": "supplier_name",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    if "item_id" in df.columns:
        df["item_id"] = df["item_id"].astype(int)
    for col in ["moq", "multiplo", "precio", "lead_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "multiplo" in df.columns:
        df["multiplo"] = df["multiplo"].fillna(1).replace(0, 1).astype(int)
    return df

def read_substitutes_csv(path: Path) -> pd.DataFrame:
    """Mapa de sustitutos (si existe). Normaliza item_id y ordena por score desc."""
    if not path.exists():
        log.info("Sustitutos no encontrados: %s (se continúa sin sustitutos).", path)
        return pd.DataFrame()
    log.info("Cargando sustitutos: %s", path)
    df = pd.read_csv(path)
    rename_map = {
        "Product_ID": "item_id",
        "product_id": "item_id",
        "item id": "item_id",
        "id_item": "item_id",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    if "item_id" not in df.columns:
        raise ValueError(f"No se encontró 'item_id' en {path}. Columnas: {list(df.columns)}")
    df["item_id"] = df["item_id"].astype(int)
    if "score" in df.columns:
        df = df.sort_values(["item_id", "score"], ascending=[True, False])
    return df

# -------------------------
# PROCESO PRINCIPAL
# -------------------------
def run(
    inventario: Path = INV_FILE,
    orders_path: Path = ORDERS_FILE,
    supplier_catalog: Path = SUPPLIER_FILE,
    substitutes: Path = SUBSTITUTES_FILE,
    outdir: Path = OUTDIR,
):
    log.info("== INICIO MOVIMIENTOS ==")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Cargar
    inv  = read_inventory_csv(inventario)
    ords = read_orders_csv(orders_path)
    cat  = read_catalog_csv(supplier_catalog)
    subs = read_substitutes_csv(substitutes)

    # 2) Ledger y alertas
    ledger_rows = []
    alerts_rows = []

    # Prepara lookups de proveedor (si hay catálogo)
    cat_lookup = None
    if not cat.empty and "item_id" in cat.columns:
        cols = ["item_id"]
        for c in ["supplier_name", "precio", "moq", "multiplo", "lead_time"]:
            if c in cat.columns: cols.append(c)
        cat_lookup = cat[cols].copy()

    for _, r in ords.iterrows():
        item = int(r["item_id"]); qty = int(r["qty"]); ts = r["date"]

        if item not in set(inv["item_id"]):
            alerts_rows.append({"timestamp": datetime.now(), "severidad": "WARN",
                                "mensaje": f"Pedido recibido para item desconocido {item}",
                                "item_id": item, "supplier_id": None})
            continue

        idx = inv.index[inv["item_id"] == item][0]
        stock_prev = int(inv.at[idx, "stock_actual"])
        stock_new  = stock_prev - qty
        inv.at[idx, "stock_actual"] = stock_new

        ledger_rows.append({
            "timestamp": ts,
            "item_id": item,
            "tipo": "venta",
            "qty": -qty,
            "stock_resultante": stock_new,
            "info": np.nan
        })

        if stock_new <= 0:
            sup = None
            if cat_lookup is not None:
                row_sup = cat_lookup[cat_lookup["item_id"] == item]
                if not row_sup.empty and "supplier_name" in row_sup.columns:
                    sup = row_sup.iloc[0]["supplier_name"]
            alerts_rows.append({
                "timestamp": datetime.now(),
                "severidad": "CRIT",
                "mensaje": f"Rotura detectada tras pedido (item {item})",
                "item_id": item,
                "supplier_id": sup
            })

    # 3) Flags post-venta
    inv["flag_rotura"] = inv["stock_actual"] <= 0
    inv["flag_bajo"]   = inv["stock_actual"] < inv.get("ROP", 0)

    # 4) Sugerencias de compra
    cands = inv[(inv["flag_rotura"]) | (inv["flag_bajo"])].copy()
    # qty sugerida básica = lo que falta hasta 0 (sin cobertura aún)
    cands["qty_sugerida"] = (-cands["stock_actual"]).clip(lower=0).astype(int)

    # Enriquecer con catálogo si existe
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

    # 5) Órdenes por proveedor (1 OC por proveedor)
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

    # 6) Sustitutos (top-3 por SKU en rotura)
    subs_out = pd.DataFrame()
    if not subs.empty:
        rot_ids = set(inv.loc[inv["flag_rotura"], "item_id"])
        sub_filt = subs[subs["item_id"].isin(rot_ids)].copy()
        if "score" in sub_filt.columns:
            sub_filt = sub_filt.sort_values(["item_id","score"], ascending=[True, False])
        subs_out = sub_filt.groupby("item_id").head(3)
        # CRIT si rotura sin sustitutos
        sin_subs = rot_ids - set(subs_out["item_id"])
        for iid in list(sin_subs)[:10000]:
            alerts_rows.append({
                "timestamp": datetime.now(),
                "severidad": "CRIT",
                "mensaje": f"Sin sustitutos disponibles para item {iid}",
                "item_id": iid,
                "supplier_id": np.nan
            })

    # 7) Exportar
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

# -------------------------
# EJECUCIÓN DIRECTA
# -------------------------
if __name__ == "__main__":
    run(
        inventario=INV_FILE,
        orders_path=ORDERS_FILE,
        supplier_catalog=SUPPLIER_FILE,
        substitutes=SUBSTITUTES_FILE,
        outdir=OUTDIR,
    )

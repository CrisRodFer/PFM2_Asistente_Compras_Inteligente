# =====================================================================
# Script: movimientos_stock.py
# Descripción:
#   Procesa pedidos de cliente y ajusta el inventario. Deja trazabilidad
#   (ledger), calcula DMD/ROP, etiqueta roturas/bajo stock, construye
#   sugerencias de compra, agrupa OC por proveedor y GENERA sugerencias
#   de SUSTITUTOS (si se aporta substitutes.csv).
#
#   - No genera pedidos por defecto; usa --autogen-demo sólo para pruebas.
#   - Funciona en consola (argparse) y en notebook (llamar run(...)).
#
# Entradas (mínimas):
#   --inventario  CSV ';' con: Product_ID;Proveedor;Nombre;Categoria;Stock Real
#   --orders      CSV con: date,item_id,qty
# Opcionales:
#   --supplier-catalog  CSV: item_id,supplier_id|supplier_name,precio,lead_time,moq,multiplo
#   --service-policy    CSV: category,objetivo_cobertura_dias,stock_seguridad,lead_time_policy
#   --substitutes       CSV: Product_ID,tipo,Substitute_Product_ID,Substitute_Supplier_ID,
#                              score,precio,disponibilidad,lead_time,prioridad,lead_time_bucket
#
# Salidas:
#   inventory_updated.csv | ledger_movimientos.csv | alerts.csv
#   sugerencias_compra.csv
#   ordenes_compra.csv | ordenes_compra_lineas.csv
#   sugerencias_sustitutos.csv   <-- NUEVO (si hay sustitutos)
#
# Dependencias: pandas, numpy, python-dateutil
# =====================================================================

from pathlib import Path
import argparse
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# ----------------------------
# CONFIG (valores por defecto)
# ----------------------------
DEFAULT_OBJECTIVE_COVERAGE = 14   # días si no hay policy
DEFAULT_LEAD_TIME = 5             # días si no hay supplier/policy
DEFAULT_SAFETY_STOCK = 0          # uds si no hay policy
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fase10.movimientos")

# ----------------------------
# UTILIDADES
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_inventory_csv(path: Path) -> pd.DataFrame:
    """
    Lee Inventario.csv (separador ';') y normaliza:
      Product_ID;Proveedor;Nombre;Categoria;Stock Real
    -> item_id, supplier_name, item_name, category, stock_actual
    """
    df = pd.read_csv(path, sep=';')
    rename = {
        "Product_ID": "item_id",
        "Proveedor": "supplier_name",
        "Nombre": "item_name",
        "Categoria": "category",
        "Stock Real": "stock_actual",
    }
    df = df.rename(columns=rename)
    needed = {"item_id", "supplier_name", "item_name", "category", "stock_actual"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Inventario sin columnas esperadas: {miss}")
    df["item_id"] = df["item_id"].astype(int)
    df["stock_actual"] = df["stock_actual"].astype(int)
    return df

def read_orders_csv(path: Path) -> pd.DataFrame:
    """Lee pedidos: date,item_id,qty -> ordenados por fecha."""
    df = pd.read_csv(path)
    need = {"date", "item_id", "qty"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Orders CSV missing columns: {miss}")
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="raise")
    df["item_id"] = df["item_id"].astype(int)
    df["qty"] = df["qty"].astype(int)
    if (df["qty"] <= 0).any():
        raise ValueError("Orders contiene qty <= 0")
    return df.sort_values("date").reset_index(drop=True)

def gen_demo_orders(inv: pd.DataFrame, n_lines=10) -> pd.DataFrame:
    """Genera pedidos DEMO (sólo si --autogen-demo)."""
    rng = np.random.default_rng(SEED)
    pick = inv.sample(min(n_lines, len(inv)), random_state=SEED).copy()
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
    dates = pd.date_range(start, periods=len(pick), freq="H")
    pick["date"] = dates.values
    max_by_stock = (pick["stock_actual"] * 0.15).clip(lower=1).round().astype(int)
    pick["qty"] = np.minimum(rng.integers(1, 9, size=len(pick)), max_by_stock)
    return pick[["date", "item_id", "qty"]].reset_index(drop=True)

def ceil_to_multiple(x: pd.Series, multiple: pd.Series) -> pd.Series:
    """Redondea hacia arriba x al múltiplo indicado (usa 1 si NaN/0)."""
    mult = multiple.fillna(1).replace(0, 1).astype(int)
    return (np.ceil(x / mult) * mult).astype(int)

# ----------------------------
# PROCESADOR PRINCIPAL
# ----------------------------
def run(
    inventario: Path,
    orders_path: Path | None,
    supplier_catalog: Path | None,
    service_policy: Path | None,
    substitutes: Path | None,     # NUEVO: fichero de sustitutos
    outdir: Path,
    autogen_demo: bool = False,
):
    ensure_dir(outdir)

    # 1) Carga inventario y pedidos
    log.info("Cargando inventario: %s", inventario)
    inv = read_inventory_csv(inventario)

    if orders_path is None:
        if autogen_demo:
            log.warning("No se pasó --orders. Generando pedidos DEMO (modo laboratorio).")
            orders = gen_demo_orders(inv)
        else:
            raise SystemExit("Falta --orders. Este script NO genera pedidos por defecto.")
    else:
        log.info("Cargando pedidos: %s", orders_path)
        orders = read_orders_csv(orders_path)

    # 2) Enriquecimiento opcional (supplier/policy)
    df = inv.copy()

    if supplier_catalog and Path(supplier_catalog).exists():
        sup = pd.read_csv(supplier_catalog)
        sup = sup.rename(columns={
            "Product_ID": "item_id",
            "supplier": "supplier_id",
            "supplier_id": "supplier_id",
            "Supplier_ID": "supplier_id",
            "precio": "precio",
            "price": "precio",
            "lead_time": "lead_time",
            "moq": "moq",
            "multiplo": "multiplo",
            "multiple": "multiplo",
        })
        if "item_id" not in sup.columns:
            raise ValueError("supplier_catalog debe contener columna item_id")
        sup["item_id"] = sup["item_id"].astype(int)
        keep = [c for c in ["item_id","supplier_id","precio","lead_time","moq","multiplo"] if c in sup.columns]
        df = df.merge(sup[keep], on="item_id", how="left")
    else:
        df["supplier_id"] = pd.NA
        df["precio"] = np.nan
        df["lead_time"] = np.nan
        df["moq"] = np.nan
        df["multiplo"] = np.nan

    if service_policy and Path(service_policy).exists():
        pol = pd.read_csv(service_policy)
        pol = pol.rename(columns={
            "Categoria": "category",
            "cluster_id": "category",
            "objetivo_cobertura_dias": "objetivo_cobertura_dias",
            "stock_seguridad": "stock_seguridad",
            "lead_time_policy": "lead_time_policy",
        })
        if "category" in pol.columns:
            df = df.merge(pol, on="category", how="left")
        else:
            log.warning("service_policy sin 'category': se ignora.")

    # Fallbacks de política
    df["objetivo_cobertura_dias"] = df.get("objetivo_cobertura_dias", pd.Series([np.nan]*len(df))).fillna(DEFAULT_OBJECTIVE_COVERAGE)
    df["stock_seguridad"] = df.get("stock_seguridad", pd.Series([np.nan]*len(df))).fillna(DEFAULT_SAFETY_STOCK)
    df["lead_time_policy"] = df.get("lead_time_policy", pd.Series([np.nan]*len(df))).fillna(DEFAULT_LEAD_TIME)
    df["lead_time_eff"] = df["lead_time"].fillna(df["lead_time_policy"]).fillna(DEFAULT_LEAD_TIME)

    # 3) DMD y ROP
    if not orders.empty:
        horizon_days = max(1, (orders["date"].max() - orders["date"].min()).days + 1)
        dem = orders.groupby("item_id")["qty"].sum().rename("consumo_total").to_frame()
        dem["demanda_media_diaria"] = dem["consumo_total"] / horizon_days
        df = df.merge(dem["demanda_media_diaria"], on="item_id", how="left")
    df["demanda_media_diaria"] = df["demanda_media_diaria"].fillna(0.0)
    df["ROP"] = (df["demanda_media_diaria"] * df["lead_time_eff"]) + df["stock_seguridad"]

    # 4) Ledger y alertas
    ledger_rows: list[dict] = []
    alerts: list[dict] = []

    def log_move(ts, item, tipo, qty, stock_after, info=""):
        ledger_rows.append({
            "timestamp": ts, "item_id": int(item), "tipo": tipo,
            "qty": int(qty), "stock_resultante": int(stock_after), "info": info
        })

    def alert(sev, msg, item_id=None, supplier_id=None):
        alerts.append({
            "timestamp": datetime.now(), "severidad": sev, "mensaje": msg,
            "item_id": item_id, "supplier_id": supplier_id
        })

    # 5) Aplicar pedidos (secuencialmente por fecha)
    items_set = set(df["item_id"])
    for _, row in orders.sort_values("date").iterrows():
        item = int(row["item_id"]); qty = int(row["qty"]); ts = row["date"]
        if item not in items_set:
            alert("WARN", f"Pedido con item_id {item} no presente en inventario", item_id=item)
            continue
        i = df.index[df["item_id"] == item][0]
        prev = int(df.at[i, "stock_actual"])
        new = prev - qty
        df.at[i, "stock_actual"] = new
        # cobertura_dias (informativa)
        dmd = float(df.at[i, "demanda_media_diaria"])
        df.at[i, "cobertura_dias"] = (new / dmd) if dmd > 0 else np.inf
        log_move(ts, item, "venta", -qty, new)
        if new <= 0:
            alert("CRIT", f"Rotura detectada tras pedido (item {item})", item_id=item,
                  supplier_id=(df.at[i, "supplier_id"] if "supplier_id" in df.columns else None))

    # 6) Etiquetas post-venta
    df["flag_rotura"] = df["stock_actual"] <= 0
    df["flag_bajo"] = df["stock_actual"] < df["ROP"]

    # 7) Sugerencias de compra
    df["stock_obj"] = df["demanda_media_diaria"] * df["objetivo_cobertura_dias"]
    cands = df[(df["flag_rotura"] | df["flag_bajo"])].copy()
    cands["qty_sugerida"] = (cands["stock_obj"] - cands["stock_actual"]).clip(lower=0).round().astype(int)

    if "moq" in cands.columns:
        cands["moq"] = cands["moq"].fillna(0).astype(int)
        cands["qty_sugerida"] = np.maximum(cands["qty_sugerida"], cands["moq"])
    if "multiplo" in cands.columns:
        cands["multiplo"] = cands["multiplo"].fillna(1).astype(int).replace(0, 1)
        cands["qty_sugerida"] = ceil_to_multiple(cands["qty_sugerida"], cands["multiplo"])
    if "precio" in cands.columns:
        cands["precio"] = cands["precio"].fillna(0.0)
        cands["importe"] = (cands["qty_sugerida"] * cands["precio"]).round(2)

    # 7.bis) Sugerencias de SUSTITUTOS (solo para roturas)
    sug_subs = pd.DataFrame()
    rot_ids = set(cands.loc[cands["flag_rotura"], "item_id"])
    if substitutes and Path(substitutes).exists() and rot_ids:
        subs_df = pd.read_csv(substitutes)
        subs_df = subs_df.rename(columns={
            "Product_ID": "item_id",
            "Substitute_Product_ID": "item_id_sub",
            "Substitute_Supplier_ID": "supplier_id_sub",
            "precio": "precio_sub",
            "lead_time": "lead_time_sub"
        })
        s = subs_df[subs_df["item_id"].isin(rot_ids)].copy()

        ext_mask = s["item_id_sub"].notna()
        s_ext = s[ext_mask].copy()
        if not s_ext.empty:
            inv_min = df[["item_id", "stock_actual", "category", "supplier_name"]].rename(
                columns={"item_id": "item_id_sub",
                         "stock_actual": "stock_sub",
                         "supplier_name": "supplier_name_sub"}
            )
            s_ext = s_ext.merge(inv_min, on="item_id_sub", how="left")

        int_mask = ~ext_mask
        s_int = s[int_mask].copy()
        if not s_int.empty:
            s_int["item_id_sub"] = s_int["item_id"]       # mismo SKU
            s_int["stock_sub"] = np.nan
            s_int["supplier_name_sub"] = s_int.get("supplier_id_sub", np.nan)

        sug_subs = pd.concat([s_ext, s_int], ignore_index=True)

        if not sug_subs.empty:
            has_stock = sug_subs["stock_sub"].fillna(0) > 0
            has_disp  = sug_subs["disponibilidad"].fillna(0) > 0
            keep_mask = has_stock | has_disp
            sug_subs = sug_subs[keep_mask].copy()

            # columnas que pueden no existir
            for col in ["score","disponibilidad","lead_time_sub","precio_sub","prioridad","lead_time_bucket"]:
                if col not in sug_subs.columns: sug_subs[col] = np.nan

            # ranking y Top-3 por item
            sug_subs = (sug_subs
                        .sort_values(["item_id","score","disponibilidad"], ascending=[True, False, False])
                        .groupby("item_id", as_index=False)
                        .head(3))

            out_cols = [
                "item_id","tipo","item_id_sub","supplier_id_sub","supplier_name_sub",
                "score","disponibilidad","lead_time_sub","precio_sub",
                "lead_time_bucket","prioridad","stock_sub"
            ]
            for c in out_cols:
                if c not in sug_subs.columns: sug_subs[c] = np.nan
            sug_subs = sug_subs[out_cols]

            # Alertas por SKU
            found_ids = set(sug_subs["item_id"].unique())
            for it in rot_ids:
                if it in found_ids:
                    n = (sug_subs["item_id"] == it).sum()
                    alert("WARN", f"Sustitutos sugeridos ({n}) para item {it}", item_id=it)
                else:
                    alert("CRIT", f"Sin sustitutos disponibles para item {it}", item_id=it)
        else:
            for it in rot_ids:
                alert("CRIT", f"Sin sustitutos definidos para item {it}", item_id=it)
    else:
        # No hay fichero o no hay roturas
        if rot_ids and not (substitutes and Path(substitutes).exists()):
            for it in rot_ids:
                alert("CRIT", f"Sin sustitutos (no se proporcionó fichero) para item {it}", item_id=it)

    # 8) Órdenes por proveedor
    prov_key = "supplier_name" if "supplier_name" in cands.columns else ("supplier_id" if "supplier_id" in cands.columns else None)
    oc_header, oc_lines = pd.DataFrame(), pd.DataFrame()
    if prov_key and not cands.empty:
        oc_lines = cands.copy()
        oc_lines["motivo"] = np.where(oc_lines["flag_rotura"], "rotura", "bajo_stock")
        base_cols = ["item_id","item_name","category",prov_key,"qty_sugerida","motivo","stock_actual","ROP"]
        price_cols = base_cols + ["precio","importe"] if "importe" in oc_lines.columns else base_cols
        oc_lines = oc_lines[price_cols]
        rows = []
        for prov, g in oc_lines.groupby(prov_key, dropna=False):
            order_id = f"PO-{str(prov)[:8]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            rows.append({
                "order_id": order_id,
                "proveedor": prov,
                "fecha": datetime.now().date(),
                "num_lineas": len(g),
                "importe_total": round(g["importe"].sum(),2) if "importe" in g.columns else np.nan
            })
            oc_lines.loc[g.index, "order_id"] = order_id
        oc_header = pd.DataFrame(rows)
        for _, r in oc_header.iterrows():
            alert("INFO", f"Pedido generado para {r['proveedor']} ({r['num_lineas']} líneas)", supplier_id=r['proveedor'])

    # ----------------------------
    # EXPORTACIÓN
    # ----------------------------
    inventory_out = outdir / "inventory_updated.csv"
    ledger_out    = outdir / "ledger_movimientos.csv"
    alerts_out    = outdir / "alerts.csv"
    cands_out     = outdir / "sugerencias_compra.csv"
    oc_head_out   = outdir / "ordenes_compra.csv"
    oc_lines_out  = outdir / "ordenes_compra_lineas.csv"
    subs_out      = outdir / "sugerencias_sustitutos.csv"

    df.to_csv(inventory_out, index=False)
    pd.DataFrame(ledger_rows).to_csv(ledger_out, index=False)
    pd.DataFrame(alerts).to_csv(alerts_out, index=False)
    cands.to_csv(cands_out, index=False)
    if not oc_header.empty:
        oc_header.to_csv(oc_head_out, index=False)
        oc_lines.to_csv(oc_lines_out, index=False)
    # export sustitutos si existen
    if 'sug_subs' in locals() and isinstance(sug_subs, pd.DataFrame) and not sug_subs.empty:
        sug_subs.to_csv(subs_out, index=False)

    log.info("Inventario actualizado → %s", inventory_out)
    log.info("Ledger → %s | Alertas → %s | Candidatas → %s", ledger_out, alerts_out, cands_out)
    if not oc_header.empty:
        log.info("Órdenes → %s (cab) y %s (líneas)", oc_head_out, oc_lines_out)
    if 'sug_subs' in locals() and isinstance(sug_subs, pd.DataFrame):
        log.info("Sugerencias de sustitutos → %s (filas: %s)", subs_out, 0 if sug_subs is None else len(sug_subs))

# ----------------------------
# CLI o Notebook
# ----------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fase 10 · Procesador de movimientos de stock (con sustitutos).")
    p.add_argument("--inventario", type=Path, required=True)
    p.add_argument("--orders", type=Path, default=None)
    p.add_argument("--supplier-catalog", type=Path, default=None)
    p.add_argument("--service-policy", type=Path, default=None)
    p.add_argument("--substitutes", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("data/processed/fase10_stock"))
    p.add_argument("--autogen-demo", action="store_true")
    return p.parse_args(argv)

def _is_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False

if __name__ == "__main__":
    if _is_notebook():
        print("Notebook detectado → llama a run(...) manualmente desde una celda.")
    else:
        args = parse_args()
        try:
            ensure_dir(args.outdir)
            run(
                inventario=args.inventario,
                orders_path=args.orders,
                supplier_catalog=args.supplier_catalog,
                service_policy=args.service_policy,
                substitutes=args.substitutes,
                outdir=args.outdir,
                autogen_demo=args.autogen_demo,
            )
        except Exception as e:
            log.exception("Error en la simulación: %s", e)
            sys.exit(1)

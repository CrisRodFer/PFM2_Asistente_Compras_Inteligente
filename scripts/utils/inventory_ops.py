# =============================================================================
# Script: inventory_ops.py
# Descripción:
#   1) Inicializa el inventario vivo 'inventory_updated.csv' copiando TODO
#      desde 'data/clean/Inventario.csv' con 5 columnas canónicas:
#      [Product_ID, Proveedor, Nombre, Categoria, Stock Real].
#   2) Aplica pedidos de cliente restando stock SOLO a los Product_ID presentes
#      en el pedido (sin merges; usando map con default 0). Reescribe el
#      inventario vivo completo tras cada actualización.
#
# Flujo del pipeline:
#   Inventario.csv  --(reset_inventory_from_base)-->  inventory_updated.csv
#   inventory_updated.csv + orders_df --(apply_customer_orders)--> inventory_updated.csv
#
# Input esperado:
#   - data/clean/Inventario.csv (maestro de inventario)
#   - orders_df con columnas: Product_ID (o equivalente) y qty (o equivalente)
#
# Output:
#   - data/processed/fase10_stock/inventory_updated.csv (inventario vivo)
#
# Dependencias:
#   pip install pandas
# =============================================================================

# ============================ 0. CONFIG ======================================
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] if (__file__ in globals()) else Path.cwd()
DATA = ROOT / "data"
CLEAN = DATA / "clean"
PROCESSED = DATA / "processed"
OUT10 = PROCESSED / "fase10_stock"
OUT10.mkdir(parents=True, exist_ok=True)

INV_BASE = CLEAN / "Inventario.csv"                # maestro
INV_LIVE = OUT10 / "inventory_updated.csv"         # inventario vivo (se reescribe siempre)

# ============================ 1. IMPORTS + LOGGING ===========================
# (logging opcional; omitido por brevedad)

# ============================ 2. UTILIDADES ==================================
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        col = low.get(c.lower().strip())
        if col:
            return col
    return None

def _pid_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _to_int_safe(s: pd.Series, default: int = 0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)

def _read_csv_auto(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, sep=";", engine="python")

# ============================ 3. LÓGICA PRINCIPAL ============================
def reset_inventory_from_base() -> Path:
    """
    Carga 'Inventario.csv' y lo normaliza a:
    [Product_ID, Proveedor, Nombre, Categoria, Stock Real].
    Escribe TODO el contenido en 'inventory_updated.csv'.
    """
    inv = _read_csv_auto(INV_BASE)
    if inv is None or inv.empty:
        # crea archivo vacío con columnas canónicas
        inv = pd.DataFrame(columns=["Product_ID", "Proveedor", "Nombre", "Categoria", "Stock Real"])

    # map de columnas posibles
    pid = _find_col(inv, ["Product_ID", "item_id", "sku", "id_producto", "producto_id", "id"])
    prov = _find_col(inv, ["Proveedor", "supplier", "preferred_supplier_id"])
    name = _find_col(inv, ["Nombre", "name", "descripcion", "descripción"])
    cat  = _find_col(inv, ["Categoria", "Categoría", "category"])
    stk  = _find_col(inv, ["Stock Real", "stock_real", "stock actual", "stock_actual", "on_hand", "stock", "existencias", "qty", "cantidad"])

    # renombrar encontrados
    ren = {}
    if pid and pid != "Product_ID": ren[pid] = "Product_ID"
    if prov and prov != "Proveedor": ren[prov] = "Proveedor"
    if name and name != "Nombre":    ren[name] = "Nombre"
    if cat  and cat  != "Categoria": ren[cat]  = "Categoria"
    if stk  and stk  != "Stock Real":ren[stk]  = "Stock Real"
    if ren:
        inv = inv.rename(columns=ren)

    # asegurar todas
    for c in ["Product_ID", "Proveedor", "Nombre", "Categoria", "Stock Real"]:
        if c not in inv.columns:
            inv[c] = "" if c != "Stock Real" else 0

    inv["Product_ID"] = _pid_series(inv["Product_ID"])
    inv["Stock Real"] = _to_int_safe(inv["Stock Real"], default=0)

    inv = inv[["Product_ID", "Proveedor", "Nombre", "Categoria", "Stock Real"]].copy()
    inv.to_csv(INV_LIVE, index=False)
    return INV_LIVE

def apply_customer_orders(orders_df: pd.DataFrame, clip_zero: bool = True) -> pd.DataFrame:
    """
    Resta stock SOLO para los Product_ID presentes en 'orders_df' (sin merges).
    - orders_df debe tener: id (Product_ID o alias) y qty (o alias).
    - Reescribe 'inventory_updated.csv' completo (todos los productos).
    """
    if not INV_LIVE.exists():
        reset_inventory_from_base()
    inv = pd.read_csv(INV_LIVE)

    pid_col = _find_col(orders_df, ["Product_ID", "item_id", "sku", "id_producto", "producto_id", "id"])
    qty_col = _find_col(orders_df, ["qty", "quantity", "cantidad", "units", "unidades", "Sales Quantity"])
    if not pid_col or not qty_col:
        raise ValueError("orders_df debe incluir columna de ID y de cantidad.")

    tmp = orders_df[[pid_col, qty_col]].copy()
    tmp[pid_col] = _pid_series(tmp[pid_col])
    tmp[qty_col] = _to_int_safe(tmp[qty_col], default=0)

    # cantidades por ID (sumadas por si llegan varias líneas del mismo producto)
    qty_by_id = tmp.groupby(pid_col)[qty_col].sum().to_dict()

    inv["Product_ID"] = _pid_series(inv["Product_ID"])
    inv["Stock Real"] = _to_int_safe(inv["Stock Real"], default=0)

    # resta sin merge: map -> fillna(0) -> int
    inv["Stock Real"] = inv["Stock Real"] - inv["Product_ID"].map(qty_by_id).fillna(0).astype(int)
    if clip_zero:
        inv["Stock Real"] = inv["Stock Real"].clip(lower=0)

    inv = inv[["Product_ID", "Proveedor", "Nombre", "Categoria", "Stock Real"]]
    inv.to_csv(INV_LIVE, index=False)
    return inv

# ============================ 4. EXPORTACIÓN / I/O ===========================
# (las funciones ya escriben/leen los CSV)

# ============================ 5. CLI / MAIN =================================
if __name__ == "__main__":
    # Ejemplo mínimo de uso desde terminal (opcional):
    #   python scripts/utils/inventory_ops.py reset
    #   python scripts/utils/inventory_ops.py apply data/processed/pedidos_hoy.csv
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "reset":
        p = reset_inventory_from_base()
        print(f"[OK] Inventario inicializado en: {p}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "apply":
        orders_path = Path(sys.argv[2])
        orders_df = _read_csv_auto(orders_path)
        new_inv = apply_customer_orders(orders_df)
        print(f"[OK] Stock actualizado. Filas: {len(new_inv)}")
    else:
        print("Uso:\n  python inventory_ops.py reset\n  python inventory_ops.py apply <ruta_orders_csv>")

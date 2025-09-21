# app/streamlit_app.py

import sys
from pathlib import Path
from datetime import datetime
import io

import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np

st.set_page_config(page_title="PFM2", page_icon="🧭", layout="wide")

# ---------------- Router sencillo en session_state ----------------
ROUTES = {
    "home": "🏠 Portada",
    "exploracion": "🔎 Exploración & Sustitutos",
    "proveedores": "🏭 Proveedores",
    "movimientos": "📦 Movimientos de stock",
    "reapro": "🧾 Reapro / Pedidos",
}
if "route" not in st.session_state:
    st.session_state["route"] = "home"

def goto(r: str):
    st.session_state["route"] = r

# ===================== SIDEBAR: Navegación =====================
with st.sidebar:
    st.header("🧭 Navegación")
    choice = st.radio(
        "Ir a:",
        list(ROUTES.keys()),
        format_func=lambda k: ROUTES.get(k, k),
        index=list(ROUTES.keys()).index(st.session_state.get("route", "home"))
    )
    if choice != st.session_state.get("route"):
        goto(choice)
        st.rerun()

# -------------- CSS mínimo --------------
st.markdown(
    """
    <style>
    .st-emotion-cache-1v0mbdj { gap: .3rem; } /* tabs compactas */
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------- JS peque autorefresh --------------
def autorefresh(interval: int | None = None):
    if not interval or interval <= 0:
        return
    st.markdown(
        f"<script>setTimeout(function(){{window.location.reload();}}, {interval});</script>",
        unsafe_allow_html=True,
    )

def st_autorefresh(interval: int, key: str = "auto_refresh"):
    """Compat alias para el nombre usado en las pestañas."""
    autorefresh(interval)

# Permitir importar desde scripts/… (raíz del proyecto)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.export.construir_vistas import construir_vistas


# ==================================================
# 1). Utils y rutas de datos
# ================================
ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / "data"
RAW   = DATA / "raw"
CLEAN = DATA / "clean"
PROC  = DATA / "processed"

UNIF  = PROC  / "substitutes_unified.csv"
MULTI = CLEAN / "supplier_catalog_multi.csv"
CAT   = PROC  / "catalog_items_enriquecido.csv"
STOCK = PROC  / "stock_positions.csv"

OUT10 = PROC / "fase10_stock"              # outputs del procesador de movimientos
ORD_UI = RAW / "customer_orders_ui.csv"    # pedidos añadidos desde la UI
ORD_AE = RAW / "customer_orders_AE.csv"    # escenarios A–E (si quieres sumarlos)
SUPPLIER_DEMO = RAW / "supplier_catalog_demo.csv"  # demo de proveedores para Fase 10

TMP_VISTAS = PROC / "_tmp_vistas_streamlit"
TMP_VISTAS.mkdir(parents=True, exist_ok=True)

def _read_csv_smart(p: Path) -> pd.DataFrame | None:
    if not p or not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        # separador desconocido / encoding
        return pd.read_csv(p, sep=None, engine="python", encoding="utf-8")

def _to_str_safe(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _to_pid_str(s: pd.Series) -> pd.Series:
    # 2141.0 -> '2141' ; preserva vacíos
    s_num = pd.to_numeric(s, errors="coerce")
    out = s_num.astype("Int64").astype(str)
    return out.replace("<NA>", "")

def _norm_pid(s: pd.Series) -> pd.Series:
    return _to_pid_str(s)

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _ensure_pid_col(df: pd.DataFrame, prefer_int: bool = True) -> pd.DataFrame:
    """
    Devuelve DF con columna 'Product_ID' garantizada.
    Acepta 'item_id' o 'Product_ID' y las normaliza.
    """
    if df is None or df.empty:
        return df
    low = {c.lower(): c for c in df.columns}
    pid_col = low.get("product_id") or low.get("item_id")
    if pid_col and pid_col != "Product_ID":
        df = df.rename(columns={pid_col: "Product_ID"})
    if "Product_ID" in df.columns:
        s = df["Product_ID"].astype(str).str.strip()
        if prefer_int:
            s = s.str.replace(r"\.0+$", "", regex=True)
        df["Product_ID"] = s
    return df

def _mtime_str(p: Path) -> str:
    try:
        ts = datetime.fromtimestamp(p.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"

def _mtime(p: Path) -> float:
    """Devuelve el mtime (epoch) del path o 0.0 si no existe/da error."""
    try:
        return p.stat().st_mtime if p and p.exists() else 0.0
    except Exception:
        return 0.0

def _call_construir_vistas_robusto(
    unif_path, multi_path, cat_path, stock_path, outdir,
    min_score, m_unif, m_multi, m_cat, m_stock
):
    # 1) Intento con nombres “largos” (lo que tenías)
    try:
        return construir_vistas(
            substitutes_unified=unif_path,
            supplier_catalog_multi=multi_path,
            catalog_items=cat_path,
            stock_positions=stock_path,
            outdir=outdir,
            min_score=min_score,
            m_unif=m_unif, m_multi=m_multi, m_cat=m_cat, m_stock=m_stock
        )
    except TypeError:
        pass

    # 2) Intento por POSICIÓN (orden más habitual)
    try:
        return construir_vistas(
            unif_path, multi_path, cat_path, stock_path,
            outdir, min_score, m_unif, m_multi, m_cat, m_stock
        )
    except TypeError:
        pass

    # 3) Intento con nombres alternativos “cortos”
    try:
        return construir_vistas(
            substitutes=unif_path,
            supplier_catalog=multi_path,
            catalog=cat_path,
            stock=stock_path,
            out=outdir,
            score=min_score,
            mu=m_unif, mm=m_multi, mc=m_cat, ms=m_stock
        )
    except TypeError as e:
        raise e


@st.cache_data(ttl=10)
def _safe_df(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

# ========= NUEVOS HELPERS =========
def _append_order_row(date_str: str, product_id: int | str, qty: int) -> None:
    """Añade una línea a customer_orders_ui.csv (si no existe lo crea)."""
    _ensure_orders_ui()
    try:
        df = pd.read_csv(ORD_UI)
    except Exception:
        df = pd.DataFrame(columns=["date", "Product_ID", "qty"])
    new_row = {"date": date_str, "Product_ID": str(product_id).strip(), "qty": int(qty)}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(ORD_UI, index=False)

def _read_inventory_base() -> pd.DataFrame:
    """
    Lee data/clean/Inventario.csv y normaliza SIEMPRE a:
      - Product_ID (string)
      - Stock_actual (int)
    Tolera nombres distintos y hasta deduce columnas si no hay cabeceras estándar.
    """
    inv_base = ROOT / "data" / "clean" / "Inventario.csv"
    df = _read_csv_smart(inv_base)  # autodetección de separador/encoding
    if df is None or df.empty:
        return pd.DataFrame(columns=["Product_ID", "Stock_actual"])

    # Mapa insensible a mayúsculas/espacios
    low = {c.lower().strip(): c for c in df.columns}

    # --- localizar columna ID ---
    pid_col = None
    for key in ("product_id", "item_id", "sku", "id_producto", "producto_id", "id"):
        if key in low:
            pid_col = low[key]
            break
    if pid_col is None:
        # si no hay cabecera reconocible, usa la PRIMERA columna
        pid_col = df.columns[0]

    # --- localizar columna stock ---
    stock_col = None
    for key in ("stock_actual", "on_hand", "stock", "existencias", "qty", "cantidad"):
        if key in low:
            stock_col = low[key]
            break
    if stock_col is None:
        # última columna con mayoría numérica; si no hay, la última
        numericish = [c for c in df.columns
                      if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.6]
        stock_col = numericish[-1] if numericish else df.columns[-1]

    # --- renombrar y tipar ---
    if pid_col != "Product_ID":
        df = df.rename(columns={pid_col: "Product_ID"})
    if stock_col != "Stock_actual":
        df = df.rename(columns={stock_col: "Stock_actual"})

    df["Product_ID"] = (
        df["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    )
    df["Stock_actual"] = pd.to_numeric(df["Stock_actual"], errors="coerce").fillna(0).astype(int)

    return df[["Product_ID", "Stock_actual"]]

def _read_working_inventory() -> pd.DataFrame:
    """
    Devuelve el inventario 'vivo' con nombres EXACTOS:
      Product_ID, Proveedor, Nombre, Categoria, Stock Real

    Prioridad:
      1) OUT10/inventory_updated.csv (si existe)
      2) data/clean/Inventario.csv  (si no existe el actualizado)
    """
    def _normalize_inv(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Product_ID", "Proveedor", "Nombre", "Categoria", "Stock Real"])

        low = {c.lower().strip(): c for c in df.columns}

        # ----- mapear columnas a canónicas -----
        # ID
        pid = (low.get("product_id") or low.get("item_id") or low.get("sku")
               or low.get("id_producto") or low.get("producto_id") or low.get("id") or "Product_ID")
        if pid != "Product_ID":
            df = df.rename(columns={pid: "Product_ID"})

        # Proveedor
        prov = (low.get("proveedor") or low.get("supplier") or low.get("scm_supplier_id")
                or low.get("preferred_supplier_id") or "Proveedor")
        if prov != "Proveedor" and prov in df.columns:
            df = df.rename(columns={prov: "Proveedor"})
        elif "Proveedor" not in df.columns:
            df["Proveedor"] = ""

        # Nombre
        name = (low.get("nombre") or low.get("name") or low.get("descripcion") or "Nombre")
        if name != "Nombre" and name in df.columns:
            df = df.rename(columns={name: "Nombre"})
        elif "Nombre" not in df.columns:
            df["Nombre"] = ""

        # Categoria
        cat = (low.get("categoria") or low.get("category") or low.get("familia") or "Categoria")
        if cat != "Categoria" and cat in df.columns:
            df = df.rename(columns={cat: "Categoria"})
        elif "Categoria" not in df.columns:
            df["Categoria"] = ""

        # Stock Real  (aceptamos variantes habituales)
        stock = (low.get("stock real") or low.get("stock_real") or low.get("stock actual") or
                 low.get("stock_actual") or low.get("on_hand") or low.get("stock") or low.get("existencias"))
        if stock and stock != "Stock Real":
            df = df.rename(columns={stock: "Stock Real"})
        if "Stock Real" not in df.columns:
            # si no hubo nada parecido, intentamos última col numérica o generamos 0
            numericish = [c for c in df.columns
                          if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.6]
            if numericish:
                df = df.rename(columns={numericish[-1]: "Stock Real"})
            else:
                df["Stock Real"] = 0

        # ----- normalizar tipos -----
        df["Product_ID"] = _to_pid_str(df["Product_ID"])
        df["Stock Real"] = pd.to_numeric(df["Stock Real"], errors="coerce").fillna(0).astype(int)

        # Devolver sólo las 5 columnas, en orden
        cols_final = ["Product_ID", "Proveedor", "Nombre", "Categoria", "Stock Real"]
        # algunas podrían no existir si el CSV no las traía; garantizamos su presencia
        for c in cols_final:
            if c not in df.columns:
                df[c] = "" if c != "Stock Real" else 0
        return df[cols_final]

    # 1) Intento con el actualizado
    upd_path = OUT10 / "inventory_updated.csv"
    if upd_path.exists():
        df_upd = _read_csv_smart(upd_path)
        if df_upd is not None and not df_upd.empty:
            return _normalize_inv(df_upd)

    # 2) Fallback: Inventario.csv base
    base_path = ROOT / "data" / "clean" / "Inventario.csv"
    df_base = _read_csv_smart(base_path)
    return _normalize_inv(df_base)

def _standardize_inventory_output(path: Path) -> None:
    """Fuerza que inventory_updated.csv salga con Product_ID + on_hand."""
    df = _read_csv_smart(path)
    if df is None or df.empty:
        return
    low = {c.lower(): c for c in df.columns}
    ren = {}

    # ID → Product_ID
    if "product_id" in low and low["product_id"] != "Product_ID":
        ren[low["product_id"]] = "Product_ID"
    elif "item_id" in low:
        ren[low["item_id"]] = "Product_ID"

    # stock → on_hand (incluye stock_actual)
    for cand in ["on_hand", "stock_actual", "stock", "qty", "quantity", "on_new", "stock_qty"]:
        if cand in low and low[cand] != "on_hand":
            ren[low[cand]] = "on_hand"
            break

    if ren:
        df = df.rename(columns=ren)

    if "Product_ID" in df.columns:
        df["Product_ID"] = (
            df["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
        )
    if "on_hand" in df.columns:
        df["on_hand"] = pd.to_numeric(df["on_hand"], errors="coerce").fillna(0).astype(int)
    else:
        df["on_hand"] = 0

    df.to_csv(path, index=False)

# ============================ helper: load_demanda_base ============================
def load_demanda_base(root: Path) -> pd.DataFrame | None:
    """
    Carga la demanda base/prevista diaria para alimentar SS/ROP.
    Intenta:
      1) data/processed/predicciones_2025_estacional.parquet (preferente)
      2) data/processed/demanda_subset.csv (fallback)
    Normaliza a columnas: Product_ID (str), Date (datetime), sales_quantity (float).
    Acepta nombres alternativos: item_id/Product_ID, fecha/Date, demanda/sales_quantity.
    """
    candidatos = [
        root / "data/processed/predicciones_2025_estacional.parquet",
        root / "data/processed/demanda_subset.csv",
    ]

    df = None
    for p in candidatos:
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
                break
            except Exception:
                df = None

    if df is None or df.empty:
        return None

    # Normalizar nombres
    low = {c.lower(): c for c in df.columns}
    pid_col = low.get("product_id") or low.get("item_id") or low.get("sku")
    date_col = low.get("date") or low.get("fecha")
    qty_col = (low.get("sales_quantity") or low.get("quantity") or low.get("qty")
               or low.get("demand") or low.get("demanda") or low.get("y") )

    # Si falta cualquiera de los 3, intentar inferir mínimo para media diaria (sin fecha)
    if not pid_col:
        # último recurso: si hay 'Product_ID' con mayúsculas exactas
        pid_col = "Product_ID" if "Product_ID" in df.columns else None
    if not qty_col:
        # último recurso: columna numérica principal
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        qty_col = num_cols[0] if num_cols else None

    if not pid_col or not qty_col:
        return None  # sin Product_ID o sin cantidad, no sirve

    out = pd.DataFrame()
    out["Product_ID"] = df[pid_col].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    out["sales_quantity"] = pd.to_numeric(df[qty_col], errors="coerce")

    if date_col:
        out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        # sin fecha: dejamos Date como NaT (seguirá funcionando con SS fijo y ROP ~ μ_d * L + SS)
        out["Date"] = pd.NaT

    # limpiar
    out = out.dropna(subset=["Product_ID"])
    return out

# ============================ helper: maestro productos ============================
def build_product_master(root: Path) -> pd.DataFrame:
    """
    Devuelve un DF con columnas: Product_ID, Nombre, Proveedor, Categoria (si existe).
    Origen:
      - Nombre/Categoria: load_catalog_items(root)
      - Proveedor: ui_products.parquet (preferred_supplier_id) y fallback a multi (scm_supplier_id)
    """
    # 1) Catálogo (Nombre/Categoria)
    cat = load_catalog_items(root)
    if cat is None or cat.empty:
        cat = pd.DataFrame(columns=["Product_ID", "Nombre", "Categoria"])
    else:
        keep_cols = [c for c in ["Product_ID", "Nombre", "Categoria"] if c in cat.columns]
        cat = cat[keep_cols].drop_duplicates("Product_ID")

    # 2) Proveedor: prioriza ui_products.parquet (preferred_supplier_id)
    prov_map = pd.DataFrame(columns=["Product_ID", "Proveedor"])
    try:
        ui_prod_path = TMP_VISTAS / "ui_products.parquet"
        if ui_prod_path.exists():
            up = pd.read_parquet(ui_prod_path)
            up = _ensure_pid_col(up)
            low_up = {c.lower(): c for c in up.columns}
            prov_col = (low_up.get("preferred_supplier_id")
                        or low_up.get("proveedor")
                        or low_up.get("supplier"))
            if prov_col:
                prov_map = (up.rename(columns={prov_col: "Proveedor"})
                              [["Product_ID", "Proveedor"]]
                              .drop_duplicates("Product_ID"))
    except Exception:
        pass

    # 3) Fallback proveedor desde multi
    if prov_map.empty:
        multi = load_supplier_catalog_multi(root)
        if multi is not None and not multi.empty:
            mm = multi.drop_duplicates("Product_ID")
            prov_map = mm[["Product_ID", "scm_supplier_id"]].rename(
                columns={"scm_supplier_id": "Proveedor"}
            )

    # 4) Normalizar IDs y unir
    for df in (cat, prov_map):
        if "Product_ID" in df.columns:
            df["Product_ID"] = (
                df["Product_ID"].astype(str).str.strip()
                .str.replace(r"\.0+$", "", regex=True)
            )

    master = pd.merge(cat, prov_map, on="Product_ID", how="outer")
    # Asegurar columnas
    for c in ["Product_ID", "Nombre", "Proveedor", "Categoria"]:
        if c not in master.columns:
            master[c] = "" if c != "Product_ID" else master.get("Product_ID", "")
    return master

# ========================= helper: append_ledger =========================
def _normalize_pid(x) -> str:
    return str(x).strip().replace(".0", "")

def append_ledger(rows: pd.DataFrame) -> None:
    """
    Escribe OUT10/ledger_movimientos.csv con el esquema CANÓNICO:
    Date, Product_ID, Nombre, Proveedor, Tipo movimiento, qty_pedido
    (acepta entrada con 'Tipo' o 'Tipo movimiento')
    """
    CANON_COLS = ["Date", "Product_ID", "Nombre", "Proveedor", "Tipo movimiento", "qty_pedido"]
    ledger_path = OUT10 / "ledger_movimientos.csv"

    # --- Normalizar columna Date tras leer CSV ---
    if ledger_path.exists():
        try:
            led = pd.read_csv(ledger_path, dtype={"Product_ID": str})
            led["Date"] = pd.to_datetime(led["Date"], errors="coerce").dt.normalize()
        except Exception:
            led = pd.DataFrame(columns=CANON_COLS)
    else:
        led = pd.DataFrame(columns=CANON_COLS)

    if rows is None or rows.empty:
        return

    r = rows.copy()

    # Aceptar 'Tipo' y normalizar a 'Tipo movimiento'
    if "Tipo movimiento" not in r.columns and "Tipo" in r.columns:
        r = r.rename(columns={"Tipo": "Tipo movimiento"})

    # Fechas / IDs / cantidades
    r["Date"] = pd.to_datetime(r.get("Date", pd.Timestamp.now()), errors="coerce").dt.normalize()
    r["Product_ID"] = (
        r["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    )
    r["qty_pedido"] = pd.to_numeric(r.get("qty_pedido", 0), errors="coerce").fillna(0).astype(int)

    # Añadir Nombre/Proveedor desde el master
    master = build_product_master(ROOT)[["Product_ID", "Nombre", "Proveedor"]]
    master["Product_ID"] = master["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    r = r.merge(master, on="Product_ID", how="left")

    # Quedarnos con el canónico y sanear
    r = r[["Date", "Product_ID", "Nombre", "Proveedor", "Tipo movimiento", "qty_pedido"]]
    r = r.dropna(subset=["Date", "Product_ID"])
    r = r[r["Product_ID"].str.isnumeric()]
    r = r[r["qty_pedido"] > 0]

    # Append ordenado
    if ledger_path.exists():
        old = pd.read_csv(ledger_path, dtype={"Product_ID": str})
        out = pd.concat([old, r], ignore_index=True)
    else:
        out = r
    out = out[CANON_COLS].sort_values(["Date", "Product_ID"]).reset_index(drop=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ledger_path, index=False, encoding="utf-8")

def repair_ledger_canonic():
    """Limpia OUT10/ledger_movimientos.csv a esquema canónico:
    Date, Product_ID, Nombre, Proveedor, Tipo movimiento, qty_pedido
    - Recalcula qty_pedido si viene mal (con delta u on_prev/on_new)
    - Normaliza Tipo movimiento (venta por defecto si venían números)
    - Reconstituye Nombre/Proveedor desde el maestro
    """
    ledger_path = OUT10 / "ledger_movimientos.csv"
    if not ledger_path.exists():
        return

    df = pd.read_csv(ledger_path)
    if df.empty:
        return
    
    # ==================== Helper en camino.  ====================
def _load_en_camino_por_pid() -> pd.DataFrame:
    """
    Devuelve: Product_ID (str), en_camino_qty (int), eta_min (float/int o NaN).
    Combina oc_en_curso*.csv con las tablas unificadas OC_HDR/OC_LIN.
    """
    def _agg_from_cur() -> pd.DataFrame | None:
        try:
            lin = _oc_read_cur_lin()
            if lin.empty:
                return None
            lin = _ensure_pid_col(lin, prefer_int=False)
            lin["Product_ID"] = _to_pid_str(lin["Product_ID"])
            lin["Cantidad_pedir"] = pd.to_numeric(lin["Cantidad_pedir"], errors="coerce").fillna(0).astype(int)

            hdr = _oc_read_cur_hdr()
            eta = hdr[["order_id", "ETA_dias"]] if not hdr.empty and "ETA_dias" in hdr.columns else None
            if eta is not None:
                lin = lin.merge(eta, on="order_id", how="left")

            return lin.groupby("Product_ID", as_index=False).agg(
                en_camino_qty=("Cantidad_pedir", "sum"),
                eta_min=("ETA_dias", "min"),
            )
        except Exception:
            return None

    def _agg_from_unified() -> pd.DataFrame | None:
        try:
            hdr = _oc_read_hdr()
            lin = _oc_read_lin()
            if hdr.empty or lin.empty:
                return None
            en = hdr[hdr["Estado"] == "en_curso"].copy()
            if en.empty:
                return None

            det = lin.merge(en[["order_id", "ETA_dias"]], on="order_id", how="inner")
            det = _ensure_pid_col(det, prefer_int=False)
            det["Product_ID"] = _to_pid_str(det["Product_ID"])

            # Normalizar cantidad (por si viniera con otro nombre)
            if "Cantidad_pedir" not in det.columns:
                low = {c.lower(): c for c in det.columns}
                for k in ("qty", "cantidad", "quantity", "cantidad_pedida"):
                    if k in low:
                        det.rename(columns={low[k]: "Cantidad_pedir"}, inplace=True)
                        break

            det["Cantidad_pedir"] = pd.to_numeric(det["Cantidad_pedir"], errors="coerce").fillna(0).astype(int)

            return det.groupby("Product_ID", as_index=False).agg(
                en_camino_qty=("Cantidad_pedir", "sum"),
                eta_min=("ETA_dias", "min"),
            )
        except Exception:
            return None

    parts = [x for x in (_agg_from_cur(), _agg_from_unified()) if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame(columns=["Product_ID", "en_camino_qty", "eta_min"])

    out = pd.concat(parts, ignore_index=True)
    out = out.groupby("Product_ID", as_index=False).agg(
        en_camino_qty=("en_camino_qty", "sum"),
        eta_min=("eta_min", "min"),
    )
    out["Product_ID"] = _to_pid_str(out["Product_ID"])
    out["en_camino_qty"] = pd.to_numeric(out["en_camino_qty"], errors="coerce").fillna(0).astype(int)
    return out[["Product_ID", "en_camino_qty", "eta_min"]]
    
def _oc_sync_append_cur(hdr_row: pd.DataFrame, lin_rows: pd.DataFrame) -> None:
    """Añade una orden recién creada a oc_en_curso.csv y oc_en_curso_lineas.csv."""
    hcols = ["order_id", "Proveedor", "Fecha", "Escenario", "ETA_dias"]
    lcols = ["order_id", "Product_ID", "Nombre", "Cantidad_pedir"]

    cur_hdr = _oc_read_cur_hdr()
    cur_lin = _oc_read_cur_lin()

    new_hdr = hdr_row[hcols].copy()
    new_lin = lin_rows[lcols].copy()

    # Normaliza por si acaso
    new_lin["Product_ID"] = _to_pid_str(new_lin["Product_ID"])
    new_lin["Cantidad_pedir"] = pd.to_numeric(new_lin["Cantidad_pedir"], errors="coerce").fillna(0).astype(int)

    _oc_write(OC_CUR_HDR, pd.concat([cur_hdr, new_hdr], ignore_index=True))
    _oc_write(OC_CUR_LIN, pd.concat([cur_lin, new_lin], ignore_index=True))


def _oc_sync_remove_cur(order_id: str) -> None:
    """Elimina una orden de los archivos 'en curso' (cuando se recibe/forza recepción)."""
    cur_hdr = _oc_read_cur_hdr()
    cur_lin = _oc_read_cur_lin()
    if not cur_hdr.empty:
        _oc_write(OC_CUR_HDR, cur_hdr[cur_hdr["order_id"] != order_id].copy())
    if not cur_lin.empty:
        _oc_write(OC_CUR_LIN, cur_lin[cur_lin["order_id"] != order_id].copy())

# ===== TOP VENTAS (outliers.csv) =====
def _load_top_ventas_from_outliers() -> pd.DataFrame:
    """
    Devuelve DF con: Product_ID (str), top_ventas (bool)
    Lee data/processed/outliers.csv (y rutas alternativas) y deduplica por Product_ID usando "any".
    """
    from pathlib import Path

    candidates = [
        ROOT / "data" / "processed" / "outliers.csv",   # ruta principal en tu proyecto
        OUT10 / "outliers.csv",                         # fallback
        Path("/mnt/data/outliers.csv"),                 # fallback en sesión actual
    ]
    df = None
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p)
                break
        except Exception:
            pass
    if df is None or df.empty:
        return pd.DataFrame(columns=["Product_ID", "top_ventas"])

    # Normalizar Product_ID
    pid_col = None
    low = {c.lower(): c for c in df.columns}
    for cand in ("product_id", "Product_ID", "id_producto", "sku"):
        if cand.lower() in low:
            pid_col = low[cand.lower()]
            break
    if not pid_col:
        return pd.DataFrame(columns=["Product_ID", "top_ventas"])

    df = df.rename(columns={pid_col: "Product_ID"})
    df["Product_ID"] = _to_pid_str(df["Product_ID"])

    # Detectar la columna de flag
    flag_col = None
    for cand in ("is_outlier", "outlier", "is_top", "top_ventas"):
        if cand.lower() in low:
            flag_col = low[cand.lower()]
            break
    if not flag_col:
        return pd.DataFrame(columns=["Product_ID", "top_ventas"])

    # Convertir a boolean (1/0, True/False, yes/no)
    s = df[flag_col]
    num = pd.to_numeric(s, errors="coerce")
    bool_num = (num.fillna(0) > 0)
    bool_txt = s.astype(str).str.strip().str.lower().isin(["true","t","yes","y","top","si","sí"])
    df["__flag__"] = bool_num | bool_txt

    # Deduplicar: True si alguna fila lo marca
    top = (df.groupby("Product_ID", as_index=False)["__flag__"].max()
             .rename(columns={"__flag__": "top_ventas"}))
    return top[["Product_ID", "top_ventas"]]

# ==================== HELPER Sustitutos ====================

@st.cache_data(ttl=60)
def _load_subs_summary() -> pd.DataFrame:
    """
    Devuelve por Product_ID:
      - subs_count: nº de sustitutos distintos
      - subs_ids:   primeros 3 IDs de sustitutos (comma-separated)
    Base: substitutes_unified.csv (vía load_substitutes_unified)
    """
    unif = load_substitutes_unified(ROOT)
    if unif is None or unif.empty:
        return pd.DataFrame(columns=["Product_ID", "subs_count", "subs_ids"])

    u = unif.copy()
    u = u.dropna(subset=["Product_ID", "Substitute_Product_ID"])
    u = u[u["Substitute_Product_ID"] != ""]
    if u.empty:
        return pd.DataFrame(columns=["Product_ID", "subs_count", "subs_ids"])

    # Asegurar ID en formato string "limpio"
    u["Product_ID"] = _to_pid_str(u["Product_ID"])
    u["Substitute_Product_ID"] = _to_pid_str(u["Substitute_Product_ID"])

    def _top3_ids(s: pd.Series) -> str:
        vals = pd.unique(s.dropna().astype(str))
        vals = [v for v in vals if v and v != "nan"]
        return ", ".join(vals[:3])

    agg = u.groupby("Product_ID", as_index=False).agg(
        subs_count=("Substitute_Product_ID", "nunique"),
        subs_ids=("Substitute_Product_ID", _top3_ids),
    )
    agg["subs_count"] = pd.to_numeric(agg["subs_count"], errors="coerce").fillna(0).astype(int)
    agg["subs_ids"] = agg["subs_ids"].fillna("")
    return agg[["Product_ID", "subs_count", "subs_ids"]]

@st.cache_data(ttl=300)
def _load_forecast_neutral() -> pd.DataFrame:
    """
    Lee la previsión neutra 2025 (baseline).
    Usa y_pred_estacional si existe; si no, cae a y_pred.
    """
    from pathlib import Path
    cands = [
        ROOT / "data" / "processed" / "predicciones_2025_estacional.parquet",
        Path(r"C:\Users\crisr\Desktop\Máster Data Science & IA\PROYECTO\PFM2_Asistente_Compras_Inteligente\data\processed\predicciones_2025_estacional.parquet"),
        Path("/mnt/data/predicciones_2025_estacional.parquet"),
    ]
    df = None
    for p in cands:
        try:
            if p.exists():
                df = pd.read_parquet(p)
                break
        except Exception:
            df = None

    if df is None or df.empty:
        return pd.DataFrame(columns=["Product_ID","Date","qty_forecast","cluster_id"])

    low = {c.lower(): c for c in df.columns}
    pid_col   = low.get("product_id") or "product_id"
    date_col  = low.get("date") or "date"
    y_est_col = low.get("y_pred_estacional")
    y_col     = y_est_col or low.get("y_pred") or "y_pred"
    cl_col    = low.get("cluster_id") or "cluster_id"

    out = df[[pid_col, date_col, y_col] + ([cl_col] if cl_col in df.columns else [])].copy()
    out = out.rename(columns={
        pid_col:  "Product_ID",
        date_col: "Date",
        y_col:    "qty_forecast",
        **({cl_col: "cluster_id"} if cl_col in df.columns else {})
    })
    out["Product_ID"]   = _to_pid_str(out["Product_ID"])
    out["Date"]         = pd.to_datetime(out["Date"], errors="coerce")
    out["qty_forecast"] = pd.to_numeric(out["qty_forecast"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["Date"])


@st.cache_data(ttl=600)
def _load_clusters_map() -> pd.DataFrame:
    """
    Devuelve: Product_ID, Cluster
    Lee demanda_subset.* (Cluster textual o cluster_id numérico).
    """
    from pathlib import Path
    cands = [
        ROOT / "data" / "processed" / "demanda_subset.csv",
        ROOT / "data" / "processed" / "demanda_subset.parquet",
        Path("/mnt/data/demanda_subset.csv"),
        Path("/mnt/data/demanda_subset.parquet"),
    ]
    df = None
    for p in cands:
        try:
            if p.exists():
                df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                break
        except Exception:
            df = None

    if df is None or df.empty:
        return pd.DataFrame(columns=["Product_ID","Cluster"])

    low = {c.lower(): c for c in df.columns}
    pid = low.get("product_id") or "Product_ID"
    cl  = low.get("cluster") or low.get("cluster_id")

    out = df[[pid, cl]].drop_duplicates().rename(columns={pid: "Product_ID", cl: "Cluster"})
    out["Product_ID"] = _to_pid_str(out["Product_ID"])
    out["Cluster"] = out["Cluster"].apply(lambda x: str(x))
    return out[["Product_ID","Cluster"]]


# Elasticidades: aceptan numérico o textual
ELASTICIDADES = {
    "0":  -0.6, "C0": -0.6,
    "1":  -1.0, "C1": -1.0,
    "2":  -1.2, "C2": -1.2,
    "3":  -0.8, "C3": -0.8,
}
def _elasticidad(cluster_val: str) -> float:
    return float(ELASTICIDADES.get(str(cluster_val).strip(), ELASTICIDADES["1"]))


def _get_cluster(pid: str, cmap: pd.DataFrame) -> str:
    row = cmap[cmap["Product_ID"] == pid]
    if row.empty:
        return "C1"
    return str(row.iloc[0]["Cluster"])


# ============= HELPERS PROVEEDORES ==========

@st.cache_data(ttl=300)
def _load_hist_demand() -> pd.DataFrame:
    """
    Carga demanda histórica 2023-2025 (hasta hoy) desde demanda_subset.* 
    y la normaliza a: Product_ID, Date, qty
    """
    from pathlib import Path
    cands = [
        ROOT / "data" / "processed" / "demanda_subset.csv",
        ROOT / "data" / "processed" / "demanda_subset.parquet",
        Path("/mnt/data/demanda_subset.csv"),
        Path("/mnt/data/demanda_subset.parquet"),
    ]
    df = None
    for p in cands:
        try:
            if p.exists():
                df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                break
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame(columns=["Product_ID","Date","qty"])

    # <-- strip() para tolerar espacios en los nombres de columnas
    low = {c.lower().strip(): c for c in df.columns}

    pid  = low.get("product_id") or low.get("item_id") or "Product_ID"
    date = low.get("date") or low.get("fecha") or "Date"

    # Preferimos y_pred_estacional; si no, y_pred; si no, sales_quantity…
    qty = (low.get("y_pred_estacional") or low.get("y_pred")
           or low.get("sales_quantity") or low.get("quantity") or low.get("qty"))

    if not pid or not date or not qty or (pid not in df.columns) or (date not in df.columns) or (qty not in df.columns):
        return pd.DataFrame(columns=["Product_ID","Date","qty"])

    out = df[[pid, date, qty]].copy()
    out = out.rename(columns={pid: "Product_ID", date: "Date", qty: "qty"})
    out["Product_ID"] = _to_pid_str(out["Product_ID"])
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["Date"])
    out = out[out["Date"] >= pd.Timestamp(2023, 1, 1)]
    return out

@st.cache_data(ttl=300)
def _load_forecast_2025() -> pd.DataFrame:
    """
    Devuelve la previsión completa de 2025 normalizada a:
      Product_ID (str), Date (datetime), qty (float)
    Reutiliza _load_forecast_neutral() y mapea qty_forecast -> qty
    """
    f = _load_forecast_neutral()
    if f is None or f.empty:
        return pd.DataFrame(columns=["Product_ID", "Date", "qty"])

    out = f.rename(columns={"qty_forecast": "qty"})[["Product_ID", "Date", "qty"]].copy()
    out["Product_ID"] = _to_pid_str(out["Product_ID"])
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["Date"])
    # Filtramos 2025 completo
    out = out[(out["Date"] >= pd.Timestamp(2025, 1, 1)) &
              (out["Date"] <= pd.Timestamp(2025, 12, 31))]
    return out

@st.cache_data(ttl=300)
def _price_map() -> pd.DataFrame:
    """
    Devuelve Product_ID, Nombre, Proveedor, Categoria, Precio.
    - Proveedor / Nombre / Categoria: build_product_master()
    - Precio: ui_products.preferred_price (si existe) y, si no, precio_medio/precio/price del catálogo enriquecido.
    """
    # ===== Base (Nombre / Proveedor / Categoria)
    master = build_product_master(ROOT)
    if master is None:
        master = pd.DataFrame()
    master = _ensure_pid_col(master)
    for c in ["Nombre", "Proveedor", "Categoria"]:
        if c not in master.columns:
            master[c] = ""
    master = master[["Product_ID", "Nombre", "Proveedor", "Categoria"]].drop_duplicates("Product_ID")

    # ---------- helper para convertir a float precios que puedan venir con coma/miles ----------
    def _clean_price(series: pd.Series) -> pd.Series:
        s = (
            series.astype(str)
                  .str.replace(r"[^\d,.\-]", "", regex=True)   # deja dígitos, coma, punto y signo
                  .str.replace(".", "", regex=False)           # elimina miles "1.234,56"
                  .str.replace(",", ".", regex=False)          # coma decimal -> punto
        )
        return pd.to_numeric(s, errors="coerce")

    # ===== Precios preferentes (ui_products.parquet)
    ui_prod_path = TMP_VISTAS / "ui_products.parquet"
    up = pd.DataFrame(columns=["Product_ID", "Precio"])
    if ui_prod_path.exists():
        try:
            tmp = pd.read_parquet(ui_prod_path)
            tmp = _ensure_pid_col(tmp)
            low = {c.lower().strip(): c for c in tmp.columns}
            prc = (low.get("preferred_price") or low.get("price") or
                   low.get("precio") or low.get("precio_medio"))
            if prc:
                up = tmp.rename(columns={prc: "Precio"})[["Product_ID", "Precio"]].copy()
                up["Precio"] = _clean_price(up["Precio"])
        except Exception:
            pass

    # ===== Fallback precios desde el catálogo enriquecido (¡sin recortar!)
    #     Ojo: NO usamos load_catalog_items(), porque recorta columnas y pierde el precio.
    from pathlib import Path
    p = CAT if CAT.exists() else (ROOT / "data" / "processed" / "catalog_items_enriquecido.csv")
    cat_prc = pd.DataFrame(columns=["Product_ID", "Precio"])
    if p.exists():
        try:
            cat_raw = _read_csv_smart(p)   # mantiene todas las columnas originales
            cat_raw = _ensure_pid_col(cat_raw)
            low = {c.lower().strip(): c for c in cat_raw.columns}
            prc_c = low.get("precio_medio") or low.get("precio") or low.get("price")
            if prc_c:
                tmp = cat_raw.rename(columns={prc_c: "Precio"})[["Product_ID", "Precio"]].copy()
                tmp["Precio"] = _clean_price(tmp["Precio"])
                cat_prc = tmp
        except Exception:
            pass

    # ===== Merge y preferencia (preferred_price > catálogo)
    left = master.copy()
    if not up.empty:
        left = left.merge(up, on="Product_ID", how="left")
    else:
        left["Precio"] = pd.NA

    if not cat_prc.empty:
        left = left.merge(cat_prc.rename(columns={"Precio": "Precio_cat"}),
                          on="Product_ID", how="left")

    left["Precio"] = pd.to_numeric(left["Precio"], errors="coerce")
    left["Precio_cat"] = pd.to_numeric(left.get("Precio_cat"), errors="coerce")
    left["Precio"] = left["Precio"].fillna(left["Precio_cat"])
    left.drop(columns=[c for c in left.columns if c.endswith("_cat")],
              inplace=True, errors="ignore")

    out = left[["Product_ID", "Nombre", "Proveedor", "Categoria", "Precio"]].drop_duplicates("Product_ID")
    out["Product_ID"] = out["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    return out


def _cat_table_by_supplier(supplier: str) -> pd.DataFrame:
    """
    Tabla Catálogo por proveedor: Product_ID, Nombre, Categoria, Stock actual.
    (Precio oculto temporalmente para evitar valores corruptos)
    """
    pm = _price_map()  # lo seguimos usando para Nombre/Proveedor/Categoria
    inv = _read_working_inventory().rename(columns={"Stock Real": "Stock actual"})

    base = pm.merge(inv[["Product_ID", "Stock actual"]], on="Product_ID", how="left")
    base["Stock actual"] = base["Stock actual"].fillna(0).astype(int)

    if supplier and supplier != "Todos":
        base = base[base["Proveedor"] == supplier]

    # Solo mostramos columnas sin precio
    keep = ["Product_ID", "Nombre", "Categoria", "Stock actual"]
    for c in keep:
        if c not in base.columns:
            base[c] = "" if c != "Stock actual" else 0

    return base[keep].sort_values("Nombre", na_position="last").reset_index(drop=True)

def _stats_by_supplier(df_qty: pd.DataFrame, pm: pd.DataFrame,
                       start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    df_qty: Product_ID, Date, qty (hist + forecast ya concatenado y filtrado por fechas)
    pm:    Product_ID, Proveedor, ...
    Devuelve:
      - rank_sup: proveedor, qty_total
      - series_sup: Date, Proveedor, qty (para línea temporal)
    """
    if df_qty.empty:
        return (pd.DataFrame(columns=["Proveedor", "qty_total"]),
                pd.DataFrame(columns=["Date", "Proveedor", "qty"]))

    df = df_qty[(df_qty["Date"] >= start) & (df_qty["Date"] <= end)].copy()
    df = df.merge(pm[["Product_ID", "Proveedor"]], on="Product_ID", how="left")

    rank = (df.groupby("Proveedor", as_index=False)
              .agg(qty_total=("qty", "sum"))
              .sort_values(["qty_total"], ascending=False))

    serie = (df.groupby([pd.Grouper(key="Date", freq="MS"), "Proveedor"], as_index=False)
               .agg(qty=("qty", "sum"))
               .sort_values(["Date", "Proveedor"]))
    return rank, serie
    
def _fmt_unidades_ceil(s: pd.Series) -> pd.Series:
    """
    Redondea al alza y aplica separador de miles (estilo 12.345).
    Solo para mostrar en tablas (no usar en cálculos).
    """
    v = pd.to_numeric(s, errors="coerce").fillna(0)
    v = np.ceil(v).astype("Int64")
    # 12,345 -> 12.345
    return v.map(lambda x: f"{x:,}".replace(",", ".") if pd.notna(x) else "")

# ==================== OC helpers ====================

OC_HDR = OUT10 / "ordenes_compra.csv"
OC_LIN = OUT10 / "ordenes_compra_lineas.csv"

def _oc_read_hdr() -> pd.DataFrame:
    exp = ["order_id","Proveedor","Fecha","Escenario","Estado","ETA_dias"]
    if OC_HDR.exists():
        try:
            df = pd.read_csv(OC_HDR)
        except Exception:
            df = pd.DataFrame(columns=exp)
    else:
        df = pd.DataFrame(columns=exp)

    # Normaliza columnas y rellena faltantes
    low = {c.lower(): c for c in df.columns}
    ren = {}
    if "order_id" not in df.columns:
        # alias muy comunes
        if "id" in low: ren[low["id"]] = "order_id"
    if "proveedor" not in df.columns and ("supplier" in low or "scm_supplier_id" in low):
        ren[low.get("supplier", low.get("scm_supplier_id"))] = "Proveedor"
    if "fecha" not in df.columns and "date" in low:
        ren[low["date"]] = "Fecha"
    if ren:
        df = df.rename(columns=ren)

    for c in exp:
        if c not in df.columns:
            df[c] = "" if c not in ("ETA_dias","Estado") else (0 if c=="ETA_dias" else "en_curso")

    # Tipos suaves
    df["Fecha"] = df["Fecha"].astype(str).str[:10]
    df["ETA_dias"] = pd.to_numeric(df["ETA_dias"], errors="coerce").fillna(0).astype(int)
    df["Estado"] = df["Estado"].replace("", "en_curso").fillna("en_curso")

    return df[exp]

def _oc_read_lin() -> pd.DataFrame:
    if OC_LIN.exists():
        return pd.read_csv(OC_LIN, dtype={"Product_ID": str})
    return pd.DataFrame(columns=["order_id","Product_ID","Nombre","Cantidad_pedir"])

def _oc_write_hdr(df: pd.DataFrame) -> None:
    df.to_csv(OC_HDR, index=False, encoding="utf-8")

def _oc_write_lin(df: pd.DataFrame) -> None:
    df.to_csv(OC_LIN, index=False, encoding="utf-8")


def _make_order_id(proveedor: str) -> str:
    ts = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    base = str(proveedor).strip().replace(" ", "_")[:16]
    return f"OC-{base}-{ts}"



    # --- marcar cabecera como recibida
    if "Estado" not in hdr.columns:
        hdr["Estado"] = "en_curso"
    hdr.loc[hdr["order_id"] == order_id, "Estado"] = "recibida"
    _oc_write_hdr(hdr)

    return True

    # --- normalizaciones básicas
def repair_ledger_canonic():
    """
    Limpia OUT10/ledger_movimientos.csv al esquema canónico:
    Date, Product_ID, Nombre, Proveedor, Tipo movimiento, qty_pedido

    - Recalcula qty_pedido si viene mal (usa |delta| o |on_prev - on_new| como fallback)
    - Normaliza 'Tipo movimiento' (pone 'Venta' si venían números o vacío)
    - Reconstituye Nombre/Proveedor desde el maestro
    """
    ledger_path = OUT10 / "ledger_movimientos.csv"
    if not ledger_path.exists():
        return

    df = pd.read_csv(ledger_path)
    if df.empty:
        return

    # --- normalizaciones básicas
    df["Product_ID"] = (
        df.get("Product_ID", "").astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    )
    df["Date"] = pd.to_datetime(df.get("Date", pd.NaT), errors="coerce").dt.normalize()

    # qty_pedido: intentar arreglar cuando esté vacío, NaN o <= 0
    if "qty_pedido" not in df.columns:
        df["qty_pedido"] = pd.NA
    df["qty_pedido"] = pd.to_numeric(df["qty_pedido"], errors="coerce")

    mask_bad_qty = df["qty_pedido"].isna() | (df["qty_pedido"] <= 0)

    # 1º: usar |delta|
    if "delta" in df.columns:
        delta_fix = pd.to_numeric(df["delta"], errors="coerce").abs()
        df.loc[mask_bad_qty & delta_fix.notna(), "qty_pedido"] = delta_fix

    # 2º: si no hay delta, usar on_prev - on_new
    if ("on_prev" in df.columns) and ("on_new" in df.columns):
        prev = pd.to_numeric(df["on_prev"], errors="coerce")
        new  = pd.to_numeric(df["on_new"], errors="coerce")
        diff = (prev - new).abs()
        df.loc[df["qty_pedido"].isna() | (df["qty_pedido"] <= 0), "qty_pedido"] = diff

    # Tipo movimiento: si venían números o vacío → "Venta"
    if "Tipo movimiento" not in df.columns:
        df["Tipo movimiento"] = "Venta"
    tm_str = df["Tipo movimiento"].astype(str).str.strip()
    bad_tm = tm_str.str.fullmatch(r"-?\d+(\.\d+)?")
    df.loc[bad_tm.fillna(True), "Tipo movimiento"] = "Venta"

    # Añadir Nombre/Proveedor desde el maestro
    master = build_product_master(ROOT)[["Product_ID", "Nombre", "Proveedor"]].copy()
    master["Product_ID"] = master["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    df = df.merge(master, on="Product_ID", how="left", suffixes=("", "_m"))

    # Si existían columnas previas con Nombre/Proveedor numéricos, prioriza las del maestro
    for col in ["Nombre", "Proveedor"]:
        mcol = f"{col}_m"
        if mcol in df.columns:
            df[col] = df[mcol].where(df[mcol].notna() & (df[mcol] != ""), df.get(col))
            df.drop(columns=[mcol], inplace=True, errors="ignore")

    # Dejar sólo el canónico y tipos correctos
    keep = ["Date", "Product_ID", "Nombre", "Proveedor", "Tipo movimiento", "qty_pedido"]
    for c in keep:
        if c not in df.columns:
            df[c] = "" if c not in ("Date", "qty_pedido") else (pd.NaT if c == "Date" else 0)

    df = df[keep].copy()
    df["qty_pedido"] = pd.to_numeric(df["qty_pedido"], errors="coerce").fillna(0).astype(int)
    df = df[(df["qty_pedido"] > 0) & df["Product_ID"].str.isnumeric()]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    df = df.sort_values(["Date", "Product_ID"]).reset_index(drop=True)
    df.to_csv(ledger_path, index=False, encoding="utf-8")

# --- Normalizador MULTI: garantiza columnas para construir_vistas ---
def _normalize_multi_for_vistas(src: Path) -> Path:
    if not src.exists():
        return src
    df = _read_csv_smart(src)
    if df is None or df.empty:
        return src

    low = {c.lower(): c for c in df.columns}
    ren = {}

    # IDs
    for cand in ["product_id", "item_id", "id_producto"]:
        if cand in low: ren[low[cand]] = "Product_ID"; break
    for cand in ["supplier_id", "id_proveedor", "proveedor"]:
        if cand in low: ren[low[cand]] = "supplier_id"; break

    # Métricas
    if "precio" not in low and "price" in low: ren[low["price"]] = "precio"
    if "lead_time" not in low and "lt" in low: ren[low["lt"]] = "lead_time"
    if "disponibilidad" not in low:
        for cand in ["availability", "stock", "on_hand"]:
            if cand in low: ren[low[cand]] = "disponibilidad"; break
    for cand in ["moq", "min_qty", "minimo_pedido"]:
        if cand in low: ren[low[cand]] = "moq"; break
    for cand in ["multiplo", "multiple", "pack_size"]:
        if cand in low: ren[low[cand]] = "multiplo"; break

    df = df.rename(columns=ren)
    out = TMP_VISTAS / "supplier_catalog_multi_norm.csv"
    df.to_csv(out, index=False)
    return out

# --- Normalizador UNIFICADO (lo puedes dejar tal cual si ya lo tienes) ---
def _normalize_unif_for_vistas(p: Path) -> Path:
    df = _read_csv_smart(p)
    if df is None or df.empty:
        return p
    low = {c.lower(): c for c in df.columns}
    ren = {}
    for cand in ["product_id", "item_id", "id_producto"]:
        if cand in low: ren[low[cand]] = "Product_ID"; break
    for cand in ["substitute_product_id", "alt_product_id", "sustituto_id", "id_sustituto"]:
        if cand in low: ren[low[cand]] = "Substitute_Product_ID"; break
    if "tipo" not in low and "type" in low: ren[low["type"]] = "tipo"
    df = df.rename(columns=ren)
    if "tipo" not in df.columns: df["tipo"] = "externo"
    if "Product_ID" in df.columns: df["Product_ID"] = _to_pid_str(df["Product_ID"])
    if "Substitute_Product_ID" in df.columns: df["Substitute_Product_ID"] = _to_pid_str(df["Substitute_Product_ID"])
    out = TMP_VISTAS / "substitutes_unified_norm.csv"
    df.to_csv(out, index=False)
    return out

# ========= HELPERS para la pestaña "Sustitutos por producto" =========

def enrich_external_subs(dfs_ext: pd.DataFrame, dfp: pd.DataFrame, cat: pd.DataFrame | None) -> pd.DataFrame:
    """
    Enriquecer sustitutos EXTERNOS con nombre/categoría/proveedor y métricas si existen.
    dfp = ui_products (para pillar precio/lt del preferente si vienen),
    cat = catálogo (nombre/categoría).
    Devuelve columnas estándar usadas en la tabla de detalle.
    """
    if dfs_ext is None or dfs_ext.empty:
        return pd.DataFrame(columns=[
            "tipo", "Substitute_Product_ID", "nombre", "categoria", "proveedor",
            "rank", "score", "precio", "lead_time", "disponibilidad"
        ])

    df = dfs_ext.copy()

    # Normalizar nombres esperados por si vienen con otros alias
    low = {c.lower(): c for c in df.columns}
    if "substitute_product_id" not in low and "Substitute_Product_ID" in df.columns:
        pass
    elif "substitute_product_id" in low and low["substitute_product_id"] != "Substitute_Product_ID":
        df = df.rename(columns={low["substitute_product_id"]: "Substitute_Product_ID"})
    if "tipo" not in df.columns:
        df["tipo"] = "externo"

    # Añadir nombre/categoría desde catálogo si está
    if cat is not None and not cat.empty:
        cc = cat.copy()
        # asegurar columnas estándar
        pid = _find_col(cc, ["Product_ID","product_id","item_id","id_producto"]) or "Product_ID"
        nom = _find_col(cc, ["Nombre","name","nombre"]) or "Nombre"
        catg = _find_col(cc, ["Categoria","Categoría","category","categoria"]) or "Categoria"
        ren = {}
        if pid != "Product_ID": ren[pid] = "Product_ID"
        if nom != "Nombre":     ren[nom] = "Nombre"
        if catg != "Categoria": ren[catg] = "Categoria"
        if ren: cc = cc.rename(columns=ren)
        cc["Product_ID"] = _to_pid_str(cc["Product_ID"])

        df = df.merge(
            cc[["Product_ID","Nombre","Categoria"]].rename(columns={
                "Product_ID": "Substitute_Product_ID",
                "Nombre": "nombre",
                "Categoria": "categoria"
            }),
            on="Substitute_Product_ID",
            how="left"
        )
    else:
        if "nombre" not in df.columns:   df["nombre"] = ""
        if "categoria" not in df.columns: df["categoria"] = ""

    # Proveedor / precio / LT / disponibilidad si existen
    prov_col = _find_col(df, ["proveedor","supplier","supplier_id"])
    if not prov_col:
        df["proveedor"] = ""
    else:
        if prov_col != "proveedor":
            df = df.rename(columns={prov_col: "proveedor"})

    # Alinear métricas principales (si vinieron con otros nombres)
    rank_col  = _find_col(df, ["rank"])
    score_col = _find_col(df, ["score"])
    price_col = _find_col(df, ["precio","price"])
    lt_col    = _find_col(df, ["lead_time","lt"])
    disp_col  = _find_col(df, ["disponibilidad","availability","stock"])

    if rank_col  and rank_col  != "rank":           df = df.rename(columns={rank_col:"rank"})
    if score_col and score_col != "score":          df = df.rename(columns={score_col:"score"})
    if price_col and price_col != "precio":         df = df.rename(columns={price_col:"precio"})
    if lt_col    and lt_col    != "lead_time":      df = df.rename(columns={lt_col:"lead_time"})
    if disp_col  and disp_col  != "disponibilidad": df = df.rename(columns={disp_col:"disponibilidad"})

    for c in ["rank","score","precio","lead_time","disponibilidad"]:
        if c not in df.columns: df[c] = pd.NA

    cols = ["tipo","Substitute_Product_ID","nombre","categoria","proveedor","rank","score","precio","lead_time","disponibilidad"]
    return df[cols]


def build_internal_subs(pid: str, scm: pd.DataFrame | None, dfp: pd.DataFrame, cat: pd.DataFrame | None) -> pd.DataFrame:
    """
    Construye sustitutos INTERNOS = otros proveedores para el MISMO Product_ID.
    - `scm` = supplier_catalog_multi normalizado (con columnas Product_ID, scm_supplier_id, scm_price, scm_lead_time, scm_availability).
    - Devuelve un dataframe con las mismas columnas que 'enrich_external_subs' para poder concatenar.
    """
    if scm is None or scm.empty:
        return pd.DataFrame(columns=[
            "tipo", "Substitute_Product_ID", "nombre", "categoria", "proveedor",
            "rank", "score", "precio", "lead_time", "disponibilidad"
        ])

    # Filtrar el multi-catálogo por el producto seleccionado
    mm = scm.copy()
    mm["Product_ID"] = _to_pid_str(mm["Product_ID"])
    mm = mm[mm["Product_ID"] == str(pid)]
    if mm.empty:
        return pd.DataFrame(columns=[
            "tipo", "Substitute_Product_ID", "nombre", "categoria", "proveedor",
            "rank", "score", "precio", "lead_time", "disponibilidad"
        ])

    # Columnas esperadas
    low = {c.lower(): c for c in mm.columns}
    sup_col = low.get("scm_supplier_id") or low.get("supplier_id") or low.get("proveedor")
    prc_col = low.get("scm_price") or low.get("price") or low.get("precio")
    lt_col  = low.get("scm_lead_time") or low.get("lead_time") or low.get("lt")
    av_col  = low.get("scm_availability") or low.get("availability") or low.get("stock") or low.get("disponibilidad")

    # Nombre/Categoría del producto original
    nombre = ""
    categoria = ""
    if cat is not None and not cat.empty:
        cc = cat.copy()
        pidc = _find_col(cc, ["Product_ID","product_id","item_id","id_producto"]) or "Product_ID"
        nomc = _find_col(cc, ["Nombre","name","nombre"]) or "Nombre"
        catc = _find_col(cc, ["Categoria","Categoría","category","categoria"]) or "Categoria"
        if pidc != "Product_ID": cc = cc.rename(columns={pidc:"Product_ID"})
        if nomc != "Nombre":     cc = cc.rename(columns={nomc:"Nombre"})
        if catc != "Categoria":  cc = cc.rename(columns={catc:"Categoria"})
        cc["Product_ID"] = _to_pid_str(cc["Product_ID"])
        row = cc[cc["Product_ID"] == str(pid)].head(1)
        if not row.empty:
            nombre = str(row.iloc[0].get("Nombre", ""))
            categoria = str(row.iloc[0].get("Categoria", ""))

    # Construir filas
    out = pd.DataFrame({
        "tipo": "interno",
        "Substitute_Product_ID": str(pid),
        "nombre": nombre,
        "categoria": categoria,
        "proveedor": mm[sup_col] if sup_col else "",
        "rank": range(1, len(mm) + 1),
        "score": pd.NA,  # no hay score para internos
        "precio": pd.to_numeric(mm[prc_col], errors="coerce") if prc_col else pd.NA,
        "lead_time": pd.to_numeric(mm[lt_col], errors="coerce") if lt_col else pd.NA,
        "disponibilidad": pd.to_numeric(mm[av_col], errors="coerce") if av_col else pd.NA,
    })

    cols = ["tipo","Substitute_Product_ID","nombre","categoria","proveedor","rank","score","precio","lead_time","disponibilidad"]
    return out[cols]

# ================================

def _norm_inv_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza cualquier inventario a:
      - Product_ID (string)
      - on_hand   (int)
    Con heurística si no hay cabeceras estándar.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Product_ID", "on_hand"])

    low = {c.lower(): c for c in df.columns}

    # 1) candidatos directos
    pid = (low.get("product_id") or low.get("item_id") or low.get("sku") or
           low.get("id_producto") or low.get("producto_id") or low.get("id"))
    stock_candidates = [
        "on_hand","stock_actual","onhand","on_hand_qty","qty_on_hand",
        "stock","stock_qty","stock_quantity","existencias","quantity","qty","on_new","cantidad"
    ]
    onh = next((low[c] for c in stock_candidates if c in low), None)

    # 2) heurística si faltan
    def _is_numericish(s: pd.Series) -> bool:
        num = pd.to_numeric(s, errors="coerce")
        return (num.notna().mean() >= 0.8)

    if pid is None:
        for c in df.columns:
            if _is_numericish(df[c]):
                pid = c; break
        if pid is None:
            pid = df.columns[0]

    if onh is None:
        for c in df.columns:
            cl = c.lower()
            if ("stock" in cl or "exist" in cl or "hand" in cl or "qty" in cl or "cant" in cl) and _is_numericish(df[c]):
                onh = c; break
        if onh is None:
            num_cols = [c for c in df.columns if _is_numericish(df[c])]
            if num_cols: onh = num_cols[-1]

    # 3) renombrar y tipar
    if pid != "Product_ID": df = df.rename(columns={pid: "Product_ID"})
    if onh and onh != "on_hand": df = df.rename(columns={onh: "on_hand"})

    df["Product_ID"] = (
        df["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    )
    if "on_hand" in df.columns:
        df["on_hand"] = pd.to_numeric(df["on_hand"], errors="coerce").fillna(0).astype(int)
    else:
        df["on_hand"] = 0

    return df[["Product_ID", "on_hand"]]

def _normalize_unif_for_vistas(p: Path) -> Path:
    """Asegura columnas esperadas para construir_vistas (y devuelve ruta al tmp)."""
    df = _read_csv_smart(p)
    if df is None or df.empty:
        return p

    # Mapear columnas flexibles
    low = {c.lower(): c for c in df.columns}
    ren = {}
    pid_cands = ["Product_ID", "product_id"]
    sub_cands = ["Substitute_Product_ID", "substitute_product_id", "alt_product_id"]
    for cand in pid_cands:
        if cand in low:
            ren[low[cand]] = "Product_ID"
            break
    for cand in sub_cands:
        if cand in low:
            ren[low[cand]] = "Substitute_Product_ID"
            break

    # Tipo (externo/interno)
    if "tipo" not in low and "type" in low:
        ren[low["type"]] = "tipo"

    df = df.rename(columns=ren)

    # Valores por defecto y tipos
    if "tipo" not in df.columns:
        df["tipo"] = "externo"
    if "Product_ID" in df.columns:
        df["Product_ID"] = _to_pid_str(df["Product_ID"])
    if "Substitute_Product_ID" in df.columns:
        df["Substitute_Product_ID"] = _to_pid_str(df["Substitute_Product_ID"])

    out = TMP_VISTAS / "substitutes_unified_norm.csv"
    df.to_csv(out, index=False)
    return out

@st.cache_data(ttl=60)
def load_views(min_score: float, m_unif: float, m_multi: float, m_cat: float, m_stock: float):
    # Rutas normalizadas
    unif_path  = _normalize_unif_for_vistas(UNIF)  if UNIF.exists()  else None
    multi_path = _normalize_multi_for_vistas(MULTI) if MULTI.exists() else None
    cat_path   = CAT   if CAT.exists()   else None
    stock_path = STOCK if STOCK.exists() else None

    # Intento 1: firma NUEVA
    try:
        paths = construir_vistas(
            substitutes_unified=unif_path,
            supplier_catalog_multi=multi_path,
            catalog_items=cat_path,
            stock_positions=stock_path,
            outdir=TMP_VISTAS,
            min_score=min_score,
            m_unif=m_unif, m_multi=m_multi, m_cat=m_cat, m_stock=m_stock
        )
    except TypeError:
        # Intento 2: firma ANTIGUA (la que tenías en el script “fino”)
        paths = construir_vistas(
            path_unificado=unif_path,
            path_multi=multi_path,
            path_catalogo=cat_path,
            path_stock=stock_path,
            path_consumo=None,
            min_score=min_score,
        )
    except Exception as e:
        st.error("Error al construir las vistas.")
        st.exception(e)
        return {"ui_products": pd.DataFrame(), "ui_substitutes": pd.DataFrame()}

    def _maybe(p) -> pd.DataFrame:
        # Si ya es un DataFrame, lo devolvemos tal cual
        if isinstance(p, pd.DataFrame):
            return p.copy()

        # Si es una ruta (str o Path), la leemos
        if isinstance(p, (str, Path)):
            try:
                pth = Path(p)
                if pth.exists():
                    return pd.read_parquet(pth) if pth.suffix.lower() == ".parquet" else pd.read_csv(pth)
            except Exception:
                # como último intento, CSV
                try:
                    return pd.read_csv(p)
                except Exception:
                    pass

        # Cualquier otro caso → DataFrame vacío
        return pd.DataFrame()

    up = _maybe(paths.get("ui_products", TMP_VISTAS / "ui_products.parquet"))
    us = _maybe(paths.get("ui_substitutes", TMP_VISTAS / "ui_substitutes.csv"))
    return {"ui_products": up, "ui_substitutes": us}

def _norm_ui_products(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    low = {c.lower(): c for c in df.columns}
    ren = {}
    for src, dst in [
        ("product_id", "Product_ID"),
        ("on_hand", "on_hand"),
        ("preferred_supplier_id", "preferred_supplier_id"),
        ("preferred_price", "preferred_price"),
        ("preferred_lead_time", "preferred_lead_time"),
        ("subs_internos_count", "subs_internos_count"),
        ("subs_externos_count", "subs_externos_count"),
        ("nombre", "up_nombre"),
        ("categoria", "up_categoria"),
        ("proveedor", "up_proveedor"),
    ]:
        if src in low and low[src] != dst:
            ren[low[src]] = dst
    df = df.rename(columns=ren)
    if "Product_ID" in df.columns:
        df["Product_ID"] = _to_pid_str(df["Product_ID"])
    return df

def _norm_ui_substitutes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    low = {c.lower(): c for c in df.columns}
    ren = {}
    for src, dst in [
        ("product_id", "Product_ID"),
        ("tipo", "tipo"),
        ("substitute_product_id", "Substitute_Product_ID"),
        ("rank", "rank"),
        ("score", "score"),
        ("nombre", "Nombre"),
        ("categoria", "Categoría"),
        ("proveedor", "Proveedor"),
        ("precio", "Precio"),
        ("lead_time", "Lead time"),
        ("disponibilidad", "Disponibilidad"),
    ]:
        if src in low and low[src] != dst:
            ren[low[src]] = dst
    df = df.rename(columns=ren)
    if "Product_ID" in df.columns:
        df["Product_ID"] = _to_pid_str(df["Product_ID"])
    if "Substitute_Product_ID" in df.columns:
        df["Substitute_Product_ID"] = _to_pid_str(df["Substitute_Product_ID"])
    return df

def load_catalog_items(root: Path) -> pd.DataFrame | None:
    p = CAT if CAT.exists() else (root / "data" / "processed" / "catalog_items_enriquecido.csv")
    df = _read_csv_smart(p) if p.exists() else None
    if df is None or df.empty:
        return None
    pid = _find_col(df, ["Product_ID", "product_id"])
    nom = _find_col(df, ["Nombre","nombre","name"])
    cat = _find_col(df, ["Categoria","Categoría","categoria","category"])
    ren = {}
    if pid and pid != "Product_ID": ren[pid] = "Product_ID"
    if nom and nom != "Nombre": ren[nom] = "Nombre"
    if cat and cat != "Categoria": ren[cat] = "Categoria"
    df = df.rename(columns=ren)
    df["Product_ID"] = _to_pid_str(df["Product_ID"])
    return df[["Product_ID","Nombre","Categoria"]].drop_duplicates("Product_ID")

def load_substitutes_unified(root: Path) -> pd.DataFrame | None:
    p = UNIF if UNIF.exists() else (root / "data" / "processed" / "substitutes_unified.csv")
    df = _read_csv_smart(p) if p.exists() else None
    if df is None or df.empty:
        return None
    pid = _find_col(df, ["Product_ID","product_id"])
    sub = _find_col(df, ["Substitute_Product_ID","substitute_product_id","alt_product_id"])
    typ = _find_col(df, ["tipo","type"])
    if not pid or not sub:
        return None
    df = df.rename(columns={pid:"Product_ID", sub:"Substitute_Product_ID"})
    if typ: df = df.rename(columns={typ:"tipo"})
    df["Product_ID"] = _to_pid_str(df["Product_ID"])
    df["Substitute_Product_ID"] = _to_pid_str(df["Substitute_Product_ID"])
    if "tipo" not in df.columns:
        df["tipo"] = "externo"
    return df

def load_supplier_catalog_multi(root: Path) -> pd.DataFrame | None:
    p = MULTI if MULTI.exists() else (root / "data" / "clean" / "supplier_catalog_multi.csv")
    df = _read_csv_smart(p) if p.exists() else None
    if df is None or df.empty:
        return None
    pid  = _find_col(df, ["Product_ID","product_id"])
    sup  = _find_col(df, ["supplier_id","id_proveedor","proveedor"])
    prc  = _find_col(df, ["price","precio"])
    lt   = _find_col(df, ["lead_time","lt"])
    av   = _find_col(df, ["availability","stock","disponibilidad"])
    ren = {}
    if pid and pid != "Product_ID": ren[pid] = "Product_ID"
    if sup and sup != "scm_supplier_id": ren[sup] = "scm_supplier_id"
    if prc and prc != "scm_price": ren[prc] = "scm_price"
    if lt and lt != "scm_lead_time": ren[lt] = "scm_lead_time"
    if av and av != "scm_availability": ren[av] = "scm_availability"
    df = df.rename(columns=ren)
    df["Product_ID"] = _to_pid_str(df["Product_ID"])
    return df

# --------------------------- PREPARACIÓN PEDIDOS PARA EL PROCESADOR ---------------------------
def _ensure_orders_ui():
    ORD_UI.parent.mkdir(parents=True, exist_ok=True)
    if not ORD_UI.exists():
        pd.DataFrame(columns=["date","Product_ID","qty"]).to_csv(ORD_UI, index=False)

def _prepare_orders_for_ms(include_ae: bool = True) -> Path | None:
    """
    Prepara el archivo de pedidos para el procesador.
    - **Archivo visible** para ti:       data/raw/orders_for_ms.csv (con columnas: date, Product_ID, qty)
    - **Copia para el procesador (ms)**: data/raw/orders_for_ms__ms.csv (con columnas: date, item_id, qty)
    Devuelve la ruta al archivo **visible** (Product_ID).
    """
    _ensure_orders_ui()

    try:
        df = pd.read_csv(ORD_UI)
    except Exception:
        return None

    # Normalizar columnas
    low = {c.lower(): c for c in df.columns}
    if "product_id" in low and low["product_id"] != "Product_ID":
        df = df.rename(columns={low["product_id"]: "Product_ID"})
    if "item_id" in low and "Product_ID" not in df.columns:
        df = df.rename(columns={low["item_id"]: "Product_ID"})
    if "qty" not in df.columns and "quantity" in low:
        df = df.rename(columns={low["quantity"]: "qty"})
    if "qty" not in df.columns and "cantidad" in low:
        df = df.rename(columns={low["cantidad"]: "qty"})
    if "date" not in df.columns and "fecha" in low:
        df = df.rename(columns={low["fecha"]: "date"})

    # Tipado básico
    if "Product_ID" in df.columns:
        df["Product_ID"] = df["Product_ID"].astype(str).str.strip()
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Mantener solo columnas esperadas
    need = {"date", "Product_ID", "qty"}
    if not need.issubset(df.columns):
        return None
    df_final = df[list(need)].copy()

    # Fusión con escenarios A–E (si aplica)
    if include_ae and ORD_AE.exists():
        try:
            ae = pd.read_csv(ORD_AE)
            ae = _ensure_pid_col(ae)
            low = {c.lower(): c for c in ae.columns}
            if "qty" not in low and "cantidad" in low:
                ae = ae.rename(columns={low["cantidad"]: "qty"})
            if "date" not in low and "fecha" in low:
                ae = ae.rename(columns={low["fecha"]: "date"})
            keep = [c for c in ["date","Product_ID","qty"] if c in ae.columns]
            ae = ae[keep].copy()
            ae["qty"] = pd.to_numeric(ae["qty"], errors="coerce").fillna(0).astype(int)
            ae["date"] = pd.to_datetime(ae["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df_final = pd.concat([df_final, ae], ignore_index=True)
        except Exception:
            pass

    # Guardar el **visible** (Product_ID)
    orders_for_ui = RAW / "orders_for_ms.csv"
    df_final.to_csv(orders_for_ui, index=False)

    # Crear **copia para el procesador** con item_id
    df_ms = df_final.rename(columns={"Product_ID": "item_id"})
    orders_for_ms = RAW / "orders_for_ms__ms.csv"
    df_ms.to_csv(orders_for_ms, index=False)

    return orders_for_ui

# ========= Generador de escenarios A–E (aleatorios y configurables) =========
import random
import numpy as np

def _generate_scenarios_AE(
    n_orders: int = 5,
    lines_per_order: int = 5,
    qty_min: int = 1,
    qty_max: int = 5,
    days_back: int = 7,
    allow_repeats_in_order: bool = False,
    sampling_mode: str = "Uniforme",  # "Uniforme" | "Ponderado por stock"
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Genera un DataFrame con columnas: date, Product_ID, qty
    y lo guarda en RAW/customer_orders_AE.csv
    """
    # Semillas (reproducibilidad opcional)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Candidatos desde el inventario vivo
    inv = _read_working_inventory().copy()
    inv = inv.dropna(subset=["Product_ID"])
    inv["Product_ID"] = _to_pid_str(inv["Product_ID"])
    inv = inv[inv["Product_ID"] != ""]
    if inv.empty:
        raise ValueError("Inventario vacío: no hay candidatos para escenarios A–E.")

    candidates = inv["Product_ID"].tolist()

    # Pesos (solo si ponderado por stock)
    weights = None
    if sampling_mode == "Ponderado por stock" and "Stock Real" in inv.columns:
        w = pd.to_numeric(inv["Stock Real"], errors="coerce").fillna(0)
        # Cortar negativos a 0 (si permites stock negativo)
        w = w.clip(lower=0)
        # Si todos quedan 0, volver a uniforme
        if w.sum() > 0:
            weights = (w / w.sum()).values  # np.array sum=1.0

    rows = []
    today = pd.Timestamp.today().normalize()

    # Normalizar parámetros
    n_orders = max(1, int(n_orders))
    lines_per_order = max(1, int(lines_per_order))
    qty_min = max(1, int(qty_min))
    qty_max = max(qty_min, int(qty_max))
    days_back = max(0, int(days_back))

    for _ in range(n_orders):
        # Fecha aleatoria en los últimos 'days_back' días
        offset = 0 if days_back == 0 else random.randint(0, days_back)
        date_str = (today - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")

        k = lines_per_order

        if allow_repeats_in_order:
            # Muestreo CON reemplazo
            if weights is not None and len(weights) == len(candidates):
                product_ids = list(np.random.choice(candidates, size=k, replace=True, p=weights))
            else:
                product_ids = [random.choice(candidates) for _ in range(k)]
        else:
            # Muestreo SIN reemplazo
            kk = min(k, len(candidates))
            if weights is not None and len(weights) == len(candidates):
                # np.random.choice permite replace=False con probabilidades
                product_ids = list(np.random.choice(candidates, size=kk, replace=False, p=weights))
            else:
                product_ids = random.sample(candidates, kk)

        # Cantidades
        qtys = [random.randint(qty_min, qty_max) for _ in range(len(product_ids))]

        rows.extend(
            {"date": date_str, "Product_ID": pid, "qty": q}
            for pid, q in zip(product_ids, qtys)
        )

    df = pd.DataFrame(rows, columns=["date", "Product_ID", "qty"])
    df["Product_ID"] = _to_pid_str(df["Product_ID"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)

    # Guardar A–E
    ORD_AE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ORD_AE, index=False)
    return df


# ==================================================
# 2). Portada
# ==================================================
def render_home():
    st.title("🧭 PFM2 — Asistente de Compras")
    st.caption("Elige un bloque para trabajar.")

    cols = st.columns(4)
    with cols[0]:
        st.subheader("🔎 Exploración & Sustitutos")
        st.write("Explora catálogo, productos y sustitutos internos/externos.")
        st.button("Entrar", use_container_width=True, on_click=goto, args=("exploracion",), key="go_exploracion")
    with cols[1]:
        st.subheader("🏭 Proveedores")
        st.write("Catálogo por proveedor, condiciones y cobertura.")
        st.button("Entrar", use_container_width=True, on_click=goto, args=("proveedores",), key="go_proveedores")
    with cols[2]:
        st.subheader("📦 Movimientos de stock")
        st.write("Entradas, salidas, transferencias y stock neto.")
        st.button("Entrar", use_container_width=True, on_click=goto, args=("movimientos",), key="go_movimientos")
    with cols[3]:
        st.subheader("🧾 Reapro / Pedidos")
        st.write("ROP, safety stock, cantidad recomendada y pedidos.")
        st.button("Entrar", use_container_width=True, on_click=goto, args=("reapro",), key="go_reapro")

    st.divider()
    st.markdown("Consejo: usa el **menú lateral** para saltar entre bloques sin volver a la portada.")



# ==================================================
# 4) BLOQUE DE PRODUCTOS Y SUSTITUTOS.
# ==================================================
def render_exploracion_sustitutos():
    # ====== Parámetros del bloque (laterales comunes) ======
    with st.sidebar:
        min_score = st.slider("Umbral score (externos)", 0.0, 1.0, 0.70, 0.01, key="min_score_ext")

    # Carga de vistas SOLO cuando estamos en este bloque (cacheada)
    views = load_views(min_score, _mtime(UNIF), _mtime(MULTI), _mtime(CAT), _mtime(STOCK))

    # --- Normalización defensiva por si las vistas traen item_id ---
    def _norm_ui_products(df: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_pid_col(df, prefer_int=False)  # fuerza Product_ID
        # nombres comunes que usamos en este bloque
        return df

    def _norm_ui_substitutes(df: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_pid_col(df, prefer_int=False)  # fuerza Product_ID
        sub = _find_col(df, ["Substitute_Product_ID", "substitute_product_id", "alt_product_id"])
        if sub and sub != "Substitute_Product_ID":
            df = df.rename(columns={sub: "Substitute_Product_ID"})
        if "Substitute_Product_ID" in df.columns:
            df["Substitute_Product_ID"] = _to_pid_str(df["Substitute_Product_ID"])
        # tipo
        tcol = _find_col(df, ["tipo", "type"])
        if tcol and tcol != "tipo":
            df = df.rename(columns={tcol: "tipo"})
        if "tipo" not in df.columns:
            df["tipo"] = "externo"
        return df

    # ====== Subpestañas persistentes (evita saltos) ======
    if "explore_subtab" not in st.session_state:
        st.session_state["explore_subtab"] = "Productos"

    subtab = st.radio(
        " ",
        ["Productos", "Sustitutos por producto"],
        horizontal=True,
        key="explore_subtab",
    )

    # ------------------------------------------------------------------
    # SUBPESTAÑA: PRODUCTOS
    # ------------------------------------------------------------------
    if subtab == "Productos":
        refresh_secs = st.sidebar.slider("⏱ Auto-refresh Productos (seg)", 0, 300, 60, key="refresh_tab1_secs")
        refresh_on   = st.sidebar.toggle("Activar auto-refresh Productos", True, key="refresh_tab1_on")
        if refresh_on and refresh_secs > 0:
            st_autorefresh(interval=int(refresh_secs * 1000), key="auto_refresh_tab1")

        st.subheader("Productos (resumen)")

        # Catálogo para añadir Nombre/Categoría (si existe)
        cat_path = ROOT / "data" / "processed" / "catalog_items.csv"
        try:
            cat = pd.read_csv(cat_path, encoding="utf-8")
            # Mapear a nombres estándar
            pid = _find_col(cat, ["Product_ID", "product_id", "item_id", "id_producto"])
            nom = _find_col(cat, ["Nombre", "name", "nombre"])
            catg = _find_col(cat, ["Categoria", "Categoría", "category", "categoria"])
            ren = {}
            if pid and pid != "Product_ID": ren[pid] = "Product_ID"
            if nom and nom != "Nombre":     ren[nom] = "Nombre"
            if catg and catg != "Categoria":ren[catg] = "Categoria"
            if ren: cat = cat.rename(columns=ren)
            cat = cat[["Product_ID","Nombre","Categoria"]].drop_duplicates("Product_ID")
            cat["Product_ID"] = _to_pid_str(cat["Product_ID"])
        except Exception:
            cat = None

        dfp = views["ui_products"].copy()
        dfp["Product_ID"] = _to_pid_str(dfp["Product_ID"])
        if cat is not None:
            dfp = dfp.merge(cat, on="Product_ID", how="left")

        rename_cols = {
            "Product_ID": "Product_ID",
            "Nombre": "Nombre",
            "Categoria": "Categoría",
            "on_hand": "Stock actual",
            "preferred_supplier_id": "Proveedor principal",
            "preferred_price": "Precio preferente",
            "preferred_lead_time": "Lead time pref.",
            "subs_internos_count": "Sustitutos internos",
            "subs_externos_count": "Sustitutos externos",
        }
        cols_final = [c for c in rename_cols if c in dfp.columns]
        dfp = dfp[cols_final].rename(columns=rename_cols)

        # --- Sincronizar "Stock actual" con el inventario vivo (inventory_updated.csv) ---
        try:
            inv_live = _read_working_inventory()  # lee OUT10/inventory_updated.csv o, si no existe, el Inventario.csv base
            if inv_live is not None and not inv_live.empty:
                inv_live = inv_live[["Product_ID", "Stock Real"]].copy()
                inv_live["Product_ID"] = _to_pid_str(inv_live["Product_ID"])
                inv_live["Stock Real"] = pd.to_numeric(inv_live["Stock Real"], errors="coerce").fillna(0).astype(int)

                if "Product_ID" in dfp.columns:
                    dfp["Product_ID"] = _to_pid_str(dfp["Product_ID"])
                    dfp = dfp.merge(inv_live, on="Product_ID", how="left")

                    # Sobrescribe SIEMPRE con el inventario vivo
                    dfp["Stock actual"] = dfp["Stock Real"].fillna(0).astype(int)
                    dfp = dfp.drop(columns=["Stock Real"], errors="ignore")
            else:
                # si no hay inventario, al menos garantiza la columna
                if "Stock actual" not in dfp.columns:
                    dfp["Stock actual"] = 0
        except Exception as e:
            # no rompas la pestaña si falla la lectura
            if "Stock actual" not in dfp.columns:
                dfp["Stock actual"] = 0
            st.warning(f"No se pudo sincronizar el stock con el inventario vivo: {e}")

        if "Stock actual" in dfp.columns:
            dfp.insert(
                dfp.columns.get_loc("Stock actual"),
                "Alerta",
                dfp["Stock actual"].apply(lambda x: "⚠️ Bajo" if pd.notnull(x) and x < 20 else "")
            )

        q = st.text_input("Buscar por nombre, ID o categoría", key="search_products")
        if q:
            ql = str(q).strip().lower()
            masks = []
            if "Product_ID" in dfp.columns:
                masks.append(dfp["Product_ID"].astype(str).str.lower().str.contains(ql, na=False))
                # comparación exacta como string (evita castear a int)
                masks.append(dfp["Product_ID"].astype(str).str.lower().eq(ql))
            if "Nombre" in dfp.columns:
                masks.append(dfp["Nombre"].fillna("").str.lower().str.contains(ql, na=False))
            if "Categoría" in dfp.columns:
                masks.append(dfp["Categoría"].fillna("").str.lower().str.contains(ql, na=False))
            if masks:
                mask = masks[0]
                for m in masks[1:]:
                    mask |= m
                dfp = dfp[mask]

        # formatos
        num_cols = ["Precio preferente", "Stock actual", "Lead time pref.",
                    "Sustitutos internos", "Sustitutos externos"]
        for c in num_cols:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

        df_show = dfp.copy()
        if "Precio preferente" in df_show.columns:
            df_show["Precio preferente"] = df_show["Precio preferente"].map(
                lambda v: "" if pd.isna(v) else f"{v:,.2f} €"
            )
        if "Stock actual" in df_show.columns:
            df_show["Stock actual"] = df_show["Stock actual"].map(
                lambda v: "" if pd.isna(v) else f"{v:,.0f}"
            )

        # tooltips
        col_cfg = {}
        if "Product_ID" in df_show.columns:
            col_cfg["Product_ID"] = st.column_config.TextColumn("Product_ID", help="Identificador único del producto.")
        if "Nombre" in df_show.columns:
            col_cfg["Nombre"] = st.column_config.TextColumn("Nombre", help="Nombre comercial (desde catálogo).")
        if "Categoría" in df_show.columns:
            col_cfg["Categoría"] = st.column_config.TextColumn("Categoría", help="Familia/categoría del catálogo.")
        if "Alerta" in df_show.columns:
            col_cfg["Alerta"] = st.column_config.TextColumn("Alerta", help="‘⚠️ Bajo’ si stock < 20.")
        if "Stock actual" in df_show.columns:
            col_cfg["Stock actual"] = st.column_config.TextColumn("Stock actual", help="Unidades en inventario.")
        if "Proveedor principal" in df_show.columns:
            col_cfg["Proveedor principal"] = st.column_config.TextColumn("Proveedor principal", help="Proveedor preferente.")
        if "Precio preferente" in df_show.columns:
            col_cfg["Precio preferente"] = st.column_config.TextColumn("Precio preferente", help="Precio del proveedor preferente.")
        if "Lead time pref." in df_show.columns:
            col_cfg["Lead time pref."] = st.column_config.NumberColumn("Lead time pref.", help="Días de suministro.", format="%d")
        if "Sustitutos internos" in df_show.columns:
            col_cfg["Sustitutos internos"] = st.column_config.NumberColumn("Sustitutos internos", help="Alternativas internas.", format="%d")
        if "Sustitutos externos" in df_show.columns:
            col_cfg["Sustitutos externos"] = st.column_config.NumberColumn("Sustitutos externos", help="Alternativas externas.", format="%d")

        preferred_order = [c for c in [
            "Product_ID", "Nombre", "Categoría",
            "Alerta", "Stock actual",
            "Proveedor principal", "Precio preferente", "Lead time pref.",
            "Sustitutos internos", "Sustitutos externos"
        ] if c in df_show.columns]
        df_show = df_show[preferred_order]

        st.dataframe(df_show, use_container_width=True, height=420, column_config=col_cfg)

    # ------------------------------------------------------------------
    # SUBPESTAÑA: SUSTITUTOS POR PRODUCTO
    # ------------------------------------------------------------------
    else:
        st.subheader("Sustitutos por producto")

        # Autorefresh SOLO Sustitutos
        refresh_secs = st.sidebar.slider("⏱ Auto-refresh Sustitutos (seg)", 0, 300, 60, key="refresh_tab2_secs")
        refresh_on   = st.sidebar.toggle("Activar auto-refresh Sustitutos", True, key="refresh_tab2_on")
        if refresh_on and refresh_secs > 0:
            st_autorefresh(interval=int(refresh_secs * 1000), key="auto_refresh_tab2")

        # --- Cargar vistas base ---
        try:
            ui_subs_views = _norm_ui_substitutes(views["ui_substitutes"].copy())
            dfp           = _norm_ui_products(views["ui_products"].copy())
        except Exception as e:
            st.error("No se pudieron cargar 'ui_substitutes' o 'ui_products'.")
            st.exception(e)
            st.stop()

        cat  = load_catalog_items(ROOT)           # catálogo (nombre/categoría/proveedor/price/lt/disp)
        uni  = load_substitutes_unified(ROOT)     # sustitutos externos (si lo tienes)
        scm  = load_supplier_catalog_multi(ROOT)  # multi-catálogo para internos

        # Uni ya viene normalizado a Product_ID; si no hay, usamos la vista
        ui_subs = uni if uni is not None else ui_subs_views
        ui_subs = _norm_ui_substitutes(ui_subs)

        # --- Normalización de IDs (defensiva) ---
        pid_col = "Product_ID"
        dfp[pid_col] = _to_pid_str(dfp[pid_col])
        if cat is not None and "Product_ID" in cat.columns:
            cat["Product_ID"] = _to_pid_str(cat["Product_ID"])
        if scm is not None and "Product_ID" in scm.columns:
            scm["Product_ID"] = _to_pid_str(scm["Product_ID"])

        # callback para NO volver a "Productos" al escribir en el buscador
        def _stay_on_subs():
            st.session_state["explore_subtab"] = "Sustitutos por producto"

        q2 = st.text_input("Buscar por ID, nombre o categoría", key="q_tab2", on_change=_stay_on_subs)

        # ---------- RESUMEN ----------
        tipo_col = "tipo"  # ya normalizado arriba

        ext_counts = (
            ui_subs[ui_subs[tipo_col].eq("externo")]
            .groupby(pid_col, as_index=False)
            .size().rename(columns={"size": "Sustitutos externos"})
        )

        if scm is not None and not scm.empty:
            pref_sup_col = _find_col(dfp, ["preferred_supplier_id"])
            base_int = scm.groupby(pid_col, as_index=False)["scm_supplier_id"].nunique() \
                          .rename(columns={"scm_supplier_id": "n_suppliers"})
            if pref_sup_col:
                df_pref = dfp[[pid_col, pref_sup_col]].drop_duplicates()
                base_int = base_int.merge(df_pref, on=pid_col, how="left")
                base_int["Sustitutos internos"] = (base_int["n_suppliers"] - 1).clip(lower=0)
            else:
                base_int["Sustitutos internos"] = base_int["n_suppliers"]
            int_counts = base_int[[pid_col, "Sustitutos internos"]]
        else:
            int_counts = pd.DataFrame(columns=[pid_col, "Sustitutos internos"])

        resumen = pd.merge(int_counts, ext_counts, on=pid_col, how="outer").fillna(0)
        for c in ["Sustitutos internos", "Sustitutos externos"]:
            if c in resumen.columns:
                resumen[c] = resumen[c].astype(int)

        if cat is not None:
            add_cols = [c for c in ["cat_nombre", "cat_categoria", "cat_proveedor"] if c in cat.columns]
            if add_cols:
                resumen = resumen.merge(cat[[pid_col, *add_cols]], on=pid_col, how="left")

        cols_join = [pid_col] + [c for c in ["preferred_price", "preferred_lead_time"] if c in dfp.columns]
        if len(cols_join) > 1:
            resumen = resumen.merge(dfp[cols_join].drop_duplicates(pid_col), on=pid_col, how="left")

        if q2:
            ql = q2.lower()
            resumen = resumen[
                _to_str_safe(resumen[pid_col]).str.lower().str.contains(ql, na=False) |
                _to_str_safe(resumen.get("cat_nombre", pd.Series("", index=resumen.index))).str.lower().str.contains(ql, na=False) |
                _to_str_safe(resumen.get("cat_categoria", pd.Series("", index=resumen.index))).str.lower().str.contains(ql, na=False)
            ]

        if not resumen.empty:
            sort_cols = [c for c in ["Sustitutos externos", "Sustitutos internos"] if c in resumen.columns]
            if sort_cols:
                resumen = resumen.sort_values(sort_cols, ascending=False)
        

        # --- Sincronizar "Stock actual" con el inventario vivo ---

        try:
            inv_live = _read_working_inventory()  # -> columnas: Product_ID, Proveedor, Nombre, Categoria, Stock Real
            if inv_live is not None and not inv_live.empty:
                inv_live = inv_live[["Product_ID", "Stock Real"]].copy()
                inv_live["Product_ID"] = _to_pid_str(inv_live["Product_ID"])
                inv_live["Stock Real"] = pd.to_numeric(inv_live["Stock Real"], errors="coerce").fillna(0).astype(int)

                # Normaliza el df de resumen y une por Product_ID
                if "Product_ID" in resumen.columns:
                    resumen["Product_ID"] = _to_pid_str(resumen["Product_ID"])
                    resumen = resumen.merge(
                        inv_live.rename(columns={"Stock Real": "Stock actual (inventario)"}),
                        on="Product_ID", how="left"
                    )

                    # Si ya existe una columna "Stock actual", la sobreescribimos con el inventario vivo
                    if "Stock actual (inventario)" in resumen.columns:
                        resumen["Stock actual"] = resumen["Stock actual (inventario)"].fillna(
                            pd.to_numeric(resumen.get("Stock actual"), errors="coerce")
                        ).fillna(0).astype(int)
                        resumen = resumen.drop(columns=["Stock actual (inventario)"])
                else:
                    # si por cualquier motivo no hay Product_ID, no rompemos la vista
                    if "Stock actual" not in resumen.columns:
                        resumen["Stock actual"] = 0
            else:
                # inventario vacío -> aseguramos la columna
                if "Stock actual" not in resumen.columns:
                    resumen["Stock actual"] = 0
        except Exception as e:
            # Failsafe: no romper la pestaña
            if "Stock actual" not in resumen.columns:
                resumen["Stock actual"] = 0
            st.warning(f"No se pudo sincronizar el stock con el inventario vivo: {e}")



        resumen_view = resumen.rename(columns={
            "cat_nombre": "Nombre",
            "cat_categoria": "Categoría",
            "cat_proveedor": "Proveedor principal",
            "preferred_price": "Precio pref.",
            "preferred_lead_time": "Lead time pref."
        })
        cols_resumen = [pid_col, "Nombre", "Categoría", "Proveedor principal",
                        "Sustitutos internos", "Sustitutos externos", "Precio pref.", "Lead time pref."]
        resumen_view = resumen_view[[c for c in cols_resumen if c in resumen_view.columns]].copy()

        def fmt_eur(x):
            if pd.isna(x): return ""
            try: return f"{float(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
            except: return str(x)
        if "Precio pref." in resumen_view.columns:
            resumen_view["Precio pref."] = pd.to_numeric(resumen_view["Precio pref."], errors="coerce").map(fmt_eur)
        for c in ["Nombre", "Categoría", "Proveedor principal"]:
            if c in resumen_view.columns:
                resumen_view[c] = _to_str_safe(resumen_view[c])

        if resumen_view.empty:
            st.info("Sin datos tras filtrar.")
            st.stop()

        visible_pids = resumen_view[pid_col].astype(str).tolist()
        cur_sel = str(st.session_state.get("sel_pid_tab2", ""))
        if cur_sel not in visible_pids:
            st.session_state["sel_pid_tab2"] = visible_pids[0]

        st.caption("Resumen de cobertura de sustitutos (haz clic en una fila para ver el detalle)")
        summary_key = "subs_summary_editor"
        # === Enriquecer 'resumen' con Nombre / Categoría / Proveedor y construir 'resumen_view' ===
        resumen_view = resumen.copy()
        resumen_view["Product_ID"] = _to_pid_str(resumen_view["Product_ID"])

        # Si aún no existen esas columnas, las traemos de catálogo y multi/ui_products
        if not {"Nombre", "Categoría", "Proveedor principal"}.issubset(resumen_view.columns):

            # 1) Nombre + Categoría (catálogo)
            cat = load_catalog_items(ROOT)  # -> Product_ID, Nombre, Categoria
            cat_mini = (cat[["Product_ID","Nombre","Categoria"]].drop_duplicates("Product_ID")
                        if cat is not None and not cat.empty else
                        pd.DataFrame(columns=["Product_ID","Nombre","Categoria"]))
            if not cat_mini.empty:
                cat_mini["Product_ID"] = _to_pid_str(cat_mini["Product_ID"])
                resumen_view = resumen_view.merge(cat_mini, on="Product_ID", how="left")
            # renombra Categoria -> Categoría para la vista
            if "Categoria" in resumen_view.columns and "Categoría" not in resumen_view.columns:
                resumen_view = resumen_view.rename(columns={"Categoria": "Categoría"})

            # 2) Proveedor preferente (ui_products si existe; si no, el más barato del multi)
            prov_mini = pd.DataFrame(columns=["Product_ID", "Proveedor"])
            try:
                ui_prod_path = TMP_VISTAS / "ui_products.parquet"
                if ui_prod_path.exists():
                    up = pd.read_parquet(ui_prod_path)
                    up = _ensure_pid_col(up)
                    low = {c.lower(): c for c in up.columns}
                    prov_col = low.get("preferred_supplier_id") or low.get("proveedor") or low.get("supplier_id")
                    if prov_col:
                        prov_mini = (up.rename(columns={prov_col: "Proveedor"})
                                        [["Product_ID", "Proveedor"]]
                                        .drop_duplicates("Product_ID"))
            except Exception:
                pass

            if prov_mini.empty:
             multi = load_supplier_catalog_multi(ROOT)
            if multi is not None and not multi.empty:
                mm = multi.copy()
                mm["Product_ID"] = _to_pid_str(mm["Product_ID"])
                mm["scm_price"] = pd.to_numeric(mm.get("scm_price"), errors="coerce")
                mm = (mm.sort_values(["Product_ID","scm_price"], na_position="last")
                        .drop_duplicates("Product_ID"))
                prov_mini = mm[["Product_ID","scm_supplier_id"]].rename(columns={"scm_supplier_id":"Proveedor"})

            if not prov_mini.empty:
                prov_mini["Product_ID"] = _to_pid_str(prov_mini["Product_ID"])
                resumen_view = resumen_view.merge(prov_mini, on="Product_ID", how="left")

            # nombre final de la columna para la UI
            if "Proveedor" in resumen_view.columns and "Proveedor principal" not in resumen_view.columns:
                resumen_view = resumen_view.rename(columns={"Proveedor": "Proveedor principal"})

        # Asegurar que existan (aunque vengan vacías)
        for c in ["Nombre", "Categoría", "Proveedor principal"]:
            if c not in resumen_view.columns:
                resumen_view[c] = ""

        # Orden final para la tabla
        cols_resumen = [
            "Product_ID", "Nombre", "Proveedor principal", "Categoría",
            "Sustitutos internos", "Sustitutos externos",
            "Precio pref.", "Lead time pref."
        ]
        resumen_view = resumen_view[[c for c in cols_resumen if c in resumen_view.columns]].copy()

        st.data_editor(
            resumen_view,
            key=summary_key,
            use_container_width=True,
            height=360,
            hide_index=True,
            disabled=True,
            column_config={
                "Product_ID": st.column_config.TextColumn("Product_ID", help="ID del producto"),
                "Nombre": st.column_config.TextColumn("Nombre", help="Nombre del producto (catálogo)"),
                "Categoría": st.column_config.TextColumn("Categoría", help="Familia/categoría (catálogo)"),
                "Proveedor principal": st.column_config.TextColumn("Proveedor principal", help="Proveedor preferente del catálogo"),
                "Sustitutos internos": st.column_config.NumberColumn("Sustitutos internos", help="Nº de proveedores alternativos", format="%d"),
                "Sustitutos externos": st.column_config.NumberColumn("Sustitutos externos", help="Nº de productos alternativos", format="%d"),
                "Precio pref.": st.column_config.TextColumn("Precio pref.", help="Precio del proveedor preferente"),
                "Lead time pref.": st.column_config.NumberColumn("Lead time pref.", help="LT del preferente", format="%d"),
            },
        )

        sel_rows = (
            st.session_state.get(summary_key, {})
            .get("selection", {})
            .get("rows", [])
        )
        if sel_rows:
            idx = next(iter(sel_rows))
            sel_pid = str(resumen_view.iloc[idx][pid_col])
            if st.session_state.get("sel_pid_tab2") != sel_pid:
                st.session_state["sel_pid_tab2"] = sel_pid
                st.session_state["explore_subtab"] = "Sustitutos por producto"
                (getattr(st, "rerun", st.experimental_rerun))()

        # ---------- DETALLE ----------
        pid = st.session_state["sel_pid_tab2"]
        st.caption(f"Detalle de sustitutos para Product_ID = {pid}")

        dfs_ext = ui_subs[(ui_subs[pid_col] == pid) & (ui_subs["tipo"].eq("externo"))].copy()
        dfs_ext = enrich_external_subs(dfs_ext, dfp, cat) if not dfs_ext.empty else pd.DataFrame(columns=[
            "tipo", "Substitute_Product_ID", "nombre", "categoria", "proveedor",
            "rank", "score", "precio", "lead_time", "disponibilidad"
        ])

        dfs_int = build_internal_subs(pid, scm, dfp, cat)

        df_show = pd.concat([dfs_int, dfs_ext], ignore_index=True, sort=False)

        # --- Sincronizar "Disponibilidad" del detalle con el inventario vivo ---
        try:
            inv_live = _read_working_inventory()  # -> columnas: Product_ID, Proveedor, Nombre, Categoria, Stock Real
            if inv_live is not None and not inv_live.empty:
                inv_live = inv_live[["Product_ID", "Stock Real"]].copy()
                inv_live["Product_ID"] = _to_pid_str(inv_live["Product_ID"])
            inv_live["Stock Real"] = pd.to_numeric(inv_live["Stock Real"], errors="coerce").fillna(0).astype(int)

            # Unimos el inventario al detalle por el ID del sustituto
            df_show = df_show.merge(
                inv_live.rename(columns={"Product_ID": "Substitute_Product_ID", "Stock Real": "disp_inv"}),
                on="Substitute_Product_ID", how="left"
            )  

            # Si ya había una columna 'disponibilidad', la sobrescribimos cuando tengamos inventario;
            # si no existía, la creamos con el inventario.
            if "disponibilidad" in df_show.columns:
                df_show["disponibilidad"] = df_show["disp_inv"].fillna(df_show["disponibilidad"])
            else:
                df_show["disponibilidad"] = df_show["disp_inv"]

            df_show = df_show.drop(columns=["disp_inv"], errors="ignore")
        except Exception as _e:
            # No rompemos la IU si algo falla
            pass

        row_pref = dfp.loc[dfp[pid_col] == pid].head(1)
        if not row_pref.empty:
            row_pref = row_pref.iloc[0]
            pref_price = pd.to_numeric(row_pref.get("preferred_price"), errors="coerce")
            pref_lt    = pd.to_numeric(row_pref.get("preferred_lead_time"), errors="coerce")
            pref_disp  = pd.to_numeric(row_pref.get("preferred_disponibilidad"), errors="coerce")
        else:
            pref_price = pref_lt = pref_disp = None

        for c in ["precio", "lead_time", "disponibilidad"]:
            if c in df_show.columns:
                df_show[c] = pd.to_numeric(df_show[c], errors="coerce")

        df_show["Δ precio"]    = df_show["precio"]         - pref_price if "precio" in df_show.columns else None
        df_show["Δ lead time"] = df_show["lead_time"]      - pref_lt    if "lead_time" in df_show.columns else None
        df_show["Δ disp"]      = df_show["disponibilidad"] - pref_disp  if "disponibilidad" in df_show.columns else None

        wanted = ["tipo", "Substitute_Product_ID", "nombre", "categoria", "proveedor",
                  "rank", "score", "precio", "lead_time", "disponibilidad", "Δ precio", "Δ lead time", "Δ disp"]
        show_cols = [c for c in wanted if c in df_show.columns]
        df_show = df_show[show_cols].rename(columns={
            "tipo": "Tipo movimiento",
            "Substitute_Product_ID": "Sustituto (Product_ID)",
            "nombre": "Nombre",
            "categoria": "Categoría",
            "proveedor": "Proveedor",
            "rank": "Rank",
            "score": "Score",
            "precio": "Precio",
            "lead_time": "Lead time",
            "disponibilidad": "Disponibilidad"
        }).reset_index(drop=True)

        if "Precio" in df_show.columns:
            df_show["Precio"] = df_show["Precio"].map(
                lambda v: "" if pd.isna(v) else f"{float(v):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
            )
        if "Δ precio" in df_show.columns:
            df_show["Δ precio"] = df_show["Δ precio"].map(
                lambda v: "" if pd.isna(v) else f"{float(v):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
            )

        st.data_editor(
            df_show,
            use_container_width=True,
            height=430,
            hide_index=True,
            disabled=True,
            column_config={
                "Tipo": st.column_config.TextColumn("Tipo", help="interno = mismo Product_ID con otro proveedor; externo = producto alternativo"),
                "Sustituto (Product_ID)": st.column_config.TextColumn("Sustituto (Product_ID)", help="ID del sustituto (externo) o el mismo Product_ID (interno)"),
                "Nombre": st.column_config.TextColumn("Nombre", help="Nombre del sustituto (catálogo/ui_products) o del original si es interno)"),
                "Categoría": st.column_config.TextColumn("Categoría", help="Categoría del sustituto (o del original si es interno)"),
                "Proveedor": st.column_config.TextColumn("Proveedor", help="Proveedor del sustituto. En internos es supplier_id del multi-catálogo"),
                "Rank": st.column_config.NumberColumn("Rank", help="Orden de preferencia (1=mejor)"),
                "Score": st.column_config.NumberColumn("Score", help="Score de similitud/prioridad para externos"),
                "Precio": st.column_config.TextColumn("Precio", help="Precio estimado del sustituto"),
                "Lead time": st.column_config.NumberColumn("Lead time", help="Lead time estimado del sustituto"),
                "Disponibilidad": st.column_config.NumberColumn("Disponibilidad", help="Disponibilidad estimada"),
                "Δ precio": st.column_config.TextColumn("Δ precio", help="Precio sustituto – precio preferente del original"),
                "Δ lead time": st.column_config.NumberColumn("Δ lead time", help="LT sustituto – LT preferente del original"),
                "Δ disp": st.column_config.NumberColumn("Δ disp", help="Disp. sustituto – Disp. preferente del original"),
            },
        )

# ==================================================
# 5) BLOQUE DE PROVEEDORES (intacto)
# ==================================================
def render_proveedores():
    st.header("🏭 Proveedores")

    tab_cat, tab_stats = st.tabs(["📘 Catálogo", "📈 Estadísticas"])

    # ---------- Pestaña: Catálogo ----------
    with tab_cat:
        st.subheader("Catálogo por proveedor")

        pm = _price_map()
        provs = ["Todos"] + sorted(pm["Proveedor"].dropna().unique().tolist())
        prov_sel = st.selectbox("Proveedor", provs, index=0)

        df_cat = _cat_table_by_supplier(prov_sel)
        st.dataframe(
            df_cat,
            use_container_width=True,
            hide_index=True,
        )

        # descarga
        csv = df_cat.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name=f"catalogo_{prov_sel or 'todos'}.csv", mime="text/csv")

    # ---------- Pestaña: Estadísticas ----------
    with tab_stats:
        st.subheader("Estadísticas 2023–2025 (histórico + previsión)")

        # Carga de datos
        pm = _price_map()
        hist = _load_hist_demand()          # 2023 → hoy (qty)
        fore = _load_forecast_2025()        # 2025 completo (qty)
        qty = pd.concat([hist, fore], ignore_index=True) if not hist.empty or not fore.empty \
          else pd.DataFrame(columns=["Product_ID","Date","qty"])

        # Filtros
        c1, c2 = st.columns(2)
        start = c1.date_input("Desde", value=dt.date(2023,1,1))
        end   = c2.date_input("Hasta", value=dt.date(2025,12,31))
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

        rank_sup, serie_sup = _stats_by_supplier(qty, pm, start_ts, end_ts)

        # KPI: solo unidades
        total_qty = float(rank_sup["qty_total"].sum()) if not rank_sup.empty else 0.0
        k1, = st.columns(1)
        k1.metric("Unidades totales", f"{total_qty:,.0f} uds")

        # Ranking proveedores (solo unidades)
        st.markdown("#### Ranking de proveedores")
        if rank_sup.empty:
            st.info("No hay datos para el rango seleccionado.")
        else:
            df_rank_view = rank_sup.rename(columns={"qty_total": "Unidades"}).copy()
            df_rank_view["Unidades"] = _fmt_unidades_ceil(df_rank_view["Unidades"])    
            st.dataframe(  
                df_rank_view,
                use_container_width=True,
                hide_index=True
            )

        # Evolución temporal (mensual) por unidades
        st.markdown("#### Evolución por proveedor (mensual)")
        if serie_sup.empty:
            st.info("No hay datos para el rango seleccionado.")
        else:
            # 1) Selector múltiple de proveedores (por defecto: todos)
            all_suppliers = sorted(serie_sup["Proveedor"].dropna().unique().tolist())
            sel_suppliers = st.multiselect(
                "Selecciona proveedores para comparar",
                options=all_suppliers,
                default=all_suppliers
            )

            # 2) Filtramos la serie según la selección
            serie_filtrada = serie_sup[serie_sup["Proveedor"].isin(sel_suppliers)].copy()

            if serie_filtrada.empty:
                st.warning("No hay proveedores seleccionados (o no hay datos para la selección).")
            else:
            # 3) Pivot y gráfica
                pivot_qty = (serie_filtrada
                                .pivot(index="Date", columns="Proveedor", values="qty")
                                .sort_index()
                                .fillna(0.0))

                st.line_chart(pivot_qty, use_container_width=True, height=280)

        # Top productos por proveedor (solo unidades)
        st.markdown("#### Top productos por proveedor")
        provs = ["(Selecciona proveedor)"] + sorted(pm["Proveedor"].dropna().unique().tolist())
        prov_pick = st.selectbox("Proveedor para ver top productos", provs, index=0)
        if prov_pick != "(Selecciona proveedor)":
            df = qty[(qty["Date"] >= start_ts) & (qty["Date"] <= end_ts)].copy()
            df = df.merge(pm[["Product_ID","Nombre","Proveedor"]], on="Product_ID", how="left")
            df = df[df["Proveedor"] == prov_pick]
            top_prod = (df.groupby(["Product_ID","Nombre"], as_index=False)
                        .agg(Unidades=("qty","sum"))
                        .sort_values(["Unidades"], ascending=False)
                        .head(25))
            
            top_prod_view = top_prod.copy()
            top_prod_view["Unidades"] = _fmt_unidades_ceil(top_prod_view["Unidades"])

            top_prod["Unidades"] = (
                top_prod["Unidades"]
                .apply(lambda x: f"{int(np.ceil(x)):,}".replace(",", "."))
            )
            st.dataframe(top_prod, use_container_width=True, hide_index=True)

# ==================================================
# 6) BLOQUE MOVIMIENTOS DE STOCK
# ==================================================
def render_movimientos_stock():
    st.title("📦 Movimientos de stock")
    st.caption("Procesa pedidos de cliente y revisa impactos en inventario.")

    def _load_ui_orders() -> pd.DataFrame:
        _ensure_orders_ui()
        try:
            return pd.read_csv(ORD_UI)
        except Exception:
            return pd.DataFrame(columns=["date","Product_ID","qty"])

    # ---------- estado / toggles ----------
    use_ae = st.sidebar.toggle("Incluir escenarios A–E", value=True)

    # ---- Configurador de escenarios A–E ----
    if use_ae:
        with st.sidebar.expander("⚗️ Configurar escenarios A–E", expanded=False):
            n_orders = st.number_input("Nº de pedidos a generar", 1, 200, 5, step=1)
            lines_per_order = st.number_input("Líneas por pedido", 1, 200, 5, step=1)
            qty_min, qty_max = st.columns(2)
            v_min = qty_min.number_input("Qty mínima", 1, 10_000, 1, step=1)
            v_max = qty_max.number_input("Qty máxima", 1, 10_000, 5, step=1)
            days_back = st.slider("Rango de fechas (días hacia atrás)", 0, 90, 7)

            allow_repeats_in_order = st.checkbox("Permitir repetir producto dentro de un pedido", value=False)
            sampling_mode = st.selectbox("Muestreo de productos", ["Uniforme", "Ponderado por stock"])
            seed_str = st.text_input("Semilla aleatoria (opcional)", value="")

            if st.button("🎲 Generar escenarios A–E", use_container_width=True):
                try:
                    seed = int(seed_str) if seed_str.strip() != "" else None
                except Exception:
                    seed = None

                try:
                    df_ae = _generate_scenarios_AE(
                        n_orders=int(n_orders),
                        lines_per_order=int(lines_per_order),
                        qty_min=int(v_min),
                        qty_max=int(v_max),
                        days_back=int(days_back),
                        allow_repeats_in_order=bool(allow_repeats_in_order),
                        sampling_mode=str(sampling_mode),
                        seed=seed,
                    )
                    st.success(f"Escenarios A–E generados ✅ · {len(df_ae)} líneas → {ORD_AE.relative_to(ROOT)}")
                    # Previsualización
                    with st.expander("Ver muestra de escenarios A–E", expanded=False):
                        st.dataframe(df_ae.head(50), use_container_width=True, height=260)
                except Exception as e:
                    st.error("No se pudieron generar los escenarios A–E.")
                    st.exception(e)

    st.sidebar.markdown("**Pedidos UI:**")
    st.sidebar.code(str(ORD_UI.relative_to(ROOT)))

    # ---------- KPIs ----------
    ui_df = _load_ui_orders()
    c1, c2, c3 = st.columns(3)
    c1.metric("Pedidos en UI", ui_df["date"].nunique())
    c2.metric("Líneas totales", int(len(ui_df)))
    c3.metric("Con escenario A–E", "Sí" if use_ae else "No")
    st.markdown("")

    # ============ Acción: PROCESAR ============
    if st.sidebar.button("▶️ Procesar movimientos", use_container_width=True, type="primary"):
        try:
            # (1) Cargar pedidos (UI + AE)
            st.info("Preparando pedidos…")
            orders_csv = _prepare_orders_for_ms(include_ae=use_ae)
            if orders_csv is None or not orders_csv.exists():
                st.warning("No hay pedidos válidos (UI/AE vacíos o sin columnas).")
                st.stop()

            df_orders = pd.read_csv(orders_csv)
            df_orders = _ensure_pid_col(df_orders)

            if "qty" not in df_orders.columns:
                st.warning("Pedidos sin columna qty.")
                st.stop()

            # Normalizar y consolidar pedidos (IDs en el MISMO formato que el inventario)
            df_orders["Product_ID"] = _to_pid_str(df_orders["Product_ID"])
            df_orders["qty"] = pd.to_numeric(df_orders["qty"], errors="coerce").fillna(0).astype(int)
            df_orders = df_orders.groupby("Product_ID", as_index=False)["qty"].sum()
            with st.expander("🔎 Ver pedido consolidado (UI + A–E)", expanded=False):
                st.write(f"Total líneas (consolidado): **{len(df_orders)}**")
                st.dataframe(df_orders.head(100), use_container_width=True, height=240)
                st.caption("Arriba se muestran hasta 100 filas. El total puede ser mayor.")
                st.markdown("**Top por cantidad pedida**")
                top_qty = df_orders.sort_values("qty", ascending=False).head(20)
                st.dataframe(top_qty, use_container_width=True, height=240)

            # (2) Inventario 'vivo' con nombres EXACTOS
            inv_live = _read_working_inventory()
            if inv_live.empty:
                st.error("Inventario vivo vacío.")
                st.stop()

            inv_live["Product_ID"] = _to_pid_str(inv_live["Product_ID"])

            # --- Diagnóstico rápido
            inv_ids = set(inv_live["Product_ID"])
            ord_ids = set(df_orders["Product_ID"])
            no_casan = sorted(list(ord_ids - inv_ids))
            if no_casan:
                st.warning(
                    "Estos Product_ID del pedido NO existen en el inventario y NO se modificarán: "
                    + ", ".join(no_casan[:20]) + ("…" if len(no_casan) > 20 else "")
                )

            # (3) Construir diccionario de cantidades pedidas por Product_ID
            qty_by_id = df_orders.set_index("Product_ID")["qty"].to_dict()

            # (4) Calcular stocks
            df = inv_live.copy()
            df["on_prev"] = pd.to_numeric(df["Stock Real"], errors="coerce").fillna(0).astype(int)
            df["resta"]   = df["Product_ID"].map(qty_by_id).fillna(0).astype(int)
            df["on_new"]  = (df["on_prev"] - df["resta"]).astype(int)
            df["Δ"]       = df["on_new"] - df["on_prev"]

            # ---- Ledger de movimientos (append) ----
            mov = df.loc[df["resta"] > 0, ["Product_ID"]].copy()
            mov["Date"] = pd.Timestamp.now().normalize()
            mov["qty_pedido"] = df.loc[df["resta"] > 0, "resta"].values
            mov["Tipo movimiento"] = "Venta"
            append_ledger(mov)
            

            # (5) Reescribir inventario vivo
            out_path = OUT10 / "inventory_updated.csv"
            df_out = df[["Product_ID", "Proveedor", "Nombre", "Categoria"]].copy()
            df_out["Stock Real"] = df["on_new"].astype(int)
            df_out = df_out[["Product_ID", "Proveedor", "Nombre", "Categoria", "Stock Real"]]
            df_out.to_csv(out_path, index=False)

            # (6) Mostrar comparativa clara
            ts_str = _mtime_str(out_path)
            st.success("Procesado OK ✅. Inventario actualizado (base = inventario 'vivo').")
            st.caption(f"`inventory_updated.csv` actualizado: {ts_str}")

            df_view = (
                df[["Product_ID", "Nombre", "Proveedor", "Categoria", "on_prev", "on_new", "Δ"]]
                .sort_values("Product_ID")
            )
            st.dataframe(df_view, use_container_width=True, height=320)

            # ---- Solo cambios (Δ != 0) + descarga ----
            df_changes = df_view[df_view["Δ"] != 0].copy()
            st.markdown(f"**Productos con cambio de stock:** {len(df_changes)}")
            st.dataframe(df_changes, use_container_width=True, height=260)

            st.download_button(
                "⬇️ Descargar cambios (CSV)",
                data=df_changes.to_csv(index=False).encode("utf-8"),
                file_name=f"diff_stock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ---- Reiniciar la pestaña: lanzar nueva orden ----
            st.markdown("---")
            c_reset1, c_reset2 = st.columns([1, 1])

            vaciar_ui = c_reset1.checkbox("Vaciar pedido UI", value=True,
                                        help="Si está marcado, deja customer_orders_ui.csv vacío para empezar de cero.")

            if c_reset2.button("🔄 Lanzar nueva orden", type="primary", use_container_width=True):
                if vaciar_ui:
                    pd.DataFrame(columns=["date", "Product_ID", "qty"]).to_csv(ORD_UI, index=False)

                try:
                    st.cache_data.clear()
                except Exception:
                    pass

                for k in ["orders_editor_explicit", "sel_pid_tab2"]:
                    st.session_state.pop(k, None)

                st.rerun()

            # (7) Limpieza de caché
            try:
                st.cache_data.clear()
            except Exception:
                pass
            pd.DataFrame(columns=["date","Product_ID","qty"]).to_csv(ORD_UI, index=False)

        except Exception as e:
            st.error("Error procesando pedidos contra el inventario.")
            st.exception(e)

    # ---------- Sidebar: salidas ----------
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Salidas (timestamp):**")
    for p in [
        OUT10 / "inventory_updated.csv",
        OUT10 / "ledger_movimientos.csv",
        OUT10 / "ordenes_compra.csv",
        OUT10 / "ordenes_compra_lineas.csv",
        OUT10 / "sugerencias_compra.csv",
        OUT10 / "alerts.csv",
    ]:
        st.sidebar.write(f"- `{p.relative_to(ROOT)}` · {_mtime_str(p)}")

    st.divider()

    # ======= Tabs de revisión rápida (UI) =======
    tab_ped, tab_stock, tab_ledger, tab_oc, tab_hist, tab_alerts = st.tabs(
        ["Pedidos cliente", "Stock actual", "📒 Ledger", "OC generadas", "📉 Demanda histórica", "🚨 Alertas"]
    )

    # ===================== TAB: Pedidos cliente =====================
    with tab_ped:
        st.subheader("Pedidos de cliente (UI)")

        _ensure_orders_ui()  # crea el CSV si no existe

        # ---------- Formulario para añadir líneas ----------
        with st.form("form_add_order", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns([1.1, 1.1, 0.9, 0.9])
            f_date = c1.date_input("Fecha", value=dt.datetime.today(), format="YYYY-MM-DD")
            f_item = c2.text_input("Product_ID", placeholder="Ej. 4347")
            f_qty  = c3.number_input("Cantidad", value=1, min_value=1, step=1)
            submitted = c4.form_submit_button("➕ Añadir")
            if submitted:
                pid = str(f_item).strip()
                if pid and f_qty > 0:
                    _append_order_row(f_date.strftime("%Y-%m-%d"), pid, int(f_qty))
                    st.success("Línea añadida al pedido.")
                    try: 
                        st.cache_data.clear()
                    except: 
                        pass
                    st.rerun()
                else:
                    st.warning("Indica Product_ID y Cantidad > 0.")

        # ---------- Cargar pedidos UI (SIN CACHE) ----------
        def _read_orders_ui_fresh() -> pd.DataFrame:
            try:
                return pd.read_csv(ORD_UI)
            except Exception:
                return pd.DataFrame(columns=["date","Product_ID","qty"])

        df_ui = _read_orders_ui_fresh()
        sel_rows_pos = []  # 🔹 Inicializamos siempre para evitar el UnboundLocalError

        if df_ui.empty:
            st.info("Aún no hay pedidos en UI.")
        else:
            # Normalizar columnas base
            low = {c.lower(): c for c in df_ui.columns}
            if "product_id" not in low and "item_id" in low:
                df_ui = df_ui.rename(columns={low["item_id"]: "Product_ID"})
            if "date" not in df_ui.columns and "fecha" in low:
                df_ui = df_ui.rename(columns={low["fecha"]: "date"})
            if "qty" not in df_ui.columns and "cantidad" in low:
                df_ui = df_ui.rename(columns={low["cantidad"]: "qty"})
            df_ui["Product_ID"] = df_ui["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
            df_ui["qty"] = pd.to_numeric(df_ui["qty"], errors="coerce").fillna(0).astype(int)

            # ---------- Enriquecer con Nombre ----------
            cat = load_catalog_items(ROOT)  # -> Product_ID, Nombre, Categoria
            if cat is not None and not cat.empty:
                cat_mini = cat[["Product_ID","Nombre"]].drop_duplicates("Product_ID")
            else:
                cat_mini = pd.DataFrame(columns=["Product_ID","Nombre"])

            # ---------- Proveedor ----------
            prov_mini = pd.DataFrame(columns=["Product_ID","Proveedor"])
            try:
                ui_prod_path = TMP_VISTAS / "ui_products.parquet"
                if ui_prod_path.exists():
                    up = pd.read_parquet(ui_prod_path)
                    up = _ensure_pid_col(up)
                    low_up = {c.lower(): c for c in up.columns}
                    prov_col = low_up.get("preferred_supplier_id") or low_up.get("up_proveedor") or low_up.get("proveedor")
                    if prov_col:
                        prov_mini = up.rename(columns={prov_col:"Proveedor"})[["Product_ID","Proveedor"]].drop_duplicates("Product_ID")
            except Exception:
                pass

            if prov_mini.empty:
                multi = load_supplier_catalog_multi(ROOT)
                if multi is not None and not multi.empty:
                    mm = multi.copy()
                    mm["scm_price"] = pd.to_numeric(mm.get("scm_price"), errors="coerce")
                    mm = mm.sort_values(["Product_ID","scm_price"], na_position="last")
                    mm = mm.drop_duplicates("Product_ID")
                    prov_mini = mm[["Product_ID","scm_supplier_id"]].rename(columns={"scm_supplier_id":"Proveedor"})

            # ---------- DF para mostrar ----------
            df_show = df_ui.copy()
            df_show = df_show.merge(cat_mini, on="Product_ID", how="left")
            if not prov_mini.empty:
                df_show = df_show.merge(prov_mini, on="Product_ID", how="left")
            for c in ["Nombre","Proveedor"]:
                if c not in df_show.columns:
                    df_show[c] = ""

            # ---------- Editor con columna de selección ----------
            SEL = "__seleccionar__"
            df_display = df_show.rename(columns={"qty": "Cantidad", "date": "Fecha"}).copy()
            if SEL not in df_display.columns:
                df_display[SEL] = False

            column_config = {
                "Product_ID": st.column_config.TextColumn("Product_ID", disabled=True),
                "Nombre":     st.column_config.TextColumn("Nombre", disabled=True),
                "Proveedor":  st.column_config.TextColumn("Proveedor", disabled=True),
                "Cantidad":   st.column_config.NumberColumn("Cantidad", format="%d", disabled=True),
                "Fecha":      st.column_config.TextColumn("Fecha", disabled=True),
                SEL:          st.column_config.CheckboxColumn("Seleccionar"),
            }
            order = [c for c in ["Product_ID","Nombre","Proveedor","Cantidad","Fecha",SEL] if c in df_display.columns]

            edited = st.data_editor(
                df_display[order],
                use_container_width=True,
                height=380,
                hide_index=True,
                num_rows="fixed",
                disabled=False,
                column_config=column_config,
                key="orders_editor_explicit",
            )

            # Fila(s) marcadas por POSICIÓN
            sel_rows_pos = edited.index[edited[SEL] == True].tolist()

            c_left, c_mid, c_right = st.columns([1,1,2])

            # Eliminar seleccionadas
            if c_left.button("🗑️ Eliminar seleccionadas", disabled=(len(sel_rows_pos) == 0), use_container_width=True):
                df_new = df_ui.drop(df_ui.index[sel_rows_pos]).reset_index(drop=True)
                df_new.to_csv(ORD_UI, index=False)
                try: st.cache_data.clear()
                except: pass
                st.success(f"Eliminadas {len(sel_rows_pos)} línea(s).")
                st.rerun()

            # Vaciar pedido completo
            if c_mid.button("🧹 Vaciar pedido completo", use_container_width=True):
                pd.DataFrame(columns=["date","Product_ID","qty"]).to_csv(ORD_UI, index=False)
                try: st.cache_data.clear()
                except: pass
                st.success("Pedido vaciado.")
                st.rerun()

            # Descargar CSV base
            csv_bytes = df_ui.to_csv(index=False).encode("utf-8")
            c_right.download_button(
                "⬇️ Descargar CSV",
                data=csv_bytes,
                file_name="customer_orders_ui.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---------------- TAB: Stock actual ----------------
    with tab_stock:
        st.subheader("📦 Estado del inventario vivo")

        inv_live = _read_working_inventory()
        if inv_live is None or inv_live.empty:
            st.warning("Inventario vivo vacío.")
        else:
            import numpy as np

            inv_live["Product_ID"] = _to_pid_str(inv_live["Product_ID"])
            inv = inv_live.copy()

            # ---------- CARGA CACHEADA: previsión (μ_d) y estacionalidad ----------
            @st.cache_data(show_spinner=False)
            def _load_demanda_cached(root: Path):
                try:
                    dem = load_demanda_base(root)  # -> Product_ID, (Date), sales_quantity
                except Exception:
                    dem = None
                if dem is None or dem.empty:
                    return None, None

                # μ diaria por SKU
                dmean = (
                    dem.groupby("Product_ID")["sales_quantity"]
                    .mean()
                    .reset_index(name="demanda_media_dia")
                )

                # Estacionalidad por DOW (si hay fecha)
                season_map = None
                if "Date" in dem.columns and dem["Date"].notna().any():
                    dem2 = dem.copy()
                    dem2["Date"] = pd.to_datetime(dem2["Date"], errors="coerce")
                    dem2["dow"] = dem2["Date"].dt.dayofweek
                    mu_total = dem2.groupby("Product_ID")["sales_quantity"].mean().rename("mu")
                    mu_dow = dem2.groupby(["Product_ID","dow"])["sales_quantity"].mean().rename("mu_dow")
                    sigma_base = dem2.groupby("Product_ID")["sales_quantity"].std(ddof=1).rename("sigma")
                    df_season = mu_dow.to_frame().join(mu_total, how="left")
                    df_season["s_dow"] = (df_season["mu_dow"] / df_season["mu"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                    season_map = {}
                    for pid, grp in df_season.reset_index().groupby("Product_ID"):
                        season_map[str(pid)] = {
                            "s_dow": grp.set_index("dow")["s_dow"],
                            "mu": float(mu_total.get(pid, np.nan)) if not pd.isna(mu_total.get(pid, np.nan)) else None,
                            "sigma": float(sigma_base.get(pid, np.nan)) if not pd.isna(sigma_base.get(pid, np.nan)) else None,
                        }
                return dmean, season_map

            dmean, season_map = _load_demanda_cached(ROOT)

            # --- unir μ_d (y redondear al ALZA sin decimales)
            if dmean is not None:
                dmean["Product_ID"] = dmean["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
                inv["Product_ID"] = inv["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
                inv = inv.merge(dmean, on="Product_ID", how="left")
            else:
                inv["demanda_media_dia"] = 0.0
            inv["demanda_media_dia"] = np.ceil(pd.to_numeric(inv["demanda_media_dia"], errors="coerce").fillna(0.0)).astype(int)

            # --- unir Lead Time (admite 'lead_time' o 'scm_lead_time')
            multi = load_supplier_catalog_multi(ROOT)
            if multi is not None and not multi.empty:
                mm = multi.copy()
                low = {c.lower(): c for c in mm.columns}
                lt_col = low.get("lead_time") or low.get("scm_lead_time")
                if lt_col:
                    mm[lt_col] = pd.to_numeric(mm[lt_col], errors="coerce")
                    mm = mm.sort_values(["Product_ID", lt_col]).drop_duplicates("Product_ID")
                    inv = inv.merge(
                        mm[["Product_ID", lt_col]].rename(columns={lt_col: "lead_time"}),
                        on="Product_ID", how="left"
                    )
            inv["lead_time"] = pd.to_numeric(inv.get("lead_time", 0), errors="coerce").fillna(0).astype(int)

            # ========= PANEL DE CÁLCULO SS / ROP =========
            c_m1, c_m2, c_m3 = st.columns([1.4, 1.1, 1.1])
            modo_ss = c_m1.selectbox("Modo de Stock de seguridad", ["Dinámico (recomendado)", "Fijo (1.5 días)"], index=0)
            redondeo = c_m2.selectbox("Redondeo SS", ["Ceil (recomendado)", "Nearest"], index=0)
            ss_min = c_m3.number_input("SS mínimo si μ>0 y L>0", min_value=0, value=1, step=1)

            nivel_serv = st.selectbox(
                "Nivel de servicio (z-score)",
                ["90% (z≈1.28)", "95% (z≈1.65)", "97.5% (z≈1.96)", "99% (z≈2.33)"],
                index=1
            )

            def _z(lbl: str) -> float:
                if "90%" in lbl: return 1.2816
                if "95%" in lbl: return 1.6449
                if "97.5%" in lbl: return 1.9599
                if "99%" in lbl: return 2.3263
                return 1.6449

            z = _z(nivel_serv)
            use_dynamic = modo_ss.startswith("Dinámico")
            use_ceil = redondeo.startswith("Ceil")
            today = pd.Timestamp.today().normalize()

            def _ss_muL_row(pid: str, mu_d: float, L: int) -> tuple[float, float]:
                """Devuelve (SS, mu_L) por SKU para el modo seleccionado."""
                mu_d = float(mu_d or 0.0)
                L = max(int(L or 0), 0)
                if L == 0:
                    return (0.0 if use_dynamic else 1.5 * mu_d), 0.0

                if not use_dynamic:
                    return 1.5 * mu_d, mu_d * L

                # Dinámico: z * sigma_L con estacionalidad por DOW (si existe)
                m = season_map.get(str(pid)) if season_map else None
                # Fallbacks robustos para sigma_d
                CV_MIN = 0.45
                sigma_d = (m.get("sigma") if m else None)
                if sigma_d is None or not np.isfinite(sigma_d) or sigma_d <= 0:
                    sigma_d = CV_MIN * max(mu_d, 0.0)
                sigma_d = max(sigma_d, np.sqrt(max(mu_d, 0.0)))  # proxy Poisson si hace falta

                if m and "s_dow" in m:
                    s_dow = m["s_dow"]
                    factors = [float(s_dow.get(int((today + pd.Timedelta(days=i)).dayofweek), 1.0)) for i in range(L)]
                else:
                    factors = [1.0] * L

                mu_days = [mu_d * f for f in factors]
                var_days = [(sigma_d * f) ** 2 for f in factors]
                mu_L = float(np.sum(mu_days))
                sigma_L = float(np.sqrt(np.sum(var_days)))
                ss = float(z * sigma_L)
                return ss, mu_L

            # --- calcular SS/ROP con controles y suelos
            mu_d_np = pd.to_numeric(inv["demanda_media_dia"], errors="coerce").fillna(0.0).to_numpy()
            L_np = pd.to_numeric(inv["lead_time"], errors="coerce").fillna(0).astype(int).to_numpy()
            pid_np = inv["Product_ID"].astype(str).to_numpy()

            ss_vals = np.zeros(len(inv), dtype=float)
            muL_vals = np.zeros(len(inv), dtype=float)
            for i in range(len(inv)):
                ss_v, muL_v = _ss_muL_row(pid_np[i], mu_d_np[i], L_np[i])
                ss_vals[i] = max(ss_v, 0.0)
                muL_vals[i] = max(muL_v, 0.0)

            # redondeo del SS y suelo mínimo
            ss_int = (np.ceil(ss_vals) if use_ceil else np.rint(ss_vals)).astype(int)
            mask_pos = (mu_d_np > 0) & (L_np > 0)
            ss_int = np.where(mask_pos, np.maximum(ss_int, ss_min), ss_int)

            inv["ss"] = ss_int
            inv["ROP"] = (np.rint(muL_vals).astype(int) + inv["ss"]).astype(int)

            # --- Cobertura (días) = floor(Stock Real / μ_d)
            denom = inv["demanda_media_dia"].replace(0, np.nan)
            inv["Cobertura (días)"] = (
                pd.to_numeric(inv["Stock Real"], errors="coerce") / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0)
            inv["Cobertura (días)"] = np.floor(inv["Cobertura (días)"]).astype(int)

            # --- flags (colores)
            stock_num = pd.to_numeric(inv["Stock Real"], errors="coerce").fillna(0)
            rop_num = pd.to_numeric(inv["ROP"], errors="coerce").fillna(0)
            inv["Rotura"] = stock_num < 0
            inv["Bajo ROP"] = (stock_num >= 0) & (stock_num < rop_num)

            # ========= Filtros / buscador =========
            c1, c2 = st.columns([2, 1])
            q = c1.text_input("Buscar por Product_ID, Nombre o Proveedor", placeholder="Ej. 1003, Whey, ProveedorX").strip()

            prov_col = "Proveedor" if "Proveedor" in inv.columns else None
            proveedores = ["(Todos)"]
            if prov_col:
                proveedores += sorted(pd.Series(inv[prov_col].astype(str)).fillna("").replace("nan", "").unique().tolist())
            prov_sel = c2.selectbox("Proveedor", proveedores, index=0)

            inv_f = inv.copy()
            if q:
                q_ci = q.casefold()
                def _ci_contains(s):
                    return s.astype(str).fillna("").str.casefold().str.contains(q_ci, na=False)
                mask = pd.Series(False, index=inv_f.index)
                for col in ["Product_ID", "Nombre", "Proveedor"]:
                    if col in inv_f.columns:
                        mask = mask | _ci_contains(inv_f[col])
                inv_f = inv_f[mask]
            if prov_col and prov_sel != "(Todos)":
                inv_f = inv_f[inv_f[prov_col].astype(str) == prov_sel]

            st.caption(f"Resultados: **{len(inv_f)}** fila(s)")

            # ---- Mostrar tabla con nombres bonitos y colores
            cols = [c for c in [
                "Product_ID","Nombre","Proveedor","Categoria","Stock Real",
                "demanda_media_dia","lead_time","ss","ROP","Cobertura (días)",
                "Rotura","Bajo ROP"
            ] if c in inv_f.columns]
            df_view = inv_f[cols].copy().rename(columns={
                "demanda_media_dia": "Demanda media diaria",
                "lead_time": "Lead Time",
                "ss": "Stock de Seguridad"
            })

            def _row_style(row):
                try:
                    stock = float(row.get("Stock Real", 0))
                    rop = float(row.get("ROP", 0))
                except Exception:
                    stock, rop = 0, 0
                if stock < 0:
                    return ["background-color: #ffe5e5"] * len(row)  # rojo suave
                elif stock < rop:
                    return ["background-color: #fff3cd"] * len(row)  # naranja/amarillo
                return [""] * len(row)

            styled = df_view.style.apply(_row_style, axis=1)
            st.dataframe(styled, use_container_width=True, height=520)

            # ---- Leyenda
            st.markdown(
                """
                **Leyenda:**  
                - 🟥 *Rojo* = **Rotura de stock** (Stock Real < 0)  
                - 🟧 *Naranja* = **Stock por debajo del ROP** (recomendado reaprovisionar)
                """
            )


    
    

    # ---------------- TAB: 📒 Ledger ----------------
    with tab_ledger:
        st.subheader("📒 Ledger de movimientos")

        CANON_COLS = ["Date", "Product_ID", "Nombre", "Proveedor", "Tipo movimiento", "qty_pedido"]
        ledger_path = OUT10 / "ledger_movimientos.csv"

        def _norm_pid(x) -> str:
            return str(x).strip().replace(".0", "")

        def _enrich_with_master(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame(columns=CANON_COLS)
            master = build_product_master(ROOT)
            if master is None or master.empty:
                for c in CANON_COLS:
                    if c not in df.columns:
                        df[c] = np.nan
                return df[CANON_COLS]
            master = master[["Product_ID", "Nombre", "Proveedor"]].copy()
            master["Product_ID"] = master["Product_ID"].astype(str).map(_norm_pid)
            df = df.merge(master, on="Product_ID", how="left", suffixes=("", "_m"))
            for c in ["Nombre", "Proveedor"]:
                base = df[c].astype(str) if c in df.columns else ""
                df[c] = base.where(base.notna() & (base != "") & (base != "None"), df[f"{c}_m"])
                df.drop(columns=[f"{c}_m"], inplace=True, errors="ignore")
            return df

        def _canonize_existing_ledger(path: Path) -> pd.DataFrame:
            try:
                led = pd.read_csv( 
                    ledger_path,
                    dtype={"Product_ID": str},
                    na_values=["", "None", "nan"],
                    keep_default_na=True,
                    low_memory=False
                )

                # Normaliza numéricos (si existen)
                for c in ["qty_pedido", "on_prev", "on_new", "delta"]:
                    if c in led.columns:
                        led[c] = pd.to_numeric(led[c], errors="coerce")
                
                # Normaliza texto (si existe)
                for c in ["Tipo", "Tipo movimiento", "Nombre", "Proveedor"]:
                    if c in led.columns:
                        led[c] = led[c].astype(str)
            except Exception:
                return pd.DataFrame(columns=CANON_COLS)
            if led.empty:
                return pd.DataFrame(columns=CANON_COLS)
            low = {c.lower(): c for c in led.columns}
            date_col = low.get("date") or low.get("timestamp") or ("Date" if "Date" in led.columns else None)
            if not date_col:
                return pd.DataFrame(columns=CANON_COLS)
            out = pd.DataFrame()
            out["Date"] = pd.to_datetime(led[date_col], errors="coerce").dt.normalize()
            pid_col = low.get("product_id") or low.get("item_id") or ("Product_ID" if "Product_ID" in led.columns else None)
            out["Product_ID"] = led.get(pid_col, pd.Series([], dtype=object)).astype(str).map(_norm_pid) if pid_col else ""
            out["Nombre"] = ""
            out["Proveedor"] = ""
            tipo_c = low.get("tipo movimiento") or low.get("tipo") or None
            if tipo_c:
                    out["Tipo movimiento"] = led[tipo_c].astype(str)
            elif "delta" in low:
                out["Tipo movimiento"] = np.where(
                    pd.to_numeric(led["delta"], errors="coerce").fillna(0) < 0, "Venta", "Entrada"
                ).astype(str)
            else:
                out["Tipo movimiento"] = "Movimiento"
            qty = None
            for cand in ["qty_pedido", "qty", "cantidad", "units"]:
                if cand in low:
                    qty = low[cand]
                    break
            if qty:
                out["qty_pedido"] = pd.to_numeric(led[qty], errors="coerce")
            elif "delta" in low:
                out["qty_pedido"] = pd.to_numeric(led[low["delta"]], errors="coerce").abs()
            else:
                out["qty_pedido"] = 0
            out = out.dropna(subset=["Date"])
            out["Product_ID"] = out["Product_ID"].replace({"nan": np.nan})
            out = out.dropna(subset=["Product_ID"])
            out["qty_pedido"] = pd.to_numeric(out["qty_pedido"], errors="coerce").fillna(0).astype(int)
            return out[CANON_COLS]

        def _historical_to_canon(root: Path) -> pd.DataFrame:
            dem = load_demanda_base(root)
            if dem is None or dem.empty:
                return pd.DataFrame(columns=CANON_COLS)
            df = dem.copy()
            needed = {"Product_ID", "Date", "sales_quantity"}
            if not needed.issubset(df.columns):
                return pd.DataFrame(columns=CANON_COLS)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date", "Product_ID", "sales_quantity"])
            df["Product_ID"] = df["Product_ID"].astype(str).map(_norm_pid)
            df["sales_quantity"] = pd.to_numeric(df["sales_quantity"], errors="coerce").fillna(0)
            cutoff = pd.Timestamp(2025, 9, 19)
            mask = (df["Date"].dt.year.eq(2024)) | (df["Date"].dt.year.eq(2025) & (df["Date"] <= cutoff))
            df = df[mask]
            if df.empty:
                return pd.DataFrame(columns=CANON_COLS)
            df["Date"] = df["Date"].dt.normalize()
            agg = df.groupby(["Product_ID", "Date"], as_index=False)["sales_quantity"].sum()
            agg = agg.rename(columns={"sales_quantity": "qty_pedido"})
            out = pd.DataFrame({
                "Date": agg["Date"],
                "Product_ID": agg["Product_ID"].astype(str).map(_norm_pid),
                "Nombre": "",
                "Proveedor": "",
                "Tipo movimiento": "Venta histórica",
                "qty_pedido": pd.to_numeric(agg["qty_pedido"], errors="coerce").fillna(0).astype(int)
            })
            return out[CANON_COLS]

        with st.expander("🧹 Reconstruir ledger canónico (históricos 2024+2025 hasta 2025-09-19 + movimientos existentes)"):
            st.caption("Reescribe **ledger_movimientos.csv** al esquema: "
                    "`Date, Product_ID, Nombre, Proveedor, Tipo movimiento, qty_pedido`.")
            if st.button("Reconstruir ledger ahora", use_container_width=True, key="rebuild_ledger_btn"):
                hist = _historical_to_canon(ROOT)
                exist = _canonize_existing_ledger(ledger_path) if ledger_path.exists() else pd.DataFrame(columns=CANON_COLS)
                all_df = pd.concat([hist, exist], ignore_index=True)
                all_df = _enrich_with_master(all_df)
                all_df["__key__"] = (
                    all_df["Product_ID"].astype(str) + "|" +
                    pd.to_datetime(all_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d") + "|" +
                    all_df["Tipo movimiento"].astype(str) + "|" +
                    pd.to_numeric(all_df["qty_pedido"], errors="coerce").fillna(0).astype(int).astype(str)
                )
                all_df = all_df.drop_duplicates("__key__").drop(columns="__key__")

                all_df["Date"] = pd.to_datetime(all_df["Date"], errors="coerce").dt.normalize()
                all_df["Product_ID"] = (
                    all_df["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
                )
                all_df["qty_pedido"] = pd.to_numeric(all_df["qty_pedido"], errors="coerce").fillna(0).astype(int)

                # Eliminar filas sin fecha o sin SKU válido
                all_df = all_df.dropna(subset=["Date", "Product_ID"])
                all_df = all_df[all_df["Product_ID"].str.isnumeric()]
                all_df = all_df[all_df["qty_pedido"] > 0]

                all_df = all_df[CANON_COLS].sort_values(["Date", "Product_ID"]).reset_index(drop=True)
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                all_df.to_csv(ledger_path, index=False, encoding="utf-8")
                st.success(f"Ledger reconstruido: **{len(all_df):,}** filas.")
                
                # --------- NUEVO: Limpieza de filas inválidas ----------
                with st.expander("🧽 Limpiar filas inválidas del ledger (1 clic)"):
                    st.caption("Elimina filas con Product_ID no numérico, qty_pedido ≤ 0 o fecha inválida y reescribe el CSV.")
                    if st.button("Eliminar filas inválidas ahora", use_container_width=True, key="cleanup_ledger_btn"):
                        try:
                            df = pd.read_csv(ledger_path, dtype={"Product_ID": str})

                            # Normalización mínima
                            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
                            df["Product_ID"] = df["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
                            df["qty_pedido"] = pd.to_numeric(df.get("qty_pedido", 0), errors="coerce").fillna(0).astype(int)

                            before = len(df)

                            # --- REGLAS DE LIMPIEZA ---
                            df = df.dropna(subset=["Date", "Product_ID"])
                            df = df[df["Product_ID"].str.isnumeric()]     # quita 'Venta histórica', vacíos, etc.
                            df = df[df["qty_pedido"] > 0]                 # quita qty = 0

                            # Guardar limpio y ordenado
                            df = df[CANON_COLS].sort_values(["Date", "Product_ID"]).reset_index(drop=True)
                            df.to_csv(ledger_path, index=False, encoding="utf-8")

                            removed = before - len(df)
                            st.success(f"Listo ✅  Eliminadas {removed:,} filas inválidas. Recarga la página si no se actualiza.")
                        except Exception as e:
                            st.error("No se pudo limpiar el ledger.")
                            st.exception(e)

        if not ledger_path.exists():
            st.info("Aún no hay ledger. Reconstrúyelo o genera movimientos.")
            led = pd.DataFrame(columns=CANON_COLS)
        else:
            try:
                led = pd.read_csv(ledger_path, dtype={"Product_ID": str})
            except Exception:
                st.error("No se pudo leer el ledger.")
                led = pd.DataFrame(columns=CANON_COLS)

        if not led.empty:
            led["Date"] = pd.to_datetime(led["Date"], errors="coerce").dt.normalize()
            led["Product_ID"] = led["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
            led["qty_pedido"] = pd.to_numeric(led["qty_pedido"], errors="coerce").fillna(0).astype(int)
        else:
            st.info("Ledger vacío.")

        c0, c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.2, 1])
        q_pid = c0.text_input("Product_ID (contiene)", placeholder="Ej. 1003, 43…", key="ledger_pid").strip()
        prov_list = ["(Todos)"]
        if "Proveedor" in led.columns and not led.empty:
            prov_list += sorted(pd.Series(led["Proveedor"]).dropna().astype(str).unique().tolist())
        prov_sel = c1.selectbox("Proveedor", prov_list, index=0, key="ledger_proveedor")

        # 🔽 nuevo: selector Tipo movimiento
        tipo_list = ["(Todos)"]
        if "Tipo movimiento" in led.columns and not led.empty:
            tipo_list += sorted(pd.Series(led["Tipo movimiento"]).dropna().astype(str).unique().tolist())
        tipo_sel = c4.selectbox("Tipo movimiento", tipo_list, index=0, key="ledger_tipo")

        min_date = pd.to_datetime(led["Date"]).min() if not led.empty else None
        max_date = pd.to_datetime(led["Date"]).max() if not led.empty else None
        f_ini = c2.date_input("Desde", value=(min_date.date() if pd.notna(min_date) else None), key="ledger_desde")
        f_fin = c3.date_input("Hasta", value=(max_date.date() if pd.notna(max_date) else None), key="ledger_hasta")
        lf = led.copy()
        if q_pid:
            q_ci = q_pid.casefold()
            lf = lf[lf["Product_ID"].astype(str).str.casefold().str.contains(q_ci, na=False)]
        if prov_sel != "(Todos)" and "Proveedor" in lf.columns:
            lf = lf[lf["Proveedor"].astype(str) == prov_sel]
        if tipo_sel != "(Todos)" and "Tipo movimiento" in lf.columns:
            lf = lf[lf["Tipo movimiento"].astype(str) == tipo_sel]
        
        # --- Filtrar por rango de fechas ---
        start = pd.to_datetime(f_ini) if f_ini else None
        end   = pd.to_datetime(f_fin) if f_fin else None

        if "Date" in lf.columns and start is not None and end is not None:
            lf = lf[(lf["Date"] >= start) & (lf["Date"] <= end)]
        st.caption(f"Líneas en vista: **{len(lf):,}**")
        st.dataframe(
            lf[CANON_COLS].sort_values("Date", ascending=False),
            use_container_width=True,
            height=460,
        )
        st.download_button(
            "⬇️ Descargar ledger (CSV)",
            data=lf[CANON_COLS].to_csv(index=False).encode("utf-8"),
            file_name=f"ledger_canonico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ==================================================
# 7) BLOQUE REAPRO (intacto)
# ==================================================

# ====================== Reapro - helpers ======================

@st.cache_data(ttl=300, show_spinner=False)
def _load_forecast_daily_scenario(root: Path, scenario: str) -> pd.DataFrame:
    """
    Carga DEMANDA DIARIA 2025 según escenario exacto y la normaliza a:
      ['Date','Product_ID','sales_quantity'].

    Tus archivos traen: ['date','cluster_id','y_pred','product_id','y_pred_estacional'].
    Nosotros usamos 'y_pred_estacional' como cantidad diaria.
    """
    mapping = {
        "Neutro":     root / "data/processed/predicciones_2025_estacional.parquet",
        "Pesimista":  root / "data/processed/predicciones_2025_pesimista.parquet",
        "Optimista":  root / "data/processed/predicciones_2025_optimista.parquet",
    }
    p = mapping.get(scenario, mapping["Neutro"])
    if not p.exists():
        return pd.DataFrame(columns=["Date","Product_ID","sales_quantity"])

    # Lee parquet/csv
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    # Renombrado explícito a canónico
    # - date -> Date
    # - product_id -> Product_ID
    # - y_pred_estacional -> sales_quantity
    ren = {}
    low = {c.lower(): c for c in df.columns}
    if "date" in low:         ren[low["date"]] = "Date"
    if "product_id" in low:   ren[low["product_id"]] = "Product_ID"
    if "y_pred_estacional" in low:
        ren[low["y_pred_estacional"]] = "sales_quantity"
    else:
        # fallback por si viniera como y_pred (no es tu caso, pero por robustez)
        if "y_pred" in low:
            ren[low["y_pred"]] = "sales_quantity"

    df = df.rename(columns=ren)

    # Nos quedamos con solo las 3 columnas necesarias
    cols_needed = ["Date", "Product_ID", "sales_quantity"]
    for c in cols_needed:
        if c not in df.columns:
            return pd.DataFrame(columns=cols_needed)

    out = df[cols_needed].copy()
    out["Product_ID"]     = _to_pid_str(out["Product_ID"])
    out["Date"]           = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["sales_quantity"] = pd.to_numeric(out["sales_quantity"], errors="coerce").fillna(0.0)
    return out

def _load_lead_times(root: Path) -> pd.DataFrame:
    """
    Lee lead times por Product_ID o por Proveedor. Columnas admitidas:
      - [Product_ID, lead_time]  o  [Proveedor, lead_time]
    """
    candidatos = [
        root / "data/processed/lead_times.csv",
        root / "data/clean/lead_times.csv",
        root / "data/processed/proveedor_lead_times.csv",
    ]
    for p in candidatos:
        if p.exists():
            try:
                df = pd.read_csv(p)
                low = {c.lower(): c for c in df.columns}
                pid_col  = low.get("product_id")
                prov_col = low.get("proveedor") or low.get("supplier") or low.get("scm_supplier_id")
                lt_col   = low.get("lead_time") or low.get("leadtime") or low.get("lt_days")
                if lt_col is None:
                    continue
                if pid_col:
                    df = df[[pid_col, lt_col]].rename(columns={pid_col:"Product_ID", lt_col:"lead_time"})
                    df["Product_ID"] = _norm_pid(df["Product_ID"])
                elif prov_col:
                    df = df[[prov_col, lt_col]].rename(columns={prov_col:"Proveedor", lt_col:"lead_time"})
                else:
                    continue
                df["lead_time"] = pd.to_numeric(df["lead_time"], errors="coerce").fillna(7).astype(int)
                return df
            except Exception:
                pass
    return pd.DataFrame(columns=["Product_ID","Proveedor","lead_time"])

def _forecast_over_lt(fore: pd.DataFrame, pid: str, start_date: pd.Timestamp, lt_days: int) -> float:
    """Suma la previsión diaria desde start_date durante lt_days (inclusive)."""
    if fore.empty or lt_days <= 0:
        return 0.0
    end = (start_date + pd.Timedelta(days=lt_days-1)).normalize()
    sub = fore[(fore["Product_ID"] == pid) & (fore["Date"].between(start_date, end))]
    return float(sub["sales_quantity"].sum()) if not sub.empty else 0.0

def _compute_rop_and_ss(f_prevision_lt: float, lt_days: int) -> tuple[float, float]:
    """Heurística por defecto: SS=20%·prev_LT; ROP=prev_LT+SS."""
    ss  = 0.20 * f_prevision_lt
    rop = f_prevision_lt + ss
    return rop, ss


# ==================== OC helpers (cabecera y líneas) ====================
OC_CUR_HDR = OUT10 / "oc_en_curso.csv"
OC_CUR_LIN = OUT10 / "oc_en_curso_lineas.csv"
OC_REC_HDR = OUT10 / "oc_recibidas.csv"
OC_REC_LIN = OUT10 / "oc_recibidas_lineas.csv"

def _oc_read(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()

def _oc_write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def _oc_read_cur_hdr() -> pd.DataFrame:
    return _oc_read(OC_CUR_HDR, ["order_id","Proveedor","Fecha","Escenario","ETA_dias"])


def _oc_read_cur_lin() -> pd.DataFrame:
    df = _oc_read(OC_CUR_LIN, ["order_id","Product_ID","Nombre","Cantidad_pedir"])
    df["Product_ID"] = df["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    df["Cantidad_pedir"] = pd.to_numeric(df["Cantidad_pedir"], errors="coerce").fillna(0).astype(int)
    return df

def _oc_read_rec_hdr() -> pd.DataFrame:
    return _oc_read(OC_REC_HDR, ["order_id","Proveedor","Fecha","Escenario","ETA_dias","Fecha_recepcion"])

def _oc_read_rec_lin() -> pd.DataFrame:
    df = _oc_read(OC_REC_LIN, ["order_id","Product_ID","Nombre","Cantidad_pedir"])
    df["Product_ID"] = df["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    df["Cantidad_pedir"] = pd.to_numeric(df["Cantidad_pedir"], errors="coerce").fillna(0).astype(int)
    return df

def _make_order_id(proveedor: str) -> str:
    ts = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    base = str(proveedor).strip().replace(" ", "_")[:16]
    return f"OC-{base}-{ts}"

# ======================= helper: ingresar orden en inventario + ledger =======================
# ======================= helper: ingresar orden en inventario + ledger =======================
def _ingresar_orden_en_inventario_y_ledger(order_id: str) -> bool:
    """
    Recibe una orden y actualiza:
      - OUT10/inventory_updated.csv  (suma cantidades)
      - OUT10/ledger_movimientos.csv (Tipo movimiento = Compra)
    Soporta dos esquemas de almacenamiento:
      1) OC_HDR / OC_LIN
      2) CUR/REC: OC_CUR_HDR/OC_CUR_LIN -> mueve a OC_REC_HDR/OC_REC_LIN
    """

    # -------- 1) ¿En qué esquema está la orden? --------
    # a) Esquema “unificado” (OC_HDR / OC_LIN)
    try:
        hdr_uni = _oc_read_hdr()
        lin_uni = _oc_read_lin()
    except Exception:
        hdr_uni = pd.DataFrame(columns=["order_id"])
        lin_uni = pd.DataFrame(columns=["order_id", "Product_ID", "Nombre", "Cantidad_pedir"])

    en_unico = order_id in set(hdr_uni.get("order_id", []))

    # b) Esquema CUR/REC
    try:
        hdr_cur = _oc_read_cur_hdr()
        lin_cur = _oc_read_cur_lin()
        en_currec = order_id in set(hdr_cur.get("order_id", []))
    except Exception:
        hdr_cur = lin_cur = None
        en_currec = False

    if not en_unico and not en_currec:
        return False  # no existe la orden en ninguno de los esquemas

    # -------- 2) Tomar líneas y normalizar cantidades --------
    lines = (lin_uni if en_unico else lin_cur)
    lines = lines[lines.get("order_id") == order_id].copy()
    if lines.empty:
        return False

    lines["Product_ID"] = (
        lines["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    )
    # cantidades EN SERIE y > 0
    lines["Cantidad_pedir"] = (
        pd.to_numeric(lines.get("Cantidad_pedir", 0), errors="coerce")
        .fillna(0).round().astype(int)
    )
    lines = lines[lines["Cantidad_pedir"] > 0]
    if lines.empty:
        return False

    # -------- 3) Actualizar inventario vivo --------
    inv = _read_working_inventory().copy()
    if inv.empty:
        return False

    inv["Product_ID"] = inv["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    inv["Stock Real"] = pd.to_numeric(inv["Stock Real"], errors="coerce").fillna(0).astype(int)
    inv = inv.set_index("Product_ID")

    add_by_pid = lines.groupby("Product_ID", as_index=True)["Cantidad_pedir"].sum()
    for pid, qty in add_by_pid.items():
        pid = str(pid).strip()
        if pid in inv.index:
            inv.loc[pid, "Stock Real"] = int(inv.loc[pid, "Stock Real"]) + int(qty)
        else:
            inv.loc[pid, "Stock Real"] = int(qty)

    inv = inv.reset_index()
    _oc_write(OUT10 / "inventory_updated.csv", inv)  # guarda actualizado

    # -------- 4) Registrar en ledger --------
    mov = (
        lines[["Product_ID", "Cantidad_pedir"]]
        .rename(columns={"Cantidad_pedir": "qty_pedido"})
        .copy()
    )
    mov["Date"] = pd.Timestamp.today().normalize()
    mov["Tipo movimiento"] = "Compra"
    if not mov.empty:
        append_ledger(mov)

    # -------- 5) Marcar como recibida / mover entre tablas --------
    if en_unico:
        hdr_uni.loc[hdr_uni["order_id"] == order_id, "Estado"] = "recibida"
        _oc_write_hdr(hdr_uni)
    else:
        hdr_rec = _oc_read_rec_hdr()
        lin_rec = _oc_read_rec_lin()

        hdr_sel = hdr_cur[hdr_cur["order_id"] == order_id].copy()
        hdr_sel["Fecha_recepcion"] = pd.Timestamp.today().strftime("%Y-%m-%d")

        _oc_write(OC_REC_HDR, pd.concat([hdr_rec, hdr_sel], ignore_index=True))
        _oc_write(OC_REC_LIN, pd.concat([lin_rec, lines], ignore_index=True))

        # eliminar de “en curso”
        _oc_write(OC_CUR_HDR, hdr_cur[hdr_cur["order_id"] != order_id].copy())
        _oc_write(OC_CUR_LIN, lin_cur[lin_cur["order_id"] != order_id].copy())
        # si usas oc_en_curso.csv para filtros/alertas, intenta también quitar la orden allí
        try:
            _oc_sync_remove_cur(order_id)
        except Exception:
            pass

    return True


# =============== BLOQUE: 📦 Órdenes en curso ===============
def render_ordenes_en_curso():
    hdr = _oc_read_hdr()
    if "Estado" not in hdr.columns:
        hdr["Estado"] = "en_curso"
    lin = _oc_read_lin()
    en_curso = hdr[hdr["Estado"] == "en_curso"].copy()

    if en_curso.empty:
        st.info("No hay órdenes en curso.")
        return

    st.dataframe(
        en_curso[["order_id","Proveedor","Fecha","Escenario","ETA_dias","Estado"]],
        use_container_width=True,
        height=220,
    )

    oc_ids = en_curso["order_id"].tolist()
    oc_sel = st.selectbox("Ver detalle de orden", oc_ids, index=0)
    det = lin[lin["order_id"] == oc_sel].copy()

    if det.empty:
        st.info("Esta orden no tiene líneas.")
        return

    if "Nombre" not in det.columns or det["Nombre"].isna().all():
        master = build_product_master(ROOT)[["Product_ID","Nombre"]].drop_duplicates("Product_ID")
        master["Product_ID"] = _norm_pid(master["Product_ID"])
        det = det.merge(master, on="Product_ID", how="left")

    st.subheader(f"Detalle {oc_sel}")
    st.dataframe(det[["Product_ID","Nombre","Cantidad_pedir"]], use_container_width=True, height=280)

    if st.button("✅ Forzar recepción", type="primary", use_container_width=True):
        ok = _ingresar_orden_en_inventario_y_ledger(oc_sel)
        if ok:
            st.success("Recepción forzada. Inventario y ledger actualizados. La orden pasa a 'recibidas'.")
            try: st.cache_data.clear()
            except: pass
            st.rerun()
        else:
            st.warning("No se pudo procesar la orden (revisa que tenga líneas y cantidades > 0).")


# =============== BLOQUE: 📥 Órdenes recibidas ===============
def render_ordenes_recibidas():
    hdr = _oc_read_hdr()
    rec = hdr[hdr["Estado"] == "recibida"].copy()

    if rec.empty:
        st.info("Aún no hay órdenes recibidas.")
        return

    if "Fecha" in rec.columns:
        rec["Fecha"] = pd.to_datetime(rec["Fecha"], errors="coerce")
        rec = rec.sort_values("Fecha", ascending=False)

    st.dataframe(
        rec[["order_id","Proveedor","Fecha","Escenario","ETA_dias","Estado"]],
        use_container_width=True,
        height=360,
    )


# ======================= BLOQUE PRINCIPAL: 🧰 Reapro / Pedidos =======================
def render_reapro():
    st.title("🧰 Reapro / Pedidos")

    tab_rank, tab_en_curso, tab_rec, tab_sim = st.tabs(
        ["📊 Ranking & Sugerencias", "📦 Órdenes en curso", "📥 Órdenes recibidas", "🔮 Simulador demanda"]
    )

    # ------------------------ TAB: Ranking & Sugerencias ------------------------
    with tab_rank:
        escenario = st.selectbox("Escenario de previsión", ["Neutro", "Pesimista", "Optimista"], index=0)

        # ===== Inventario =====
        inv = _read_working_inventory()
        if inv.empty:
            st.warning("Inventario vivo vacío.")
            st.stop()

        inv["Product_ID"] = _to_pid_str(inv["Product_ID"])
        inv["Stock Real"] = pd.to_numeric(inv["Stock Real"], errors="coerce").fillna(0).astype(int)

        # ===== Demanda =====
        fore = load_demanda_base(ROOT)
        if fore is None:
            fore = pd.DataFrame(columns=["Product_ID", "Date", "sales_quantity"])
        else:
            fore = fore.copy()
            fore["Product_ID"] = _to_pid_str(fore["Product_ID"])
            fore["sales_quantity"] = pd.to_numeric(fore["sales_quantity"], errors="coerce").fillna(0)

        # ===== Catálogo (lead times) =====
        lt_df = load_supplier_catalog_multi(ROOT)

        # ===== Base de trabajo =====
        base_cols = ["Product_ID", "Nombre", "Proveedor"]
        work = inv[base_cols + ["Stock Real"]].copy().rename(columns={"Stock Real": "stock_actual"})

        # Lead time por Product_ID o por Proveedor
        lt_by_pid, lt_by_prov = {}, {}
        if lt_df is not None and not lt_df.empty:
            tmp = lt_df.copy()
            if "Product_ID" in tmp.columns:
                tmp["Product_ID"] = _to_pid_str(tmp["Product_ID"])
            if "scm_lead_time" in tmp.columns:
                tmp["scm_lead_time"] = pd.to_numeric(tmp["scm_lead_time"], errors="coerce").fillna(7).astype(int)
                if "Product_ID" in tmp.columns:
                    lt_by_pid = dict(zip(tmp["Product_ID"], tmp["scm_lead_time"]))
            if "scm_supplier_id" in tmp.columns and "scm_lead_time" in tmp.columns:
                lt_by_prov = dict(zip(tmp["scm_supplier_id"].astype(str), tmp["scm_lead_time"]))

        work["lead_time"] = (
            work["Product_ID"].map(lt_by_pid) if lt_by_pid else work["Proveedor"].map(lt_by_prov)
        ).fillna(7)

        # Ventas medias
        if not fore.empty:
            daily_mean = (
                fore.groupby("Product_ID", as_index=False)["sales_quantity"]
                .mean()
                .rename(columns={"sales_quantity": "sales_mean"})
            )
            work = work.merge(daily_mean, on="Product_ID", how="left")
            work["sales_mean"] = work["sales_mean"].fillna(0)
        else:
            work["sales_mean"] = 0.0

        # Cálculos ROP/SS básicos
        work["prev_lt"] = (work["sales_mean"] * pd.to_numeric(work["lead_time"], errors="coerce").fillna(7)).clip(lower=0)
        work["stock_seguridad"] = (work["sales_mean"] * 0.5 * work["lead_time"]).fillna(0)
        work["ROP"] = (work["sales_mean"] * work["lead_time"] + work["stock_seguridad"]).fillna(0)

        # Flags de urgencia (rotura solo si stock < 0)
        work["rotura"]   = work["stock_actual"] < 0
        work["rop_bajo"] = work["stock_actual"] < work["ROP"]
        work["ss_bajo"]  = work["stock_actual"] < work["stock_seguridad"]

        work["recomendada"] = (work["ROP"] + work["prev_lt"] - work["stock_actual"]).clip(lower=0).round().astype(int)

        # ===== EN CAMINO =====
        en_camino = _load_en_camino_por_pid()
        if en_camino is None or en_camino.empty:
            en_camino = pd.DataFrame(columns=["Product_ID", "en_camino_qty", "eta_min"])
        en_camino["Product_ID"] = _to_pid_str(en_camino.get("Product_ID", ""))
        en_camino["en_camino_qty"] = pd.to_numeric(en_camino.get("en_camino_qty", 0), errors="coerce").fillna(0).astype(int)
        work = work.merge(en_camino[["Product_ID", "en_camino_qty", "eta_min"]], on="Product_ID", how="left")
        work["en_camino_qty"] = work["en_camino_qty"].fillna(0).astype(int)

        # ===== TOP VENTAS (outliers.csv) =====
        tv = _load_top_ventas_from_outliers()
        if tv is not None and not tv.empty:
            work = work.merge(tv, on="Product_ID", how="left")
            work["top_ventas"] = work["top_ventas"].fillna(False).astype(bool)
        else:
            work["top_ventas"] = False

        # ===== SUSTITUTOS =====
        subs_sum = _load_subs_summary()
        if subs_sum is not None and not subs_sum.empty:
            work = work.merge(subs_sum, on="Product_ID", how="left")
        else:
            work["subs_count"] = 0
            work["subs_ids"] = ""

        work["subs_count"] = pd.to_numeric(work.get("subs_count", 0), errors="coerce").fillna(0).astype(int)
        work["subs_ids"] = work.get("subs_ids", "").fillna("")

        # ===== Ranking de proveedores =====
        prov_rank = (
            work.groupby("Proveedor")
                .agg(roturas=("rotura","sum"), rop_bajos=("rop_bajo","sum"), ss_bajos=("ss_bajo","sum"), total=("Product_ID","count"))
                .reset_index()
                .sort_values(["roturas","rop_bajos","ss_bajos","total"], ascending=[False, False, False, True])
        )

        st.subheader("📦 Proveedores priorizados")
        st.dataframe(prov_rank, use_container_width=True, height=260)

        # ===== Selector y filtro de detalle =====
        c1, c2 = st.columns([1,1])
        prov_sel = c1.selectbox("Ver detalle de proveedor", prov_rank["Proveedor"].tolist() if not prov_rank.empty else [], index=0 if not prov_rank.empty else None)
        solo_criticos = c2.toggle("Solo críticos (rotura / ROP bajo / SS bajo)", value=True)

        if prov_sel:
            dfp = work[work["Proveedor"] == prov_sel].copy()
            if solo_criticos:
                dfp = dfp[dfp["rotura"] | dfp["rop_bajo"] | dfp["ss_bajo"]].copy()

            if dfp.empty:
                st.info("No hay productos críticos para este proveedor en el escenario seleccionado.")
            else:
                # --- chips de alertas (ACUMULABLES)
                def _build_alertas_row(r):
                    chips = []
                    if r.get("rotura", False):
                        chips.append("🔴 Rotura")
                        sc = int(r.get("subs_count", 0) or 0)
                        if sc > 0:
                            ns = str(r.get("subs_ids", "") or "")
                            chips.append("♻️ Sustitutos: " + str(sc) + (f" [{ns}]" if ns else ""))
                    if r.get("rop_bajo", False):
                        chips.append("🟠 ROP bajo")
                    if r.get("ss_bajo", False):
                        chips.append("🟡 SS bajo")
                    ec = int(r.get("en_camino_qty", 0) or 0)
                    if ec > 0:
                        eta = r.get("eta_min", pd.NA)
                        chips.append(f"🚚 En camino: {ec}" + (f" (ETA~{int(eta)}d)" if pd.notna(eta) and str(eta) != "" else ""))
                    if r.get("top_ventas", False):
                        chips.append("🏆 Top ventas")
                    return " · ".join(chips)

                dfp["Alertas"] = dfp.apply(_build_alertas_row, axis=1)

                # Orden por severidad
                dfp = dfp.sort_values(["rotura","rop_bajo","ss_bajo","Product_ID"],
                                      ascending=[False, False, False, True])

                # --- columnas a mostrar
                mostrar = dfp[[
                    "Product_ID","Nombre","stock_actual","prev_lt","recomendada",
                    "en_camino_qty","ROP","stock_seguridad",
                    "subs_count","subs_ids",
                    "top_ventas","Alertas"
                ]].rename(columns={
                    "stock_actual":"Stock actual",
                    "prev_lt":"Previsión (LT)",
                    "recomendada":"Cantidad recomendada",
                    "stock_seguridad":"Stock de seguridad",
                    "en_camino_qty":"En camino",
                    "subs_count":"Sustitutos (n)",
                    "subs_ids":"Sustitutos (IDs)",
                    "top_ventas":"Top ventas",
                }).copy()

                # Asegurar columna editable
                if "Cantidad a pedir" not in mostrar.columns:
                    mostrar["Cantidad a pedir"] = 0
                mostrar["Cantidad a pedir"] = pd.to_numeric(mostrar["Cantidad a pedir"], errors="coerce").fillna(0).astype("Int64")

                # Clave base para widgets
                _preview_key = f"oc_preview_{prov_sel}_{escenario}"
                SEL = "__incluir__"
                if SEL not in mostrar.columns:
                    mostrar[SEL] = False

                st.subheader(f"🧾 Recomendaciones de compra — {prov_sel} · Escenario: {escenario}")

                cfg = {
                    "Product_ID":           st.column_config.TextColumn("Product_ID", disabled=True),
                    "Nombre":               st.column_config.TextColumn("Nombre", disabled=True),
                    "Stock actual":         st.column_config.NumberColumn("Stock actual", format="%d", disabled=True),
                    "Previsión (LT)":       st.column_config.NumberColumn("Previsión (LT)", format="%.0f", disabled=True),
                    "Cantidad recomendada": st.column_config.NumberColumn("Cantidad recomendada", format="%d", disabled=True),
                    "En camino":            st.column_config.NumberColumn("En camino", format="%d", disabled=True),
                    "Cantidad a pedir":     st.column_config.NumberColumn("Cantidad a pedir", min_value=0, step=1),
                    "ROP":                  st.column_config.NumberColumn("ROP", format="%.0f", disabled=True),
                    "Stock de seguridad":   st.column_config.NumberColumn("Stock de seguridad", format="%.0f", disabled=True),
                    "Sustitutos (n)":       st.column_config.NumberColumn("Sustitutos (n)", format="%d", disabled=True),
                    "Sustitutos (ejemplos)":st.column_config.TextColumn("Sustitutos (ejemplos)", disabled=True),
                    "Top ventas":           st.column_config.CheckboxColumn("🏆 Top ventas", disabled=True),
                    "Alertas":              st.column_config.TextColumn("Alertas", disabled=True),
                    SEL:                    st.column_config.CheckboxColumn("✓ Incluir"),
                }

                edited = st.data_editor(
                    mostrar,
                    use_container_width=True,
                    height=420,
                    hide_index=True,
                    key=f"reapro_{prov_sel}_{escenario}",
                    column_config=cfg,
                    disabled=False,
                    num_rows="fixed",
                )


                # ---- Acciones preview (con keys únicos)
                if _preview_key not in st.session_state:
                    st.session_state[_preview_key] = pd.DataFrame(
                        columns=["Product_ID", "Nombre", "Cantidad_pedir"]
                    )

                c_sel_add, c_sel_clear = st.columns([1, 1])
                if c_sel_add.button("➕ Añadir seleccionadas al preview",
                                    key=f"{_preview_key}_add_top",
                                    use_container_width=True):
                    tmp = edited.copy()
                    tmp["Cantidad a pedir"] = pd.to_numeric(tmp["Cantidad a pedir"], errors="coerce").fillna(0).astype(int)
                    tmp = tmp[(tmp[SEL] == True) & (tmp["Cantidad a pedir"] > 0)][
                        ["Product_ID", "Nombre", "Cantidad a pedir"]
                    ].rename(columns={"Cantidad a pedir": "Cantidad_pedir"})

                    if not tmp.empty:
                        prev = st.session_state[_preview_key]
                        prev = pd.concat([prev, tmp], ignore_index=True)
                        prev = (
                            prev.groupby(["Product_ID", "Nombre"], as_index=False)["Cantidad_pedir"]
                            .sum()
                            .sort_values(["Product_ID"])
                        )
                        st.session_state[_preview_key] = prev
                        st.success(f"Añadidas {len(tmp)} línea(s) al preview.")
                    else:
                        st.info("Marca alguna línea y pon 'Cantidad a pedir' > 0.")

                if c_sel_clear.button("🧹 Vaciar preview",
                                      key=f"{_preview_key}_clear_top",
                                      use_container_width=True):
                    st.session_state[_preview_key] = pd.DataFrame(
                        columns=["Product_ID", "Nombre", "Cantidad_pedir"]
                    )
                    st.info("Preview vaciado.")

                # ---- Preview
                st.markdown("#### 🧾 Orden de compra definitiva (preview)")
                prev = st.session_state[_preview_key]
                if prev.empty:
                    st.caption("No hay líneas todavía. Marca filas arriba y pulsa **Añadir seleccionadas al preview**.")
                else:
                    st.dataframe(prev, use_container_width=True, height=220)

                # ---- Botonera inferior del preview
                cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])
                if cc1.button("🚀 Lanzar orden con el preview",
                              key=f"{_preview_key}_launch",
                              use_container_width=True):
                    if prev.empty:
                        st.warning("El preview está vacío.")
                    else:
                        order_id = _make_order_id(prov_sel)
                        hdr_cur = _oc_read_hdr()
                        eta_dias = int(dfp["lead_time"].iloc[0]) if "lead_time" in dfp.columns and not dfp.empty else 7
                        new_hdr = pd.DataFrame([{
                            "order_id": order_id,
                            "Proveedor": prov_sel,
                            "Fecha": pd.Timestamp.today().strftime("%Y-%m-%d"),
                            "Escenario": escenario,
                            "ETA_dias": eta_dias,
                            "Estado": "en_curso",
                        }])
                        _oc_write_hdr(pd.concat([hdr_cur, new_hdr], ignore_index=True))

                        lin_cur = _oc_read_lin()
                        new_lin = prev.copy()
                        new_lin.insert(0, "order_id", order_id)
                        _oc_write_lin(pd.concat([lin_cur, new_lin], ignore_index=True))

                        # sincronizar en_curso.csv para alertas
                        try:
                            _oc_sync_append_cur(new_hdr, new_lin)
                        except Exception as e:
                            st.warning(f"No se pudo actualizar 'oc_en_curso.csv': {e}")

                        st.session_state[_preview_key] = pd.DataFrame(
                            columns=["Product_ID", "Nombre", "Cantidad_pedir"]
                        )
                        st.success(f"Orden {order_id} creada en 'Órdenes en curso'.")

                if cc2.button("🧹 Vaciar preview",
                              key=f"{_preview_key}_clear_bottom",
                              use_container_width=True):
                    st.session_state[_preview_key] = pd.DataFrame(
                        columns=["Product_ID", "Nombre", "Cantidad_pedir"]
                    )
                    st.info("Preview vaciado.")

                if cc3.button("🗑️ Eliminar seleccionadas del preview",
                              key=f"{_preview_key}_del_sel",
                              use_container_width=True):
                    st.info("De momento, usa 'Vaciar preview' y vuelve a añadir solo las deseadas.")

                if cc4.button("🗓️ Retrasar / Descartar orden",
                              key=f"{_preview_key}_postpone",
                              use_container_width=True):
                    st.info("Orden descartada por ahora. Podrás revisarla mañana de nuevo.")

    # ------------------------ TAB: Órdenes en curso ------------------------
    with tab_en_curso:
        hdr = _oc_read_hdr()
        if "Estado" not in hdr.columns:
            hdr["Estado"] = "en_curso"
        lin = _oc_read_lin()
        en_curso = hdr[hdr["Estado"] == "en_curso"].copy()

        if en_curso.empty:
            st.info("No hay órdenes en curso.")
        else:
            st.dataframe(en_curso[["order_id","Proveedor","Fecha","Escenario","ETA_dias","Estado"]],
                         use_container_width=True, height=220)

            oc_ids = en_curso["order_id"].tolist()
            oc_sel = st.selectbox("Ver detalle de orden", oc_ids, index=0)

            det = lin[lin["order_id"] == oc_sel].copy()
            det = _ensure_pid_col(det, prefer_int=False)
            det["Product_ID"] = _to_pid_str(det["Product_ID"])

            if "Cantidad_pedir" not in det.columns:
                low = {c.lower(): c for c in det.columns}
                for alt in ("qty", "cantidad", "cantidad_pedida", "quantity"):
                    if alt in low:
                        det = det.rename(columns={low[alt]: "Cantidad_pedir"})
                        break
            if "Cantidad_pedir" not in det.columns:
                det["Cantidad_pedir"] = 0

            if "Nombre" not in det.columns or det["Nombre"].isna().all() or (det["Nombre"] == "").all():
                master = build_product_master(ROOT)[["Product_ID", "Nombre"]].drop_duplicates("Product_ID")
                master["Product_ID"] = master["Product_ID"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
                det = det.merge(master, on="Product_ID", how="left")
                if "Nombre" not in det.columns:
                    det["Nombre"] = ""

            st.subheader(f"Detalle {oc_sel}")
            st.dataframe(det[["Product_ID","Nombre","Cantidad_pedir"]], use_container_width=True, height=280)

            if st.button("✅ Forzar recepción", type="primary", use_container_width=True, key=f"force_rx_{oc_sel}"):
                ok = _ingresar_orden_en_inventario_y_ledger(oc_sel)
                if ok:
                    st.success("Recepción forzada. Inventario y ledger actualizados. La orden pasó a 'Recibidas'.")
                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.warning("No se pudo procesar la orden (revisa que tenga líneas y cantidades > 0).")

    # ------------------------ TAB: Órdenes recibidas ------------------------
    with tab_rec:
        hdr = _oc_read_hdr()
        rec1 = hdr[hdr.get("Estado") == "recibida"].copy()
        try:
            rec2 = _oc_read_rec_hdr()
        except Exception:
            rec2 = pd.DataFrame(columns=["order_id","Proveedor","Fecha","Escenario","ETA_dias","Estado"])
        rec = pd.concat([rec1, rec2], ignore_index=True)
        if rec.empty:
            st.info("Aún no hay órdenes recibidas.")
        else:
            rec = rec.drop_duplicates(subset=["order_id"], keep="last").copy()
            if "Fecha" in rec.columns:
                rec["Fecha"] = pd.to_datetime(rec["Fecha"], errors="coerce")
                rec = rec.sort_values("Fecha", ascending=False)

            st.subheader("📥 Órdenes recibidas")
            st.dataframe(
                rec[["order_id","Proveedor","Fecha","Escenario","ETA_dias","Estado"]],
                use_container_width=True,
                height=260,
            )

            st.markdown("#### 🔎 Ver detalle de una orden recibida")
            oc_ids = ["— Selecciona una orden —"] + rec["order_id"].tolist()
            oc_sel = st.selectbox("Orden:", oc_ids, index=0)
            if oc_sel != "— Selecciona una orden —":
                try:
                    lin_rec = _oc_read_rec_lin()
                except Exception:
                    lin_rec = pd.DataFrame(columns=["order_id","Product_ID","Nombre","Cantidad_pedir"])
                det = lin_rec[lin_rec.get("order_id") == oc_sel].copy()
                if det.empty:
                    lin = _oc_read_lin()
                    det = lin[lin.get("order_id") == oc_sel].copy()

                det = _ensure_pid_col(det, prefer_int=False)
                det["Product_ID"] = _to_pid_str(det.get("Product_ID", ""))
                if "Cantidad_pedir" not in det.columns:
                    low = {c.lower(): c for c in det.columns}
                    for alt in ("qty", "cantidad", "cantidad_pedida", "quantity"):
                        if alt in low:
                            det.rename(columns={low[alt]: "Cantidad_pedir"}, inplace=True)
                            break
                det["Cantidad_pedir"] = pd.to_numeric(det.get("Cantidad_pedir", 0), errors="coerce").fillna(0).astype(int)

                if "Nombre" not in det.columns or det["Nombre"].isna().all() or (det["Nombre"] == "").all():
                    master = build_product_master(ROOT)[["Product_ID","Nombre"]].drop_duplicates("Product_ID")
                    master["Product_ID"] = _to_pid_str(master["Product_ID"])
                    det = det.merge(master, on="Product_ID", how="left")
                    if "Nombre" not in det.columns:
                        det["Nombre"] = ""

                st.dataframe(
                    det[["Product_ID","Nombre","Cantidad_pedir"]],
                    use_container_width=True,
                    height=280,
                )
    
    # ==================== TAB DE SIMULADOR ====================

    with tab_sim:
        st.subheader("🔮 Simulador de escenarios de demanda")

        forecast = _load_forecast_neutral()
        cmap     = _load_clusters_map()

        # selector de producto (con placeholder)
        prods = forecast["Product_ID"].unique().tolist()
        options = ["--Seleccione o introduzca Product_ID para comenzar simulación--"] + prods
        prod_sel = st.selectbox("Selecciona un producto", options, index=0)

        # detener hasta que el usuario elija un Product_ID válido
        if prod_sel == "--Seleccione o introduzca Product_ID para comenzar simulación--":
            st.info("Selecciona un Product_ID para iniciar la simulación.")
            st.stop()

        if prod_sel:
            dfp = forecast[forecast["Product_ID"] == prod_sel].copy()
            clus = _get_cluster(prod_sel, cmap)
            eps  = _elasticidad(clus)

            st.caption(f"Cluster del producto: **{clus}** · Elasticidad precio: **{eps:.2f}**")

            # ===== Parámetros de simulación (precio + promo + evento) =====
            st.markdown("#### Parámetros de simulación")
            c1, c2, c3 = st.columns([1, 1, 1])

            # 1) Precio
            delta_precio = c1.slider("Δ precio (%)", -50, 50, 0, step=1,
                                    help="Variación relativa del precio. Un -10% reduce el precio; la demanda cambia según la elasticidad del clúster.")

            # 2) Promo
            promo_on = c2.checkbox("Promo (packs/bundles)", value=False,
                                help="No es descuento; no afecta al precio. Impacta directamente a la demanda.")
            promo_pct = 0
            if promo_on:
                promo_pct = st.slider("Impacto promo (% sobre demanda)", 0, 100, 15, step=5,
                                    help="Efecto multiplicativo sobre la demanda. 15% ⇒ x1.15")

            # 3) Evento especial
            evento_on = c3.checkbox("Evento especial", value=False)
            evento_pct = 0
            if evento_on:
                evento_pct = st.slider("Impacto evento (% sobre demanda)", 0, 200, 30, step=10,
                                help="Efecto multiplicativo sobre la demanda. 30% ⇒ x1.30")

            # ===== Factores multiplicativos =====
            M_price_scalar = max(0.01, 1.0 + delta_precio / 100.0) ** eps
            M_promo_scalar = 1.0 + promo_pct / 100.0
            M_event_scalar = 1.0 + evento_pct / 100.0

            # Demanda base y simulada
            dfp["qty_base"] = dfp["qty_forecast"].astype(float)
            dfp["qty_sim"]  = (dfp["qty_base"] * M_price_scalar * M_promo_scalar * M_event_scalar).clip(lower=0.0)

            # ===== Métricas resumen =====
            base_sum = float(dfp["qty_base"].sum())
            sim_sum  = float(dfp["qty_sim"].sum())
            delta_pct_total = (sim_sum / base_sum - 1.0) * 100.0 if base_sum > 0 else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Demanda base (rango)", f"{base_sum:,.0f} uds")
            m2.metric("Demanda simulada", f"{sim_sum:,.0f} uds", delta=f"{delta_pct_total:+.1f}%")
            m3.metric("Δ precio aplicado", f"{delta_precio:+d} %")

            # ===== Gráfica =====
            chart_df = dfp[["Date", "qty_base", "qty_sim"]].set_index("Date").sort_index()
            st.line_chart(chart_df, height=280, use_container_width=True)

            # ===== Tabla detalle =====
            show_tbl = st.toggle("Mostrar detalle diario", value=False)
            if show_tbl:
                det = dfp[["Date","qty_base","qty_sim"]].copy()
                det = det.rename(columns={"qty_base":"Demanda base", "qty_sim":"Escenario modificado"})
                st.dataframe(det, use_container_width=True, height=320)

# ------------------ Render según ruta ----------------------------
route = st.session_state["route"]
if route == "home":
    render_home()
elif route == "exploracion":
    render_exploracion_sustitutos()
elif route == "proveedores":
    render_proveedores()
elif route == "movimientos":
    render_movimientos_stock()
elif route == "reapro":
    render_reapro()
else:
    st.error("Ruta desconocida. Volviendo a portada…")
    goto("home")
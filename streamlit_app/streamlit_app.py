# app/streamlit_app.py

import sys
from pathlib import Path
from datetime import datetime
import io

import streamlit as st
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
            "tipo": "Tipo",
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
    # (… tu bloque de proveedores actual …)
    pass


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

            # (2) Inventario 'vivo' con nombres EXACTOS (Product_ID, Proveedor, Nombre, Categoria, Stock Real)
            inv_live = _read_working_inventory()
            if inv_live.empty:
                st.error("Inventario vivo vacío.")
                st.stop()

            # Asegurar mismo formato de ID en el inventario
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

            # ==================== CAMBIO CLAVE: sin merge, con map ====================
            # (3) Construir diccionario de cantidades pedidas por Product_ID
            qty_by_id = df_orders.set_index("Product_ID")["qty"].to_dict()

            # (4) Calcular stocks: solo resta si el Product_ID aparece en el pedido
            df = inv_live.copy()
            df["on_prev"] = pd.to_numeric(df["Stock Real"], errors="coerce").fillna(0).astype(int)
            df["resta"]   = df["Product_ID"].map(qty_by_id).fillna(0).astype(int)
            df["on_new"]  = (df["on_prev"] - df["resta"]).astype(int)
            df["Δ"]       = df["on_new"] - df["on_prev"]

            # ---- Ledger de movimientos (append) ----
            ledger_path = OUT10 / "ledger_movimientos.csv"
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ledger_cols = ["timestamp", "Product_ID", "Nombre", "Proveedor", "Categoria",
                            "qty_pedido", "on_prev", "on_new", "delta"]

            ledger_df = df[["Product_ID", "Nombre", "Proveedor", "Categoria", "resta", "on_prev", "on_new", "Δ"]].copy()
            ledger_df.insert(0, "timestamp", stamp)
            ledger_df = ledger_df.rename(columns={"resta": "qty_pedido", "Δ": "delta"})

            try:
                if ledger_path.exists():
                    old = pd.read_csv(ledger_path)
                    merged = pd.concat([old, ledger_df[ledger_cols]], ignore_index=True)
                    merged.to_csv(ledger_path, index=False)
                else:
                    ledger_df[ledger_cols].to_csv(ledger_path, index=False)
            
            except Exception as e:
                st.warning(f"No se pudo escribir el ledger: {e}")
            # ========================================================================

            # (5) Reescribir inventario 'vivo' con EXACTAMENTE las 5 columnas (transcribiendo todo)
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


            # (7) Limpieza de caché (y vaciar pedidos UI para no recontar)
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
            f_date = c1.date_input("Fecha", value=datetime.today(), format="YYYY-MM-DD")
            f_item = c2.text_input("Product_ID", placeholder="Ej. 4347")
            f_qty  = c3.number_input("Cantidad", value=1, min_value=1, step=1)
            submitted = c4.form_submit_button("➕ Añadir")
            if submitted:
                pid = str(f_item).strip()
                if pid and f_qty > 0:
                    _append_order_row(f_date.strftime("%Y-%m-%d"), pid, int(f_qty))
                    st.success("Línea añadida al pedido.")
                    try: st.cache_data.clear()
                    except: pass
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
        if df_ui.empty:
            st.info("Aún no hay pedidos en UI.")
            st.stop()

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

        # ---------- Proveedor: ui_products.parquet o fallback al multi-catálogo ----------
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

        # Fila(s) marcadas por POSICIÓN (no necesitamos el índice auxiliar)
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
            st.subheader("Stock actual (inventario vivo)")
            inv_live = _read_working_inventory()
            if inv_live.empty:
                st.info("Inventario vacío.")
            else:
                st.caption(f"`inventory_updated.csv` · {_mtime_str(OUT10 / 'inventory_updated.csv')}")
                st.dataframe(inv_live, use_container_width=True, height=420)
    
    # ---------------- TAB: 📒 Ledger (NUEVA) ----------------
        with tab_ledger:
            st.subheader("📒 Ledger de movimientos")
            ledger_path = OUT10 / "ledger_movimientos.csv"
            if not ledger_path.exists():
                st.info("Aún no hay ledger. Procesa algún pedido para generarlo.")
            else:
                try:
                    led = pd.read_csv(ledger_path)
                except Exception as e:
                    st.error("No se pudo leer el ledger.")
                    st.exception(e)
                    st.stop()

                if "timestamp" in led.columns:
                    led["timestamp"] = pd.to_datetime(led["timestamp"], errors="coerce")
                for c in ["qty_pedido","on_prev","on_new","delta"]:
                    if c in led.columns:
                        led[c] = pd.to_numeric(led[c], errors="coerce")

                c1, c2, c3 = st.columns([1.2,1,1])
                min_date = pd.to_datetime(led["timestamp"].min()) if "timestamp" in led.columns else None
                max_date = pd.to_datetime(led["timestamp"].max()) if "timestamp" in led.columns else None
                f_ini = c1.date_input("Desde", value=min_date.date() if min_date is not None else None)
                f_fin = c2.date_input("Hasta", value=max_date.date() if max_date is not None else None)
                solo_cambios = c3.toggle("Mostrar solo cambios (Δ ≠ 0)", value=True)

                lf = led.copy()
                if "timestamp" in lf.columns and f_ini and f_fin:
                    lf = lf[(lf["timestamp"].dt.date >= f_ini) & (lf["timestamp"].dt.date <= f_fin)]
                if solo_cambios and "delta" in lf.columns:
                    lf = lf[lf["delta"] != 0]

                total_lines = len(lf)
                total_delta = lf["delta"].sum() if "delta" in lf.columns else 0
                st.caption(f"Líneas en vista: **{total_lines}** · Suma Δ: **{int(total_delta)}**")

                show_cols = [c for c in ["timestamp","Product_ID","Nombre","Proveedor","Categoria","qty_pedido","on_prev","on_new","delta"] if c in lf.columns]
                st.dataframe(lf[show_cols].sort_values("timestamp", ascending=False), use_container_width=True, height=420)

                st.download_button(
                    "⬇️ Descargar ledger filtrado (CSV)",
                    data=lf[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name=f"ledger_filtrado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )   

# ==================================================
# 7) BLOQUE REAPRO (intacto)
# ==================================================
def render_reapro():
    # (… tu bloque actual …)
    pass


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

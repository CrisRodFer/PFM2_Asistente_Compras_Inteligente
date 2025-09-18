# app/streamlit_app.py

import sys
from pathlib import Path
from datetime import datetime
import io

import streamlit as st
import pandas as pd

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

def goto(route: str):
    st.session_state["route"] = route

# ----------------- Barra lateral (navegación global) --------------
with st.sidebar:
    st.markdown("### Navegación")
    choice = st.radio(
        "Bloques",
        list(ROUTES.keys()),
        format_func=lambda k: ROUTES[k],
        index=list(ROUTES.keys()).index(st.session_state["route"]),
        key="route_radio",
    )
    if choice != st.session_state["route"]:
        st.session_state["route"] = choice

    st.markdown("---")
    st.markdown("**Atajos**")
    st.button("🏠 Portada", use_container_width=True, on_click=goto, args=("home",), key="go_home_btn")

# --- Auto-refresh: usa el paquete si está instalado; si no, fallback JS ---
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
except Exception:
    def st_autorefresh(interval: int | None = None, key: str | None = None, **_):
        if not interval or interval <= 0:
            return
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}}, {interval});</script>",
            unsafe_allow_html=True,
        )

# Permitir importar desde scripts/… (raíz del proyecto)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.export.construir_vistas import construir_vistas


# ==================================================
# 1). Utils y rutas de datos
# ==================================================
ROOT  = Path.cwd()
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

def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0

# =================================================================
# 2). Normalizadores y helpers de carga
# =================================================================
TMP_VISTAS = PROC / "_tmp_views"
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

def _normalize_multi_for_vistas(src: Path) -> Path:
    """
    supplier_catalog_multi.csv -> garantiza columnas:
    Product_ID, supplier_id, precio, lead_time, disponibilidad, moq, multiplo
    (nombres exactos que espera construir_vistas)
    """
    if not src.exists():
        return src

    df = _read_csv_smart(src)
    if df is None or df.empty:
        return src

    low = {c.lower(): c for c in df.columns}
    ren = {}

    # IDs de producto
    for cand in ["product_id", "item_id", "id_producto"]:
        if cand in low:
            ren[low[cand]] = "Product_ID"
            break

    # Proveedor
    for cand in ["supplier_id", "id_proveedor", "proveedor"]:
        if cand in low:
            ren[low[cand]] = "supplier_id"
            break

    # Precio -> *precio* (NO 'price')
    if "precio" in low:
        pass
    elif "price" in low:
        ren[low["price"]] = "precio"

    # Lead time
    if "lead_time" in low:
        pass
    elif "lt" in low:
        ren[low["lt"]] = "lead_time"

    # Disponibilidad
    if "disponibilidad" in low:
        pass
    elif "availability" in low:
        ren[low["availability"]] = "disponibilidad"
    elif "stock" in low:
        ren[low["stock"]] = "disponibilidad"

    # MOQ y múltiplos (si existen)
    for cand in ["moq", "min_qty", "minimo_pedido"]:
        if cand in low:
            ren[low[cand]] = "moq"
            break
    for cand in ["multiplo", "multiple", "pack_size"]:
        if cand in low:
            ren[low[cand]] = "multiplo"
            break

    df = df.rename(columns=ren)

    # Exportar a tmp con los nombres normalizados
    out = TMP_VISTAS / "supplier_catalog_multi_norm.csv"
    df.to_csv(out, index=False)
    return out

def _normalize_unif_for_vistas(src: Path) -> Path:
    """
    Normaliza el CSV de sustitutos unificado para construir_vistas:
    columnas clave -> Product_ID, Substitute_Product_ID, tipo (externo/interno).
    Mantiene el resto de columnas (rank, score, etc.) si existen.
    """
    if not src.exists():
        return src

    df = _read_csv_smart(src)
    if df is None or df.empty:
        return src

    low = {c.lower(): c for c in df.columns}
    ren = {}

    # IDs
    for cand in ["product_id", "id_producto", "item_id"]:
        if cand in low:
            ren[low[cand]] = "Product_ID"
            break
    for cand in ["substitute_product_id", "alt_product_id", "sustituto_id", "id_sustituto"]:
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
    """
    Carga vistas de ui_products/ui_substitutes construidas por construir_vistas,
    pasando rutas normalizadas y manejando ausencias con mensajes claros.
    """
    # UNIFICADO
    unif_path = _normalize_unif_for_vistas(UNIF) if UNIF.exists() else UNIF

    # MULTI
    if MULTI.exists():
        multi_path = _normalize_multi_for_vistas(MULTI)
    else:
        candidate = TMP_VISTAS / "supplier_catalog_multi_norm.csv"
        multi_path = candidate if candidate.exists() else None

    if multi_path is None:
        st.error("No encuentro `supplier_catalog_multi.csv` (ni su versión normalizada).")
        st.stop()

    # Catálogo & stock
    cat_path   = CAT   if CAT.exists()   else None
    stock_path = STOCK if STOCK.exists() else None

    return construir_vistas(
        path_unificado=unif_path,
        path_multi=multi_path,
        path_catalogo=cat_path,
        path_stock=stock_path,
        path_consumo=None,
        min_score=min_score,
    )

# -------------------- Cargas específicas (exploración) --------------------
def load_catalog_items(root: Path) -> pd.DataFrame | None:
    posibles = [
        root / "data" / "processed" / "catalog_items_enriquecido.csv",
        root / "data" / "processed" / "catalog_items.csv",
        root / "data" / "clean"     / "catalog_items.csv",
    ]
    pth = next((p for p in posibles if p.exists()), None)
    if not pth:
        return None
    cat = _read_csv_smart(pth)
    if cat is None or cat.empty:
        return None

    pid  = _find_col(cat, ["Product_ID","product_id","id_producto"])
    name = _find_col(cat, ["Nombre","name","nombre"])
    catg = _find_col(cat, ["Categoría","categoria","category","Categoria"])
    prov = _find_col(cat, ["Proveedor","proveedor","supplier","supplier_name"])
    prc  = _find_col(cat, ["precio","price","preferred_price","cat_precio"])
    lt   = _find_col(cat, ["lead_time","Lead_time","preferred_lead_time","lt","cat_lead_time"])
    av   = _find_col(cat, ["disponibilidad","availability","preferred_disponibilidad","stock","on_hand","cat_disponibilidad"])
    if not pid:
        return None

    ren = {pid: "Product_ID"}
    if name: ren[name] = "cat_nombre"
    if catg: ren[catg] = "cat_categoria"
    if prov: ren[prov] = "cat_proveedor"
    if prc:  ren[prc]  = "cat_precio"
    if lt:   ren[lt]   = "cat_lead_time"
    if av:   ren[av]   = "cat_disponibilidad"

    cat = cat.rename(columns=ren)

    keep = ["Product_ID"]
    for c in ["cat_nombre","cat_categoria","cat_proveedor","cat_precio","cat_lead_time","cat_disponibilidad"]:
        if c in cat.columns: keep.append(c)
    cat = cat[keep].drop_duplicates("Product_ID")

    cat["Product_ID"] = _to_pid_str(cat["Product_ID"])
    for c in ["cat_precio","cat_lead_time","cat_disponibilidad"]:
        if c in cat.columns: cat[c] = pd.to_numeric(cat[c], errors="coerce")
    for c in ["cat_nombre","cat_categoria","cat_proveedor"]:
        if c in cat.columns: cat[c] = _to_str_safe(cat[c])

    return cat

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
    if pid: ren[pid] = "Product_ID"
    if sup: ren[sup] = "scm_supplier_id"
    if prc: ren[prc] = "scm_precio"
    if lt:  ren[lt]  = "scm_lead_time"
    if av:  ren[av]  = "scm_disp"
    df = df.rename(columns=ren)

    if "Product_ID" in df.columns:
        df["Product_ID"] = _to_pid_str(df["Product_ID"])
    for c in ["scm_precio","scm_lead_time","scm_disp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ==================================================
# 3). Enriquecidos y helpers de sustitutos
# ==================================================
def enrich_external_subs(dfs: pd.DataFrame, ui_products: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    out = dfs.copy()
    sub_id_col = _find_col(out, ["Substitute_Product_ID","substitute_product_id","alt_product_id"])
    if not sub_id_col:
        return out

    out[sub_id_col] = _to_pid_str(out[sub_id_col])

    # preferred del sustituto desde ui_products (fallback y externos)
    up = ui_products.copy()
    up["Product_ID"] = _to_pid_str(up["Product_ID"])
    cols_map = ["Product_ID"]
    for c in ["name","category","preferred_supplier_id",
              "preferred_price","preferred_lead_time","preferred_disponibilidad"]:
        if c in up.columns: cols_map.append(c)
    up = up[cols_map].drop_duplicates("Product_ID")

    # Catálogo (si hay)
    if catalog is not None:
        cols_cat = ["Product_ID","cat_nombre","cat_categoria","cat_proveedor"]
        for c in ["cat_precio","cat_lead_time","cat_disponibilidad"]:
            if c in catalog.columns: cols_cat.append(c)
        out = out.merge(
            catalog[cols_cat],
            left_on=sub_id_col, right_on="Product_ID", how="left"
        ).drop(columns=["Product_ID"], errors="ignore")

    # Fallback ui_products
    out = out.merge(
        up.rename(columns={
            "name": "up_nombre",
            "category": "up_categoria",
            "preferred_supplier_id": "up_proveedor",
            "preferred_price": "up_precio",
            "preferred_lead_time": "up_lead_time",
            "preferred_disponibilidad": "up_disp"
        }),
        left_on=sub_id_col, right_on="Product_ID", how="left"
    ).drop(columns=["Product_ID"], errors="ignore")

    # -- Helper: siempre devuelve Series (Float64) del tamaño de out --
    def _num_series(colname: str) -> pd.Series:
        s = out.get(colname)
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors="coerce")
        return pd.Series([pd.NA] * len(out), dtype="Float64")

    # Campos visibles texto con fallback cat -> ui_products
    out["nombre"]    = out.get("cat_nombre")    if "cat_nombre"    in out.columns else pd.Series([pd.NA]*len(out))
    out["categoria"] = out.get("cat_categoria") if "cat_categoria" in out.columns else pd.Series([pd.NA]*len(out))
    out["proveedor"] = out.get("cat_proveedor") if "cat_proveedor" in out.columns else pd.Series([pd.NA]*len(out))

    if "up_nombre" in out.columns:    out["nombre"]    = out["nombre"].fillna(out["up_nombre"])
    if "up_categoria" in out.columns: out["categoria"] = out["categoria"].fillna(out["up_categoria"])
    if "up_proveedor" in out.columns: out["proveedor"] = out["proveedor"].fillna(out["up_proveedor"])

    # Precio / LT / Disp: base catálogo + fallback preferred del sustituto
    base_prec = _num_series("cat_precio")
    base_lt   = _num_series("cat_lead_time")
    base_disp = _num_series("cat_disponibilidad")

    up_prec = _num_series("up_precio")
    up_lt   = _num_series("up_lead_time")
    up_disp = _num_series("up_disp")

    out["precio"]         = base_prec.fillna(up_prec)
    out["lead_time"]      = base_lt.fillna(up_lt)
    out["disponibilidad"] = base_disp.fillna(up_disp)

    # Limpiar auxiliares
    out.drop(columns=[c for c in out.columns if c.startswith(("cat_","up_"))], inplace=True, errors="ignore")

    # Tipos finales
    for c in ["precio","lead_time","disponibilidad"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["nombre","categoria","proveedor"]:
        if c in out.columns:
            out[c] = _to_str_safe(out[c])

    return out

def build_internal_subs(pid: str, scm: pd.DataFrame | None, ui_products: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """
    Devuelve sustitutos internos (mismo Product_ID, distinto proveedor) para pid.
    Columns: tipo, Substitute_Product_ID, nombre, categoria, proveedor, precio, lead_time, disponibilidad.
    """
    if scm is None or scm.empty:
        return pd.DataFrame(columns=["tipo","Substitute_Product_ID","nombre","categoria","proveedor","precio","lead_time","disponibilidad"])

    pid = str(pid)
    df = scm[scm["Product_ID"] == pid].copy()
    if df.empty:
        return pd.DataFrame(columns=["tipo","Substitute_Product_ID","nombre","categoria","proveedor","precio","lead_time","disponibilidad"])

    # nombre/categoría del propio producto desde catálogo o ui_products (para mostrar en internos)
    base_name = base_cat = None
    if catalog is not None:
        row = catalog.loc[catalog["Product_ID"] == pid]
        if not row.empty:
            base_name = row["cat_nombre"].iloc[0] if "cat_nombre" in row.columns else None
            base_cat  = row["cat_categoria"].iloc[0] if "cat_categoria" in row.columns else None
    if (base_name is None or base_cat is None) and ui_products is not None:
        row = ui_products.loc[ui_products["Product_ID"] == pid]
        if not row.empty:
            if base_name is None:
                base_name = row["name"].iloc[0] if "name" in row.columns else None
            if base_cat is None:
                base_cat  = row["category"].iloc[0] if "category" in row.columns else None

    out = pd.DataFrame({
        "tipo": "interno",
        "Substitute_Product_ID": pid,
        "nombre": base_name,
        "categoria": base_cat,
        "proveedor": _to_str_safe(df.get("scm_supplier_id", pd.Series(dtype=str))),
        "precio": pd.to_numeric(df.get("scm_precio"), errors="coerce"),
        "lead_time": pd.to_numeric(df.get("scm_lead_time"), errors="coerce"),
        "disponibilidad": pd.to_numeric(df.get("scm_disp"), errors="coerce"),
    })
    # Ordenar opcional: por precio asc / lead time
    out = out.sort_values(["precio","lead_time"], na_position="last", ignore_index=True)
    return out


# ----------------- Portada: tarjetas de acceso -------------------
def render_home():
    st.title("PFM2 · Portada")
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
            cat = cat[["Product_ID", "Nombre", "Categoria"]].drop_duplicates("Product_ID")
        except Exception:
            cat = None

        dfp = views["ui_products"].copy()
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
                if q.isdigit():
                    masks.append(dfp["Product_ID"] == int(q))
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
            col_cfg["Product_ID"] = st.column_config.NumberColumn("Product_ID", help="Identificador único del producto.")
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
        import numpy as np

        st.subheader("Sustitutos por producto")

        # Autorefresh SOLO Sustitutos
        refresh_secs = st.sidebar.slider("⏱ Auto-refresh Sustitutos (seg)", 0, 300, 60, key="refresh_tab2_secs")
        refresh_on   = st.sidebar.toggle("Activar auto-refresh Sustitutos", True, key="refresh_tab2_on")
        if refresh_on and refresh_secs > 0:
            st_autorefresh(interval=int(refresh_secs * 1000), key="auto_refresh_tab2")

        # --- Cargar vistas base ---
        try:
            ui_subs_views = views["ui_substitutes"].copy()
            dfp           = views["ui_products"].copy()
        except Exception as e:
            st.error("No se pudieron cargar 'ui_substitutes' o 'ui_products'.")
            st.exception(e)
            st.stop()

        cat  = load_catalog_items(ROOT)           # catálogo (nombre/categoría/proveedor/price/lt/disp)
        uni  = load_substitutes_unified(ROOT)     # sustitutos externos (si lo tienes)
        scm  = load_supplier_catalog_multi(ROOT)  # multi-catálogo para internos

        ui_subs = uni if uni is not None else ui_subs_views

        # --- Normalización de IDs ---
        pid_col = "Product_ID"
        ui_subs[pid_col] = _to_pid_str(ui_subs[pid_col])
        dfp[pid_col]     = _to_pid_str(dfp[pid_col])
        if cat is not None and "Product_ID" in cat.columns:
            cat["Product_ID"] = _to_pid_str(cat["Product_ID"])
        if scm is not None and "Product_ID" in scm.columns:
            scm["Product_ID"] = _to_pid_str(scm["Product_ID"])

        # callback para NO volver a "Productos" al escribir en el buscador
        def _stay_on_subs():
            st.session_state["explore_subtab"] = "Sustitutos por producto"

        q2 = st.text_input("Buscar por ID, nombre o categoría", key="q_tab2", on_change=_stay_on_subs)

        # ---------- RESUMEN ----------
        tipo_col = _find_col(ui_subs, ["tipo","type"])
        if not tipo_col:
            ui_subs["__dummy_tipo__"] = "externo"
            tipo_col = "__dummy_tipo__"

        ext_counts = (
            ui_subs[ui_subs[tipo_col].eq("externo")]
            .groupby(pid_col, as_index=False)
            .size().rename(columns={"size":"Sustitutos externos"})
        )

        if scm is not None and not scm.empty:
            pref_sup_col = _find_col(dfp, ["preferred_supplier_id"])
            base_int = scm.groupby(pid_col, as_index=False)["scm_supplier_id"].nunique() \
                          .rename(columns={"scm_supplier_id":"n_suppliers"})
            if pref_sup_col:
                df_pref = dfp[[pid_col, pref_sup_col]].drop_duplicates()
                base_int = base_int.merge(df_pref, on=pid_col, how="left")
                base_int["Sustitutos internos"] = (base_int["n_suppliers"] - 1).clip(lower=0)
            else:
                base_int["Sustitutos internos"] = base_int["n_suppliers"]
            int_counts = base_int[[pid_col,"Sustitutos internos"]]
        else:
            int_counts = pd.DataFrame(columns=[pid_col,"Sustitutos internos"])

        resumen = pd.merge(int_counts, ext_counts, on=pid_col, how="outer").fillna(0)
        for c in ["Sustitutos internos","Sustitutos externos"]:
            if c in resumen.columns:
                resumen[c] = resumen[c].astype(int)

        if cat is not None:
            add_cols = [c for c in ["cat_nombre","cat_categoria","cat_proveedor"] if c in cat.columns]
            if add_cols:
                resumen = resumen.merge(cat[[pid_col, *add_cols]], on=pid_col, how="left")

        cols_join = [pid_col] + [c for c in ["preferred_price","preferred_lead_time"] if c in dfp.columns]
        if len(cols_join) > 1:
            resumen = resumen.merge(dfp[cols_join].drop_duplicates(pid_col), on=pid_col, how="left")

        if q2:
            ql = q2.lower()
            resumen = resumen[
                _to_str_safe(resumen[pid_col]).str.contains(ql, na=False) |
                _to_str_safe(resumen.get("cat_nombre","")).str.lower().str.contains(ql, na=False) |
                _to_str_safe(resumen.get("cat_categoria","")).str.lower().str.contains(ql, na=False)
            ]

        if not resumen.empty:
            sort_cols = [c for c in ["Sustitutos externos","Sustitutos internos"] if c in resumen.columns]
            if sort_cols:
                resumen = resumen.sort_values(sort_cols, ascending=False)

        resumen_view = resumen.rename(columns={
            "cat_nombre":"Nombre",
            "cat_categoria":"Categoría",
            "cat_proveedor":"Proveedor principal",
            "preferred_price":"Precio pref.",
            "preferred_lead_time":"Lead time pref."
        })
        cols_resumen = [pid_col,"Nombre","Categoría","Proveedor principal",
                        "Sustitutos internos","Sustitutos externos","Precio pref.","Lead time pref."]
        resumen_view = resumen_view[[c for c in cols_resumen if c in resumen_view.columns]].copy()

        def fmt_eur(x):
            if pd.isna(x): return ""
            try: return f"{float(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
            except: return str(x)
        if "Precio pref." in resumen_view.columns:
            resumen_view["Precio pref."] = pd.to_numeric(resumen_view["Precio pref."], errors="coerce").map(fmt_eur)
        for c in ["Nombre","Categoría","Proveedor principal"]:
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

        dfs_ext = ui_subs[(ui_subs[pid_col] == pid) & (ui_subs[tipo_col].eq("externo"))].copy()
        dfs_ext = enrich_external_subs(dfs_ext, dfp, cat) if not dfs_ext.empty else pd.DataFrame(columns=[
            "tipo","Substitute_Product_ID","nombre","categoria","proveedor",
            "rank","score","precio","lead_time","disponibilidad"
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

        for c in ["precio","lead_time","disponibilidad"]:
            if c in df_show.columns:
                df_show[c] = pd.to_numeric(df_show[c], errors="coerce")

        df_show["Δ precio"]    = df_show["precio"]         - pref_price if "precio" in df_show.columns else None
        df_show["Δ lead time"] = df_show["lead_time"]      - pref_lt    if "lead_time" in df_show.columns else None
        df_show["Δ disp"]      = df_show["disponibilidad"] - pref_disp  if "disponibilidad" in df_show.columns else None

        wanted = ["tipo","Substitute_Product_ID","nombre","categoria","proveedor",
                  "rank","score","precio","lead_time","disponibilidad","Δ precio","Δ lead time","Δ disp"]
        show_cols = [c for c in wanted if c in df_show.columns]
        df_show = df_show[show_cols].rename(columns={
            "tipo":"Tipo",
            "Substitute_Product_ID":"Sustituto (Product_ID)",
            "nombre":"Nombre",
            "categoria":"Categoría",
            "proveedor":"Proveedor",
            "rank":"Rank",
            "score":"Score",
            "precio":"Precio",
            "lead_time":"Lead time",
            "disponibilidad":"Disponibilidad"
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
# 5) BLOQUE DE PROVEEDORES. 
# ==================================================
def render_proveedores():
    st.title("🏭 Proveedores")
    tab1, tab2 = st.tabs(["Catálogo por proveedor", "KPIs de proveedor"])

    with tab1:
        st.subheader("Catálogo por proveedor")
        st.info("Pega aquí tu lógica real: selector de proveedor + catálogo (puedes usar supplier_catalog_multi).")

    with tab2:
        st.subheader("KPIs de proveedor")
        st.info("Muestra lead time medio, OTIF, nº de referencias, % cobertura, etc. (placeholder).")


# ==================================================
# 6) BLOQUE DE MOVIMIENTOS DE STOCK. 
# ==================================================
def _ensure_orders_ui() -> None:
    """Crea el CSV de pedidos UI si no existe."""
    if not ORD_UI.exists():
        ORD_UI.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["date", "item_id", "qty"]).to_csv(ORD_UI, index=False)

def _append_order_row(date_iso: str, item_id: int, qty: int) -> None:
    _ensure_orders_ui()
    df = _read_csv_smart(ORD_UI) or pd.DataFrame(columns=["date", "item_id", "qty"])
    df = pd.concat([df, pd.DataFrame([{"date": date_iso, "item_id": int(item_id), "qty": int(qty)}])], ignore_index=True)
    df.to_csv(ORD_UI, index=False)

def _combine_orders(include_ae: bool) -> Path:
    """Devuelve una ruta temporal con los pedidos a procesar (UI +/- A–E)."""
    _ensure_orders_ui()
    ui = _read_csv_smart(ORD_UI) or pd.DataFrame(columns=["date","item_id","qty"])
    frames = [ui]
    if include_ae and ORD_AE.exists():
        ae = _read_csv_smart(ORD_AE) or pd.DataFrame(columns=["date","item_id","qty"])
        frames.append(ae)
    combo = pd.concat(frames, ignore_index=True)
    tmp = RAW / "customer_orders_to_process.csv"
    combo.to_csv(tmp, index=False)
    return tmp

def _mtime_str(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if p.exists() else "—"

def _safe_df(path: Path) -> pd.DataFrame:
    df = _read_csv_smart(path)
    return df if df is not None else pd.DataFrame()

def render_movimientos_stock():
    st.title("📦 Movimientos de stock")

    # ====== Panel lateral (acciones globales del bloque) ======
    st.sidebar.markdown("### Acciones (procesador)")
    use_ae = st.sidebar.toggle("Incluir escenarios A–E", value=True, help="Suma los pedidos de prueba A–E a los pedidos creados desde la UI.")
    st.sidebar.caption(f"Pedidos UI: `{ORD_UI.relative_to(ROOT)}`")

    # Botón de procesar (llama a tu script de movimientos)
    if st.sidebar.button("▶️ Procesar movimientos", use_container_width=True, type="primary"):
        try:
            # Importamos el run del script (ruta relativa al proyecto)
            sys.path.append(str(ROOT))  # garantía: raíz del proyecto en sys.path
            from scripts.operativa.movimientos_stock import run  # noqa: E402

            orders_to_process = _combine_orders(include_ae=use_ae)

            run(
                inventario=ROOT / "data" / "clean" / "Inventario.csv",
                orders_path=orders_to_process,
                supplier_catalog=SUPPLIER_DEMO,
                substitutes=PROC / "substitutes_unified.csv",
                outdir=OUT10,
            )
            st.success("Procesado OK. Revisa ‘Stock actual’, ‘Ledger’ y ‘OC generadas’.")
        except Exception as e:
            st.error("Error al ejecutar el procesador de movimientos.")
            st.exception(e)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Estado de ficheros (outputs):**")
    for p in [
        OUT10 / "inventory_updated.csv",
        OUT10 / "ledger_movimientos.csv",
        OUT10 / "ordenes_compra.csv",
        OUT10 / "ordenes_compra_lineas.csv",
        OUT10 / "sugerencias_compra.csv",
        OUT10 / "alerts.csv",
        OUT10 / "sugerencias_sustitutos.csv",
    ]:
        st.sidebar.write(f"- `{p.relative_to(ROOT)}` · {_mtime_str(p)}")

    # ====== Tabs principales del bloque ======
    tab_ped, tab_stock, tab_ledger, tab_oc, tab_dem = st.tabs([
        "🧾 Pedidos cliente", "📦 Stock actual", "📜 Ledger", "🛒 OC generadas", "📈 Demanda histórica"
    ])

    # 1) Pedidos de cliente (formulario + carga CSV + tabla UI)
    with tab_ped:
        st.subheader("Pedidos de cliente (UI)")
        with st.form("form_add_order", clear_on_submit=True):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                f_date = st.date_input("Fecha", value=datetime.today(), format="YYYY-MM-DD")
            with c2:
                f_item = st.number_input("Item ID", value=0, min_value=0, step=1)
            with c3:
                f_qty  = st.number_input("Cantidad", value=1, min_value=1, step=1)

            add = st.form_submit_button("➕ Añadir a UI")
            if add:
                if f_item > 0 and f_qty > 0:
                    _append_order_row(f_date.strftime("%Y-%m-%d"), int(f_item), int(f_qty))
                    st.success("Línea añadida a `customer_orders_ui.csv`.")
                else:
                    st.warning("Revisa Item ID y Cantidad.")

        st.markdown("**Cargar CSV de pedidos (append)**")
        up = st.file_uploader("Sube un CSV con columnas `date,item_id,qty`", type=["csv"])
        if up is not None:
            try:
                df_up = pd.read_csv(io.BytesIO(up.getvalue()))
                req = {"date","item_id","qty"}
                if not req.issubset({c.lower() for c in df_up.columns}):
                    st.error("El CSV debe incluir columnas: date, item_id, qty.")
                else:
                    low = {c.lower(): c for c in df_up.columns}
                    df_up = df_up.rename(columns={low["date"]:"date", low["item_id"]:"item_id", low["qty"]:"qty"})
                    _ensure_orders_ui()
                    base = _read_csv_smart(ORD_UI) or pd.DataFrame(columns=["date","item_id","qty"])
                    base = pd.concat([base, df_up[["date","item_id","qty"]]], ignore_index=True)
                    base.to_csv(ORD_UI, index=False)
                    st.success(f"{len(df_up)} líneas añadidas a `{ORD_UI.name}`.")
            except Exception as e:
                st.error("No se pudo leer el CSV subido.")
                st.exception(e)

        _ensure_orders_ui()
        df_ui = _safe_df(ORD_UI)
        st.caption(f"Pedidos en UI ({len(df_ui)} líneas)")
        st.dataframe(df_ui, use_container_width=True, height=320)

        if st.button("🗑️ Vaciar pedidos UI", use_container_width=True):
            pd.DataFrame(columns=["date","item_id","qty"]).to_csv(ORD_UI, index=False)
            st.info("`customer_orders_ui.csv` vaciado.")

    # 2) Stock actual
    with tab_stock:
        st.subheader("Stock actual")
        inv_upd = _safe_df(OUT10 / "inventory_updated.csv")
        if inv_upd.empty:
            st.info("Aún no hay `inventory_updated.csv`. Pulsa **Procesar movimientos** en la barra lateral.")
        else:
            q = st.text_input("Buscar por Item ID o nombre (si existe)", key="search_invupd")
            df_show = inv_upd.copy()
            low = {c.lower(): c for c in df_show.columns}
            if q and "item_id" in low:
                df_show = df_show[df_show[low["item_id"]].astype(str).str.contains(str(q), na=False)]
            st.dataframe(df_show, use_container_width=True, height=420)

    # 3) Ledger
    with tab_ledger:
        st.subheader("Ledger de movimientos")
        led = _safe_df(OUT10 / "ledger_movimientos.csv")
        if led.empty:
            st.info("Aún no hay `ledger_movimientos.csv`.")
        else:
            st.dataframe(led, use_container_width=True, height=420)

    # 4) OC generadas
    with tab_oc:
        st.subheader("Órdenes de compra generadas")
        oc_h = _safe_df(OUT10 / "ordenes_compra.csv")
        oc_l = _safe_df(OUT10 / "ordenes_compra_lineas.csv")

        if oc_h.empty and oc_l.empty:
            st.info("Sin órdenes de compra generadas todavía.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Cabeceras")
                st.dataframe(oc_h, use_container_width=True, height=280)
            with c2:
                st.caption("Líneas")
                st.dataframe(oc_l, use_container_width=True, height=280)

            st.markdown("> **Recepción forzada**: la implementamos en la siguiente subfase (entrada de mercancía).")

    # 5) Demanda histórica (derivada del ledger)
    with tab_dem:
        st.subheader("Demanda histórica (desde ledger)")
        led = _safe_df(OUT10 / "ledger_movimientos.csv")
        if led.empty:
            st.info("Necesitamos el ledger para calcular demanda por día (tipo='venta').")
            st.stop()

        low = {c.lower(): c for c in led.columns}
        ok = all(k in low for k in ["timestamp","tipo","item_id","qty"])
        if not ok:
            st.warning("El ledger no tiene las columnas esperadas (timestamp/tipo/item_id/qty).")
            st.dataframe(led.head(10), use_container_width=True)
            st.stop()

        df = led.rename(columns={low["timestamp"]:"timestamp",
                                 low["tipo"]:"tipo",
                                 low["item_id"]:"item_id",
                                 low["qty"]:"qty"})
        df["fecha"] = pd.to_datetime(df["timestamp"]).dt.date
        demanda = (
            df[df["tipo"].astype(str).str.lower().eq("venta")]
            .assign(demanda=lambda x: (-pd.to_numeric(x["qty"], errors="coerce")).clip(lower=0))
            .groupby(["fecha","item_id"], as_index=False)["demanda"].sum()
        )

        st.caption("Demanda diaria (todas las referencias). Usa filtros para centrarte en una SKU.")
        col1, col2 = st.columns([1,1])
        with col1:
            item_filter = st.text_input("Filtrar por Item ID (opcional)", key="dem_item")
        with col2:
            year_filter = st.selectbox("Año", options=["Todos"] + sorted({d.year for d in map(pd.Timestamp, demanda["fecha"])}), index=0)

        df_show = demanda.copy()
        if item_filter:
            df_show = df_show[df_show["item_id"].astype(str) == str(item_filter)]
        if year_filter != "Todos":
            df_show = df_show[pd.to_datetime(df_show["fecha"]).dt.year == int(year_filter)]

        st.dataframe(df_show.sort_values(["fecha","item_id"]), use_container_width=True, height=420)


# ==================================================
# 7) BLOQUE DE REAPROVIONAMIENTO Y PEDIDOS.
# ==================================================
def render_reapro():
    st.title("🧾 Reapro / Pedidos")
    st.sidebar.markdown("### Parámetros de Reapro")
    z_service = st.sidebar.number_input("Nivel de servicio (z)", value=1.65, step=0.05)
    target_cobertura = st.sidebar.number_input("Cobertura objetivo (días)", value=30, step=1)
    st.sidebar.caption("Ajusta z y cobertura y actualiza la recomendación.")

    tab1, tab2, tab3 = st.tabs(["Recomendación", "Alertas", "Simulador"])
    with tab1:
        st.subheader("Recomendación de pedido")
        st.info("Calcula ROP/SS/Qty recomendado. Integraremos tus DF cuando los tengamos enlazados.")
    with tab2:
        st.subheader("Alertas de ruptura / cobertura baja")
        st.info("Listado de productos con cobertura < umbral o stock ≤ ROP.")
    with tab3:
        st.subheader("Simulador de cobertura")
        st.info("Ajusta LT, demanda o SS y observa impacto en fecha de ruptura y coste.")

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

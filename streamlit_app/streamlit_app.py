
# app/streamlit_app.py

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# --- Auto-refresh: usa el paquete si está instalado; si no, fallback JS ---
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
except Exception:
    def st_autorefresh(interval: int, key: str = None):
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}}, {interval});</script>",
            unsafe_allow_html=True,
        )

# Permitir importar desde scripts/…
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.export.construir_vistas import construir_vistas

# =========================
# 1) Imports & Config
# =========================

st.set_page_config(page_title="Compras Inteligente · Fase 9", layout="wide")
st.title("Exploración y sustitutos")

# --- [NUEVO] Rutas de entrada y utilidades de invalidación ---
ROOT  = Path.cwd()
UNIF  = ROOT / "data" / "processed" / "substitutes_unified.csv"
MULTI = ROOT / "data" / "clean" / "supplier_catalog_multi.csv"
CAT   = ROOT / "data" / "processed" / "catalog_items_enriquecido.csv"
STOCK = ROOT / "data" / "processed" / "stock_positions.csv"

def _mtime(p: Path) -> float:
    """modified time del fichero (0.0 si no existe)."""
    return p.stat().st_mtime if p.exists() else 0.0

# --- [NUEVO] Carga cacheada con TTL + invalidación por mtime ---
@st.cache_data(ttl=60)   # fuerza recarga como máximo cada 60s
def load_views(min_score: float, m_unif: float, m_multi: float, m_cat: float, m_stock: float):
    return construir_vistas(
        path_unificado=UNIF,
        path_multi=MULTI,
        path_catalogo=(CAT if CAT.exists() else None),
        path_stock=(STOCK if STOCK.exists() else None),
        path_consumo=None,
        min_score=min_score,
    )

# =========================
# 2) Parámetros UI (sidebar)
# =========================

min_score = st.sidebar.slider("Umbral score (externos)", 0.0, 1.0, 0.70, 0.01)



# =========================
# 3) Carga / Preparación de datos
# =========================

# --- [NUEVO] Usamos la función cacheada con los mtimes como clave ---
views = load_views(
    min_score,
    _mtime(UNIF), _mtime(MULTI), _mtime(CAT), _mtime(STOCK)
)

# =========================
# 4) Sección: Productos & Sustitutos
# =========================

tab1, tab2, tab3 = st.tabs(["Productos", "Sustitutos por producto", "Reapro (opcional)"])



# ========= Pestaña 1: Productos (resumen) con nombre/categoría + encabezados ES + tooltips =========

# --- [NUEVO] Control del auto-refresh ---

refresh_secs = st.sidebar.slider("⏱️ Auto-refresh (seg)", 10, 300, 60)
st_autorefresh(interval=refresh_secs * 1000, key="auto-refresh")

with tab1:
    import pandas as pd
    import streamlit as st
    from pathlib import Path

    st.subheader("Productos (resumen)")
    root = Path.cwd()

    # --- Cargar catálogo para añadir Nombre y Categoría (fijo) ---
    cat_path = root / "data" / "processed" / "catalog_items.csv"
    try:
        cat = pd.read_csv(cat_path, encoding="utf-8")
        cat = cat[["Product_ID", "Nombre", "Categoria"]].drop_duplicates("Product_ID")
    except Exception as e:
        st.warning(f"No se pudo cargar catálogo en {cat_path}: {e}")
        cat = None

    # --- Vista base ---
    dfp = views["ui_products"].copy()

    # Enriquecer con catálogo (si existe)
    if cat is not None:
        dfp = dfp.merge(cat, on="Product_ID", how="left")

    # Selección/renombrado de columnas (ES)
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

    # --- Columna de alerta (stock bajo) ---
    if "Stock actual" in dfp.columns:
        dfp.insert(
            dfp.columns.get_loc("Stock actual"),
            "Alerta",
            dfp["Stock actual"].apply(lambda x: "⚠️ Bajo" if pd.notnull(x) and x < 20 else "")
        )

    # --- Buscador por ID, nombre o categoría ---

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

    # --- Normalizar numéricas (interno) ---
    num_cols = ["Precio preferente", "Stock actual", "Lead time pref.",
                "Sustitutos internos", "Sustitutos externos"]
    for c in num_cols:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    # --- DataFrame de presentación con formato fijo (TEXTO) ---
    df_show = dfp.copy()
    if "Precio preferente" in df_show.columns:
        df_show["Precio preferente"] = df_show["Precio preferente"].map(
            lambda v: "" if pd.isna(v) else f"{v:,.2f} €"
        )
    if "Stock actual" in df_show.columns:
        df_show["Stock actual"] = df_show["Stock actual"].map(
            lambda v: "" if pd.isna(v) else f"{v:,.0f}"
        )

    # --- Config de columnas con tooltips ---
    col_cfg = {}
    if "Product_ID" in df_show.columns:
        col_cfg["Product_ID"] = st.column_config.NumberColumn(
            "Product_ID", help="Identificador único del producto."
        )
    if "Nombre" in df_show.columns:
        col_cfg["Nombre"] = st.column_config.TextColumn(
            "Nombre", help="Nombre comercial (desde catálogo, si enlaza)."
        )
    if "Categoría" in df_show.columns:
        col_cfg["Categoría"] = st.column_config.TextColumn(
            "Categoría", help="Familia/categoría del catálogo."
        )
    if "Alerta" in df_show.columns:
        col_cfg["Alerta"] = st.column_config.TextColumn(
            "Alerta", help="Muestra '⚠️ Bajo' cuando el stock es inferior a 20 unidades."
        )
    if "Stock actual" in df_show.columns:
        col_cfg["Stock actual"] = st.column_config.TextColumn(
            "Stock actual", help="Unidades en inventario."
        )
    if "Proveedor principal" in df_show.columns:
        col_cfg["Proveedor principal"] = st.column_config.TextColumn(
            "Proveedor principal", help="Proveedor preferente seleccionado por reglas."
        )
    if "Precio preferente" in df_show.columns:
        col_cfg["Precio preferente"] = st.column_config.TextColumn(
            "Precio preferente", help="Precio del proveedor preferente (2 decimales, €)."
        )
    if "Lead time pref." in df_show.columns:
        col_cfg["Lead time pref."] = st.column_config.NumberColumn(
            "Lead time pref.", help="Días de suministro del proveedor principal.", format="%d"
        )
    if "Sustitutos internos" in df_show.columns:
        col_cfg["Sustitutos internos"] = st.column_config.NumberColumn(
            "Sustitutos internos", help="Alternativas internas del mismo producto.", format="%d"
        )
    if "Sustitutos externos" in df_show.columns:
        col_cfg["Sustitutos externos"] = st.column_config.NumberColumn(
            "Sustitutos externos", help="Alternativas externas (otros productos de su categoría).", format="%d"
        )

    preferred_order = [c for c in [
        "Product_ID", "Nombre", "Categoría",
        "Alerta", "Stock actual",
        "Proveedor principal", "Precio preferente", "Lead time pref.",
        "Sustitutos internos", "Sustitutos externos"
    ] if c in df_show.columns]
    df_show = df_show[preferred_order]

    st.dataframe(df_show, use_container_width=True, height=420, column_config=col_cfg)


# ===================== PESTAÑA: Sustitutos por producto ======================
# ============================ HELPERS (idempotentes) ============================
import pandas as pd
from pathlib import Path
import streamlit as st
import numpy as np

if '_find_col' not in globals():
    def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        low = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

if '_to_str_safe' not in globals():
    def _to_str_safe(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip()

if '_to_pid_str' not in globals():
    def _to_pid_str(s: pd.Series) -> pd.Series:
        """
        Normaliza IDs numéricos a string sin decimales: 2141.0 -> '2141'.
        Mantiene NaN como '' para no romper joins.
        """
        s_num = pd.to_numeric(s, errors="coerce")
        out = s_num.astype("Int64").astype(str)
        return out.replace("<NA>", "")

# ===================== CARGAR CATALOGO (debe ir ANTES de tab2) ======================
def load_catalog_items(root: Path):
    posibles = [
        root / "data/processed/catalog_items.csv",
        root / "data/clean/catalog_items.csv",
    ]
    pth = next((p for p in posibles if p.exists()), None)
    if not pth:
        return None

    try:
        cat = pd.read_csv(pth)
    except Exception:
        cat = pd.read_csv(pth, encoding="utf-8", sep=None, engine="python")

    # Detectar columnas y normalizar nombres
    pid  = _find_col(cat, ["Product_ID","product_id","id_producto"])
    name = _find_col(cat, ["Nombre","name","nombre"])
    catg = _find_col(cat, ["Categoría","categoria","category"])
    prov = _find_col(cat, ["Proveedor","proveedor","supplier","supplier_name"])
    prc  = _find_col(cat, ["precio","price","preferred_price"])
    lt   = _find_col(cat, ["lead_time","Lead_time","preferred_lead_time","lt"])
    av   = _find_col(cat, ["disponibilidad","availability","preferred_disponibilidad","stock","on_hand"])
    if not pid:
        return None

    ren = {pid:"Product_ID"}
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

    # Normalizar IDs y tipos
    cat["Product_ID"] = _to_pid_str(cat["Product_ID"])
    for c in ["cat_precio","cat_lead_time","cat_disponibilidad"]:
        if c in cat.columns: cat[c] = pd.to_numeric(cat[c], errors="coerce")
    for c in ["cat_nombre","cat_categoria","cat_proveedor"]:
        if c in cat.columns: cat[c] = _to_str_safe(cat[c])
    return cat

# ===================== ENRIQUECER DETALLE CON CATALOGO (versión fillna) ============
def enrich_subs_detail_with_catalog(
    dfs: pd.DataFrame,
    ui_products: pd.DataFrame,
    catalog: pd.DataFrame | None
):
    out = dfs.copy()

    sub_id_col  = _find_col(out, ["Substitute_Product_ID","substitute_product_id","alt_product_id"])
    tipo_col    = _find_col(out, ["tipo","type"])
    if not sub_id_col:
        return out

    # Normalizar IDs para casar joins
    out[sub_id_col] = _to_pid_str(out[sub_id_col])

    # preferred del sustituto desde ui_products (fallback y externos)
    up = ui_products.copy()
    up["Product_ID"] = _to_pid_str(up["Product_ID"])
    cols_map = ["Product_ID"]
    for c in ["name","category","preferred_supplier_id",
              "preferred_price","preferred_lead_time","preferred_disponibilidad"]:
        if c in up.columns: cols_map.append(c)
    up = up[cols_map].drop_duplicates("Product_ID")

    # Unir catálogo si existe
    if catalog is not None:
        cols_cat = ["Product_ID","cat_nombre","cat_categoria","cat_proveedor"]
        for c in ["cat_precio","cat_lead_time","cat_disponibilidad"]:
            if c in catalog.columns: cols_cat.append(c)
        out = out.merge(
            catalog[cols_cat],
            left_on=sub_id_col, right_on="Product_ID", how="left"
        ).drop(columns=["Product_ID"], errors="ignore")

    # Unir preferred del sustituto
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

    # Fallbacks texto: catálogo -> preferred
    if "cat_nombre" in out or "up_nombre" in out:
        out["nombre"] = out.get("cat_nombre")
        if "up_nombre" in out: out["nombre"] = out["nombre"].fillna(out["up_nombre"])
    if "cat_categoria" in out or "up_categoria" in out:
        out["categoria"] = out.get("cat_categoria")
        if "up_categoria" in out: out["categoria"] = out["categoria"].fillna(out["up_categoria"])
    if "cat_proveedor" in out or "up_proveedor" in out:
        out["proveedor"] = out.get("cat_proveedor")
        if "up_proveedor" in out: out["proveedor"] = out["proveedor"].fillna(out["up_proveedor"])

    # Helper: SIEMPRE Series numéricas del tamaño de out
    def _num_series(colname: str) -> pd.Series:
        if colname in out:
            return pd.to_numeric(out[colname], errors="coerce")
        return pd.Series([pd.NA] * len(out), dtype="Float64")

    # Precio / LT / Disp (base catálogo + fallback preferred; externos priorizan preferred)
    base_prec = _num_series("cat_precio")
    base_lt   = _num_series("cat_lead_time")
    base_disp = _num_series("cat_disponibilidad")
    up_prec   = _num_series("up_precio")
    up_lt     = _num_series("up_lead_time")
    up_disp   = _num_series("up_disp")

    out["precio"]         = base_prec.fillna(up_prec)
    out["lead_time"]      = base_lt.fillna(up_lt)
    out["disponibilidad"] = base_disp.fillna(up_disp)

    if tipo_col and tipo_col in out.columns:
        mask_ext = out[tipo_col].eq("externo")
        out.loc[mask_ext, "precio"]         = up_prec[mask_ext].fillna(out.loc[mask_ext, "precio"])
        out.loc[mask_ext, "lead_time"]      = up_lt[mask_ext].fillna(out.loc[mask_ext, "lead_time"])
        out.loc[mask_ext, "disponibilidad"] = up_disp[mask_ext].fillna(out.loc[mask_ext, "disponibilidad"])

    # Limpiar auxiliares y normalizar tipos finales
    out.drop(columns=[c for c in out.columns if c.startswith(("cat_","up_"))], inplace=True, errors="ignore")
    for c in ["precio","lead_time","disponibilidad"]:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["nombre","categoria","proveedor"]:
        if c in out.columns: out[c] = _to_str_safe(out[c])

    return out

# ===================== PESTAÑA: Sustitutos por producto ======================
with tab2:
    st.subheader("Sustitutos por producto")

    # Cargar vistas base
    try:
        ui_subs = views["ui_substitutes"].copy()
        dfp     = views["ui_products"].copy()
    except Exception as e:
        st.error("No se pudieron cargar las vistas 'ui_substitutes' o 'ui_products'.")
        st.exception(e)
        st.stop()

    # Catálogo opcional
    cat = None
    try:
        root = Path.cwd()
        cat  = load_catalog_items(root)  # None si no existe
        if cat is None:
            st.warning("No se pudo cargar 'catalog_items.csv'. Se continuará sin catálogo.")
    except Exception as e:
        st.warning("No se pudo cargar 'catalog_items.csv'. Se continuará sin catálogo.")
        st.exception(e)
        cat = None

    # Normalización de IDs
    pid_col = "Product_ID"
    ui_subs[pid_col] = _to_pid_str(ui_subs[pid_col])
    dfp[pid_col]     = _to_pid_str(dfp[pid_col])
    if cat is not None and "Product_ID" in cat.columns:
        cat["Product_ID"] = _to_pid_str(cat["Product_ID"])

    # Buscador (lo mantenemos)
    q2 = st.text_input("Buscar por ID, nombre o categoría", key="q_tab2")

    # Columna tipo defensiva
    tipo_col = _find_col(ui_subs, ["tipo","type"])
    if not tipo_col:
        ui_subs["__dummy_tipo__"] = "externo"
        tipo_col = "__dummy_tipo__"

    # Resumen
    try:
        resumen = (
            ui_subs
            .assign(is_int=ui_subs[tipo_col].eq("interno").astype(int),
                    is_ext=ui_subs[tipo_col].eq("externo").astype(int))
            .groupby(pid_col, as_index=False)[["is_int","is_ext"]].sum()
            .rename(columns={"is_int":"Sustitutos internos","is_ext":"Sustitutos externos"})
        )
    except Exception as e:
        st.error("No se pudo construir el resumen de sustitutos.")
        st.exception(e)
        st.stop()

    # Enriquecer resumen con catálogo y preferidos
    if cat is not None:
        try:
            add_cols = [c for c in ["cat_nombre","cat_categoria","cat_proveedor"] if c in cat.columns]
            if add_cols:
                resumen = resumen.merge(cat[[pid_col, *add_cols]], on=pid_col, how="left")
        except Exception as e:
            st.warning("Fallo al unir catálogo en el resumen.")
            st.exception(e)

    cols_join = [pid_col]
    for c in ["preferred_price","preferred_lead_time"]:
        if c in dfp.columns: cols_join.append(c)
    if len(cols_join) > 1:
        try:
            resumen = resumen.merge(dfp[cols_join].drop_duplicates(pid_col), on=pid_col, how="left")
        except Exception as e:
            st.warning("Fallo al unir ui_products (preferidos) en el resumen.")
            st.exception(e)

    # Filtro
    if q2:
        ql = q2.lower()
        resumen = resumen[
            _to_str_safe(resumen[pid_col]).str.contains(ql, na=False) |
            _to_str_safe(resumen.get("cat_nombre","")).str.lower().str.contains(ql, na=False) |
            _to_str_safe(resumen.get("cat_categoria","")).str.lower().str.contains(ql, na=False)
        ]

    # Orden
    if not resumen.empty:
        sort_cols = [c for c in ["Sustitutos externos","Sustitutos internos"] if c in resumen.columns]
        if sort_cols:
            resumen = resumen.sort_values(sort_cols, ascending=False)

    # Vista amigable
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

    # Formato €
    def fmt_eur(x):
        if pd.isna(x): return ""
        try: return f"{float(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
        except: return str(x)
    if "Precio pref." in resumen_view.columns:
        resumen_view["Precio pref."] = pd.to_numeric(resumen_view["Precio pref."], errors="coerce").map(fmt_eur)
    for c in ["Nombre","Categoría","Proveedor principal"]:
        if c in resumen_view.columns:
            resumen_view[c] = _to_str_safe(resumen_view[c])

    # Diagnóstico si está vacío
    if resumen_view.empty:
        st.warning("No hay filas en el resumen. Diagnóstico rápido:")
        st.write({
            "ui_subs.shape": ui_subs.shape,
            "dfp.shape": dfp.shape,
            "cat is None": cat is None,
            "n_unique Product_ID en ui_subs": ui_subs[pid_col].nunique(),
            "n_unique Product_ID en dfp": dfp[pid_col].nunique(),
            "ejemplo ui_subs PID": ui_subs[pid_col].head(5).tolist(),
        })
        st.stop()

    # ✅ Reset index para que la selección sea posicional (0..N-1)
    resumen_view = resumen_view.reset_index(drop=True)

    # Tabla resumen (clica para ver el detalle)
    st.caption("Resumen de cobertura de sustitutos (haz clic en una fila para ver el detalle)")
    st.data_editor(
        resumen_view,
        key="subs_summary_editor",
        use_container_width=True,
        height=360,
        disabled=True,
        hide_index=True,
        column_config={
            "Product_ID": st.column_config.TextColumn("Product_ID", help="ID del producto"),
            "Nombre": st.column_config.TextColumn("Nombre", help="Nombre del producto"),
            "Categoría": st.column_config.TextColumn("Categoría", help="Familia/categoría"),
            "Proveedor principal": st.column_config.TextColumn("Proveedor principal", help="Proveedor del catálogo"),
            "Sustitutos internos": st.column_config.NumberColumn("Sustitutos internos", help="Conteo internos"),
            "Sustitutos externos": st.column_config.NumberColumn("Sustitutos externos", help="Conteo externos"),
            "Precio pref.": st.column_config.TextColumn("Precio pref.", help="Precio del preferente (ui_products)"),
            "Lead time pref.": st.column_config.NumberColumn("Lead time pref.", help="LT del preferente (ui_products)"),
        },
    )

    # --- Capturar clic y sincronizar PID en session_state ---
    sel_rows = st.session_state.get("subs_summary_editor", {}).get("selection", {}).get("rows", [])
    if sel_rows:
        idx = next(iter(sel_rows))  # ya es posicional
        sel_pid = resumen_view.iloc[idx][pid_col]
        if st.session_state.get("sel_pid_tab2") != sel_pid:
            st.session_state["sel_pid_tab2"] = sel_pid
            (getattr(st, "rerun", st.experimental_rerun))()

    # Si aún no hay selección, usar la primera fila visible como valor inicial
    if "sel_pid_tab2" not in st.session_state:
        st.session_state["sel_pid_tab2"] = resumen_view.iloc[0][pid_col]

    # ---------- DETALLE (sin selectbox) ----------
    pid = st.session_state["sel_pid_tab2"]
    st.caption(f"Detalle de sustitutos para Product_ID = {pid}")

    dfs = ui_subs.query("Product_ID == @pid").copy()
    if dfs.empty:
        st.info("Este producto no tiene sustitutos.")
        st.stop()

    # Enriquecer detalle
    try:
        dfs = enrich_subs_detail_with_catalog(dfs, dfp, cat)
    except Exception as e:
        st.error("Fallo al enriquecer el detalle con catálogo/ui_products.")
        st.exception(e)
        st.stop()

    # Métricas vs preferente del original
    row_pref = dfp.loc[dfp[pid_col] == pid]
    if not row_pref.empty:
        row_pref = row_pref.iloc[0]
        pref_price = pd.to_numeric(row_pref.get("preferred_price"), errors="coerce")
        pref_lt    = pd.to_numeric(row_pref.get("preferred_lead_time"), errors="coerce")
        pref_disp  = pd.to_numeric(row_pref.get("preferred_disponibilidad"), errors="coerce")
    else:
        pref_price = pref_lt = pref_disp = None

    for c in ["precio","lead_time","disponibilidad"]:
        if c in dfs.columns: dfs[c] = pd.to_numeric(dfs[c], errors="coerce")

    dfs["Δ precio"]    = dfs["precio"]         - pref_price if "precio" in dfs.columns else None
    dfs["Δ lead time"] = dfs["lead_time"]      - pref_lt    if "lead_time" in dfs.columns else None
    dfs["Δ disp"]      = dfs["disponibilidad"] - pref_disp  if "disponibilidad" in dfs.columns else None

    # Columnas y formato
    show_cols = [c for c in [
        "tipo","Substitute_Product_ID","nombre","categoria","proveedor",
        "rank","score","precio","lead_time","disponibilidad","Δ precio","Δ lead time","Δ disp"
    ] if c in dfs.columns]

    def _fmt_if_exists(df, col, fn):
        if col in df.columns:
            df[col] = df[col].map(fn)

    _fmt_if_exists(dfs, "precio", fmt_eur)
    _fmt_if_exists(dfs, "Δ precio", fmt_eur)

    df_show = dfs[show_cols].rename(columns={
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

    colcfg = {
        "Tipo": st.column_config.TextColumn("Tipo", help="interno/externo"),
        "Sustituto (Product_ID)": st.column_config.TextColumn("Sustituto (Product_ID)", help="ID del sustituto"),
        "Nombre": st.column_config.TextColumn("Nombre", help="Nombre (catálogo/ui_products)"),
        "Categoría": st.column_config.TextColumn("Categoría", help="Familia/categoría"),
        "Proveedor": st.column_config.TextColumn("Proveedor", help="Proveedor"),
        "Rank": st.column_config.NumberColumn("Rank", help="Orden de preferencia (1 = mejor)"),
        "Score": st.column_config.NumberColumn("Score", help="Score para externos"),
        "Precio": st.column_config.TextColumn("Precio", help="Precio estimado del sustituto"),
        "Lead time": st.column_config.NumberColumn("Lead time", help="Lead time estimado"),
        "Disponibilidad": st.column_config.NumberColumn("Disponibilidad", help="Disponibilidad estimada"),
        "Δ precio": st.column_config.TextColumn("Δ precio", help="Precio sustituto – precio preferente del original"),
        "Δ lead time": st.column_config.NumberColumn("Δ lead time", help="LT sustituto – LT preferente del original"),
        "Δ disp": st.column_config.NumberColumn("Δ disp", help="Disp. sustituto – Disp. preferente del original"),
    }

    st.dataframe(
        df_show,
        use_container_width=True,
        column_config=colcfg,
        height=430
    )


# ========= Pestaña 3: Vista de reaprovisionamiento (opcional) =========
with tab3:
    st.subheader("Vista de reaprovisionamiento (opcional)")
    if "engine_replenishment_view" in views:
        dfr = views["engine_replenishment_view"].copy()
        keep = [c for c in ["Product_ID","preferred_supplier_id","preferred_price","on_hand","substitutes_n","preferred_lead_time","preferred_disponibilidad"] if c in dfr.columns]
        st.dataframe(dfr[keep], use_container_width=True, height=420)
    else:
        st.info("Sin datos de reapro todavía.")
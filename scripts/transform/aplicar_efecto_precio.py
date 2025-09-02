# =============================================================================
# Script: aplicar_efecto_precio.py
# Propósito:
#   - Aplicar el impacto de precio (ventanas + elasticidades) sobre la baseline
#     para 2022–2024 y generar un dataset ajustado al calendario real.
#   - Dejar preparado un "calendario" de multiplicadores útil para Streamlit.
#
# Qué hace:
#   1) Lee baseline: data/processed/demanda_subset_final.parquet
#   2) Lee ventanas: data/auxiliar/ventanas_precio.csv
#   3) Lee eventos reales (SHIFT): outputs/tables/validacion_calendario_real_SHIFT_*.csv
#   4) Construye multiplicadores por día y clúster / producto aplicando
#        M = (1 + discount) ** epsilon_cluster,
#      con guardarraíles CAP/FLOOR (+50% CAP si día con evento real) y tratamiento
#      de outliers (no amplificar si M>1 en días outlier).
#   5) Resuelve solapes (elige mayor |lift|, y prioriza product_id sobre clúster/global).
#   6) Genera:
#        - data/processed/demanda_price_adjusted.parquet
#        - (opcional) outputs/tables/price_calendar.parquet
#
# Edición/escenarios:
#   - Ajusta descuentos/fechas/scope en data/auxiliar/ventanas_precio.csv
#   - (Opcional) activa bloque de ESCENARIO (comentado) para aplicar sólo algunas ventanas.
#
# Dependencias:
#   - pandas, numpy, pyarrow (parquet), xlsxwriter u openpyxl (no obligatorio)
# =============================================================================

from __future__ import annotations

from pathlib import Path
import logging
import os
import re
import pandas as pd
import numpy as np

# =========================
# CONFIGURACIÓN Y RUTAS
# =========================
# Soporte notebook / entorno: si no hay __file__, usa PFM2_ROOT o la ruta local conocida
try:
    ROOT_DIR = Path(__file__).resolve().parents[2]   # .../scripts/transform/ -> raíz del repo
except NameError:
    ROOT_DIR = Path(os.getenv("PFM2_ROOT", r"C:\Users\crisr\Desktop\Máster Data Science & IA\PROYECTO\PFM2_Asistente_Compras_Inteligente"))

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
AUXILIAR_DIR = DATA_DIR / ("auxiliar" if (DATA_DIR / "auxiliar").exists() else "aux")
OUTPUTS_DIR = ROOT_DIR / "outputs" / "tables"

BASELINE_PARQUET = PROCESSED_DIR / "demanda_subset_final.parquet"
VENTANAS_CSV     = AUXILIAR_DIR / "ventanas_precio.csv"

# SHIFT (calendario real observado)
SHIFT_PATTERNS = [
    "validacion_calendario_real_SHIFT_*_20*.csv",
    "validacion_calendario_real_SHIFT_localk*_s*_*20*.csv",
]

# Salidas
OUT_PARQUET  = PROCESSED_DIR / "demanda_price_adjusted.parquet"
OUT_CALENDAR = OUTPUTS_DIR / "price_calendar.parquet"  # útil para Streamlit (opcional)

# Elasticidades por clúster (C0..C3)
ELASTICITIES = {
    0: -0.6,  # Estables / fondo (baja sensibilidad)
    1: -1.0,  # Mainstream (media)
    2: -1.2,  # Altamente promocionable / value (alta)
    3: -0.8,  # Premium / nicho (media-baja)
}

# Guardarraíles
CAPS_SIN_EVENTO = {0: 1.8, 1: 2.2, 2: 2.8, 3: 2.0}  # tope multiplicador sin evento por clúster
EVENT_BONUS = 1.5                                   # +50% del CAP si día de evento real
FLOOR_MULT  = 0.5                                   # suelo general

# Tratamiento outliers:
# - si is_outlier==1 y M>1: no amplificar (M_final = min(M, 1))
# - si M<1 (subes precio): sí permitimos reducir.
NO_AMPLIFY_OUTLIERS = True

# Exportar calendario auxiliar para UI
EXPORT_CALENDAR = True

# -------------- ESCENARIO (opcional; desactivado por defecto) ----------------
# Si quieres aplicar sólo algunas ventanas (e.g., para pruebas/Streamlit),
# descomenta el bloque de más abajo y define aquí los IDs a usar.
# APPLY_ONLY_WINDOW_IDS = ["bf_2024", "rebajas_invierno_2024"]
# -----------------------------------------------------------------------------


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem if "__file__" in globals() else "aplicar_efecto_precio")


# =========================
# UTILIDADES
# =========================
def ensure_dirs(*dirs: Path) -> None:
    """Crea (si no existen) los directorios indicados."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def _read_parquet_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Lee parquet de forma segura; devuelve DataFrame vacío si falla."""
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as e:
        log.warning("No pude leer %s: %s", path, e)
        return pd.DataFrame()

def _fix_header_mojibake(s: str) -> str:
    """Normaliza mojibake típico de 'Año' en CSV SHIFT."""
    s = s.replace("AÃ±o","Año").replace("AÂ±o","Año").replace("Año","Año")
    s = s.replace("aÃ±o","Año").replace("aÂ±o","Año")
    s = s.replace("Â","").replace("Ã","")
    return s

def _parse_scope_values(s: str) -> list[int]:
    """Convierte '1|2' o '1,2' en [1,2]; vacío → []."""
    if s is None or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).replace(",", "|").split("|") if str(x).strip()!=""]


# =========================
# CARGA: VENTANAS OBSERVADAS (SHIFT)
# =========================
def load_observed_windows(shift_dir: Path) -> tuple[pd.DataFrame, dict[int, set]]:
    """
    Carga CSV SHIFT (calendario real) y construye un set de fechas por año.

    Returns:
        obs: DataFrame ['Año','Evento','Inicio_obs','Fin_obs']
        event_dates_by_year: dict {año -> set(fecha)}
    """
    files = []
    for pat in SHIFT_PATTERNS:
        files += list(shift_dir.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No encontré CSV SHIFT en {shift_dir}")

    frames = []
    for p in files:
        df = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(p, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            log.warning("No pude leer %s con encodings estándar", p.name)
            continue

        df.columns = [_fix_header_mojibake(c) for c in df.columns]

        ren = {}
        for c in df.columns:
            lc = c.lower()
            if "evento" in lc and "compar" not in lc: ren[c] = "Evento"
            elif "inicio" in lc and ("obs" in lc or "shift" in lc): ren[c] = "Inicio_obs"
            elif "fin"    in lc and ("obs" in lc or "shift" in lc): ren[c] = "Fin_obs"
            elif lc in ("año","ano","year"): ren[c] = "Año"
        df = df.rename(columns=ren)

        # deduplicar y fusionar clave
        df = df.loc[:, ~df.columns.duplicated()].copy()
        for key in ["Año","Evento","Inicio_obs","Fin_obs"]:
            same = [c for c in df.columns if c == key]
            if len(same) > 1:
                merged = df[same].bfill(axis=1).iloc[:, 0]
                df = df.drop(columns=same).assign(**{key: merged})

        if "Año" not in df.columns:
            m = re.search(r"(20\d{2})", p.stem)
            if m: df["Año"] = int(m.group(1))

        need = [c for c in ["Año","Evento","Inicio_obs","Fin_obs"] if c in df.columns]
        if {"Inicio_obs","Fin_obs"}.issubset(need):
            df = df[need].copy()
            df["Inicio_obs"] = pd.to_datetime(df["Inicio_obs"], errors="coerce").dt.date
            df["Fin_obs"]    = pd.to_datetime(df["Fin_obs"],    errors="coerce").dt.date
            df = df.dropna(subset=["Inicio_obs","Fin_obs"])
            frames.append(df)

    if not frames:
        raise RuntimeError("No pude estandarizar ninguna tabla SHIFT.")

    obs = pd.concat(frames, ignore_index=True)
    obs = obs.loc[:, ~df.columns.duplicated()].copy()
    obs = obs.sort_values(["Año","Evento"]).reset_index(drop=True)

    event_dates_by_year: dict[int, set] = {}
    for y, grp in obs.groupby("Año"):
        days = set()
        for _, r in grp.iterrows():
            for d in pd.date_range(r["Inicio_obs"], r["Fin_obs"], freq="D"):
                days.add(d.date())
        event_dates_by_year[int(y)] = days

    log.info("SHIFT cargado. Años: %s · Registros: %s",
             ", ".join(map(str, sorted(event_dates_by_year.keys()))), len(obs))
    return obs, event_dates_by_year


# =========================
# BASELINE: columnas clave
# =========================
def detect_baseline_columns(df: pd.DataFrame) -> dict:
    """
    Detecta columnas clave en baseline y devuelve un mapeo estándar.
    Retorna dict con keys: date, demand, cluster, product_id, is_outlier, price (opcional).
    """
    cols = {c.lower(): c for c in df.columns}
    out = {}

    # Date
    for k in ("date","fecha"):
        if k in cols: out["date"] = cols[k]; break

    # Demand
    for k in ("demand_day","sales_quantity","quantity","qty"):
        if k in cols: out["demand"] = cols[k]; break

    # Cluster
    for k in ("cluster","clúster"):
        if k in cols: out["cluster"] = cols[k]; break

    # product_id
    for k in ("product_id","producto_id","id_producto"):
        if k in cols: out["product_id"] = cols[k]; break

    # is_outlier
    for k in ("is_outlier","outlier","es_outlier","flag_outlier"):
        if k in cols: out["is_outlier"] = cols[k]; break

    # price (opcional)
    for k in ("price","precio","price_mean","mean_price","avg_price","precio_medio"):
        if k in cols: out["price"] = cols[k]; break

    missing = [k for k in ("date","demand","cluster","product_id") if k not in out]
    if missing:
        raise KeyError(f"Faltan columnas clave en baseline: {missing}")

    return out


# =========================
# PRODUCTO → CLÚSTER
# =========================
def product_to_cluster_map(df: pd.DataFrame, col_product: str, col_cluster: str) -> dict[int,int]:
    """Construye un mapeo product_id → cluster usando la moda por producto."""
    tmp = (df[[col_product, col_cluster]]
           .dropna()
           .groupby(col_product)[col_cluster]
           .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]))
    return tmp.to_dict()


# =========================
# CALENDARIO DE MULTIPLICADORES
# =========================
def build_price_multiplier_calendar(windows: pd.DataFrame,
                                    elasticities: dict[int,float],
                                    event_dates_by_year: dict[int,set],
                                    p2c: dict[int,int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye dos calendarios:
      - m_cluster: Date, Cluster, M, source_window_id
      - m_product: Date, product_id, M, source_window_id
    Resuelve CAP/FLOOR por día y marca solapes en el cálculo del CAP.
    """
    rows_c = []
    rows_p = []

    for _, w in windows.iterrows():
        w_id = str(w["id"])
        d0 = pd.to_datetime(w["start"]).date()
        d1 = pd.to_datetime(w["end"]).date()
        discount = float(w["discount"])
        scope_type = str(w["scope_type"]).lower()
        scope_vals = w["scope_values"] if isinstance(w["scope_values"], list) else _parse_scope_values(w["scope_values"])

        # fechas de la ventana
        dates = [d.date() for d in pd.date_range(d0, d1, freq="D")]

        if scope_type == "global":
            targets = sorted(elasticities.keys())
            for c in targets:
                eps = float(elasticities[c])
                base_M = (1.0 + discount) ** eps
                for dt in dates:
                    cap = CAPS_SIN_EVENTO.get(c, 2.0) * (EVENT_BONUS if (dt in event_dates_by_year.get(dt.year, set())) else 1.0)
                    M = max(FLOOR_MULT, min(base_M, cap))
                    rows_c.append((dt, c, float(M), w_id))

        elif scope_type == "cluster":
            targets = [int(c) for c in scope_vals] or sorted(elasticities.keys())
            for c in targets:
                eps = float(elasticities[c])
                base_M = (1.0 + discount) ** eps
                for dt in dates:
                    cap = CAPS_SIN_EVENTO.get(c, 2.0) * (EVENT_BONUS if (dt in event_dates_by_year.get(dt.year, set())) else 1.0)
                    M = max(FLOOR_MULT, min(base_M, cap))
                    rows_c.append((dt, c, float(M), w_id))

        elif scope_type == "product_id":
            targets = [int(pid) for pid in scope_vals if int(pid) in p2c]
            for pid in targets:
                c = int(p2c[pid])
                eps = float(elasticities.get(c, -1.0))
                base_M = (1.0 + discount) ** eps
                for dt in dates:
                    cap = CAPS_SIN_EVENTO.get(c, 2.0) * (EVENT_BONUS if (dt in event_dates_by_year.get(dt.year, set())) else 1.0)
                    M = max(FLOOR_MULT, min(base_M, cap))
                    rows_p.append((dt, pid, float(M), w_id))

        else:
            # desconocido → tratar como global
            targets = sorted(elasticities.keys())
            for c in targets:
                eps = float(elasticities[c])
                base_M = (1.0 + discount) ** eps
                for dt in dates:
                    cap = CAPS_SIN_EVENTO.get(c, 2.0) * (EVENT_BONUS if (dt in event_dates_by_year.get(dt.year, set())) else 1.0)
                    M = max(FLOOR_MULT, min(base_M, cap))
                    rows_c.append((dt, c, float(M), w_id))

    m_cluster = pd.DataFrame(rows_c, columns=["Date","Cluster","M","source_window_id"]) if rows_c else pd.DataFrame(columns=["Date","Cluster","M","source_window_id"])
    m_product = pd.DataFrame(rows_p, columns=["Date","product_id","M","source_window_id"]) if rows_p else pd.DataFrame(columns=["Date","product_id","M","source_window_id"])

    # Resolver solapes dentro de cada grupo: escoger mayor |lift| (|M-1|)
    if not m_cluster.empty:
        m_cluster["lift_abs"] = (m_cluster["M"] - 1.0).abs()
        m_cluster = (m_cluster.sort_values(["Date","Cluster","lift_abs"], ascending=[True,True,False])
                              .drop_duplicates(subset=["Date","Cluster"], keep="first")
                              .drop(columns=["lift_abs"]))
    if not m_product.empty:
        m_product["lift_abs"] = (m_product["M"] - 1.0).abs()
        m_product = (m_product.sort_values(["Date","product_id","lift_abs"], ascending=[True,True,False])
                              .drop_duplicates(subset=["Date","product_id"], keep="first")
                              .drop(columns=["lift_abs"]))

    return m_cluster, m_product


# =========================
# APLICAR A BASELINE (ROBUSTO)
# =========================
def apply_multipliers_to_baseline(base: pd.DataFrame,
                                  cols: dict,
                                  m_cluster: pd.DataFrame,
                                  m_product: pd.DataFrame,
                                  elasticities: dict[int,float],
                                  p2c: dict[int,int] | None = None) -> pd.DataFrame:
    """
    Aplica multiplicadores a la baseline, priorizando product_id sobre clúster.
    Añadido: coerción robusta de tipos e imputación de cluster por product_id.
    Devuelve DF con:
      - demand_multiplier
      - Demand_Day_priceAdj
      - price_factor_effective
      - Price_virtual (si hay precio) o Price_index_virtual (si no)
    """
    b = base.copy()

    # --- Fecha en date ---
    b[cols["date"]] = pd.to_datetime(b[cols["date"]], errors="coerce").dt.date

    # --- product_id a numérico (descartar filas sin product_id) ---
    pid = pd.to_numeric(b[cols["product_id"]], errors="coerce")
    pid_na = pid.isna()
    if pid_na.any():
        log.warning("Filas sin product_id: %s → se descartan para el ajuste de precio.", int(pid_na.sum()))
        b = b.loc[~pid_na].copy()
        pid = pid.loc[b.index]

    # --- cluster a numérico (coerce), imputar desde p2c y rellenar con moda ---
    cluster_raw = pd.to_numeric(b[cols["cluster"]], errors="coerce")

    # Imputar por mapeo product_id -> cluster si está disponible
    if p2c:
        mask_na = cluster_raw.isna()
        if mask_na.any():
            imputed = b.loc[mask_na, cols["product_id"]].map(p2c)
            imputed = pd.to_numeric(imputed, errors="coerce")
            cluster_raw.loc[mask_na] = imputed

    # Si aún quedan NaN, usar la moda de los clusters válidos; si no existe moda, usar 0
    if cluster_raw.isna().any():
        valid = cluster_raw.dropna()
        if len(valid):
            mode_val = valid.mode().iloc[0]
        else:
            mode_val = 0
        n_fill = int(cluster_raw.isna().sum())
        log.warning("Clusters nulos tras imputación: %s → se rellenan con la moda=%s", n_fill, mode_val)
        cluster_raw = cluster_raw.fillna(mode_val)

    # Cast final
    b["__Cluster__"]    = cluster_raw.astype(int)
    b["__product_id__"] = pid.astype(int)

    # --- Merge de multiplicadores: prioridad product_id > cluster/global ---
    if not m_product.empty:
        mp = m_product.rename(columns={"Date":"__Date__","product_id":"__product_id__","M":"M_prod"})
        b["__Date__"] = b[cols["date"]]
        b = b.merge(mp[["__Date__","__product_id__","M_prod"]], how="left", on=["__Date__","__product_id__"])
    else:
        b["M_prod"] = np.nan

    if not m_cluster.empty:
        mc = m_cluster.rename(columns={"Date":"__Date__","Cluster":"__Cluster__","M":"M_clu"})
        if "__Date__" not in b.columns:
            b["__Date__"] = b[cols["date"]]
        b = b.merge(mc[["__Date__","__Cluster__","M_clu"]], how="left", on=["__Date__","__Cluster__"])
    else:
        b["M_clu"] = np.nan

    b["demand_multiplier"] = b["M_prod"].fillna(b["M_clu"]).fillna(1.0).astype(float)

    # --- Outliers: no amplificar si M>1 ---
    if "is_outlier" in cols and cols["is_outlier"] in b.columns and NO_AMPLIFY_OUTLIERS:
        mask_up = (b["demand_multiplier"] > 1.0) & (b[cols["is_outlier"]].astype(int) == 1)
        b.loc[mask_up, "demand_multiplier"] = 1.0

    # --- Demanda ajustada ---
    b["Demand_Day_priceAdj"] = b[cols["demand"]].astype(float) * b["demand_multiplier"]

    # --- Price factor efectivo e (opcional) precio virtual ---
    def _eps_for_row(row):
        c = int(row["__Cluster__"])
        return float(elasticities.get(c, -1.0))

    b["epsilon_row"] = b.apply(_eps_for_row, axis=1)
    b["price_factor_effective"] = b.apply(
        lambda r: (r["demand_multiplier"] ** (1.0 / r["epsilon_row"])) if r["epsilon_row"] != 0 else 1.0,
        axis=1
    )

    price_col = cols.get("price")
    if price_col and price_col in b.columns:
        b["Price_virtual"] = b[price_col].astype(float) * b["price_factor_effective"]
    else:
        b["Price_index_virtual"] = b["price_factor_effective"]

    # Limpieza temporal
    drop_cols = ["__Date__","M_prod","M_clu","epsilon_row"]
    b = b.drop(columns=[c for c in drop_cols if c in b.columns])

    return b


# =========================
# MAIN
# =========================
def main() -> None:
    """Pipeline completo de aplicación del efecto precio a la baseline."""
    try:
        log.info("Inicio aplicar_efecto_precio.py")
        ensure_dirs(PROCESSED_DIR, OUTPUTS_DIR)

        # 1) Baseline (cargamos sólo columnas necesarias + precio si existe)
        preview = _read_parquet_safe(BASELINE_PARQUET, columns=None)
        if preview.empty:
            raise FileNotFoundError(f"No pude leer baseline en {BASELINE_PARQUET}")
        cols_map = detect_baseline_columns(preview)
        need_cols = [cols_map["date"], cols_map["demand"], cols_map["cluster"], cols_map["product_id"]]
        if "is_outlier" in cols_map: need_cols.append(cols_map["is_outlier"])
        if "price" in cols_map: need_cols.append(cols_map["price"])

        base = _read_parquet_safe(BASELINE_PARQUET, columns=list(set(need_cols)))
        if base.empty:
            raise RuntimeError("Baseline cargada vacía con columnas necesarias.")
        log.info("Baseline cargada: filas=%s · columnas=%s", len(base), list(base.columns))

        # 2) Mapeo producto → clúster (para ventanas por product_id)
        p2c = product_to_cluster_map(base, cols_map["product_id"], cols_map["cluster"])

        # 3) Ventanas de precio
        if not VENTANAS_CSV.exists():
            raise FileNotFoundError(f"No existe {VENTANAS_CSV}. Genera primero ventanas_precio.csv.")
        windows = pd.read_csv(VENTANAS_CSV)
        windows["start"] = pd.to_datetime(windows["start"], errors="coerce")
        windows["end"]   = pd.to_datetime(windows["end"], errors="coerce")
        windows["discount"] = windows["discount"].astype(float)

        # --- ESCENARIO (opcional): aplicar sólo ciertas ventanas -----------------
        # if 'APPLY_ONLY_WINDOW_IDS' in globals() and len(APPLY_ONLY_WINDOW_IDS) > 0:
        #     windows = windows[windows["id"].isin(APPLY_ONLY_WINDOW_IDS)].copy()
        #     log.info("ESCENARIO: usando sólo ventanas %s", APPLY_ONLY_WINDOW_IDS)
        # ------------------------------------------------------------------------

        # 4) Calendario real (para CAP con bonus por evento)
        obs, event_dates_by_year = load_observed_windows(OUTPUTS_DIR)

        # 5) Construir calendario de multiplicadores
        windows["scope_values"] = windows["scope_values"].apply(_parse_scope_values)
        m_cluster, m_product = build_price_multiplier_calendar(
            windows=windows,
            elasticities=ELASTICITIES,
            event_dates_by_year=event_dates_by_year,
            p2c=p2c
        )
        log.info("Calendario construido: m_cluster=%s filas · m_product=%s filas",
                 len(m_cluster), len(m_product))

        # 6) Aplicar a baseline (ahora con imputación robusta de clúster)
        adjusted = apply_multipliers_to_baseline(
            base=base,
            cols=cols_map,
            m_cluster=m_cluster,
            m_product=m_product,
            elasticities=ELASTICITIES,
            p2c=p2c  # ← imputar cluster desde product_id si falta
        )

        # 7) Exportar
        adjusted.to_parquet(OUT_PARQUET, index=False)
        log.info("Parquet ajustado escrito en: %s (filas=%s)", OUT_PARQUET, len(adjusted))

        if EXPORT_CALENDAR:
            packs = []
            if not m_cluster.empty:
                packs.append(m_cluster.assign(level="cluster"))
            if not m_product.empty:
                packs.append(m_product.assign(level="product"))
            if packs:
                cal = pd.concat(packs, ignore_index=True)
                cal.to_parquet(OUT_CALENDAR, index=False)
                log.info("Calendario de multiplicadores escrito en: %s (filas=%s)", OUT_CALENDAR, len(cal))

        log.info("Proceso terminado correctamente.")

    except Exception as e:
        log.exception("Fallo en la ejecución: %s", e)
        raise


if __name__ == "__main__":
    main()

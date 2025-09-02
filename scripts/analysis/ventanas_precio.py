# =============================================================================
# Script: ventanas_precio.py
# Descripción:
#   - Genera una plantilla de "ventanas de precio" (desde eventos observados en
#     CSV SHIFT) y ejecuta un pre-chequeo (preflight):
#       * Lifts esperados por clúster (según elasticidades)
#       * Solape con calendario real
#       * Conteo de outliers dentro de la ventana
#       * Guardarraíles (cap / floor)
#
# Flujo (sin argumentos, autoejecutable):
#   1) Cargar ventanas observadas (SHIFT) desde outputs/tables/.
#   2) Generar data/auxiliar/ventanas_precio.csv (si no existe) con limpieza:
#        - Ignora eventos tipo "Festivo Nacional"
#        - Descarta eventos > MAX_EVENT_DAYS
#        - Fusiona duplicados por id y asegura duraciones mínimas por tipo
#   3) (Opcional) Aplicar overrides definidos en este script (si los activas).
#   4) Ejecutar preflight y exportar outputs/tables/preflight_ventanas.xlsx.
#
# Edición de ventanas:
#   - Recomendado: editar data/auxiliar/ventanas_precio.csv y dejar
#     REGENERATE_VENTANAS_CSV = False (así no se pisa).
#
# Overrides (desactivados por defecto):
#   - Puedes activar un bloque de "OVERRIDES" para cambiar descuentos, fechas
#     y/o scope directamente desde el script (para pruebas rápidas).
#   - Para activarlo, descomenta:
#       * las variables APPLY_OVERRIDES / PERSIST_OVERRIDES / OVERRIDES
#       * la llamada a apply_overrides_to_csv(...) en main()
#
# Dependencias:
#   - pandas, numpy, pyarrow (parquet), xlsxwriter (u openpyxl)
#
# Salidas:
#   - data/auxiliar/ventanas_precio.csv
#   - outputs/tables/preflight_ventanas.xlsx
# =============================================================================

from __future__ import annotations

from pathlib import Path
import logging
import re
import pandas as pd
import numpy as np

# ==== 0. CONFIG (RUTAS Y PARÁMETROS) =========================================
# Soporte notebook: si no existe __file__, usa la raíz del proyecto en tu equipo.
try:
    ROOT_DIR = Path(__file__).resolve().parents[2]   # .../scripts/analysis/ -> raíz
except NameError:
    ROOT_DIR = Path(r"C:\Users\crisr\Desktop\Máster Data Science & IA\PROYECTO\PFM2_Asistente_Compras_Inteligente")

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
AUXILIAR_DIR = DATA_DIR / ("auxiliar" if (DATA_DIR / "auxiliar").exists() else "aux")
OUTPUTS_DIR = ROOT_DIR / "outputs" / "tables"

VENTANAS_CSV = AUXILIAR_DIR / "ventanas_precio.csv"
PREFLIGHT_XLSX = OUTPUTS_DIR / "preflight_ventanas.xlsx"

SHIFT_PATTERNS = [
    "validacion_calendario_real_SHIFT_*_20*.csv",
    "validacion_calendario_real_SHIFT_localk*_s*_*20*.csv",
]

# Elasticidades por clúster (ajustadas a C0..C3)
ELASTICITIES = {
    0: -0.6,  # Estables / fondo de armario (baja sensibilidad)
    1: -1.0,  # Mainstream (media)
    2: -1.2,  # Altamente promocionable / value (alta)
    3: -0.8,  # Premium / nicho (media-baja)
}

# Guardarraíles
CAPS_SIN_EVENTO = {0: 1.8, 1: 2.2, 2: 2.8, 3: 2.0}  # tope multiplicador sin evento
EVENT_BONUS = 1.5                                   # +50% si solapa evento real
FLOOR_MULT  = 0.5                                   # suelo general

# Reglas de generación/limpieza
IGNORE_EVENT_PATTERNS = ["festivo nacional"]  # ampliar si quieres
MAX_EVENT_DAYS = 45                            # descarta eventos demasiado largos
MIN_LEN_BY_PREFIX = {                          # duración mínima por tipo
    "bf": 7,            # Black Friday / Cyber
    "rebajas": 25,      # Rebajas (3–4 semanas)
    "navidad": 10,      # Navidad
    "verano": 7,        # Verano (ejemplo)
    "evento": 0,        # genérico
}

# Por defecto NO regenerar el CSV si ya existe (así no pisas tus ediciones)
REGENERATE_VENTANAS_CSV = False

# ==== Overrides opcionales (DESACTIVADOS: deja comentado para no usarlos) ====
# APPLY_OVERRIDES = True          # activa overrides
# PERSIST_OVERRIDES = True        # guarda el CSV con los cambios
# OVERRIDES = {
#     "bf_2024": {  # cambia % y fechas para BF 2024
#         "discount": -0.20,
#         "scope_type": "cluster",
#         "scope_values": "1|2",
#         "start": "2024-11-24",
#         "end":   "2024-12-01",
#     },
#     "rebajas_2025": {  # sólo % (resto igual)
#         "discount": -0.18
#     },
#     # "navidad_2024": {"scope_type": "cluster", "scope_values": "0"}
# }

# ==== 1. LOGGING ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem if "__file__" in globals() else "ventanas_precio")

# ==== 2. UTILIDADES ===========================================================
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

def _excel_writer(path: Path):
    """Devuelve un ExcelWriter con el mejor engine disponible."""
    try:
        return pd.ExcelWriter(path, engine="xlsxwriter", date_format="yyyy-mm-dd", datetime_format="yyyy-mm-dd")
    except Exception:
        return pd.ExcelWriter(path, engine="openpyxl")

# ==== 3. CARGA VENTANAS OBSERVADAS (SHIFT) ===================================
def load_observed_windows(shift_dir: Path) -> tuple[pd.DataFrame, dict[int, set]]:
    """
    Carga los CSV SHIFT del calendario real, arregla cabeceras con mojibake
    (e.g., 'AÂ±o' → 'Año'), deduplica columnas y crea un set de fechas por año.

    Returns:
        obs: DataFrame con columnas ['Año','Evento','Inicio_obs','Fin_obs'].
        event_dates_by_year: dict {año -> set(fecha)}
    """
    def _fix_header(s: str) -> str:
        s = s.replace("AÃ±o","Año").replace("AÂ±o","Año").replace("Año","Año")
        s = s.replace("aÃ±o","Año").replace("aÂ±o","Año")
        s = s.replace("Â","").replace("Ã","")
        return s

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

        df.columns = [_fix_header(c) for c in df.columns]

        ren = {}
        for c in df.columns:
            lc = c.lower()
            if "evento" in lc and "compar" not in lc: ren[c] = "Evento"
            elif "inicio" in lc and (("obs" in lc) or ("shift" in lc)): ren[c] = "Inicio_obs"
            elif "fin"    in lc and (("obs" in lc) or ("shift" in lc)): ren[c] = "Fin_obs"
            elif lc in ("año","ano","year"): ren[c] = "Año"
        df = df.rename(columns=ren)

        # deduplicar y fusionar (coalesce) columnas clave
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
    obs = obs.loc[:, ~obs.columns.duplicated()].copy()
    obs = obs.sort_values(["Año","Evento"]).reset_index(drop=True)

    event_dates_by_year: dict[int, set] = {}
    for y, grp in obs.groupby("Año"):
        days = set()
        for _, r in grp.iterrows():
            for d in pd.date_range(r["Inicio_obs"], r["Fin_obs"], freq="D"):
                days.add(d.date())
        event_dates_by_year[int(y)] = days

    log.info("Ventanas observadas cargadas. Años: %s · Registros: %s",
             ", ".join(map(str, sorted(event_dates_by_year.keys()))), len(obs))
    return obs, event_dates_by_year

# ==== 4. GENERACIÓN DE VENTANAS DE PRECIO ====================================
def classify_event_for_price(evento: str) -> dict:
    """Clasifica el evento y devuelve propuesta de (id_prefix, discount, scope)."""
    s = (evento or "").lower()
    if any(k in s for k in ["black", "bf", "cyber"]):
        return {"id_prefix":"bf", "discount":-0.25, "scope_type":"cluster", "scope_values":"1|2"}
    if ("rebaj" in s) or ("sale" in s):
        return {"id_prefix":"rebajas", "discount":-0.15, "scope_type":"cluster", "scope_values":"0|1|2|3"}
    if ("navid" in s) or ("christ" in s) or ("xmas" in s):
        return {"id_prefix":"navidad", "discount":-0.10, "scope_type":"cluster", "scope_values":"0"}
    if ("verano" in s) or ("summer" in s):
        return {"id_prefix":"verano", "discount":-0.10, "scope_type":"cluster", "scope_values":"0"}
    return {"id_prefix":"evento", "discount":-0.10, "scope_type":"cluster", "scope_values":"0|1|2|3"}

def _should_ignore_event(nombre: str) -> bool:
    """Devuelve True si el evento debe ignorarse (por patrón)."""
    s = (nombre or "").lower()
    return any(pat in s for pat in IGNORE_EVENT_PATTERNS)

def _enforce_min_len(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Fusiona duplicados por 'id' (start=min, end=max) y asegura una duración
    mínima según el prefijo del id (BF, rebajas, etc.).
    """
    g = (df_rows.groupby("id", as_index=False)
                .agg({"start":"min","end":"max","discount":"mean","scope_type":"first","scope_values":"first"}))
    out = []
    for _, r in g.iterrows():
        pref = str(r["id"]).split("_")[0]
        min_len = MIN_LEN_BY_PREFIX.get(pref, 0)
        start = pd.to_datetime(r["start"])
        end   = pd.to_datetime(r["end"])
        days = (end - start).days + 1
        if min_len > 0 and days < min_len:
            pad_total = min_len - days
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            start = start - pd.Timedelta(days=pad_left)
            end   = end   + pd.Timedelta(days=pad_right)
        r["start"], r["end"] = start.date(), end.date()
        out.append(r)
    return pd.DataFrame(out)

def generate_ventanas_precio(obs: pd.DataFrame, out_csv: Path) -> Path:
    """
    Genera el CSV de ventanas de precio a partir de las ventanas observadas,
    aplicando filtros (ignorar patrones, duración máxima) y reglas de mínimos.
    """
    filtered = []
    for _, r in obs.iterrows():
        if _should_ignore_event(str(r.get("Evento",""))):
            continue
        dur = (pd.to_datetime(r["Fin_obs"]) - pd.to_datetime(r["Inicio_obs"])).days + 1
        if dur > MAX_EVENT_DAYS:
            continue
        filtered.append(r)
    obs_use = pd.DataFrame(filtered) if filtered else obs.copy()

    rows = []
    for _, r in obs_use.iterrows():
        meta = classify_event_for_price(str(r.get("Evento","")))
        wid = f'{meta["id_prefix"]}_{r["Año"]}'
        rows.append([wid, str(r["Inicio_obs"]), str(r["Fin_obs"]),
                     meta["discount"], meta["scope_type"], meta["scope_values"]])

    ventanas_precio = pd.DataFrame(rows, columns=["id","start","end","discount","scope_type","scope_values"])
    if not ventanas_precio.empty:
        ventanas_precio = _enforce_min_len(ventanas_precio)

    ensure_dirs(out_csv.parent)
    ventanas_precio.to_csv(out_csv, index=False)
    log.info("CSV de ventanas de precio escrito en: %s", out_csv)
    return out_csv

# ==== 5. PREFLIGHT ============================================================
def parse_scope_values(s: str) -> list[int]:
    """Convierte '1|2' o '1,2' en [1,2]; vacío → []."""
    if s is None or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).replace(",", "|").split("|") if str(x).strip()!=""]

def product_to_cluster_map(baseline_path: Path) -> dict[int,int]:
    """Construye un mapeo product_id → cluster usando el parquet (moda por producto)."""
    base = _read_parquet_safe(baseline_path)
    if base.empty:
        return {}
    ren = {}
    for c in base.columns:
        lc = str(c).lower()
        if lc == "product_id": ren[c] = "product_id"
        if lc == "cluster":    ren[c] = "cluster"
    base = base.rename(columns=ren)
    if not {"product_id","cluster"}.issubset(base.columns):
        return {}
    return (base.dropna(subset=["product_id","cluster"])
                .groupby("product_id")["cluster"]
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
                .to_dict())

def preflight(ventanas_csv: Path,
              event_dates_by_year: dict[int,set],
              outliers_path: Path,
              baseline_path: Path,
              elasticities: dict[int,float]) -> pd.DataFrame:
    """
    Calcula, por ventana × clúster:
      - M_expected y lift_expected_pct
      - solape con calendario real (overlap_event_*),
      - outliers dentro,
      - cap_limit + cap_ok y floor_ok.
    """
    windows = pd.read_csv(ventanas_csv)
    windows["start"] = pd.to_datetime(windows["start"], errors="coerce").dt.date
    windows["end"]   = pd.to_datetime(windows["end"],   errors="coerce").dt.date
    windows["discount"] = windows["discount"].astype(float)
    windows["scope_values"] = windows["scope_values"].apply(parse_scope_values)
    windows["price_factor"] = 1.0 + windows["discount"]

    # Outliers
    outlier_dates = set()
    if outliers_path.exists():
        o = _read_parquet_safe(outliers_path)
        if not o.empty:
            date_col = None
            for c in o.columns:
                if str(c).lower() in ("date","fecha"):
                    date_col = c; break
            if date_col is None: date_col = o.columns[0]
            flag_col = None
            for c in o.columns:
                if str(c).lower() in ("is_outlier","outlier","es_outlier","flag"):
                    flag_col = c; break
            if flag_col is not None:
                o = o[o[flag_col].astype(bool)]
            outlier_dates = set(pd.to_datetime(o[date_col], errors="coerce").dt.date.dropna().tolist())

    # Mapping product->cluster (si scope por producto)
    p2c = product_to_cluster_map(PROCESSED_DIR / "demanda_subset_final.parquet")

    rows = []
    for _, w in windows.iterrows():
        # clústeres afectados
        if w["scope_type"] == "global":
            clusters = sorted(elasticities.keys())
        elif w["scope_type"] == "cluster":
            clusters = [int(c) for c in w["scope_values"]] or sorted(elasticities.keys())
        elif w["scope_type"] == "product_id":
            if p2c:
                clusters = sorted({int(p2c.get(pid, -1)) for pid in w["scope_values"] if p2c.get(pid, -1) != -1})
                if not clusters: clusters = sorted(elasticities.keys())
            else:
                clusters = sorted(elasticities.keys())
        else:
            clusters = sorted(elasticities.keys())

        fechas_win = set(pd.date_range(w["start"], w["end"], freq="D").date)
        years = sorted({d.year for d in fechas_win})
        event_days = set().union(*(event_dates_by_year.get(y, set()) for y in years))
        overlap_days = len(fechas_win & event_days)
        overlap_flag = overlap_days > 0
        out_in_window = len(fechas_win & outlier_dates)

        for c in clusters:
            eps = float(elasticities.get(c, -1.0))
            M   = float((1.0 + float(w["discount"])) ** eps)
            lift = M - 1.0
            cap_limit = CAPS_SIN_EVENTO.get(c, 2.0) * (EVENT_BONUS if overlap_flag else 1.0)
            cap_ok   = "OK" if M <= cap_limit else "REVISAR"
            floor_ok = "OK" if M >= FLOOR_MULT else "REVISAR"
            rows.append({
                "window_id": w["id"],
                "scope_type": w["scope_type"],
                "scope_values": "|".join(map(str, w["scope_values"])) if isinstance(w["scope_values"], list) else w["scope_values"],
                "cluster": c,
                "start": w["start"], "end": w["end"], "days": (w["end"] - w["start"]).days + 1,
                "discount": w["discount"], "price_factor": 1.0 + w["discount"],
                "epsilon": eps,
                "M_expected": round(M, 4),
                "lift_expected_pct": round(lift, 4),
                "overlap_event_days": overlap_days,
                "overlap_event_flag": overlap_flag,
                "outliers_in_window": out_in_window,
                "cap_limit": round(cap_limit, 2),
                "cap_ok": cap_ok,
                "floor_ok": floor_ok,
            })

    return (pd.DataFrame(rows)
              .sort_values(["window_id","cluster"])
              .reset_index(drop=True))

# ==== 6. EXPORTACIÓN ==========================================================
def export_preflight(obs: pd.DataFrame, ventanas_csv: Path, precheck: pd.DataFrame, out_xlsx: Path) -> Path:
    """Exporta el Excel con hojas: observadas, precio (CSV) y preflight."""
    ensure_dirs(out_xlsx.parent)
    with _excel_writer(out_xlsx) as wr:
        obs.to_excel(wr, sheet_name="VENTANAS_OBSERVADAS", index=False)
        pd.read_csv(ventanas_csv).to_excel(wr, sheet_name="VENTANAS_PRECIO", index=False)
        precheck.to_excel(wr, sheet_name="PREFLIGHT", index=False)
    log.info("Preflight exportado en: %s", out_xlsx)
    return out_xlsx

# ==== 7. OVERRIDES (FUNCIONALIDAD OPCIONAL) ===================================
def apply_overrides_to_csv(csv_path: Path,
                           overrides: dict[str, dict],
                           persist: bool = True) -> pd.DataFrame:
    """
    Aplica cambios por 'id' de ventana al CSV (discount, fechas, scope), útil
    para pruebas rápidas sin abrir el CSV. Si 'persist' es True, guarda cambios.
    """
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        return df

    def _norm_scope_vals(v):
        if isinstance(v, str):
            return "|".join([x.strip() for x in v.replace(",", "|").split("|") if x.strip()!=""])
        if isinstance(v, (list, tuple)):
            return "|".join(map(str, v))
        return v

    for win_id, patch in overrides.items():
        mask = (df["id"] == win_id)
        if not mask.any():
            continue
        for k, val in patch.items():
            if k in ("start","end"):
                df.loc[mask, k] = pd.to_datetime(val).date().isoformat()
            elif k == "scope_values":
                df.loc[mask, k] = _norm_scope_vals(val)
            else:
                df.loc[mask, k] = val

    if persist:
        df.to_csv(csv_path, index=False)
    return df

# ==== 8. MAIN ================================================================
def main() -> None:
    """Ejecución del pipeline: carga SHIFT, genera/usa CSV, (opcional overrides), preflight y export."""
    try:
        log.info("Inicio ventanas_precio.py")
        ensure_dirs(AUXILIAR_DIR, OUTPUTS_DIR)

        # 1) Observadas (calendario real)
        obs, event_dates_by_year = load_observed_windows(OUTPUTS_DIR)

        # 2) CSV de ventanas de precio (generar o respetar existente)
        if VENTANAS_CSV.exists() and REGENERATE_VENTANAS_CSV:
            backup = VENTANAS_CSV.with_suffix(".bak.csv")
            VENTANAS_CSV.replace(backup)
            log.info("Backup de CSV existente: %s", backup)
        if (not VENTANAS_CSV.exists()) or REGENERATE_VENTANAS_CSV:
            generate_ventanas_precio(obs, VENTANAS_CSV)
        else:
            log.info("Usando CSV existente: %s", VENTANAS_CSV)

        # 3) (OPCIONAL) Overrides desde el propio script
        # if 'APPLY_OVERRIDES' in globals() and APPLY_OVERRIDES and VENTANAS_CSV.exists():
        #     apply_overrides_to_csv(VENTANAS_CSV, OVERRIDES, persist=('PERSIST_OVERRIDES' in globals() and PERSIST_OVERRIDES))

        # 4) Preflight
        precheck = preflight(
            ventanas_csv=VENTANAS_CSV,
            event_dates_by_year=event_dates_by_year,
            outliers_path=PROCESSED_DIR / "outliers.parquet",
            baseline_path=PROCESSED_DIR / "demanda_subset_final.parquet",
            elasticities=ELASTICITIES,
        )

        # 5) Export
        export_preflight(obs, VENTANAS_CSV, precheck, PREFLIGHT_XLSX)
        log.info("Proceso terminado correctamente.")

    except Exception as e:
        log.exception("Fallo en la ejecución: %s", e)
        raise

if __name__ == "__main__":
    main()

# =============================================================================
# Script: controles_calidad_trazabilidad.py
# Descripción:
# Controles de calidad y trazabilidad para el dataset unificado de sustitutos.
#
# Qué hace:
# 1) Carga el dataset unificado de sustitutos.
# 2) Dedupe: detecta duplicados residuales (y opcionalmente los elimina).
# 3) Simetría (externos): comprueba reciprocidad A↔B y reporta faltantes.
# 4) Sustitutos "raros":
#    - Externos: score < umbral.
#    - Internos: métricas incoherentes (precio<=0, lead_time<0, etc.).
# 5) Cobertura por categoría (si se aporta mapping Product_ID→categoría).
# 6) Snapshot de versión (JSON) con métricas clave para trazabilidad.
#
# Entradas (por defecto):
#   - data/processed/substitutes_unified.csv
#   - (opcional) data/processed/catalog_items_enriquecido.csv  # debe incluir: Product_ID, category (o similar)
#
# Salidas (por defecto):
#   - reports/quality/9_6_snapshot.json
#   - reports/quality/9_6_duplicados.csv
#   - reports/quality/9_6_asimetricos.csv
#   - reports/quality/9_6_raros.csv
#   - reports/quality/9_6_cobertura_categoria.csv   (si hay mapping de categoría)
#   - (opcional) data/processed/substitutes_unified_dedup.csv  (--write_dedup)
#
# Dependencias:
#   - pandas, numpy
#
# Ejemplo de ejecución:
#   python scripts/transform/controles_calidad_trazabilidad.py \
#       --unificado data/processed/substitutes_unified.csv \
#       --catalogo data/processed/catalog_items_enriquecido.csv \
#       --min_score 0.70 --write_dedup
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG RUTAS BASE ====================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]  # scripts/transform -> proyecto
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "quality"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)


# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"No se encontró ninguna de {candidates} en {list(df.columns)}")


# ==== 3. DEDUPE ===============================================================
def detectar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.duplicated()].copy()


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


# ==== 4. SIMETRÍA (EXTERNOS) ==================================================
def calcular_asimetrias_externos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta pares externos A->B que no tienen el reverso B->A.
    Implementación robusta basada en conjuntos de pares.
    Devuelve columnas: Product_ID, Substitute_Product_ID.
    """
    externos = df[df["tipo"] == "externo"].copy()
    if externos.empty:
        return pd.DataFrame(columns=["Product_ID", "Substitute_Product_ID"])

    # Sanea y tipa como string
    a = externos["Product_ID"].astype(str)
    b = externos["Substitute_Product_ID"].astype(str)
    mask_valid = a.notna() & b.notna()
    a = a[mask_valid]
    b = b[mask_valid]

    pares = set(zip(a.tolist(), b.tolist()))
    if not pares:
        return pd.DataFrame(columns=["Product_ID", "Substitute_Product_ID"])

    faltantes = [(x, y) for (x, y) in pares if (y, x) not in pares]
    if not faltantes:
        return pd.DataFrame(columns=["Product_ID", "Substitute_Product_ID"])

    return pd.DataFrame(faltantes, columns=["Product_ID", "Substitute_Product_ID"])


# ==== 5. SUSTITUTOS "RAROS" ===================================================
def detectar_raros(df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    frames = []

    # Externos con score bajo
    ext = df[df["tipo"] == "externo"].copy()
    if "score" in ext.columns:
        score_num = pd.to_numeric(ext["score"], errors="coerce")
        ext_low = ext[score_num < float(min_score)].copy()
        if not ext_low.empty:
            ext_low["motivo"] = f"externo_score<{min_score}"
            frames.append(ext_low)

    # Internos con métricas incoherentes
    inter = df[df["tipo"] == "interno"].copy()
    incoh_mask = pd.Series(False, index=inter.index)
    if "precio" in inter.columns:
        incoh_mask |= pd.to_numeric(inter["precio"], errors="coerce") <= 0
    if "lead_time" in inter.columns:
        incoh_mask |= pd.to_numeric(inter["lead_time"], errors="coerce") < 0
    if "disponibilidad" in inter.columns:
        incoh_mask |= pd.to_numeric(inter["disponibilidad"], errors="coerce") < 0
    if "prioridad" in inter.columns:
        incoh_mask |= inter["prioridad"].isna()

    inter_bad = inter[incoh_mask].copy()
    if not inter_bad.empty:
        inter_bad["motivo"] = "interno_metricas_incoherentes"
        frames.append(inter_bad)

    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame(columns=list(df.columns) + ["motivo"])


# ==== 6. COBERTURA POR CATEGORÍA ==============================================
def cobertura_por_categoria(
    df_unificado: pd.DataFrame,
    df_catalogo: pd.DataFrame,
    categoria_cols: list[str] = ("category", "categoria", "Category"),
) -> pd.DataFrame:
    pid_col = _find_col(df_catalogo, ["Product_ID", "product_id", "id_producto"])
    cat_col = _find_col(df_catalogo, list(categoria_cols))

    productos_con_sust = df_unificado["Product_ID"].astype(str).unique()
    m = df_catalogo[[pid_col, cat_col]].rename(columns={pid_col: "Product_ID", cat_col: "category"})
    m["Product_ID"] = m["Product_ID"].astype(str)

    base = m.groupby("category")["Product_ID"].nunique().rename("productos_totales").reset_index()
    con = m[m["Product_ID"].isin(productos_con_sust)].groupby("category")["Product_ID"].nunique().rename("productos_con_sust").reset_index()

    cov = base.merge(con, on="category", how="left").fillna({"productos_con_sust": 0})
    cov["pct_cobertura"] = (cov["productos_con_sust"] / cov["productos_totales"] * 100).round(1)
    return cov.sort_values("pct_cobertura", ascending=False).reset_index(drop=True)


# ==== 7. SNAPSHOT ==============================================================
def escribir_snapshot(
    path: Path,
    *,
    n_rows: int,
    n_products: int,
    counts_tipo: dict,
    n_dupes: int,
    n_asimetricos: int,
    n_raros: int,
    extra: dict | None = None,
) -> Path:
    snap = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_rows": int(n_rows),
        "n_products": int(n_products),
        "counts_tipo": {k: int(v) for k, v in counts_tipo.items()},
        "duplicados": int(n_dupes),
        "asimetricos_externos": int(n_asimetricos),
        "raros": int(n_raros),
    }
    if extra:
        snap["extra"] = extra
    ensure_dirs(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    return path


# ==== 8. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controles de calidad y trazabilidad del dataset unificado de sustitutos (9.6).")
    p.add_argument("--unificado", type=str, default=str(PROCESSED_DIR / "substitutes_unified.csv"),
                   help="Ruta del CSV unificado de sustitutos.")
    p.add_argument("--catalogo", type=str, default=None,
                   help="(Opcional) Catálogo con Product_ID y categoría para calcular coberturas.")
    p.add_argument("--min_score", type=float, default=0.70,
                   help="Umbral de score mínimo para considerar válidos los sustitutos externos.")
    p.add_argument("--write_dedup", action="store_true",
                   help="Si se activa, exporta un CSV deduplicado de substitutes_unified.")
    p.add_argument("--outdir", type=str, default=str(REPORTS_DIR),
                   help="Directorio donde escribir reportes/snapshot.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    unificado_path = Path(args.unificado)
    outdir = Path(args.outdir)
    ensure_dirs(outdir)

    if not unificado_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo unificado: {unificado_path}")

    log.info("Leyendo unificado: %s", unificado_path)
    df = pd.read_csv(unificado_path)

    # Métricas base
    counts_tipo = df["tipo"].value_counts().to_dict()
    n_rows = len(df)
    n_products = df["Product_ID"].astype(str).nunique()

    # 1) Dedupe
    log.info("Detectando duplicados…")
    dups = detectar_duplicados(df)
    n_dupes = len(dups)
    if n_dupes > 0:
        dups_path = outdir / "9_6_duplicados.csv"
        dups.to_csv(dups_path, index=False)
        log.info("Duplicados exportados: %s", dups_path)
        if args.write_dedup:
            df_dedup = eliminar_duplicados(df)
            out_dedup = PROCESSED_DIR / "substitutes_unified_dedup.csv"
            df_dedup.to_csv(out_dedup, index=False)
            log.info("Unificado deduplicado exportado en: %s", out_dedup)

    # 2) Simetría (externos)
    log.info("Comprobando simetrías (externos)…")
    asim = calcular_asimetrias_externos(df)
    n_asim = len(asim)
    if n_asim > 0:
        asim_path = outdir / "9_6_asimetricos.csv"
        asim.to_csv(asim_path, index=False)
        log.info("Asimetrías externas exportadas: %s", asim_path)

    # 3) Raros
    log.info("Detectando registros 'raros'… (min_score=%.2f)", args.min_score)
    raros = detectar_raros(df, args.min_score)
    n_raros = len(raros)
    if n_raros > 0:
        raros_path = outdir / "9_6_raros.csv"
        raros.to_csv(raros_path, index=False)
        log.info("Raros exportados: %s", raros_path)

    # 4) Cobertura por categoría (opcional)
    extra = {}
    if args.catalogo:
        catalogo_path = Path(args.catalogo)
        if not catalogo_path.exists():
            log.warning("No se encontró el catálogo de categorías: %s", catalogo_path)
        else:
            log.info("Leyendo catálogo para cobertura por categoría: %s", catalogo_path)
            df_cat = pd.read_csv(catalogo_path)
            df_validos = pd.concat(
                [
                    df[df["tipo"] == "interno"],
                    df[(df["tipo"] == "externo") & (pd.to_numeric(df["score"], errors="coerce") >= float(args.min_score))],
                ],
                ignore_index=True,
            )
            cov = cobertura_por_categoria(df_validos, df_cat)
            cov_path = outdir / "9_6_cobertura_categoria.csv"
            cov.to_csv(cov_path, index=False)
            log.info("Cobertura por categoría exportada en: %s", cov_path)
            extra["coverage_categories_rows"] = int(len(cov))

    # 5) Snapshot
    snap_path = outdir / "9_6_snapshot.json"
    escribir_snapshot(
        snap_path,
        n_rows=n_rows,
        n_products=n_products,
        counts_tipo=counts_tipo,
        n_dupes=n_dupes,
        n_asimetricos=n_asim,
        n_raros=n_raros,
        extra=extra,
    )
    log.info("Snapshot escrito en: %s", snap_path)

    log.info("9.6 Controles de calidad y trazabilidad: COMPLETADO")


if __name__ == "__main__":
    main()

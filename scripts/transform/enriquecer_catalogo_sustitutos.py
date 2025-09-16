# =============================================================================
# Script: enriquecer_catalogo_para_sustitutos.py
# Autor : Equipo PFM2
# Fecha : 2025-09-15
#
# Descripción
# ----------
# Enriquecer el catálogo canónico con campos auxiliares para construir el
# mapa de sustitutos (Fase 9.4):
#   - nombre_normalizado : nombre homogéneo (minúsculas, sin tildes, espacios simples).
#   - pack_size          : tamaño del pack extraído del nombre (p. ej. 100, 300, 60...).
#   - uom                : unidad asociada (g, ml, capsulas, sobres, viales, uds).
#
# Parche anti-gel:
#   Si el nombre contiene “gel” o “geles”, NO se extraen pack_size/uom (devuelve None, None).
#
# Entradas por defecto
# --------------------
#   data/processed/catalog_items.parquet   (o .csv equivalente)
#
# Salidas por defecto
# -------------------
#   data/processed/catalog_items_enriquecido.parquet
#   data/processed/catalog_items_enriquecido.csv
# =============================================================================

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# ------------------------- logging -------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("enriquecer_catalogo")


# ------------------------- helpers -------------------------------------------
def _strip_accents(s: str) -> str:
    """Quita acentos/diacríticos."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if unicodedata.category(c) != "Mn"
    )


def normalizar_nombre(s: pd.Series) -> pd.Series:
    """Limpieza suave: minúsculas, sin acentos, espacios normalizados."""
    ss = s.fillna("").astype(str)
    ss = ss.str.lower().map(_strip_accents)
    ss = ss.str.replace(r"\s+", " ", regex=True).str.strip()
    return ss


def _extract_first_number_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extrae (numero, unidad) del texto.
    Unidades soportadas: g, gr, gramos, ml, caps, capsulas, cápsulas, uds, unidades.
    También considera patrones 'x 60 ml' (toma la unidad a la derecha del 'x').
    """
    if not text:
        return None, None

    t = _strip_accents(text.lower())

    # 1) patrón "x 60 ml" o "1 gel x 60 ml"
    m_x = re.search(r"x\s*(\d+(?:[.,]\d+)?)\s*(ml|g|gr|gramos|capsulas?|uds|unidades)\b", t)
    if m_x:
        num = float(m_x.group(1).replace(",", "."))
        unit = m_x.group(2)
        unit = (
            "ml"
            if unit in {"ml"}
            else "g"
            if unit in {"g", "gr", "gramos"}
            else "capsulas"
            if unit.startswith("caps")
            else "uds"
            if unit in {"uds", "unidades"}
            else None
        )
        return num, unit

    # 2) patrón general "60 ml"
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(ml|g|gr|gramos|capsulas?|uds|unidades)\b", t)
    if m:
        num = float(m.group(1).replace(",", "."))
        raw = m.group(2)
        unit = (
            "ml"
            if raw in {"ml"}
            else "g"
            if raw in {"g", "gr", "gramos"}
            else "capsulas"
            if raw.startswith("caps")
            else "uds"
            if raw in {"uds", "unidades"}
            else None
        )
        return num, unit

    return None, None


def extraer_pack_y_uom_desde_nombre(
    df: pd.DataFrame, col_nombre: str = "Nombre"
) -> pd.DataFrame:
    """Crea columnas pack_size y uom a partir de 'Nombre'."""
    if col_nombre not in df.columns:
        raise ValueError(f"Falta columna '{col_nombre}' en el DataFrame.")

    sizes = []
    units = []
    for txt in df[col_nombre].fillna("").astype(str):
        num, unit = _extract_first_number_unit(txt)
        sizes.append(num)
        units.append(unit)

    df["pack_size"] = sizes
    df["uom"] = units
    return df


def aplicar_reglas_geles(df: pd.DataFrame, col_nombre: str = "Nombre") -> pd.DataFrame:
    """
    Parche rápido: si el nombre contiene 'gel' y la unidad resulta 'g', lo marcamos ambiguo.
    (Se refuerza luego con el guardarraíl por categoría).
    """
    mask_gel = (
        df[col_nombre].fillna("").astype(str).str.lower().str.contains(r"\bgel(es)?\b")
    )
    df.loc[mask_gel & (df["uom"] == "g"), ["pack_size", "uom"]] = (np.nan, np.nan)
    return df


def armonizar_uom_por_categoria(df: pd.DataFrame, col_cat: str = "Categoria") -> pd.DataFrame:
    """
    Guardarraíl por categoría:
     - Categorías de VOLÚMEN (ml): geles/isotónicos/hidratación/bebidas.
     - Categorías de PESO (g): barritas/amino/proteínas/creatina/carbohidratos.
    Si la unidad extraída no coincide con lo esperado -> anular pack_size/uom.
    """
    if col_cat not in df.columns:
        return df

    cat_norm = df[col_cat].fillna("").astype(str).str.lower()

    # volumen (ml)
    cat_vol_regex = r"(gel|geles|isoton|hidrataci[oó]n|bebida)"
    # peso (g)
    cat_peso_regex = r"(barrita|barritas|amino|prote[ií]n|creatina|carbohidratos?)"

    m_vol = cat_norm.str.contains(cat_vol_regex, regex=True)
    m_peso = cat_norm.str.contains(cat_peso_regex, regex=True)

    conflict_vol = m_vol & df["uom"].notna() & ~df["uom"].isin(["ml"])
    conflict_peso = m_peso & df["uom"].notna() & ~df["uom"].isin(["g"])

    df.loc[conflict_vol | conflict_peso, ["pack_size", "uom"]] = (np.nan, np.nan)
    return df


# ------------------------- IO -----------------------------------------------
def read_catalog_items(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe catalog_items: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype=str, low_memory=False)

    # tipados básicos
    if "precio_medio" in df.columns:
        df["precio_medio"] = pd.to_numeric(df["precio_medio"], errors="coerce")
    if "Stock Real" in df.columns:
        df["Stock Real"] = pd.to_numeric(df["Stock Real"], errors="coerce")

    need = {"Product_ID", "Nombre", "Categoria"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"catalog_items necesita columnas {sorted(need)}; faltan: {sorted(miss)}")

    return df


def write_outputs(df: pd.DataFrame, out_parquet: Path | None, out_csv: Path | None) -> None:
    if out_parquet:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        log.info("Exportado parquet a: %s", out_parquet)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        log.info("Exportado csv a: %s", out_csv)


# ------------------------- main ---------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enriquece catálogo para sustitución (Fase 9.4).")
    p.add_argument(
        "--in",
        dest="inp",
        default=str(Path("data/processed") / "catalog_items.parquet"),
        help="Ruta de entrada (parquet/csv).",
    )
    p.add_argument(
        "--out-parquet",
        default=str(Path("data/processed") / "catalog_items_enriquecido.parquet"),
        help="Ruta de salida parquet.",
    )
    p.add_argument(
        "--out-csv",
        default=str(Path("data/processed") / "catalog_items_enriquecido.csv"),
        help="Ruta de salida csv.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.inp)
    out_parquet = Path(args.out_parquet) if args.out_parquet else None
    out_csv = Path(args.out_csv) if args.out_csv else None

    log.info("Enriqueciendo catálogo...")

    cat = read_catalog_items(in_path)
    cat["nombre_normalizado"] = normalizar_nombre(cat["Nombre"])
    cat = extraer_pack_y_uom_desde_nombre(cat, col_nombre="Nombre")
    cat = aplicar_reglas_geles(cat, col_nombre="Nombre")
    cat = armonizar_uom_por_categoria(cat, col_cat="Categoria")

    # resumen rápido
    total = len(cat)
    n_pack = cat["pack_size"].notna().sum()
    n_uom = cat["uom"].notna().sum()

    print("== Resumen enriquecimiento ==")
    print(f"- Filas catálogo: {total:,}")
    print(f"- pack_size no nulos: {n_pack:,}")
    print(f"- uom no nulos: {n_uom:,}")

    if n_uom:
        top_uom = cat["uom"].value_counts().head(10)
        print("\nTop uom detectados:")
        print(top_uom)

    write_outputs(cat, out_parquet, out_csv)


if __name__ == "__main__":
    main()
from __future__ import annotations
import re
from typing import Iterable, Literal, Optional
import pandas as pd
from unidecode import unidecode

from .config import (
    DATA_RAW, DATA_CLEAN, PROCESSED, OUTPUTS,
    COL_ID, COL_DATE, COL_QTY
)
from .logging_conf import get_logger

logger = get_logger("pfm2.utils")

# ---------- IO ----------
def read_xlsx(path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    parse_dates = parse_dates or [COL_DATE]
    df = pd.read_excel(path, engine="openpyxl")
    if COL_DATE in parse_dates and COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    return df

def save_parquet(df: pd.DataFrame, path) -> None:
    path = str(path)
    df.to_parquet(path, index=False)
    logger.info(f"Guardado Parquet: {path} ({len(df):,} filas)")

def save_xlsx(df: pd.DataFrame, path) -> None:
    path = str(path)
    df.to_excel(path, index=False, engine="openpyxl")
    logger.info(f"Guardado Excel: {path} ({len(df):,} filas)")

# ---------- Limpieza ----------
def normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return s
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s

def slugify(s: Optional[str]) -> Optional[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return s
    s = normalize_text(s)
    s = unidecode(s).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

def make_slugs(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        sc = f"{c}_slug"
        if c in df.columns:
            df[sc] = df[c].apply(slugify)
    return df

def remove_negatives(df: pd.DataFrame, col: str = COL_QTY,
                     how: Literal["clip0", "drop"] = "clip0") -> pd.DataFrame:
    negs = (df[col] < 0).sum() if col in df.columns else 0
    if negs:
        logger.warning(f"{negs} negativos en {col}")
    if col in df.columns:
        if how == "drop":
            df = df[df[col] >= 0].copy()
        else:
            df[col] = df[col].clip(lower=0)
    return df

def deduplicate(df: pd.DataFrame, keys: Iterable[str],
                method: Literal["sum","max","first"]="sum") -> pd.DataFrame:
    keys = list(keys)
    dups = df.duplicated(subset=keys).sum()
    if not dups:
        return df
    logger.warning(f"Resolviendo {dups} duplicados por {keys} con '{method}'")
    # Agregación numérica para no perder volumen
    agg = {}
    for c in df.columns:
        if c in keys:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and method in ("sum","max"):
            agg[c] = method
        else:
            agg[c] = "first"
    grouped = df.groupby(keys, as_index=False).agg(agg)
    return grouped

# ---------- Fechas / Años ----------
def change_year(df: pd.DataFrame, year: int, date_col: str = COL_DATE) -> pd.DataFrame:
    """Cambia el año; 29/02 de años bisiestos se mapea a 28/02.
       Idempotente si ya está en 'year'."""
    df = df.copy()
    if date_col not in df.columns:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if (df[date_col].dt.year == year).all():
        logger.info(f"Fechas ya en {year}; no se cambia.")
        return df

    df["_was_leap_adjustment"] = False
    def map_ts(ts):
        if pd.isna(ts):
            return ts
        try:
            return ts.replace(year=year)
        except ValueError:
            # 29/02 -> 28/02 para conservar volumen
            return pd.Timestamp(year, 2, 28)

    # Marca los ajustes de 29/02 por si quieres auditarlos
    mask_feb29 = (df[date_col].dt.month == 2) & (df[date_col].dt.day == 29)
    df.loc[mask_feb29, "_was_leap_adjustment"] = True
    df[date_col] = df[date_col].apply(map_ts)
    return df

# ---------- Auditoría ----------
def audit(before: pd.DataFrame, after: pd.DataFrame, name: str) -> dict:
    d = {
        "dataset": name,
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "removed": int(len(before) - len(after)),
        "nulls_after": int(after.isna().sum().sum()),
        "dups_after_id_date": int(after.duplicated(subset=[COL_ID, COL_DATE]).sum())
    }
    logger.info(f"AUDIT {name}: {d}")
    return d

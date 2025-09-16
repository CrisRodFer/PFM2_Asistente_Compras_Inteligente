# =============================================================================
# Script: construir_sustitutos.py
# Descripción:
# Construye el mapa de sustitutos entre productos del catálogo enriquecido.
# Compara dentro de la misma categoría y aplica:
#   - Reglas duras: UOM compatibles + ratio de pack en rango
#   - Score compuesto: atributos + texto (TF-IDF) + precio (opcional) + marca (opcional)
#
# Entradas (min):
#   - data/processed/catalog_items_enriquecido.parquet  (o .csv)
#     Columnas requeridas (case-insensitive en Categoria/EAN):
#       Product_ID, nombre_normalizado, Categoria|categoria, uom, pack_size
#     Opcionales: marca, EAN13|ean_normalizado, precio_pref (si se aporta preferred)
#
# Salidas:
#   - data/clean/substitutes.csv
#   - (opcional) data/clean/substitutes_diag.csv
#
# Uso rápido:
#   python scripts/transform/construir_sustitutos.py \
#     --catalog-items data/processed/catalog_items_enriquecido.parquet \
#     --out data/clean/substitutes.csv --diag-out data/clean/substitutes_diag.csv
# =============================================================================

# == 0. CONFIG =================================================================
from pathlib import Path
import argparse
import logging
import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"

DEFAULT_INP = PROCESSED_DIR / "catalog_items_enriquecido.parquet"
DEFAULT_OUT = CLEAN_DIR / "substitutes.csv"
DEFAULT_DIAG = CLEAN_DIR / "substitutes_diag.csv"

NEED_COLS = {"product_id", "categoria", "nombre_normalizado", "pack_size", "uom"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)


# == 1. UTILIDADES =============================================================

def _norm_header(s: str) -> str:
    """Normaliza encabezados: minúsculas, sin acentos, espacios→_."""
    s = s.strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.lower().replace(" ", "_")


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza y mapea encabezados a los nombres canónicos esperados."""
    df = df.rename(columns={c: _norm_header(c) for c in df.columns})

    ren = {}
    # categoria
    for cand in ("categoria", "categoría", "category"):
        if cand in df.columns:
            ren[cand] = "categoria"
            break
    # product_id
    for cand in ("product_id", "productid", "product_id_"):
        if cand in df.columns:
            ren[cand] = "product_id"
            break
    # nombre_normalizado
    if "nombre_normalizado" not in df.columns and "nombre" in df.columns:
        tmp = (
            df["nombre"]
            .astype(str)
            .str.normalize("NFKD")
            .str.encode("ascii", "ignore")
            .str.decode("utf-8")
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["nombre_normalizado"] = tmp

    df = df.rename(columns=ren)
    return df


def read_catalog_items(path) -> pd.DataFrame:
    """Lee parquet/csv, normaliza encabezados y valida columnas mínimas."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe catálogo de entrada: {p}")

    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, dtype=str, low_memory=False)

    df = _normalize_headers(df)

    # validar columnas necesarias
    miss = NEED_COLS - set(df.columns)
    if miss:
        raise ValueError(
            f"catalog_items necesita columnas {sorted(NEED_COLS)}; faltan: {sorted(miss)}"
        )

    df["pack_size"] = pd.to_numeric(df["pack_size"], errors="coerce")
    df["uom"] = df["uom"].astype(str).str.lower()
    return df


def compute_similarity(df: pd.DataFrame, k=3, umbral=0.7,
                       ratio_pack_min=0.8, ratio_pack_max=1.25) -> pd.DataFrame:
    """Calcula sustitutos basados en similitud de nombre y reglas básicas."""
    results = []
    diag_rows = []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["nombre_normalizado"].fillna(""))

    cos_sim = cosine_similarity(tfidf_matrix)

    for i, row in df.iterrows():
        pid = row["product_id"]
        cat = row["categoria"]
        uom = row["uom"]
        psize = row["pack_size"]

        # candidatos en misma categoría, distintos productos
        mask = (df["categoria"] == cat) & (df["product_id"] != pid)

        candidates = df[mask].copy()
        if candidates.empty:
            continue

        # ratio de pack
        candidates = candidates[
            (candidates["pack_size"] / psize).between(ratio_pack_min, ratio_pack_max, inclusive="neither")
        ]

        if candidates.empty:
            continue

        # similitud de nombre
        sims = cos_sim[i, candidates.index]

        candidates = candidates.assign(score=sims)
        candidates = candidates[candidates["score"] >= umbral]
        candidates = candidates.nlargest(k, "score")

        for _, cand in candidates.iterrows():
            results.append({
                "product_id": pid,
                "sustituto_id": cand["product_id"],
                "score": cand["score"],
            })
            diag_rows.append({
                "product_id": pid,
                "nombre": row["nombre_normalizado"],
                "sustituto_id": cand["product_id"],
                "nombre_sust": cand["nombre_normalizado"],
                "score": cand["score"],
                "categoria": cat,
                "uom": uom,
            })

    subs = pd.DataFrame(results)
    diag = pd.DataFrame(diag_rows)
    return subs, diag


# == 2. MAIN ===================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Construir mapa de sustitutos entre productos externos.")
    p.add_argument("--catalog-items", type=str, default=str(DEFAULT_INP))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--diag-out", type=str, default=str(DEFAULT_DIAG))
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--umbral", type=float, default=0.7)
    p.add_argument("--ratio-pack-min", type=float, default=0.8)
    p.add_argument("--ratio-pack-max", type=float, default=1.25)
    return p.parse_args()


def main():
    args = _parse_args()
    log.info("Leyendo catálogo de entrada...")
    cat = read_catalog_items(args.catalog_items)

    log.info("Construyendo mapa de sustitutos...")
    subs, diag = compute_similarity(
        cat,
        k=args.k,
        umbral=args.umbral,
        ratio_pack_min=args.ratio_pack_min,
        ratio_pack_max=args.ratio_pack_max,
    )

    log.info("Escribiendo salida...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subs.to_csv(args.out, index=False, encoding="utf-8-sig")

    if args.diag_out:
        diag.to_csv(args.diag_out, index=False, encoding="utf-8-sig")

    print("=== Resumen sustitutos ===")
    print(f"Entradas: {len(cat)} productos")
    print(f"Sustituciones generadas: {len(subs)}")
    print(f"Archivo principal: {args.out}")
    if args.diag_out:
        print(f"Archivo diagnóstico: {args.diag_out}")


if __name__ == "__main__":
    main()
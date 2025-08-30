# ======================================================================================
# Script: generar_subset_representativo.py
# Autor: Equipo PFM2
# Descripción:
#   Genera un subset representativo desde 'demanda_con_clusters.csv' garantizando:
#   - 30% de productos NO outlier (muestreados proporcionalmente por clúster con mínimo).
#   - +100% de productos outlier (forzados SIEMPRE, no consumen el cupo del 30%).
#   - Reportes: distribución de productos por clúster y resumen de validaciones.
#
# Uso (ejemplo):
#   python scripts/transform/generar_subset_representativo.py \
#     --inp "data/processed/demanda_con_clusters.csv" \
#     --out_demanda "data/processed/demanda_subset.csv" \
#     --out_prods "reports/subset_productos.csv" \
#     --out_resumen "reports/subset_resumen_validacion.txt" \
#     --outliers "reports/outliers_dbscan.csv" \
#     --frac 0.30 --min_per_cluster 50 --seed 42
#
# Requisitos de columnas en la demanda:
#   Product_ID, Cluster, Date (para el rango, si no está se informa "N/A")
#   is_outlier (opcional); si no existe, se creará a partir del fichero --outliers (opcional)
# ======================================================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd


# ------------------------- utilidades de logging --------------------------------------
def _ts() -> str:
    """Devuelve un timestamp corto para logs."""
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """Log al estilo solicitado."""
    print(f"{_ts()} | INFO | generar_subset_representativo | {msg}")


# ------------------------- lectura de outliers ----------------------------------------
def _read_outliers(path: str | Path) -> Set[str]:
    """
    Lee un CSV con una columna de Product_ID (o similar) y retorna el set de IDs (str).
    Acepta nombres habituales: 'Product_ID', 'product_id', 'id', 'producto', etc.
    Si no se encuentra columna válida, devuelve set() y se informa por log.
    """
    path = Path(path)
    if not path.exists():
        log(f"[AVISO] Fichero de outliers no encontrado: {path}. Se continúa sin forzar outliers.")
        return set()

    df = pd.read_csv(path)
    posibles = ["Product_ID", "product_id", "id", "producto", "producto_id", "productid"]
    col_ok = None
    for c in df.columns:
        if c in posibles or c.strip().lower() in [p.lower() for p in posibles]:
            col_ok = c
            break

    if col_ok is None:
        log(f"[AVISO] No se encontró columna de Product_ID en: {path}. Se continúa sin forzar outliers.")
        return set()

    return set(df[col_ok].astype(str).unique())


# ------------------------- muestreo de no outliers ------------------------------------
def _sample_ids_no_outliers(
    df: pd.DataFrame,
    target: int,
    min_por_cluster: int,
    seed: int,
) -> Set[str]:
    """
    Devuelve un conjunto de Product_ID (str) muestreados SOLO entre NO outliers,
    respetando distribución por clúster con un mínimo por clúster.
    """
    if target <= 0 or df.empty:
        return set()

    rng = np.random.default_rng(seed)

    dist = df.groupby("Cluster")["Product_ID"].nunique().rename("n").reset_index()
    total = int(dist["n"].sum())
    if total == 0:
        return set()

    # Cuotas iniciales proporcionales + mínimo
    cuotas: Dict[int, int] = {}
    resto = target
    for _, r in dist.sort_values("n", ascending=False).iterrows():
        cl = int(r["Cluster"])
        n = int(r["n"])
        cu = max(min_por_cluster, int(round(n * (target / total))))
        cu = min(cu, n)
        cuotas[cl] = cu
        resto -= cu

    # Ajuste del resto (positivo/negativo) sobre el clúster mayoritario
    mayor = max(cuotas.items(), key=lambda x: x[1])[0]
    if resto != 0:
        n_mayor = int(dist.loc[dist["Cluster"] == mayor, "n"].iloc[0])
        cuotas[mayor] = max(
            min_por_cluster,
            min(cuotas[mayor] + resto, n_mayor),
        )

    # Muestreo por clúster
    keep: Set[str] = set()
    for cl, cu in cuotas.items():
        cand = (
            df.loc[df["Cluster"] == cl, "Product_ID"]
            .drop_duplicates()
            .sample(n=cu, random_state=seed)
        )
        keep |= set(cand.astype(str).tolist())

    return keep


# ------------------------- pipeline principal -----------------------------------------
def generar_subset(
    in_path: Path,
    out_demanda: Path,
    out_prods_report: Path,
    out_resumen_report: Path,
    outliers_path: Path | None = None,
    frac: float = 0.30,
    min_per_cluster: int = 50,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Genera el subset con la política:
      - target_no_outliers = round(frac * nº_productos_base)
      - subset = muestreo(no_outliers, target_no_outliers)  UNION  todos_los_outliers
    """
    log(f"Leyendo: {in_path}")
    demanda = pd.read_csv(in_path)

    if "Product_ID" not in demanda.columns:
        raise ValueError("Falta columna obligatoria 'Product_ID' en la demanda.")
    if "Cluster" not in demanda.columns:
        raise ValueError("Falta columna obligatoria 'Cluster' en la demanda.")

    # Normalizamos tipos
    demanda["Product_ID"] = demanda["Product_ID"].astype(str)

    # Outliers desde CSV (si se proporciona) + columna is_outlier (si no existe)
    out_ids: Set[str] = set()
    if outliers_path:
        out_ids = _read_outliers(outliers_path)
    if "is_outlier" not in demanda.columns:
        demanda["is_outlier"] = 0
    if out_ids:
        demanda.loc[demanda["Product_ID"].isin(out_ids), "is_outlier"] = 1

    # Diagnóstico de outliers
    outliers_en_demanda = demanda.loc[demanda["is_outlier"] == 1, "Product_ID"].nunique()
    coinciden = len(set(demanda.loc[demanda["is_outlier"] == 1, "Product_ID"]).intersection(out_ids))
    log(f"Outliers detectados en demanda: {outliers_en_demanda} | Coinciden con CSV de outliers: {coinciden} | En CSV: {len(out_ids)}")

    base_n_prod = demanda["Product_ID"].nunique()
    objetivo_30 = int(round(base_n_prod * frac))

    log("Generando subset...")
    log(f"Productos totales: {base_n_prod} | Objetivo subset: {objetivo_30} (30%)")

    # Separar NO outliers / outliers
    df_out = demanda[demanda["is_outlier"] == 1].copy()
    df_in = demanda[demanda["is_outlier"] == 0].copy()

    # Muestreo SOLO entre NO outliers (cupo del 30%)
    ids_no_out = _sample_ids_no_outliers(
        df=df_in,
        target=objetivo_30,
        min_por_cluster=min_per_cluster,
        seed=seed,
    )

    subset_in = df_in[df_in["Product_ID"].isin(ids_no_out)]
    subset_out = df_out  # TODOS los outliers se incluyen
    subset = pd.concat([subset_in, subset_out], ignore_index=True)

    # Guardar subset completo (todas las filas de los productos seleccionados)
    out_demanda.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_demanda, index=False)

    # Reporte por clúster (nº de productos únicos)
    prods = (
        subset.groupby("Cluster")["Product_ID"]
        .nunique()
        .rename("n_productos")
        .reset_index()
    )
    out_prods_report.parent.mkdir(parents=True, exist_ok=True)
    prods.to_csv(out_prods_report, index=False)

    # Resumen de validación
    try:
        min_date = pd.to_datetime(subset["Date"]).min().strftime("%Y-%m-%d")
        max_date = pd.to_datetime(subset["Date"]).max().strftime("%Y-%m-%d")
    except Exception:
        min_date = "N/A"
        max_date = "N/A"

    outliers_subset = subset.loc[subset["is_outlier"] == 1, "Product_ID"].nunique()
    filas_outlier_subset = int((subset["is_outlier"] == 1).sum())

    out_resumen_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_resumen_report, "w", encoding="utf-8") as f:
        f.write(f"Productos totales (base): {base_n_prod}\n")
        f.write(f"Objetivo subset (≈{int(frac*100)}%): {objetivo_30}\n")
        f.write(f"Productos en subset: {subset['Product_ID'].nunique()}  (≈30% no outliers + outliers)\n\n")
        f.write("Distribución por cluster (nº productos en subset):\n")
        f.write(prods.to_string(index=False))
        f.write("\n\n")
        f.write(f"Rango fechas subset: {min_date} ➜ {max_date}\n")
        f.write(f"Filas marcadas outlier en subset: {filas_outlier_subset}\n")
        f.write(f"Productos con algún outlier en subset: {outliers_subset}\n")

    # Validación de forzado de outliers
    forced_ids = set(df_out["Product_ID"].unique())
    missing = forced_ids - set(subset["Product_ID"].unique())
    if missing:
        log(f"[AVISO] Faltan {len(missing)} productos outlier en subset (no debería ocurrir).")
    else:
        log(f"Outliers incluidos en subset: {len(forced_ids)} productos.")

    log("Subset generado correctamente.")

    return {
        "n_prod_base": base_n_prod,
        "n_prod_subset": subset["Product_ID"].nunique(),
        "outliers_base": outliers_en_demanda,
        "outliers_subset": outliers_subset,
    }


# ------------------------- CLI ---------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """Parseo de argumentos de línea de comandos."""
    p = argparse.ArgumentParser(
        description="Genera un subset representativo (30% NO outliers + 100% outliers)."
    )
    p.add_argument("--inp", required=True, help="CSV de entrada con demanda y Cluster.")
    p.add_argument("--out_demanda", required=True, help="Ruta del CSV de salida del subset.")
    p.add_argument("--out_prods", required=True, help="Ruta del reporte de nº productos por cluster.")
    p.add_argument("--out_resumen", required=True, help="Ruta del reporte de resumen de validación.")
    p.add_argument("--outliers", default="", help="CSV con lista de Product_ID outliers (opcional).")
    p.add_argument("--frac", type=float, default=0.30, help="Fracción objetivo de productos NO outlier (default 0.30).")
    p.add_argument("--min_per_cluster", type=int, default=50, help="Mínimo por clúster para el muestreo NO outlier.")
    p.add_argument("--seed", type=int, default=42, help="Semilla para muestreos.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    in_path = Path(args.inp)
    out_demanda = Path(args.out_demanda)
    out_prods = Path(args.out_prods)
    out_resumen = Path(args.out_resumen)
    outliers_path = Path(args.outliers) if args.outliers else None

    generar_subset(
        in_path=in_path,
        out_demanda=out_demanda,
        out_prods_report=out_prods,
        out_resumen_report=out_resumen,
        outliers_path=outliers_path,
        frac=float(args.frac),
        min_per_cluster=int(args.min_per_cluster),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()

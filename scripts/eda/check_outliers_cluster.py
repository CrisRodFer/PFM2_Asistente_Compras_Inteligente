# =============================================================================
# Script: check_outliers_clusters.py
# Descripción:
# Comprobaciones específicas de outliers y clúster:
# - Distribución de __cluster__
# - Igualdad cluster vs __cluster__ en NO-outliers
# - Asignación de clúster en productos outlier
# No genera archivos salvo que se pase --report.
#
# Input:
#   - data/processed/subset_modelado.parquet (por defecto)
#
# Output (opcional):
#   - reports/outliers/summary_outliers_clusters.txt
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "outliers"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

def build_report(df: pd.DataFrame) -> str:
    lines: list[str] = []

    assert "__cluster__" in df.columns or "cluster" in df.columns, "Falta columna de clúster"
    cluster_col = "__cluster__" if "__cluster__" in df.columns else "cluster"

    # 1) Distribución de __cluster__
    lines.append("=== Distribución de clusters ===")
    vc = df[cluster_col].value_counts(dropna=False).sort_index()
    lines.append(str(vc.head(20)))
    lines.append(f"Valores únicos: {df[cluster_col].nunique()} | Min: {df[cluster_col].min()} | Max: {df[cluster_col].max()}")

    # 2) Igualdad en no-outliers
    if {"cluster","__cluster__","is_outlier"}.issubset(df.columns):
        no_out = df["is_outlier"].eq(0)
        iguales = (df.loc[no_out,"cluster"].fillna(-1).astype(int) ==
                   df.loc[no_out,"__cluster__"].astype(int)).all()
        lines.append("\n=== Coherencia NO-outliers ===")
        lines.append(f"Cluster y __cluster__ idénticos en NO-outliers: {bool(iguales)}")

    # 3) Outliers y su asignación
    if "is_outlier" in df.columns:
        outliers = df.loc[df["is_outlier"]==1, ["product_id", cluster_col]].drop_duplicates()
        lines.append("\n=== Outliers ===")
        lines.append(f"Productos outlier totales: {outliers['product_id'].nunique()}")
        lines.append(f"Clusters distintos asignados a outliers: {outliers[cluster_col].nunique()}")
        # muestra
        lines.append(str(outliers.head(20)))

    return "\n".join(lines)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chequeos específicos de outliers vs clusters (no escribe por defecto).")
    p.add_argument("--in", dest="inp", type=str, default=str(PROCESSED_DIR / "subset_modelado.parquet"),
                   help="Ruta de entrada (PARQUET).")
    p.add_argument("--report", dest="report", type=str, default="",
                   help="Ruta de TXT para volcar el resumen (opcional).")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    inp = Path(args.inp)
    rep = Path(args.report) if args.report else None

    log.info("Leyendo: %s", inp)
    df = pd.read_parquet(inp)

    report = build_report(df)
    print(report)

    if rep:
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(report, encoding="utf-8")
        log.info("Resumen guardado en: %s", rep)

if __name__ == "__main__":
    main()

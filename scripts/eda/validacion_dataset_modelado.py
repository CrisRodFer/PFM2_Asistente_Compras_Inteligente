# =============================================================================
# Script: validacion_dataset_modelado.py
# Descripción:
#   Validación inicial del dataset de modelado:
#   - Columnas y tipos (incluye captura de df.info())
#   - Target (nulos/negativos/descr.)
#   - Cobertura temporal y duplicados (product_id, date)
#   - Continuidad diaria y series constantes
#   - Chequeo rápido de precio si existe
#   - Clústeres y columnas de outliers
#   NO genera archivos salvo que se pase --report.
#
# Uso:
#   python scripts/eda/validacion_dataset_modelado.py
#   python scripts/eda/validacion_dataset_modelado.py --in data/processed/subset_modelado.parquet
#   python scripts/eda/validacion_dataset_modelado.py --report reports/validacion_dataset.txt
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import io
import pandas as pd
import numpy as np

# ---------- Rutas por defecto (ajusta si cambias estructura) -----------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # <raíz del proyecto>
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ---------- Logging -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ---------- Core --------------------------------------------------------------
def validate_dataset(df: pd.DataFrame, target: str = "demand_final_noised") -> str:
    lines: list[str] = []

    # 1) Columnas / tipos
    lines.append("=== ENCABEZADOS ===")
    lines.append(str(list(df.columns)))

    lines.append("\n=== INFO ===")
    buf = io.StringIO()                 # <-- FIX: buffer con .write()
    df.info(buf=buf, show_counts=True)  # captura df.info()
    lines.extend(buf.getvalue().splitlines())

    # 2) Target
    assert target in df.columns, f"No existe la columna objetivo '{target}'"
    tgt = df[target]
    lines.append(f"\n=== TARGET ({target}) ===")
    lines.append(f"Nulos: {int(tgt.isna().sum())}")
    lines.append(f"Negativos: {int((tgt < 0).sum())}")
    lines.append(str(tgt.describe()))

    # 3) Fechas y cobertura
    assert "date" in df.columns, "Falta columna 'date'"
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    lines.append("\n=== COBERTURA GLOBAL DE FECHAS ===")
    lines.append(f"Min: {df['date'].min()}  |  Max: {df['date'].max()}")

    # Duplicados product_id+date
    dups = int(df.duplicated(["product_id", "date"]).sum())
    lines.append(f"\nDuplicados (product_id, date): {dups}")

    # Continuidad diaria por producto
    span = df.groupby("product_id")["date"].agg(["min", "max", "count"])
    span["dias_esperados"] = (span["max"] - span["min"]).dt.days + 1
    span["completitud_%"] = (span["count"] / span["dias_esperados"] * 100).round(2)
    faltantes = int((span["completitud_%"] < 100).sum())
    lines.append(f"Productos con fechas faltantes: {faltantes}")
    lines.append(f"Completitud media %: {span['completitud_%'].mean().round(2)}")

    # Series constantes (sin variación)
    var0 = int((df.groupby("product_id")[target].nunique() == 1).sum())
    lines.append(f"Productos con demanda constante (único valor): {var0}")

    # 4) Precio (si existe)
    for col in ["precio_medio", "price_factor_effective"]:
        if col in df.columns:
            lines.append(f"\n=== CHEQUEO {col} ===")
            lines.append(
                f"Nulos: {int(df[col].isna().sum())} | Negativos: {int((df[col] < 0).sum())} "
                f"| Min: {df[col].min()} | Max: {df[col].max()}"
            )

    # 5) Clústeres
    cluster_col = "__cluster__" if "__cluster__" in df.columns else ("cluster" if "cluster" in df.columns else None)
    assert cluster_col is not None, "No hay columna de cluster ni __cluster__"
    lines.append(f"\n=== CLÚSTERES ({cluster_col}) ===")
    lines.append(f"Productos únicos: {df['product_id'].nunique()}")
    lines.append(f"Productos con cluster: {df.loc[df[cluster_col].notna(), 'product_id'].nunique()}")
    lines.append(f"Productos SIN cluster: {df.loc[df[cluster_col].isna(), 'product_id'].nunique()}")

    if {"cluster", "__cluster__", "is_outlier"}.issubset(df.columns):
        no_out = df["is_outlier"].eq(0)
        iguales = (df.loc[no_out, "cluster"].fillna(-1).astype(int)
                   == df.loc[no_out, "__cluster__"].astype(int)).all()
        lines.append(f"Cluster y __cluster__ idénticos en NO-outliers: {bool(iguales)}")

    # 6) Outliers
    outlier_cols = [c for c in df.columns if "outlier" in c.lower()]
    lines.append("\n=== COLUMNAS OUTLIERS ===")
    lines.append(str(outlier_cols))
    if "is_outlier" in df.columns:
        n_out = int(df.query("is_outlier == 1")["product_id"].nunique())
        lines.append(f"Productos outlier: {n_out}")
        asign = df.loc[df["is_outlier"] == 1, ["product_id", cluster_col]].drop_duplicates()
        lines.append(f"Clusters distintos en outliers: {asign[cluster_col].nunique()}")

    # 7) Semáforo
    checks = {
        "target_sin_nulos": int(tgt.isna().sum()) == 0,
        "target_sin_negativos": int((tgt < 0).sum()) == 0,
        "sin_duplicados_pid_fecha": dups == 0,
        "cluster_cubierto": df.loc[df[cluster_col].isna(), "product_id"].nunique() == 0,
    }
    lines.append("\n=== RESUMEN (OK=True) ===")
    for k, v in checks.items():
        lines.append(f"{k}: {bool(v)}")

    return "\n".join(lines)

# ---------- CLI ----------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validación inicial del dataset de modelado (no escribe por defecto).")
    p.add_argument("--in", dest="inp", type=str, default=str(PROCESSED_DIR / "subset_modelado.parquet"),
                   help="Ruta de entrada (PARQUET).")
    p.add_argument("--report", dest="report", type=str, default="",
                   help="Ruta TXT para volcar el reporte (opcional).")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    inp = Path(args.inp)
    rep = Path(args.report) if args.report else None

    log.info("Leyendo: %s", inp)
    df = pd.read_parquet(inp)

    log.info("Validando…")
    report = validate_dataset(df)

    print(report)

    if rep:
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(report, encoding="utf-8")
        log.info("Reporte guardado en: %s", rep)

if __name__ == "__main__":
    main()

# =============================================================================
# Script: fi_reprocess.py
# Descripción:
# Re-procesa importancias ya calculadas (fi_long_RF/XGB.csv) para:
#   - Excluir target y derivados directos (sales_quantity, demand_day_priceadj…).
#   - Normalizar importancias por clúster (suma 100) si procede.
#   - Recalcular resúmenes globales (media entre clústeres).
#   - Generar Top-k y barplots globales.
#
# Flujo del pipeline:
# 1) Lectura de fi_long_RF/XGB.csv desde carpeta de outputs.
# 2) Limpieza (exclusión + normalización por clúster).
# 3) Re-escritura de fi_long limpios.
# 4) Exportación de fi_summary, fi_topk y barplots globales.
#
# Input:
#   - outputs/modeling/ml/feature_importance/fi_long_RF.csv
#   - outputs/modeling/ml/feature_importance/fi_long_XGB.csv
#
# Output:
#   - outputs/modeling/ml/feature_importance/fi_long_{MODEL}.csv (limpio)
#   - outputs/modeling/ml/feature_importance/fi_summary_{MODEL}.csv
#   - outputs/modeling/ml/feature_importance/fi_top{K}_{MODEL}.csv
#   - outputs/modeling/ml/feature_importance/{model}_top{K}_summary.png
#
# Dependencias:
#   - pandas
#   - numpy
#   - matplotlib
#
# Instalación rápida:
#   pip install pandas numpy matplotlib
# =============================================================================

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== 0. CONFIG (RUTAS BASE) ==================================================
ROOT_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT_DIR / "outputs" / "modeling" / "ml" / "feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "sales_quantity"
EXCLUDE = {TARGET, "demand_day_priceadj", "demand_adjust"}

# ==== 1. IMPORTS + LOGGING ====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _filter_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    if "feature" not in df.columns:
        return df
    return df[~df["feature"].isin(EXCLUDE)].copy()

def _normalize_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    if "cluster" not in df.columns:
        return df
    df = df.copy()
    df["importance"] = df.groupby("cluster")["importance"].transform(
        lambda s: (s / s.sum()) * 100 if s.sum() != 0 else s
    )
    return df

def _plot_summary(mean_imp: pd.DataFrame, topk: int, title: str, out: Path):
    top = mean_imp.head(topk).iloc[::-1]
    plt.figure(figsize=(8, max(3, 0.35 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.title(title)
    plt.xlabel("Importancia media (%)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("Guardado: %s", out)

def _process_one(tag: str, path: Path, outdir: Path, topk: int):
    df = _read_csv(path)
    if df is None or df.empty:
        log.warning("[%s] fi_long no encontrado o vacío", tag)
        return
    df = _normalize_by_cluster(_filter_exclusions(df))
    df.to_csv(outdir / f"fi_long_{tag}.csv", index=False)

    mean_imp = df.groupby("feature", as_index=False)["importance"].mean()
    mean_imp = mean_imp.sort_values("importance", ascending=False)
    mean_imp.to_csv(outdir / f"fi_summary_{tag}.csv", index=False)
    mean_imp.head(topk).to_csv(outdir / f"fi_top{topk}_{tag}.csv", index=False)

    _plot_summary(mean_imp, topk, f"{tag} Top {topk} (media global)",
                  outdir / f"{tag.lower()}_top{topk}_summary.png")

# ==== 3. LÓGICA PRINCIPAL =====================================================
def reprocess_importances(fi_long_rf: Path, fi_long_xgb: Path, outdir: Path, topk: int):
    _process_one("RF", fi_long_rf, outdir, topk)
    _process_one("XGB", fi_long_xgb, outdir, topk)

# ==== 4. EXPORTACIÓN / I/O OPCIONAL ========================================== #
# (ya integrada en _process_one)

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-procesa fi_long_* para generar resúmenes/plots limpios.")
    p.add_argument("--fi-long-rf", type=str, default=str(OUT_DIR / "fi_long_RF.csv"))
    p.add_argument("--fi-long-xgb", type=str, default=str(OUT_DIR / "fi_long_XGB.csv"))
    p.add_argument("--outdir", type=str, default=str(OUT_DIR))
    p.add_argument("--topk", type=int, default=12)
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    reprocess_importances(Path(args.fi_long_rf), Path(args.fi_long_xgb),
                          Path(args.outdir), args.topk)

if __name__ == "__main__":
    main()

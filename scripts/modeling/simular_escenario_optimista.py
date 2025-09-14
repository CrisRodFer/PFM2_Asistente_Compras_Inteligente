# =============================================================================
# Script: simular_escenario_optimista.py
# Descripción:
#   Genera el escenario OPTIMISTA a partir del baseline estacionalizado,
#   usando factores derivados del WAPE medio por clúster (backtesting 8.5).
#
# Flujo del pipeline:
# 1) Carga baseline y métricas (WAPE por clúster)
# 2) Cálculo de factores (optimista = 1 + WAPE_%/100)
# 3) Aplicación de factores al baseline
# 4) Validación y exportación (parquet + controles CSV)
#
# Input (por defecto):
#   - data/processed/predicciones_2025_estacional.parquet
#   - reports/backtests/metrics_all.csv
#
# Output (por defecto):
#   - data/processed/predicciones_2025_optimista.parquet
#   - outputs/controles_escenarios/control_totales_optimista.csv
#   - outputs/controles_escenarios/control_por_cluster_optimista.csv
#
# Dependencias:
#   - pandas
#   - numpy
#
# Instalación rápida:
#   pip install pandas numpy pyarrow
# =============================================================================

from __future__ import annotations

from pathlib import Path
import sys, json, datetime as dt, importlib, importlib.util
import argparse, logging, pandas as pd
from typing import Dict, Any

# ==== 0) CONFIG DE RUTAS Y sys.path ===========================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
UTILS_DIR = SCRIPTS_DIR / "utils"

for p in (str(ROOT_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ==== 1) IMPORTS + LOGGING ====================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(Path(__file__).stem)

def _import_utils() -> Dict[str, Any]:
    """Carga la utilidad simular_escenario de utils."""
    candidates = ["scripts.utils.simular_escenario", "scripts.utils.simular_escenarios"]
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            return {
                "load_baseline": mod.load_baseline,
                "load_metrics": mod.load_metrics,
                "calcular_factores": mod.calcular_factores,
                "aplicar_factores": mod.aplicar_factores,
                "validate_structure": mod.validate_structure,
                "generar_controles": mod.generar_controles,
                "ensure_dirs": mod.ensure_dirs,
            }
        except ModuleNotFoundError:
            pass
    # fallback: carga por ruta
    util_path = next(UTILS_DIR.glob("simular_escen*.py"))
    spec = importlib.util.spec_from_file_location(util_path.stem, util_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return {
        "load_baseline": mod.load_baseline,
        "load_metrics": mod.load_metrics,
        "calcular_factores": mod.calcular_factores,
        "aplicar_factores": mod.aplicar_factores,
        "validate_structure": mod.validate_structure,
        "generar_controles": mod.generar_controles,
        "ensure_dirs": mod.ensure_dirs,
    }

U = _import_utils()

# ==== 2) CLI ==================================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera el escenario OPTIMISTA.")
    p.add_argument("--baseline", type=Path, default=PROCESSED_DIR / "predicciones_2025_estacional.parquet")
    p.add_argument("--metrics",  type=Path, default=REPORTS_DIR / "backtests" / "metrics_all.csv")
    p.add_argument("--out",      type=Path, default=PROCESSED_DIR / "predicciones_2025_optimista.parquet")
    p.add_argument("--controles-dir", type=Path, default=OUTPUTS_DIR / "controles_escenarios")
    p.add_argument("--date-col",    type=str, default="date")
    p.add_argument("--cluster-col", type=str, default="cluster_id")
    p.add_argument("--yhat-col",    type=str, default="y_pred_estacional")
    p.add_argument("--metrics-cluster-col", type=str, default="cluster")
    p.add_argument("--metrics-wape-col",    type=str, default="WAPE_%")
    p.add_argument("--year", type=int, default=2025)
    return p.parse_args()

# ==== 3) LÓGICA PRINCIPAL =====================================================
def run(args: argparse.Namespace) -> None:
    log.info("Leyendo baseline: %s", args.baseline)
    df_base = U["load_baseline"](args.baseline, date_col=args.date_col,
                                 cluster_col=args.cluster_col, yhat_col=args.yhat_col)

    log.info("Leyendo métricas: %s", args.metrics)
    metrics = U["load_metrics"](args.metrics, cluster_col=args.metrics_cluster_col,
                                wape_col=args.metrics_wape_col)

    log.info("Calculando factores (optimista = 1 + WAPE_%/100)")
    factors = U["calcular_factores"](metrics, escenario="optimista",
                                     cluster_col=args.metrics_cluster_col,
                                     wape_col=args.metrics_wape_col)

    # 🔑 Ajuste de columna: cluster → cluster_id
    factors = factors.rename(columns={args.metrics_cluster_col: args.cluster_col})

    U["ensure_dirs"](args.controles_dir)
    factors.to_csv(args.controles_dir / "factores_optimista.csv", index=False)

    df_opt = U["aplicar_factores"](df_base, factors, on=[args.cluster_col], yhat_col=args.yhat_col)

    U["validate_structure"](df_opt, date_col=args.date_col, cluster_col=args.cluster_col,
                            yhat_col=args.yhat_col, expect_year=args.year)

    U["ensure_dirs"](args.out.parent)
    df_opt.to_parquet(args.out, index=False)

    U["generar_controles"](df_base, df_opt, args.controles_dir, escenario="optimista",
                           cluster_col=args.cluster_col, yhat_col=args.yhat_col)

    meta = {
        "escenario": "optimista",
        "baseline": str(args.baseline),
        "metrics": str(args.metrics),
        "out_parquet": str(args.out),
        "total_base": float(df_base[args.yhat_col].sum()),
        "total_escenario": float(df_opt[args.yhat_col].sum()),
        "cols": {"date_col": args.date_col, "cluster_col": args.cluster_col, "yhat_col": args.yhat_col,
                 "metrics_cluster_col": args.metrics_cluster_col, "metrics_wape_col": args.metrics_wape_col},
    }
    (args.controles_dir / "meta_optimista.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info("✅ Escenario OPTIMISTA generado correctamente.")

# ==== 4) MAIN =================================================================
def main() -> None:
    args = _parse_args()
    run(args)

if __name__ == "__main__":
    main()
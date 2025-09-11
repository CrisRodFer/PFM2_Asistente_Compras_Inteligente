# scripts/transform/construir_metricas_unificadas.py
# =============================================================================
# Descripción
# ----------
# Construye las *métricas unificadas* para la fase 8.3.4 a partir del fichero
# generado por el entrenamiento SARIMAX. Usa la herramienta genérica
# `integrar_metricas()` (definida en scripts/utils/integrar_metricas.py) para:
#   1) Leer `outputs/modeling/sarimax/metrics_val_2024.csv`.
#   2) Normalizar columnas y unificar el nombre de los modelos.
#   3) Sustituir los sMAPE por los valores de autoridad validados.
#   4) Exportar:
#        - `metrics_unificados.csv`  (tabla completa)
#        - `ganadores_por_cluster.csv` (modelo con menor sMAPE por clúster)
#
# Entradas
# --------
# - outputs/modeling/sarimax/metrics_val_2024.csv
#     (producido por `scripts/modeling/sarimax_por_cluster.py`)
#
# Salidas
# -------
# - reports/metrics_unificados.csv
# - reports/ganadores_por_cluster.csv
#
# Uso
# ---
#   (.venv) python scripts/transform/construir_metricas_unificadas.py
#
# Dependencias
# ------------
#   pip install pandas numpy
#
# Notas
# -----
# - Este script **no** re-entrena nada: únicamente integra y guarda métricas.
# - Si cambian los valores “autoridad” de sMAPE, actualízalos en
#   `scripts/utils/integrar_metricas.py`.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import pandas as pd

from pathlib import Path
import sys

# Herramienta de integración (no genera ficheros por sí misma)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.utils.integrar_metricas import integrar_metricas


# ----------------------------- Rutas del proyecto -----------------------------
# Raíz del repo (…/PFM2_Asistente_Compras_Inteligente/)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Entrada (métricas crudas del entrenamiento SARIMAX)
INP_METRICS = ROOT_DIR / "outputs" / "modeling" / "sarimax" / "metrics_val_2024.csv"

# Carpeta y ficheros de salida (en transform)
OUT_DIR = ROOT_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_METRICS = OUT_DIR / "metrics_unificados.csv"
OUT_WINNERS = OUT_DIR / "ganadores_por_cluster.csv"


# --------------------------------- Ejecución ---------------------------------
if __name__ == "__main__":
    # 1) Integrar métricas (NO guarda dentro de la herramienta)
    df = integrar_metricas(path_sarimax=INP_METRICS, save=False)

    # 2) Guardar tabla completa
    df.to_csv(OUT_METRICS, index=True)
    print(f"[OK] Métricas unificadas guardadas en: {OUT_METRICS}")

    # 3) Calcular y guardar ganadores por clúster (mínimo sMAPE)
    winners = df.loc[df.groupby("cluster")["smape"].idxmin(), ["cluster", "model", "smape", "status"]]
    winners = winners.sort_values("cluster").reset_index(drop=True)
    winners.to_csv(OUT_WINNERS, index=False)
    print(f"[OK] Ganadores por clúster guardados en: {OUT_WINNERS}")

    # (opcional) Ver por pantalla
    print("\nGanador por clúster (por sMAPE):")
    print(winners)

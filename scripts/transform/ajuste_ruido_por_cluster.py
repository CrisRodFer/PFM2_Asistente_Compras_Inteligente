# =============================================================================
# Script: ajuste_ruido_por_cluster.py
# Descripción:
# Script ejecutable que aplica, de extremo a extremo, el ajuste de ruido por
# clúster usando las utilidades de `utils/ajuste_ruido.py`. Lee un parquet de
# entrada, calcula métricas, propone factores, aplica el ajuste y escribe un
# parquet de salida.
#
# Flujo del pipeline:
# 1) Leer parquet de entrada (procesado).
# 2) Calcular métricas de ruido por clúster.
# 3) Proponer factores k_c (con opción de override manual).
# 4) Ajustar N^adj = F + (N - F) * k_c.
# 5) Guardar parquet de salida.
#
# Input (por defecto):
#   - data/processed/demanda_all_adjusted.parquet
#
# Output (por defecto):
#   - data/processed/demanda_all_adjusted_postnoise.parquet
#
# Dependencias:
#   - pandas
#   - numpy
#   - pyarrow
#
# Instalación rápida:
#   pip install pandas numpy pyarrow
# =============================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# === 0) Rutas (ajusta si tu estructura difiere) ==============================
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

P_IN = DATA_DIR / "demanda_all_adjusted.parquet"
P_OUT = DATA_DIR / "demanda_all_adjusted_postnoise.parquet"

# Config
CLUSTER_COL = "Cluster"   # nombre tal y como sale del dataset
OUT_COL = "Demand_Final_Noiseds_adj"  # nueva columna con el ruido ajustado

# Target de ruido medio por clúster (22%)
TGT = 0.22
EPS = 1e-9


# === Funciones (idénticas a la lógica del notebook) ==========================
def _detectar_columnas(df: pd.DataFrame) -> tuple[str, str]:
    """
    Devuelve (col_final, col_noised) detectando de forma robusta
    variaciones de nombre (mayús, minús, 'noised', 'noises', etc.).
    """
    cols = list(df.columns)
    cols_low = [c.lower() for c in cols]

    # Candidatas exactas / habituales
    alias_final = {"demand_final", "demand day final", "demand_day_final", "final"}
    alias_noised = {
        "demand_final_noised", "demand_final_noiseds", "demand_final_noises",
        "noised", "noises"
    }

    # 1) exact match por alias
    cand_final = [c for c, cl in zip(cols, cols_low) if cl in alias_final or cl == "demand_final"]
    cand_noised = [c for c, cl in zip(cols, cols_low) if cl in alias_noised]

    # 2) si no encuentra, heurística por sufijo/contención
    if not cand_final:
        cand_final = [c for c, cl in zip(cols, cols_low)
                      if cl.endswith("demand_final") or ("demand" in cl and "final" in cl)]
    if not cand_noised:
        cand_noised = [c for c, cl in zip(cols, cols_low)
                       if ("noised" in cl and "demand" in cl) or ("noises" in cl and "demand" in cl)]

    if not cand_final:
        raise KeyError(f"No se encontró la columna de demanda FINAL. Columnas disponibles: {cols[:15]}...")
    if not cand_noised:
        raise KeyError(f"No se encontró la columna de demanda NOISED. Columnas disponibles: {cols[:15]}...")

    col_final = cand_final[0]
    col_noised = cand_noised[0]
    print(f"→ Columnas detectadas: FINAL='{col_final}' | NOISED='{col_noised}'")
    return col_final, col_noised


def _resumen_ruido(df: pd.DataFrame, cluster_col: str, final_col: str, noised_col: str) -> pd.DataFrame:
    """Resumen por clúster de medias y % ruido."""
    g = (df.groupby(cluster_col)[[final_col, noised_col]].mean()
           .rename(columns={final_col: "mean_final", noised_col: "mean_noised"}))
    g["noise_pct"] = (g["mean_noised"] - g["mean_final"]) / (g["mean_final"] + EPS)
    return g


def _factores_kc(g_before: pd.DataFrame, tgt: float) -> pd.Series:
    """
    Calcula factores k_c por clúster:
        delta_c = (mean_noised - mean_final) / mean_final
        k_c = min(1, tgt / max(delta_c, EPS))
    """
    delta = g_before["noise_pct"].astype(float).clip(lower=EPS)
    k = (tgt / delta).clip(upper=1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    # presentación similar a la del notebook:
    tabla = pd.DataFrame({"k_c": k.round(2)}).T
    print("\nFactores k_c propuestos (truncados a 1, redondeo 2 dec.):")
    print(tabla)
    return k


def _aplicar_ajuste(df: pd.DataFrame,
                    k: pd.Series,
                    cluster_col: str,
                    final_col: str,
                    noised_col: str,
                    out_col: str) -> pd.Series:
    """
    Aplica el ajuste SOLO sobre el componente de ruido:
        delta_t = (Noised - Final)
        Noised_adj = Final + delta_t * k_c[cluster]
        (clip inferior a 0 para seguridad)
    """
    delta = df[noised_col] - df[final_col]
    k_vec = df[cluster_col].map(k).astype(float)
    return (df[final_col] + delta * k_vec).clip(lower=0)


def main() -> None:
    assert P_IN.exists(), f"No existe el fichero de entrada: {P_IN}"

    # 1) Carga
    df = pd.read_parquet(P_IN)

    # 2) Detección de columnas robusta (como en el notebook)
    final_col, noised_col = _detectar_columnas(df)

    # 3) Métricas ANTES
    print("\n=== Ruido medio ANTES (por clúster) ===")
    g_before = _resumen_ruido(df, CLUSTER_COL, final_col, noised_col)
    print(g_before)

    # 4) Factores k_c
    k = _factores_kc(g_before, TGT)

    # 5) Aplicar ajuste
    df[OUT_COL] = _aplicar_ajuste(df, k, CLUSTER_COL, final_col, noised_col, OUT_COL)

    # 6) Métricas DESPUÉS
    print("\n=== Ruido medio DESPUÉS (por clúster) ===")
    g_after = (df.groupby(CLUSTER_COL)[[final_col, OUT_COL]].mean()
                 .rename(columns={final_col: "mean_final", OUT_COL: "mean_noised_adj"}))
    g_after["noise_pct_adj"] = (g_after["mean_noised_adj"] - g_after["mean_final"]) / (g_after["mean_final"] + EPS)
    print(g_after)

    # 7) Guardar
    df.to_parquet(P_OUT, index=False)
    print(f"\n✅ Guardado con éxito: {P_OUT}")

    # 8) Resumen compacto
    compact = (g_before[["mean_final", "mean_noised", "noise_pct"]]
               .join(g_after[["mean_noised_adj", "noise_pct_adj"]]))
    print("\nResumen compacto (medias y % ruido):")
    print(compact)


if __name__ == "__main__":
    main()
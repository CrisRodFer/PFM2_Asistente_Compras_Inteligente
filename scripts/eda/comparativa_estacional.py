

# scripts/eda/comparativa_estacional.py
# =============================================================================
# Comparativa estacional 2022–2024:
#   1) Evolución diaria total (alineada por MM-DD, sin 29/02) con curva cruda + suavizada (rolling 7)
#   2) Media diaria mensual
# KPI:
#   - Correlaciones entre curvas diarias
#   - CV mensual (sd/mean) entre años
# Salidas:
#   outputs/figures/evolucion_diaria_total.png
#   outputs/figures/evolucion_mensual_media.png
#   outputs/tables/correlacion_curvas_diarias.csv
#   outputs/tables/media_mensual_y_cv.csv
#   outputs/tables/resumen_kpis_comparativa.csv
# =============================================================================

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Bootstrapping rutas -----------------------------
def find_project_root(start: Path, marker: str = "scripts", max_up: int = 8) -> Path:
    p = start.resolve()
    for _ in range(max_up):
        if (p / marker).is_dir():
            return p
        p = p.parent
    raise RuntimeError(f"No se encontró la carpeta '{marker}' hacia arriba.")

ROOT = find_project_root(Path(__file__).parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
FIGS = ROOT / "outputs" / "figures"
TBLS = ROOT / "outputs" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TBLS.mkdir(parents=True, exist_ok=True)

FILES = {
    2022: PROC / "demanda_diaria_2022.csv",
    2023: PROC / "demanda_diaria_2023.csv",
    2024: PROC / "demanda_diaria_2024.csv",
}

# --------------------------- Utilidades de carga -----------------------------
def load_daily_total(path: Path, year: int) -> pd.DataFrame:
    """
    Lee la desagregación diaria de un año y devuelve:
      Date (datetime), Year (int), Total (float)
    donde Total = suma sobre productos.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.groupby("Date", as_index=False)["Demand_Day"].sum()
    df = df.rename(columns={"Demand_Day": "Total"})
    df["Year"] = year
    return df

def align_by_mmdd(daily_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Alinea por MM-DD (excluye 29/02) y devuelve una matriz con index MM-DD
    y columnas = años, con la demanda total diaria.
    """
    mat = {}
    for y, df in daily_dict.items():
        tmp = df.copy()
        tmp["mmdd"] = tmp["Date"].dt.strftime("%m-%d")
        tmp = tmp[tmp["mmdd"] != "02-29"]  # quitar 29/02
        ser = tmp.groupby("mmdd")["Total"].sum()
        mat[y] = ser
    mat_df = pd.concat(mat, axis=1).sort_index()
    return mat_df

# --------------------------- Gráfica: evolución diaria -----------------------
def plot_evolucion_diaria(daily_dict: dict[int, pd.DataFrame], out_path: Path, out_corr: Path) -> pd.DataFrame:
    """
    Dibuja curvas diarias totales por año (cruda + suavizada rolling(7)).
    Guarda correlaciones en CSV y devuelve la matriz diaria alineada.
    """
    mat_df = align_by_mmdd(daily_dict)
    corr = mat_df.corr(method="pearson")
    corr.round(4).to_csv(out_corr, index=True)
    print(f"📄 Correlaciones guardadas: {out_corr}")

    # Suavizado 7 días centrado
    mat_smooth = mat_df.rolling(7, center=True, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(14, 4))
    x = range(1, len(mat_df) + 1)
    # cruda (fina y translúcida)
    for y in sorted(mat_df.columns):
        plt.plot(x, mat_df[y].values, alpha=0.25, linewidth=1.0, label=f"{y} (cruda)")
    # suavizada (más visible)
    for y in sorted(mat_smooth.columns):
        plt.plot(x, mat_smooth[y].values, linewidth=2.0, label=f"{y} (suav.)")
    plt.title("Evolución diaria total por año (alineada por MM-DD, sin 29/02)")
    plt.xlabel("Día del año (1..365)")
    plt.ylabel("Demanda total diaria")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"🖼️  Guardado: {out_path}")

    return mat_df

# --------------------------- Gráfica: media mensual --------------------------
def plot_media_mensual(daily_dict: dict[int, pd.DataFrame], out_path: Path, out_tbl: Path, out_resume: Path) -> None:
    """
    Calcula la media diaria por mes y la dibuja por año.
    Guarda tabla con medias por año + CV mensual, y un resumen con % meses CV bajo.
    """
    monthly = {}
    for y, df in daily_dict.items():
        tmp = df.copy()
        tmp["Month"] = tmp["Date"].dt.month
        monthly[y] = tmp.groupby("Month")["Total"].mean()

    mon_df = pd.DataFrame(monthly).sort_index()
    # CV entre años por mes
    cv = (mon_df.std(axis=1) / mon_df.mean(axis=1)).rename("CV")
    tabla = mon_df.copy()
    tabla["CV"] = cv
    tabla.index.name = "Mes"
    tabla.round(4).to_csv(out_tbl)
    print(f"📄 Medias mensuales + CV guardadas: {out_tbl}")

    # Resumen con umbral CV
    thr_cv = 0.15
    pct_ok = float((cv <= thr_cv).mean())  # 0..1
    resume = pd.DataFrame({
        "umbral_cv": [thr_cv],
        "porc_meses_cv_ok": [pct_ok]
    })
    resume.round(4).to_csv(out_resume, index=False)
    print(f"📄 Resumen KPIs (CV) guardado: {out_resume} | % meses CV≤{thr_cv:.2f} = {pct_ok*100:.1f}%")

    # Plot
    plt.figure(figsize=(9, 4))
    for y in sorted(mon_df.columns):
        plt.plot(mon_df.index, mon_df[y].values, marker="o", label=str(y))
    plt.title("Demanda media diaria mensual por año")
    plt.xlabel("Mes")
    plt.ylabel("Demanda media diaria")
    plt.xticks(range(1, 13))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"🖼️  Guardado: {out_path}")

# --------------------------- Main -------------------------------------------
def main():
    # 1) Cargar y preparar
    daily = {}
    for y, p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {p}")
        daily[y] = load_daily_total(p, y)
        print(f"✓ {y}: filas={len(daily[y])}, rango={daily[y]['Date'].min().date()} → {daily[y]['Date'].max().date()}")

    # 2) Evolución diaria total (sin 29/02) + correlaciones
    out1 = FIGS / "evolucion_diaria_total.png"
    out_corr = TBLS / "correlacion_curvas_diarias.csv"
    _ = plot_evolucion_diaria(daily, out1, out_corr)

    # 3) Media diaria mensual + CV
    out2 = FIGS / "evolucion_mensual_media.png"
    out_tbl = TBLS / "media_mensual_y_cv.csv"
    out_resume = TBLS / "resumen_kpis_comparativa.csv"
    plot_media_mensual(daily, out2, out_tbl, out_resume)

    print("✅ Visualizaciones y KPIs generados.")

if __name__ == "__main__":
    main()

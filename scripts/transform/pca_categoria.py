
# =============================================================================
# Script: pca_categoria.py
# Ubicación: scripts/transform/pca_categoria.py
#
# Objetivo:
#   Reducir dimensionalidad en el espacio de CATEGORÍAS mediante PCA.
#   Se modela cada categoría con un vector de rasgos derivados de su
#   serie temporal agregada (perfil mensual, semanal, totales por año,
#   crecimientos, volatilidad, asimetría robusta).
#
# Flujo:
#   1) Cargar data/processed/demanda_filtrada.csv
#   2) (Opcional) Unificar categorías poco representativas en "Otros"
#   3) Construir matriz Categoria x Rasgos
#   4) Estandarizar y aplicar PCA (varianza objetivo o min_components)
#   5) Exportar features, scores, componentes y scree plot
#
# Entradas:
#   data/processed/demanda_filtrada.csv
#
# Salidas:
#   data/processed/pca/categoria_features.csv
#   data/processed/pca/categoria_scores.csv
#   data/processed/pca/pca_componentes.csv     (cargas por rasgo)
#   reports/figures/pca_scree.png              (gráfico varianza explicada)
#
# Uso:
#   - Terminal:  python scripts/transform/pca_categoria.py
#   - Notebook:  main([])
#
# Parámetros clave (CLI):
#   --min-share       Proporción mínima de demanda para mantener categoría (por defecto 0.01 = 1%)
#   --min-products    Nº mínimo de productos por categoría (por defecto 10)
#   --target-var      Varianza explicada objetivo (por defecto 0.90)
#   --min-components  Nº mínimo de componentes (por defecto 5)
#   --start / --end   Rango temporal (por defecto 2022-01-01 a 2024-12-31)
#
# Dependencias:
#   pandas, numpy, scikit-learn, matplotlib
# =============================================================================

from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np

# sklearn/mpl solo se usan en este script
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ---------- RUTAS BASE (notebook/terminal) ----------
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    here = Path.cwd()
    ROOT_DIR = here if (here / "data" / "processed").exists() else here.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PCA_DIR = PROCESSED_DIR / "pca"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

INPUT_PATH = PROCESSED_DIR / "demanda_filtrada.csv"
FEATURES_OUT = PCA_DIR / "categoria_features.csv"
SCORES_OUT = PCA_DIR / "categoria_scores.csv"
COMP_OUT = PCA_DIR / "pca_componentes.csv"
SCREE_OUT = FIGURES_DIR / "pca_scree.png"

REQUIRED_COLS = {"Product_ID", "Date", "Demand_Day", "Categoria", "Estado_Producto"}

# ---------- UTILIDADES I/O ----------
def exportar(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

def guardar_scree_plot(var_ratio_cum: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(var_ratio_cum) + 1), var_ratio_cum, marker="o")
    plt.xlabel("Número de Componentes")
    plt.ylabel("Varianza explicada acumulada")
    plt.title("Scree plot PCA (varianza explicada acumulada)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def cargar_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    df = pd.read_csv(path)
    faltan = REQUIRED_COLS - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {faltan}")
    # Tipos
    df["Product_ID"] = df["Product_ID"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Demand_Day"] = pd.to_numeric(df["Demand_Day"], errors="coerce")
    # Sanidad mínima
    if df["Date"].isna().any():
        raise ValueError("Existen fechas no parseables en 'Date'.")
    if (df["Demand_Day"] < 0).any():
        raise ValueError("Se detectaron demandas negativas. Revisa el filtrado previo.")
    # Solo activos por seguridad
    mask_activo = df["Estado_Producto"].astype(str).str.strip().str.lower().eq("activo")
    if not mask_activo.all():
        df = df[mask_activo].copy()
    return df

# ---------- AGREGACIÓN Y RASGOS POR CATEGORÍA ----------
def dias_en_anio(y: int) -> int:
    return 366 if pd.Timestamp(year=y, month=12, day=31).dayofyear == 366 else 365

def unificar_categorias_minor(df: pd.DataFrame, min_share: float, min_products: int) -> pd.DataFrame:
    """
    Reasigna a 'Otros' categorías con poca representatividad:
      - share de demanda total < min_share  O
      - nº de productos < min_products
    """
    tot_cat = df.groupby("Categoria")["Demand_Day"].sum().sort_values(ascending=False)
    share_cat = tot_cat / tot_cat.sum()
    prod_cat = df.groupby("Categoria")["Product_ID"].nunique()

    minor_flags = (share_cat < min_share) | (prod_cat < min_products)
    minor_cats = set(share_cat.index[minor_flags])

    if minor_cats:
        df = df.copy()
        df["Categoria"] = df["Categoria"].where(~df["Categoria"].isin(minor_cats), "Otros")
    return df

def construir_rasgos_categoria(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Devuelve DF de rasgos a nivel de categoría:
      - Perfil mensual (12 rasgos normalizados por suma de meses)
      - Perfil semanal (7 rasgos normalizados por suma semanal)
      - Totales por año (2022, 2023, 2024) y crecimientos (2023/2022, 2024/2023)
      - Volatilidad (std/mean) a nivel diario (robusta si mean>0)
      - Asimetría robusta (p95 / mediana)
    Todas las proporciones se calculan con salvaguardas ante divisiones por 0.
    """
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DoW"] = df["Date"].dt.dayofweek  # 0=Lunes

    # Serie diaria por categoría (para volatilidad y p95/mediana)
    daily = df.groupby(["Categoria", "Date"])["Demand_Day"].sum().reset_index()

    # Perfil mensual (proporciones mes / total)
    month_sum = df.groupby(["Categoria", "Month"])["Demand_Day"].sum().unstack(fill_value=0)
    month_tot = month_sum.sum(axis=1).replace(0, np.nan)
    month_share = month_sum.div(month_tot, axis=0).fillna(0)
    month_share.columns = [f"m_{m:02d}" for m in month_share.columns]  # m_01..m_12

    # Perfil semanal (proporciones dow / total)
    dow_sum = df.groupby(["Categoria", "DoW"])["Demand_Day"].sum().unstack(fill_value=0)
    dow_tot = dow_sum.sum(axis=1).replace(0, np.nan)
    dow_share = dow_sum.div(dow_tot, axis=0).fillna(0)
    dow_share.columns = [f"dow_{d}" for d in dow_share.columns]  # dow_0..dow_6

    # Totales por año y growth
    yearly = df.groupby(["Categoria", "Year"])["Demand_Day"].sum().unstack(fill_value=0)
    for y in [2022, 2023, 2024]:
        if y not in yearly.columns:
            yearly[y] = 0
    yearly = yearly[[2022, 2023, 2024]]
    yearly.columns = ["y2022", "y2023", "y2024"]

    def safe_growth(a, b):
        return (b / a) - 1 if a > 0 else np.nan

    growth_2322 = (yearly["y2022"].replace(0, np.nan), yearly["y2023"])
    growth_2423 = (yearly["y2023"].replace(0, np.nan), yearly["y2024"])
    g_23_22 = growth_2322[1] / growth_2322[0] - 1
    g_24_23 = growth_2423[1] / growth_2423[0] - 1
    g_23_22 = g_23_22.replace([np.inf, -np.inf], np.nan).fillna(0)
    g_24_23 = g_24_23.replace([np.inf, -np.inf], np.nan).fillna(0)
    growth = pd.DataFrame({"g_23_22": g_23_22, "g_24_23": g_24_23})

    # Volatilidad diaria y asimetría robusta
    vol = (daily.groupby("Categoria")["Demand_Day"]
           .agg(dmean="mean", dstd="std", p95=lambda s: np.percentile(s, 95), med="median"))
    vol["cv_daily"] = np.where(vol["dmean"] > 0, vol["dstd"] / vol["dmean"], 0.0)
    vol["p95_over_med"] = np.where(vol["med"] > 0, vol["p95"] / vol["med"], 0.0)
    vol = vol[["cv_daily", "p95_over_med"]]

    # Ensamble de rasgos
    feats = month_share.join(dow_share, how="outer")
    feats = feats.join(yearly, how="outer")
    feats = feats.join(growth, how="outer")
    feats = feats.join(vol, how="outer")
    feats = feats.fillna(0.0)

    # Orden estable de columnas
    ordered_cols = (
        [f"m_{i:02d}" for i in range(1, 13)]
        + [f"dow_{i}" for i in range(0, 7)]
        + ["y2022", "y2023", "y2024", "g_23_22", "g_24_23", "cv_daily", "p95_over_med"]
    )
    feats = feats.reindex(columns=ordered_cols).fillna(0.0)
    feats.index.name = "Categoria"
    return feats.reset_index()

# ---------- PCA ----------
def aplicar_pca(df_feats: pd.DataFrame, target_var: float, min_components: int):
    """
    Estandariza rasgos y aplica PCA.
    Devuelve:
      - scaler (fit)
      - pca (fit)
      - X_std (matriz estandarizada)
      - scores (coordenadas por categoría)
      - componentes (cargas por rasgo)
    """
    cat = df_feats["Categoria"].values
    X = df_feats.drop(columns=["Categoria"]).values

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)

    # Ajuste progresivo hasta cubrir target_var, con mínimo min_components
    pca_full = PCA(svd_solver="full", random_state=42).fit(X_std)
    var_ratio_cum = np.cumsum(pca_full.explained_variance_ratio_)
    # nº comp que llegan al objetivo
    k_target = int(np.searchsorted(var_ratio_cum, target_var) + 1)
    k = max(min_components, k_target)
    k = min(k, X_std.shape[1])  # no más que número de rasgos

    pca = PCA(n_components=k, svd_solver="full", random_state=42)
    scores = pca.fit_transform(X_std)

    # DataFrames de salida
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(k)])
    scores_df.insert(0, "Categoria", cat)

    componentes_df = pd.DataFrame(
        pca.components_,
        columns=df_feats.drop(columns=["Categoria"]).columns,
        index=[f"PC{i+1}" for i in range(k)]
    ).T.reset_index().rename(columns={"index": "feature"})

    return scaler, pca, X_std, scores_df, componentes_df, np.cumsum(pca.explained_variance_ratio_)

# ---------- CLI ----------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PCA sobre categorías a partir de demanda_filtrada.csv")
    p.add_argument("--in", dest="inp", default=str(INPUT_PATH), help="CSV de entrada (demanda_filtrada)")
    p.add_argument("--out-features", default=str(FEATURES_OUT))
    p.add_argument("--out-scores", default=str(SCORES_OUT))
    p.add_argument("--out-components", default=str(COMP_OUT))
    p.add_argument("--out-scree", default=str(SCREE_OUT))
    p.add_argument("--start", default="2022-01-01", help="Fecha inicio del período")
    p.add_argument("--end", default="2024-12-31", help="Fecha fin del período")
    p.add_argument("--min-share", type=float, default=0.01, help="Share mínimo de demanda para mantener categoría")
    p.add_argument("--min-products", type=int, default=10, help="Nº mínimo de productos por categoría")
    p.add_argument("--target-var", type=float, default=0.90, help="Varianza explicada objetivo (0–1)")
    p.add_argument("--min-components", type=int, default=5, help="Nº mínimo de componentes")
    return p.parse_args(argv)

# ---------- MAIN ----------
def main(argv=None):
    args = parse_args(argv)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    # 1) Cargar
    df = cargar_dataset(Path(args.inp))

    # 2) Unificar categorías minoritarias
    df_u = unificar_categorias_minor(df, min_share=args.min_share, min_products=args.min_products)

    # 3) Construir rasgos por categoría
    feats = construir_rasgos_categoria(df_u, start=start, end=end)

    # 4) PCA
    scaler, pca, X_std, scores_df, comp_df, var_ratio_cum = aplicar_pca(
        feats, target_var=args.target_var, min_components=args.min_components
    )

    # 5) Exportar
    PCA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    exportar(feats, Path(args.out_features))
    exportar(scores_df, Path(args.out_scores))
    exportar(comp_df, Path(args.out_components))
    guardar_scree_plot(var_ratio_cum, Path(args.out_scree))

    # 6) Log básico
    print(f"[OK] Features por categoría -> {args.out_features} (categorías: {feats['Categoria'].nunique()}, rasgos: {feats.shape[1]-1})")
    print(f"[OK] Scores PCA -> {args.out_scores} (PCs: {scores_df.shape[1]-1})")
    print(f"[OK] Componentes (cargas) -> {args.out_components}")
    print(f"[OK] Scree plot -> {args.out_scree}")
    print(f"[INFO] Varianza explicada acumulada: {', '.join([f'{v:.3f}' for v in var_ratio_cum])}")

    try:
        from IPython.display import display  # noqa
        # Vista rápida de resultados
        print("\nResumen primeros PCs:")
        display(scores_df.head())
    except Exception:
        pass

    return {
        "features": feats,
        "scores": scores_df,
        "components": comp_df,
        "var_ratio_cum": var_ratio_cum
    }

# ---------- ENTRYPOINT ----------
if __name__ == "__main__":
    if any("ipykernel" in arg for arg in sys.argv):
        main([])   
    else:
        main()    

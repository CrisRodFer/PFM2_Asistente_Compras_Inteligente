# =============================================================================
# Script: analisis_coherencia_historicos.py
# Descripción:
# Valida la coherencia entre los históricos simulados de demanda para 2022, 2023
# y 2024 mediante tres bloques: (1) análisis visual, (2) análisis estadístico y
# (3) contraste de hipótesis. Opcionalmente, restringe el análisis al conjunto
# de productos presente en los tres años y/o excluye novedades apoyándose en el
# catálogo limpio.
#
# Flujo del pipeline:
# 1) Cargar históricos.
# 2) Filtrar cohorte común / excluir novedades (opcional)
# 3) Visualizaciones: evolución mensual + boxplot media por producto
# 4) Estadísticos: descriptivos y correlaciones interanuales 
# 5) Contrastes: Friedman (diseño intra-sujetos) + Wilcoxon pareadas 
# 6) Exportar reportes a data/reports y figuras a outputs/figures 
#
# Input:
#   - data/clean/Historico_Ventas_2022.csv 
#   - data/clean/Historico_Ventas_2023.csv 
#   - data/clean/Historico_Ventas_2024.csv 
# 
# Output:
#   - outputs/figures/evolucion_mensual.png
#   - outputs/figures/boxplot_media_diaria_por_producto.png
#   - data/reports/estadisticos_descriptivos_coherencia.csv
#   - data/reports/correlaciones_coherencia.csv
#   - data/reports/tests_coherencia.json
#
# Dependencias:
#   - pandas, numpy, scipy, seaborn, matplotlib
#
# Instalación rápida:
#   pip install pandas numpy scipy seaborn matplotlib
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import json
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)
sns.set(style="whitegrid")

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    """Crea directorios si no existen."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def _find_input_for_year(base_dir: Path, year: int) -> Path:
    """Localiza el archivo de histórico para un año admitiendo csv/xlsx/parquet."""
    candidates = [
        base_dir / f"Historico_Ventas_{year}.csv",
        base_dir / f"Historico_Ventas_{year}.xlsx",
        base_dir / f"Historico_Ventas_{year}.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No se encontró el histórico para {year} en {base_dir} "
        f"(buscado como CSV/XLSX/Parquet)."
    )

def _read_any(path: Path) -> pd.DataFrame:
    """Lee CSV/XLSX/Parquet a DataFrame, normalizando columnas clave."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_parquet(path)
    # normalización mínima esperada
    rename = {c: c.strip() for c in df.columns}
    df.rename(columns=rename, inplace=True)
    assert {"Date", "Product_ID"}.issubset(df.columns), (
        f"Faltan columnas esperadas en {path}: {df.columns.tolist()}"
    )
    if "Sales Quantity" not in df.columns:
        if "Demand" in df.columns:
            df["Sales Quantity"] = df["Demand"]
        else:
            raise KeyError("No se encontró 'Sales Quantity' ni 'Demand' en el archivo.")
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Product_ID", "Date", "Sales Quantity"]].copy()

# ==== 3. LÓGICA PRINCIPAL =====================================================
def cargar_historicos(p2022: Path, p2023: Path, p2024: Path) -> pd.DataFrame:
    """Carga y concatena los históricos 2022–2024 añadiendo columna Year."""
    log.info("📥 Cargando históricos:\n  • %s\n  • %s\n  • %s", p2022, p2023, p2024)
    df22, df23, df24 = _read_any(p2022), _read_any(p2023), _read_any(p2024)
    df22["Year"], df23["Year"], df24["Year"] = 2022, 2023, 2024
    df = pd.concat([df22, df23, df24], ignore_index=True)
    log.info("🧩 Historicos unidos: %s filas, %s productos únicos.",
             f"{len(df):,}", f"{df['Product_ID'].nunique():,}")
    return df

def aplicar_cohorte_comun(df: pd.DataFrame, usar_cohorte: bool) -> pd.DataFrame:
    """Restringe (opcionalmente) el análisis a productos presentes en 2022–2024."""
    if not usar_cohorte:
        return df
    pres = df.groupby("Year")["Product_ID"].unique()
    missing = [y for y in (2022, 2023, 2024) if y not in pres.index]
    if missing:
        raise ValueError(f"Faltan años en los datos para cohorte común: {missing}")
    common = set(pres.loc[2022]) & set(pres.loc[2023]) & set(pres.loc[2024])
    before = df["Product_ID"].nunique()
    out = df[df["Product_ID"].isin(common)]
    after = out["Product_ID"].nunique()
    log.info("🔗 Cohorte común aplicada: %d → %d productos.", before, after)
    return out

# ---------- BLOQUE 1. VISUALIZACIONES ----------------------------------------
def plot_evolucion_mensual(df: pd.DataFrame, figs_dir: Path) -> Path:
    """Grafica y guarda la evolución mensual total por año."""
    ensure_dirs(figs_dir)
    mensual = (
        df.assign(Mes=lambda x: x["Date"].dt.to_period("M"))
          .groupby(["Mes", "Year"])["Sales Quantity"].sum()
          .reset_index()
    )
    mensual["Mes"] = mensual["Mes"].dt.to_timestamp()

    plt.figure(figsize=(12, 5))
    ax = sns.lineplot(data=mensual, x="Mes", y="Sales Quantity", hue="Year", marker="o")
    ax.set_title("Evolución mensual de la demanda total (2022–2024)")
    ax.set_xlabel("Mes"); ax.set_ylabel("Unidades")
    plt.xticks(rotation=45); plt.tight_layout()
    outpath = figs_dir / "evolucion_mensual.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    log.info("🖼️ Figura guardada: %s", outpath)
    return outpath

def plot_box_media_por_producto(df: pd.DataFrame, figs_dir: Path, log_scale: bool=False) -> Path:
    """Grafica y guarda boxplot de la media diaria por producto, por año."""
    ensure_dirs(figs_dir)
    mean_per_prod = df.groupby(["Product_ID", "Year"])["Sales Quantity"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=mean_per_prod, x="Year", y="Sales Quantity", palette="pastel", showfliers=False)
    ax.set_title("Distribución de la media diaria por producto")
    ax.set_xlabel("Año"); ax.set_ylabel("Media diaria (uds)")
    if log_scale:
        ax.set_yscale("log")
    plt.tight_layout()
    outpath = figs_dir / "boxplot_media_diaria_por_producto.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    log.info("🖼️ Figura guardada: %s", outpath)
    return outpath

# ---------- BLOQUE 2. ESTADÍSTICOS -------------------------------------------
def media_diaria_por_producto(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla ancha: filas=Product_ID, columnas=Year, valores=media diaria."""
    wide = df.groupby(["Product_ID", "Year"])["Sales Quantity"].mean().unstack()
    return wide.dropna()

def estadisticos_descriptivos(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula n, media, mediana, std, CV, IQR de la media diaria por producto."""
    wide = media_diaria_por_producto(df)
    stats_year = {}
    for y in [2022, 2023, 2024]:
        s = wide[y]
        stats_year[y] = {
            "n": s.size,
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(ddof=1),
            "cv": s.std(ddof=1) / (s.mean() if s.mean() != 0 else np.nan),
            "iqr": s.quantile(0.75) - s.quantile(0.25),
        }
    res = pd.DataFrame(stats_year).T
    log.info("📊 Descriptivos (media diaria por producto):\n%s", res.round(4))
    return res

def correlaciones_interanuales(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson y Spearman entre pares (22–23, 23–24, 22–24) de medias por producto."""
    wide = media_diaria_por_producto(df)

    def pair(a: int, b: int) -> Dict[str, float]:
        pear = stats.pearsonr(wide[a], wide[b])
        spear = stats.spearmanr(wide[a], wide[b])
        return {
            "pearson_r": float(pear.statistic) if hasattr(pear, "statistic") else float(pear[0]),
            "pearson_p": float(pear.pvalue) if hasattr(pear, "pvalue") else float(pear[1]),
            "spearman_rho": float(spear.correlation) if hasattr(spear, "correlation") else float(spear[0]),
            "spearman_p": float(spear.pvalue) if hasattr(spear, "pvalue") else float(spear[1]),
            "n": int(len(wide)),
        }

    rows = {
        "2022–2023": pair(2022, 2023),
        "2023–2024": pair(2023, 2024),
        "2022–2024": pair(2022, 2024),
    }
    dfcorr = pd.DataFrame(rows).T
    log.info("🔗 Correlaciones interanuales:\n%s", dfcorr.round(4))
    return dfcorr

# ---------- BLOQUE 3. CONTRASTES ---------------------------------------------
def tests_coherencia(df: pd.DataFrame) -> Dict[str, object]:
    """Contrastes sobre la media diaria por producto (intra-sujetos)."""
    wide = media_diaria_por_producto(df)
    x22, x23, x24 = wide[2022].values, wide[2023].values, wide[2024].values

    # Normalidad (D’Agostino K²)
    p_norm = {y: float(stats.normaltest(wide[y].values).pvalue) for y in [2022, 2023, 2024]}
    # Homocedasticidad (Levene)
    p_levene = float(stats.levene(x22, x23, x24, center="median").pvalue)

    # Friedman (intra-sujetos)
    p_friedman = float(stats.friedmanchisquare(x22, x23, x24).pvalue)

    # Post-hoc Wilcoxon pareadas + Bonferroni
    pairs = {
        "22–23": float(stats.wilcoxon(x22, x23).pvalue),
        "23–24": float(stats.wilcoxon(x23, x24).pvalue),
        "22–24": float(stats.wilcoxon(x22, x24).pvalue),
    }
    m = len(pairs)
    pairs_bonf = {k: min(v * m, 1.0) for k, v in pairs.items()}

    # Complemento (independientes)
    p_anova = float(stats.f_oneway(x22, x23, x24).pvalue)
    p_kruskal = float(stats.kruskal(x22, x23, x24).pvalue)

    out = {
        "normalidad_p": p_norm,
        "levene_p": p_levene,
        "friedman_p": p_friedman,
        "wilcoxon_bonferroni_p": pairs_bonf,
        "anova_p": p_anova,
        "kruskal_p": p_kruskal,
        "n_productos": int(len(wide)),
    }
    log.info("🧪 Tests de coherencia:\n%s", json.dumps(out, indent=2))
    return out

# ==== 4. EXPORTACIÓN / I/O ====================================================
def exportar_reportes(desc: pd.DataFrame, corr: pd.DataFrame, tests: Dict[str, object], reports_dir: Path) -> Dict[str, Path]:
    """Exporta tablas y tests a data/reports. Devuelve rutas escritas."""
    ensure_dirs(reports_dir)
    p_desc = reports_dir / "estadisticos_descriptivos_coherencia.csv"
    p_corr = reports_dir / "correlaciones_coherencia.csv"
    p_tests = reports_dir / "tests_coherencia.json"
    desc.to_csv(p_desc, index=True)
    corr.to_csv(p_corr, index=True)
    with p_tests.open("w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, ensure_ascii=False)
    log.info("📄 Reportes guardados:\n  • %s\n  • %s\n  • %s", p_desc, p_corr, p_tests)
    return {"descriptivos": p_desc, "correlaciones": p_corr, "tests": p_tests}

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Análisis de coherencia entre históricos 2022–2024.")
    p.add_argument("--y2022", type=str, default=str(_find_input_for_year(CLEAN_DIR, 2022)),
                   help="Ruta al histórico 2022 (csv/xlsx/parquet).")
    p.add_argument("--y2023", type=str, default=str(_find_input_for_year(CLEAN_DIR, 2023)),
                   help="Ruta al histórico 2023 (csv/xlsx/parquet).")
    p.add_argument("--y2024", type=str, default=str(_find_input_for_year(CLEAN_DIR, 2024)),
                   help="Ruta al histórico 2024 (csv/xlsx/parquet).")
    p.add_argument("--cohort", action="store_true",
                   help="Si se indica, restringe a la cohorte común (productos presentes en 2022–2024).")
    p.add_argument("--log-scale", action="store_true",
                   help="Usar escala log en boxplot.")
    p.add_argument("--figs-dir", type=str, default=str(FIGURES_DIR),
                   help="Directorio de salida para figuras.")
    p.add_argument("--reports-dir", type=str, default=str(REPORTS_DIR),
                   help="Directorio de salida para reportes.")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    p2022, p2023, p2024 = Path(args.y2022), Path(args.y2023), Path(args.y2024)
    figs_dir = Path(args.figs_dir)
    reports_dir = Path(args.reports_dir)

    try:
        log.info("🚀 Iniciando análisis de coherencia 2022–2024")
        log.info("📍 ROOT_DIR: %s", ROOT_DIR)

        df = cargar_historicos(p2022, p2023, p2024)
        df = aplicar_cohorte_comun(df, usar_cohorte=args.cohort)

        # Bloque 1: visual
        plot_evolucion_mensual(df, figs_dir)
        plot_box_media_por_producto(df, figs_dir, log_scale=args.log_scale)

        # Bloque 2: estadísticos
        desc = estadisticos_descriptivos(df)
        corr = correlaciones_interanuales(df)

        # Bloque 3: contrastes
        tests = tests_coherencia(df)

        # Exportar reportes
        exportar_reportes(desc, corr, tests, reports_dir)

        log.info("✅ Análisis de coherencia completado con éxito.")

    except Exception as e:
        log.exception("💥 Error en la ejecución: %s", e)
        raise

if __name__ == "__main__":
    main()
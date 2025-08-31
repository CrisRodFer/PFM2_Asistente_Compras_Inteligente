# ==============================================================
# Script: unir_subset_y_outliers.py
# Objetivo: Fusionar el subset representativo (30%) con las filas
#           atípicas (outliers) para obtener el subset final.
#
# Entradas:
#   - data/processed/demanda_subset.csv
#   - data/processed/outliers.csv
#
# Salidas:
#   - data/processed/demanda_subset_final.parquet
#   - reports/unir_subset_y_outliers.txt
#
# Notas:
#   - Se normalizan tipos (Product_ID como string y Date como datetime).
#   - Si una misma clave (Product_ID, Date) aparece en ambos ficheros, se
#     conserva la fila con `is_outlier=1`.
# ==============================================================

from pathlib import Path
import pandas as pd
import sys


def main() -> None:
    # --- Rutas base (ajusta si tu estructura difiere) ------------------------
    ROOT_DIR = Path(__file__).resolve().parents[2]     # .../PFM2_Asistente_Compras_Inteligente/
    DATA_DIR = ROOT_DIR / "data"
    PROC_DIR = DATA_DIR / "processed"
    REPO_DIR = ROOT_DIR / "reports"

    # --- Entradas / salidas --------------------------------------------------
    SUBSET30    = PROC_DIR / "demanda_subset.parquet"     # Paso 3 (30% sin outliers)
    OUTLIERS    = PROC_DIR / "outliers.parquet"            # Paso 2b (solo outliers)
    FINAL_PARQ  = PROC_DIR / "demanda_subset_final.parquet"
    OUT_REPORT  = REPO_DIR / "unir_subset_y_outliers.txt"

    # Comprobaciones de existencia
    for p in (SUBSET30, OUTLIERS):
        if not p.exists():
            print(f"[ERROR] No encuentro: {p}")
            sys.exit(1)

    REPO_DIR.mkdir(parents=True, exist_ok=True)

    # --- Carga ---------------------------------------------------------------
    df_sub = pd.read_parquet(SUBSET30)   # contiene Cluster; no tiene is_outlier (o todo 0)
    df_out = pd.read_parquet(OUTLIERS)   # contiene is_outlier=1 y mismas claves

    # --- Normalización de tipos ---------------------------------------------
    df_sub["Product_ID"] = df_sub["Product_ID"].astype(str)
    df_out["Product_ID"] = df_out["Product_ID"].astype(str)

    df_sub["Date"] = pd.to_datetime(df_sub["Date"], errors="coerce")
    df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")

    if "is_outlier" not in df_sub.columns:
        df_sub["is_outlier"] = 0
    else:
        df_sub["is_outlier"] = (
            pd.to_numeric(df_sub["is_outlier"], errors="coerce")
              .fillna(0)
              .astype("int8")
        )

    # Por definición, las filas del fichero de outliers llevan 1
    df_out["is_outlier"] = 1

    # --- Unión y resolución de duplicados por clave -------------------------
    # Si hay filas duplicadas para la misma clave, nos quedamos con la que
    # tenga is_outlier=1 (ordenamos y nos quedamos con la última).
    df_final = pd.concat([df_sub, df_out], ignore_index=True)
    df_final.sort_values(["Product_ID", "Date", "is_outlier"], inplace=True)
    df_final = df_final.drop_duplicates(["Product_ID", "Date"], keep="last")

    # --- Guardado ------------------------------------------------------------
    df_final.to_parquet(FINAL_PARQ, index=False)

    # --- Métrica rápida y reporte -------------------------------------------
    n_out_rows   = int(df_final["is_outlier"].eq(1).sum()) if "is_outlier" in df_final.columns else 0
    n_out_prods  = int(df_final.loc[df_final["is_outlier"].eq(1), "Product_ID"].nunique())
    n_rows       = len(df_final)
    n_products   = int(df_final["Product_ID"].nunique())
    min_date     = df_final["Date"].min()
    max_date     = df_final["Date"].max()
    by_cluster   = (
        df_final.groupby("Cluster")["Product_ID"].nunique().to_dict()
        if "Cluster" in df_final.columns else {}
    )

    # Consola
    print("[OK] demanda_subset_final.parquet guardado")
    print(f"Filas totales            : {n_rows:,}")
    print(f"Productos únicos         : {n_products:,}")
    print(f"Rango temporal           : {min_date} -> {max_date}")
    print(f"Outliers en final (filas): {n_out_rows:,}")
    print(f"Outliers en final (prods): {n_out_prods:,}")
    if by_cluster:
        print(f"Productos únicos por cluster: {by_cluster}")

    # Reporte
    lines = []
    lines.append("=== UNION SUBSET 30% + OUTLIERS ===")
    lines.append(f"IN  subset : {SUBSET30}")
    lines.append(f"IN  outliers: {OUTLIERS}")
    lines.append(f"OUT final : {FINAL_PARQ}")
    lines.append("")
    lines.append(f"Filas totales          : {n_rows:,}")
    lines.append(f"Productos únicos       : {n_products:,}")
    lines.append(f"Rango temporal         : {min_date} -> {max_date}")
    lines.append(f"Outliers (filas)       : {n_out_rows:,}")
    lines.append(f"Outliers (productos)   : {n_out_prods:,}")
    if by_cluster:
        lines.append(f"Productos únicos por cluster: {by_cluster}")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Reporte guardado en: {OUT_REPORT}")


if __name__ == "__main__":
    main()
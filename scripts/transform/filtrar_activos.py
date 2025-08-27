
# =============================================================================
# Script: filtrar_activos.py
# FASE 3 · 3.2.B — Filtrar Estado_Producto == "Activo" (tras enriquecer)
#
# Flujo:
#   1) Cargar data/processed/demanda_enriquecida.csv
#   2) Filtrar Estado_Producto == "Activo" (case/espacios tolerantes)
#   3) Dejar columnas: Product_ID, Date, Demand_Day, Categoria, Estado_Producto
#   4) Exportar demanda_con_catalogo.csv
#
# Inputs:
#   data/processed/demanda_enriquecida.csv
# Outputs: s
#   data/processed/demanda_con_catalogo.csv
#   reports/validation/resumen_filtrado_activos.csv
# =============================================================================

from pathlib import Path
import argparse, sys
import pandas as pd

# --- Rutas base ---
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    here = Path.cwd()
    ROOT_DIR = here if (here / "data" / "processed").exists() else here.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"

ENRIQ_IN   = PROCESSED_DIR / "demanda_enriquecida.csv"
SALIDA_OUT = PROCESSED_DIR / "demanda_con_catalogo.csv"
RESUMEN_OUT= VALIDATION_DIR / "resumen_filtrado_activos.csv"

REQ_COLS = {"Product_ID","Date","Demand_Day","Categoria","Estado_Producto"}

def cargar_enriquecida(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe demanda_enriquecida: {path}")
    df = pd.read_csv(path)
    faltan = REQ_COLS - set(df.columns)
    if faltan:
        raise ValueError(f"demanda_enriquecida sin columnas esperadas: {faltan}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def exportar(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Filtrar Estado_Producto == 'Activo' en demanda_enriquecida.")
    p.add_argument("--in", dest="inp", default=str(ENRIQ_IN))
    p.add_argument("--out", default=str(SALIDA_OUT))
    p.add_argument("--val-out", default=str(RESUMEN_OUT))
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    df = cargar_enriquecida(Path(args.inp))

    mask = df["Estado_Producto"].astype(str).str.strip().str.lower().eq("activo")
    df_act = df.loc[mask, ["Product_ID","Date","Demand_Day","Categoria","Estado_Producto"]].copy()

    # validaciones
    dups = int(df_act.duplicated(subset=["Product_ID","Date"]).sum())
    in_min, in_max = df["Date"].min(), df["Date"].max()
    out_min, out_max = (df_act["Date"].min(), df_act["Date"].max()) if not df_act.empty else (pd.NaT, pd.NaT)
    rango_ok = bool((in_min == out_min) and (in_max == out_max)) if not df_act.empty else False

    resumen = pd.DataFrame({
        "registros_enriquecidos":[len(df)],
        "registros_salida_activos":[len(df_act)],
        "productos_enriquecidos":[df["Product_ID"].nunique()],
        "productos_salida_activos":[df_act["Product_ID"].nunique()],
        "duplicados_en_salida":[dups],
        "fecha_min_entrada":[in_min],
        "fecha_max_entrada":[in_max],
        "fecha_min_salida":[out_min],
        "fecha_max_salida":[out_max],
        "rango_fechas_igual":[rango_ok],
    })

    out_path = exportar(df_act, Path(args.out))
    val_path = exportar(resumen, Path(args.val_out))

    print(f"[OK] Activos -> {out_path} ({len(df_act)} filas, {df_act['Product_ID'].nunique()} productos)")
    print(f"[OK] Resumen -> {val_path}")
    print(resumen.to_string(index=False))
    return df_act, resumen

if __name__ == "__main__":
    if any("ipykernel" in arg for arg in sys.argv):
        main([])
    else:
        main()

        

# ============================================================================
# Script: normalizar_features_productos.py
# Descripción:
#   Estandariza las variables numéricas de productos_features_clean.csv para
#   que todas tengan media 0 y desviación 1 (StandardScaler), dejando
#   Product_ID sin transformar. Genera productos_features_norm.csv.
#
# Entradas (por defecto):
#   - data/processed/productos_features_clean.csv
#
# Salidas:
#   - data/processed/productos_features_norm.csv
#   - models/scalers/standard_scaler_productos.pkl  (scaler serializado)
#
# Uso:
#   python scripts/transform/normalizar_features_productos.py
#   python scripts/transform/normalizar_features_productos.py \
#       --in  data/processed/productos_features_clean.csv \
#       --out data/processed/productos_features_norm.csv
# ============================================================================

from pathlib import Path
import sys, argparse, logging, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --------------------------- Raíz del proyecto ------------------------------
def _detect_root_when_no_file():
    here = Path().resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir():
            return p
    return here

if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[2]
else:
    ROOT_DIR = _detect_root_when_no_file()

DATA_DIR      = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = ROOT_DIR / "models" / "scalers"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------- Logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("normalizar_features")

# --- Parche Jupyter: elimina --f=... del kernel para argparse ----------------
if "ipykernel" in sys.modules or "IPython" in sys.modules:
    sys.argv = [sys.argv[0]]
# -----------------------------------------------------------------------------

NUM_COLS_DEFAULT = [
    "d_total", "d_media", "d_std", "cv", "p95", "mediana",
    "precio_medio", "PC1", "PC2", "PC3"
]

def normalizar_features(in_path: Path, out_path: Path, cols: list[str] | None = None, save_scaler: bool = True):
    in_path  = Path(in_path)
    out_path = Path(out_path)
    cols = cols or NUM_COLS_DEFAULT

    # 1) Cargar
    logger.info(f"Cargando features limpias: {in_path}")
    df = pd.read_csv(in_path)

    # 2) Validaciones básicas
    if "Product_ID" not in df.columns:
        raise KeyError("Se requiere columna 'Product_ID' en el dataset de entrada.")

    faltantes = [c for c in cols if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas esperadas para normalizar: {faltantes}")

    # Comprobar NaNs previos
    nan_prev = df[cols].isna().sum().to_dict()
    if any(v > 0 for v in nan_prev.values()):
        logger.warning("Se han detectado NaNs previos en columnas a escalar: %s", nan_prev)

    n_productos = df["Product_ID"].nunique()
    logger.info("Productos únicos (entrada): %d", n_productos)

    # 3) Ajustar StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = df[cols].astype(float).values
    X_scaled = scaler.fit_transform(X)

    # 4) Construir dataframe normalizado
    df_scaled = df.copy()
    df_scaled[cols] = X_scaled

    # 5) Validaciones post-escalado
    means = df_scaled[cols].mean().to_dict()
    stds  = df_scaled[cols].std(ddof=0).to_dict()  # ddof=0 para comparar con sklearn
    approx_mean_ok = all(abs(m) < 1e-6 for m in means.values())
    approx_std_ok  = all(abs(s - 1.0) < 1e-6 for s in stds.values())

    logger.info("=== VALIDACIONES POST-ESCALADO ===")
    logger.info("Media aprox ≈ 0 por columna: %s", "OK" if approx_mean_ok else "REVISAR")
    logger.info("Std   aprox ≈ 1 por columna: %s", "OK" if approx_std_ok else "REVISAR")
    logger.info("Means: %s", {k: round(v, 6) for k, v in means.items()})
    logger.info("Stds : %s", {k: round(v, 6) for k, v in stds.items()})

    # NaNs post-escalado
    nan_post = df_scaled[cols].isna().sum().to_dict()
    logger.info("NaNs tras escalado: %s", nan_post)

    # 6) Guardar dataset normalizado
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(out_path, index=False)
    logger.info(f"Guardado: {out_path} (filas={len(df_scaled)}, cols={df_scaled.shape[1]})")

    # 7) Guardar scaler (opcional)
    if save_scaler:
        scaler_path = MODELS_DIR / "standard_scaler_productos.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump({"scaler": scaler, "cols": cols}, f)
        logger.info(f"Scaler guardado en: {scaler_path}")

    return df_scaled, scaler

# ------------------------------------ CLI -----------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Normaliza features de productos con StandardScaler.")
    p.add_argument("--in",  dest="inp",  type=str, default=str(PROCESSED_DIR / "productos_features_clean.csv"))
    p.add_argument("--out", dest="outp", type=str, default=str(PROCESSED_DIR / "productos_features_norm.csv"))
    p.add_argument("--no-save-scaler", action="store_true", help="No guardar el scaler serializado.")
    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []
    args, _ = p.parse_known_args(argv)
    logger.info("ARGS -> in=%s | out=%s | save_scaler=%s", args.inp, args.outp, not args.no_save_scaler)
    return args

def main():
    args = parse_args()
    try:
        normalizar_features(Path(args.inp), Path(args.outp), NUM_COLS_DEFAULT, save_scaler=not args.no_save_scaler)
        logger.info("Normalización finalizada correctamente.")
    except Exception as e:
        logging.exception(f"Error en normalización: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

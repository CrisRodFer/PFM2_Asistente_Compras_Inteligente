# =============================================================================
# Script: generar_historico_2024.py
# Descripción:
# Genera el histórico diario de 2024 a partir de la previsión limpia, exporta
# CSV y Parquet en data/clean y reporte de huecos en data/reports.
#
# Flujo del pipeline:
# 1) Cargar previsión limpia
# 2) Integrar en calendario 2024 (365/366) y validar
# 3) Exportar histórico + reporte de huecos + métricas por pantalla
#
# Input:
#   - data/clean/Prevision_Demanda_2025_Limpia.xlsx (por defecto)
#
# Output:
#   - data/clean/Historico_Ventas_2024.csv
#   - data/clean/Historico_Ventas_2024.parquet
#   - data/reports/reporte_huecos_historico_2024.csv
#
# Dependencias:
#   - pandas, pyarrow, openpyxl
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS + PATH DINÁMICO) ======================================
from pathlib import Path
import sys, logging, importlib.util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,  # garantiza que se vea en consola
)
log = logging.getLogger("generar_historico_2024")

# 1) Punto de partida = carpeta de este script
HERE = Path(__file__).resolve()
log.info("Ejecutando script: %s", HERE)

# 2) Subir niveles para estimar la raíz (donde suele estar /data)
ROOT_DIR = HERE
for _ in range(6):
    if (ROOT_DIR / "data").exists() or (ROOT_DIR / ".git").exists():
        break
    ROOT_DIR = ROOT_DIR.parent
log.info("ROOT_DIR estimado: %s", ROOT_DIR)

# 3) Buscar el archivo generar_historicos.py en todo el proyecto (subcarpetas incluidas)
try:
    MOD_PATH = next(ROOT_DIR.rglob("generar_historicos.py"))
except StopIteration:
    MOD_PATH = None

log.info("Ruta encontrada de generar_historicos.py: %s", MOD_PATH)

# 4) Asegurar import: si lo encuentro, lo cargo por ruta.
if MOD_PATH is None:
    raise FileNotFoundError(
        "No se ha encontrado 'generar_historicos.py' en el proyecto.\n"
        f"Raíz buscada: {ROOT_DIR}\n"
        "Comprueba que el archivo existe y su nombre exacto."
    )

if str(MOD_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MOD_PATH.parent))

spec = importlib.util.spec_from_file_location("generar_historicos", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader, "No se pudo preparar el import de generar_historicos.py"
spec.loader.exec_module(mod)  # type: ignore

# Exponemos las funciones que necesitamos
generar_historico_df = mod.generar_historico_df
exportar_historico   = mod.exportar_historico

# 5) Directorios de datos tomados respecto a ROOT_DIR
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

# ==== 1. IMPORTS DEL MÓDULO BASE =============================================
# Intento normal
try:
    from generar_historicos import generar_historico_df, exportar_historico
except ModuleNotFoundError as e:
    # Fallback absoluto por ruta (por si el path aún se resiste)
    log.warning("Fallo import estándar (%s). Probando import absoluto por ruta…", e)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generar_historicos",
        ROOT_DIR / "generar_historicos.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "No se pudo preparar el import absoluto de generar_historicos.py"
    spec.loader.exec_module(mod)  # type: ignore
    generar_historico_df = mod.generar_historico_df
    exportar_historico = mod.exportar_historico
    log.info("Import absoluto cargado correctamente.")

# ==== 2. RESTO DE IMPORTS =====================================================
import argparse
import pandas as pd

# ==== 3. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

# ==== 4. LÓGICA PRINCIPAL =====================================================
def run(
    input_path: Path,
    out_dir: Path,
    *,
    year: int = 2024,
    date_col: str = "Date",
    id_col: str = "Product_ID",
    qty_col: str = "Sales Quantity",  # cambia a "Demand" si procede
    skip_checks: bool = False,
    csv_compressed: bool = False,
) -> dict[str, Path]:

    log.info("Leyendo previsión: %s", input_path)
    historico_df, gaps_df, metrics = generar_historico_df(
        input_path=input_path,
        year=year,
        date_col=date_col,
        id_col=id_col,
        qty_col=qty_col,
        skip_checks=skip_checks,
    )

    ensure_dirs(out_dir, REPORTS_DIR)
    paths = exportar_historico(
        historico=historico_df,
        gaps=gaps_df,
        out_dir=out_dir,
        year=year,
        date_col=date_col,
        id_col=id_col,
        qty_col=qty_col,
        csv_compressed=csv_compressed,
    )

    log.info("Métricas %s: %s", year, metrics)
    log.info("CSV: %s", paths["csv"])
    log.info("Parquet: %s", paths["parquet"])
    log.info("Reporte huecos: %s", paths["gaps"])

    try:
        log.info("Preview histórico (head):\n%s", historico_df.head(3).to_string(index=False))
        log.info("Preview huecos (head):\n%s", gaps_df.head(3).to_string(index=False))
    except Exception:
        pass

    return paths

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generar histórico diario de 2024.")
    p.add_argument("--input", type=str, default=str(CLEAN_DIR / "Prevision_Demanda_2025_Limpia.xlsx"),
                   help="Ruta a la previsión limpia (xlsx).")
    p.add_argument("--outdir", type=str, default=str(CLEAN_DIR),
                   help="Directorio de salida para el histórico.")
    p.add_argument("--date-col", type=str, default="Date", help="Nombre de la columna de fecha.")
    p.add_argument("--id-col", type=str, default="Product_ID", help="Nombre de la columna de ID.")
    p.add_argument("--qty-col", type=str, default="Sales Quantity",
                   help="Columna de cantidad (p.ej. 'Demand' o 'Sales Quantity').")
    p.add_argument("--skip-checks", action="store_true", help="Omitir checks de calendario.")
    p.add_argument("--csv-compressed", action="store_true", help="Escribir CSV.gz en lugar de CSV.")
    return p.parse_args()

def main() -> None:
    args = _parse_args()                   # ← ya lo tienes
    print(">>> Iniciando generar_historico_2024.py")          # señal visible en consola
    log.info(">>> Iniciando generar_historico_2024.py")       # (opcional) mismo mensaje por logging
    log.info("Args: input=%s outdir=%s qty_col=%s",
             args.input, args.outdir, args.qty_col)           # (opcional) ver parámetros

    run(
        input_path=Path(args.input),
        out_dir=Path(args.outdir),
        year=2024,
        date_col=args.date_col,
        id_col=args.id_col,
        qty_col=args.qty_col,
        skip_checks=args.skip_checks,
        csv_compressed=args.csv_compressed,
    )

if __name__ == "__main__":
    main()
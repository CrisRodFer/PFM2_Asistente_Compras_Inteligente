# =============================================================================
# Script: generar_historico_2022.py
# Descripción:
# Genera el histórico diario de 2022 a partir del archivo de entrada
# (por defecto: data/raw/Historico_Ventas_2023_Corregido.xlsx),
# exporta CSV/Parquet en data/clean y el reporte de huecos en data/reports.
# Versión robusta: autodescubre 'generar_historicos.py' y rutas del proyecto.
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS + IMPORT DINÁMICO) ====================================
from pathlib import Path
import sys, logging, importlib.util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,   # asegura salida en la consola de VSCode
    force=True,
)
log = logging.getLogger("generar_historico_2022")

HERE = Path(__file__).resolve()
log.info("Ejecutando script: %s", HERE)

# Estimar raíz del proyecto subiendo niveles hasta ver 'data' o '.git'
ROOT_DIR = HERE
for _ in range(6):
    if (ROOT_DIR / "data").exists() or (ROOT_DIR / ".git").exists() or (ROOT_DIR / "generar_historicos.py").exists():
        break
    ROOT_DIR = ROOT_DIR.parent
log.info("ROOT_DIR estimado: %s", ROOT_DIR)

# Buscar el módulo base en todo el proyecto
try:
    MOD_PATH = next(ROOT_DIR.rglob("generar_historicos.py"))
except StopIteration:
    raise FileNotFoundError(f"No se ha encontrado 'generar_historicos.py' bajo {ROOT_DIR}")

log.info("Ruta encontrada de generar_historicos.py: %s", MOD_PATH)

# Cargar por ruta hallada (independiente del sys.path)
spec = importlib.util.spec_from_file_location("generar_historicos", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader, "No se pudo preparar el import de generar_historicos.py"
spec.loader.exec_module(mod)  # type: ignore

# Exponer funciones del módulo base
generar_historico_df = mod.generar_historico_df
exportar_historico   = mod.exportar_historico

# Directorios de datos relativos a la raíz
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"       # ← usamos RAW como origen por defecto (según tu indicación)
CLEAN_DIR  = DATA_DIR / "clean"
REPORTS_DIR= DATA_DIR / "reports"

# ==== 1. IMPORTS RESTO =======================================================
import argparse
import pandas as pd

# ==== 2. UTILIDADES ==========================================================
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# ==== 3. LÓGICA PRINCIPAL ====================================================
def run(
    input_path: Path,
    out_dir: Path,
    *,
    year: int = 2022,
    date_col: str = "Date",
    id_col: str = "Product_ID",
    qty_col: str = "Sales Quantity",   
    skip_checks: bool = False,
    csv_compressed: bool = False,
) -> dict[str, Path]:

    log.info("Leyendo archivo de entrada: %s", input_path)
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

    # Previews útiles en consola
    try:
        log.info("Preview histórico (head):\n%s", historico_df.head(3).to_string(index=False))
        log.info("Preview huecos (head):\n%s", gaps_df.head(3).to_string(index=False))
    except Exception:
        pass

    return paths

# ==== 4. CLI / MAIN ==========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generar histórico diario de 2022.")
    p.add_argument(
        "--input",
        type=str,
        # Usamos como default el fichero que me adjuntaste y que está en RAW
        default=str(RAW_DIR / "Historico_Ventas_2023_Corregido.xlsx"),
        help="Ruta al archivo de entrada (xlsx/csv) para construir el histórico 2022.",
    )
    p.add_argument("--outdir", type=str, default=str(CLEAN_DIR),
                   help="Directorio de salida para el histórico.")
    p.add_argument("--date-col", type=str, default="Date", help="Nombre de la columna de fecha.")
    p.add_argument("--id-col", type=str, default="Product_ID", help="Nombre de la columna de ID.")
    p.add_argument(
        "--qty-col",
        type=str,
        default="Sales Quantity",   # <-- ajusta si tu columna es 'Demand'
        help="Columna de cantidad (p.ej. 'Sales Quantity' o 'Demand').",
    )
    p.add_argument("--skip-checks", action="store_true", help="Omitir checks de calendario.")
    p.add_argument("--csv-compressed", action="store_true", help="Escribir CSV.gz en lugar de CSV.")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    print(">>> Iniciando generar_historico_2022.py")
    # Comprobación preventiva de la ruta de entrada
    ip = Path(args.input)
    if not ip.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ip}")

    log.info("Args: input=%s outdir=%s qty_col=%s", args.input, args.outdir, args.qty_col)
    run(
        input_path=ip,
        out_dir=Path(args.outdir),
        year=2022,
        date_col=args.date_col,
        id_col=args.id_col,
        qty_col=args.qty_col,
        skip_checks=args.skip_checks,
        csv_compressed=args.csv_compressed,
    )

if __name__ == "__main__":
    main()

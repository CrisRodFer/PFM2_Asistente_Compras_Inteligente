from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Raíz del proyecto (carpeta que contiene src/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Estructura (según tu data/)
DATA_DIR   = BASE_DIR / "data"
DATA_RAW   = DATA_DIR / "raw"
DATA_CLEAN = DATA_DIR / "clean"
PROCESSED  = DATA_DIR / "processed"   # <- processed dentro de data
OUTPUTS    = BASE_DIR / "outputs"
FIGURES    = OUTPUTS / "figures"
REPORTS    = OUTPUTS / "reports"
LOGS       = BASE_DIR / "logs"

# Crea carpetas si no existen (idempotente)
for p in (DATA_RAW, DATA_CLEAN, PROCESSED, OUTPUTS, FIGURES, REPORTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

# Columnas clave (ajusta si cambian nombres)
COL_ID    = "Product_ID"
COL_DATE  = "Date"
COL_QTY   = "Sales Quantity"
COL_TREND = "Demand Trend"

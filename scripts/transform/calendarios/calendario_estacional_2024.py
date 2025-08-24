

# scripts/transform/calendarios/calendario_estacional_2024.py
# =============================================================================

# 0) Bootstrapping de rutas 
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]  # ← sube a la RAÍZ del proyecto
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 1) Imports necesarios
from scripts.utils.generar_calendario_estacional import generar_calendario_estacional

# 2) Config
YEAR = 2024
SALIDA = ROOT / "outputs" / f"calendario_estacional_{YEAR}.csv"

# 3) Lógica principal
def run():
    df = generar_calendario_estacional(anio=YEAR, leap_strategy="explicit")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"✅ Guardado: {SALIDA}")

# 4) Entry point
if __name__ == "__main__":
    run()
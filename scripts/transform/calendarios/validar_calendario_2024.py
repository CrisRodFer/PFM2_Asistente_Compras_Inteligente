
# scripts/transform/calendarios/validar_calendario_2024.py
# =============================================================================

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.validar_calendario_estacional import validar_calendario_estacional

YEAR = 2024
CAL = ROOT / "outputs" / f"calendario_estacional_{YEAR}.csv"

def main():
    print(f"🔎 Validando calendario {YEAR}: {CAL}")
    res = validar_calendario_estacional(str(CAL), year=YEAR, verbose=True)
    if not res["ok"]:
        print("❌ Validación fallida.")
        raise SystemExit(1)
    print("✅ Validación OK.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
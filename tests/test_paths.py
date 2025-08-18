import sys
from pathlib import Path

# Añadir raíz del proyecto al sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from src.config import DATA_RAW, DATA_CLEAN

# Crear un DataFrame de prueba
df = pd.DataFrame({
    "Product_ID": [1000, 1001, 1002],
    "Sales Quantity": [10, 20, 30]
})

# Guardar en DATA_RAW
raw_file = DATA_RAW / "test_data.csv"
df.to_csv(raw_file, index=False)
print(f"Archivo guardado en: {raw_file}")

# Leer desde DATA_RAW
df_loaded = pd.read_csv(raw_file)
print("Contenido leído:")
print(df_loaded)

# Guardar en DATA_CLEAN
clean_file = DATA_CLEAN / "test_data_clean.csv"
df_loaded.to_csv(clean_file, index=False)
print(f"Archivo procesado guardado en: {clean_file}")

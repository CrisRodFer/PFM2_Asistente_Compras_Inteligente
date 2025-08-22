# =============================================================================
# Script: limpiar_catalogo_productos.py
# Descripción:
# Limpia y estructura el catálogo de productos para su uso posterior
# (joins con históricos, previsiones y desagregación). Se eliminan
# columnas no usadas, se reubica Product_ID al frente y se corrigen
# errores de codificación (mojibake UTF‑8) en Nombre y Categoria.
#
# Flujo del pipeline:
# 1) Cargar catálogo bruto (Excel)
# 2) Eliminar columnas innecesarias
# 3) Reubicar Product_ID al inicio
# 4) Normalizar texto en Nombre y Categoria (mojibake, espacios, regex)
# 5) Exportar catálogo limpio (Excel + Parquet)
#
# Input:
#   - data/raw/Catalogo_Productos_Con_Estado.xlsx
#
# Output:
#   - data/clean/Catalogo_Productos_Limpio.xlsx
#
# Dependencias:
#   - pandas
#   - openpyxl
#   - pyarrow (opcional, para Parquet)
#
# Instalación rápida:
#   pip install pandas openpyxl pyarrow
# =============================================================================

from __future__ import annotations

# ==== 0. CONFIG (RUTAS BASE) ==================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]   # <repo>/
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

# Salvaguarda si el script estuviera un nivel más profundo:
if not DATA_DIR.exists():
    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    CLEAN_DIR = DATA_DIR / "clean"
    INTERIM_DIR = DATA_DIR / "interim"
    PROCESSED_DIR = DATA_DIR / "processed"
    REPORTS_DIR = DATA_DIR / "reports"

# ==== 1. IMPORTS + LOGGING ====================================================
import argparse
import logging
import re
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(Path(__file__).stem)

# ==== 2. UTILIDADES ===========================================================
def ensure_dirs(*dirs: Path) -> None:
    """Crea directorios si no existen."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def _preserve_na_str(x):
    """Convierte a str SIN transformar NaN en la cadena 'nan'."""
    return x if pd.isna(x) else str(x)

# Mapa de correcciones típico de mojibake UTF‑8 (Ã¡, Ã©, Ã±, Â…)
UTF8_MOJIBAKE_MAP: dict[str, str] = {
    "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
    "ÃÁ": "Á", "Ã‰": "É", "Ã\x8d": "Í", "Ã“": "Ó", "Ãš": "Ú",
    "Ã±": "ñ", "Ã‘": "Ñ",
    "Â": "",   # suele colarse antes de símbolos/espacios
    "Ã ": "à", "Ã¨": "è", "Ã¬": "ì", "Ã²": "ò", "Ã¹": "ù",
}

def fix_utf8_mojibake_pairs(text):
    """Corrige secuencias mojibake UTF‑8 frecuentes (p. ej. 'Ã¡' → 'á').

    Parameters
    ----------
    text : Any
        Valor textual (o NaN).

    Returns
    -------
    Any
        Texto corregido conservando NaN cuando corresponda.
    """
    if pd.isna(text):
        return text
    s = str(text)
    if "Ã" in s or "Â" in s:
        for bad, good in UTF8_MOJIBAKE_MAP.items():
            s = s.replace(bad, good)
    return s

def _collapse_spaces(text):
    """Colapsa espacios múltiples y recorta extremos (conserva NaN)."""
    if pd.isna(text):
        return text
    s = str(text)
    return " ".join(s.split()).strip()

# ==== 3. LÓGICA PRINCIPAL =====================================================
DROP_COLS_DEFAULT = ["Fecha Alta", "Referencia", "Ventas 30 dias", "Ventas 60 dias", "Ventas 90 dias"]
TEXT_COLS_DEFAULT = ["Nombre", "Categoria"]

def cargar_catalogo(path: Path) -> pd.DataFrame:
    """Lee el catálogo original desde Excel."""
    return pd.read_excel(path)

def eliminar_columnas(df: pd.DataFrame, columnas_a_eliminar: list[str]) -> pd.DataFrame:
    """Elimina del DataFrame las columnas indicadas si existen."""
    cols = [c for c in columnas_a_eliminar if c in df.columns]
    return df.drop(columns=cols, errors="ignore")

def mover_columna_al_inicio(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Reordena columnas colocando `columna` en primera posición (si existe)."""
    if columna in df.columns:
        return df[[columna] + [c for c in df.columns if c != columna]]
    return df

def limpiar_caracteres_columnas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Limpia texto en columnas: preserva NaN → corrige mojibake → colapsa espacios."""
    for col in columnas:
        if col not in df.columns:
            continue
        s = df[col]
        s = s.map(_preserve_na_str)
        s = s.map(fix_utf8_mojibake_pairs)
        s = s.map(_collapse_spaces)
        df[col] = s
    return df

def corregir_codificacion_catalogo(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Aplica reemplazos (regex) específicos para afinar casos conocidos."""
    reemplazos = {
        # Errores modernos corregidos tras arreglar pares UTF8
        r"HidrataciÃ³n": "Hidratación",
        r"Geles EnergÃ©ticos": "Geles Energéticos",
        r"ProteÃ­nas": "Proteínas",
        r"CÃ¡psula?s?": "Cápsula",
        r"EnergÃ­a": "Energía",
        r"BaÃ±o": "Baño",
        # Antiguos (por si persisten)
        r"A3n": "ón", r"A1": "á", r"A9": "é", r"A\(c\)": "é", r"Aa": "á",
        r"CA!psulas": "Cápsulas",
        r"Energá": "Energía",
        r"Geles EnergA\(c\)ticos": "Geles Energéticos",
        r"CosmA\(c\)tica": "Cosmética",
        r"TA\(c\), Infusiones y Rooibos": "Té, Infusiones y Rooibos",
        r"CafA\(c\) y expreso": "Café y expreso",
        r"BaA\+\-o e higiene personal": "Baño e higiene personal",
        r"éeites Esenciales": "Aceites Esenciales",
        r"éeites, vinagres y aliA\+\-os": "Aceites, vinagres y aliños",
        r"Perfumerá": "Perfumería",
        r"ProteAnas": "Proteínas",
    }
    for col in columnas:
        if col not in df.columns:
            continue
        s = df[col].astype("string")
        for pattern, repl in reemplazos.items():
            s = s.str.replace(pattern, repl, regex=True)
        df[col] = s
    return df

# ==== 4. EXPORTACIÓN / I/O OPCIONAL ==========================================
def exportar_resultados(df: pd.DataFrame, out_xlsx: Path, out_parquet: Path | None = None) -> dict[str, Path]:
    """Exporta Excel y, opcionalmente, Parquet. Devuelve rutas escritas."""
    ensure_dirs(out_xlsx.parent)
    df.to_excel(out_xlsx, index=False)
    paths = {"xlsx": out_xlsx}
    if out_parquet is not None:
        df.to_parquet(out_parquet, index=False)
        paths["parquet"] = out_parquet
    return paths

# ==== 5. CLI / MAIN ===========================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Limpieza del catálogo de productos.")
    p.add_argument("--in",  dest="inp", type=str, default=str(RAW_DIR   / "Catalogo_Productos_Con_Estado.xlsx"),
                   help="Ruta de entrada (Excel).")
    p.add_argument("--out", dest="out", type=str, default=str(CLEAN_DIR / "Catalogo_Productos_Limpio.xlsx"),
                   help="Ruta de salida (Excel).")
    p.add_argument("--no-parquet", action="store_true", help="No escribir Parquet adicional.")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    inp  = Path(args.inp)
    outx = Path(args.out)
    outp = outx.with_suffix(".parquet") if not args.no_parquet else None

    try:
        log.info("📍 ROOT_DIR: %s", ROOT_DIR)
        log.info("📥 Leyendo catálogo: %s", inp)
        df = cargar_catalogo(inp)

        log.info("🧹 Eliminando columnas innecesarias…")
        df = eliminar_columnas(df, DROP_COLS_DEFAULT)

        log.info("🔁 Reubicando Product_ID al frente…")
        df = mover_columna_al_inicio(df, "Product_ID")

        log.info("✍️ Normalizando texto en Nombre/Categoria…")
        df = limpiar_caracteres_columnas(df, TEXT_COLS_DEFAULT)
        df = corregir_codificacion_catalogo(df, TEXT_COLS_DEFAULT)

        log.info("💾 Exportando resultados…")
        paths = exportar_resultados(df, outx, outp)
        for k, v in paths.items():
            icon = "📘" if k == "xlsx" else "🧱"
            log.info("%s Salida (%s): %s", icon, k, v)

        log.info("✅ Catálogo limpio generado correctamente.")

    except Exception as e:
        log.exception("💥 Fallo en la ejecución: %s", e)
        raise

if __name__ == "__main__":
    main()
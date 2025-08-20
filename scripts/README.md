# Carpeta `scripts`

Aquí se encuentran los **scripts de Python** que implementan los distintos pasos del pipeline de datos.

La organización sigue un enfoque modular:

- **`clean/`**  
  Scripts para la **limpieza de datos** (detección de nulos, estandarización de formatos, etc.).

- **`eda/`**  
  Scripts para el **análisis exploratorio de datos** (gráficas, estadísticas descriptivas, correlaciones).

- **`transform/`**  
  Scripts que realizan **transformaciones y generación de históricos**.  
  Ejemplo: `generar_historicos.py`.

- **`utils/`**  
  Funciones auxiliares y utilidades reutilizables (lectura de datos, validaciones, logs).

---

## Convenciones

- Los scripts deben contener **funciones reutilizables** y, cuando aplique, un bloque `if __name__ == "__main__":` para ejecuciones directas.  
- Se recomienda mantener una **estructura de imports limpia**, priorizando primero librerías estándar, luego externas, y al final módulos internos.  
- Los nombres de los scripts deben ser **descriptivos y en minúsculas**, usando `snake_case`.

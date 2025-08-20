# Carpeta `data`

Esta carpeta contiene los **datasets** utilizados en el proyecto. Se organiza en varios niveles según el estado de los datos:

- **`raw/`**  
  Datos en bruto tal y como se reciben de la fuente original.  
  ⚠️ Nunca deben modificarse, sirven como referencia histórica.

- **`clean/`**  
  Datos depurados tras aplicar procesos de limpieza inicial (formato, duplicados, valores inconsistentes, etc.).

- **`interim/`**  
  Datos intermedios generados durante el pipeline de transformación.  
  Son temporales y pueden ser sobrescritos.

- **`processed/`**  
  Datos finales listos para el modelado y el análisis.  
  Aquí se guardan outputs consistentes y validados.

- **`reports/`**  
  Archivos de reporte generados automáticamente (por ejemplo: reportes de huecos, validaciones, resúmenes estadísticos).

---

## Convenciones

- Todos los archivos deben tener nombres **descriptivos y en minúsculas**, usando guiones bajos (`snake_case`).  
- Los archivos **Parquet** (`.parquet`) se priorizan sobre Excel/CSV por eficiencia.  
- Los datos sensibles **no deben incluirse en este repositorio**.


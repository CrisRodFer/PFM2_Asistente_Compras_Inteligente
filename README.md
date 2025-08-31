# 📌 PFM2_Asistente_Compras_Inteligente

## 📖 Descripción

Trabajo de fin de máster en **Data Science & IA**.\
Este proyecto consiste en el desarrollo de un **Asistente Inteligente de
Compras**, cuyo objetivo es optimizar la gestión de inventario y stock
de una empresa mediante técnicas de análisis de datos, simulación de
demanda y modelado predictivo.

------------------------------------------------------------------------

## 📂 Estructura del proyecto

``` plaintext
PFM2_Asistente_Compras_Inteligente/
├── data/               # Datos en diferentes estados
│   ├── raw/            # Datos originales sin procesar
│   ├── interim/        # Datos intermedios (transformaciones temporales)
│   ├── clean/          # Datos limpios y validados
│   ├── processed/      # Datos preparados
│   └── reports/        # Reportes automáticos (ej. huecos en históricos)
├── outputs/            # Resultados y salidas
│   ├── figures/        # Gráficas
│   └── reports/        # Informes
├── logs/               # Registros de ejecución
├── notebooks/          # Notebooks del proyecto
├── scripts/            # Scripts ejecutables
├── sql/                # Consultas SQL
├── streamlit_app/      # Interfaz Streamlit
├── src/                # Código reusable
│   ├── config.py
│   ├── logging_conf.py
│   └── utils_data.py
├── requirements.txt    # Dependencias
├── .gitignore
└── .gitattributes
```

------------------------------------------------------------------------

## 🚀 Instalación

1.  Clonar el repositorio:

    ``` bash
    git clone https://github.com/<usuario>/PFM2_Asistente_Compras_Inteligente.git
    ```

2.  Crear y activar un entorno virtual:

    ``` bash
    python -m venv venv
    source venv/bin/activate  # En Linux/Mac
    venv\Scripts\activate     # En Windows
    ```

3.  Instalar dependencias:

    ``` bash
    pip install -r requirements.txt
    ```

------------------------------------------------------------------------

## 🎯 Objetivos del proyecto

-   Simulación de la demanda base diaria.
-   Optimización de compras según stock disponible.
-   Identificación de proveedores críticos.
-   Desarrollo de una interfaz en Streamlit para interactuar con el
    modelo.
-   Integración futura con SQL para gestión eficiente de catálogos y
    consultas.

------------------------------------------------------------------------

## 📊 Tecnologías utilizadas

-   **Python**
-   **Pandas / NumPy**
-   **Matplotlib / Seaborn**
-   **Scikit-learn**
-   **Streamlit**
-   **SQLite**
-   **Git / GitHub**

------------------------------------------------------------------------

👉 A partir de aquí, iremos completando con: 
- **Detalles de datasets** 
- **Proceso metodológico (EDA, simulaciones, modelado,validación)** 
- **Resultados** 
- **Líneas futuras**

------------------------------------------------------------------------

La carpeta data/ incluye un README específico donde se documenta en detalle la 
estructura de las subcarpetas de datos. Lo mismo ocurre con la carpeta scripts/.

------------------------------------------------------------------------

## 📑 Metodología – Fase 1 (EDA y preparación de datos)

Durante esta fase se llevaron a cabo las siguientes tareas:

1. **Limpieza y validación de la previsión de demanda 2025**  
   - Revisión de fechas (inclusión de 29/02 y 31/12).  
   - Eliminación de nulos y duplicados.  
   - Generación de reportes automáticos de huecos.  

2. **Generación de históricos simulados (2022–2024)**  
   - Creación de históricos diarios a partir de la previsión de 2025.  
   - Ajustes por coherencia temporal y progresión lógica entre años.  
   - Exportación en CSV/Parquet y reporte de validación.  

3. **Limpieza del catálogo de productos**  
   - Normalización de caracteres y eliminación de inconsistencias.  
   - Reordenación de columnas y exportación en formato limpio.  
   - Exportación final en Excel y Parquet para integración con la demanda.  

4. **Análisis de coherencia entre históricos (2022–2024)**  
   - **Visual**: evolución mensual y boxplots de medias por producto.  
   - **Estadístico**: descriptivos (media, std, CV) y correlaciones interanuales (>0.98).  
   - **Contrastes de hipótesis**: ANOVA y Kruskal-Wallis confirman ausencia de diferencias significativas entre años.  

📌 **Conclusión de la fase**:  
Los históricos generados presentan un comportamiento **estable, coherente y robusto**, validando que los datos están en condiciones óptimas para pasar a la siguiente fase de desagregación de demanda y modelado predictivo.

------------------------------------------------------------------------

### Justificación del uso de patrones estacionales

Aunque el dataset inicial es artificial, se ha enriquecido mediante la aplicación de un **patrón estacional basado en factores reales del comercio electrónico en España** (ciclos de ingresos mensuales, estacionalidad semanal, campañas clave como rebajas, Black Friday o Navidad, y ajustes en festivos/vacaciones).  

El objetivo no es replicar al detalle un histórico pasado, sino **incorporar estructuras realistas** que permitan que los modelos de predicción entrenados posteriormente sean **robustos y aplicables a escenarios futuros**.  
De este modo, cada ajuste o modificación sobre el calendario se entiende como una **etapa de calibración deliberada**, cuyo fin es generar un dataset artificial más realista, coherente y predictivo.  

-------------------------------------------------------------------------

##  📑 Metodología – Fase 2 (Generación de históricos y validación estacional).

1. **Patrón estacional**
   - Calendario artificial enriquecido con factores del ecommerce en España (rebajas, Black Friday, Prime Day, Navidad, Semana Santa, festivos y agosto).
   - La función `generar_calendario_estacional(...)` se aplica por año sin I/O y los scripts guardan el calendario por año.

2. **Desagregación de demanda anual**
   - Conversión a demanda **diaria** para 2022, 2023 y 2024.
   - Entradas: `data/processed/demanda_diaria_YYYY.csv` (columnas: `Date`, `Demand_Day`).
   - Salidas EDA: `outputs/figures/*` y `outputs/tables/*`.

3. **Comparativa entre años (EDA)**
   - Evolución diaria total por año, medias mensuales con CV, correlaciones interanuales y KPIs de consistencia.

4. **Validación con calendario real (iterativa)**
   - Objetivo: comprobar si los picos/vales de la demanda desagregada coinciden con eventos reales.
   - Se probaron diferentes enfoques: baseline mensual, filtro de activos, baseline DoW, baseline local ±k y ventanas desplazadas ±shift.
   - **Configuración final** seleccionada: **baseline local ±7 con ventanas ±3**.
   - Salidas:
     - `outputs/tables/validacion_calendario_real_SHIFT_localk7_s3_YYYY.csv`
     - `outputs/tables/validacion_calendario_real_kpis_SHIFT_k7_s3.csv`
     - `outputs/figures/evolucion_2024_con_eventos_SHIFT_k7_s3.png`

📓 **Nota metodológica (validación estacional)**  
Se elaboró un **Notebook bitácora** con todas las iteraciones de validación frente a calendario real.  
Este README y el notebook **PFM2_Fase2** recogen únicamente la **versión final consolidada** (baseline local ±7 y ventanas ±3).  
La bitácora está disponible en la carpeta `/notebooks/` como material complementario.


### Reproducibilidad (script)

# Desde la raíz del proyecto
python scripts/eda/validacion_calendario_real.py \
  --years 2022 2023 2024 --k 7 --shift 3


📌 Conclusión de la Fase 2:
La validación estacional confirma que los históricos desagregados reflejan un comportamiento coherente con eventos de mercado. La configuración final 
seleccionada (baseline local ±7 y ventanas ±3) se utilizará como feature base en la siguiente fase, donde se integrarán variables de precio y externas.

-------------------------------------------------------------------------
## 📑 Metodología – Fase 3 (Clustering de productos y generación de subset representativo)

### 3.1 Preparación de features de producto
- Limpieza y normalización de categorías.
- Eliminación de productos sin PCs relevantes (categorías no representativas).
- Obtención de `productos_features_clean.csv` como base de entrada al análisis de clustering.

### 3.2 Clustering de productos
- Comparación de técnicas: **K-Means, GMM y DBSCAN**.
- Métricas internas: silhouette score (K-Means con *k=4* ≈ 0.32).
- Interpretación de clusters:
  - **C0** → nicho reducido (~210 productos).
  - **C1** → productos estables y de alta demanda (~1.233).
  - **C2** → cluster mayoritario (~3.394).
  - **C3** → miscelánea (~1.101).
- Decisión final: **K-Means (k=4)** como modelo de referencia.  
  GMM y DBSCAN se emplearon como técnicas de contraste.

### 3.3 Validación complementaria del clustering
- Uso del notebook auxiliar `PFM2_Fase3_pruebas_clustering.ipynb`.
- Confirmación de la robustez de K-Means frente a alternativas.

### 3.4 Asignación de clusters a la demanda desagregada
- Script `asignar_cluster_a_demanda.py`:
  - **Entrada**: `demanda_filtrada_enriquecida_sin_nans.csv` + `productos_clusters.csv`.
  - **Salida**: `demanda_con_clusters.csv`.
  - Exclusión de productos sin cluster asignado (~113 productos, <2%).
- Validaciones:
  - Cobertura temporal completa.
  - Asignación de clusters consistente con el catálogo.

### 3.5 Generación del subset representativo
- **Criterios de selección**:
  - Mantener cobertura completa de fechas (2022–2024).
  - Conservar casi completos los clústeres minoritarios (C0, C1, C3).
  - Reducir proporcionalmente el clúster mayoritario (C2) al ~30%.
  - Incluir **todos los outliers** como casos especiales (no eliminar top ventas atípicos).
- **Outputs intermedios**:
  - `demanda_subset.parquet` (30% sin outliers).
  - `outliers.parquet` (productos con `is_outlier=1`).
  - `demanda_subset_final.parquet` (fusión subset + outliers).

### 3.6 Validaciones del subset final
- **Validación rápida pre-outliers**:
  - Reglas de reducción de clusters aplicadas correctamente.
  - Integridad de claves sin duplicados ni nulos.
- **Validación robusta final** (sobre parquet):
  - Cobertura temporal 2022–2024 intacta.
  - Sin duplicados ni NaNs en claves.
  - Subset ⊆ catálogo original.
  - Todos los outliers conservados (220.296 filas, 201 productos).
  - Distribución por cluster coherente tras reducción.
  - Tamaño final: 3.596 productos (~60% del catálogo).


📌 **Conclusión de la Fase 3**:  
Se ha construido un **subset representativo, coherente y manejable** (30% + outliers), que mantiene diversidad de clusters, top ventas atípicos y cobertura temporal completa. Este subset servirá como base para la **Fase 4**, centrada en el análisis del impacto del precio y variables externas.

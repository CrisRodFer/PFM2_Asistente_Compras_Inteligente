# 📌 PFM2_Asistente_Compras_Inteligente

## 📖 Descripción

Trabajo de fin de máster en **Data Science & IA**.\
Este proyecto consiste en el desarrollo de un **Asistente Inteligente de
Compras**, cuyo objetivo es optimizar la gestión de inventario y stock
de una empresa mediante técnicas de análisis de datos, simulación de
demanda y modelado predictivo.

------------------------------------------------------------------------

## 📂 Estructura del proyecto

📂 PFM2_Asistente_Compras_Inteligente
├── 📂 data
│   ├── 📂 raw
│   ├── 📂 clean
│   │   └── supplier_catalog_multi.csv
│   ├── 📂 processed
│   │   ├── demanda_all_adjusted.parquet
│   │   ├── demanda_all_adjusted_postnoise.parquet
│   │   ├── subset_modelado.parquet
│   │   ├── dataset_modelado_ready.parquet
│   │   ├── predicciones_2025.parquet
│   │   ├── predicciones_2025_estacional.parquet
│   │   ├── predicciones_2025_optimista.parquet
│   │   ├── predicciones_2025_pesimista.parquet
│   │   ├── products.parquet                 # ← Fase 9
│   │   ├── suppliers.parquet                # ← Fase 9
│   │   └── substitutes_unified.parquet      # ← Fase 9
│   └── 📂 external
│
├── 📂 outputs
│   ├── 📂 figuras
│   └── 📂 controles_escenarios
│       ├── control_totales_optimista.csv
│       ├── control_por_cluster_optimista.csv
│       ├── control_totales_pesimista.csv
│       └── control_por_cluster_pesimista.csv
│
├── 📂 reports
│   └── fase9_validations_summary.json       # ← métricas validación F9
│
├── 📂 scripts
│   ├── 📂 eda
│   ├── 📂 transform
│   ├── 📂 modeling
│   ├── 📂 utils
│   └── 📂 export
│       └── construir_vistas.py              # ← export vistas F9
│
├── 📂 notebooks
│   └── PFM2_Modelado_y_app.ipynb
│
├── README.md
└── requirements.txt



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


-------------------------------------------------------------------------

## 📑 Metodología – Fase 4 (Impacto del precio sobre la demanda)

### 4.1 Objetivo, datos de partida y diseño del efecto (ventanas + elasticidades)
- **Objetivo**: simular cómo cambian las unidades cuando decidimos modificar el precio (descuentos/subidas), manteniendo separados otros efectos (promos no-precio, externos). Se trabaja sobre el subset final (2022–2024) y se alinea con el calendario real validado en Fase 2.
- **Datos de partida** (dataset `demanda_subset_final`):
  - Demanda → `Demand_Day`
  - Producto → `Product_ID`
  - Fecha → `Date`
  - Clúster → `Cluster`
  - Outliers → `is_outlier`
  - Precio base (si existe) → `precio_medio`  
  > Si no hay serie de precio real, se genera **precio virtual** a partir de las ventanas.
- **Ventanas**: `start`, `end`, `discount` (p. ej. −0.10), `scope_type` (`global|cluster|product_id`) y `scope_values`.
- **Elasticidades por clúster (arranque)**:
  - C0 = −0.6 (poco sensible)
  - C1 = −1.0 (media)
  - C2 = −1.2 (alta)
  - C3 = −0.8 (media-baja)
- **Fórmula del efecto**  
  Multiplicador de unidades por día:  
  `M_price,t = (1 + d_t)^(ε_{g(i)})`  ⇒  `qty_{i,t} = baseline_{i,t} * M_price,t`
- **Guardarraíles y política de outliers**
  - **CAP por clúster (sin evento)**: C0 1.8×, C1 2.2×, C2 2.8×, C3 2.0×.  
    En día de **evento real** (SHIFT) → CAP × **1.5**.
  - **FLOOR** global: **0.5×**.
  - **Outliers**: si `is_outlier==1` y `M>1`, **no amplificar** (forzar `M=1`).  
  - **Solapes**: prioridad `product_id` > `cluster` > `global` (se elige el mayor `|M−1|`).


### 4.2 Preflight de ventanas — `scripts/analysis/ventanas_precio.py`
- Genera/actualiza **`data/auxiliar/ventanas_precio.csv`** y **`data/auxiliar/preflight_ventanas.xlsx`** (sanity de ventanas).
- Entradas: SHIFT `outputs/tables/validacion_calendario_real_SHIFT_*.csv`.
- Recomendado para revisar coberturas, solapes y “lifts” esperados por clúster antes de aplicar.


### 4.3 Aplicación del efecto — `scripts/transform/aplicar_efecto_precio.py`
- **Inputs**: `demanda_subset_final.parquet`, `ventanas_precio.csv`, y SHIFT `outputs/tables/validacion_calendario_real_SHIFT_*.csv`.
- **Outputs**:
  - `data/processed/demanda_price_adjusted.parquet`  
    (añade `demand_multiplier`, `Demand_Day_priceAdj`, `price_factor_effective`, `Price_virtual` o `Price_index_virtual`).
  - `outputs/tables/price_calendar.parquet` (calendario de multiplicadores; útil para la app).


### 4.4 Validación rápida (sanity)
- **Rangos**: `M ∈ [0.5, CAP×1.5]` sin valores fuera de tope/suelo.  
- **Cobertura**: % de días con `M≠1` acorde a duración de campañas.  
- **Outliers**: si `is_outlier==1` y `M>1` ⇒ `M=1`.  
- **Consistencia**: `price_factor_effective ≈ M^(1/ε)` (error ≈ 0).  
- **Clúster**: mayor lift en C2, intermedio C1/C3, menor C0 (coherente con elasticidades).



### 4.5 Validación adicional: alineamiento con calendario real
- Alineamiento precio vs. ventanas (±3 días): **precision ≈ 1.00** y **recall ≈ 0.67** por año.  
  ⇒ todo el efecto cae **dentro** de ventanas; no todas las fechas de ventana se usan (diseño esperado por umbral/cobertura).
- Gráficas anuales: picos/mesetas de la serie ajustada coinciden con zonas sombreadas (rebajas, agosto, BF, Navidad).



**📌 Conclusión de la Fase 4**  
Fase 4 **OK**. Dataset listo para el **modelado** y para la app de escenarios (*what-if* de precio).

**⏭️ Reproducibilidad (Fase 4)**

```bash
# 1) Ventanas de precio (CSV + preflight)
python scripts/analysis/ventanas_precio.py

# 2) Aplicar efecto precio a la baseline (parquet ajustado + calendario)
python scripts/transform/aplicar_efecto_precio.py


➡️ **Entradas previas necesarias**
- `data/processed/demanda_subset_final.parquet (de Fases 1–3)`.
- SHIFT en `outputs/tables/validacion_calendario_real_SHIFT_*.csv (Fase 2)`.

⬅️ **Salidas clave**
- `data/auxiliar/ventanas_precio.csv`, `data/auxiliar/preflight_ventanas.xlsx`
- `data/processed/demanda_price_adjusted.parquet`
- `outputs/tables/price_calendar.parquet`

-------------------------------------------------------------------------

## 📑 Metodología – Fase 5 (Factores externos, ruido y validación)

### 5.1 Preparación de los datos de entrada
- Punto de partida: `data/processed/demanda_price_adjusted.parquet` (output de la Fase 4).
- Columnas clave: `Demand_Day`, `Demand_Final`, `Demand_Final_Noiseds`, `demand_multiplier`, `Factors_Applied`.
- Revisión inicial de integridad y consistencia antes de aplicar factores externos.

### 5.2 Definición de factores externos
- Factores considerados: **inflación, promociones no-precio, competencia, estacionalidad extra y eventos específicos** (ej. agosto).
- Cada factor se implementa como columna multiplicadora `M_factor`, aplicada sobre la serie de referencia.
- Se evita solapamiento con el precio (ya ajustado en Fase 4).
- Se añaden tolerancias y ventanas alineadas al calendario real validado en Fase 2.

### 5.3 Diseño del modelo de aplicación
- Fórmula general de la demanda ajustada: Demand_Final = Demand_Day × (Π M_factor_i)
- Cuando ningún factor aplica → `Demand_Final = Demand_Day`.
- En cada fila se registra en `Factors_Applied` la lista exacta de factores activos.
- Esta trazabilidad permite auditar el efecto de cada multiplicador.

### 5.4 Implementación en código
- Scripts principales:
- `ventanas_externos.py` → genera ventanas externas (`calendar_externos.parquet`).
- `enriquecer_ventas_externos.py` → aplica y valida ventanas externas (sanity checks).
- `aplicar_factores.py` → aplica los multiplicadores a la demanda.
- Outputs intermedios:  
- `data/auxiliar/ventanas_externos.csv`, `preflight_externos.xlsx`
- `data/processed/demanda_all_adjusted.parquet`, `calendar_total_externos.parquet`

### 5.5 Validación de coherencia y robustez.

**5.5.1 Validación de coherencia del precio**  
- Revisión de rangos de multiplicadores (`M ∈ [0.5, CAP×1.5]`).
- Confirmación de que los valores son consistentes con elasticidades y ventanas de Fase 4.

**5.5.2 Validación adicional (alineamiento ventanas)**  
- Comparación con el calendario real ±3 días.  
- Métricas: *Precision* = 1.0; *Recall* ≈ 0.67–0.69; *F1* ≈ 0.80.  
- Gráficas muestran picos alineados con ventanas de rebajas, agosto, Black Friday, etc.

**5.5.3. Comparativa de demanda.**  
- Introducción de ruido lognormal para enriquecer la serie y aumentar la variabilidad de forma controlada.  
- La comparativa entre `Demand_Day`, `Demand_Final` y `Demand_Final_Noiseds` mostró que algunos clústers presentaban un 
  exceso de ruido (40–60%), mientras que otros mantenían niveles razonables (~20%).  
- Para estabilizar la serie se aplicó un ajuste por clúster (`ajuste_ruido_por_cluster.py`), reduciendo los casos con exceso a ~22% y 
  manteniendo sin cambios los clústers ya estables.  
- El resultado es un dataset más realista y consistente, que conserva la señal original pero mejora la 
  robustez para el modelado predictivo.

**5.5.4 Validación de trazabilidad**  
- Confirmación de que `Factors_Applied` refleja exactamente los multiplicadores ≠ 1.  
- Top combinaciones:  
- `inflation|seasonExtra` (~3.2M filas)  
- `agosto_nonprice|inflation|seasonExtra` (~334K filas)  
- `inflation|promo|seasonExtra` (~237K filas)  
- Se confirma que no aparecen factores espurios ni inconsistencias.

📌 **Conclusión de la Fase 5**:  
La demanda ajustada resultante es **estadísticamente coherente, trazable y alineada con el calendario real**, constituyendo una base sólida para la siguiente fase de **modelado predictivo**.

-------------------------------------------------------------------------

## 📑 Metodología – Fase 6 (Análisis y tratamiento de outliers)

### 6.1 Validación complementaria (`is_outlier = 0`)
- Punto de partida: `data/processed/demanda_all_adjusted_postnoise.parquet`.
- Criterios aplicados: MAD z-score y percentiles P95/P99.
- Clasificación automática: top_venta, pico_aislado, mixto.
- Outputs:
  - `reports/outliers/outliers_candidatos_nuevos_dias.csv`
  - `reports/outliers/outliers_candidatos_nuevos_productos.csv`

### 6.2 Revisión de outliers DBSCAN (`is_outlier = 1`)
- Se aplica la misma lógica que en 6.1 sobre los casos detectados inicialmente con DBSCAN.
- Clasificación y decisiones por producto-año.
- Outputs:
  - `reports/outliers/outliers_dbscan_dias.csv`
  - `reports/outliers/outliers_dbscan_productos.csv`

### 6.3 Consolidado y decisiones finales
- Unión y priorización de tipologías: top_venta > mixto > pico_aislado.
- Decisiones aplicadas: sin_cambio, suavizado_a015, alerta_pendiente.
- Outputs:
  - `reports/outliers/outliers_resumen.csv`
  - `reports/outliers/outliers_resumen_metricas.csv`

### 6.4 Implicaciones para modelado y subset final
- Se mantiene la columna original is_outlier (DBSCAN).
- Se añaden columnas de trazabilidad anual:
  - `tipo_outlier_year`
  - `decision_outlier_year`
- Output final:
  -`data/processed/subset_modelado.parquet`

📌  **Conclusión de la Fase 6**.
Los picos aislados quedan justificados por calendario real y los top ventas se mantienen; no se detectan outliers espurios. El dataset resultante está validado y listo para la Fase 7 (modelado y aplicación).

-------------------------------------------------------------------------

## 📑 Metodología – Fase 7 (Validación y preparación del dataset para modelado)

### 7.1 Validación inicial del dataset
- Script principal: `scripts/eda/validacion_dataset_modelado.py`
- Objetivos:
  - Verificar integridad del target (`demand_final_noised`).
  - Confirmar cobertura temporal completa (2022–2024).
  - Detectar duplicados en (`product_id`, `date`).
  - Validar la coherencia de los clústeres y la trazabilidad de los outliers.
  - Generar reporte tipo “semáforo” con indicadores críticos (OK/NO-OK).
- Script auxiliar: `scripts/eda/check_outliers_clusters.py`
  - Objetivo: auditar la coherencia entre `cluster` y `__cluster__` y confirmar la asignación de outliers.
  - Se documenta como herramienta complementaria, no obligatoria en el pipeline.
- Resultados:
  - Target sin nulos ni negativos.
  - Cobertura temporal completa hasta 2024-12-31.
  - Sin duplicados por (`product_id`, `date`).
  - Todos los productos con clúster asignado (0–3).
  - Outliers asignados al clúster mayoritario, garantizando cobertura.
- Conclusión: el dataset `subset_modelado.parquet` queda validado como punto de partida fiable para el modelado.


### 7.2 Preparación del dataset para modelado
- Script: `scripts/transform/preparacion_dataset_modelado.py`
- Transformaciones aplicadas:
  - Renombrado de columnas clave:
    - `__cluster__` → `cluster_id`
    - `demand_final_noised` → `sales_quantity`
  - Eliminación de columnas redundantes:
    - `cluster`, `__product_id__`, `demand_final_noiseds_adj`
  - Normalización de tipos:  
    - `date` → datetime  
    - `product_id` → string  
    - `cluster_id` → int
  - Control de duplicados y nulos:
    - Sin duplicados en (`product_id`, `date`)  
    - Sin nulos en `sales_quantity`
  - Selección final de variables: identificadores, target, precio, factores externos y trazabilidad.
- Output final:  
  - `data/processed/dataset_modelado_ready.parquet` → dataset limpio y consolidado para modelado.
- Verificación post-transformación:
  - Confirmada cobertura temporal completa.  
  - `sales_quantity` sin nulos ni negativos.  
  - `product_id` válido (sin 0 ni cadenas vacías).  
  - Sin duplicados en (`product_id`, `date`).  
  - `cluster_id` completo y dentro del rango esperado (0–3).  


### 7.3 Target y features disponibles
- **Target definido:**
  - `sales_quantity` → demanda diaria final por producto, consolidada y validada.
- **Features disponibles:**
  - Identificadores y estructura temporal:
    - `product_id`, `date`, `cluster_id`
  - Precio y derivados:
    - `precio_medio`, `price_virtual`, `price_factor_effective`, `demand_day_priceadj`
  - Factores externos:
    - `m_agosto_nonprice`, `m_competition`, `m_inflation`, `m_promo`, etc.
  - Outliers y trazabilidad (opcionales):
    - `is_outlier`, `tipo_outlier_year`, `decision_outlier_year`
- Implicaciones:
  - Los modelos de series temporales clásicos (p.ej. SARIMAX, Holt-Winters) usarán el target y exógenas seleccionadas.
  - Los modelos de machine learning (Ridge, Random Forest) podrán explotar un conjunto más amplio de features.
  - Este listado define el universo de variables disponibles, dejando la selección específica para la Fase 8.

📌 **Conclusión de la Fase 7**:  
El dataset `dataset_modelado_ready.parquet` constituye la **base estable, homogénea y trazable** para el modelado.  
Con esta fase se cierra todo el bloque de preparación y se garantiza que los modelos de la Fase 8 se entrenarán sobre datos limpios, validados y estructurados.

-------------------------------------------------------------------------
📑 Metodología – Fase 8 (Modelado de la demanda).

### 8.1 Preparación del dataset para entrenamiento
- **Script principal:** `scripts/eda/preparacion_dataset_ml.py`
- **Objetivos:**
  - Preparar el dataset `dataset_modelado_ready.parquet` como entrada para modelos de series temporales y de ML.
  - Asegurar consistencia en el target (`sales_quantity`) y en las features seleccionadas.
- **Resultados:**
  - Dataset preparado y verificado como base común para los experimentos de modelado.
- **Conclusión:** El dataset queda listo para entrenar modelos bajo diferentes enfoques.



### 8.2 Baselines
- **Scripts principales:**  
  - `scripts/modeling/seasonal_naive.py`  
  - `scripts/modeling/holt_winters.py`  
- **Objetivos:**
  - Establecer modelos de referencia iniciales (benchmarks).
  - Evaluar enfoques sencillos: baseline por clúster, Seasonal Naive y Holt-Winters (ETS).
- **Resultados:**
  - Se generan métricas de error para cada baseline.
  - Se confirma que los modelos clásicos capturan cierta estacionalidad pero no son robustos en clústeres con alta variabilidad.
- **Conclusión:** Es necesario avanzar hacia modelos más complejos para mejorar precisión y estabilidad.


### 8.3 Modelos clásicos de series temporales
- **Script principal:** `scripts/modeling/sarimax_cluster.py`
- **Objetivos:**
  - Entrenar y evaluar modelos SARIMAX por clúster.
  - Incluir factores exógenos relevantes en la predicción.
- **Resultados:**
  - SARIMAX obtiene resultados aceptables en clústeres estables.
  - En clústeres más variables, el modelo presenta limitaciones.
- **Conclusión:** SARIMAX aporta valor pero no ofrece mejoras consistentes frente a los baselines en todos los casos.


### 8.4 Modelos de regresión y ML
- **Scripts principales:**  
  - `scripts/modeling/regresion_ml.py`  
  - `scripts/modeling/evaluacion_global.py`  
- **Objetivos:**
  - Aplicar modelos de regresión y Random Forest para capturar relaciones no lineales.
  - Comparar el rendimiento frente a SARIMAX y baselines.
  - Analizar la importancia de variables y su interpretabilidad.
- **Resultados:**
  - Random Forest ofrece los mejores resultados globales.
  - Se identifican como variables clave: precio medio, promociones y factores externos.
- **Conclusión:** Random Forest se establece como el modelo más adecuado para este caso de uso.


### 8.5 Backtesting y comparación
- **Script principal:** `scripts/modeling/backtesting.py`
- **Objetivos:**
  - Validar los modelos mediante backtesting.
  - Comparar métricas globales y por clúster (WAPE, MAPE, RMSE).
- **Resultados:**
  - Random Forest mantiene un rendimiento más estable frente a SARIMAX y baselines.
- **Conclusión:** El modelo seleccionado es consistente bajo distintos periodos de validación.


### 8.6 Predicciones finales (2025) – escenario neutro con estacionalidad
- **Script principal:** `scripts/modeling/predicciones_2025.py`
- **Objetivos:**
  - Generar el escenario neutro estacionalizado para 2025.
  - Establecer una base de comparación para escenarios alternativos.
- **Resultados:**
  - Se produce el archivo `predicciones_2025_estacional.parquet`.
- **Conclusión:** El escenario neutro estacionalizado queda validado como baseline para simulaciones.


### 8.7 Simulación de escenarios optimista y pesimista
- **Scripts principales:**  
  - `scripts/utils/simular_escenario.py`  
  - `scripts/modeling/simular_escenario_optimista.py`  
  - `scripts/modeling/simular_escenario_pesimista.py`  
- **Objetivos:**
  - Simular escenarios alternativos aplicando factores derivados de las métricas de backtesting.
  - Optimista: multiplicar predicciones por `1 + WAPE_%`.
  - Pesimista: multiplicar predicciones por `1 - WAPE_%` (con límite ≥ 0).
- **Resultados:**
  - Escenarios alternativos generados en formato Parquet (`predicciones_2025_optimista.parquet`, `predicciones_2025_pesimista.parquet`).
  - Controles comparativos exportados en CSV (`outputs/controles_escenarios`).
- **Conclusión:** Los escenarios reflejan un rango realista de incertidumbre basado en la precisión histórica del modelo.


### 8.8 Conclusiones y líneas futuras
- **Conclusiones:**
  - Random Forest se confirma como el modelo más robusto para predecir la demanda base.
  - El backtesting valida la coherencia del escenario neutro y los escenarios alternativos.
  - El enfoque por clúster resulta clave para capturar patrones diferenciados.
- **Líneas de mejora:**
  - Probar otros modelos avanzados (Prophet, LSTM).
  - Refinar la simulación de factores externos (exógenas).
  - Incorporar un módulo de ajuste manual para eventos extraordinarios.



📌 **Conclusión de la Fase 8**:  
La Fase 8 consolida el bloque de modelado, confirmando que Random Forest es el modelo más robusto para predecir la demanda. El backtesting valida la coherencia del escenario neutro y de los escenarios alternativos, asegurando un rango realista de proyecciones. Se cierra así la fase de modelado con una base sólida para la toma de decisiones en compras y planificación.

-------------------------------------------------------------------------
## 📑 Metodología – Fase 9 (Catálogo multiproveedor y sustitutos + export)

### 9.1 Objetivo
Enriquecer el catálogo con **proveedores alternativos** y validar reglas clave antes de exportar las **vistas** que consumirá la app de Streamlit (la explicación de Streamlit va en su apartado propio).

### 9.2 Datos de entrada
- `data/processed/preferred_supplier.csv`  (proveedor preferente por `product_id`)
- `data/clean/supplier_catalog_multi.csv`  (catálogo con candidatos alternativos)
- `data/clean/substitutes.csv`             (sustitutos por similitud de producto)

### 9.3 Validaciones multiproveedor
Se realizan sobre el cruce `preferred_supplier` + `supplier_catalog_multi`:

- **V1 Proveedor distinto**: el alternativo nunca coincide con el preferente.  
- **V2 Precio modificado**: `precio_alt` difiere de `precio_pref` (tolerancia ±0.5%).  
- **V3 Stock modificado**: `disponibilidad_alt` ≠ `disp_pref` (se permite 0 para simular rotura).  
- **V4 Lead time bucket**: coherente con `lead_time` (`2-4`, `5-7`, `10-15`).  

Las validaciones son duras: si alguna falla se muestra una muestra de filas y **se corta** la ejecución.

**Salida de control**: `reports/fase9_validations_summary.json` (métricas agregadas).

### 9.4 Esquema de salida (para la app)
> Este bloque es **documentación**: no altera la app ni sus inputs.

- **products**: `product_id`, `categoria`, `pack_size`, `uom`, `precio_medio`  
- **suppliers**: `product_id`, `supplier_id`, `prioridad`, `precio`, `disponibilidad`, `lead_time`  
- **substitutes**: `product_id`, `sustituto_id`, `score`, `categoria`

### 9.5 Export de vistas (consumo por Streamlit)
Script: `scripts/export/construir_vistas.py`

**Salidas**
- `data/processed/products.parquet`  
- `data/processed/suppliers.parquet`  
- `data/processed/substitutes_unified.parquet`  

### 9.6 Reproducibilidad (Fase 9)
```bash
# Desde la raíz del repo
python scripts/export/construir_vistas.py

📌 Conclusión de la Fase 9
Catálogo multiproveedor validado (V1–V4) y vistas exportadas para la app.
-------------------------------------------------------------------------
## 📑 Metodología – Fase 10 (Motor de movimientos de stock)

### 10.1 Objetivo
Desarrollar un **procesador de movimientos de stock** que actualice el inventario en función de pedidos de cliente, genere ledger de movimientos, dispare alertas y sugiera órdenes de compra, incluyendo el manejo de sustitutos.

### 10.2 Datos de entrada
- `data/clean/Inventario.csv` (inventario inicial; columnas: Product_ID, Proveedor, Nombre, Categoria, Stock Real).  
- `customer_orders.csv` (pedidos simulados; columnas: date, item_id, qty).  
- `supplier_catalog.csv` (catálogo enriquecido con precio, lead_time, moq, múltiplo).  
- `service_policy.csv` (políticas de cobertura y stock de seguridad por categoría).  
- `substitutes.csv` (sustitutos internos y externos por Product_ID).  

### 10.3 Lógica implementada
1. **Carga y validación de inventario y pedidos**.  
   - Control de columnas obligatorias y formatos.  
   - Posibilidad de generar pedidos DEMO en ausencia de input.  

2. **Enriquecimiento con catálogos y políticas**.  
   - Merge con catálogo multiproveedor y política de servicio.  
   - Fallbacks: cobertura=14 días, lead_time=5 días, SS=0 uds.  

3. **Cálculo de métricas base**.  
   - Demanda media diaria (`μ_d`).  
   - Reorder Point (ROP = μ_d·LT + SS).  

4. **Procesado de pedidos**.  
   - Actualización secuencial de `stock_actual`.  
   - Ledger cronológico de movimientos (ventas, recepciones).  
   - Generación de alertas (WARN, CRIT) ante inconsistencias o roturas.  

5. **Etiquetado y sugerencias de compra**.  
   - Flags `flag_rotura` y `flag_bajo`.  
   - Cálculo de `qty_sugerida` con validación de MOQ y múltiplos.  
   - Cálculo de importes (`qty_sugerida` × `precio`).  

6. **Sustitutos**.  
   - Búsqueda de hasta 3 alternativas por SKU en rotura.  
   - Filtros por disponibilidad y ranking por score.  
   - Alertas críticas si no se encuentran sustitutos.  

7. **Órdenes de compra**.  
   - Generación de cabecera y líneas por proveedor.  
   - Control de motivo (rotura/bajo stock) y trazabilidad en ledger.  

### 10.4 Salidas
- `inventory_updated.csv` → inventario vivo post-pedidos.  
- `ledger_movimientos.csv` → histórico de movimientos.  
- `alerts.csv` → alertas generadas.  
- `sugerencias_compra.csv` → candidatos a reaprovisionamiento.  
- `ordenes_compra.csv` y `ordenes_compra_lineas.csv` → órdenes agrupadas por proveedor.  
- `sugerencias_sustitutos.csv` → propuestas de sustitutos para roturas.  

### 10.5 Validación
- Reglas aplicadas: stock ≥ 0, qty > 0 en pedidos, ROP coherente con μ_d y lead_time.  
- Se verificó consistencia de salidas y ausencia de NaNs críticos.  
- Análisis de resultados: detección correcta de roturas y bajo ROP, ledger cronológico sin inconsistencias, sustitutos sugeridos para >85% de SKUs en rotura.

📌 **Conclusión de la Fase 10**  
El motor de movimientos de stock queda consolidado: actualiza inventario, mantiene trazabilidad con ledger, dispara alertas y genera sugerencias de compra junto con sustitutos y órdenes de compra. Sus salidas alimentan directamente la app en Streamlit.

**⏭️ Reproducibilidad (Fase 10)**

```bash
# Desde la raíz del repo
python scripts/operativa/movimientos_stock.py \
  --inventario data/clean/Inventario.csv \
  --orders data/clean/customer_orders.csv \
  --supplier-catalog data/clean/supplier_catalog.csv \
  --service-policy data/clean/service_policy.csv \
  --substitutes data/clean/substitutes.csv \
  --outdir data/processed/fase10_stock

-------------------------------------------------------------------------

## 📑 Metodología – Fase 11 (Despliegue en Streamlit — SupplyMind)

### 11.1 Objetivo
Desplegar el asistente de compras en una **app interactiva en Streamlit**, conectando las salidas de la Fase 10 con una interfaz usable para el técnico de compras.

### 11.2 Bloques principales de la app
1. **Home**: portada con navegación mediante tarjetas a las distintas secciones.  
2. **Exploración & Sustitutos**: consulta de inventario vivo y sustitutos unificados, con edición opcional de pedidos UI y acceso a ledger, órdenes y alertas.  
3. **Proveedores**: vista por proveedor con catálogo, precios y análisis de históricos 2023–2025 (ranking y tendencias).  
4. **Movimientos de stock**: inventario vivo con flags de rotura/bajo ROP, buscador y filtrado por proveedor/categoría.  
5. **Reapro / Pedidos**: consolidación de sugerencias, gestión de órdenes en curso y recibidas, y simulador de escenarios de demanda.  
6. **Mensajes y descargas**: alertas INFO/WARN/CRIT alineadas con `alerts.csv` y botones para descargar CSVs.  

### 11.3 Datos consumidos
- Outputs de Fase 10 (`inventory_updated.csv`, `ledger_movimientos.csv`, `alerts.csv`, `sugerencias_compra.csv`, `ordenes_compra*.csv`, `sugerencias_sustitutos.csv`).  
- Vistas de catálogo multiproveedor (`products.parquet`, `suppliers.parquet`, `substitutes_unified.parquet`).  
- Predicciones 2025 (`predicciones_2025*.parquet`) para el simulador.  

### 11.4 Validaciones de la app
- Se verificó coherencia en la carga de CSV/Parquet y alineación de nombres de columnas (`Product_ID` vs `item_id`).  
- La navegación mantiene el estado entre secciones gracias a `st.session_state`.  
- Se comprobó la descarga correcta de archivos y la consistencia de mensajes de alerta.  
- El simulador reproduce escenarios alternativos coherentes con los definidos en Fase 8.  

📌 **Conclusión de la Fase 11**  
La app en Streamlit convierte SupplyMind en un **asistente operativo**: permite visualizar inventario y alertas, gestionar sustitutos y multiproveedor, generar y recibir órdenes, y simular escenarios de demanda. Con ello, el proyecto culmina en una herramienta práctica y directamente utilizable en el área de compras.

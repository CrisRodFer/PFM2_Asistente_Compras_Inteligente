# 📌 PFM2_Asistente_Compras_Inteligente

## 📖 Descripción

Trabajo de fin de máster en **Data Science & IA**.\
Este proyecto consiste en el desarrollo de un **Asistente Inteligente de
Compras**, cuyo objetivo es optimizar la gestión de inventario y stock
de una empresa mediante técnicas de análisis de datos, simulación de
demanda y modelado predictivo.

------------------------------------------------------------------------

## 📂 Estructura del proyecto

PFM2_Asistente_Compras_Inteligente/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── clean/
│   ├── processed/         # demanda_subset_final.parquet, demanda_price_adjusted.parquet, ...
│   └── auxiliar/          # ventanas_precio.csv, preflight_ventanas.xlsx (antes "aux")
├── outputs/
│   ├── figures/
│   └── tables/            # validacion_calendario_real_SHIFT_*.csv, price_calendar.parquet, ...
├── notebooks/
├── scripts/
│   ├── analysis/          # ventanas_precio.py  (genera/valida ventanas de precio)
│   └── transform/         # aplicar_efecto_precio.py (aplica efecto a la baseline)
├── requirements.txt
└── ...

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

-------------------------------------------------------------------------


📌 **Conclusión de la Fase 5**:  
La demanda ajustada resultante es **estadísticamente coherente, trazable y alineada con el calendario real**, constituyendo una base sólida para la siguiente fase de **modelado predictivo**.


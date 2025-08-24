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
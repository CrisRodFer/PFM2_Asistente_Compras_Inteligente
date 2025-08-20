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
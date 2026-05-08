# Predicción Educativa - TalentoTech 🎓

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

Este repositorio contiene el flujo completo de Machine Learning (bajo la metodología CRISP-ML) para predecir la Tasa de Aprobación y el Riesgo de Deserción Escolar en municipios de Colombia. Todo construido a partir de los datos originales de `EDUCACION.csv`.

El proyecto está diseñado para funcionar en un entorno **Python 3.9, 3.10 o 3.11**, siendo completamente compatible con Streamlit Cloud.

## Novedades del Dashboard Integral

La aplicación de Streamlit (`app.py`) ha sido transformada en un Dashboard Integral que incluye:
- 📊 **Análisis Exploratorio y Gráficos:** Filtros por departamento, caja de bigotes, gráficos de dispersión y diagrama de barras top 10 poblacional. Todo renderizado interactivamente.
- 📈 **Modelo 1: Regresión Lineal:** Predice de manera continua el porcentaje exacto estimado de la tasa de *aprobación*.
- 🔮 **Modelo 2: Regresión Logística:** Clasificador entrenado para identificar si un municipio tiene un *alto riesgo de deserción* basado en la cobertura.
- 📝 **Quiz Interactivo:** Un test dinámico estilo juego integrado en el dashboard.

## Contenido del Proyecto

Se ha dividido el ciclo de vida de los datos en tres cuadernos principales, detallados y comentados a profundidad:

1. **`01_ETL.ipynb`**: Extracción, Transformación y Carga de datos. Limpieza de strings numéricos, imputación de nulos y generación del archivo `EDUCACION_limpio.csv`.
2. **`02_EDA.ipynb`**: Análisis Exploratorio de Datos. Incluye revisión de la cabecera, información del dataset y gráficas descriptivas para entender las correlaciones y distribuciones.
3. **`03_Modelo.ipynb`**: Entrenamiento, evaluación y exportación paso a paso.
4. **`index.html`**: Landing page web diseñada para ser desplegada en **GitHub Pages**. (Contiene botones apuntando directamente a la app final y al repo de GitHub).
5. **`app.py`**: El dashboard interactivo en **Streamlit**.
6. **`requirements.txt`**: Archivo con las dependencias necesarias.

## Despliegue

### 1. Landing Page (GitHub Pages)
Puedes visualizar la landing page subiendo este código a tu cuenta de GitHub (por ejemplo al repositorio: `https://github.com/jhoansystem/deser_edu.git`) y activando **GitHub Pages**.

### 2. Aplicación Streamlit
La aplicación principal se puede visualizar en este enlace público proporcionado: `https://deseredu.streamlit.app/`.

Para ejecutar la aplicación localmente, asegúrate de usar Python 3.9+:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desarrollado para
**TalentoTech**

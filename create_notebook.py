import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Ciclo de Vida del Machine Learning - CRISP-ML\n## Predicción de la Tasa de Aprobación Escolar en Colombia"),
    nbf.v4.new_markdown_cell("### 1. Identificación del Problema\n**Objetivo:** Predecir la tasa de aprobación escolar (`APROBACIÓN`) a nivel municipal utilizando variables demográficas y de cobertura educativa.\n**Problema:** La deserción y reprobación escolar son problemas críticos. Identificar qué factores influyen en la aprobación escolar permite a los gobiernos locales focalizar sus estrategias educativas.\n**Metodología:** CRISP-ML."),
    
    nbf.v4.new_markdown_cell("### 2. Recolección de Datos\nCargamos el conjunto de datos `EDUCACION.csv`."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Cargar los datos\ndf = pd.read_csv('EDUCACION.csv')\ndf.head()"),
    
    nbf.v4.new_markdown_cell("### 3. Preparación de Datos (ETL y EDA)\nLimpieza de formatos numéricos (comas a puntos, eliminación del símbolo %), imputación de valores nulos y análisis exploratorio."),
    nbf.v4.new_code_cell("def clean_percentage(val):\n    if pd.isna(val):\n        return np.nan\n    if isinstance(val, str):\n        return float(val.replace('%', '').replace(',', '.'))\n    return val\n\n# Variables a limpiar\ncols_to_clean = ['APROBACIÓN', 'DESERCIÓN', 'REPROBACIÓN', 'COBERTURA_NETA', 'COBERTURA_BRUTA', 'POBLACIÓN_5_16']\n\nfor col in cols_to_clean:\n    if col in df.columns:\n        df[col] = df[col].apply(clean_percentage)\n\n# Rellenar valores nulos con la mediana\ndf[cols_to_clean] = df[cols_to_clean].fillna(df[cols_to_clean].median())\n\n# EDA: Matriz de correlación\nplt.figure(figsize=(8, 6))\nsns.heatmap(df[cols_to_clean].corr(), annot=True, cmap='coolwarm')\nplt.title('Correlación entre variables')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("### 4. Ingeniería de Modelos\nSelección de variables (features) y separación en conjuntos de entrenamiento y prueba. Usaremos un modelo de **Regresión Lineal**."),
    nbf.v4.new_code_cell("from sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\n\n# Selección de variables\nX = df[['POBLACIÓN_5_16', 'COBERTURA_NETA', 'COBERTURA_BRUTA']]\ny = df['APROBACIÓN']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)\n\n# Modelo de Regresión Lineal\nmodel = LinearRegression()\nmodel.fit(X_train_scaled, y_train)"),
    
    nbf.v4.new_markdown_cell("### 5. Evaluación del Modelo\nEvaluamos el desempeño del modelo usando el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R2)."),
    nbf.v4.new_code_cell("from sklearn.metrics import mean_squared_error, r2_score\n\ny_pred = model.predict(X_test_scaled)\n\nmse = mean_squared_error(y_test, y_pred)\nr2 = r2_score(y_test, y_pred)\n\nprint(f'MSE: {mse}')\nprint(f'R^2: {r2}')"),
    
    nbf.v4.new_markdown_cell("### 6. Despliegue\nEl modelo entrenado se exportará utilizando `joblib` para ser consumido en una aplicación web interactiva usando Streamlit (`app.py`)."),
    nbf.v4.new_code_cell("import joblib\n\n# Guardar el modelo y el escalador\njoblib.dump(model, 'modelo_regresion.pkl')\njoblib.dump(scaler, 'scaler.pkl')\nprint('Modelo y escalador exportados correctamente.')"),
    
    nbf.v4.new_markdown_cell("### 7. Mantenimiento y Actualización\nEl modelo debe ser reentrenado periódicamente a medida que se publiquen nuevos datos educativos anuales. Se implementará un sistema de monitoreo en la aplicación para medir el drift de los datos de entrada.")
]

with open('Pipeline_Educacion.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

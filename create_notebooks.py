import nbformat as nbf
import pandas as pd
import numpy as np

# ==========================================
# 1. CUADERNO ETL
# ==========================================
nb_etl = nbf.v4.new_notebook()
nb_etl['cells'] = [
    nbf.v4.new_markdown_cell("# Fase 1: ETL (Extracción, Transformación y Carga)\nEn este cuaderno realizaremos la limpieza profunda del dataset original `EDUCACION.csv`."),
    
    nbf.v4.new_markdown_cell("### 1. Extracción de Datos\nCargamos la base de datos original."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\n\n# Cargamos el dataset\n# Utilizamos el encoding adecuado si es necesario (ej. utf-8 o latin1)\ndf = pd.read_csv('EDUCACION.csv')\nprint('Dimensiones del dataset:', df.shape)"),
    
    nbf.v4.new_markdown_cell("### 2. Transformación de Datos\nMuchas variables vienen en formato de texto con el símbolo `%` y comas como separador decimal. Procedemos a limpiarlos y transformarlos a numéricos."),
    nbf.v4.new_code_cell("# Funciones de limpieza\ndef clean_percentage(val):\n    if pd.isna(val):\n        return np.nan\n    if isinstance(val, str):\n        return float(val.replace('%', '').replace('.', '').replace(',', '.'))\n    return val\n\ndef clean_population(val):\n    if pd.isna(val):\n        return np.nan\n    if isinstance(val, str):\n        return float(val.replace(',', '').replace('.', ''))\n    return val\n\n# Columnas de porcentajes\ncols_percentage = [\n    'APROBACIÓN', 'DESERCIÓN', 'REPROBACIÓN', \n    'COBERTURA_NETA', 'COBERTURA_BRUTA'\n]\n\n# Limpiar porcentajes\nfor col in cols_percentage:\n    if col in df.columns:\n        df[col] = df[col].apply(clean_percentage)\n\n# Limpiar población\nif 'POBLACIÓN_5_16' in df.columns:\n    df['POBLACIÓN_5_16'] = df['POBLACIÓN_5_16'].apply(clean_population)\n\ncols_all = cols_percentage + ['POBLACIÓN_5_16']\ndf[cols_all].head()"),
    
    nbf.v4.new_markdown_cell("### 3. Manejo de Valores Nulos\nImputamos los valores nulos utilizando la mediana para no afectar la distribución general por valores atípicos."),
    nbf.v4.new_code_cell("# Rellenamos los valores nulos con la mediana de cada columna\ndf[cols_to_clean] = df[cols_to_clean].fillna(df[cols_to_clean].median())\n\n# Verificamos si quedan nulos\nprint('Valores nulos restantes:\\n', df[cols_to_clean].isnull().sum())"),
    
    nbf.v4.new_markdown_cell("### 4. Carga (Guardar el Dataset Limpio)\nExportamos el dataset limpio para que sea utilizado en la siguiente fase (EDA)."),
    nbf.v4.new_code_cell("# Guardamos el nuevo archivo CSV limpio\ndf.to_csv('EDUCACION_limpio.csv', index=False)\nprint('Dataset guardado exitosamente como EDUCACION_limpio.csv')")
]

# ==========================================
# 2. CUADERNO EDA
# ==========================================
nb_eda = nbf.v4.new_notebook()
nb_eda['cells'] = [
    nbf.v4.new_markdown_cell("# Fase 2: EDA (Análisis Exploratorio de Datos)\nExploraremos el dataset limpio para encontrar patrones, distribuciones y correlaciones."),
    
    nbf.v4.new_markdown_cell("### 1. Cargar Datos y Cabecera"),
    nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Configuración visual de seaborn\nsns.set_theme(style='whitegrid')\n\n# Cargar los datos limpios\ndf_limpio = pd.read_csv('EDUCACION_limpio.csv')\n\n# Mostrar la cabecera\ndf_limpio.head()"),
    
    nbf.v4.new_markdown_cell("### 2. Información General del Dataset\nVisualizamos los tipos de datos y estadísticas descriptivas para entender la magnitud de las variables."),
    nbf.v4.new_code_cell("# Información de tipos de datos\nprint('INFORMACIÓN DEL DATASET:')\ndf_limpio.info()\n\n# Estadísticas descriptivas\nprint('\\nESTADÍSTICAS DESCRIPTIVAS:')\ndf_limpio.describe()"),
    
    nbf.v4.new_markdown_cell("### 3. Distribución de la Tasa de Aprobación\nVeamos cómo se distribuye nuestra variable objetivo."),
    nbf.v4.new_code_cell("plt.figure(figsize=(10, 6))\nsns.histplot(df_limpio['APROBACIÓN'], bins=30, kde=True, color='blue')\nplt.title('Distribución de la Tasa de Aprobación')\nplt.xlabel('Tasa de Aprobación (%)')\nplt.ylabel('Frecuencia')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("### 4. Matriz de Correlación\nRevisamos cómo se relacionan las variables de cobertura y población con la aprobación y deserción."),
    nbf.v4.new_code_cell("cols_numericas = ['APROBACIÓN', 'DESERCIÓN', 'REPROBACIÓN', 'COBERTURA_NETA', 'COBERTURA_BRUTA', 'POBLACIÓN_5_16']\n\nplt.figure(figsize=(10, 8))\ncorrelacion = df_limpio[cols_numericas].corr()\nsns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\nplt.title('Mapa de Calor de Correlaciones')\nplt.show()")
]

# ==========================================
# 3. CUADERNO MODELO
# ==========================================
nb_modelo = nbf.v4.new_notebook()
nb_modelo['cells'] = [
    nbf.v4.new_markdown_cell("# Fase 3: Modelo de Machine Learning\nCrearemos y evaluaremos un modelo de Regresión Lineal para predecir la Tasa de Aprobación."),
    
    nbf.v4.new_markdown_cell("### Paso 1: Cargar Librerías y Dataset Limpio"),
    nbf.v4.new_code_cell("import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import mean_squared_error, r2_score\nimport joblib\n\n# Cargamos los datos limpios\ndf = pd.read_csv('EDUCACION_limpio.csv')"),
    
    nbf.v4.new_markdown_cell("### Paso 2: Selección de Variables (Features y Target)\n- **X (Variables Independientes):** Población, Cobertura Neta, Cobertura Bruta.\n- **y (Variable Dependiente/Target):** Aprobación."),
    nbf.v4.new_code_cell("# Definir X e y\nX = df[['POBLACIÓN_5_16', 'COBERTURA_NETA', 'COBERTURA_BRUTA']]\ny = df['APROBACIÓN']\n\nprint('Dimensiones de X:', X.shape)\nprint('Dimensiones de y:', y.shape)"),
    
    nbf.v4.new_markdown_cell("### Paso 3: División de Datos (Train y Test)\nSeparamos el 80% de los datos para entrenar el modelo y el 20% para evaluarlo."),
    nbf.v4.new_code_cell("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nprint('Tamaño de entrenamiento:', X_train.shape[0])\nprint('Tamaño de prueba:', X_test.shape[0])"),
    
    nbf.v4.new_markdown_cell("### Paso 4: Escalado de Características\nEstandarizamos las variables para que tengan una media de 0 y varianza de 1, lo cual mejora el rendimiento de la regresión lineal."),
    nbf.v4.new_code_cell("scaler = StandardScaler()\n\n# Ajustamos y transformamos el conjunto de entrenamiento\nX_train_scaled = scaler.fit_transform(X_train)\n\n# Transformamos el conjunto de prueba (NUNCA se ajusta el scaler con los datos de test)\nX_test_scaled = scaler.transform(X_test)"),
    
    nbf.v4.new_markdown_cell("### Paso 5: Entrenamiento del Modelo\nInstanciamos y entrenamos la Regresión Lineal Múltiple."),
    nbf.v4.new_code_cell("modelo_regresion = LinearRegression()\n\n# Entrenamos el modelo con los datos de entrenamiento escalados\nmodelo_regresion.fit(X_train_scaled, y_train)\nprint('Modelo entrenado correctamente.')"),
    
    nbf.v4.new_markdown_cell("### Paso 6: Evaluación del Modelo\nRealizamos predicciones sobre el conjunto de test y evaluamos su precisión usando R² y MSE."),
    nbf.v4.new_code_cell("y_pred = modelo_regresion.predict(X_test_scaled)\n\nmse = mean_squared_error(y_test, y_pred)\nr2 = r2_score(y_test, y_pred)\n\nprint(f'Error Cuadrático Medio (MSE): {mse:.2f}')\nprint(f'Coeficiente de Determinación (R²): {r2:.4f}')"),
    
    nbf.v4.new_markdown_cell("### Paso 7: Exportación del Modelo\nGuardamos el modelo y el escalador en archivos `.pkl` para ser usados en la aplicación de Streamlit."),
    nbf.v4.new_code_cell("# Exportar artefactos usando joblib\njoblib.dump(modelo_regresion, 'modelo_regresion.pkl')\njoblib.dump(scaler, 'scaler.pkl')\n\nprint('Artefactos exportados: modelo_regresion.pkl y scaler.pkl')")
]

# Guardar los cuadernos
with open('01_ETL.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_etl, f)
with open('02_EDA.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_eda, f)
with open('03_Modelo.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_modelo, f)

print("Cuadernos generados exitosamente.")

# ==========================================
# GENERAR CSV LIMPIO AHORA MISMO
# ==========================================
print("Generando CSV limpio...")
df = pd.read_csv('EDUCACION.csv')
def clean_percentage(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        return float(val.replace('%', '').replace('.', '').replace(',', '.'))
    return val

def clean_population(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        return float(val.replace(',', '').replace('.', ''))
    return val

cols_percentage = ['APROBACIÓN', 'DESERCIÓN', 'REPROBACIÓN', 'COBERTURA_NETA', 'COBERTURA_BRUTA']

for col in cols_percentage:
    if col in df.columns:
        df[col] = df[col].apply(clean_percentage)

if 'POBLACIÓN_5_16' in df.columns:
    df['POBLACIÓN_5_16'] = df['POBLACIÓN_5_16'].apply(clean_population)

cols_all = cols_percentage + ['POBLACIÓN_5_16']
df[cols_all] = df[cols_all].fillna(df[cols_all].median())
df.to_csv('EDUCACION_limpio.csv', index=False)
print("EDUCACION_limpio.csv generado exitosamente.")

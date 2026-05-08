import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import os

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Educativo y Predictivo - Colombia",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos de Seaborn
sns.set_theme(style="whitegrid")

# Cargar Datos Limpios
@st.cache_data
def load_data():
    if os.path.exists('EDUCACION_limpio.csv'):
        df = pd.read_csv('EDUCACION_limpio.csv')
        return df
    return None

df = load_data()

# Entrenamiento en tiempo real para asegurar que siempre funcione
@st.cache_resource
def train_models(data):
    # Variables predictoras
    X = data[['POBLACIÓN_5_16', 'COBERTURA_NETA', 'COBERTURA_BRUTA']].fillna(0)
    
    # Target Lineal (Aprobación)
    y_lin = data['APROBACIÓN'].fillna(0)
    
    # Target Logístico (Deserción Alta > 5% = 1, Baja <= 5% = 0)
    y_log = (data['DESERCIÓN'].fillna(0) > 5.0).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modelo Lineal
    lin_model = LinearRegression()
    lin_model.fit(X_scaled, y_lin)
    
    # Modelo Logístico
    log_model = LogisticRegression(class_weight='balanced')
    log_model.fit(X_scaled, y_log)
    
    return scaler, lin_model, log_model

if df is not None:
    scaler, lin_model, log_model = train_models(df)

# Panel de Navegación Lateral
st.sidebar.image("https://images.unsplash.com/photo-1503676260728-1c00da094a0b?auto=format&fit=crop&w=300", use_container_width=True)
st.sidebar.title("🧭 Panel de Navegación")
opcion = st.sidebar.radio(
    "Selecciona una opción:",
    ["🏠 Inicio", "📊 Dashboard y EDA", "📈 Modelo: Regresión Lineal", "🔮 Modelo: Regresión Logística", "📝 Quiz Interactivo"]
)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard Integral desarrollado para **TalentoTech** 🚀")

# ----------------------------------------
# 1. INICIO
# ----------------------------------------
if opcion == "🏠 Inicio":
    st.title("🎓 Predicción y Análisis de la Educación en Colombia")
    st.markdown("""
    Bienvenido al **Dashboard Integral Predictivo**. Aquí podrás explorar cómo se comporta el sistema educativo a nivel municipal, analizar variables mediante filtros y utilizar modelos de Machine Learning avanzados para proyectar la aprobación y deserción escolar.
    
    ### Secciones Disponibles:
    - 📊 **Dashboard y EDA:** Analiza gráficamente la información (Caja y Bigotes, Dispersión, Barras) aplicando filtros interactivos.
    - 📈 **Regresión Lineal:** Predice con exactitud porcentual la tasa de *aprobación* estudiantil.
    - 🔮 **Regresión Logística:** Clasifica si un municipio tiene *riesgo alto de deserción* basándose en sus características.
    - 📝 **Quiz Interactivo:** Pon a prueba tus conocimientos al finalizar el recorrido.
    """)
    st.success("¡Usa el menú de la izquierda para comenzar a navegar!")

# ----------------------------------------
# 2. DASHBOARD Y EDA
# ----------------------------------------
elif opcion == "📊 Dashboard y EDA":
    st.title("📊 Análisis Exploratorio de Datos (Dashboard)")
    
    if df is not None:
        st.sidebar.subheader("Filtros del Dashboard")
        deptos = df['DEPARTAMENTO'].unique().tolist()
        depto_seleccionado = st.sidebar.multiselect("Filtrar por Departamento:", options=deptos, default=deptos[:3])
        
        # Aplicar filtro
        if depto_seleccionado:
            df_filtered = df[df['DEPARTAMENTO'].isin(depto_seleccionado)]
        else:
            df_filtered = df
            
        st.markdown(f"Mostrando datos para **{len(df_filtered)}** municipios.")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📦 Caja y Bigotes", "📉 Dispersión (Scatter)", "📊 Barras y Top 10", "📋 Datos Crudos"])
        
        with tab1:
            st.subheader("Caja y Bigotes: Distribución por Departamento")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='DEPARTAMENTO', y='APROBACIÓN', data=df_filtered, palette="viridis", ax=ax)
            plt.xticks(rotation=45)
            ax.set_title("Distribución de la Tasa de Aprobación")
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Dispersión: Cobertura Neta vs. Aprobación")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='COBERTURA_NETA', y='APROBACIÓN', hue='DEPARTAMENTO', size='POBLACIÓN_5_16', sizes=(20, 400), data=df_filtered, ax=ax2, alpha=0.7)
            ax2.set_title("Relación entre Cobertura y Aprobación")
            st.pyplot(fig2)
            
        with tab3:
            st.subheader("Top 10 Municipios con Mayor Población Escolar (Filtrados)")
            top10 = df_filtered.nlargest(10, 'POBLACIÓN_5_16')
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='POBLACIÓN_5_16', y='MUNICIPIO', data=top10, palette="magma", ax=ax3)
            ax3.set_title("Población Estudiantil 5-16 años")
            st.pyplot(fig3)
            
        with tab4:
            st.subheader("Vista de Datos Filtrados")
            st.dataframe(df_filtered)
            
    else:
        st.warning("⚠️ No se encontró el archivo `EDUCACION_limpio.csv`. Por favor ejecuta el ETL primero.")

# ----------------------------------------
# 3. REGRESIÓN LINEAL
# ----------------------------------------
elif opcion == "📈 Modelo: Regresión Lineal":
    st.title("📈 Regresión Lineal: Predicción de Aprobación")
    st.markdown("Este modelo estima el porcentaje exacto de **Aprobación** en base a la cobertura y población.")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ingresa los Parámetros")
            pob = st.number_input("Población Estudiantil", min_value=100, max_value=2000000, value=15000, step=1000)
            cob_neta = st.slider("Cobertura Neta (%)", min_value=0.0, max_value=150.0, value=85.0, step=0.5)
            cob_bruta = st.slider("Cobertura Bruta (%)", min_value=0.0, max_value=200.0, value=95.0, step=0.5)
            
            btn_predecir_lin = st.button("🚀 Calcular Aprobación", use_container_width=True)

        with col2:
            st.subheader("Resultado de la Predicción")
            if btn_predecir_lin:
                st.balloons()
                X_in = np.array([[pob, cob_neta, cob_bruta]])
                pred = lin_model.predict(scaler.transform(X_in))[0]
                pred = min(max(pred, 0), 100) # Limitar entre 0 y 100
                
                st.success(f"### 🎉 Aprobación Estimada: **{pred:.2f}%**")
                st.info("La regresión lineal traza una línea de mejor ajuste a través de los datos para inferir esta tendencia continua.")

# ----------------------------------------
# 4. REGRESIÓN LOGÍSTICA
# ----------------------------------------
elif opcion == "🔮 Modelo: Regresión Logística":
    st.title("🔮 Regresión Logística: Clasificador de Riesgo de Deserción")
    st.markdown("Este modelo clasifica si un municipio tiene un **Riesgo Alto de Deserción** (mayor al 5%) o un **Riesgo Bajo**.")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Simular Escenario")
            pob_log = st.number_input("Población Estudiantil", min_value=100, max_value=2000000, value=5000, step=1000, key="plog")
            cob_neta_log = st.slider("Cobertura Neta (%)", min_value=0.0, max_value=150.0, value=60.0, step=0.5, key="cnlog")
            cob_bruta_log = st.slider("Cobertura Bruta (%)", min_value=0.0, max_value=200.0, value=70.0, step=0.5, key="cblog")
            
            btn_predecir_log = st.button("⚠️ Evaluar Riesgo", use_container_width=True)

        with col2:
            st.subheader("Clasificación del Modelo")
            if btn_predecir_log:
                st.snow()
                X_in_log = np.array([[pob_log, cob_neta_log, cob_bruta_log]])
                pred_class = log_model.predict(scaler.transform(X_in_log))[0]
                prob = log_model.predict_proba(scaler.transform(X_in_log))[0]
                
                if pred_class == 1:
                    st.error("### 🚨 RIESGO ALTO DE DESERCIÓN")
                    st.write(f"Probabilidad asignada: **{prob[1]*100:.2f}%**")
                    st.markdown("> Se recomiendan intervenciones urgentes para mejorar la retención de estudiantes.")
                else:
                    st.success("### ✅ RIESGO BAJO DE DESERCIÓN")
                    st.write(f"Probabilidad de estar a salvo: **{prob[0]*100:.2f}%**")
                    st.markdown("> Las condiciones actuales favorecen la permanencia estudiantil.")

# ----------------------------------------
# 5. QUIZ
# ----------------------------------------
elif opcion == "📝 Quiz Interactivo":
    st.title("📝 Quiz Interactivo: Data y Modelos")
    st.markdown("Evalúa tus conocimientos sobre los modelos implementados y la metodología.")
    
    with st.form("quiz_form"):
        q1 = st.radio(
            "1. ¿Para qué se utiliza la Regresión Logística en este Dashboard?",
            ["A) Para predecir un valor continuo como el porcentaje de aprobación.", "B) Para clasificar el riesgo de deserción en categorías (Alto/Bajo).", "C) Para crear gráficos de caja y bigotes."]
        )
        
        q2 = st.radio(
            "2. ¿Cuál es el objetivo principal del análisis exploratorio (EDA)?",
            ["A) Desplegar la aplicación web.", "B) Limpiar y reemplazar valores nulos únicamente.", "C) Descubrir patrones, distribuciones y relaciones entre las variables."]
        )
        
        q3 = st.radio(
            "3. ¿Qué mide el Coeficiente de Determinación (R²) en la Regresión Lineal?",
            ["A) La proporción de varianza de la variable dependiente explicada por el modelo.", "B) El número de datos nulos en el dataset.", "C) El peso computacional del modelo exportado."]
        )
        
        submit_btn = st.form_submit_button("Verificar Respuestas")
        
    if submit_btn:
        puntaje = 0
        if q1.startswith("B"): puntaje += 1
        if q2.startswith("C"): puntaje += 1
        if q3.startswith("A"): puntaje += 1
        
        if puntaje == 3:
            st.success("¡Puntaje Perfecto: 3/3! 🎉 ¡Tienes madera de Data Scientist!")
            st.balloons()
        elif puntaje == 2:
            st.warning("Puntaje: 2/3. ¡Casi perfecto! Repasa las diferencias entre regresión lineal y logística.")
        else:
            st.error(f"Puntaje: {puntaje}/3. ¡Sigue practicando! Lee la documentación del proyecto.")

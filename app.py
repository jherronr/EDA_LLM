# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Título de la aplicación
st.title('📊 Herramienta de Análisis Exploratorio de Datos (EDA)')
st.write('Sube un archivo CSV para generar un análisis completo y visualizaciones.')

# Carga de archivo
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado exitosamente.")
    
    # ----------------------------------------------------
    # Parte 1: Resumen y Estadísticas
    # ----------------------------------------------------
    st.header('🔍 Resumen de los Datos')
    
    st.subheader('Información General')
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.subheader('Primeras 5 Filas')
    st.dataframe(df.head())
    
    st.subheader('Estadísticas Descriptivas')
    st.dataframe(df.describe().T)
    
    st.subheader('Valores Nulos y Duplicados')
    st.write("Conteo de valores nulos por columna:")
    st.dataframe(df.isnull().sum())
    st.write(f"Total de filas duplicadas: **{df.duplicated().sum()}**")
    
    # ----------------------------------------------------
    # Parte 2: Visualizaciones del EDA
    # ----------------------------------------------------
    st.header('📈 Visualizaciones de los Datos')
    
    # Obtener columnas numéricas y categóricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Visualizaciones numéricas
    if not numeric_cols.empty:
        st.subheader('Distribución de Variables Numéricas')
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Histograma de {col}')
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(10, 2))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f'Boxplot de {col}')
            st.pyplot(fig)
    
    # Visualizaciones categóricas
    if not cat_cols.empty:
        st.subheader('Distribución de Variables Categóricas')
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
            ax.set_title(f'Conteo de {col}')
            st.pyplot(fig)
    
    # Matriz de Correlación
    if len(numeric_cols) > 1:
        st.subheader('Matriz de Correlación')
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Matriz de Correlación entre Variables Numéricas')
        st.pyplot(fig)
    
    st.balloons()


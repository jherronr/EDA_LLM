# eda_script.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sys

# Configuración visual
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_data(file_path):
    """Carga el archivo CSV y retorna un DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print(f"\n✅ Archivo cargado correctamente: {file_path}")
        print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        return df
    except Exception as e:
        print(f"Error cargando el archivo: {e}")
        sys.exit(1)

def basic_info(df):
    """Muestra información básica del DataFrame"""
    print("\n📌 Información general:")
    print(df.info())
    print("\nPrimeras 5 filas:")
    print(df.head())
    print("\nTipos de datos:")
    print(df.dtypes)

def missing_and_duplicates(df):
    """Revisa valores nulos y duplicados"""
    print("\n🔍 Valores nulos por columna:")
    print(df.isnull().sum())
    print(f"\nTotal de duplicados: {df.duplicated().sum()}")

def descriptive_stats(df):
    """Estadísticas descriptivas"""
    print("\n📊 Estadísticas descriptivas (variables numéricas):")
    print(df.describe().T)
    print("\n📊 Estadísticas descriptivas (variables categóricas):")
    print(df.describe(include=['object']).T)

def detect_outliers(df):
    """Detecta outliers usando Z-score"""
    print("\n🚨 Detección de outliers (Z-score > 3):")
    numeric_cols = df.select_dtypes(include=np.number).columns
    outliers = {}
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers[col] = (z_scores > 3).sum()
    print(outliers)

def visualize_data(df):
    """Genera visualizaciones relevantes para EDA"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    # Histograma variables numéricas
    df[numeric_cols].hist(bins=30, figsize=(15, 10), color='skyblue')
    plt.suptitle("Distribución de variables numéricas")
    plt.show()

    # Boxplot variables numéricas
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df[col], color='orange')
        plt.title(f"Boxplot de {col}")
        plt.show()

    # Heatmap de correlación
    plt.figure(figsize=(10, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Matriz de Correlación")
    plt.show()

    # Conteo de variables categóricas
    for col in cat_cols:
        plt.figure()
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Distribución de {col}")
        plt.show()

    # Pairplot (solo si no hay demasiadas columnas)
    if len(numeric_cols) <= 5:
        sns.pairplot(df[numeric_cols])
        plt.suptitle("Relaciones entre variables numéricas", y=1.02)
        plt.show()

def run_eda(file_path):
    """Ejecuta todo el flujo de EDA"""
    df = load_data(file_path)
    basic_info(df)
    missing_and_duplicates(df)
    descriptive_stats(df)
    detect_outliers(df)
    visualize_data(df)
    print("\n✅ EDA completado.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python eda_script.py <ruta_del_archivo.csv>")
    else:
        run_eda(sys.argv[1])

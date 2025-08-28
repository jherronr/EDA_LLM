import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# --- CONFIGURACIÓN DE LA PÁGINA Y ESTILOS ---
st.set_page_config(page_title="Análisis de Datos con LLM", layout="wide", initial_sidebar_state="collapsed")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title('📊 Herramienta de Análisis de Datos y Asistente LLM')
st.write('Sube un archivo CSV para generar un análisis exploratorio completo y obtener una conclusión generada por un LLM.')

# --- INICIALIZACIÓN DEL ESTADO DE LA SESIÓN ---
# Esto es crucial para mantener los datos y mensajes entre interacciones.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "eda_summary" not in st.session_state:
    st.session_state.eda_summary = None
if "llm_conclusion" not in st.session_state:
    st.session_state.llm_conclusion = None

# --- FUNCIONES AUXILIARES ---

def generate_eda_summary(df: pd.DataFrame) -> str:
    """
    Genera un resumen textual del Análisis Exploratorio de Datos (EDA).
    Este resumen será el contexto principal para el LLM.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    summary = f"""
    ### Resumen del Análisis Exploratorio de Datos (EDA)

    #### 1. Información General y Tipos de Datos:
    ```
    {info_str}
    ```

    #### 2. Estadísticas Descriptivas (Variables Numéricas):
    ```
    {df.describe().T.to_string()}
    ```

    #### 3. Conteo de Valores Nulos por Columna:
    ```
    {df.isnull().sum().to_string()}
    ```

    #### 4. Total de Filas Duplicadas:
    **{df.duplicated().sum()}**

    """
    if len(numeric_cols) > 1:
        summary += f"""
    #### 5. Matriz de Correlación (Variables Numéricas):
    ```
    {df[numeric_cols].corr().to_string()}
    ```
    """
    return summary

def get_llm_conclusion(eda_summary: str) -> str:
    """
    Llama al LLM para generar una conclusión basada en el resumen del EDA.
    """
    try:
        if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
            return "⚠️ **Advertencia:** La clave de API de Groq no está configurada. No se puede generar la conclusión."

        llm = ChatGroq(
            temperature=0.1,
            groq_api_key=os.environ["GROQ_API_KEY"],
            model_name="gemma2-9b-it"
        )
        
        system_prompt = (
            "Eres un analista de datos experto y comunicador excepcional. "
            "Tu tarea es analizar el siguiente resumen de un Análisis Exploratorio de Datos (EDA) y redactar una conclusión clara y concisa en español. "
            "Tu conclusión debe ser fácil de entender para una audiencia no técnica. "
            "**Estructura tu respuesta de la siguiente manera:**\n"
            "1.  **Calidad de los Datos:** Comenta sobre valores nulos, duplicados y tipos de datos. ¿El dataset está limpio o requiere preparación?\n"
            "2.  **Principales Hallazgos:** Describe las tendencias, patrones o distribuciones más importantes que observas en las estadísticas.\n"
            "3.  **Correlaciones Relevantes:** Si hay una matriz de correlación, menciona las correlaciones fuertes (positivas o negativas) y lo que podrían significar.\n"
            "4.  **Recomendaciones:** Basado en el análisis, sugiere los siguientes pasos o áreas de interés para una investigación más profunda.\n\n"
            "**IMPORTANTE:** Basa tu conclusión ÚNICAMENTE en el resumen del EDA proporcionado o la base de datos entregada por el usuario. No inventes información. No generes código. No respondas temas que no estén directamente relacionados con la base de datos entregada."
        )
        
        human_prompt = "Aquí está el resumen del EDA:\n\n---\n\n{eda_summary}\n\n---\n\nPor favor, genera la conclusión."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        llm_chain = LLMChain(prompt=prompt_template, llm=llm)
        response = llm_chain.invoke({"eda_summary": eda_summary})['text']
        return response

    except Exception as e:
        return f"❌ Ocurrió un error al contactar al modelo LLM: {e}"

# --- CARGA DEL ARCHIVO ---
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None and st.session_state.df is None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Archivo cargado exitosamente. Generando análisis...")
        
        # Generar y guardar el resumen del EDA y la conclusión del LLM una sola vez.
        st.session_state.eda_summary = generate_eda_summary(df)
        with st.spinner('🤖 El asistente LLM está analizando los datos para generar una conclusión...'):
            st.session_state.llm_conclusion = get_llm_conclusion(st.session_state.eda_summary)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
        st.session_state.df = None # Resetear en caso de error

# --- VISUALIZACIÓN DEL ANÁLISIS (si el df existe) ---
if st.session_state.df is not None:
    df = st.session_state.df
    
    # --- Pestañas para organizar el contenido ---
    tab1, tab2, tab3 = st.tabs(["📄 Resumen y Estadísticas", "📈 Visualizaciones", "🤖 Conclusión y Chat"])

    with tab1:
        st.header('🔍 Resumen de los Datos')
        
        st.subheader('Información General de las Columnas')
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader('Primeras 5 Filas')
        st.dataframe(df.head())
        
        st.subheader('Estadísticas Descriptivas')
        st.dataframe(df.describe().T)
        
        st.subheader('Valores Nulos y Duplicados')
        col1, col2 = st.columns(2)
        with col1:
            st.write("Conteo de valores nulos:")
            st.dataframe(df.isnull().sum())
        with col2:
            st.write(f"Total de filas duplicadas:")
            st.metric("Duplicados", df.duplicated().sum())

    with tab2:
        st.header('📈 Visualizaciones del EDA')
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include='object').columns
        
        if not numeric_cols.empty:
            st.subheader('Distribución de Variables Numéricas')
            for col in numeric_cols:
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                sns.histplot(df[col], kde=True, ax=ax[0])
                ax[0].set_title(f'Histograma de {col}')
                sns.boxplot(x=df[col], ax=ax[1])
                ax[1].set_title(f'Boxplot de {col}')
                st.pyplot(fig)

        if not cat_cols.empty:
            st.subheader('Distribución de Variables Categóricas')
            for col in cat_cols:
                fig, ax = plt.subplots()
                sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
                ax.set_title(f'Conteo de {col}')
                st.pyplot(fig)
        
        if len(numeric_cols) > 1:
            st.subheader('Matriz de Correlación')
            fig, ax = plt.subplots()
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Matriz de Correlación')
            st.pyplot(fig)

    with tab3:
        st.header('🤖 Conclusión del Asistente LLM')
        if st.session_state.llm_conclusion:
            st.markdown(st.session_state.llm_conclusion)
        else:
            st.info("La conclusión del LLM aparecerá aquí después de cargar los datos.")

        st.markdown("---")
        st.header('💬 Chatea sobre tus Datos')
        st.write('Haz preguntas específicas sobre el análisis o los datos.')

        # Mostrar mensajes del chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input del usuario para el chat
        if prompt := st.chat_input("Ej: ¿Cuál es la correlación entre la columna A y B?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
                            st.warning("La clave de API de Groq no está configurada.")
                            continue

                        llm = ChatGroq(temperature=0, model_name="gemma2-9b-it")
                        
                        system_prompt_qa = (
                            "Eres un asistente de análisis de datos. Se te ha proporcionado un resumen del EDA y el dataset completo. "
                            "Tu tarea es responder preguntas específicas del usuario sobre los datos. "
                            "Utiliza la información del resumen y del dataset para formular tus respuestas. "
                            "No respondas preguntas que estén por fuera de la base de datos o que no sean del tema de la base de datos"
                            "Sé directo y conciso. No intentes generar gráficos ni código.\n\n"
                            f"--- RESUMEN DEL EDA ---\n{st.session_state.eda_summary}\n\n"
                            f"--- DATASET (primeras 100 filas) ---\n{df.to_string()}\n---"
                        )
                        
                        prompt_template_qa = ChatPromptTemplate.from_messages([
                            ("system", system_prompt_qa),
                            ("human", "{user_question}")
                        ])
                        
                        llm_chain = LLMChain(prompt=prompt_template_qa, llm=llm)
                        response = llm_chain.invoke({"user_question": prompt})['text']
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        st.error(f"Ocurrió un error al llamar al LLM: {e}")


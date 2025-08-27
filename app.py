
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# Importar las clases necesarias de LangChain desde la ubicación correcta
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

# Configuración visual de los gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Configurar el título y la descripción de la aplicación
st.title('📊 Herramienta de Análisis de Datos y Asistente LLM')
st.write('Sube un archivo CSV para generar un análisis completo y chatear con un LLM sobre tus datos.')

# Inicializar el historial del chat en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "df" not in st.session_state:
    st.session_state.df = None

# Aceptar la entrada del archivo por el usuario
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
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
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include='object').columns
        
        if not numeric_cols.empty:
            st.subheader('Distribución de Variables Numéricas')
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Histograma de {col}')
                st.pyplot(fig)
                
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f'Boxplot de {col}')
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
            ax.set_title('Matriz de Correlación entre Variables Numéricas')
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
        st.stop()


# ----------------------------------------------------
# Parte 3: Integración del LLM
# ----------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df

    st.header('🤖 Asistente LLM sobre tus Datos')
    st.write('Pregunta lo que quieras sobre el análisis de datos que acabas de ver.')
    
    # Mostrar los mensajes anteriores del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capturar los insights del EDA dinámicamente
    file_name = uploaded_file.name
    num_rows, num_cols = df.shape
    columns_info = df.dtypes.to_dict()
    descriptive_stats_str = df.describe().T.to_string()
    missing_data_info = df.isnull().sum().to_dict()
    categorical_info = {col: df[col].nunique() for col in df.select_dtypes(include='object').columns}

    # Crear el prompt del sistema con los insights
    system_prompt = (
        "Eres un analista de datos experto. Tienes acceso a un dataset con las siguientes características:\n"
        f"- Nombre del archivo: {file_name}\n"
        f"- Dimensiones: {num_rows} filas, {num_cols} columnas\n"
        f"- Columnas y Tipos: {columns_info}\n"
        f"- Estadísticas Descriptivas:\n{descriptive_stats_str}\n"
        f"- Valores Nulos: {missing_data_info}\n"
        f"- Cardinalidad Categórica: {categorical_info}\n\n"
        "Tu tarea es responder preguntas de los usuarios sobre este dataset, usando la información proporcionada. "
        "Sé conciso y ve al punto. No inventes información."
    )
    
    # Aceptar la entrada del usuario
    if prompt := st.chat_input("Pregunta sobre tu dataset..."):
        # Añadir el mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Aquí se configura y llama al modelo de Groq
            try:
                # Se obtiene la clave de API de las variables de entorno o de los "secrets" de Streamlit Cloud
                if "GROQ_API_KEY" in os.environ:
                    llm = ChatGroq(
                        temperature=0,
                        groq_api_key=os.environ["GROQ_API_KEY"],
                        model_name="mixtral-8x7b-32768"
                    )
                    
                    # Se crea la cadena con el prompt y el LLM
                    prompt_template = ChatPromptTemplate.from_messages(
                        [("system", system_prompt), ("human", "{user_question}")]
                    )
                    
                    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
                    
                    # Llamada al modelo con el prompt completo
                    # La corrección está en cómo se pasan las variables al .invoke()
                    response = llm_chain.invoke({"user_question": prompt})['text']
                    
                    st.markdown(response)
                    
                    # Añadir la respuesta del asistente al historial
                    st.session_state.messages.append({"role": "assistant", "content": response})

                else:
                    st.warning("¡Advertencia! La clave de API de Groq no está configurada.")
                    st.info("Para que el LLM funcione, debes configurar tu clave de API en los 'secrets' de Streamlit Cloud con el nombre 'GROQ_API_KEY'.")
                    st.warning("Respuesta simulada del LLM para demostración.")
                    
                    # Simular la respuesta del LLM para la demostración
                    if "promedio" in prompt.lower():
                        response = f"El monto promedio de las transacciones es de ${df['Amount'].mean():.2f}."
                    elif "categoría" in prompt.lower():
                        response = f"La categoría de gasto más común es '{df['Category'].value_counts().idxmax()}'."
                    else:
                        response = "No puedo responder a esa pregunta con los datos disponibles."
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Ocurrió un error al llamar al modelo LLM: {e}")
                st.warning("Asegúrate de que la clave de API es válida y el modelo está disponible.")



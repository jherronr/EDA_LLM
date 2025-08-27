
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

# Configuración visual de los gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Configurar el título y la descripción de la aplicación
st.title('📊 Herramienta de Análisis de Datos y Asistente LLM')
st.write('Sube un archivo CSV para generar un análisis completo y chatear con un LLM sobre tus datos.')

# Inicializar el historial del chat, el DataFrame y la conclusión en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "df" not in st.session_state:
    st.session_state.df = None

if "initial_conclusion" not in st.session_state:
    st.session_state.initial_conclusion = None

# Aceptar la entrada del archivo por el usuario
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None and st.session_state.df is None:
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
            
        # ----------------------------------------------------
        # Parte 3: Generar Conclusión Inicial del LLM
        # ----------------------------------------------------
        # Obtener la clave de API de las variables de entorno o de los "secrets" de Streamlit Cloud
        if "GROQ_API_KEY" in os.environ:
            with st.spinner('Generando conclusión inicial sobre el dataset...'):
                df_string = df.to_string(index=False)
                system_prompt_initial = (
                    "Eres un analista de datos experto. Tienes un dataset en formato de texto a continuación. "
                    "Analiza el dataset, extrae los insights más relevantes y proporciona una conclusión concisa y directa. "
                    "No incluyas nada más en tu respuesta. "
                    "Dataset:\n"
                    "```\n"
                    f"{df_string}\n"
                    "```"
                )
                
                llm = ChatGroq(
                    temperature=0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                    model_name="gemma2-9b-it"
                )
                
                initial_prompt_template = ChatPromptTemplate.from_messages(
                    [("system", system_prompt_initial)]
                )
                
                initial_llm_chain = LLMChain(prompt=initial_prompt_template, llm=llm)
                
                st.session_state.initial_conclusion = initial_llm_chain.invoke({})['text']
        else:
            st.session_state.initial_conclusion = "No se pudo generar la conclusión. Por favor, configura tu clave de API de Groq en los 'secrets' de Streamlit Cloud."

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
        st.stop()
        
if st.session_state.df is not None:
    # ----------------------------------------------------
    # Sección de Conclusión
    # ----------------------------------------------------
    st.markdown("---")
    st.header('💡 Conclusión del Análisis de Datos')
    if st.session_state.initial_conclusion:
        st.info(st.session_state.initial_conclusion)

    # ----------------------------------------------------
    # Sección de Chat Interactivo
    # ----------------------------------------------------
    st.markdown("---")
    st.header('🤖 Asistente LLM sobre tus Datos')
    st.write('Pregunta lo que quieras sobre el análisis de datos que acabas de ver.')
    
    # Mostrar los mensajes anteriores del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    df_string = st.session_state.df.to_string(index=False)
    
    # Crear el prompt del sistema para las preguntas interactivas
    system_prompt_qa = (
        "Eres un analista de datos experto. Tienes un dataset en formato de texto a continuación. "
        "Analiza el dataset y responde a las preguntas del usuario. "
        "Sé conciso y ve al punto. No inventes información. "
        "Dataset:\n"
        "```\n"
        f"{df_string}\n"
        "```"
    )

    # Aceptar la entrada del usuario
    if prompt := st.chat_input("Pregunta sobre tu dataset..."):
        # Añadir el mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                if "GROQ_API_KEY" in os.environ:
                    llm = ChatGroq(
                        temperature=0,
                        groq_api_key=os.environ["GROQ_API_KEY"],
                        model_name="gemma2-9b-it"
                    )
                    
                    prompt_template_qa = ChatPromptTemplate.from_messages(
                        [("system", system_prompt_qa), ("human", "{user_question}")]
                    )
                    
                    llm_chain = LLMChain(prompt=prompt_template_qa, llm=llm)
                    
                    response = llm_chain.invoke({"user_question": prompt})['text']
                    
                    st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})

                else:
                    st.warning("¡Advertencia! La clave de API de Groq no está configurada.")
                    st.info("Para que el LLM funcione, debes configurar tu clave de API en los 'secrets' de Streamlit Cloud con el nombre 'GROQ_API_KEY'.")
                    st.warning("Respuesta simulada del LLM para demostración.")
                    
                    if "promedio" in prompt.lower():
                        if "Amount" in st.session_state.df.columns:
                            response = f"El monto promedio de las transacciones es de ${st.session_state.df['Amount'].mean():.2f}."
                        else:
                            response = "No se encontró la columna 'Amount'."
                    else:
                        response = "No puedo responder a esa pregunta con los datos disponibles."
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Ocurrió un error al llamar al modelo LLM: {e}")
                st.warning("Asegúrate de que la clave de API es válida y el modelo está disponible.")

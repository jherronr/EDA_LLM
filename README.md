📊 Asistente de Análisis de Datos con LLM

Una aplicación web interactiva construida con Streamlit que automatiza el Análisis Exploratorio de Datos (EDA) de cualquier archivo CSV y permite a los usuarios obtener conclusiones e interactuar con un asistente inteligente basado en un Modelo de Lenguaje Grande (LLM) a través de la API de Groq.

📜 Descripción General

Esta herramienta está diseñada para acelerar el proceso de análisis de datos. Simplemente sube un archivo CSV y la aplicación generará automáticamente un completo reporte de EDA, que incluye:

    Resumen de Datos: Información general, tipos de datos, y una vista previa de las primeras filas.

    Estadísticas Descriptivas: Métricas clave como media, mediana, desviación estándar, etc.

    Calidad de los Datos: Análisis de valores nulos y filas duplicadas.

    Visualizaciones: Histogramas, boxplots para variables numéricas y gráficos de barras para variables categóricas.

    Análisis de Correlación: Una matriz de correlación para identificar relaciones entre variables numéricas.

Además del EDA, la aplicación utiliza un LLM (a través de LangChain y Groq) para:

    Generar una Conclusión Automática: El LLM analiza el resumen del EDA y proporciona una conclusión experta sobre la calidad de los datos, los principales hallazgos y las recomendaciones.

    Chat Interactivo: Permite a los usuarios hacer preguntas específicas en lenguaje natural sobre sus datos y recibir respuestas instantáneas del asistente.

✨ Características Principales

    Carga Fácil de Datos: Interfaz simple para subir archivos CSV.

    EDA Automatizado: Generación instantánea de estadísticas y visualizaciones clave.

    Conclusiones con IA: Un LLM interpreta el EDA y genera un resumen ejecutivo.

    Chat Inteligente: Resuelve dudas específicas sobre los datos de forma conversacional.

    Interfaz Intuitiva: Organizada en pestañas para una navegación clara y sencilla.

    Despliegue Sencillo: Lista para ser desplegada en plataformas como Streamlit Community Cloud.

🛠️ Tecnologías Utilizadas

    Framework: Streamlit

    Análisis de Datos: Pandas

    Visualización: Matplotlib y Seaborn

    Orquestación de LLM: LangChain

    Modelo de Lenguaje: Groq API (usando los modelos gemma2-9b-it y llama3-70b-8192)

🚀 Instalación y Uso Local

Sigue estos pasos para ejecutar la aplicación en tu máquina local.
Prerrequisitos

    Python 3.8 o superior

    Una clave de API de Groq

Pasos

    Clona el repositorio:

    git clone https://github.com/tu-usuario/tu-repositorio.git
    cd tu-repositorio

    Crea un entorno virtual (recomendado):

    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate

    Instala las dependencias:

    pip install -r requirements.txt

    Configura tu clave de API:
    Crea un archivo llamado .env en la raíz del proyecto y añade tu clave de API de Groq:

    GROQ_API_KEY="tu_clave_secreta_aqui"

    Alternativamente, puedes configurarla como una variable de entorno del sistema.

    Ejecuta la aplicación:

    streamlit run app.py

¡Abre tu navegador en http://localhost:8501 y comienza a analizar tus datos!
🤝 Contribuciones

Las contribuciones son bienvenidas. Si tienes ideas para mejorar la aplicación, por favor abre un issue para discutirlo o envía un pull request.
📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

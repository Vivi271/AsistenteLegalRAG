# ⚖️ Asistente Legal Inteligente (RAG)

Este proyecto es un asistente experto diseñado para analizar reglamentos, contratos y leyes utilizando la arquitectura **RAG (Retrieval-Augmented Generation)**. La aplicación permite cargar documentos en formato PDF o texto plano y realizar consultas legales precisas basadas exclusivamente en el conocimiento proporcionado, mitigando alucinaciones y garantizando trazabilidad legal.

**Desarrollado por:**
*   **Viviana García** - [Konrad Lorenz]
*   **Braian Ramirez** - [Konrad Lorenz]

---

## 🏗️ Arquitectura del Sistema (RAG Flow - Avance 2)

El flujo de información se implementó utilizando LangChain y ChromaDB para operar de manera local y privada, garantizando la precisión de las respuestas:

1. **Carga de Documentos:** Se extrae el texto de archivos PDF (Reglamentos, manuales o leyes) usando `PyPDFLoader`.
2. **División en Fragmentos (Chunking):** El texto extraído se divide en pequeños fragmentos de 500 caracteres (con un solapamiento de 50) usando `RecursiveCharacterTextSplitter`.
3. **Vectorización (Embeddings):** Cada fragmento se convierte en un vector numérico multidimensional utilizando el modelo `gemini-embedding-001`.
4. **Almacenamiento Local:** Estos vectores se guardan en una base de datos vectorial local basada en **ChromaDB**.
5. **Recuperación (Retrieval):** Ante una consulta del usuario, el sistema vectoriza la pregunta y extrae de ChromaDB los `k=5` fragmentos semánticamente más similares.
6. **Prompt Aumentado:** El sistema construye un prompt estricto donde los fragmentos recuperados se inyectan como el *único* contexto permitido.
7. **Generación Segura:** El LLM (`gemini-3.1-flash-lite-preview` a temperatura 0.0) procesa la consulta usando exclusivamente el contexto inyectado y un formato de salida estructurado, eliminando alucinaciones.
8. **Evaluación (RAGAS):** El desempeño general se mide evaluando `faithfulness`, `answer_relevancy` y `context_precision` usando el framework RAGAS.

---

## 🛠️ Tecnologías Utilizadas

*   **Core:** Python 3.9+
*   **IA:** Google Gemini 3 Flash (vía SDK `google-genai`)
*   **Frontend:** Streamlit (Custom CSS para diseño premium)
*   **Documentación:** `pypdf` (Motor de extracción de PDF)
*   **Entorno:** `python-dotenv` para gestión de API Keys seguras.

---

## 🚀 Guía de Instalación y Uso

### 1. Preparación del Entorno
```bash
# Crear entorno virtual
python3 -m venv env
source env/bin/activate  # MacOS/Linux
# .\env\Scripts\activate # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración
Crea un archivo `.env` en la raíz con tu clave de API:
```env
GEMINI_API_KEY=tu_clave_aqui
```

### 3. Ejecución
```bash
streamlit run app.py
```

---

## 🧠 Ingeniería de Prompts (Requisitos Académicos)

Este sistema implementa las 4 estrategias clave de Prompt Engineering exigidas:

1.  **System Instruction:** Se define una "Identidad de Sistema" que obliga al modelo a actuar como un jurista experto y prohíbe el uso de conocimiento externo al documento.
2.  **Few-Shot Prompting:** Se inyectan ejemplos de entrenamiento rápido en el prompt para asegurar que la IA aprenda el formato de análisis y la estructura JSON en un solo paso.
3.  **Delimitadores XML:** Se utilizan tags `<contexto>` y `<consulta>` para jerarquizar la información y evitar confusiones en el modelo durante el procesamiento de textos largos.
4.  **Structured Output (JSON):** Se fuerza al modelo a responder exclusivamente en JSON técnico (`response_mime_type`), lo que permite que la interfaz muestre alertas de colores y tarjetas de riesgo automáticamente.

---

## 🌟 Características Destacadas (Actualizado)

*   ✅ **Soporte PDF:** Carga de reglamentos directamente desde archivos PDF.
*   ✅ **Diseño Premium:** Interfaz con tarjetas de reporte visuales y colores semánticos (Verde=Válido, Rojo=Inválido).
*   ✅ **Exportación:** Generación automática de reportes de caso en formato texto descargable.
*   ✅ **Seguridad Jurídica:** Clasificación de niveles de riesgo (Bajo, Medio, Alto) en cada análisis.

---

## 📝 Notas de Versión
*   **v2.0 (Avance 2):** Implementación de almacenamiento local con base de datos vectorial ChromaDB, pipeline completo (embeddings, chunking) y cuaderno de evaluación mediante el framework RAGAS.
*   **v1.0 (Avance 1):** Implementación de arquitectura inicial, motor de lectura PDF y estrategias de Prompting (Few-Shot, System Instructions).

---
© 2026 - Proyecto Académico Konrad Lorenz

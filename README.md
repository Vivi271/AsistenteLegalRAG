# 🧠 Consultor Especialista en Neuroanatomía (RAG)

Sistema de IA que actúa como consultor científico especializado en neuroanatomía. Responde preguntas basándose **exclusivamente** en los artículos científicos cargados como base de conocimientos, usando la arquitectura **RAG (Retrieval-Augmented Generation)** con almacenamiento vectorial local.

**Desarrollado por:**
*   **Viviana García** — Konrad Lorenz
*   **Braian Ramirez** — Konrad Lorenz

---

## 🏗️ Arquitectura del Sistema — Flujo RAG Completo

El pipeline procesa los documentos científicos en **8 pasos** encadenados:

```
📄 PDFs Científicos (Neuroanatomía)
 │
 ▼
 ✂️  PASO 1 — Carga de documentos (PyPDFLoader)
 │
 ▼
 🧩 PASO 2 — División en fragmentos (Chunking)
 │           chunk_size=600 · overlap=80
 ▼
 🔢 PASO 3 — Generación de Embeddings
 │           modelo: gemini-embedding-001
 ▼
 🗄️  PASO 4 — Base Vectorial local (ChromaDB)
 │           similitud coseno · persistente en disco
 ▼
 ❓  Consulta del usuario
 │
 ▼
 🔍 PASO 5 — Recuperación vectorial (k=5 fragmentos)
 │
 ▼
 📝 PASO 6 — Construcción del Prompt Aumentado
 │           contexto inyectado con delimitadores XML
 ▼
 🤖 PASO 7 — LLM generador (gemini-1.5-flash · T=0.1)
 │           respuesta anclada al contexto
 ▼
 📊 PASO 8 — Evaluación con RAGAS
             faithfulness · answer_relevancy · context_precision
```

---

## 🗂️ Estructura del Proyecto

```
AsistenteLegalRAG/
│
├── app.py                      # Interfaz Streamlit del consultor
├── asistente_legal.py          # Pipeline RAG (chunks, embeddings, ChromaDB, consulta)
│
├── 0717-9502-ijmorphol-41-04-996.pdf   # Artículo 1 — International J. of Morphology
├── SCT_2025_1250.pdf                    # Artículo 2 — Surgical & Clinical Trials
├── circir_25_93_2_197-201.pdf           # Artículo 3 — Cirugía y Cirujanos
│
├── chroma_neuro_db/            # Base vectorial local (generada automáticamente)
├── Flujo_RAG_Evaluacion.ipynb  # Notebook de evaluación con RAGAS
├── Evaluacion_RAG.md           # Reporte de métricas de evaluación
├── requirements.txt            # Dependencias del proyecto
└── .env                        # API Key (no subir a GitHub)
```

---

## 🛠️ Tecnologías Utilizadas

| Componente | Tecnología |
|---|---|
| **Lenguaje** | Python 3.9+ |
| **Frontend** | Streamlit (CSS personalizado dark-mode) |
| **LLM generador** | Google Gemini 1.5 Flash (`langchain-google-genai`) |
| **Embeddings** | Google `embedding-001` |
| **Base vectorial** | ChromaDB (almacenamiento local persistente) |
| **Pipeline RAG** | LangChain + LangChain-Chroma |
| **Lector de PDFs** | PyPDFLoader (`langchain-community`) |
| **Evaluación** | RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`) |
| **Variables de entorno** | `python-dotenv` |

---

## 🧠 Ingeniería de Prompts Implementada

El sistema implementa **4 estrategias clave** de Prompt Engineering:

1. **System Instruction:** Identidad de consultor científico con restricción estricta de no usar conocimiento externo al contexto.
2. **Few-Shot Prompting:** Ejemplos de análisis inyectados en el prompt del sistema para guiar el formato de respuesta científica.
3. **Delimitadores XML:** Tags `<contexto>` y `<pregunta>` para jerarquizar claramente la información recuperada vs. la consulta del usuario.
4. **Structured Retrieval (RAG):** Los fragmentos más relevantes se recuperan semánticamente desde ChromaDB y se inyectan como único contexto válido antes de llamar al LLM.

---

## 🚀 Guía de Instalación y Uso

### 1. Preparación del Entorno
```bash
# Crear entorno virtual
python3 -m venv env
source env/bin/activate   # macOS / Linux
# .\env\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración
Crea o edita el archivo `.env` en la raíz del proyecto:
```env
GEMINI_API_KEY=tu_clave_aqui
```

### 3. Ejecución
```bash
streamlit run app.py
```
> **Primera ejecución:** El sistema vectorizará automáticamente los 3 PDFs y creará la base ChromaDB. Tarda ~1-2 minutos. Las siguientes ejecuciones cargan instantáneamente desde el disco.

---

## 📊 Resultados de Evaluación RAGAS

| Métrica | Promedio |
|---|---|
| `faithfulness` | **1.00** — No se generó ninguna alucinación |
| `answer_relevancy` | **0.79** — Respuestas pertinentes a la consulta |
| `context_precision` | **0.30** — Área de mejora: reducir `chunk_size` |

Ver análisis completo → [`Evaluacion_RAG.md`](./Evaluacion_RAG.md)

---

## 📝 Control de Versiones

| Versión | Tag Git | Descripción |
|---|---|---|
| **v1.0** | `v1.0-asistente-legal` | Asistente Legal RAG con Few-Shot y Structured Output |
| **v2.0** | `main` (actual) | Consultor Neuroanatomía con ChromaDB + RAGAS |

---

© 2026 — Proyecto Académico · Universidad Konrad Lorenz

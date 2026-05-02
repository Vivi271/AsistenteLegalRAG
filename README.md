# 🧠 Consultor Especialista en Neuroanatomía (RAG)

Sistema de IA que actúa como **consultor científico especializado en neuroanatomía**. Responde preguntas basándose **exclusivamente** en los artículos científicos cargados como base de conocimientos, usando la arquitectura **RAG (Retrieval-Augmented Generation)** con almacenamiento vectorial local.

**Desarrollado por:**
- **Viviana García** — Universidad Konrad Lorenz
- **Braian Ramírez** — Universidad Konrad Lorenz

---

## 📌 Descripción del Proyecto

Sistema RAG (Retrieval-Augmented Generation) local que procesa artículos científicos de neuroanatomía en formato PDF, los vectoriza mediante modelos de embeddings de Gemini, y permite realizar consultas en lenguaje natural respondidas exclusivamente con información de los documentos cargados.

**Preservación de privacidad:** La base vectorial se almacena completamente en disco local (ChromaDB con SQLite), sin enviar documentos a servidores externos.

---

## 🏗️ Flujo del Sistema RAG — Implementación Completa

El pipeline procesa documentos y consultas en **8 pasos encadenados**:

```
╔══════════════════════════════════════════════════════════╗
║           FASE 1: INDEXACIÓN (Offline)                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📄 Carpeta Docs/  →  Detección automática de PDFs      ║
║         │                                                ║
║         ▼                                                ║
║  [PASO 1] Carga de Documentos                           ║
║           PyPDFLoader — extrae texto página a página     ║
║         │                                                ║
║         ▼                                                ║
║  [PASO 2] Chunking (División en Fragmentos)             ║
║           RecursiveCharacterTextSplitter                 ║
║           chunk_size=600 · overlap=80 caracteres         ║
║         │                                                ║
║         ▼                                                ║
║  [PASO 3] Generación de Embeddings                      ║
║           Modelo: gemini-embedding-2-preview (3072 dims) ║
║           Procesado en lotes de 50 · pausa anti-429      ║
║         │                                                ║
║         ▼                                                ║
║  [PASO 4] Base Vectorial Local                          ║
║           ChromaDB (SQLite) · similitud coseno           ║
║           Local: chroma_neuro_db/ + backup ~/.neuro_db_permanent/ ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║           FASE 2: CONSULTA (Online/Tiempo Real)         ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  ❓ Pregunta del usuario (lenguaje natural)              ║
║         │                                                ║
║         ▼                                                ║
║  [PASO 5] Recuperación Vectorial (Retrieval)            ║
║           Búsqueda por similitud semántica               ║
║           k fragmentos más relevantes (ajustable 3-8)    ║
║         │                                                ║
║         ▼                                                ║
║  [PASO 6] Construcción del Prompt Aumentado             ║
║           Contexto inyectado con delimitadores XML        ║
║           <contexto> + <pregunta> estructurados          ║
║         │                                                ║
║         ▼                                                ║
║  [PASO 7] Generación con LLM                            ║
║           Gemini (fallback multi-modelo automático)      ║
║           Respuesta anclada al contexto con citas        ║
║         │                                                ║
║         ▼                                                ║
║  [PASO 8] Presentación y Evaluación                     ║
║           Respuesta con fuentes · métricas de uso        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

## 🧠 Ingeniería de Prompts

### System Prompt (Identidad del Consultor)

```python
SYSTEM_INSTRUCTION = """Eres un consultor especialista en neuroanatomía con formación
en investigación científica. Tu misión es responder preguntas EXCLUSIVAMENTE
basándote en la información del <contexto>.

REGLAS ESTRICTAS (de mayor a menor prioridad):
1. Si la <pregunta> es un saludo o NO es una pregunta científica, responde SOLO con:
   "Por favor, formula una pregunta específica sobre el contenido de los artículos."
2. Si la respuesta NO está en el contexto, responde:
   "Esta información no se encuentra en los documentos científicos disponibles."
3. Si la respuesta SÍ está en el contexto:
   - Responde de forma científica y precisa
   - Cita SIEMPRE el artículo y página de donde proviene la información
   - Usa terminología médica apropiada
   - Estructura la respuesta con secciones claras si aplica
"""
```

### Prompt Aumentado (Template RAG)

```python
prompt = f"""
<contexto>
{fragmentos_recuperados}
</contexto>

<pregunta>
{consulta_del_usuario}
</pregunta>

Responde basándote EXCLUSIVAMENTE en el contexto anterior.
Cita el artículo fuente y la página para cada afirmación.
"""
```

### Configuración del Sistema (Generation Config)

```python
config = genai_types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.1,       # Respuestas deterministas y precisas
    max_output_tokens=2048, # Respuestas completas sin truncar
    top_p=0.8,             # Control de diversidad de tokens
)
```

### Formato de Salida

Las respuestas siguen esta estructura obligatoria:
- **Respuesta directa** basada en el contexto recuperado
- **Citas** con formato: `(Título del artículo, p. X)`
- **Advertencia** si la información no está en los documentos

---

## ⚙️ Configuración del Sistema RAG

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `chunk_size` | 600 | Caracteres por fragmento |
| `chunk_overlap` | 80 | Superposición entre fragmentos para contexto |
| `embedding_model` | `gemini-embedding-2-preview` | Modelo de vectorización (3072 dims) |
| `llm_model` | `gemini-flash-latest` + fallback automático | Intenta múltiples modelos si hay 429 |
| `temperature` | 0.1 | Precisión sobre creatividad |
| `k_retrieval` | 5 (ajustable 3-8) | Fragmentos recuperados por consulta |
| `similarity` | Coseno | Métrica de similitud vectorial |
| `vector_db` | ChromaDB (SQLite local) | Base vectorial persistente |
| `batch_size` | 50 | Fragmentos por lote de vectorización |
| `backup_permanente` | `~/.neuro_db_permanent/` | Restauración automática si se borra la DB |

---

## 🗂️ Estructura del Proyecto

```
ConsultorNeuroanatomia/
│
├── app.py                      # Interfaz Streamlit — UI completa
├── rag_pipeline.py             # Pipeline RAG completo (Indexación + Consulta)
├── Flujo_RAG_Evaluacion.ipynb  # Notebook de evaluación del sistema
│
├── Docs/                       # Base de conocimientos (3 artículos PDF)
│   ├── 0717-9502-ijmorphol-41-04-996.pdf   # Regla Simple - Aprendizaje Neuroanatomía
│   ├── SCT_2025_1250.pdf                    # Tecnologías Inmersivas en Enseñanza
│   └── circir_25_93_2_197-201.pdf           # Modelos 3D y Realidad Aumentada
│
├── chroma_neuro_db/            # Base vectorial local — 171 vectores (auto-generada)
├── .streamlit/config.toml      # Configuración de tema y UI
├── requirements.txt            # Dependencias del proyecto
├── .env                        # Variables de entorno (NO incluido en git)
├── .gitignore                  # Exclusiones de control de versiones
└── README.md                   # Este archivo
```

> 💡 **Backup automático:** La DB se guarda también en `~/.neuro_db_permanent/`. Si la carpeta local desaparece, se restaura automáticamente al iniciar sin necesidad de re-vectorizar.

---

## 🚀 Instalación y Ejecución

### Prerrequisitos
- Python 3.11+
- API Key de Google Gemini ([aistudio.google.com](https://aistudio.google.com/apikey))

### Pasos

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd ConsultorNeuroanatomia

# 2. Crear entorno virtual
python3 -m venv env
source env/bin/activate   # macOS/Linux
# env\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API Key
echo "GEMINI_API_KEY=tu_key_aqui" > .env

# 5. Vectorizar documentos (PRIMERA VEZ — tarda ~5 min)
python3 -c "from rag_pipeline import build_vector_store; build_vector_store(force_rebuild=True)"

# 6. Iniciar la aplicación
streamlit run app.py
```

> ⚠️ **Importante:** El paso 5 solo es necesario la primera vez o cuando se agregan nuevos documentos. Desde la app, usa el botón **"Reconstruir VectorDB"** en el panel lateral.

---

## 📊 Evaluación del Sistema

El notebook `Flujo_RAG_Evaluacion.ipynb` contiene:
- Análisis de calidad de fragmentación (chunking)
- Métricas de recuperación (precisión, cobertura)
- Pruebas de coherencia de respuestas
- Ejemplos de consultas con sus respuestas y fuentes

---

## 🔒 Privacidad y Seguridad

- Los documentos PDF **nunca se envían** a servidores externos
- Solo el **texto de los fragmentos** recuperados se envía al LLM (no el PDF completo)
- La base vectorial es **completamente local** (SQLite en disco)
- El archivo `.env` con la API Key está excluido del repositorio via `.gitignore`

---

© 2026 — Proyecto Académico · Universidad Konrad Lorenz

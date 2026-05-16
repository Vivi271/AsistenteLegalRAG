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
| `embedding_model` | `gemini-embedding-001` | Modelo de vectorización (768 dims, soporte multilingüe) |
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

## 🔍 Pruebas de Similitud de Coseno (Búsqueda Semántica)

Para cumplir con el requisito de recuperación exitosa usando lenguaje coloquial o sinónimos, el sistema aprovecha los embeddings de Google para realizar mapeos semánticos avanzados. 

**Ejemplo de Recuperación Exitosa (Lenguaje coloquial → Lenguaje científico):**
- **Consulta del usuario (Coloquial):** *"¿Es útil usar gafas de videojuegos o de compu para estudiar la cabeza o el cerebro?"*
- **Contexto Recuperado por el Sistema:** Fragmentos sobre *"tecnologías inmersivas"*, *"Realidad Virtual (RV)"* y *"modelos 3D anatómicos"*.
- **Resultado:** A través de la similitud de coseno, el motor vectorial entiende que "gafas de videojuegos" y "cabeza" son semánticamente cercanos a "tecnologías inmersivas / RV" y "neuroanatomía" en el contexto cargado, logrando devolver la respuesta correcta y citada basándose en los artículos de innovación académica.

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

## 📊 Informe de Resultados — 10 Preguntas de Evaluación

### Configuración de la Evaluación

| Parámetro | Valor |
|---|---|
| Documentos | 3 artículos peer-reviewed de neuroanatomía |
| Modelo embeddings | `gemini-embedding-001` |
| LLM generador | `gemini-2.0-flash-lite` + fallback automático |
| Chunk size / overlap | 600 / 80 caracteres |
| k (chunks recuperados) | 5 |
| Temperatura | 0.1 |

### Tabla de Resultados

| # | Pregunta | Categoría | Faithfulness | Relevancy | Precision | Estado |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | ¿Qué es la regla simple de neuronas aferentes? | Directa | 1.00 | 0.96 | 0.85 | ✅ Éxito |
| 2 | ¿Características morfológicas de neuronas aferentes? | Directa | 1.00 | 0.93 | 0.80 | ✅ Éxito |
| 3 | ¿Cómo se enseña el SN con realidad virtual? | Semántica | 1.00 | 0.89 | 0.65 | ✅ Éxito |
| 4 | ¿Los modelos 3D ayudan a entender la anatomía cerebral? | Semántica | 1.00 | 0.87 | 0.70 | ✅ Éxito |
| 5 | ¿Ventajas/desventajas tecnologías inmersivas vs. convencionales? | Multi-chunk | 0.92 | 0.91 | 0.55 | ✅ Éxito |
| 6 | ¿Metodología y hallazgos morfológicos de neuronas aferentes? | Multi-chunk | 0.95 | 0.93 | 0.60 | ✅ Éxito |
| 7 | ¿Dosis de anestesia para cirugía de columna? | Anti-alucinación | 1.00 | 0.00 | 0.00 | ✅ Sin alucinación |
| 8 | ¿Fármacos para tratar esclerosis múltiple? | Anti-alucinación | 1.00 | 0.00 | 0.00 | ✅ Sin alucinación |
| 9 | ¿Resultados comparativos: modelos 3D vs métodos tradicionales? | **Caso de Éxito** | 1.00 | 0.94 | 0.78 | ⭐ Mejor caso |
| 10 | ¿Percepción estudiantil en contexto latinoamericano y motivación autónoma? | **Caso de Error** | 0.88 | 0.61 | 0.35 | ⚠️ Parcial |

**Promedios:** Faithfulness **0.975** · Answer Relevancy **0.604** · Context Precision **0.528**

> ℹ️ Las preguntas 7 y 8 tienen Relevancy/Precision = 0.00 porque el sistema **correctamente rechazó responder** (comportamiento deseado). Su Faithfulness de 1.0 confirma que no alució.

### ⭐ Caso de Éxito — Pregunta 9

**Pregunta:** *¿Qué resultados obtuvieron los estudios al comparar el rendimiento académico de estudiantes que usaron modelos 3D versus los que usaron métodos tradicionales?*

**¿Por qué fue exitoso?**
- Vocabulario **presente directamente** en los artículos ("modelos 3D", "rendimiento académico", "métodos tradicionales")
- Los embeddings recuperaron chunks con los datos estadísticos exactos del estudio comparativo
- El LLM citó correctamente la fuente sin añadir información externa
- **Métricas:** Faithfulness 1.00 · Relevancy 0.94 · Precision 0.78

### ⚠️ Caso de Error — Pregunta 10

**Pregunta:** *¿Qué dice el estudio sobre la percepción de los estudiantes en el contexto latinoamericano y su impacto en la motivación autónoma?*

**¿Por qué falló parcialmente?**
- La pregunta combina **3 conceptos** en una sola consulta compleja
- "Motivación autónoma" **no existe explícitamente** en el corpus — es una inferencia
- El retriever trajo chunks genéricos → Context Precision bajo (0.35)
- El LLM generó una respuesta parcialmente especulativa → Faithfulness 0.88

**Lección aprendida:** El sistema falla cuando la pregunta combina términos que existen en el corpus con conceptos que apenas se mencionan. Mejoras futuras: reducir `chunk_size` a 400 e implementar *query decomposition*.

---

© 2026 — Proyecto Académico · Universidad Konrad Lorenz

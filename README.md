# Consultor Especialista en Neuroanatomía (RAG)

Sistema de Inteligencia Artificial que actúa como **consultor científico especializado en neuroanatomía**. Diseñado para responder consultas académicas basándose **exclusivamente** en literatura científica indexada localmente, implementando la arquitectura **RAG (Retrieval-Augmented Generation)** de forma 100% local y soberana.

---

## 📌 Descripción del Proyecto

Este sistema RAG procesa artículos científicos de neuroanatomía (en formato PDF o DOCX), los fragmenta, genera sus correspondientes representaciones vectoriales (embeddings) de manera local y persistente, y permite realizar consultas en lenguaje natural. El sistema responde con base en las citas y referencias exactas de los documentos, previniendo alucinaciones en ámbitos médicos y psicológicos.

**Privacidad y Soberanía Tecnológica:** 
Tanto la base vectorial como el modelo de lenguaje (LLM) corren localmente mediante **Ollama** y **ChromaDB**. Ningún documento ni consulta sale del equipo local, garantizando la privacidad absoluta de los datos.

---

## 🏗️ Arquitectura de la Aplicación

El sistema está dividido en capas con una separación estricta de responsabilidades:

### 1. Capa de Presentación (Frontend Modular)
- **`app.py`**: Punto de entrada de la aplicación. Maneja el ciclo de vida y orquesta los componentes.
- **`style.css`**: Archivo de hojas de estilo centralizado. Contiene toda la identidad visual institucional.
- **`components/`**: Módulos independientes que componen la interfaz gráfica:
  - `header.py`: Cabecera corporativa y branding universitario.
  - `sidebar.py`: Control de carga de documentos, nivel de usuario, parámetros y autenticación de administrador.
  - `consultor.py`: Campo de consulta principal y listado de ejemplos rápidos.
  - `resultados.py`: Tarjeta de respuesta personalizada y listado expandible de evidencias científicas (citas/páginas).
  - `admin_panel.py`: Panel de evaluación del RAGAS, gráficos de latencia y gestión del banco de preguntas evaluativas.

### 2. Capa de Lógica de Negocio (Backend RAG)
- **`rag_pipeline.py`**: Administra la carga de PDFs/DOCX, la fragmentación de texto (chunking), la indexación incremental y las consultas de búsqueda semántica de cosenos a la base de datos vectorial ChromaDB.
- **`config.py`**: Almacena las variables globales, PINs de administración y constantes operativas.

### 3. Capa de Persistencia (Bases de Datos)
- **`chroma_neuro_db/`**: Base de datos vectorial persistente donde se almacenan los embeddings de los documentos científicos.
- **`database.py`** y **`neuro_metrics.db`**: Base de datos relacional SQLite que registra en tiempo real el historial de consultas, latencias de Ollama, distribución de niveles (Básico/Avanzado) y aciertos del cuestionario de autoevaluación.

---

## ⚙️ Configuración del Pipeline RAG

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **LLM Local** | `qwen2.5:1.5b` (Ollama) | Modelo de lenguaje de 1.5 mil millones de parámetros (2GB), rápido y preciso. |
| **Embeddings** | `nomic-embed-text` | Modelo local de vectorización (768 dimensiones). |
| **chunk_size** | 800 caracteres | Tamaño máximo del fragmento de texto procesado. |
| **chunk_overlap** | 80 caracteres | Solapamiento entre fragmentos para preservar contexto en los límites. |
| **Similarity** | Similitud de Coseno | Métrica para la recuperación semántica. |
| **k_retrieval** | Ajustable (3 a 8) | Cantidad de fragmentos recuperados para inyectar en el LLM. |
| **Temperatura** | 0.1 | Configuración determinista óptima para respuestas académicas. |

---

## 🚀 Instalación y Ejecución Local

### Prerrequisitos

1. **Python 3.10+** instalado en tu sistema.
2. **Ollama** instalado y corriendo en tu PC. Puedes descargarlo gratis desde [ollama.com](https://ollama.com).
3. Tener descargados los modelos locales necesarios. Ejecuta en tu terminal:
   ```bash
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ```

### Pasos para iniciar el sistema

```bash
# 1. Clonar el repositorio y entrar a la carpeta
cd ConsultorNeuroanatomia

# 2. Crear y activar un entorno virtual de Python
python3 -m venv env
source env/bin/activate  # En macOS/Linux
# env\Scripts\activate   # En Windows

# 3. Instalar las dependencias de Python
pip install -r requirements.txt

# 4. Iniciar la aplicación Streamlit
streamlit run app.py
```

El navegador abrirá automáticamente la aplicación en la dirección: **`http://localhost:8502`**.

---

## 🛠️ Panel del Administrador

Accediendo con el PIN de administrador (configurado por defecto en `config.py`), se habilitan opciones avanzadas:
1. **Indexación y Gestión:** Cargar nuevos artículos científicos y removerlos en tiempo real de la base de datos sin necesidad de reconstrucciones manuales.
2. **Métricas en Tiempo Real:** Gráficos y reportes de rendimiento del sistema basados en la telemetría almacenada en SQLite.
3. **Gestor de Evaluaciones:** Agregar, modificar o eliminar preguntas del banco de autoevaluaciones que toman los estudiantes.

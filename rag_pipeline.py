"""
rag_pipeline.py — Pipeline RAG para el Consultor Especialista en Neuroanatomía
Versión 2.0: 100% LOCAL con Ollama (sin API keys, sin cuotas, sin internet)
- Embeddings: nomic-embed-text (via Ollama)
- LLM: llama3.2 (via Ollama)
- VectorDB: ChromaDB local (SQLite)
"""

import os
import shutil
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "Docs")

# _get_docs_files es dinámico — se lee en cada llamada para capturar archivos nuevos (.pdf y .docx)
def _get_docs_files():
    if not os.path.exists(DOCS_DIR):
        return []
    return sorted([
        os.path.join(DOCS_DIR, f)
        for f in os.listdir(DOCS_DIR)
        if f.lower().endswith((".pdf", ".docx"))
    ])

def _load_any_document(file_path: str) -> list:
    """Carga un PDF usando PyPDFLoader o un DOCX usando un parser local de XML."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_path.lower().endswith(".docx"):
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(file_path) as docx:
                tree = ET.parse(docx.open('word/document.xml'))
                root = tree.getroot()
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                text = ' '.join(n.text for n in root.findall('.//w:t', ns) if n.text)
            
            # Retorna como una sola página de Documento (será fragmentada en el split)
            return [Document(page_content=text, metadata={"source": file_path, "page": 1})]
        except Exception as e:
            print(f"  [!] Error leyendo Word {os.path.basename(file_path)}: {e}")
            return []
    return []

PERSIST_DIR = os.path.join(BASE_DIR, "chroma_neuro_db")
COLLECTION_NAME = "neuroanatomia_cientifica"

# ─────────────────────────────────────────────
# 2. MODELOS — 100% LOCAL via Ollama
# ─────────────────────────────────────────────
# Modelo de embeddings local: nomic-embed-text (274 MB, sin cuotas)
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL   = "qwen2.5:1.5b"   # 2 GB — ya instalado. Cambiar a llama3.1:8b o qwen2.5:7b si hay más RAM

embeddings_model = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL,
)

# ─────────────────────────────────────────────
# 3. SYSTEM PROMPT — Identidad del consultor
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION_BASICO = """Eres un consultor y divulgador de neuroanatomía para público general sin formación en medicina.
Tu misión es explicar de manera sencilla, didáctica y clara, utilizando analogías y metáforas de la vida cotidiana para facilitar el aprendizaje. Evita tecnicismos excesivos y traduce conceptos médicos difíciles a un lenguaje accesible.

Tu respuesta debe basarse EXCLUSIVAMENTE en la información del <contexto>.

REGLAS ESTRICTAS (de mayor a menor prioridad):
1. Si la <pregunta> es un saludo, una expresión social (hola, gracias, qué tal, etc.) o NO es una pregunta científica, responde Únicamente con:
   "Por favor, formula una pregunta específica sobre neuroanatomía para poder ayudarte. Ejemplo: ¿Cuáles son las partes del cerebro?"
   No agregues nada más. No resumas los documentos.
2. Si el contexto contiene información relacionada, descríbela de forma didáctica.
3. Si el tema de la pregunta NO aparece de ninguna forma en el contexto, responde:
   "Esta información no se encuentra en los documentos científicos disponibles."
4. Cita siempre la fuente (nombre del archivo y página) de forma simplificada al final.
5. Nunca inventes información o uses conocimiento externo."""

SYSTEM_INSTRUCTION_AVANZADO = """Eres un consultor especialista y docente en neuroanatomía clínica para nivel universitario y profesional.
Tu misión es responder con alta rigurosidad académica, precisión terminológica y profundidad conceptual. Detalla los núcleos específicos, tractos neuronales, relaciones espaciales y correlaciones clínicas correspondientes.

Tu respuesta debe basarse EXCLUSIVAMENTE en la información del <contexto>.

REGLAS ESTRICTAS (de mayor a menor prioridad):
1. Si la <pregunta> es un saludo, una expresión social (hola, gracias, qué tal, etc.) o NO es una pregunta científica, responde Únicamente con:
   "Por favor, formula una pregunta específica sobre el contenido de los artículos científicos. Ejemplo: ¿Qué estructuras neuroanatómicas se describen?"
   No agregues nada más. No resumas los documentos.
2. Si el contexto contiene información relacionada con la pregunta, sintetiza y presenta la información disponible detalladamente.
3. Si el tema de la pregunta NO aparece de ninguna forma en el contexto, responde:
   "Esta información no se encuentra en los documentos científicos disponibles."
4. Cita siempre la fuente (nombre del archivo y página) al final de cada respuesta científica.
5. Usa terminología científica precisa. Nunca uses conocimiento externo."""

PROMPT_TEMPLATE = """
<contexto>
{context}
</contexto>

<pregunta>
{question}
</pregunta>

Responde como consultor basándote EXCLUSIVAMENTE en el contexto anterior. Al final, indica en qué fuente(s) encontraste la información."""


# ─────────────────────────────────────────────
# 3.5. HELPER — Restaurar backup si el rebuild falla
# ─────────────────────────────────────────────
def _restaurar_backup(temp_dir: str, backup_dir: str, persist_dir: str) -> None:
    """Limpia carpeta temporal y restaura el backup si existe."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(backup_dir):
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        shutil.copytree(backup_dir, persist_dir)
        shutil.rmtree(backup_dir)
        print("[RESTORE] ✅ Base de datos anterior restaurada exitosamente.")
    else:
        print("[RESTORE] ⚠️ No se encontró backup para restaurar.")


# ─────────────────────────────────────────────
# 4. CONSTRUCCIÓN DE LA BASE VECTORIAL
# ─────────────────────────────────────────────
def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    PASO 1-4 del pipeline RAG:
    Carga PDFs → Chunking → Vectorización (Embeddings locales) → ChromaDB

    Backup permanente en ~/.neuro_db_permanent/ — se restaura automáticamente
    si la DB local desaparece.
    """
    PERMANENT_BACKUP = os.path.join(os.path.expanduser("~"), ".neuro_db_permanent")

    if not force_rebuild:
        # Modo carga: SOLO cargar si existe, NUNCA reconstruir automáticamente
        if not os.path.exists(PERSIST_DIR):
            if os.path.exists(PERMANENT_BACKUP):
                print("[RESTORE] DB no encontrada localmente. Restaurando desde backup permanente...")
                shutil.copytree(PERMANENT_BACKUP, PERSIST_DIR)
                print("[RESTORE] ✔ DB restaurada desde ~/.neuro_db_permanent/")
            else:
                raise FileNotFoundError(
                    "Base vectorial no encontrada. "
                    "Usa el botón 'Reconstruir VectorDB' para crearla."
                )
        print(f"[OK] Cargando base vectorial existente desde: {PERSIST_DIR}")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME,
        )

    # ── Construir en carpeta TEMPORAL + Backup de seguridad ──
    TEMP_DIR   = PERSIST_DIR + "_temp"
    BACKUP_DIR = PERSIST_DIR + "_backup"

    # 1. Limpiar temp anterior si existe
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    # 2. Hacer backup de la DB actual ANTES de tocarla
    if os.path.exists(PERSIST_DIR):
        if os.path.exists(BACKUP_DIR):
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(PERSIST_DIR, BACKUP_DIR)
        print(f"[BACKUP] DB respaldada en {os.path.basename(BACKUP_DIR)}/")

    # PASO 1 — Carga de documentos (PDF y DOCX)
    docs_files = _get_docs_files()
    print(f"\n[PASO 1] Cargando documentos de neuroanatomía... ({len(docs_files)} archivos en Docs/)")
    documents = []
    for file_path in docs_files:
        if not os.path.exists(file_path):
            print(f"  [!] Archivo no encontrado: {os.path.basename(file_path)}")
            continue
        pages = _load_any_document(file_path)
        documents.extend(pages)
        print(f"  ✔ {os.path.basename(file_path)}: {len(pages)} páginas/secciones cargadas")
    print(f"  Total de páginas/secciones cargadas: {len(documents)}")

    # PASO 2 — Chunking
    print("\n[PASO 2] Dividiendo en fragmentos (chunk_size=800, overlap=100)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Fragmentos generados: {len(chunks)}")

    # PASO 3 & 4 — Embeddings locales + ChromaDB
    # OllamaEmbeddings no tiene límite de cuota — 100% local
    print(f"\n[PASO 3 & 4] Vectorizando con Ollama ({OLLAMA_EMBED_MODEL}) — sin cuotas, 100% local...")
    print("  (lotes de 50 fragmentos)")

    BATCH_SIZE = 50
    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        lote = chunks[i:i + BATCH_SIZE]
        numero_lote = i // BATCH_SIZE + 1
        total_lotes = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Lote {numero_lote}/{total_lotes}: fragmentos {i+1}–{min(i+BATCH_SIZE, len(chunks))}...")

        try:
            if vector_store is None:
                vector_store = Chroma.from_documents(
                    documents=lote,
                    embedding=embeddings_model,
                    persist_directory=TEMP_DIR,
                    collection_name=COLLECTION_NAME,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                vector_store.add_documents(lote)
        except Exception as e:
            _restaurar_backup(TEMP_DIR, BACKUP_DIR, PERSIST_DIR)
            raise RuntimeError(f"Error vectorizando: {e}") from e

    # ── Swap seguro ──
    total = vector_store._collection.count()
    print(f"  ✔ {total} vectores listos. Cerrando conexión temporal...")

    try:
        vector_store._client._system.stop()
    except Exception:
        pass
    del vector_store

    # Checkpoint WAL de SQLite
    import sqlite3 as _sqlite3
    sqlite_file = os.path.join(TEMP_DIR, "chroma.sqlite3")
    if os.path.exists(sqlite_file):
        try:
            _conn = _sqlite3.connect(sqlite_file)
            _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _conn.close()
            print("  ✔ SQLite WAL checkpoint completado")
        except Exception as _e:
            print(f"  [WARN] No se pudo hacer checkpoint: {_e}")

    # Swap: temporal → real
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    shutil.copytree(TEMP_DIR, PERSIST_DIR)
    shutil.rmtree(TEMP_DIR)

    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
        print(f"  ✔ Backup eliminado — DB nueva confirmada ({total} vectores)")

    print(f"  ✔ DB actualizada en {os.path.basename(PERSIST_DIR)}/ — {total} vectores indexados")

    # Backup permanente en home
    try:
        PERMANENT_BACKUP = os.path.join(os.path.expanduser("~"), ".neuro_db_permanent")
        if os.path.exists(PERMANENT_BACKUP):
            shutil.rmtree(PERMANENT_BACKUP)
        shutil.copytree(PERSIST_DIR, PERMANENT_BACKUP)
        print(f"  ✔ Backup permanente guardado en ~/.neuro_db_permanent/ ({total} vectores)")
    except Exception as _e:
        print(f"  [WARN] No se pudo guardar backup permanente: {_e}")

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME,
    )


# ─────────────────────────────────────────────
# 4a. ELIMINAR VECTORES DE UN PDF ESPECÍFICO
# ─────────────────────────────────────────────
def remove_documents_from_store(pdf_filename: str, vs_existente=None):
    """
    Elimina de ChromaDB todos los vectores que provienen del PDF indicado.
    """
    vs = vs_existente
    if vs is None:
        if not os.path.exists(PERSIST_DIR):
            return None, 0
        vs = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME,
        )

    todos = vs._collection.get(include=["metadatas"])
    ids_a_borrar = [
        doc_id
        for doc_id, meta in zip(todos["ids"], todos["metadatas"])
        if meta and pdf_filename in (meta.get("source", ""))
    ]

    if ids_a_borrar:
        vs._collection.delete(ids=ids_a_borrar)
        print(f"  ✔ {len(ids_a_borrar)} vectores eliminados de '{pdf_filename}'")
    else:
        print(f"  [!] No se encontraron vectores para '{pdf_filename}'")

    total = vs._collection.count()
    print(f"  ✔ DB ahora tiene {total} vectores totales")
    return vs, len(ids_a_borrar)


# ─────────────────────────────────────────────
# 4b. INDEXACIÓN INCREMENTAL — solo archivos nuevos
# ─────────────────────────────────────────────
def add_documents_incremental(new_pdf_paths: list, vs_existente=None):
    """
    Agrega solo los archivos nuevos a la base vectorial existente.
    """
    BATCH_SIZE = 50

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )

    documents = []
    for doc_path in new_pdf_paths:
        if not os.path.exists(doc_path):
            print(f"  [!] No encontrado: {os.path.basename(doc_path)}")
            continue
        pages = _load_any_document(doc_path)
        documents.extend(pages)
        print(f"  ✔ {os.path.basename(doc_path)}: {len(pages)} páginas/secciones cargadas")

    if not documents:
        raise ValueError("No se pudo cargar ningún documento de los archivos dados.")

    chunks = splitter.split_documents(documents)
    print(f"  Fragmentos nuevos: {len(chunks)}")

    vs = vs_existente
    if vs is None and os.path.exists(PERSIST_DIR):
        vs = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )

    for i in range(0, len(chunks), BATCH_SIZE):
        lote = chunks[i:i + BATCH_SIZE]
        numero_lote = i // BATCH_SIZE + 1
        total_lotes = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Lote {numero_lote}/{total_lotes}: fragmentos {i+1}–{min(i+BATCH_SIZE, len(chunks))}...")

        try:
            if vs is None:
                vs = Chroma.from_documents(
                    documents=lote,
                    embedding=embeddings_model,
                    persist_directory=PERSIST_DIR,
                    collection_name=COLLECTION_NAME,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                vs.add_documents(lote)
        except Exception as e:
            raise RuntimeError(f"Error vectorizando: {e}") from e

    total = vs._collection.count()
    print(f"  ✔ DB ahora tiene {total} vectores totales")
    return vs


# ─────────────────────────────────────────────
# 5. CONSULTA RAG — 100% LOCAL con Ollama LLM
# ─────────────────────────────────────────────
def consultar(pregunta: str, vector_store: Chroma, k: int = 5, nivel: str = "avanzado") -> dict:
    """
    PASOS 5-7 del pipeline RAG:
    Recuperación vectorial → Prompt aumentado → Generación local con Ollama
    """
    # PASO 5 — Recuperación
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(pregunta)

    # PASO 6 — Construcción del prompt aumentado
    context_parts = []
    for i, doc in enumerate(docs, 1):
        fuente = os.path.basename(doc.metadata.get("source", "desconocido"))
        pagina = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Fragmento {i} — {fuente}, Pág. {pagina}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_instruction = SYSTEM_INSTRUCTION_AVANZADO if nivel.lower() == "avanzado" else SYSTEM_INSTRUCTION_BASICO

    prompt_completo = (
        system_instruction
        + "\n\n"
        + PROMPT_TEMPLATE.format(context=context, question=pregunta)
    )

    # PASO 7 — Generación LOCAL con Ollama (sin internet, sin cuotas)
    llm = ChatOllama(
        model=OLLAMA_LLM_MODEL,
        temperature=0.1,
        num_predict=600,   # Límite explícito para respuestas ágiles en CPU Intel
        num_ctx=2048,      # Ventana de contexto reducida para mayor velocidad
    )

    # pyrefly: ignore [missing-import]
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=PROMPT_TEMPLATE.format(context=context, question=pregunta)),
    ]
    response = llm.invoke(messages)
    texto = response.content

    return {
        "pregunta": pregunta,
        "respuesta": texto,
        "fragmentos": docs,
        "tokens_contexto_aprox": len(context) // 4,
    }


def stream_consultar(pregunta: str, vector_store, k: int = 5, nivel: str = "avanzado"):
    """
    Igual que consultar() pero devuelve un GENERADOR de tokens.
    Se usa con st.write_stream() en Streamlit para streaming en tiempo real.
    Retorna: (generator, docs, context_tokens)
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(pregunta)

    context_parts = []
    for i, doc in enumerate(docs, 1):
        fuente = os.path.basename(doc.metadata.get("source", "desconocido"))
        pagina = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Fragmento {i} — {fuente}, Pág. {pagina}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_instruction = SYSTEM_INSTRUCTION_AVANZADO if nivel.lower() == "avanzado" else SYSTEM_INSTRUCTION_BASICO

    # pyrefly: ignore [missing-import]
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = ChatOllama(
        model=OLLAMA_LLM_MODEL,
        temperature=0.1,
        num_predict=600,
        num_ctx=2048,
    )
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=PROMPT_TEMPLATE.format(context=context, question=pregunta)),
    ]

    def _token_generator():
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    return _token_generator(), docs, len(context) // 4


# ─────────────────────────────────────────────
# 6. EJECUCIÓN DIRECTA (modo script / prueba)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("🧠 CONSULTOR RAG — NEUROANATOMÍA (100% LOCAL con Ollama)")
    print("=" * 65)

    vs = build_vector_store(force_rebuild=False)

    preguntas_prueba = [
        "¿Cuáles son las principales estructuras neuroanatómicas descritas?",
        "¿Qué hallazgos morfológicos o histológicos se reportan?",
        "¿Es útil usar tecnología 3D para estudiar el cerebro?",
        "¿Cuál es la dosis de anestesia recomendada para una cirugía de columna?",
    ]

    for pregunta in preguntas_prueba:
        print(f"\n{'─'*65}")
        print(f"❓ {pregunta}")
        resultado = consultar(pregunta, vs)
        print(f"\n🤖 {resultado['respuesta']}")
        print(f"\n   [~{resultado['tokens_contexto_aprox']} tokens | "
              f"{len(resultado['fragmentos'])} fragmentos recuperados]")

    print(f"\n{'='*65}")
    print("Sistema listo.")

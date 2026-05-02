"""
rag_pipeline.py — Pipeline RAG para el Consultor Especialista en Neuroanatomía
Carga PDFs científicos, los vectoriza con Google Gemini Embeddings y ChromaDB,
y responde consultas usando Gemini como LLM, anclado exclusivamente al contenido.
"""

import os
import shutil
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from google import genai
from google.genai import types as genai_types

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("No se encontró GEMINI_API_KEY en el archivo .env")

os.environ["GOOGLE_API_KEY"] = API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "Docs")
# PDF_FILES es dinámico — se lee en cada llamada para capturar archivos nuevos
def _get_pdf_files():
    if not os.path.exists(DOCS_DIR):
        return []
    return sorted([
        os.path.join(DOCS_DIR, f)
        for f in os.listdir(DOCS_DIR)
        if f.lower().endswith(".pdf")
    ])

PERSIST_DIR = os.path.join(BASE_DIR, "chroma_neuro_db")
COLLECTION_NAME = "neuroanatomia_cientifica"

# ─────────────────────────────────────────────
# 2. MODELOS
# ─────────────────────────────────────────────
# Modelo de embeddings: gemini-embedding-001 (cambiado porque gemini-embedding-2-preview agotó su cuota de 1000 requests)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY,
)

# Cliente Gemini directo (sin forzar api_version)
_genai_client = genai.Client(api_key=API_KEY)

# ─────────────────────────────────────────────
# 3. SYSTEM PROMPT — Identidad del consultor
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION = """Eres un consultor especialista en neuroanatomía con formación en investigación científica.
Tu misión es responder preguntas EXCLUSIVAMENTE basándote en la información del <contexto>.

REGLAS ESTRICTAS (de mayor a menor prioridad):
1. Si la <pregunta> es un saludo, una expresión social (hola, gracias, qué tal, etc.) o NO es una pregunta científica, responde Únicamente con:
   "Por favor, formula una pregunta específica sobre el contenido de los artículos científicos. Ejemplo: ¿Qué estructuras neuroanatómicas se describen?"
   No agregues nada más. No resumas los documentos. No menciones su contenido.
2. Si la respuesta a una pregunta científica NO está en el contexto, responde:
   "Esta información no se encuentra en los documentos científicos disponibles."
3. Siempre cita la fuente (nombre del PDF y página) al final de cada respuesta científica.
4. Usa terminología científica precisa. Nunca uses conocimiento externo a los documentos.
5. Si se necesita combinar información de varios fragmentos, intégralos de forma coherente."""

PROMPT_TEMPLATE = """
<contexto>
{context}
</contexto>

<pregunta>
{question}
</pregunta>

Responde como consultor científico especialista en neuroanatomía basándote EXCLUSIVAMENTE
en el contexto anterior. Al final, indica en qué fuente(s) encontraste la información."""


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
    Carga PDFs → Chunking → Vectorización (Embeddings) → ChromaDB

    Backup permanente en ~/.neuro_db_permanent/ — se restaura automáticamente
    si la DB local desaparece (git clean, rm -rf accidental, etc.)
    """
    PERMANENT_BACKUP = os.path.join(os.path.expanduser("~"), ".neuro_db_permanent")

    if not force_rebuild:
        # Modo carga: SOLO cargar si existe, NUNCA reconstruir automáticamente
        if not os.path.exists(PERSIST_DIR):
            # ── Intentar restaurar desde backup permanente antes de fallar ──
            if os.path.exists(PERMANENT_BACKUP):
                print(f"[RESTORE] DB no encontrada localmente. Restaurando desde backup permanente...")
                shutil.copytree(PERMANENT_BACKUP, PERSIST_DIR)
                print(f"[RESTORE] ✔ DB restaurada desde ~/.neuro_db_permanent/")
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

    # PASO 1 — Carga de documentos PDF (lectura dinámica de Docs/)
    pdf_files = _get_pdf_files()
    print(f"\n[PASO 1] Cargando PDFs de neuroanatomía... ({len(pdf_files)} archivos en Docs/)")
    documents = []
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"  [!] Archivo no encontrado: {os.path.basename(pdf_path)}")
            continue
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        documents.extend(pages)
        print(f"  ✔ {os.path.basename(pdf_path)}: {len(pages)} páginas")
    print(f"  Total páginas cargadas: {len(documents)}")

    # PASO 2 — Chunking (RecursiveCharacterTextSplitter)
    print("\n[PASO 2] Dividiendo en fragmentos (chunk_size=600, overlap=80)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Fragmentos generados: {len(chunks)}")

    # PASO 3 & 4 — Embeddings + ChromaDB en carpeta TEMPORAL (la original no se toca hasta el éxito)
    print("\n[PASO 3 & 4] Vectorizando hacia carpeta temporal (la DB original queda intacta hasta confirmar éxito)...")
    print("  (lotes de 100 fragmentos, pausa 5s entre lotes, reintento automático en 429)")

    BATCH_SIZE = 100
    PAUSE_SECONDS = 5
    MAX_RETRIES = 3
    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        lote = chunks[i:i + BATCH_SIZE]
        numero_lote = i // BATCH_SIZE + 1
        total_lotes = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Lote {numero_lote}/{total_lotes}: fragmentos {i+1}–{min(i+BATCH_SIZE, len(chunks))}...")

        # Retry con backoff exponencial en caso de 429
        for intento in range(1, MAX_RETRIES + 1):
            try:
                if vector_store is None:
                    vector_store = Chroma.from_documents(
                        documents=lote,
                        embedding=embeddings_model,
                        persist_directory=TEMP_DIR,       # ← carpeta temporal
                        collection_name=COLLECTION_NAME,
                        collection_metadata={"hnsw:space": "cosine"},
                    )
                else:
                    vector_store.add_documents(lote)
                break  # Éxito — salir del loop de reintentos
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or \
                   "503" in err_str or "UNAVAILABLE" in err_str:
                    espera = 30 * intento  # 30s, 60s, 90s solo en caso de error real
                    print(f"  ⚠️ Error temporal ({'429' if '429' in err_str else '503'}). Esperando {espera}s (intento {intento}/{MAX_RETRIES})...")
                    time.sleep(espera)
                    if intento == MAX_RETRIES:
                        _restaurar_backup(TEMP_DIR, BACKUP_DIR, PERSIST_DIR)
                        raise RuntimeError(
                            f"API no disponible después de {MAX_RETRIES} reintentos. "
                            "La base de datos anterior fue restaurada automáticamente."
                        ) from e
                else:
                    _restaurar_backup(TEMP_DIR, BACKUP_DIR, PERSIST_DIR)
                    raise

        # No forzamos pausas, si hay 429 el except se encarga

    # ── Swap seguro: cerrar conexión y hacer checkpoint SQLite antes de copiar ──
    total = vector_store._collection.count()
    print(f"  ✔ {total} vectores listos. Cerrando conexión temporal y guardando DB...")

    # 1. Cerrar la conexión ChromaDB al directorio temporal
    try:
        vector_store._client._system.stop()
    except Exception:
        pass
    del vector_store

    # 2. Checkpoint del WAL de SQLite (Write-Ahead Log) para que todos los
    #    datos queden en el archivo .sqlite3 principal antes de copiar.
    #    Sin esto, la DB copiada puede fallar con "default_tenant not found".
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

    # 3. Swap: reemplazar la DB real con la temporal
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    shutil.copytree(TEMP_DIR, PERSIST_DIR)
    shutil.rmtree(TEMP_DIR)

    # Eliminar backup ahora que la nueva DB está confirmada
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
        print(f"  ✔ Backup eliminado — DB nueva confirmada ({total} vectores)")

    print(f"  ✔ DB actualizada en {os.path.basename(PERSIST_DIR)}/ — {total} vectores indexados")

    # ── Guardar backup PERMANENTE en home (~/.neuro_db_permanent/) ──
    # Este backup sobrevive a git clean, rm -rf en el proyecto, etc.
    try:
        PERMANENT_BACKUP = os.path.join(os.path.expanduser("~"), ".neuro_db_permanent")
        if os.path.exists(PERMANENT_BACKUP):
            shutil.rmtree(PERMANENT_BACKUP)
        shutil.copytree(PERSIST_DIR, PERMANENT_BACKUP)
        print(f"  ✔ Backup permanente guardado en ~/.neuro_db_permanent/ ({total} vectores)")
    except Exception as _e:
        print(f"  [WARN] No se pudo guardar backup permanente: {_e}")

    # Devolver instancia apuntando a la carpeta real
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
    Usa el campo metadata['source'] para identificarlos.
    Devuelve (vs, n_eliminados).
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

    # Buscar todos los IDs cuyo metadata['source'] contenga el nombre del PDF
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
    # El backup permanente NO se actualiza aqui (evita error 1032 SQLITE_READONLY_DBMOVED)
    # Solo se actualiza en el rebuild completo donde se cierra la conexion antes de copiar
    return vs, len(ids_a_borrar)


# ─────────────────────────────────────────────
# 4b. INDEXACIÓN INCREMENTAL — solo archivos nuevos
# ─────────────────────────────────────────────
def add_documents_incremental(new_pdf_paths: list, vs_existente=None):
    """
    Agrega solo los archivos nuevos a la base vectorial existente.
    Si se pasa vs_existente (objeto Chroma ya abierto), lo reutiliza
    para evitar el error de doble conexion SQLite (codigo 1032).
    """
    BATCH_SIZE = 100
    MAX_RETRIES = 3

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )

    # Cargar y chunkear solo los nuevos PDFs
    documents = []
    for pdf_path in new_pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"  [!] No encontrado: {os.path.basename(pdf_path)}")
            continue
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        documents.extend(pages)
        print(f"  ✔ {os.path.basename(pdf_path)}: {len(pages)} páginas")

    if not documents:
        raise ValueError("No se pudo cargar ningún documento de los archivos dados.")

    chunks = splitter.split_documents(documents)
    print(f"  Fragmentos nuevos: {len(chunks)}")

    # Reutilizar conexion existente si se pasa (evita error 1032 de doble conexion)
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

        for intento in range(1, MAX_RETRIES + 1):
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
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    espera = 15 * intento  # 15s, 30s max — falla rápido en UI
                    print(f"  ⚠️ 429 — esperando {espera}s...")
                    time.sleep(espera)
                    if intento == MAX_RETRIES:
                        raise RuntimeError(
                            "Cuota de embeddings agotada (límite diario gratuito). "
                            "Se resetea automáticamente mañana. "
                            "O genera una nueva API Key en https://aistudio.google.com/apikey"
                        ) from e
                else:
                    raise

        # Sin pausas forzadas entre lotes para maximizar velocidad

    total = vs._collection.count()
    print(f"  ✔ DB ahora tiene {total} vectores totales")
    # El backup permanente NO se actualiza aqui (evita error 1032 SQLITE_READONLY_DBMOVED)
    # Solo se actualiza en el rebuild completo donde se cierra la conexion antes de copiar
    return vs

# ─────────────────────────────────────────────
def consultar(pregunta: str, vector_store: Chroma, k: int = 5) -> dict:
    """
    PASOS 5-7 del pipeline RAG:
    Recuperación vectorial → Prompt aumentado → Generación con LLM
    """
    # PASO 5 — Recuperación (k fragmentos más similares)
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

    prompt_completo = (
        SYSTEM_INSTRUCTION
        + "\n\n"
        + PROMPT_TEMPLATE.format(context=context, question=pregunta)
    )

    # PASO 7 — Generación LLM via REST (intenta modelos en orden de disponibilidad)
    import requests as _req
    _llm_models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash-preview-04-17",
    ]
    _body = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"parts": [{"text": PROMPT_TEMPLATE.format(context=context, question=pregunta)}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    texto = ""
    for _model in _llm_models:
        _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model}:generateContent?key={API_KEY}"
        _resp = _req.post(_url, json=_body, timeout=30)
        if _resp.status_code == 200:
            texto = _resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            break
        elif _resp.status_code == 429:
            continue  # cuota agotada → probar siguiente modelo
        else:
            err = _resp.json().get("error", {}).get("message", "")
            if "not found" in err.lower() or "404" in str(_resp.status_code):
                continue  # modelo no disponible → probar siguiente
            raise RuntimeError(f"Error LLM ({_resp.status_code}): {err[:120]}")
    if not texto:
        raise RuntimeError(
            "Cuota LLM agotada en todos los modelos disponibles. "
            "Por favor agrega una nueva GEMINI_API_KEY en el archivo .env y reinicia la app."
        )

    return {
        "pregunta": pregunta,
        "respuesta": texto,
        "fragmentos": docs,
        "tokens_contexto_aprox": len(context) // 4,
    }


# ─────────────────────────────────────────────
# 6. EJECUCIÓN DIRECTA (modo script / prueba)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("🧠 CONSULTOR RAG — NEUROANATOMÍA CIENTÍFICA")
    print("=" * 65)

    vs = build_vector_store(force_rebuild=False)

    preguntas_prueba = [
        "¿Cuáles son las principales estructuras neuroanatómicas descritas?",
        "¿Qué hallazgos morfológicos o histológicos se reportan?",
        "¿Qué metodología de investigación utilizaron los autores?",
        "¿Cuáles son las conclusiones principales de los artículos?",
        "¿Qué variaciones anatómicas se identificaron en los estudios?",
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

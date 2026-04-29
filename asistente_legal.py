"""
rag_pipeline.py — Pipeline RAG para el Consultor Especialista en Neuroanatomía
Carga los 3 PDFs científicos, los vectoriza con ChromaDB y responde consultas
usando Google Gemini como LLM generador, anclado exclusivamente al contenido.
"""

import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("No se encontró GEMINI_API_KEY en el archivo .env")

os.environ["GOOGLE_API_KEY"] = API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILES = [
    os.path.join(BASE_DIR, "0717-9502-ijmorphol-41-04-996.pdf"),
    os.path.join(BASE_DIR, "SCT_2025_1250.pdf"),
    os.path.join(BASE_DIR, "circir_25_93_2_197-201.pdf"),
]

PERSIST_DIR = os.path.join(BASE_DIR, "chroma_neuro_db")
COLLECTION_NAME = "neuroanatomia_cientifica"

# ─────────────────────────────────────────────
# 2. MODELOS
# ─────────────────────────────────────────────
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    google_api_key=API_KEY,
)

# ─────────────────────────────────────────────
# 3. SYSTEM PROMPT — Identidad del consultor
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION = """Eres un consultor especialista en neuroanatomía con formación en investigación científica.
Tu misión es analizar y responder preguntas ÚNICAMENTE basándote en la información contenida
en los artículos científicos proporcionados en el <contexto>.

REGLAS ESTRICTAS:
1. SOLO puedes usar la información del <contexto>. Nunca uses conocimiento externo.
2. Si la respuesta NO está en el contexto, responde exactamente:
   "Esta información no se encuentra en los documentos científicos disponibles."
3. Siempre cita la fuente (nombre del PDF y página) al final de tu respuesta.
4. Usa terminología científica precisa y apropiada para el área de neuroanatomía.
5. Si la pregunta requiere combinar información de varios fragmentos, intégralos de forma coherente.

EJEMPLO 1:
<contexto>El nervio facial emerge del tronco encefálico a nivel del surco bulbopontino... (Fragmento 1, pág. 2)</contexto>
<pregunta>¿Desde dónde emerge el nervio facial?</pregunta>
Respuesta: Según el documento, el nervio facial emerge del tronco encefálico a nivel del surco bulbopontino.
Fuente: Fragmento 1, pág. 2.

EJEMPLO 2:
<contexto>...descripción de variaciones en la arteria cerebral media... (Fragmento 3, pág. 7)</contexto>
<pregunta>¿Cuáles son las variantes de la arteria basilar?</pregunta>
Respuesta: Esta información no se encuentra en los documentos científicos disponibles."""

PROMPT_TEMPLATE = """
<contexto>
{context}
</contexto>

<pregunta>
{question}
</pregunta>

Responde como consultor científico especialista en neuroanatomía basándote EXCLUSIVAMENTE
en el contexto anterior. Al final, indica en qué fuente(s) encontraste la información."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("human", PROMPT_TEMPLATE),
])


# ─────────────────────────────────────────────
# 4. CONSTRUCCIÓN DE LA BASE VECTORIAL
# ─────────────────────────────────────────────
def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    PASO 1-4 del pipeline RAG:
    Carga PDFs → Chunking → Vectorización (Embeddings) → ChromaDB
    """
    if os.path.exists(PERSIST_DIR) and not force_rebuild:
        print(f"[OK] Cargando base vectorial existente desde: {PERSIST_DIR}")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME,
        )

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    # PASO 1 — Carga de documentos PDF
    print("\n[PASO 1] Cargando PDFs de neuroanatomía...")
    documents = []
    for pdf_path in PDF_FILES:
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

    # PASO 3 & 4 — Embeddings + ChromaDB
    print("\n[PASO 3 & 4] Vectorizando y almacenando en ChromaDB...")
    print("  (modelo: gemini-embedding-001 · puede tardar ~1-2 min)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    total = vector_store._collection.count()
    print(f"  ✔ {total} vectores almacenados en {PERSIST_DIR}/")
    return vector_store


# ─────────────────────────────────────────────
# 5. PIPELINE RAG — CONSULTA
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

    augmented_prompt = prompt_template.invoke({
        "context": context,
        "question": pregunta,
    })

    # PASO 7 — Generación con LLM (temperatura baja → respuestas precisas)
    respuesta = llm.invoke(augmented_prompt)
    texto = respuesta.content if isinstance(respuesta.content, str) else str(respuesta.content)

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

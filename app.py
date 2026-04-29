"""
app.py — Interfaz Streamlit del Consultor Científico en Neuroanatomía RAG
Avance 2: Sistema RAG completo con LangChain + ChromaDB + Google Gemini
"""

import streamlit as st
import os
from dotenv import load_dotenv

# ── Configuración de página (DEBE ser la primera instrucción de Streamlit) ──
st.set_page_config(
    page_title="Consultor Neuroanatomía RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Premium ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
    border-right: 1px solid rgba(100, 200, 255, 0.15);
}

.hero-header {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 50%, #0d2137 100%);
    border: 1px solid rgba(100, 200, 255, 0.2);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    text-align: center;
}
.hero-header h1 { color: #64c8ff; font-size: 2rem; font-weight: 700; margin: 0; }
.hero-header p  { color: #8ab4d4; font-size: 1rem; margin: 8px 0 0 0; }

.response-card {
    background: linear-gradient(135deg, rgba(13,33,55,0.95) 0%, rgba(17,34,64,0.95) 100%);
    border: 1px solid rgba(100, 200, 255, 0.25);
    border-left: 4px solid #64c8ff;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 16px;
    color: #cce8ff;
    line-height: 1.75;
    font-size: 0.97rem;
}

.metric-row { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
.metric-chip {
    background: rgba(100, 200, 255, 0.08);
    border: 1px solid rgba(100, 200, 255, 0.2);
    border-radius: 20px;
    padding: 6px 16px;
    color: #64c8ff;
    font-size: 0.82rem;
    font-weight: 600;
}

.stTextArea textarea {
    background: rgba(13,33,55,0.8) !important;
    border: 1px solid rgba(100,200,255,0.3) !important;
    border-radius: 10px !important;
    color: #cce8ff !important;
    font-size: 0.95rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a6fa8 0%, #1a82c8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 24px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2080c0 0%, #209ae0 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(100,200,255,0.3) !important;
}
hr { border-color: rgba(100,200,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ── Carga de la API key ──
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("🔑 No se encontró GEMINI_API_KEY en el archivo .env")
    st.stop()
os.environ["GOOGLE_API_KEY"] = API_KEY

# ── Importaciones del RAG ──
try:
    from asistente_legal import build_vector_store, consultar
except ImportError as e:
    st.error(f"Error al importar el pipeline RAG: {e}")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🧠 Neuroanatomía RAG")
    st.markdown("---")
    st.markdown("**Base de conocimientos:**")
    pdfs = [
        "📄 International J. of Morphology (2023)",
        "📄 Surgical & Clinical Trials (2025)",
        "📄 Cirugía y Cirujanos (2025)",
    ]
    for p in pdfs:
        st.markdown(f"- {p}")
    st.markdown("---")
    st.markdown("**Configuración del pipeline:**")
    k_chunks = st.slider("Fragmentos recuperados (k)", 3, 8, 5)
    st.markdown("---")
    st.markdown("**Modelo LLM:** `gemini-1.5-flash`")
    st.markdown("**Embeddings:** `embedding-001`")
    st.markdown("**Temperatura:** `0.1`")
    st.markdown("**chunk_size:** `600` | **overlap:** `80`")
    st.markdown("---")
    if st.button("🔄 Reconstruir base vectorial"):
        with st.spinner("Reconstruyendo..."):
            st.cache_resource.clear()
            st.session_state.pop("vector_store", None)
        st.success("¡Base vectorial reconstruida!")
        st.rerun()
    st.markdown("---")
    st.caption("© 2026 · Konrad Lorenz · V. García & B. Ramirez")

# ── Header principal ──
st.markdown("""
<div class="hero-header">
    <h1>🧠 Consultor Especialista en Neuroanatomía</h1>
    <p>Sistema RAG · Respuestas fundamentadas exclusivamente en artículos científicos peer-reviewed</p>
</div>
""", unsafe_allow_html=True)

# ── Inicialización de la base vectorial (cacheada) ──
@st.cache_resource(show_spinner=False)
def get_vector_store():
    return build_vector_store(force_rebuild=False)

if "vector_store" not in st.session_state:
    with st.spinner("🔬 Inicializando base de conocimientos neuroanatómica (primera vez ~1-2 min)..."):
        st.session_state["vector_store"] = get_vector_store()
    st.success("✅ Base vectorial lista. ¡Puedes realizar tu consulta!")

vs = st.session_state["vector_store"]

# ── Preguntas sugeridas ──
st.markdown("#### 💡 Preguntas frecuentes")
col1, col2 = st.columns(2)
ejemplos = [
    "¿Cuáles son las estructuras neuroanatómicas estudiadas?",
    "¿Qué metodología utilizaron los investigadores?",
    "¿Qué hallazgos morfológicos se reportan?",
    "¿Cuáles son las conclusiones clínicas principales?",
    "¿Qué variaciones anatómicas se identificaron?",
    "¿Qué técnicas de imagen o histología se usaron?",
]
for i, ej in enumerate(ejemplos):
    with (col1 if i % 2 == 0 else col2):
        if st.button(f"🔹 {ej}", key=f"ej_{i}"):
            st.session_state["pregunta_input"] = ej

st.markdown("---")

# ── Input de la consulta ──
st.markdown("#### ❓ Tu consulta científica")
pregunta = st.text_area(
    label="Consulta:",
    value=st.session_state.get("pregunta_input", ""),
    height=120,
    placeholder="Ej: ¿Qué variaciones morfológicas del nervio facial se identificaron en los estudios?",
    key="pregunta_ta",
    label_visibility="collapsed",
)

if st.button("🔬 Consultar al especialista"):
    if not pregunta.strip():
        st.warning("Por favor, escribe una pregunta antes de consultar.")
    else:
        with st.spinner("🧬 Recuperando fragmentos relevantes y generando respuesta..."):
            resultado = consultar(pregunta, vs, k=k_chunks)

        # ── Respuesta ──
        st.markdown("#### 📋 Respuesta del consultor")
        st.markdown(f"""
        <div class="response-card">
            {resultado["respuesta"].replace(chr(10), "<br>")}
        </div>
        """, unsafe_allow_html=True)

        # ── Métricas del pipeline ──
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-chip">📦 {len(resultado["fragmentos"])} fragmentos recuperados</span>
            <span class="metric-chip">🔤 ~{resultado["tokens_contexto_aprox"]} tokens de contexto</span>
            <span class="metric-chip">🤖 gemini-1.5-flash · T=0.1</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Fuentes recuperadas expandibles ──
        st.markdown("#### 📚 Fuentes científicas consultadas")
        for i, doc in enumerate(resultado["fragmentos"], 1):
            fuente = os.path.basename(doc.metadata.get("source", "desconocido"))
            pagina = doc.metadata.get("page", "?")
            with st.expander(f"Fragmento {i} — {fuente} · Pág. {pagina}"):
                st.markdown(f"```\n{doc.page_content[:600]}\n```")

        # ── Descarga del reporte ──
        fuentes_txt = "\n".join([
            f"  [{i+1}] {os.path.basename(d.metadata.get('source','?'))} — Pág. {d.metadata.get('page','?')}"
            for i, d in enumerate(resultado["fragmentos"])
        ])
        reporte = f"""CONSULTA NEUROANATÓMICA — REPORTE RAG
=========================================
PREGUNTA:
{resultado["pregunta"]}

RESPUESTA:
{resultado["respuesta"]}

FUENTES RECUPERADAS:
{fuentes_txt}

=========================================
Generado por: Consultor RAG · Neuroanatomía Científica
Konrad Lorenz · V. García & B. Ramirez · 2026
"""
        st.download_button(
            label="📥 Descargar reporte científico",
            data=reporte,
            file_name="reporte_neuroanatomia.txt",
            mime="text/plain",
        )

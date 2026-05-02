"""
app.py — Interfaz Streamlit del Consultor Científico en Neuroanatomía RAG
Avance 2: Sistema RAG completo con LangChain + ChromaDB + Google Gemini
"""

import streamlit as st
import os
import html as html_module
from dotenv import load_dotenv

# ── Configuración de página (DEBE ser la primera instrucción de Streamlit) ──
st.set_page_config(
    page_title="Consultor Neuroanatomía",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Premium y Responsivo ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --primary: #0ea5e9;
    --primary-dark: #0284c7;
    --bg-base: #0f172a;
    --bg-glass: rgba(30, 41, 59, 0.7);
    --border-glass: rgba(14, 165, 233, 0.2);
    --text-main: #f8fafc;
    --text-muted: #cbd5e1;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .hero-header h1 {
    font-family: 'Outfit', sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at top right, #1e293b 0%, var(--bg-base) 100%);
    color: var(--text-main);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid var(--border-glass);
}

.sidebar-content {
    padding: 1rem;
}

/* Hero Header with Animation */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-header {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(2, 132, 199, 0.05) 100%);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    animation: fadeInDown 0.8s ease-out forwards;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.hero-header h1 {
    color: var(--primary);
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.hero-header p {
    color: var(--text-muted);
    font-size: 1.1rem;
    margin: 10px 0 0 0;
    font-weight: 300;
}

/* Response Card with Glassmorphism */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.response-card {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-left: 4px solid var(--primary);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1rem;
    color: var(--text-main);
    line-height: 1.8;
    font-size: 1.05rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    animation: fadeInUp 0.6s ease-out forwards;
}

/* Metrics row */
.metric-row {
    display: flex;
    gap: 12px;
    margin-top: 20px;
    flex-wrap: wrap;
    animation: fadeInUp 0.8s ease-out forwards;
}

.metric-chip {
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 6px 16px;
    color: #38bdf8;
    font-size: 0.85rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
}

.metric-chip:hover {
    background: rgba(14, 165, 233, 0.2);
    transform: translateY(-2px);
}

/* Input area styling */
.stTextArea textarea {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(14, 165, 233, 0.3) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4) !important;
    background: linear-gradient(135deg, #38bdf8 0%, var(--primary) 100%) !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}

hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
    margin: 2rem 0 !important;
}

/* Author section */
.author-badge {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}

.author-name {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #e2e8f0;
    margin: 5px 0;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none !important;}

</style>
""", unsafe_allow_html=True)

# ── Inicialización de Estado ──
if "historial" not in st.session_state:
    st.session_state.historial = []

# ── Carga de la API key ──
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("🔑 No se encontró GEMINI_API_KEY en el archivo .env")
    st.stop()
os.environ["GOOGLE_API_KEY"] = API_KEY

# ── Importaciones del RAG ──
try:
    from rag_pipeline import build_vector_store, consultar
except ImportError as e:
    st.error(f"Error al importar el pipeline RAG: {e}")
    st.stop()

# ── Carga del vector store ANTES del sidebar (para que el conteo sea correcto) ──
@st.cache_resource(show_spinner=False)
def get_vector_store():
    try:
        return build_vector_store(force_rebuild=False)
    except FileNotFoundError:
        return None  # DB no existe aún — app mostrará botón de reconstrucción
    except Exception as e:
        # DB corrupta — eliminarla para poder reconstruir limpiamente
        if any(k in str(e).lower() for k in ["tenant", "default_tenant", "sqlite", "database"]):
            import shutil as _sh
            _db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_neuro_db")
            if os.path.exists(_db):
                _sh.rmtree(_db)
        return None  # La app mostrará botón de reconstrucción

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = get_vector_store()

# Nombres legibles de PDFs (sidebar + resultados)
mapeo_nombres_sidebar = {
    "0717-9502-ijmorphol-41-04-996.pdf": "Regla Simple para el Aprendizaje de la Neuroanatomía",
    "circir_25_93_2_197-201.pdf": "Modelos 3D y Realidad Aumentada en Neuroanatomía",
    "SCT_2025_1250.pdf": "Tecnologías Inmersivas vs. Convencionales en la Enseñanza",
}

with st.sidebar:
    st.markdown("""
    <div class="author-badge">
        <img src="https://international.udemedellin.edu.co/wp-content/uploads/2021/07/17538-2-40.jpg" width="120" style="border-radius: 8px; margin-bottom: 10px;">
        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Desarrollado por</div>
        <div class="author-name">Viviana García</div>
        <div class="author-name">Braian Ramírez</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Base de conocimientos dinámica - lee archivos reales de Docs/
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    pdfs_disponibles = sorted([f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")])

    with st.expander(f"📂 Base de Conocimientos ({len(pdfs_disponibles)} docs)", expanded=False):

        # ── Subir nuevos PDFs ──
        nuevos_archivos = st.file_uploader(
            "📤 Agregar documentos PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploader_docs",
            help="Sube uno o más PDFs. Luego haz clic en Reconstruir VectorDB."
        )
        # Guardar archivos solo si no los hemos guardado ya en este ciclo
        if nuevos_archivos:
            ya_guardados = st.session_state.get("_uploads_guardados", set())
            nuevos = [uf for uf in nuevos_archivos if uf.name not in ya_guardados]
            if nuevos:
                for uf in nuevos:
                    destino = os.path.join(docs_dir, uf.name)
                    with open(destino, "wb") as f:
                        f.write(uf.getbuffer())
                    ya_guardados.add(uf.name)
                st.session_state["_uploads_guardados"] = ya_guardados
                st.success(f"✅ {len(nuevos)} archivo(s) guardado(s) en Docs/. Haz clic en **Reconstruir VectorDB**.")
        else:
            # Limpiar el set cuando el usuario quita los archivos del uploader
            st.session_state["_uploads_guardados"] = set()

        # ── Lista de PDFs con opción de eliminar ──
        pdfs_actuales = sorted([f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")])
        if pdfs_actuales:
            st.markdown("<div style='margin-top:8px; font-size:0.8rem; color:#64748b;'>Documentos en la base:</div>", unsafe_allow_html=True)
            for pdf in pdfs_actuales:
                nombre = mapeo_nombres_sidebar.get(pdf, pdf.replace(".pdf", "").replace("-", " ").replace("_", " "))
                col_n, col_d = st.columns([5, 1])
                with col_n:
                    st.markdown(
                        f"<div style='font-size:0.82rem; padding:5px 8px; background:rgba(255,255,255,0.05); "
                        f"border-radius:6px; border-left:2px solid #0ea5e9; color:#cbd5e1; "
                        f"margin-bottom:3px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;' "
                        f"title='{pdf}'>📄 {nombre}</div>",
                        unsafe_allow_html=True
                    )
                with col_d:
                    if st.button("🗑️", key=f"del_{pdf}", help=f"Eliminar {pdf}"):
                        st.session_state["_pending_delete"] = pdf

            # Confirmación de eliminación
            pending = st.session_state.get("_pending_delete", None)
            if pending and pending in pdfs_actuales:
                nombre_pending = mapeo_nombres_sidebar.get(pending, pending)
                st.warning(f"⚠️ ¿Eliminar **{nombre_pending}**? Esta acción no se puede deshacer.")
                col_si, col_no = st.columns(2)
                with col_si:
                    if st.button("✅ Sí, eliminar", key="confirmar_delete", use_container_width=True):
                        os.remove(os.path.join(docs_dir, pending))
                        st.session_state["_pending_delete"] = None
                        st.session_state["_uploads_guardados"] = set()
                        st.rerun()
                with col_no:
                    if st.button("❌ Cancelar", key="cancelar_delete", use_container_width=True):
                        st.session_state["_pending_delete"] = None
                        st.rerun()
        else:
            st.info("💭 No hay documentos. Sube un PDF para comenzar.")

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros del Motor")
    k_chunks = st.slider("Fragmentos a recuperar (k)", min_value=3, max_value=8, value=5, help="Mayor cantidad trae más contexto pero consume más tokens.")
    
    st.markdown("""
    <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; font-size: 0.82rem; color: #94a3b8; margin-top: 10px;">
        <b>LLM:</b> Gemini Flash
        <span title="Modelo de lenguaje que genera la respuesta final usando los fragmentos recuperados. Gemini Flash es rápido y gratuito en la capa básica." style="cursor:help; color:#0ea5e9;"> ℹ️</span><br>
        <b>Embeddings:</b> gemini-embedding-001
        <span title="Modelo que convierte texto en vectores numéricos para la búsqueda semántica. Compara tu pregunta con los fragmentos de los PDFs para encontrar los más relevantes." style="cursor:help; color:#0ea5e9;"> ℹ️</span><br>
        <b>Temp:</b> 0.1
        <span title="Temperatura del LLM (0 = respuestas muy precisas y deterministas, 1 = respuestas más creativas y variables). 0.1 asegura respuestas académicas concretas y reproducibles." style="cursor:help; color:#0ea5e9;"> ℹ️</span>
        | <b>Chunk:</b> 600
        <span title="Tamaño de cada fragmento de texto en caracteres. Los PDFs se dividen en trozos de 600 caracteres para la búsqueda vectorial. Fragmentos más pequeños = más precisión; más grandes = más contexto." style="cursor:help; color:#0ea5e9;"> ℹ️</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    db_existe = os.path.exists(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_neuro_db")
    )

    # Conteo real desde la base (si ya está cargada en session_state)
    try:
        _vs_temp = st.session_state.get("vector_store", None)
        _sidebar_count = _vs_temp._collection.count() if _vs_temp is not None else 0
    except Exception:
        _sidebar_count = 0

    if _sidebar_count > 0:
        st.success(f"✅ Base lista — {_sidebar_count} vectores indexados")
    elif db_existe:
        st.warning("⚠️ Base detectada pero vacía — Reconstruye")
    else:
        st.error("❌ Base no encontrada — Reconstruye")


    if st.button("🔄 Reconstruir VectorDB", use_container_width=True):
        st.warning("⏳ Reconstruyendo... tarda ~3-5 min. No cierres la ventana.")
        with st.spinner("🔬 Leyendo PDFs → Chunks → Vectorizando → ChromaDB..."):
            try:
                # Liberar la conexión activa ANTES de reconstruir
                # (ChromaDB usa SQLite, no soporta dos procesos simultáneos)
                vs_actual = st.session_state.pop("vector_store", None)
                if vs_actual is not None:
                    try:
                        vs_actual._client._system.stop()
                    except Exception:
                        pass
                    del vs_actual
                st.cache_resource.clear()

                nuevo_vs = build_vector_store(force_rebuild=True)
                st.session_state["vector_store"] = nuevo_vs
                st.success(f"✅ ¡Base vectorial reconstruida — {nuevo_vs._collection.count()} vectores!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        st.rerun()

    # Historial en sidebar
    if st.session_state.historial:
        st.markdown("---")
        st.markdown("### 🕒 Consultas Recientes")
        for i, h in enumerate(reversed(st.session_state.historial[-5:])):
            st.markdown(f"<div style='font-size: 0.85rem; padding: 6px 10px; background: rgba(255,255,255,0.05); border-radius: 6px; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border: 1px solid rgba(255,255,255,0.05);' title='{html_module.escape(h)}'>💬 {html_module.escape(h)}</div>", unsafe_allow_html=True)

# ── Header principal ──
st.markdown("""
<div class="hero-header">
    <h1>🧠 Consultor IA en Neuroanatomía</h1>
    <p>Asistente RAG fundamentado exclusivamente en literatura científica</p>
</div>
""", unsafe_allow_html=True)

vs = st.session_state["vector_store"]

# ── Auto-detección de base vacía ──
try:
    _count = vs._collection.count()
except Exception:
    _count = 0

if _count == 0:
    st.warning("⚠️ **Base vectorial vacía** — los documentos aún no han sido indexados.")
    st.info("🔄 Haz clic en **Reconstruir VectorDB** en el panel izquierdo. Tarda ~3 minutos la primera vez porque procesa los PDFs y genera los embeddings.")
    st.stop()

# ── CSS para Ctrl+Enter ──
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const btn = document.querySelector('button[kind="primary"]');
        if (btn) btn.click();
    }
});
</script>
""", unsafe_allow_html=True)

# ── Layout principal (Responsive: columnas en desktop) ──
col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.markdown("### 💡 Ejemplos de consulta")
    ejemplos = [
        ("¿Cuáles son las estructuras neuroanatómicas estudiadas?",
         "Pregunta directa sobre el contenido de los artículos. Trae fragmentos con descripciones morfológicas."),
        ("¿Qué metodología utilizaron los investigadores?",
         "Consulta el diseño de investigación, tipo de estudio y métodos usados en los 3 artículos."),
        ("¿Qué hallazgos morfológicos se reportan?",
         "Busca resultados anatomícos concretos: medidas, variaciones, características de tejidos."),
        ("¿Qué técnicas de imagen o histología se usaron?",
         "Identifica los métodos de visualización: resonancia, tomografía, tinción histológica, etc.")
    ]
    for i, (ej, tooltip) in enumerate(ejemplos):
        if st.button(f"👉 {ej}", key=f"ej_{i}", use_container_width=True, help=tooltip):
            st.session_state["pregunta_ta"] = ej
            st.rerun()

with col1:
    st.markdown("### ❓ Tu consulta científica")
    pregunta = st.text_area(
        label="Consulta:",
        height=130,
        placeholder="Ej: ¿Qué variaciones morfológicas del nervio facial se identificaron en los estudios?",
        key="pregunta_ta",
        label_visibility="collapsed",
    )

    consultar_btn = st.button("🔬 Analizar Literatura", use_container_width=True)

# ── Procesamiento de la Consulta ──
mapeo_nombres = {
    "0717-9502-ijmorphol-41-04-996.pdf": "Regla Simple para el Aprendizaje de la Neuroanatomía",
    "circir_25_93_2_197-201.pdf": "Modelos 3D y Realidad Aumentada en Neuroanatomía",
    "SCT_2025_1250.pdf": "Tecnologías Inmersivas vs. Convencionales en la Enseñanza"
}

if consultar_btn:
    if not pregunta.strip():
        st.warning("⚠️ Por favor, escribe una pregunta antes de consultar.")
    else:
        st.session_state.historial.append(pregunta)
        try:
            with st.spinner("🧬 Analizando vectores y sintetizando respuesta..."):
                resultado = consultar(pregunta, vs, k=k_chunks)

            st.markdown("---")
            st.markdown("### 📋 Síntesis del Consultor")
            
            # Escapar HTML para evitar crash
            respuesta_segura = html_module.escape(resultado["respuesta"]).replace("\n", "<br>")
            
            st.markdown(
                f'<div class="response-card">{respuesta_segura}</div>',
                unsafe_allow_html=True,
            )

            # Métricas con tooltips
            st.markdown(f"""
            <div class="metric-row">
                <span class="metric-chip" title="Chunks recuperados: fragmentos de los PDFs que el sistema consideró más relevantes para tu pregunta. Más chunks = más contexto, pero más tokens consumidos.">📄 {len(resultado["fragmentos"])} chunks recuperados</span>
                <span class="metric-chip" title="Tokens de contexto: estimación de la cantidad de texto enviado al modelo Gemini. 1 token ≈ 4 caracteres. La API gratuita tiene límite de tokens por minuto.">⚡ ~{resultado["tokens_contexto_aprox"]} tokens de contexto</span>
                <span class="metric-chip" title="Modelo LLM generador: Gemini Flash es el modelo de Google usado para generar la respuesta a partir de los fragmentos recuperados. Es rápido y eficiente.">🧠 Gemini Flash</span>
                <span class="metric-chip" title="Temperatura 0.1: controla la creatividad del modelo. 0.0 = muy literal y preciso, 1.0 = muy creativo. Para consultas científicas usamos 0.1 para mayor exactitud.">⏱️ Temp: 0.1</span>
            </div>
            """, unsafe_allow_html=True)

            # Detectar si la pregunta es un saludo / no científica
            SALUDOS = {"hola", "hello", "hi", "buenas", "buenos días", "buenas tardes",
                       "buenas noches", "gracias", "de nada", "ok", "okay", "sí", "no",
                       "perfecto", "genial", "bien", "mal", "cómo estás", "adios", "bye"}
            pregunta_corta = len(pregunta.strip()) < 12
            es_saludo = any(s in pregunta.strip().lower() for s in SALUDOS)

            # Detectar si la RESPUESTA indica que no hay información
            NO_INFO_PHRASES = [
                "no se encuentra en los documentos",
                "no está en los documentos",
                "no hay información",
                "no tengo información",
                "plantee su consulta",
                "formula una pregunta",
                "pregunta específica",
                "no encontr",
            ]
            es_respuesta_sin_info = any(p in resultado["respuesta"].lower() for p in NO_INFO_PHRASES)

            mostrar_evidencia = not (pregunta_corta or es_saludo or es_respuesta_sin_info)

            if mostrar_evidencia:
                st.markdown("<br>### 📚 Evidencia Documental", unsafe_allow_html=True)
                for i, doc in enumerate(resultado["fragmentos"], 1):
                    file_name = os.path.basename(doc.metadata.get("source", "desconocido"))
                    nombre_revista = mapeo_nombres.get(file_name, file_name)
                    pagina = doc.metadata.get("page", "?")
                    with st.expander(f"📌 Fragmento {i} — {nombre_revista} (Pág. {pagina})"):
                        st.markdown(f"<div style='font-size:0.95rem; color:#cbd5e1; line-height:1.6; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;'>{html_module.escape(doc.page_content)}</div>", unsafe_allow_html=True)

            # Botón de descarga
            fuentes_txt = "\n".join([
                f"  [{i+1}] {mapeo_nombres.get(os.path.basename(d.metadata.get('source','?')), os.path.basename(d.metadata.get('source','?')))} — Pág. {d.metadata.get('page','?')}"
                for i, d in enumerate(resultado["fragmentos"])
            ])
            reporte = f"""=========================================
CONSULTA NEUROANATÓMICA — REPORTE RAG
=========================================
FECHA: 2026

PREGUNTA:
{resultado["pregunta"]}

RESPUESTA:
{resultado["respuesta"]}

FUENTES RECUPERADAS:
{fuentes_txt}

=========================================
Generado por: Consultor IA en Neuroanatomía
Autores: Viviana García & Braian Ramírez
Universidad Konrad Lorenz
=========================================
"""
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
            with col_dl2:
                st.download_button(
                    label="📥 Exportar Reporte Académico (TXT)",
                    data=reporte,
                    file_name="reporte_neuroanatomia.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error al procesar la consulta: {str(e)}")
            st.info("💡 Intenta reformular tu pregunta o reconstruye la base vectorial desde el panel lateral.")

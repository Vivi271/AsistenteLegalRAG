"""
app.py — Punto de entrada limpio del Consultor de Neuroanatomía RAG
"""
import streamlit as st
import os
import time

# ── 1. Configuración de página (DEBE ser la primera instrucción de Streamlit) ──
st.set_page_config(
    page_title="Consultor Neuroanatomía",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Cargar y aplicar estilos CSS desde style.css ──
CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# ── 3. Inicialización de Estado ──
if "historial" not in st.session_state:
    st.session_state.historial = []
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

import urllib.request as _urllib_request
import os as _os

_gemini_key = _os.environ.get("GEMINI_API_KEY", _os.environ.get("GOOGLE_API_KEY", ""))

if not _gemini_key:
    _ollama_host = _os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not _ollama_host.startswith("http"):
        _ollama_host = f"http://{_ollama_host}"

    try:
        _urllib_request.urlopen(f"{_ollama_host}/api/tags", timeout=3)
    except Exception:
        st.error("Ollama no está corriendo. Inicia Ollama en tu PC y recarga la página.")
        st.info(f"Asegúrate de que Ollama esté activo en: {_ollama_host}\n\nEn macOS abre la aplicación Ollama o ejecuta `ollama serve` en la terminal.")
        st.stop()

try:
    from rag_pipeline import build_vector_store, consultar
    from database import registrar_consulta, registrar_evaluacion, obtener_preguntas_por_nivel
except ImportError as e:
    st.error(f"Error al importar módulos del sistema: {e}")
    st.stop()

# Carga del vector store (cacheado globalmente)
@st.cache_resource
def get_vector_store():
    try:
        return build_vector_store(force_rebuild=False)
    except Exception as e:
        err = str(e).lower()
        if "default_tenant" in err and "does not exist" in err:
            import shutil as _sh
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_neuro_db")
            if os.path.exists(db_path):
                _sh.rmtree(db_path)
        return None

vs = get_vector_store()

# Auto-detección de base vacía al arrancar
try:
    _count = vs._collection.count() if vs is not None else 0
except Exception:
    _count = 0

if _count == 0:
    st.warning("Base de datos de conocimiento vacía.")
    st.info("Inicia sesión como administrador en la barra lateral y haz clic en 'Reconstruir VectorDB' para indexar tus archivos PDFs.")
    # Permitir que renderice de todos modos para que el admin pueda ingresar el PIN
    
# Cargar Componentes de Interfaz
from components.header import render_header
from components.sidebar import render_sidebar
from components.consultor import render_consultor
from components.resultados import render_resultados
from components.admin_panel import render_admin_panel

# --- DIALOG DE AUTOEVALUACIÓN ---
@st.dialog("Autoevaluación de Neuroanatomía", width="large")
def mostrar_evaluacion(nivel):
    st.caption(f"Responde el cuestionario para evaluar tus conocimientos del nivel {nivel} basados en el material del laboratorio.")
    db_preguntas = obtener_preguntas_por_nivel(nivel)
    preguntas_quiz = []
    
    for i, q in enumerate(db_preguntas, 1):
        opciones = [
            f"A) {q['opcion_a']}",
            f"B) {q['opcion_b']}",
            f"C) {q['opcion_c']}",
            f"D) {q['opcion_d']}"
        ]
        letra_correcta = q['correcta'].upper()
        opcion_correcta = ""
        if letra_correcta == 'A': opcion_correcta = opciones[0]
        elif letra_correcta == 'B': opcion_correcta = opciones[1]
        elif letra_correcta == 'C': opcion_correcta = opciones[2]
        elif letra_correcta == 'D': opcion_correcta = opciones[3]
        
        preguntas_quiz.append({
            "id": q['id'],
            "num": i,
            "pregunta": q['pregunta'],
            "opciones": opciones,
            "correcta": opcion_correcta,
            "explicacion": q['explicacion']
        })
        
    if not preguntas_quiz:
        st.info(f"No hay preguntas de evaluación registradas para el nivel {nivel}.")
        return
        
    if "quiz_respuestas" not in st.session_state:
        st.session_state.quiz_respuestas = {}
    if "quiz_evaluado" not in st.session_state:
        st.session_state.quiz_evaluado = False
        
    form_quiz = st.form(key="evaluacion_form_dialog")
    with form_quiz:
        for p in preguntas_quiz:
            st.markdown(f"**{p['num']}. {p['pregunta']}**")
            st.session_state.quiz_respuestas[p['id']] = st.radio(
                "Selecciona una opción:",
                options=p["opciones"],
                key=f"q_dlg_{p['id']}",
                label_visibility="collapsed"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            
        enviar_btn = st.form_submit_button("Enviar Respuestas")
        
    if enviar_btn:
        st.session_state.quiz_evaluado = True
        
    if st.session_state.quiz_evaluado:
        st.markdown("#### Resultados de tu evaluación")
        aciertos = 0
        for p in preguntas_quiz:
            resp_usr = st.session_state.quiz_respuestas.get(p['id'])
            if not resp_usr:
                resp_usr = p["opciones"][0]
            es_correcta = p['correcta'] in resp_usr
            if es_correcta:
                aciertos += 1
                st.success(f"**Pregunta {p['num']}:** Correcto. Tu respuesta: {resp_usr}\n\n{p['explicacion']}")
            else:
                st.error(f"**Pregunta {p['num']}:** Incorrecto. Tu respuesta: {resp_usr} (Correcta: {p['correcta']})\n\n{p['explicacion']}")
                
            if not st.session_state.get(f"dlg_q_logged_{p['id']}"):
                registrar_evaluacion(p['pregunta'], resp_usr, p['correcta'], es_correcta, p['explicacion'])
                st.session_state[f"dlg_q_logged_{p['id']}"] = True
                
        nota = (aciertos / len(preguntas_quiz)) * 5.0
        st.metric("Calificación Final", f"{nota:.2f} / 5.00", f"{aciertos} de {len(preguntas_quiz)} correctas")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Reiniciar Examen", key="reset_quiz_dlg_btn"):
                st.session_state.quiz_evaluado = False
                for p in preguntas_quiz:
                    st.session_state.pop(f"dlg_q_logged_{p['id']}", None)
                st.rerun()
        with col_r2:
            if st.button("Cerrar Ventana", key="close_quiz_dlg_btn"):
                st.session_state.quiz_evaluado = False
                for p in preguntas_quiz:
                    st.session_state.pop(f"dlg_q_logged_{p['id']}", None)
                st.rerun()

# ── 4. Renderizar Componentes de UI ──
render_header()
with st.sidebar:
    nivel, k_chunks, is_admin, lanzar_evaluacion, vs = render_sidebar(vs)

# Activar dialog de autoevaluación si se pulsó el botón
if lanzar_evaluacion:
    mostrar_evaluacion(nivel)

# Panel de administración (pestañas RAGAS y preguntas)
if is_admin:
    render_admin_panel()

# Consultor RAG (caja de texto y ejemplos)
pregunta, consultar_btn, col_principal = render_consultor()

# ── 5. Procesamiento de la Consulta en Tiempo Real ──
if consultar_btn:
    if not pregunta.strip():
        st.warning("Por favor, escribe una pregunta antes de consultar.")
    else:
        st.session_state.historial.append(pregunta)
        waiting_area = st.empty()
        
        # Animación de espera
        waiting_area.markdown(
            """
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;padding:40px 0;">
                <div class="loader-dots" style="display:flex;gap:8px;">
                    <span style="background:#5dade2;width:12px;height:12px;border-radius:50%;display:inline-block;animation:bounce 1.4s infinite ease-in-out both;animation-delay:-0.32s;"></span>
                    <span style="background:#5dade2;width:12px;height:12px;border-radius:50%;display:inline-block;animation:bounce 1.4s infinite ease-in-out both;animation-delay:-0.16s;"></span>
                    <span style="background:#5dade2;width:12px;height:12px;border-radius:50%;display:inline-block;animation:bounce 1.4s infinite ease-in-out both;"></span>
                </div>
                <span style="color:#5dade2;font-size:16px;font-weight:600;letter-spacing:.5px;">
                    Pensando la respuesta, por favor espera un momento...
                </span>
            </div>
            <style>
            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1.0); }
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        try:
            t_start = time.time()
            resultado = consultar(pregunta, vs, k=k_chunks, nivel=nivel)
            latencia = time.time() - t_start

            waiting_area.empty()
            registrar_consulta(pregunta, resultado["respuesta"], nivel.lower(), latencia)
            st.session_state["_ultimo_resultado"] = resultado
            st.session_state["_ultima_pregunta"] = pregunta
            st.rerun()

        except Exception as e:
            err_str = str(e)
            waiting_area.empty()
            if "1032" in err_str or "readonly" in err_str.lower():
                try:
                    vs_fresh = get_vector_store()
                    t_start = time.time()
                    resultado = consultar(pregunta, vs_fresh, k=k_chunks, nivel=nivel)
                    latencia = time.time() - t_start
                    registrar_consulta(pregunta, resultado["respuesta"], nivel.lower(), latencia)
                    st.session_state["_ultimo_resultado"] = resultado
                    st.session_state["_ultima_pregunta"] = pregunta
                    st.rerun()
                except Exception as e2:
                    st.error(f"Error de base de datos: {str(e2)[:200]}")
            else:
                st.error(f"Error: {err_str[:200]}")

# Mostrar resultado si está guardado en session_state
if st.session_state.get("_ultimo_resultado"):
    render_resultados(st.session_state["_ultimo_resultado"], col_principal)

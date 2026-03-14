import streamlit as st
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pypdf import PdfReader
import io

# Configuración estética de la app
st.set_page_config(page_title="Asistente Legal RAG", page_icon="⚖️", layout="centered")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .report-card {
        padding: 25px;
        border-radius: 12px;
        background-color: white;
        color: #1f2937;
        border-left: 6px solid #007bff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Carga de configuración
load_dotenv(override=True)
if not os.getenv("GEMINI_API_KEY"):
    st.error("Error: GEMINI_API_KEY no configurada en .env")
    st.stop()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error al leer el PDF: {e}")
        return ""

def procesar_rag(consulta, documento):
    # 1. System Instruction (Configuración del Sistema)
    sys_instruction = """Eres un Asistente Legal experto. Tu tarea es analizar consultas basadas UNICAMENTE en el texto proporcionado en <contexto>.
    Si la información no está en el contexto, indica que no se encontró base legal.
    
    REGLA DE FORMATO: Debes responder OBLIGATORIAMENTE en formato JSON estricto."""

    # 2. Few-Shot Prompting (Ejemplos para guiar el formato y razonamiento)
    ejemplos = """
    EJEMPLO 1:
    <contexto>Art 1. La jornada semanal es de 40 horas.</contexto>
    <consulta>¿Puedo trabajar 50 horas?</consulta>
    Respuesta: {"es_valido": false, "ref": "Art 1", "explica": "La jornada máxima permitida es de 40 horas según el reglamento.", "riesgo": "Alto"}

    EJEMPLO 2:
    <contexto>Art 5. Se permite el teletrabajo para cargos administrativos.</contexto>
    <consulta>Soy secretaria, ¿puedo pedir teletrabajo?</consulta>
    Respuesta: {"es_valido": true, "ref": "Art 5", "explica": "Su cargo es administrativo y el artículo 5 lo permite.", "riesgo": "Bajo"}
    """
    
    # 3. Estrategia de Delimitadores (XML Tags)
    prompt = f"{ejemplos}\n\nAHORA ANALIZA LO SIGUIENTE:\n<contexto>\n{documento}\n</contexto>\n\n<consulta>\n{consulta}\n</consulta>"

    # Generación con Gemini
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    return json.loads(response.text)

# --- Interfaz de Usuario ---
st.title("⚖️ Asistente Legal Inteligente")
st.write("Carga tu reglamento en PDF o pega el texto para analizar casos legales con precisión.")

st.divider()

# Sección de entrada de conocimiento
st.subheader("1. Base de Conocimientos (Documento Legal)")

# Opción de subir PDF
uploaded_file = st.file_uploader("📂 Sube un reglamento en PDF", type=["pdf"])

pdf_text = ""
if uploaded_file is not None:
    with st.spinner("Extrayendo texto del PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("Texto extraído correctamente del PDF.")

# Área de texto (se llena con el PDF o manualmente)
default_text = "Reglamento Interno de Seguridad - Art. 22:\nEl uso de teléfonos móviles personales está terminantemente prohibido durante la operación de vehículos o maquinaria pesada. La infracción de esta norma de seguridad constituye una falta gravísima que justifica el despido inmediato."
doc_content = st.text_area(
    "✏️ Contenido legal (Texto extraído o pegado):", 
    height=200, 
    value=pdf_text if pdf_text else default_text
)

# Sección de consulta
st.subheader("2. Caso a Analizar")
query_area = st.text_area(
    "❓ Consulta o situación a evaluar:", 
    height=100, 
    placeholder="Ej: ¿Es legal despedir a alguien por usar el celular en la bodega?"
)

if st.button("🚀 Iniciar Análisis Legal"):
    if not query_area.strip():
        st.warning("Por favor, ingresa una consulta para analizar.")
    elif not doc_content.strip():
        st.warning("Por favor, proporciona un documento legal (PDF o texto).")
    else:
        try:
            with st.spinner("Analizando con IA..."):
                resultado = procesar_rag(query_area, doc_content)
                
                # Mostrar resultados
                st.markdown("### 📋 Resultados del Análisis")
                if resultado.get("es_valido"):
                    st.success("Acción legalmente procedente (VÁLIDA)")
                else:
                    st.error("Acción no procedente (INVÁLIDA)")
                
                st.markdown(f"""
                <div class="report-card">
                    <strong>Ref. Legal:</strong> {resultado.get('ref')}<br><br>
                    <strong>Análisis:</strong> {resultado.get('explica')}<br><br>
                    <strong>Nivel de Riesgo:</strong> {resultado.get('riesgo')}
                </div>
                """, unsafe_allow_html=True)

                # Botón para descargar el reporte
                reporte_txt = f"""INFORME DE ANÁLISIS LEGAL
-------------------------
RESULTADO: {"VÁLIDO" if resultado.get("es_valido") else "INVÁLIDO"}
REFERENCIA: {resultado.get('ref')}
RIESGO: {resultado.get('riesgo')}

ANÁLISIS:
{resultado.get('explica')}

Generado por: Asistente Legal RAG (V. García & B. Ramirez)
"""
                st.download_button(
                    label="📥 Descargar Informe del Caso (TXT)",
                    data=reporte_txt,
                    file_name="informe_legal.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            if "429" in str(e):
                st.error("Límite de cuota excedido. Por favor, espera 60 segundos y reintenta.")
            else:
                st.error(f"Error técnico: {str(e)}")

# Sidebar institucional
st.sidebar.image("https://ascofapsi.org.co/wp-content/uploads/2022/06/konrad_lorenz.png", width=180)
st.sidebar.markdown("---")
st.sidebar.markdown("**Integrantes:**")
st.sidebar.markdown("- Viviana García")
st.sidebar.markdown("- Braian Ramirez")
st.sidebar.markdown("---")
st.sidebar.markdown("Proyecto RAG - Avance 1")

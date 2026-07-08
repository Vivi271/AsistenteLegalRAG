import streamlit as st

def render_header():
    st.markdown("""
    <div class="hero-header">
        <div style="display:flex; align-items:center; justify-content:center; gap:16px; margin-bottom:12px;">
            <div style="font-size:1.2rem; font-weight:800; color:#5dade2; font-family:'Outfit',sans-serif; line-height:1.1; text-align:left;">
                <span style="color:#ffffff;">KONRAD</span> <span style="color:#5dade2;">LORENZ</span>
            </div>
            <div style="width:1px; height:26px; background:rgba(16, 185, 129, 0.3);"></div>
            <div style="text-align:left; line-height:1.25;">
                <div style="font-size:0.68rem; color:#e2e8f0; text-transform:uppercase; letter-spacing:1px; font-weight:600;">Fundación Universitaria</div>
                <div style="font-size:0.65rem; color:#94a3b8; font-weight:500;">Acreditación Institucional de Alta Calidad</div>
            </div>
        </div>
        <h1>Consultor IA en Neuroanatomía</h1>
        <p>Asistente RAG fundamentado exclusivamente en literatura científica</p>
        <div class="kl-subtitle">Programa de Psicología · Laboratorio de Neurociencias</div>
    </div>
    """, unsafe_allow_html=True)

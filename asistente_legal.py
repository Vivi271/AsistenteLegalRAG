import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(override=True)

# Configurar Cliente
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analizar_consulta_legal(consulta_usuario: str, documento_contexto: str):
    """
    Simula la fase de Generación (Generation) en un flujo RAG (Retrieval-Augmented Generation),
    donde ya se ha recuperado un documento de la base de conocimientos y se le pasa a la IA.
    """
    
    # 1. DISEÑO DE PROMPTS: System Prompt estructurado. Define el rol, comportamiento y fomato.
    system_instruction = """Eres un Asistente Legal Analítico experto en leyes laborales y reglamentos de la empresa.
Tu objetivo es analizar las consultas de los usuarios y determinar la validez legal basándote ÚNICAMENTE en los documentos de referencia proporcionados.

Reglas estrictas de comportamiento:
1. No inventes información (cero alucinaciones). Si la respuesta no está en el documento, indica que no hay información en "explicacion" y pon "es_valido" en null.
2. Tu respuesta DEBE ser obligatoriamente un objeto JSON válido para ser consumido por una API.
3. El JSON debe respetar exactamente esta estructura:
   {
     "es_valido": boolean,
     "articulo_referencia": "string o null",
     "explicacion": "string",
     "nivel_riesgo_legal": "Alto" | "Medio" | "Bajo"
   }
"""

    # 2. FEW-SHOT PROMPTING: Se incluyen ejemplos dentro del prompt para guiar la salida de la IA
    # hacia el formato JSON estricto esperado, demostrando cómo extraer el artículo y razonar.
    few_shot_examples = """
A continuación te muestro ejemplos de cómo debes razonar y formatear tus respuestas:

--- EJEMPLO 1 ---
<contexto_legal>
Art. 10: El despido justificado requiere faltas graves y reiteradas. Llegadas tardías menores a 10 minutos no son causal de despido inmediato sin al menos 3 advertencias previas formales.
</contexto_legal>
<consulta>
Quiero despedir a un empleado de contabilidad porque llegó 5 minutos tarde ayer por primera vez.
</consulta>

Respuesta Esperada:
{
  "es_valido": false,
  "articulo_referencia": "Art. 10",
  "explicacion": "Una única llegada tardía de 5 minutos no se considera una falta grave que justifique el despido sin advertencias previas formales según el Artículo 10.",
  "nivel_riesgo_legal": "Alto"
}

--- EJEMPLO 2 ---
<contexto_legal>
Art. 45: Los trabajadores tienen derecho a 15 días hábiles de vacaciones remuneradas tras completar exactamente un año continuo de servicio en la organización.
</contexto_legal>
<consulta>
Llevo trabajando acá 14 meses ininterrumpidos. ¿Cuántos días de vacaciones me corresponden?
</consulta>

Respuesta Esperada:
{
  "es_valido": true,
  "articulo_referencia": "Art. 45",
  "explicacion": "Al superar el año continuo de servicio (14 meses), tiene derecho a 15 días hábiles de vacaciones remuneradas.",
  "nivel_riesgo_legal": "Bajo"
}
"""

    # 3. ESTRATEGIAS DE DELIMITADORES: Uso de etiquetas XML (<contexto_legal>, <consulta>) 
    # para separar claramente la información de referencia de la pregunta del usuario.
    prompt_usuario = f"""
Por favor, analiza la siguiente consulta real basándote EXCLUSIVAMENTE en el contexto legal proporcionado.

<contexto_legal>
{documento_contexto}
</contexto_legal>

<consulta>
{consulta_usuario}
</consulta>
"""

    # Combinamos el Few-Shot con el Prompt final
    prompt_final = f"{few_shot_examples}\n\n{prompt_usuario}"

    try:
        # Generar contenido usando el nuevo SDK
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt_final,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1 # Nivel de creatividad bajo (0.1) para que sea analítico y estricto
            )
        )
        
        # Parsear la respuesta a diccionario Python para asegurar que devolvió un JSON correcto
        resultado_json = json.loads(response.text)
        return resultado_json

    except Exception as e:
        return {"error": f"Error al procesar con Gemini: {str(e)}"}

if __name__ == "__main__":
    print("=========================================================")
    print("⚖️ ASISTENTE LEGAL RAG - PRUEBA DE PROMPTS EXPERTOS ⚖️")
    print("=========================================================\n")
    
    # Simulación de un documento recuperado de la base vectorial
    contexto_simulado = """
Reglamento Interno de Seguridad - Art. 22:
El uso de teléfonos móviles personales está terminantemente prohibido durante la operación de vehículos o maquinaria pesada. 
La infracción de esta norma de seguridad constituye una falta gravísima que deriva en despido justificado inmediato sin derecho a preaviso por poner en riesgo la vida de los trabajadores.
    """
    
    consulta_real = "Identificamos a un operario de montacargas viendo videos en su celular móvil mientras movía estibas en la bodega. Queremos terminar su contrato laboral hoy mismo. ¿Procedemos?"
    
    print(f"[1] Buscando en Base de Conocimientos (Recuperación RAG simulada)...")
    print(f"Contexto Recuperado:\n{contexto_simulado}")
    
    print(f"[2] Consulta del Usuario:\n{consulta_real}\n")
    print("[3] Enviando a Gemini con (System Prompt + Few-Shot + XML Delimiters)...\n")
    
    resultado = analizar_consulta_legal(consulta_real, contexto_simulado)
    
    print("RESULTADO DEL ASISTENTE (JSON ESTRUCTURADO):")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

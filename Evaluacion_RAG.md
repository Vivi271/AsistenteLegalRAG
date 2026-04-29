# Evaluación RAG - Asistente Legal

## Objetivo
Medir el desempeño del pipeline RAG implementado usando la librería RAGAS y presentar los resultados con análisis crítico.

## Preparación de la Evaluación

| Parámetro | Valor |
| :--- | :--- |
| **Documento(s)** | Documentos de la Base Legal (Reglamento Interno, Políticas) |
| **Modelo de embeddings** | gemini-embedding-001 (Google) |
| **chunk_size / overlap** | 500 caracteres / 50 caracteres de solapamiento |
| **k (chunks recuperados)** | 5 |
| **LLM generador** | gemini-3.1-flash-lite-preview (Temperatura 0.1) |
| **LLM juez (RAGAS)** | gemini-3.1-flash-lite-preview |

## Casos de Prueba

Se definieron 8 casos de prueba divididos en las siguientes categorías obligatorias:

1. **La respuesta está textualmente en el documento**
   - *Pregunta 1:* ¿Cuál es la jornada laboral semanal máxima permitida?
   - *Pregunta 2:* ¿A qué tipo de sanción se expone un trabajador si abandona el sitio de trabajo sin previo aviso?
2. **Vocabulario diferente al del documento (prueba de embeddings)**
   - *Pregunta 3:* ¿Me echan del trabajo si me pongo a chatear por el celular mientras manejo el montacargas? (Prueba de sinonimia con "uso de teléfono móvil en maquinaria pesada").
   - *Pregunta 4:* ¿Puedo solicitar trabajar desde mi casa si pertenezco a la junta directiva? (Prueba de sinonimia con "teletrabajo para cargos administrativos").
3. **Requiere combinar información de varios chunks**
   - *Pregunta 5:* Si llego tarde por una cita médica, ¿qué documentos debo presentar a recursos humanos y en qué plazo?
   - *Pregunta 6:* ¿Cuáles son los pasos completos para reportar un accidente laboral según las normativas vigentes?
4. **El sistema no debería tener la respuesta (detecta alucinaciones)**
   - *Pregunta 7:* ¿Cuál es el menú del casino o cafetería de la empresa para los días viernes?
   - *Pregunta 8:* ¿Cuáles son los requisitos legales para afiliar a mi mascota a la EPS de la empresa?

## Resultados de RAGAS

| Pregunta | Faithfulness | answer_relevancy | context_precision | Análisis |
| :--- | :--- | :--- | :--- | :--- |
| ¿Cuál es la jornada laboral semanal máxima permitida? | 1.00 | 0.95 | 0.80 | Excelente desempeño. El modelo recuperó la regla exacta y la respuesta está totalmente fundamentada en el contexto. |
| ¿A qué tipo de sanción se expone un trabajador si abandona el sitio de trabajo sin previo aviso? | 1.00 | 0.92 | 0.85 | Buena recuperación del contexto. La respuesta es directa y no presenta alucinaciones. |
| ¿Me echan del trabajo si chateo manejando el montacargas? | 1.00 | 0.88 | 0.60 | Los embeddings lograron captar la similitud semántica, pero se trajeron fragmentos adicionales de políticas móviles generales que bajaron un poco la precisión de contexto. |
| ¿Puedo solicitar trabajar desde mi casa si soy gerente? | 1.00 | 0.85 | 0.65 | Respondió correctamente conectando los roles con cargos elegibles, sin inventar información adicional. |
| Si llego tarde por cita médica, ¿qué presento y cuándo? | 0.90 | 0.90 | 0.50 | El modelo combinó chunks de justificación de ausencias y plazos de RRHH de manera coherente, con mínima pérdida de contexto. |
| ¿Cuáles son los pasos para reportar un accidente laboral? | 0.95 | 0.94 | 0.55 | Aunque la respuesta fue excelente y fiel, la precisión del contexto fue media ya que la política estaba muy dispersa en el documento. |
| ¿Cuál es el menú de la empresa para los viernes? | 1.00 | 0.00 | 0.00 | Correctamente indicó que no tiene información. Faithfulness se mantiene en 1.0 porque no alucinó datos, simplemente denegó la respuesta. |
| ¿Requisitos legales para afiliar mi mascota a la EPS? | 1.00 | 0.00 | 0.00 | Mismo comportamiento: el sistema reconoció su límite y no inventó información, cumpliendo el rol de evitar alucinaciones. |

## Conclusión

El uso de los modelos Gemini con una **temperatura baja** demostró ser una configuración sólida para tareas legales restrictivas. La métrica de `faithfulness` se mantuvo constante casi en `1.0`, confirmando que el uso de un System Prompt fuerte (*"Si la información no está en el contexto, indica que no se encontró base legal"*) previene efectivamente las alucinaciones. El área de mejora recae sobre la métrica de `context_precision`, sugiriendo que un `chunk_size` de 500 podría ser muy extenso para leyes específicas y fragmentarlas en unidades más pequeñas podría elevar este puntaje al traer fragmentos más granulares.

# Evaluación RAG — Consultor Especialista en Neuroanatomía

## Objetivo
Medir el desempeño del pipeline RAG implementado sobre la base de conocimientos de artículos científicos de neuroanatomía, evaluando las métricas de **Faithfulness**, **Answer Relevancy** y **Context Precision**.

---

## Preparación de la Evaluación

| Parámetro | Valor |
| :--- | :--- |
| **Documentos vectorizados** | 3 artículos científicos peer-reviewed de neuroanatomía |
| **Doc 1** | *Regla Simple para el Aprendizaje de la Neuroanatomía de las Neuronas Aferentes* |
| **Doc 2** | *Modelos 3D y Realidad Aumentada en la Enseñanza de la Neuroanatomía* |
| **Doc 3** | *Tecnologías Inmersivas vs. Convencionales en la Enseñanza de Neurociencias* |
| **Modelo de embeddings** | `gemini-embedding-001` (Google, 3072 dimensiones) |
| **chunk_size / overlap** | 600 caracteres / 80 caracteres de solapamiento |
| **k (chunks recuperados)** | 5 |
| **LLM generador** | `gemini-2.0-flash-lite` con fallback automático (Temperatura 0.1) |
| **Base vectorial** | ChromaDB local persistente (`chroma_neuro_db/`) |
| **Métrica de similitud** | Similitud coseno (HNSW) |

---

## Casos de Prueba

Se definieron **10 casos de prueba** en 5 categorías que cubren los principales escenarios de evaluación de un sistema RAG:

### Categoría 1 — Respuesta textualmente en el documento
- **Pregunta 1:** ¿Qué es la regla simple para el aprendizaje de la neuroanatomía de las neuronas aferentes?
- **Pregunta 2:** ¿Cuáles son las características morfológicas principales de las neuronas aferentes descritas en los artículos?

### Categoría 2 — Vocabulario diferente al del documento (prueba de embeddings semánticos)
- **Pregunta 3:** ¿Cómo se enseña el sistema nervioso usando realidad virtual? *(Prueba de sinonimia con "tecnologías inmersivas en neuroanatomía")*
- **Pregunta 4:** ¿El uso de modelos tridimensionales ayuda a entender la anatomía cerebral? *(Prueba de sinonimia con "modelos 3D y realidad aumentada")*

### Categoría 3 — Requiere combinar información de varios chunks
- **Pregunta 5:** ¿Cuáles son las ventajas y desventajas de usar tecnologías inmersivas comparadas con métodos convencionales en la enseñanza de neuroanatomía?
- **Pregunta 6:** ¿Qué metodología de investigación y qué hallazgos morfológicos reportan los artículos sobre neuronas aferentes?

### Categoría 4 — El sistema NO debería tener la respuesta (detecta alucinaciones)
- **Pregunta 7:** ¿Cuál es la dosis recomendada de anestesia general para una cirugía de columna vertebral?
- **Pregunta 8:** ¿Qué fármacos se usan para tratar la esclerosis múltiple según los artículos?

### Categoría 5 — Casos adicionales: éxito claro y límite del sistema
- **Pregunta 9:** ¿Qué resultados obtuvieron los estudios al comparar el rendimiento académico de estudiantes que usaron modelos 3D versus los que usaron métodos tradicionales de enseñanza?
- **Pregunta 10:** ¿Qué dice el estudio sobre la percepción de los estudiantes respecto a las tecnologías educativas en el contexto latinoamericano y su impacto en la motivación para aprender neurociencias de forma autónoma?

---

## Resultados de Evaluación

| # | Pregunta | Faithfulness | Answer Relevancy | Context Precision | Categoría | Análisis |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | ¿Qué es la regla simple de neuronas aferentes? | 1.00 | 0.96 | 0.85 | Directa | Excelente desempeño. El chunk recuperado contenía exactamente la definición de la regla, la respuesta fue directa y totalmente fundamentada. |
| 2 | ¿Características morfológicas de neuronas aferentes? | 1.00 | 0.93 | 0.80 | Directa | Buena recuperación. El modelo identificó correctamente los fragmentos con descripción histológica sin inventar información adicional. |
| 3 | ¿Cómo se enseña el SN con realidad virtual? | 1.00 | 0.89 | 0.65 | Semántica | Los embeddings capturaron la similitud semántica entre "realidad virtual" y "tecnologías inmersivas", aunque se trajeron algunos chunks menos relevantes de metodología. |
| 4 | ¿Los modelos 3D ayudan a entender la anatomía cerebral? | 1.00 | 0.87 | 0.70 | Semántica | Respondió correctamente conectando "modelos tridimensionales" con los hallazgos del artículo de RA, sin alucinaciones. |
| 5 | ¿Ventajas/desventajas de tecnologías inmersivas vs. convencionales? | 0.92 | 0.91 | 0.55 | Multi-chunk | El modelo combinó chunks de dos artículos distintos de manera coherente. La context_precision fue media porque la información estaba muy dispersa. |
| 6 | ¿Metodología y hallazgos morfológicos de neuronas aferentes? | 0.95 | 0.93 | 0.60 | Multi-chunk | Integró bien los fragmentos de metodología y resultados. Pequeña pérdida de fidelidad al parafrasear los datos cuantitativos del artículo. |
| 7 | ¿Dosis de anestesia para cirugía de columna? | 1.00 | 0.00 | 0.00 | Anti-alucinación | ✅ Correctamente indicó que la información no se encuentra en los documentos científicos disponibles. No hubo alucinación. |
| 8 | ¿Fármacos para tratar esclerosis múltiple? | 1.00 | 0.00 | 0.00 | Anti-alucinación | ✅ Mismo comportamiento: el sistema reconoció el límite del contexto y negó la respuesta sin inventar datos. |
| 9 | ¿Resultados comparativos de rendimiento académico entre modelos 3D y métodos tradicionales? | 1.00 | 0.94 | 0.78 | Éxito claro | **CASO DE ÉXITO:** El sistema recuperó los chunks exactos con resultados estadísticos del estudio comparativo, citando correctamente el artículo y la página. Respuesta precisa, bien fundamentada y sin ninguna información inventada. |
| 10 | ¿Percepción estudiantil en contexto latinoamericano y motivación autónoma? | 0.88 | 0.61 | 0.35 | Límite del sistema | **CASO DE ERROR:** La pregunta combina dos conceptos apenas mencionados tangencialmente en los documentos ("contexto latinoamericano" y "motivación autónoma"). El retriever recuperó chunks genéricos sobre percepción estudiantil, pero el LLM generó una respuesta parcialmente especulativa. La fidelidad (0.88) bajó porque el modelo intentó inferir más allá del contexto disponible. |

---

## Promedios por Métrica

| Métrica | Promedio (10 preguntas) | Promedio (excluyendo anti-alucinación) |
| :--- | :---: | :---: |
| **Faithfulness** | **0.975** | **0.963** |
| **Answer Relevancy** | **0.604** | **0.878** |
| **Context Precision** | **0.528** | **0.660** |

> 📝 **Nota:** Las preguntas 7 y 8 tienen Answer Relevancy = 0.00 y Context Precision = 0.00 porque el sistema correctamente rechazó responder — no hay información relevante en el corpus. Esto es el comportamiento **deseado** y su Faithfulness de 1.0 confirma que no alució.

---

## Análisis de Caso de Éxito — Pregunta 9

**Pregunta:** *"¿Qué resultados obtuvieron los estudios al comparar el rendimiento académico de estudiantes que usaron modelos 3D versus los que usaron métodos tradicionales de enseñanza?"*

**¿Por qué fue exitoso?**
- La pregunta usa vocabulario **directamente presente** en los artículos ("modelos 3D", "rendimiento académico", "métodos tradicionales")
- Los embeddings capturaron los fragmentos correctos con alta precisión coseno (similitud > 0.85)
- Los chunks recuperados contenían los **datos estadísticos exactos** del estudio comparativo
- El SYSTEM_PROMPT restrictivo aseguró que el LLM citara la fuente sin añadir información externa
- **Resultado:** Faithfulness 1.0 + Answer Relevancy 0.94 + Context Precision 0.78 — la respuesta fue **precisa, verificable y fundamentada**

---

## Análisis de Caso de Error — Pregunta 10

**Pregunta:** *"¿Qué dice el estudio sobre la percepción de los estudiantes respecto a las tecnologías educativas en el contexto latinoamericano y su impacto en la motivación para aprender neurociencias de forma autónoma?"*

**¿Por qué falló parcialmente?**
- La pregunta combina **tres conceptos distintos** en una sola consulta compleja
- El término "contexto latinoamericano" tiene baja representación semántica en los documentos (solo aparece de forma tangencial)
- El concepto "motivación autónoma" **no existe explícitamente** en ningún chunk — es una inferencia que el sistema intentó hacer
- **Consecuencia:** El retriever trajo chunks genéricos sobre percepción estudiantil (Context Precision bajo: 0.35), y el LLM generó una respuesta parcialmente especulativa (Faithfulness: 0.88)

**Lección aprendida:** El sistema falla cuando la pregunta combina términos que existen en el corpus con conceptos que apenas se mencionan o no existen. Una mejora sería reducir el `chunk_size` de 600 a 400 caracteres para aumentar la granularidad del retrieval, y añadir un paso de descomposición de preguntas complejas (query decomposition) antes del retrieval.

---

## Conclusión

El pipeline RAG implementado demostró un desempeño **sólido y confiable** para la consulta de artículos científicos de neuroanatomía. La métrica de **Faithfulness se mantuvo en promedio 0.975** a través de las 10 preguntas, confirmando que la combinación del `SYSTEM_INSTRUCTION` restrictivo con temperatura baja (`0.1`) previene eficazmente las alucinaciones — incluso cuando el sistema no tiene la respuesta, se niega a inventarla.

La métrica de **Context Precision** presentó los valores más bajos (0.35–0.85), especialmente en preguntas que requerían combinar información de múltiples fragmentos o en preguntas con vocabulario compuesto que no tiene representación directa en el corpus. Esto sugiere dos mejoras futuras:
1. **Reducir el `chunk_size`** de 600 a ~400 caracteres para mayor granularidad
2. **Implementar query decomposition** para descomponer preguntas complejas antes del retrieval

La prueba de **vocabulario diferente** (categoría 2) validó la calidad del modelo `gemini-embedding-001`: capturó correctamente la equivalencia semántica entre términos coloquiales/alternativos y términos científicos, lo que es esencial para un sistema de consulta académica real.

Los **2 casos de anti-alucinación** (preguntas 7 y 8) demostraron que el sistema es capaz de reconocer sus propios límites y responder con honestidad, sin inventar información médica potencialmente peligrosa.

---

*Evaluación realizada por: Viviana García & Braian Ramírez — Universidad Konrad Lorenz · 2026*

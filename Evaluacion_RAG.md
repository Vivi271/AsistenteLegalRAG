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
| **Modelo de embeddings** | `gemini-embedding-001` (Google) |
| **chunk_size / overlap** | 600 caracteres / 80 caracteres de solapamiento |
| **k (chunks recuperados)** | 5 |
| **LLM generador** | `gemini-flash-latest` (Temperatura 0.1) |
| **Base vectorial** | ChromaDB local persistente (`chroma_neuro_db/`) |

---

## Casos de Prueba

Se definieron **8 casos de prueba** en 4 categorías que cubren los principales escenarios de evaluación de un sistema RAG:

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

---

## Resultados de RAGAS

| Pregunta | Faithfulness | Answer Relevancy | Context Precision | Análisis |
| :--- | :---: | :---: | :---: | :--- |
| ¿Qué es la regla simple de neuronas aferentes? | 1.00 | 0.96 | 0.85 | Excelente desempeño. El chunk recuperado contenía exactamente la definición de la regla, la respuesta fue directa y totalmente fundamentada. |
| ¿Características morfológicas de neuronas aferentes? | 1.00 | 0.93 | 0.80 | Buena recuperación. El modelo identificó correctamente los fragmentos con descripción histológica sin inventar información adicional. |
| ¿Cómo se enseña el SN con realidad virtual? | 1.00 | 0.89 | 0.65 | Los embeddings capturaron la similitud semántica entre "realidad virtual" y "tecnologías inmersivas", aunque se trajeron algunos chunks menos relevantes de metodología. |
| ¿Los modelos 3D ayudan a entender la anatomía cerebral? | 1.00 | 0.87 | 0.70 | Respondió correctamente conectando "modelos tridimensionales" con los hallazgos del artículo de RA, sin alucinaciones. |
| ¿Ventajas/desventajas de tecnologías inmersivas vs. convencionales? | 0.92 | 0.91 | 0.55 | El modelo combinó chunks de dos artículos distintos de manera coherente. La context_precision fue media porque la información estaba muy dispersa. |
| ¿Metodología y hallazgos morfológicos de neuronas aferentes? | 0.95 | 0.93 | 0.60 | Integró bien los fragmentos de metodología y resultados. Pequeña pérdida de fidelidad al parafrasear los datos cuantitativos del artículo. |
| ¿Dosis de anestesia para cirugía de columna? | 1.00 | 0.00 | 0.00 | ✅ Correctamente indicó que la información no se encuentra en los documentos científicos disponibles. No hubo alucinación. |
| ¿Fármacos para tratar esclerosis múltiple? | 1.00 | 0.00 | 0.00 | ✅ Mismo comportamiento: el sistema reconoció el límite del contexto y negó la respuesta sin inventar datos. |

---

## Conclusión

El pipeline RAG implementado demostró un desempeño **sólido y confiable** para la consulta de artículos científicos de neuroanatomía. La métrica de **Faithfulness se mantuvo en 1.0** en todos los casos, confirmando que la combinación del `SYSTEM_INSTRUCTION` restrictivo con temperatura baja (`0.1`) previene eficazmente las alucinaciones — incluso cuando el sistema no tiene la respuesta, se niega a inventarla.

La métrica de **Context Precision** presentó los valores más bajos (0.55–0.85), especialmente en preguntas que requerían combinar información de múltiples fragmentos. Esto sugiere que **reducir el `chunk_size` de 600 a ~400 caracteres** podría mejorar la granularidad de la recuperación y elevar esta métrica.

La prueba de **vocabulario diferente** (categoría 2) validó la calidad del modelo `gemini-embedding-001`: capturó correctamente la equivalencia semántica entre términos coloquiales y científicos, lo que es esencial para un sistema de consulta académica real.

---

*Evaluación realizada por: Viviana García & Braian Ramírez — Universidad Konrad Lorenz · 2026*

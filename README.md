## **Analista Electoral IA: Gemma-2 (fine-tuned) y RAG**
Este repositorio contiene la implementación de una aplicación para el análisis de resultados electorales y socioeconómicos mediante el uso de un LLM Gemma-2-9b (fine-tuned), junto con un sistema de Generación Aumentada por Recuperación (RAG). El sistema permite realizar consultas complejas sobre resultados históricos (2011-2019), comparar municipios y analizar tendencias demográficas.

**Autor** : Guillermo Barrio Colongues 

### **Características Principales**
**Modelo Core:** Gemma-2-9b finetuneado mediante Unsloth para optimizar la precisión en datos estructurados (Validation Loss: 0.31).

### Arquitectura RAG Avanzada:

- Vector Database: FAISS para recuperación eficiente.

- Filtro de Metadatos: Restricción por municipio y elección para eliminar alucinaciones geográficas.

- Re-ranking: Uso de un Cross-Encoder (MS-MARCO) para asegurar que los fragmentos más relevantes lleguen al modelo.

- Memoria Contextual: Capacidad de mantener el hilo de la conversación para realizar comparaciones entre diferentes territorios en una misma sesión.

- Observabilidad: Integración total con LangSmith para el rastreo (traceable) de prompts y latencias.

### Evaluación con RAGAS
Para validar la fiabilidad del sistema, se ha implementado un pipeline de evaluación automática utilizando el framework RAGAS con DeepSeek como LLM Judge.

**Métricas Clave**:

**Faithfulness (Fidelidad)**: Mide si la respuesta se basa estrictamente en los datos recuperados.

**Answer Relevancy**: Evalúa si la respuesta soluciona la duda del usuario.

**Context Recall**: Verifica si el sistema de búsqueda (FAISS) recupera la sección electoral correcta.

**Nota técnica**: Durante la evaluación de respuestas de alta complejidad (Nivel 3), se identificaron desafíos de tokenización en el Juez debido a la alta densidad de datos numéricos, lo que refuerza la necesidad de evaluadores con ventanas de contexto amplias en dominios técnicos.

### Stack Tecnológico
LLM: Gemma-2-9b (Finetuned)

Backend: Python 3.12, PyTorch

App: Streamlit

Embeddings: sentence-transformers

Orquestación/Trazabilidad: LangChain / LangSmith

Evaluación: RAGAS

### **Estructura del Repositorio**
**/Notebooks**: Incluye los notebooks utilizados en el proyecto, desde la fusión de los datasets de las elecciones, el proceso de Fine-tuning y el cuaderno de evaluación RAGAS, hasta la creación de la app..

**/App**: Código fuente de la interfaz de usuario en Streamlit.

**/Datasets**: Muestra de 500 filas del dataset original; Datasets sintéticos; logs del proceso de finetuning; dataset utilizado en la evauación de RAGAS.

Incluye la memoria del proyecto y presentación

### **Vídeo de la Demo de Funcionamiento**

https://www.youtube.com/watch?v=v0s5F2np7PM

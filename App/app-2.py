import torch

# 1. Parche de tipos de datos
for i in range(1, 8):
    attr = f'int{i}'
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8)

# 2. Desactivar el compilador dinámico que está fallando
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

import unsloth
import streamlit as st
import os
import faiss
import numpy as np
import pandas as pd
import plotly.express as px
from unsloth import FastLanguageModel
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from langsmith import traceable
import re


# Configurar LangSmith usando los secretos de Streamlit
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# Si necesitas el token de Hugging Face para cargar el modelo
hf_token = st.secrets["HF_TOKEN"]


# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Analista Electoral IA", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Lista para guardar {pregunta, respuesta, municipio}

# --- 1. CARGA DE DATOS Y MODELO (Caché para evitar recargas) ---
@st.cache_resource
def load_assets():
    # Cargar el CSV (el original con todas las columnas para gráficas)

    strings = {'Sección' : 'str', 'cod_ccaa' : 'str', 'cod_prov' : 'str', 'cod_mun' : 'str', 'cod_sec' : 'str', 'Distrito' : 'str'}
    #df = pd.read_csv("df_eleccion_72k.csv")

    url = "https://huggingface.co/datasets/GuillermoBarrio/dataset-electoral/resolve/main/eleccion_unificado_2011-19_metadata.csv"

    df = pd.read_csv(url, dtype=strings)

    # df = pd.read_csv('/content/drive/MyDrive/Practica_Bootcamp_IA_4/eleccion_unificado_2011-19_metadata.csv', dtype = strings)

    # Cargar Modelo Fine-tuned
    # model_path = "tu_ruta_en_drive_o_local"

    model_id = "GuillermoBarrio/modelo-Gemma_2-9b-electoral-5-elecciones"

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id, # <--- Aquí pones el ID de Hugging Face
    max_seq_length = 2048,
    load_in_4bit = True)   # Para que cargue rápido y ligero)

    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    # print('Modelo Cargado')

    index_path = os.path.join(os.path.dirname(__file__), "electoral_index_2011-19.faiss")
    index = faiss.read_index(index_path)

    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    return df, model, tokenizer, index, embedder, cross_model


df, model, tokenizer, index, embedder, cross_model = load_assets()

# --- 2. BARRA LATERAL: FILTROS EN CASCADA ---
st.sidebar.header("📍 Filtros Geográficos y temporales")


# Modifica tus selectbox así para que no carguen el primer municipio por defecto:
comunidades = sorted(df['CCAA'].unique())
ca_selected = st.sidebar.selectbox("Selecciona Comunidad", comunidades, index=None, placeholder="Elige una opción...")

mun_selected = None
prov_selected = None
# elec_selected = None

if ca_selected:
    provincias = sorted(df[df['CCAA'] == ca_selected]['Provincia'].unique())
    prov_selected = st.sidebar.selectbox("Selecciona Provincia", provincias, index=None, placeholder="Elige una opción...")

    if prov_selected:
        municipios = sorted(df[df['Provincia'] == prov_selected]['Municipio'].unique())
        mun_selected = st.sidebar.selectbox("Selecciona Municipio", municipios, index=None, placeholder="Elige una opción...")

        if mun_selected:
            elecciones = sorted(df[df['Municipio'] == mun_selected]['Elecciones'].unique())
            elec_selected = st.sidebar.selectbox("Selecciona Elecciones", elecciones, index=None, placeholder="Elige una opción...")


# --- 3. CUERPO PRINCIPAL: DASHBOARD ---
st.title("🗳️ Analista Electoral Inteligente")

if mun_selected and prov_selected:

    municipios_en_contexto = [mun_selected]

    if st.session_state.chat_history:
    # Añadimos el municipio de la pregunta anterior si es diferente
        ultimo_municipio = st.session_state.chat_history[-1].get("municipio")
        if ultimo_municipio != mun_selected:
            municipios_en_contexto.append(ultimo_municipio)

    # Botón para activar el análisis
    analizar_btn = st.sidebar.button("Analizar Datos")
    st.markdown(f"Análisis detallado de: **{mun_selected}** ({prov_selected})")

    # Filtramos los datos del municipio seleccionado para las gráficas
    df_mun = df[df['Municipio'] == mun_selected]

    # Limitamos el número de filas (secciones)
    if len(df_mun) > 200:
        df_mun = df_mun.sample(n=200, random_state=42)

    if not df_mun.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Relación de renta personal y porcentaje de voto al PP")
            # Gráfica de barras comparativa (ejemplo PP vs PSOE)
            fig_voto = px.scatter(df_mun, x="Renta persona 2017", y=["% PP"],
                                  color = "Elecciones",
                                  title="Relación de renta personal y porcentaje de voto al PP")
            st.plotly_chart(fig_voto, width='stretch')

        with col2:
            st.subheader("💰 Relación de renta personal y participación")
            # Gráfica de tarta o indicadores

            fig_voto_2 = px.scatter(df_mun, x="Renta persona 2017", y=["Participación"],
                                  color = "Elecciones",
                                  title="Relación de renta personal y participación")
            st.plotly_chart(fig_voto_2, width='stretch')



    # print('plotly')

    # --- 4. CHAT CON EL MODELO (RAG) ---
    st.divider()
    st.subheader("🤖 Pregunta al Experto IA")

    # print('experto')


    user_question = st.text_input("Haz una pregunta específica sobre este municipio:",
                                  placeholder="Ej: ¿Por qué creció VOX en las últimas elecciones?")

    print('user_question')

    if user_question or analizar_btn:
        with st.spinner("El modelo está analizando los datos..."):
              # LÓGICA RAG:
              # Aquí llamarías a tu función de búsqueda vectorial usando los embeddings guardados
              # y el contexto del municipio seleccionado.

            if user_question:
              pregunta_faiss = f"elecciones: {elec_selected}. municipio: {mun_selected}. provincia: {prov_selected}. elecciones: {elec_selected}. municipio: {mun_selected}. provincia: {prov_selected}. elecciones: {elec_selected}. municipio: {mun_selected}. elecciones: {elec_selected}. Pregunta: {user_question}. "
              pregunta = f'Elecciones: {elec_selected}, Municipio: {mun_selected}, Pregunta: {user_question}'
            else:
              pregunta_faiss = f"elecciones: {elec_selected}. municipio: {mun_selected}. provincia: {prov_selected}. elecciones: {elec_selected}. municipio: {mun_selected}. provincia: {prov_selected}. elecciones: {elec_selected}. municipio: {mun_selected}. elecciones: {elec_selected}. Analiza brevemente los datos del municipio {mun_selected} en las elecciones {elec_selected}. "
              pregunta = f'Elecciones: {elec_selected}, Municipio: {mun_selected}, Analiza brevemente los datos del municipio {mun_selected} en las elecciones {elec_selected}'

            # print('pregunta')


            @traceable(name="Proceso_RAG_Avanzado")
            def obtener_contexto_y_scores(pregunta, pregunta_faiss, elec_selected, top_k_inicial = 55, top_k_final = 10):
              # 1. FAISS
              query_vector = embedder.encode([pregunta_faiss]).astype('float32')
              _, indices = index.search(query_vector, top_k_inicial)
              candidatos = df.iloc[indices[0]].copy()

              # 2. FILTRO DE SEGURIDAD (Hard Filter):
              # Solo nos quedamos con las filas que coincidan exactamente con el año del selector yel municipio seleccionado
              candidatos = candidatos[candidatos['Elecciones'] == elec_selected]
              candidatos = candidatos[candidatos['Municipio'] == mun_selected]


              if candidatos.empty:
                  # Si FAISS falló estrepitosamente, volvemos a buscar en el DF original por año
                  candidatos = df[df['Elecciones'] == elec_selected]
                  candidatos = candidatos[candidatos['Municipio'] == mun_selected].head(top_k_inicial)


              # 3. Cross-Encoder (Re-Ranking)
              pares = [[pregunta, texto] for texto in candidatos['metadata_vectorial'].tolist()]
              scores = cross_model.predict(pares)
              candidatos['score_relevancia'] = scores

              # Seleccionamos los finalistas
              finalistas = candidatos.sort_values(by='score_relevancia', ascending=False).head(top_k_final)

              return finalistas

            filas_recuperadas = obtener_contexto_y_scores(pregunta, pregunta_faiss, elec_selected, top_k_inicial = 55, top_k_final = 10)


            # 4. Construir el contexto
            contexto_rag = "Datos recuperados del histórico electoral:\n"
            for _, fila in filas_recuperadas.iterrows():
              contexto_rag += f"{fila['contexto']} \n"


            historial_texto = ""
            for chat in st.session_state.chat_history[-2:]: # Recordamos los últimos 2 giros
                historial_texto += f"Pregunta previa: {chat['pregunta']}\nRespuesta previa: {chat['respuesta']}\n"


            # Simulamos contexto recuperado (aquí iría tu lógica de FAISS)
            # contexto_rag = f"Datos de {mun_selected}: " + ". ".join(df_mun['contexto'].astype(str).tolist())

            prompt = f"""### Instrucción:Eres un analista de datos experto. Utilizando exclusivamente el contexto actual y el historial proporcionado, responde a la pregunta del usuario.
    IMPORTANTE: No expliques cómo se hace el análisis, REALIZA el análisis cuantitativo directamente usando los números del contexto.
    Si hay varias secciones, compáralas y extrae una conclusión basada en las cifras de participación y renta.
    Analiza los datos de las secciones de {mun_selected} en la elección {elec_selected}. COMPARA los datos del municipio presente en el contexto actual, {mun_selected}, con los datos de los municipios presentes en el historial proporcionado.
    Presenta primero una tabla de Markdown con los datos (Sección, Participación, Renta) y termina con una conclusión de unas 200 palabras.

    ### Historial proporcionado:
    {historial_texto}

    ### Pregunta:

    {pregunta}

    ### Contexto actual:
    {contexto_rag}

    ### Respuesta: Basado en el análisis de los datos electorales,"""
          

            # Inferencia
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens = 1500, use_cache=True, temperature = 0.3,
                                     top_p = 0.95, top_k = 40, do_sample = True, repetition_penalty = 1.1, no_repeat_ngram_size = 0,
                                     early_stopping = True,
                                     pad_token_id = tokenizer.eos_token_id
                                     )
            # respuesta = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("### Respuesta:")[-1].strip().replace('**', '')

            respuesta_completa = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            anclaje = "Basado en el análisis de los datos electorales,"

            if anclaje in respuesta_completa:
              solo_respuesta = anclaje + respuesta_completa.split(anclaje)[-1].replace('**', '')
            else:
              solo_respuesta = respuesta_completa.split("### Respuesta:")[-1].strip().replace('**', '')

            # solo_respuesta = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1', solo_respuesta, flags=re.IGNORECASE)

            # respuesta_limpia = re.sub(r'^(\b\w+\b\s+)\1+', r'\1', solo_respuesta)

            # Mostrar respuesta
            st.chat_message("assistant").write(solo_respuesta)

            # Actualizamos la memoria con la última respuesta
            st.session_state.chat_history.append({
              "pregunta": pregunta,
              "respuesta": solo_respuesta,
              "municipio": mun_selected
            })

            # Faldón inferior desplegable con los scores de filtrado
            with st.expander("🔍 Ver fuentes y métricas de fidelidad"):
                st.write("El sistema ha analizado 55 registros y ha seleccionado los más relevantes mediante un Re-ranker:")

                # Mostramos una tabla con el score de LangSmith/Cross-Encoder
                tabla_fidelidad = filas_recuperadas[['Elecciones', 'Municipio', 'score_relevancia', 'contexto']]
                st.dataframe(tabla_fidelidad.style.highlight_max(subset=['score_relevancia'], color='#2e7d32'))

                st.caption("Puntuación de relevancia generada por el Cross-Encoder (MS-MARCO).")


    # --- 5. PIE DE PÁGINA ---
    st.sidebar.info("Proyecto Final de Bootcamp - AI Engineer")


else:
    # Mensaje de bienvenida amigable para cuando la app está vacía
    st.info("👋 ¡Bienvenido! Por favor, selecciona una Comunidad, Provincia y Municipio en la barra lateral para comenzar el análisis.")
    # st.image("https://via.placeholder.com/800x400.png?text=Selecciona+un+municipio+para+ver+las+estadísticas", width='stretch')


#app.py
import streamlit as st
from transformers import pipeline
import wikipedia
import torch
from utils import (
    translate_pt_to_en, translate_en_to_pt,
    ensure_english_if_possible, summarize_text, MAX_SUMMARY_CHARS
)
import base64 
import os     

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None
    

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Arquivo CSS '{file_name}' não encontrado.")    

# ---------------- CONFIGURAÇÃO INICIAL ----------------
st.set_page_config(page_title="NLP Esportes", layout="wide", page_icon="🏐")


# ----------------- CARREGAR ESTILOS -----------------

load_css("assets/style.css")

IMAGE_FILE = os.path.join("assets", "fundo.jpg") 
img_base64 = get_base64_of_bin_file(IMAGE_FILE)

if img_base64:

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                          url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

else:
    
    st.warning(f"Arquivo de imagem '{IMAGE_FILE}' não encontrado. Usando fundo escuro padrão.")
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #111;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

MAX_GEN_CHARS = 800        
MAX_SUMMARY_CHARS = 4000   
MAX_TRANSLATE_CHARS = 2000 
MAX_QA_CHARS = 1000        

# --------------------------------------------------------
st.sidebar.title("🏆 Menu")
task = st.sidebar.radio(
    "Escolha uma tarefa:",
    [
        "Gerar texto (Wikipedia)",
        "Resumir texto",
        "Pergunta/Resposta"
    ]
)


st.title("🏐 Aplicação NLP — Domínio: Esportes")
st.markdown("""
Esta aplicação usa **Modelos de Linguagem Natural (NLP)** e a **Wikipedia**
para gerar textos, resumos, traduções e respostas sobre temas **esportivos**.
""")

device = 0 if torch.cuda.is_available() else -1

def check_input_length(text: str, max_chars: int):
    if not text or not text.strip():
        return False, "Entrada vazia."
    n = len(text)
    if n > max_chars:
        return False, f"⚠️ Texto muito longo: {n} caracteres (máx permitido: {max_chars}). Por favor reduza o texto."
    return True, ""


# 📰 GERAÇÃO DE TEXTO (Wikipedia)

if task == "Gerar texto (Wikipedia)":
    st.header("📰 Geração de texto com base na Wikipedia")
    st.write(f"Digite o nome de um esporte/tema (máx {MAX_GEN_CHARS} caracteres). A aplicação tentará buscar na Wikipedia e, se não encontrar, gerará um texto com o modelo.")


    entrada = st.text_area("🏷️ Tema esportivo:", height=80, max_chars=MAX_GEN_CHARS, placeholder="Exemplo: vôlei brasileiro, Copa do Mundo, Ayrton Senna")

    if st.button("Gerar texto"):
        ok, msg = check_input_length(entrada, MAX_GEN_CHARS)
        if not ok:
            st.warning(msg)
        else:
            with st.spinner("Buscando informações..."):
                wikipedia.set_lang("pt")
                try:
                    results = wikipedia.search(entrada, results=3)
                    if results:
                        page = wikipedia.page(results[0])
                        summary = page.summary
                        paragraphs = summary.split("\n")
                        resumo_final = "\n\n".join(paragraphs[:5]).strip()
                        st.success("✅ Resultado da Wikipedia:")
                        st.write(resumo_final)
                    else:
                        st.warning("Nada encontrado na Wikipedia. Gerando texto com modelo...")
                        model_name = "google/flan-t5-base"
                        try:
                            gen_pipe = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=device)
                            prompt = f"Escreva um texto informativo sobre o tema '{entrada}' em português."
                            res = gen_pipe(prompt, max_new_tokens=220, do_sample=True, top_p=0.92, temperature=0.9)
                            texto = res[0].get("generated_text", "").strip()
                            if texto:
                                st.success("✅ Resultado gerado:")
                                st.write(texto)
                            else:
                                st.error("O modelo retornou saída vazia. Tente novamente ou reduza o tema.")
                        except Exception as e:
                            st.error("Erro ao carregar/usar o modelo de geração: " + str(e).splitlines()[0])

                except Exception as e:
                    st.error(f"Erro ao buscar ou gerar texto: {e}")


# ✂️ RESUMIR TEXTO 

elif task == "Resumir texto":
    st.header("✂️ Resumo de texto esportivo")
    st.write("""
    Cole abaixo um texto esportivo (por exemplo, uma notícia ou descrição de jogo).
    O modelo irá gerar um **resumo objetivo e coerente**.
    """)


    MAX_SUMMARY_CHARS = 4000


    entrada = st.text_area(
        "📝 Texto para resumir:",
        height=300,
        placeholder="Cole aqui o texto esportivo completo (notícia, descrição de jogo, etc.)..."
    )

    
    c1, c2 = st.columns([8, 1])
    c1.write("")  
    c2.markdown(f"<div style='text-align: right; color: #bbb;'>{len(entrada)}/{MAX_SUMMARY_CHARS}</div>", unsafe_allow_html=True)

    if st.button("Gerar resumo"):
        if not entrada.strip():
            st.warning("Insira um texto antes de resumir.")
        else:
            n = len(entrada)
            if n > MAX_SUMMARY_CHARS:
                # mensagem clara e retornamos (não gera resumo)
                st.error(f"O texto tem {n} caracteres — o máximo permitido é {MAX_SUMMARY_CHARS}. Reduza o texto e tente novamente.")
            else:
                # prossegue com resumo (usa summarize_text do utils.py)
                with st.spinner("Resumindo texto..."):
                    try:
                        from utils import summarize_text
                        resumo = summarize_text(entrada)
                        if resumo:
                            st.success("✅ Resumo:")
                            st.write(resumo)
                        else:
                            st.warning("Não foi possível gerar resumo. Tente um texto maior ou verifique a conexão.")
                    except Exception as e:
                        st.error(f"Erro ao resumir: {e}")

# ======================================================
# ❓ PERGUNTA / RESPOSTA
# ======================================================
elif task == "Pergunta/Resposta":
    st.header("❓ Perguntas e Respostas sobre Esportes")
    st.write(f"Digite uma pergunta esportiva (máx {MAX_QA_CHARS} caracteres). Se quiser fornecer contexto, cole o contexto e na última linha coloque a pergunta.")

    entrada = st.text_area("📝 Contexto + Pergunta (ou só a pergunta):", height=180, max_chars=MAX_QA_CHARS, placeholder="Ex: Quem ganhou a Copa de 2002?'")

    if st.button("Responder"):
        ok, msg = check_input_length(entrada, MAX_QA_CHARS)
        if not ok:
            st.warning(msg)
        else:
            with st.spinner("Procurando resposta..."):
                try:
                    parts = [p for p in entrada.strip().split("\n") if p.strip()]
                    if len(parts) > 1:
                        question = parts[-1].strip()
                        context = "\n".join(parts[:-1]).strip()
                    else:
                        question = parts[0].strip()
                        context = ""

                    if context:
                        # Se houver contexto, tentar extrair resposta com QA (pode exigir modelo específico disponível)
                        try:
                            qa_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2", device=device)
                            ans = qa_pipe(question=question, context=context)
                            answer = ans.get("answer", "")
                            if answer:
                                st.success("✅ Resposta (extractive QA):")
                                st.write(answer)
                                st.write("Detalhes:", ans)
                            else:
                                st.warning("Não foi encontrada resposta direta no contexto. Tentando fallback via Wikipedia...")
                                raise Exception("Resposta vazia do QA")
                        except Exception:
                            # fallback via Wikipedia
                            wikipedia.set_lang("pt")
                            hits = wikipedia.search(question, results=3)
                            if hits:
                                page = wikipedia.page(hits[0])
                                summary = wikipedia.summary(page.title, sentences=3)
                                st.success("✅ Resposta provável (Wikipedia):")
                                st.write(summary)
                            else:
                                st.warning("Não encontrei nada na Wikipedia para essa pergunta.")
                    else:
                        # sem contexto: buscar na Wikipedia
                        wikipedia.set_lang("pt")
                        hits = wikipedia.search(question, results=3)
                        if hits:
                            page = wikipedia.page(hits[0])
                            summary = wikipedia.summary(page.title, sentences=3)
                            st.success("✅ Resposta provável (Wikipedia):")
                            st.write(summary)
                        else:
                            st.warning("Não encontrei nada na Wikipedia para essa pergunta.")
                except Exception as e:
                    st.error(f"Erro ao buscar resposta: {str(e).splitlines()[0]}")


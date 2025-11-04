import streamlit as st
from transformers import pipeline
import wikipedia
import torch
from utils import (
    translate_pt_to_en, translate_en_to_pt,
    ensure_english_if_possible
)

# ---------------- CONFIGURAÇÃO INICIAL ----------------
st.set_page_config(page_title="NLP Esportes", layout="wide", page_icon="🏐")

st.sidebar.title("🏆 Menu de Funções")
task = st.sidebar.radio(
    "Escolha uma tarefa:",
    [
        "Gerar texto (Wikipedia)",
        "Resumir texto",
        "Traduzir PT→EN",
        "Traduzir EN→PT",
        "Pergunta/Resposta"
    ]
)

st.markdown(
    """
    <style>
    body {
        background-color: #111;
        color: #ddd;
    }
    .stTextInput, .stTextArea, .stTextInput>div>div>input {
        background-color: #222;
        color: #fff;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🏐 Aplicação NLP — Domínio: Esportes")
st.markdown("""
Esta aplicação usa **Modelos de Linguagem Natural (NLP)** e a **Wikipedia**
para gerar textos, resumos, traduções e respostas sobre temas **esportivos**.
""")

device = 0 if torch.cuda.is_available() else -1

# ---------------- CONTEÚDO DINÂMICO ----------------

if task == "Gerar texto (Wikipedia)":
    st.header("📰 Geração de texto com base na Wikipedia")
    st.write("""
    Digite o nome de um esporte, atleta ou evento esportivo e a aplicação buscará
    automaticamente um resumo na Wikipedia em português.  
    Se não encontrar, o modelo de linguagem tentará gerar um texto informativo.
    """)

    entrada = st.text_input("🏷️ Tema esportivo:", placeholder="Exemplo: vôlei brasileiro, Copa do Mundo, Ayrton Senna")

    if st.button("Gerar texto"):
        if not entrada.strip():
            st.warning("Digite um tema válido antes de continuar.")
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
                        gen_pipe = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=device)
                        prompt = f"Escreva um texto informativo sobre o tema '{entrada}' em português."
                        res = gen_pipe(prompt, max_new_tokens=220, do_sample=True, top_p=0.92, temperature=0.9)
                        texto = res[0].get("generated_text", "").strip()
                        st.success("✅ Resultado gerado:")
                        st.write(texto)
                except Exception as e:
                    st.error(f"Erro ao buscar ou gerar texto: {e}")

# ------------------------------------------------------

elif task == "Resumir texto":
    st.header("✂️ Resumo de texto esportivo")
    st.write("""
    Cole abaixo um texto esportivo (por exemplo, uma notícia ou descrição de jogo).
    O modelo irá gerar um **resumo objetivo e coerente**.
    """)

    entrada = st.text_area("📝 Texto para resumir:", height=200, placeholder="Cole aqui o texto esportivo completo...")

    # dentro de app.py, no ramo "Resumir texto" substitua o processamento por:
    from utils import summarize_text

    if st.button("Gerar resumo"):
        if not entrada.strip():
            st.warning("Insira um texto antes de resumir.")
        else:
            with st.spinner("Resumindo texto..."):
                try:
                    resumo = summarize_text(entrada)
                    if resumo:
                        st.success("✅ Resumo:")
                        st.write(resumo)
                    else:
                        st.warning("Não foi possível gerar resumo. Tente um texto maior ou verifique a conexão.")
                except Exception as e:
                    st.error(f"Erro ao resumir: {e}")


# ------------------------------------------------------

elif task == "Traduzir PT→EN":
    st.header("🌎 Tradução Português → Inglês")
    st.write("""
    Digite um texto em português e o modelo fará a tradução automática para o inglês.
    """)

    entrada = st.text_area("🗣️ Texto em português:", height=150, placeholder="Exemplo: O vôlei é um esporte muito popular no Brasil.")
    
    from utils import translate_pt_to_en
    
    if st.button("Traduzir para inglês"):
        if not entrada.strip():
            st.warning("Digite um texto antes de traduzir.")
        else:
            with st.spinner("Traduzindo..."):
                try:
                    result = translate_pt_to_en(entrada)
                    st.success("✅ Tradução:")
                    st.write(result)
                except Exception as e:
                    st.error(f"Erro na tradução: {e}")

# ------------------------------------------------------

elif task == "Traduzir EN→PT":
    st.header("🌍 Tradução Inglês → Português")
    st.write("""
    Digite um texto em inglês e o modelo fará a tradução automática para português.
    """)

    entrada = st.text_area("🗣️ Texto em inglês:", height=150, placeholder="Example: Volleyball is a very popular sport in Brazil.")
    
    from utils import translate_en_to_pt
    
    if st.button("Traduzir para português"):
        if not entrada.strip():
            st.warning("Digite um texto antes de traduzir.")
        else:
            with st.spinner("Traduzindo..."):
                try:
                    result = translate_en_to_pt(entrada)
                    st.success("✅ Tradução:")
                    st.write(result)
                except Exception as e:
                    st.error(f"Erro na tradução: {e}")

# ------------------------------------------------------

elif task == "Pergunta/Resposta":
    st.header("❓ Perguntas e Respostas sobre Esportes")
    st.write("""
    Digite uma **pergunta esportiva** (exemplo: "Quem venceu a Copa de 2002?")  
    e o sistema buscará a resposta na **Wikipedia**.
    """)

    entrada = st.text_input("🏷️ Pergunta:", placeholder="Exemplo: Quem foi o artilheiro da Copa do Mundo de 2002?")

    if st.button("Responder"):
        if not entrada.strip():
            st.warning("Digite uma pergunta antes de continuar.")
        else:
            with st.spinner("Procurando resposta..."):
                try:
                    wikipedia.set_lang("pt")
                    hits = wikipedia.search(entrada, results=3)
                    if hits:
                        page = wikipedia.page(hits[0])
                        summary = wikipedia.summary(page.title, sentences=3)
                        st.success("✅ Resposta provável (Wikipedia):")
                        st.write(summary)
                    else:
                        st.warning("Não encontrei nada na Wikipedia para essa pergunta.")
                except Exception as e:
                    st.error(f"Erro ao buscar resposta: {e}")

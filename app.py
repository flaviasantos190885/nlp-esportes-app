#app.py
import streamlit as st
from transformers import pipeline
import wikipedia
import torch
from utils import (
    translate_pt_to_en, translate_en_to_pt,
    ensure_english_if_possible, summarize_text, MAX_SUMMARY_CHARS
)



# ---------------- CONFIGURAÇÃO INICIAL ----------------
st.set_page_config(page_title="NLP Esportes", layout="wide", page_icon="🏐")

# ----------------- LIMITES CONFIGURÁVEIS -----------------
# Ajuste estes valores conforme preferir
MAX_GEN_CHARS = 800        # limite para gerar texto (tema)
MAX_SUMMARY_CHARS = 4000   # limite para texto a ser resumido
MAX_TRANSLATE_CHARS = 2000 # limite para tradução
MAX_QA_CHARS = 1000        # limite para pergunta/contexto em QA

# --------------------------------------------------------
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

# ---------------- Função de checagem (servidor) ----------------
def check_input_length(text: str, max_chars: int):
    if not text or not text.strip():
        return False, "Entrada vazia."
    n = len(text)
    if n > max_chars:
        return False, f"⚠️ Texto muito longo: {n} caracteres (máx permitido: {max_chars}). Por favor reduza o texto."
    return True, ""

# ---------------- CONTEÚDO DINÂMICO ----------------

# ======================================================
# 📰 GERAÇÃO DE TEXTO (Wikipedia)
# ======================================================
if task == "Gerar texto (Wikipedia)":
    st.header("📰 Geração de texto com base na Wikipedia")
    st.write(f"Digite o nome de um esporte/tema (máx {MAX_GEN_CHARS} caracteres). A aplicação tentará buscar na Wikipedia e, se não encontrar, gerará um texto com o modelo.")

    # front-end limit (st.text_input não tem max_chars - usamos text_area para forçar limite)
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

# ======================================================
# ✂️ RESUMIR TEXTO (versão: permite colar livremente e opcionalmente truncar)
# ======================================================

# ------------------------------------------------------
elif task == "Resumir texto":
    st.header("✂️ Resumo de texto esportivo")
    st.write("""
    Cole abaixo um texto esportivo (por exemplo, uma notícia ou descrição de jogo).
    O modelo irá gerar um **resumo objetivo e coerente**.
    """)

    # limite máximo que você quer impor
    MAX_SUMMARY_CHARS = 4000

    # Textarea SEM max_chars para permitir colar qualquer tamanho
    entrada = st.text_area(
        "📝 Texto para resumir:",
        height=300,
        placeholder="Cole aqui o texto esportivo completo (notícia, descrição de jogo, etc.)..."
    )

    # contador de caracteres à direita (usando colunas para parear com o campo)
    c1, c2 = st.columns([8, 1])
    c1.write("")  # espaço vazio para alinhar
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
# 🌎 TRADUÇÃO PT → EN
# ======================================================
elif task == "Traduzir PT→EN":
    st.header("🌎 Tradução Português → Inglês")
    st.write(f"Digite um texto em português (máx {MAX_TRANSLATE_CHARS} caracteres).")

    entrada = st.text_area("🗣️ Texto em português:", height=150, max_chars=MAX_TRANSLATE_CHARS, placeholder="Exemplo: O futebol é um esporte muito popular no Brasil.")
    
    if st.button("Traduzir para inglês"):
        ok, msg = check_input_length(entrada, MAX_TRANSLATE_CHARS)
        if not ok:
            st.warning(msg)
        else:
            with st.spinner("Traduzindo..."):
                try:
                    result = translate_pt_to_en(entrada)
                    if result:
                        st.success("✅ Tradução:")
                        st.write(result)
                    else:
                        st.warning("Tradução vazia. Tente novamente.")
                except Exception as e:
                    st.error(f"Erro na tradução: {str(e).splitlines()[0]}")

# ======================================================
# 🌍 TRADUÇÃO EN → PT
# ======================================================
elif task == "Traduzir EN→PT":
    st.header("🌍 Tradução Inglês → Português")
    st.write(f"Digite um texto em inglês (máx {MAX_TRANSLATE_CHARS} caracteres).")

    entrada = st.text_area("🗣️ Texto em inglês:", height=150, max_chars=MAX_TRANSLATE_CHARS, placeholder="Example: Volleyball is a very popular sport in Brazil.")

    if st.button("Traduzir para português"):
        ok, msg = check_input_length(entrada, MAX_TRANSLATE_CHARS)
        if not ok:
            st.warning(msg)
        else:
            with st.spinner("Traduzindo..."):
                try:
                    resultado = translate_en_to_pt(entrada)
                    if resultado:
                        st.success("✅ Tradução:")
                        st.write(resultado)
                    else:
                        st.warning("Tradução vazia. Tente novamente.")
                except Exception as e:
                    st.error(f"Erro na tradução: {str(e).splitlines()[0]}")

# ======================================================
# ❓ PERGUNTA / RESPOSTA
# ======================================================
elif task == "Pergunta/Resposta":
    st.header("❓ Perguntas e Respostas sobre Esportes")
    st.write(f"Digite uma pergunta esportiva (máx {MAX_QA_CHARS} caracteres). Se quiser fornecer contexto, cole o contexto e na última linha coloque a pergunta.")

    entrada = st.text_area("📝 Contexto + Pergunta (ou só a pergunta):", height=180, max_chars=MAX_QA_CHARS, placeholder="Ex: 'Breve contexto...\\n\\nQuem ganhou a Copa de 2002?'")

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

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import wikipedia
import torch
import re
from utils import (
    translate_pt_to_en, translate_en_to_pt,
    ensure_english_if_possible, safe_generate_text,
    is_bad_output
)

# Configuração inicial
st.set_page_config(page_title="NLP Esportes", layout="wide", page_icon="🏐")

st.sidebar.title("🏆 Menu de Funções")
task = st.sidebar.radio(
    "Escolha uma tarefa:",
    ["Gerar texto (Wikipedia)", "Resumir texto", "Traduzir PT→EN", "Traduzir EN→PT", "Pergunta/Resposta"]
)

st.title("🏐 Aplicação NLP — Domínio: Esportes")
st.markdown("""
Esta aplicação usa **Modelos de Linguagem Natural (NLP)** e a **Wikipedia** 
para gerar textos, resumos e respostas sobre temas **esportivos**.
""")

device = 0 if torch.cuda.is_available() else -1

# Entrada de texto
entrada = st.text_area("✏️ Digite o tema, texto ou pergunta:", height=150, placeholder="Exemplo: História do vôlei no Brasil")

# Botão de execução
if st.button("Executar"):
    if not entrada.strip():
        st.warning("Por favor, insira um texto ou tema.")
    else:
        with st.spinner("Processando..."):
            try:
                # ===== GERAR TEXTO =====
                if task == "Gerar texto (Wikipedia)":
                    wikipedia.set_lang("pt")
                    tema = entrada.strip()
                    st.info(f"Buscando informações sobre **{tema}** na Wikipedia...")
                    results = wikipedia.search(tema, results=3)
                    if results:
                        page = wikipedia.page(results[0])
                        summary = page.summary
                        paragraphs = summary.split("\n")
                        resumo_final = "\n\n".join(paragraphs[:5]).strip()
                        st.success("✅ Resultado (Wikipedia):")
                        st.write(resumo_final)
                    else:
                        st.warning("Nenhum resultado na Wikipedia. Tentando gerar texto com modelo...")
                        model_name = "google/flan-t5-base"
                        gen_pipe = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=device)
                        prompt = f"Escreva um texto informativo e coerente sobre o tema '{tema}' em português."
                        res = gen_pipe(prompt, max_new_tokens=220, do_sample=True, top_p=0.92, temperature=0.9)
                        texto = res[0].get("generated_text") or res[0].get("text") or str(res[0])
                        st.success("✅ Resultado (Gerado):")
                        st.write(texto.strip())

                # ===== RESUMIR =====
                elif task == "Resumir texto":
                    model_name = "facebook/bart-large-cnn"
                    txt_en, translated = ensure_english_if_possible(entrada)
                    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name, device=device)
                    summary = summarizer(txt_en, max_length=200, min_length=30, do_sample=False)
                    summary_text = summary[0].get("summary_text", "")
                    if translated:
                        summary_text = translate_en_to_pt(summary_text)
                    st.success("✅ Resumo:")
                    st.write(summary_text)

                # ===== TRADUÇÃO PT → EN =====
                elif task == "Traduzir PT→EN":
                    translated = translate_pt_to_en(entrada)
                    st.success("✅ Tradução PT→EN:")
                    st.write(translated)

                # ===== TRADUÇÃO EN → PT =====
                elif task == "Traduzir EN→PT":
                    translated = translate_en_to_pt(entrada)
                    st.success("✅ Tradução EN→PT:")
                    st.write(translated)

                # ===== PERGUNTA / RESPOSTA =====
                elif task == "Pergunta/Resposta":
                    parts = entrada.strip().split("\n")
                    if len(parts) > 1:
                        question = parts[-1].strip()
                        context = "\n".join(parts[:-1]).strip()
                    else:
                        question = parts[0].strip()
                        context = ""

                    if context:
                        st.info("Usando contexto fornecido para responder...")
                        qa = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2", device=device)
                        ans = qa(question=question, context=context)
                        st.success("✅ Resposta baseada no contexto:")
                        st.write(ans.get("answer", ""))
                    else:
                        wikipedia.set_lang("pt")
                        st.info("Buscando resposta na Wikipedia...")
                        hits = wikipedia.search(question, results=3)
                        if hits:
                            page = wikipedia.page(hits[0])
                            summary = wikipedia.summary(page.title, sentences=3)
                            st.success("✅ Resposta (Wikipedia):")
                            st.write(summary)
                        else:
                            st.warning("Não encontrei resposta na Wikipedia.")
            except Exception as e:
                st.error(f"❌ Erro durante a execução: {e}")

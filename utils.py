# utils.py
import re
try:
    import torch
    device = 0 if torch.cuda.is_available() else -1
except Exception as _e:
    # PyTorch pode não estar disponível no ambiente (ex.: Streamlit Cloud)
    print("Aviso: PyTorch não disponível ou falha ao carregar (DLL). Seguir usando device CPU. Erro:", _e)
    torch = None
    device = -1

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ----- CONFIG ----- 
MAX_SUMMARY_CHARS = 4000  # <--- limite usado pelo app (exportar/importar em app.py)
MODEL_FLAN = "google/flan-t5-base"
MODEL_MT5 = "google/mt5-small"
MODEL_SUMMARIZER = "facebook/bart-large-cnn"  # summarizer em EN (fallback)
MODEL_MARIAN_PT_EN = "Helsinki-NLP/opus-mt-pt-en"
MODEL_MARIAN_EN_PT = "Helsinki-NLP/opus-mt-en-pt"

_loaded = {}

# ---------- helpers ----------
def strip_extra_ids(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_pipeline(task, model_name):
    """Cria pipeline com tratamento simples de erros (retorna None se falhar)."""
    try:
        return pipeline(task, model=model_name, tokenizer=model_name, device=device)
    except Exception as e:
        print(f"safe_pipeline: falha ao carregar {model_name} para task {task}: {e}")
        return None

def safe_generate(pipe, prompt, max_new_tokens=200, deterministic=True):
    if pipe is None:
        return ""
    try:
        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if deterministic:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update({"do_sample": True, "top_p": 0.92, "temperature": 0.8})
        out = pipe(prompt, **gen_kwargs, num_return_sequences=1)
        if isinstance(out, list) and out:
            return strip_extra_ids(out[0].get("generated_text") or out[0].get("text") or str(out[0]))
        return str(out)
    except Exception as e:
        print("safe_generate erro:", e)
        return ""

# ---------- Traduções ----------
def translate_pt_to_en(text: str) -> str:
    pipe = safe_pipeline("translation", MODEL_MARIAN_PT_EN)
    if pipe:
        try:
            res = pipe(text, max_length=512)
            return strip_extra_ids(res[0].get("translation_text", ""))
        except Exception as e:
            print("Marian PT->EN falhou:", e)
    # fallback via MT5
    gen = safe_pipeline("text2text-generation", MODEL_MT5)
    return safe_generate(gen, f"Translate to English: {text}", max_new_tokens=256)

def translate_en_to_pt(text: str) -> str:
    pipe = safe_pipeline("translation", MODEL_MARIAN_EN_PT)
    if pipe:
        try:
            res = pipe(text, max_length=512)
            return strip_extra_ids(res[0].get("translation_text", ""))
        except Exception as e:
            print("Marian EN->PT falhou:", e)
    gen = safe_pipeline("text2text-generation", MODEL_MT5)
    return safe_generate(gen, f"Translate to Portuguese: {text}", max_new_tokens=256)

def ensure_english_if_possible(text: str):
    pt_indicators = [" que ", " não ", " para ", " por ", " com ", " é ", " está "]
    if any(w in text.lower() for w in pt_indicators):
        try:
            return translate_pt_to_en(text), True
        except Exception:
            return text, False
    return text, False

# ---------- Resumo ----------
def summarize_text(text: str) -> str:
    """
    Versão robusta de summarize_text:
    - tenta pipeline('summarization') com MODEL_SUMMARY (BART)
    - se falhar, tenta pipeline('text2text-generation') com FLAN
    - limpa tokens <extra_id_*> e strings vazias
    - em caso de saída vazia, retorna debug com o raw_output (útil para ver o que deu errado)
    """
    from transformers import Pipeline
    from transformers.pipelines import AggregationStrategy

    if not text or not text.strip():
        return ""

    # parâmetros
    MODEL_SUMMARY = "facebook/bart-large-cnn"   # bom para resumo em inglês
    MODEL_MT5 = "google/mt5-small"              # fallback multilingual
    MODEL_FLAN = "google/flan-t5-base"          # fallback instruction
    MAX_TOKENS_SUMMARY = 180
    MIN_TOKENS_SUMMARY = 30

    txt = text.strip()
    try:
        # 1) tentar pipeline summarization (melhor para resumos)
        try:
            summarizer = pipeline("summarization", model=MODEL_SUMMARY, tokenizer=MODEL_SUMMARY, device=device)
            out = summarizer(txt, max_length=MAX_TOKENS_SUMMARY, min_length=MIN_TOKENS_SUMMARY, do_sample=False)
            # a saída costuma ser [{'summary_text': '...'}]
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                summary_text = out[0].get("summary_text") or ""
            else:
                # se não no formato esperado, converte para string bruta
                summary_text = str(out)
            summary_text = strip_extra_ids(summary_text).strip()
            if summary_text:
                return summary_text
        except Exception as e_summ:
            # não fatal, tentamos fallback abaixo
            print("Aviso: summarizer pipeline falhou:", e_summ)

        # 2) fallback: MT5 multilingual (útil se texto não for inglês)
        try:
            mtpipe = pipeline("text2text-generation", model=MODEL_MT5, tokenizer=MODEL_MT5, device=device)
            prompt = f"Resuma em português de forma clara e objetiva:\n\n{txt}"
            res = mtpipe(prompt, max_new_tokens=MAX_TOKENS_SUMMARY, do_sample=False, num_return_sequences=1)
            # res formato variável: lista de dicts com 'generated_text' ou 'text'
            if isinstance(res, list) and len(res) > 0:
                candidate = res[0].get("generated_text") or res[0].get("text") or str(res[0])
            else:
                candidate = str(res)
            candidate = strip_extra_ids(candidate).strip()
            if candidate:
                return candidate
        except Exception as e_mt:
            print("Aviso: MT5 fallback falhou:", e_mt)

        # 3) fallback final: FLAN (instruído)
        try:
            flan = pipeline("text2text-generation", model=MODEL_FLAN, tokenizer=MODEL_FLAN, device=device)
            prompt = f"Instruction: Resuma o texto a seguir de forma objetiva e clara em português.\n\nInput: {txt}\n\nOutput:"
            res2 = flan(prompt, max_new_tokens=MAX_TOKENS_SUMMARY, do_sample=False, num_return_sequences=1)
            if isinstance(res2, list) and len(res2) > 0:
                candidate2 = res2[0].get("generated_text") or res2[0].get("text") or str(res2[0])
            else:
                candidate2 = str(res2)
            candidate2 = strip_extra_ids(candidate2).strip()
            if candidate2:
                return candidate2
        except Exception as e_flan:
            print("Aviso: FLAN fallback falhou:", e_flan)

        # se chegamos aqui, nenhum método gerou texto limpo — devolve debug para identificar
        debug_msg = "(sem resumo gerado) Raw outputs podem estar nos logs do servidor."
        print("DEBUG: summarize_text não conseguiu gerar texto limpo; verifique logs.")
        return debug_msg

    except Exception as e:
        print("Erro inesperado em summarize_text:", e)
        return "(erro ao gerar resumo)"


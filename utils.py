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
    Resume o texto (espera receber um texto com tamanho OK — o app verifica o limite antes).
    - Se o texto estiver em PT: traduz PT->EN, resume em EN, traduz EN->PT.
    - Se houver falha com summarizer, tenta fallback com FLAN.
    """
    txt = (text or "").strip()
    if not txt:
        return ""

    try:
        txt_en, was_translated = ensure_english_if_possible(txt)

        # tenta summarizer (primário)
        summ_pipe = safe_pipeline("summarization", MODEL_SUMMARIZER)
        summary_en = ""
        if summ_pipe:
            try:
                out = summ_pipe(txt_en, max_length=200, min_length=30, do_sample=False)
                summary_en = out[0].get("summary_text", "")
            except Exception as e:
                print("summarizer pipeline falhou:", e)

        # fallback generator se summarizer falhar
        if not summary_en:
            gen = safe_pipeline("text2text-generation", MODEL_FLAN)
            prompt = f"Summarize briefly and clearly the following text:\n\n{txt_en}"
            summary_en = safe_generate(gen, prompt, max_new_tokens=220)

        summary_en = strip_extra_ids(summary_en)
        if was_translated and summary_en:
            return translate_en_to_pt(summary_en)
        return summary_en or "(sem resumo gerado)"
    except Exception as e:
        print("Erro em summarize_text:", e)
        return f"(erro ao gerar resumo: {e})"

# utils.py
# Funções auxiliares: tradução, resumo e geração segura.
# Observações:
# - Se usar repositórios privados/gated do Hugging Face, exporte
#   HF_TOKEN como variável de ambiente: export HF_TOKEN="seu_token"  (Linux/mac)
#   setx HF_TOKEN "seu_token" (Windows, depois reinicie o terminal)
# - Usa modelos públicos Helsinki-NLP e google/flan-t5-base como fallback.

import re
import os
try:
    import torch
    device = 0 if torch.cuda.is_available() else -1
except Exception as _e:
    # Se não carregar o torch (ex: Windows com DLL problem), seguimos com CPU
    torch = None
    device = -1

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Modelos (padrões públicos)
MODEL_FLAN = "google/flan-t5-base"
MODEL_MT5 = "google/mt5-small"
MARIAN_PT_EN = "Helsinki-NLP/opus-mt-pt-en"
MARIAN_EN_PT = "Helsinki-NLP/opus-mt-en-pt"
SUMMARIZER_EN = "facebook/bart-large-cnn"  # sumarizador em inglês

# Limites úteis (o app usa estes valores)
MAX_TRANSLATE_CHARS = 2000
MAX_SUMMARY_CHARS = 4000

# cache simples
_loaded = {}

def strip_extra_ids(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_pipeline(task, model_name, **kwargs):
    """
    Carrega pipeline com tratamento de token se houver HF token.
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    use_auth = {"use_auth_token": hf_token} if hf_token else {}
    return pipeline(task, model=model_name, device=device, **use_auth, **kwargs)

def safe_generate_text(pipe, prompt, max_new_tokens=150, deterministic=True):
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if deterministic:
        gen_kwargs.update({"do_sample": False})
    else:
        gen_kwargs.update({"do_sample": True, "top_p": 0.92, "temperature": 0.8})
    try:
        res = pipe(prompt, **gen_kwargs, num_return_sequences=1)
        if isinstance(res, list) and len(res) > 0:
            # diferentes pipelines usam chaves diferentes
            txt = res[0].get("generated_text") or res[0].get("translation_text") or res[0].get("text") or res[0].get("summary_text")
            return strip_extra_ids(str(txt))
        return strip_extra_ids(str(res))
    except Exception as e:
        # retorno vazio no fallback
        return ""

# ---------- Tradução PT->EN e EN->PT ----------
def translate_pt_to_en(text: str) -> str:
    if not text:
        return ""
    try:
        # tenta Marian (rápido, eficiente)
        pipe = safe_pipeline("translation", MARIAN_PT_EN)
        out = pipe(text, max_length=MAX_TRANSLATE_CHARS+50)
        txt = out[0].get("translation_text") or out[0].get("generated_text") or str(out[0])
        return strip_extra_ids(txt)
    except Exception:
        # fallback para MT5/FLAN
        try:
            pipe = safe_pipeline("text2text-generation", MODEL_MT5)
            prompt = f"Translate to English: {text}"
            return safe_generate_text(pipe, prompt, max_new_tokens=200, deterministic=True)
        except Exception:
            return text  # retorna original como último recurso

def translate_en_to_pt(text: str) -> str:
    if not text:
        return ""
    try:
        pipe = safe_pipeline("translation", MARIAN_EN_PT)
        out = pipe(text, max_length=MAX_TRANSLATE_CHARS+50)
        txt = out[0].get("translation_text") or out[0].get("generated_text") or str(out[0])
        return strip_extra_ids(txt)
    except Exception:
        try:
            pipe = safe_pipeline("text2text-generation", MODEL_MT5)
            prompt = f"Translate to Portuguese: {text}"
            return safe_generate_text(pipe, prompt, max_new_tokens=200, deterministic=True)
        except Exception:
            return text

# ---------- Função para resumir texto (workflows seguros) ----------
def summarize_text(text: str) -> str:
    """
    Resumir texto esportivo:
    - Se texto > MAX_SUMMARY_CHARS: retorna aviso (não faz truncagem automática).
    - Workflow: Portugal (PT) -> traduzir para EN -> sumarizar em EN -> traduzir de volta para PT.
    - Usa summarizer em inglês (BART) via pipeline summarization; fallback via FLAN-MT5.
    """
    txt = (text or "").strip()
    if not txt:
        return ""

    if len(txt) > MAX_SUMMARY_CHARS:
        # O app quer apenas notificar — não truncaremos automaticamente
        return f"[TOO_LONG] O texto tem {len(txt)} caracteres — excede o limite de {MAX_SUMMARY_CHARS}. Por favor reduza o texto antes de resumir."

    try:
        # Detecta se é português (heurística simples)
        pt_indicators = [" que ", " não ", " para ", " por ", " com ", " é ", " está "]
        needs_translate = any(w in txt.lower() for w in pt_indicators)

        if needs_translate:
            en_text = translate_pt_to_en(txt)
        else:
            en_text = txt

        # Summarize (em inglês)
        try:
            summarizer = safe_pipeline("summarization", SUMMARIZER_EN)
            summ = summarizer(en_text, max_length=200, min_length=30, do_sample=False)
            summary_en = summ[0].get("summary_text") or str(summ[0])
            summary_en = strip_extra_ids(summary_en)
        except Exception:
            # fallback: usar FLAN como generator para resumir
            gen = safe_pipeline("text2text-generation", MODEL_FLAN)
            prompt = f"Summarize briefly and clearly the following text:\n\n{en_text}"
            summary_en = safe_generate_text(gen, prompt, max_new_tokens=200, deterministic=True)

        if needs_translate and summary_en:
            summary_pt = translate_en_to_pt(summary_en)
            return summary_pt
        return summary_en or "(sem resumo gerado)"
    except Exception as e:
        return f"(erro ao gerar resumo: {e})"

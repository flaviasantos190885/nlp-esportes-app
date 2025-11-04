# início de utils.py — import torch de forma segura
import re
try:
    import torch
    device = 0 if torch.cuda.is_available() else -1
except Exception as _e:
    # não encontrou/erro ao carregar torch -> continua com device CPU (-1)
    print("Aviso: PyTorch não disponível ou falha ao carregar (DLL). Seguir usando device CPU. Erro:", _e)
    torch = None
    device = -1

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# Modelos estáveis em CPU
MODEL_FLAN = "google/flan-t5-base"
MODEL_PT_T5 = "unicamp-dl/ptt5-base-portuguese-vocab"
MODEL_MARIAN_PT_EN = "Helsinki-NLP/opus-mt-pt-en"
MODEL_MARIAN_EN_PT = "Helsinki-NLP/opus-mt-en-pt"

device = 0 if torch.cuda.is_available() else -1
_loaded = {}


# ---------------------- FUNÇÕES AUXILIARES ----------------------

def strip_extra_ids(text: str) -> str:
    """Remove tokens extras (<extra_id_0>) e espaços duplicados."""
    if not text:
        return ""
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_generate(pipe, prompt, max_new_tokens=200):
    """Geração com limpeza e fallback seguro."""
    try:
        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if isinstance(out, list) and len(out) > 0:
            text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
            return strip_extra_ids(text)
    except Exception as e:
        print("Erro em safe_generate:", e)
    return ""


# ---------------------- TRADUÇÃO ----------------------

def translate_pt_to_en(text: str) -> str:
    """Tradução Português → Inglês usando modelo MarianMT estável."""
    try:
        pipe = pipeline("translation", model=MODEL_MARIAN_PT_EN, device=device)
        result = pipe(text, max_length=400)
        return strip_extra_ids(result[0]["translation_text"])
    except Exception as e:
        print("Erro Marian PT->EN:", e)
        # fallback FLAN
        pipe = pipeline("text2text-generation", model=MODEL_FLAN, device=device)
        return safe_generate(pipe, f"Translate to English: {text}")


def translate_en_to_pt(text: str) -> str:
    """Tradução Inglês → Português usando modelo MarianMT estável."""
    try:
        pipe = pipeline("translation", model=MODEL_MARIAN_EN_PT, device=device)
        result = pipe(text, max_length=400)
        return strip_extra_ids(result[0]["translation_text"])
    except Exception as e:
        print("Erro Marian EN->PT:", e)
        pipe = pipeline("text2text-generation", model=MODEL_FLAN, device=device)
        return safe_generate(pipe, f"Traduza para português: {text}")


def ensure_english_if_possible(text: str):
    """Se texto parece português, traduz para inglês antes de processar."""
    pt_words = [" que ", " não ", " para ", " por ", " com ", " é ", " está ", " será ", " também "]
    if any(w in text.lower() for w in pt_words):
        try:
            return translate_pt_to_en(text), True
        except Exception:
            return text, False
    return text, False


# ---------------------- RESUMO ----------------------

def summarize_text(text: str) -> str:
    """
    Resume texto esportivo:
      - Se for em português, traduz -> resume -> traduz de volta.
      - Usa FLAN-T5 para gerar resumos curtos e coerentes.
    """
    text = text.strip()
    if not text:
        return ""

    try:
        txt_en, translated = ensure_english_if_possible(text)
        pipe = pipeline("text2text-generation", model=MODEL_FLAN, device=device)
        prompt = (
            f"Summarize this sports-related text briefly and clearly:\n\n{txt_en}"
        )
        summary = safe_generate(pipe, prompt)
        summary = strip_extra_ids(summary)
        if translated and summary:
            summary = translate_en_to_pt(summary)
        return summary or "(sem resumo gerado)"
    except Exception as e:
        print("Erro no resumo:", e)
        return "(erro ao gerar resumo)"

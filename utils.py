# utils.py
import re
import math
from typing import Tuple

# ----------------- import torch de forma segura -----------------
try:
    import torch
    device = 0 if torch.cuda.is_available() else -1
except Exception as _e:
    # Ao não conseguir carregar torch (Win DLL error etc), prosseguir em CPU (-1)
    print("Aviso: PyTorch não disponível ou falha ao carregar. Seguir usando device CPU. Erro:", _e)
    torch = None
    device = -1

# transformers
from transformers import pipeline

# ----------------- model names (ajuste se quiser) -----------------
MODEL_FLAN = "google/flan-t5-base"
MODEL_MARIAN_PT_EN = "Helsinki-NLP/opus-mt-pt-en"
MODEL_MARIAN_EN_PT = "Helsinki-NLP/opus-mt-en-pt"

# ----------------- helpers -----------------
def strip_extra_ids(text: str) -> str:
    """Remove tokens do tipo <extra_id_0> e espaços duplicados."""
    if not text:
        return ""
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_generate(pipe, prompt: str, max_new_tokens: int = 200, do_sample: bool = False) -> str:
    """
    Gera texto com um pipeline (text2text-generation ou text-generation) e limpa o retorno.
    Retorna string limpa (ou '' em caso de erro).
    """
    try:
        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, num_return_sequences=1)
        if isinstance(out, list) and len(out) > 0:
            # modelos T5-style: 'generated_text' ou 'text' ou 'summary_text'
            text = out[0].get("generated_text") or out[0].get("text") or out[0].get("summary_text") or str(out[0])
            return strip_extra_ids(text)
        # às vezes pipeline retorna dict
        if isinstance(out, dict):
            text = out.get("generated_text") or out.get("text") or out.get("summary_text") or str(out)
            return strip_extra_ids(text)
    except Exception as e:
        print("safe_generate erro:", e)
    return ""

# ----------------- Traduções -----------------
# ----- em utils.py: importe strip_extra_ids no topo (se já não tiver) -----
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# (garanta que strip_extra_ids está definido neste arquivo)

def translate_pt_to_en(text: str) -> str:
    """Tradução PT -> EN com fallback. Retorna apenas o texto limpo."""
    text = text.strip()
    if not text:
        return ""
    try:
        # pipeline de tradução (Marian) - sensor: 'translation' funciona para modelos Marian
        pipe = pipeline("translation", model=MODEL_MARIAN_PT_EN, device=device)
        out = pipe(text, max_length=512)
        # alguns pipelines retornam "translation_text"
        translated = out[0].get("translation_text") if isinstance(out, list) else None
        if not translated:
            # fallback ao formato padrão
            translated = out[0].get("generated_text") if isinstance(out, list) else str(out)
        return strip_extra_ids(translated)
    except Exception as e:
        # fallback via gerador (FLAN) caso Marian falhe
        try:
            gen = pipeline("text2text-generation", model=MODEL_FLAN, tokenizer=MODEL_FLAN, device=device)
            prompt = f"Translate to English (short and natural): {text}"
            out = gen(prompt, max_new_tokens=200, do_sample=False, num_return_sequences=1)
            # extrai resultado e remove tokens extras
            candidate = out[0].get("generated_text") or out[0].get("text") or str(out[0])
            return strip_extra_ids(candidate)
        except Exception as e2:
            print("Fallback FLAN PT->EN falhou:", e2)
            return text  # devolve original como último recurso


def translate_en_to_pt(text: str) -> str:
    """Tradução EN -> PT com fallback. Retorna apenas o texto limpo."""
    text = text.strip()
    if not text:
        return ""
    try:
        pipe = pipeline("translation", model=MODEL_MARIAN_EN_PT, device=device)
        out = pipe(text, max_length=512)
        translated = out[0].get("translation_text") if isinstance(out, list) else None
        if not translated:
            translated = out[0].get("generated_text") if isinstance(out, list) else str(out)
        return strip_extra_ids(translated)
    except Exception as e:
        try:
            gen = pipeline("text2text-generation", model=MODEL_FLAN, tokenizer=MODEL_FLAN, device=device)
            prompt = f"Translate to Portuguese (short and natural): {text}"
            out = gen(prompt, max_new_tokens=200, do_sample=False, num_return_sequences=1)
            candidate = out[0].get("generated_text") or out[0].get("text") or str(out[0])
            return strip_extra_ids(candidate)
        except Exception as e2:
            print("Fallback FLAN EN->PT falhou:", e2)
            return text


def ensure_english_if_possible(text: str) -> Tuple[str, bool]:
    """
    Detecta se o texto aparenta ser PT (por heurística simples).
    Se for PT, traduz PT->EN e retorna (texto_em_ingles, True).
    Se não, retorna (texto, False).
    """
    text = (text or "").strip()
    if not text:
        return "", False
    pt_indicators = [" que ", " não ", " para ", " por ", " com ", " é ", " está ", " também ", "será"]
    low = text.lower()
    if any(tok in low for tok in pt_indicators):
        try:
            en = translate_pt_to_en(text)
            return en or text, True
        except Exception:
            return text, False
    return text, False

# ----------------- Chunking para resumo -----------------
def _chunk_text(text: str, max_chars: int = 3000):
    """
    Divide o texto em pedaços com até ~max_chars caracteres, preferindo cortar em espaços.
    Retorna lista de strings.
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        if end >= L:
            chunks.append(text[start:L].strip())
            break
        # tenta cortar no último espaço antes de 'end'
        cut = text.rfind(" ", start, end)
        if cut <= start:
            cut = end  # força corte
        chunks.append(text[start:cut].strip())
        start = cut
    return chunks

# ----------------- Resumo (robusto) -----------------
def summarize_text(text: str) -> str:
    """
    Resumo robusto:
      - traduz PT->EN se detectado PT
      - divide o texto em chunks
      - resume cada chunk com FLAN-T5
      - junta parciais e resume final (se múltiplas parciais)
      - traduz de volta para PT se necessário
    """
    text = (text or "").strip()
    if not text:
        return ""

    try:
        txt_en, was_translated = ensure_english_if_possible(text)
    except Exception as e:
        print("ensure_english falhou:", e)
        txt_en, was_translated = text, False

    # pipeline FLAN
    try:
        pipe = pipeline("text2text-generation", model=MODEL_FLAN, device=device)
    except Exception as e:
        print("Erro ao criar pipeline FLAN (resumo):", e)
        return "(erro ao criar pipeline de resumo)"

    # chunking
    chunks = _chunk_text(txt_en, max_chars=2800)
    if not chunks:
        return ""

    partials = []
    for i, ch in enumerate(chunks):
        try:
            prompt = f"Summarize this sports-related text briefly and clearly:\n\n{ch}"
            s = safe_generate(pipe, prompt, max_new_tokens=180, do_sample=False)
            s = strip_extra_ids(s)
            if s:
                partials.append(s)
        except Exception as e:
            print(f"Erro resumindo chunk {i}:", e)

    if not partials:
        return "(sem resumo gerado)"

    # se houver várias parciais, gerar resumo final
    if len(partials) > 1:
        joined = "\n\n".join(partials)
        try:
            prompt_final = f"Summarize the following text into a short, clear paragraph:\n\n{joined}"
            final_summary = safe_generate(pipe, prompt_final, max_new_tokens=220, do_sample=False)
            final_summary = strip_extra_ids(final_summary)
        except Exception as e:
            print("Erro ao gerar resumo final:", e)
            final_summary = " ".join(partials)[:1000]
    else:
        final_summary = partials[0]

    # traduz de volta para PT se o original foi PT
    if was_translated and final_summary:
        try:
            final_summary = translate_en_to_pt(final_summary)
        except Exception as e:
            print("Falha ao traduzir resumo de volta para PT:", e)

    return final_summary or "(sem resumo gerado)"

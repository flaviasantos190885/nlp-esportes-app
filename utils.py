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
    Resumir texto esportivo:
    - Se o texto exceder MAX_SUMMARY_CHARS: trunca automaticamente aos primeiros MAX_SUMMARY_CHARS caracteres.
    - Workflow: se for PT, traduz PT->EN -> sumariza em EN -> traduz EN->PT.
    - Retorna o resumo (ou mensagem de erro).
    """
    txt = (text or "").strip()
    if not txt:
        return ""

    # Se exceder o limite, trunca automaticamente (sem pedir confirmação)
    truncated = False
    if len(txt) > MAX_SUMMARY_CHARS:
        txt = txt[:MAX_SUMMARY_CHARS]
        truncated = True

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
            # se foi truncado, acrescenta aviso curto (opcional) no retorno
            if truncated:
                return f"(entrada truncada a {MAX_SUMMARY_CHARS} caracteres) \n\n{summary_pt}"
            return summary_pt
        else:
            if truncated:
                return f"(entrada truncada a {MAX_SUMMARY_CHARS} caracteres) \n\n{summary_en}"
            return summary_en or "(sem resumo gerado)"
    except Exception as e:
        return f"(erro ao gerar resumo: {e})"

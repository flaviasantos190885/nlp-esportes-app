# utils.py (trecho a atualizar/substituir)
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_MT5 = "google/mt5-small"
MARIA_PT_EN = "Helsinki-NLP/opus-mt-pt-en"   # público
MARIA_EN_PT = "Helsinki-NLP/opus-mt-en-pt"   # público
SUMMARIZER_EN = "facebook/bart-large-cnn"   # summarizer confiável em inglês
device = 0 if torch.cuda.is_available() else -1
_loaded = {}

def strip_extra_ids(text: str) -> str:
    if not text:
        return text
    # remove tokens do tipo <extra_id_0>, <extra_id_1> etc. e espaços duplicados
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def try_load_marien(name):
    key = f"trans_{name}"
    if key in _loaded:
        return _loaded[key]
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        _loaded[key] = (tok, model)
        return tok, model
    except Exception:
        return None, None

def translate_pt_to_en(text: str) -> str:
    """Tenta Marian; se falhar usa MT5 como fallback. Sempre limpa extra_id tokens."""
    tok, model = try_load_marien(MARIA_PT_EN)
    if tok:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        outs = model.generate(**inputs, max_length=512)
        out = tok.decode(outs[0], skip_special_tokens=True)
        return strip_extra_ids(out)
    # fallback via MT5 pipeline
    try:
        pipe = pipeline("text2text-generation", model=MODEL_MT5, tokenizer=MODEL_MT5, device=device)
        out = pipe(f"Translate to English: {text}", max_new_tokens=256, do_sample=False, num_return_sequences=1)
        out_txt = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        return strip_extra_ids(out_txt)
    except Exception:
        return text

def translate_en_to_pt(text: str) -> str:
    """Tenta Marian; se falhar usa MT5. Limpa tokens sentinel."""
    tok, model = try_load_marien(MARIA_EN_PT)
    if tok:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        outs = model.generate(**inputs, max_length=512)
        out = tok.decode(outs[0], skip_special_tokens=True)
        return strip_extra_ids(out)
    try:
        pipe = pipeline("text2text-generation", model=MODEL_MT5, tokenizer=MODEL_MT5, device=device)
        out = pipe(f"Translate to Portuguese: {text}", max_new_tokens=256, do_sample=False, num_return_sequences=1)
        out_txt = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        return strip_extra_ids(out_txt)
    except Exception:
        return text

def ensure_english_if_possible(text: str):
    """Se texto aparenta ser PT, retorna versão EN e flag True; senão retorna text, False."""
    pt_indicators = [" que ", " não ", " para ", " por ", " com ", " é ", " está ", " será ", " também "]
    if any(w in text.lower() for w in pt_indicators):
        try:
            en = translate_pt_to_en(text)
            return en, True
        except Exception:
            return text, False
    return text, False

def summarize_text(text: str) -> str:
    """
    Função segura de resumo:
      - tenta resumir diretamente (se texto em EN usa BART)
      - se texto em PT: traduz PT->EN, resume em EN (BART), traduz EN->PT
      - se saída conter <extra_id_*> aplica fallback gerado por MT5
    """
    text = (text or "").strip()
    if not text:
        return ""

    # detecta PT simples
    en_text, translated = ensure_english_if_possible(text)
    try:
        # usa BART (robusto) para resumir em inglês (ou texto já em inglês)
        summarizer = pipeline("summarization", model=SUMMARIZER_EN, tokenizer=SUMMARIZER_EN, device=device)
        summary = summarizer(en_text, max_length=200, min_length=30, do_sample=False)
        summary_text = summary[0].get("summary_text") or ""
        summary_text = strip_extra_ids(summary_text)
        if translated:
            # traduzir de volta para PT
            try:
                summary_text = translate_en_to_pt(summary_text)
            except Exception:
                pass
        # se ainda tiver sentinel ou for muito curto, tenta fallback via mt5
        if not summary_text or re.search(r"<extra_id_\d+>", summary_text) or len(summary_text) < 20:
            # fallback via MT5 generator (prompts em PT direto)
            gen = pipeline("text2text-generation", model=MODEL_MT5, tokenizer=MODEL_MT5, device=device)
            prompt = f"Resuma em português de forma clara e curta: {text}"
            out = gen(prompt, max_new_tokens=180, do_sample=False, num_return_sequences=1)
            s = out[0].get("generated_text") or out[0].get("text") or str(out[0])
            return strip_extra_ids(s)
        return summary_text
    except Exception:
        # fallback simples: usar MT5 direto
        try:
            gen = pipeline("text2text-generation", model=MODEL_MT5, tokenizer=MODEL_MT5, device=device)
            prompt = f"Resuma em português de forma clara e curta: {text}"
            out = gen(prompt, max_new_tokens=180, do_sample=False, num_return_sequences=1)
            s = out[0].get("generated_text") or out[0].get("text") or str(out[0])
            return strip_extra_ids(s)
        except Exception:
            return ""

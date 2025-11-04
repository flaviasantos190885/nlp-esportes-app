import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_MT5 = "google/mt5-small"
MARIA_PT_EN = "Helsinki-NLP/opus-mt-pt-en"
MARIA_EN_PT = "Helsinki-NLP/opus-mt-en-pt"
device = 0 if torch.cuda.is_available() else -1
_loaded = {}

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

def translate_pt_to_en(text):
    tok, model = try_load_marien(MARIA_PT_EN)
    if tok:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        outs = model.generate(**inputs, max_length=512)
        return tok.decode(outs[0], skip_special_tokens=True)
    mtpipe = pipeline("text2text-generation", model=MODEL_MT5, tokenizer=MODEL_MT5, device=device)
    out = mtpipe(f"Translate to English: {text}", max_new_tokens=256)
    return out[0]["generated_text"]

def translate_en_to_pt(text):
    tok, model = try_load_marien(MARIA_EN_PT)
    if tok:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        outs = model.generate(**inputs, max_length=512)
        return tok.decode(outs[0], skip_special_tokens=True)
    mtpipe = pipeline("text2text-generation", model=MODEL_MT5, tokenizer=MODEL_MT5, device=device)
    out = mtpipe(f"Translate to Portuguese: {text}", max_new_tokens=256)
    return out[0]["generated_text"]

def ensure_english_if_possible(text):
    pt_indicators = [" que ", " não ", " para ", " por ", " com ", " é ", " está ", " será ", " também "]
    if any(w in text.lower() for w in pt_indicators):
        try:
            return translate_pt_to_en(text), True
        except Exception:
            return text, False
    return text, False

def is_bad_output(text, prompt):
    if not text or len(text) < 10:
        return True
    if text.lower().startswith(prompt.strip().lower()[:40]):
        return True
    return False

def safe_generate_text(pipe, prompt, max_new_tokens=150, deterministic=True):
    gen_kwargs = {"max_new_tokens": max_new_tokens, "early_stopping": True, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2}
    gen_kwargs.update({"do_sample": not deterministic, "top_p": 0.92, "temperature": 0.8})
    res = pipe(prompt, **gen_kwargs, num_return_sequences=1)
    return res[0].get("generated_text") or res[0].get("text") or str(res[0])

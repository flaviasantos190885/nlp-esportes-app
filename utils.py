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
MODEL_MARIAN_PT_EN = "Helsinki-NLP/opus-mt-pt-en"   # português -> inglês (padrão)
MODEL_MARIAN_EN_PT = "Helsinki-NLP/opus-mt-en-pt"   # inglês -> português (padrão)



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
# substitua as duas funções de tradução por este bloco no utils.py

def _extract_translation_result(res):
    """
    Normaliza a saída dos pipelines: aceita formatos diversos e retorna string limpa.
    """
    try:
        if res is None:
            return ""
        # pipeline('translation') normalmente retorna [{'translation_text': '...'}]
        if isinstance(res, list) and len(res) > 0:
            first = res[0]
            if isinstance(first, dict):
                # chaves possíveis
                for k in ("translation_text", "generated_text", "text", "summary_text"):
                    if k in first and first[k]:
                        return strip_extra_ids(str(first[k]))
                # nada encontrado na dict -> fallback para string do objeto
                return strip_extra_ids(str(first))
            else:
                # lista de strings ou outro
                return strip_extra_ids(str(first))
        # se receberam uma string simples
        if isinstance(res, str):
            return strip_extra_ids(res)
        # encapsulamentos inesperados -> str()
        return strip_extra_ids(str(res))
    except Exception as e:
        print("DEBUG _extract_translation_result erro:", e)
        try:
            return strip_extra_ids(str(res))
        except Exception:
            return ""


def _is_bad_translation(s: str) -> bool:
    """Detecta saídas inválidas: vazias, só pontuação ou muito curtas."""
    if not s:
        return True
    ss = s.strip()
    # só pontuação
    if all(ch in " \n\t\r.,;:!?-—()[]{}" for ch in ss):
        return True
    # poucas letras (ajuste se quiser)
    letters = sum(1 for ch in ss if ch.isalpha())
    if letters < 2:   # frase muito curta / inválida
        return True
    return False


from transformers import pipeline

def translate_pt_to_en(text: str) -> str:
    if not text or not text.strip():
        return ""
    try:
        pipe = pipeline("translation", model=MODEL_MARIAN_PT_EN, tokenizer=MODEL_MARIAN_PT_EN)
        result = pipe(text)
        return result[0]['translation_text']
    except Exception as e:
        print("Erro PT→EN:", e)
        return "(erro na tradução — tente novamente)"

def translate_en_to_pt(text: str) -> str:
    if not text or not text.strip():
        return ""
    try:
        pipe = pipeline("translation", model=MODEL_MARIAN_EN_PT, tokenizer=MODEL_MARIAN_EN_PT)
        result = pipe(text)
        return result[0]['translation_text']
    except Exception as e:
        print("Erro EN→PT:", e)
        return "(erro na tradução — tente novamente)"

    # 3) fallback FLAN
    try:
        flan = safe_pipeline("text2text-generation", MODEL_FLAN)
        if flan:
            prompt = f"Instruction: Translate the following English text to Portuguese clearly and concisely.\n\nInput: {text}\n\nOutput:"
            res3 = flan(prompt, max_new_tokens=256, do_sample=False, num_return_sequences=1)
            out3 = _extract_translation_result(res3)
            if not _is_bad_translation(out3):
                return out3
            else:
                print("DEBUG translate_en_to_pt: saída FLAN inválida:", repr(out3))
    except Exception as e:
        print("DEBUG translate_en_to_pt FLAN exception:", e)

    print("translate_en_to_pt: nenhum método produziu tradução válida. Verifique logs.")
    return "(tradução indisponível; ver logs)"

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
    Versão melhorada:
    - tenta pipeline('summarization') (BART) primeiro
    - fallbacks: MT5 -> FLAN
    - rejeita saídas muito curtas ou que contenham só pontuação
    - imprime debug nos logs se nada válido for gerado
    """
    if not text or not text.strip():
        return ""

    # modelos / parâmetros
    MODEL_SUMMARY = "facebook/bart-large-cnn"
    MODEL_MT5 = "google/mt5-small"
    MODEL_FLAN = "google/flan-t5-base"
    MAX_TOKENS_SUMMARY = 220
    MIN_TOKENS_SUMMARY = 30

    def looks_good(s: str) -> bool:
        if not s:
            return False
        s2 = s.strip()
        # rejeita só pontuação (ex: ".", "..." etc)
        if all(ch in " \n\t\r.,;:!?-—()[]{}" for ch in s2):
            return False
        # tamanho mínimo razoável (número de letras)
        letters = sum(1 for ch in s2 if ch.isalpha())
        if letters < 20:  # ajuste se quiser mais ou menos sensibilidade
            return False
        return True

    txt = text.strip()

    # helper para mostrar raw output nos logs
    def debug_log(prefix, obj):
        try:
            print(f"[summarize_text DEBUG] {prefix}: {repr(obj)[:1000]}")
        except Exception:
            print(f"[summarize_text DEBUG] {prefix}: (erro ao mostrar)")

    # 1) pipeline summarization (preferível)
    try:
        try:
            summarizer = pipeline("summarization", model=MODEL_SUMMARY, tokenizer=MODEL_SUMMARY, device=device)
            out = summarizer(txt, max_length=MAX_TOKENS_SUMMARY, min_length=MIN_TOKENS_SUMMARY, do_sample=False)
            # normalmente [{'summary_text': '...'}]
            if isinstance(out, list) and out and isinstance(out[0], dict):
                summary_text = out[0].get("summary_text", "") or ""
            else:
                summary_text = str(out)
            summary_text = strip_extra_ids(summary_text).strip()
            debug_log("BART raw", out)
            if looks_good(summary_text):
                return summary_text
            else:
                debug_log("BART rejected (too short/punct)", summary_text)
        except Exception as ex:
            debug_log("BART exception", ex)
    except Exception as e:
        print("summarize_text: erro inesperado no bloco BART:", e)

    # 2) fallback: MT5 text2text
    try:
        try:
            mtpipe = pipeline("text2text-generation", model=MODEL_MT5, tokenizer=MODEL_MT5, device=device)
            prompt = f"Resuma em português de forma clara e objetiva:\n\n{txt}"
            res = mtpipe(prompt, max_new_tokens=MAX_TOKENS_SUMMARY, do_sample=False, num_return_sequences=1)
            debug_log("MT5 raw", res)
            if isinstance(res, list) and res:
                candidate = res[0].get("generated_text") or res[0].get("text") or str(res[0])
            else:
                candidate = str(res)
            candidate = strip_extra_ids(candidate).strip()
            if looks_good(candidate):
                return candidate
            else:
                debug_log("MT5 rejected (too short/punct)", candidate)
        except Exception as ex:
            debug_log("MT5 exception", ex)
    except Exception as e:
        print("summarize_text: erro inesperado no bloco MT5:", e)

    # 3) fallback: FLAN (instruído)
    try:
        try:
            flan = pipeline("text2text-generation", model=MODEL_FLAN, tokenizer=MODEL_FLAN, device=device)
            prompt = f"Instruction: Resuma o texto a seguir de forma objetiva e clara em português.\n\nInput: {txt}\n\nOutput:"
            res2 = flan(prompt, max_new_tokens=MAX_TOKENS_SUMMARY, do_sample=False, num_return_sequences=1)
            debug_log("FLAN raw", res2)
            if isinstance(res2, list) and res2:
                candidate2 = res2[0].get("generated_text") or res2[0].get("text") or str(res2[0])
            else:
                candidate2 = str(res2)
            candidate2 = strip_extra_ids(candidate2).strip()
            if looks_good(candidate2):
                return candidate2
            else:
                debug_log("FLAN rejected (too short/punct)", candidate2)
        except Exception as ex:
            debug_log("FLAN exception", ex)
    except Exception as e:
        print("summarize_text: erro inesperado no bloco FLAN:", e)

    # nada válido — retorna mensagem curta para a UI e já deixou debug nos logs
    print("summarize_text: nenhum método gerou resumo válido. Verifique os logs debug acima.")
    return "(sem resumo gerado — consulte os logs do servidor para detalhes)"

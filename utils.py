import re
import warnings

try:
    import torch
    device = 0 if torch.cuda.is_available() else -1
except Exception as _e:
    print("Aviso: PyTorch não disponível ou falha ao carregar (DLL). Seguir usando device CPU. Erro:", _e)
    torch = None
    device = -1

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")

# ----- CONFIG ----- 
MAX_SUMMARY_CHARS = 4000  
MODEL_FLAN = "google/flan-t5-base"
MODEL_MT5 = "google/mt5-small"
MODEL_SUMMARIZER = "facebook/bart-large-cnn"  
MODEL_MARIAN_PT_EN = "Helsinki-NLP/opus-mt-pt-en"   
MODEL_MARIAN_EN_PT = "Helsinki-NLP/opus-mt-en-pt"   

_loaded = {}

# ---------- util helpers ----------

def strip_extra_ids(text: str) -> str:
    """Remove tokens sentinel <extra_id_n> e normaliza espaços."""
    if not text:
        return ""
    text = re.sub(r"<extra_id_\d+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_trivial_text(s: str) -> bool:
    """Detecta saídas triviais como '.' ou strings com poucos caracteres/letras significativas."""
    if not s:
        return True
    s2 = s.strip()
    # se só pontuação / símbolos
    if all(ch in " \n\t\r.,;:!?-—()[]{}\"'«»" for ch in s2):
        return True
    # conta letras relevantes
    letters = sum(1 for ch in s2 if ch.isalpha())
    if letters < 5:  # menos de 5 letras => provavelmente muito curto
        return True
    # conta palavras alfanuméricas
    words = re.findall(r"\w+", s2)
    if len(words) <= 2:
        return True
    return False

def safe_pipeline(task, model_name):
    """Cria pipeline com tratamento simples de erros (retorna None se falhar)."""
    try:
        return pipeline(task, model=model_name, tokenizer=model_name, device=device)
    except Exception as e:
        print(f"safe_pipeline: falha ao carregar {model_name} para task {task}: {e}")
        return None

def safe_generate(pipe, prompt, max_new_tokens=200, deterministic=True):
    """Gera texto a partir de um pipeline text2text gerenciando erros e normalizando saída."""
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
            # tenta extrair campos comuns
            first = out[0]
            if isinstance(first, dict):
                for k in ("generated_text","summary_text","translation_text","text"):
                    if k in first and first[k]:
                        return strip_extra_ids(str(first[k]))
            return strip_extra_ids(str(first))
        return strip_extra_ids(str(out))
    except Exception as e:
        print("safe_generate erro:", e)
        return ""

# ---------- Traduções ----------

def _extract_translation_result(res):
    """
    Normaliza a saída dos pipelines: aceita formatos diversos e retorna string limpa.
    """
    try:
        if res is None:
            return ""

        if isinstance(res, list) and len(res) > 0:
            first = res[0]
            if isinstance(first, dict):
                for k in ("translation_text", "generated_text", "text", "summary_text"):
                    if k in first and first[k]:
                        return strip_extra_ids(str(first[k]))
                return strip_extra_ids(str(first))
            else:
                return strip_extra_ids(str(first))

        if isinstance(res, str):
            return strip_extra_ids(res)

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
    if all(ch in " \n\t\r.,;:!?-—()[]{}" for ch in ss):
        return True
    letters = sum(1 for ch in ss if ch.isalpha())
    if letters < 2:
        return True
    return False

def translate_pt_to_en(text: str) -> str:
    """Traduz PT->EN preferindo Marian; fallback para mT5/flan se necessário."""
    if not text or not text.strip():
        return ""
    # tentativa Marian (mais direta)
    try:
        pipe = safe_pipeline("translation", MODEL_MARIAN_PT_EN)
        if pipe:
            res = pipe(text)
            out = _extract_translation_result(res)
            if not _is_bad_translation(out):
                return out
    except Exception as e:
        print("Erro PT→EN Marian:", e)

    # fallback mT5 / flan
    try:
        mtpipe = safe_pipeline("text2text-generation", MODEL_MT5)
        if mtpipe:
            prompt = f"Translate to English:\n\n{text}"
            out = safe_generate(mtpipe, prompt, max_new_tokens=256, deterministic=True)
            if not _is_bad_translation(out):
                return out
    except Exception as e:
        print("Erro PT→EN mT5:", e)

    try:
        flan = safe_pipeline("text2text-generation", MODEL_FLAN)
        if flan:
            prompt = f"Instruction: Translate the following Portuguese text to English clearly and concisely.\n\nInput: {text}\n\nOutput:"
            out = safe_generate(flan, prompt, max_new_tokens=256, deterministic=True)
            if not _is_bad_translation(out):
                return out
    except Exception as e:
        print("Erro PT→EN FLAN:", e)

    return "(tradução indisponível)"

def translate_en_to_pt(text: str) -> str:
    """Traduz EN->PT preferindo Marian; fallback para flan/mT5 se necessário."""
    if not text or not text.strip():
        return ""
    # tentativa Marian
    try:
        pipe = safe_pipeline("translation", MODEL_MARIAN_EN_PT)
        if pipe:
            res = pipe(text)
            out = _extract_translation_result(res)
            if not _is_bad_translation(out):
                return out
    except Exception as e:
        print("Erro EN→PT Marian:", e)

    # fallback flan / mT5
    try:
        flan = safe_pipeline("text2text-generation", MODEL_FLAN)
        if flan:
            prompt = f"Instruction: Translate the following English text to Portuguese clearly and concisely.\n\nInput: {text}\n\nOutput:"
            out = safe_generate(flan, prompt, max_new_tokens=256, deterministic=True)
            if not _is_bad_translation(out):
                return out
    except Exception as e:
        print("Erro EN→PT FLAN:", e)

    try:
        mtpipe = safe_pipeline("text2text-generation", MODEL_MT5)
        if mtpipe:
            prompt = f"Translate to Portuguese:\n\n{text}"
            out = safe_generate(mtpipe, prompt, max_new_tokens=256, deterministic=True)
            if not _is_bad_translation(out):
                return out
    except Exception as e:
        print("Erro EN→PT mT5:", e)

    return "(tradução indisponível)"

def ensure_english_if_possible(text: str):
    """
    Se detectar provável PT por indicadores simples, tenta traduzir para EN.
    Retorna (texto_em_ingles_ou_original, boolean_foi_traduzido)
    """
    pt_indicators = [" que ", " não ", " para ", " por ", " com ", " é ", " está ", " será ", " também "]
    if any(w in text.lower() for w in pt_indicators):
        try:
            tr = translate_pt_to_en(text)
            if not _is_bad_translation(tr):
                return tr, True
            else:
                return text, False
        except Exception:
            return text, False
    return text, False

# ---------- Summarization robusto ----------

def _chunk_text_by_words(text: str, chunk_size_words: int = 400):
    """Divide texto em blocos de ~chunk_size_words palavras, preservando ordem."""
    words = text.split()
    if not words:
        return []
    chunks = [" ".join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
    return chunks

def summarize_text(text: str) -> str:
    """
    Resumidor robusto:
    - Separa textos longos em blocos (~400 palavras por bloco).
    - Resume cada bloco (BART se disponível; fallback mT5).
    - Junta os resumos e faz um resumo final curto (ou devolve os resumos concatenados).
    - Evita saídas triviais e aplica fallbacks com prompts diferentes.
    """
    if not text or not text.strip():
        return ""

    txt = text.strip()
    # limita entradas absurdamente longas (o app checa tamanho, mas aqui defendemos novamente)
    if len(txt) > MAX_SUMMARY_CHARS * 5:
        txt = txt[:MAX_SUMMARY_CHARS * 5]  # cortar para não travar totalmente

    # configuração de geração
    CHUNK_WORDS = 400
    PER_CHUNK_MAX = 180
    PER_CHUNK_MIN = 40
    FINAL_MAX = 120
    FINAL_MIN = 30

    # tentamos carregar summarizer BART (mais confiável para summarization)
    summarizer = safe_pipeline("summarization", MODEL_SUMMARIZER)
    use_bart = summarizer is not None

    # criar chunks
    chunks = _chunk_text_by_words(txt, chunk_size_words=CHUNK_WORDS)
    if not chunks:
        return ""

    summaries = []
    # resume cada chunk
    for idx, chunk in enumerate(chunks):
        got = ""
        # se chunk curto suficiente, talvez não precise resumir (mas mantemos regra)
        try:
            if use_bart:
                try:
                    out = summarizer(chunk, max_length=PER_CHUNK_MAX, min_length=PER_CHUNK_MIN, do_sample=False)
                    if isinstance(out, list) and out and isinstance(out[0], dict):
                        got = out[0].get("summary_text") or out[0].get("generated_text") or ""
                    else:
                        got = str(out)
                    got = strip_extra_ids(got).strip()
                except Exception as e:
                    print(f"[summarize_text] BART chunk {idx} exception:", e)
                    got = ""
            # fallback mT5 for chunk if BART missing or failed
            if not got:
                mtpipe = safe_pipeline("text2text-generation", MODEL_MT5)
                if mtpipe:
                    prompt = f"Resuma em português de forma clara e objetiva (2-4 frases):\n\n{chunk}"
                    got = safe_generate(mtpipe, prompt, max_new_tokens=PER_CHUNK_MAX, deterministic=True)
                    got = strip_extra_ids(got).strip()
            # se ainda vazio/ruim, tentar FLAN deterministic
            if not got:
                flanpipe = safe_pipeline("text2text-generation", MODEL_FLAN)
                if flanpipe:
                    prompt = f"Instruction: Resuma o texto abaixo de forma objetiva e clara em português.\n\nInput: {chunk}\n\nOutput:"
                    got = safe_generate(flanpipe, prompt, max_new_tokens=PER_CHUNK_MAX, deterministic=True)
                    got = strip_extra_ids(got).strip()

            if got and not is_trivial_text(got):
                summaries.append(got)
            else:
                print(f"[summarize_text] chunk {idx} gerou saída trivial/ vazia; será ignorado ou usado fallback adicional.")
        except Exception as e:
            print(f"[summarize_text] erro ao resumir chunk {idx}:", e)
            continue

    # se não conseguimos resumos úteis de chunks, tentar um resumo direto do texto inteiro com fallback agressivo
    if not summaries:
        print("[summarize_text] Nenhum summary gerado por chunks. Tentando resumo direto (fallback agressivo).")
        # tentativa direta mT5 com sampling e retries
        mtpipe = safe_pipeline("text2text-generation", MODEL_MT5)
        if mtpipe:
            prompts = [
                f"Resuma em português em 3 frases curtas e objetivas:\n\n{txt}",
                f"Resuma em português em 2 frases muito diretas:\n\n{txt}",
                f"Resuma em 1-2 frases muito diretas:\n\n{txt}"
            ]
            for i, p in enumerate(prompts):
                try:
                    deterministic = (i == 0)
                    out = safe_generate(mtpipe, p, max_new_tokens=FINAL_MAX, deterministic=deterministic)
                    out = strip_extra_ids(out).strip()
                    if out and not is_trivial_text(out):
                        return out
                except Exception as e:
                    print("[summarize_text] fallback direto falhou:", e)

        # flan fallback
        flanpipe = safe_pipeline("text2text-generation", MODEL_FLAN)
        if flanpipe:
            try:
                prompt = f"Instruction: Faça um resumo curto e objetivo em português (2-4 frases):\n\nInput: {txt}\n\nOutput:"
                out = safe_generate(flanpipe, prompt, max_new_tokens=FINAL_MAX, deterministic=False)
                out = strip_extra_ids(out).strip()
                if out and not is_trivial_text(out):
                    return out
            except Exception as e:
                print("[summarize_text] flan fallback falhou:", e)

        # se ainda nada, devolve aviso
        return "(sem resumo gerado — consulte os logs do servidor para detalhes)"

    # Se temos vários summaries, juntamos e fazemos um resumo final (se possível)
    combined = " ".join(summaries)
    # se apenas um resumo obtido, talvez já seja suficiente; tentamos ainda gerar um resumo final mais condensado
    try:
        if use_bart:
            final_summarizer = summarizer
            out_final = final_summarizer(combined, max_length=FINAL_MAX, min_length=FINAL_MIN, do_sample=False)
            if isinstance(out_final, list) and out_final and isinstance(out_final[0], dict):
                final_txt = out_final[0].get("summary_text") or out_final[0].get("generated_text") or ""
            else:
                final_txt = str(out_final)
            final_txt = strip_extra_ids(final_txt).strip()
            if final_txt and not is_trivial_text(final_txt):
                return final_txt
            else:
                print("[summarize_text] resumo final BART foi trivial/ inválido; retornando concat de chunk summaries.")
        else:
            # sem BART, tentar mT5 no combined deterministically
            mtpipe = safe_pipeline("text2text-generation", MODEL_MT5)
            if mtpipe:
                prompt = f"Resuma em português de forma objetiva (2-3 frases):\n\n{combined}"
                out = safe_generate(mtpipe, prompt, max_new_tokens=FINAL_MAX, deterministic=True)
                out = strip_extra_ids(out).strip()
                if out and not is_trivial_text(out):
                    return out
    except Exception as e:
        print("[summarize_text] exception ao gerar resumo final:", e)

    # último recurso: retorna os primeiros dois summaries concatenados (ou apenas o combined se for curto)
    if len(summaries) == 1:
        return summaries[0]
    return " ".join(summaries[:2])

# Se quiser expor funções úteis ao importar utils
__all__ = [
    "translate_pt_to_en",
    "translate_en_to_pt",
    "ensure_english_if_possible",
    "summarize_text",
    "MAX_SUMMARY_CHARS",
    "strip_extra_ids"
]

# ---------- Traduções via Hugging Face Inference API (sem carregar modelos locais) ----------
import os
import requests

# Escolha do modelo (troque se quiser outro)
MODEL_PT_EN = "facebook/mbart-large-50-many-to-many-mmt"   # recomendado para PT->EN
MODEL_EN_PT = "facebook/mbart-large-50-many-to-many-mmt"   # o mesmo suporta EN->PT

HF_API_URL_PT_EN = f"https://api-inference.huggingface.co/models/{MODEL_PT_EN}"
HF_API_URL_EN_PT = f"https://api-inference.huggingface.co/models/{MODEL_EN_PT}"

# Lê HF token (prefira configurar no Streamlit secrets: HF_API_KEY="seu_token")
HF_API_KEY = ""
try:
    HF_API_KEY = st.secrets.get("HF_API_KEY", "")  # funciona no Streamlit Cloud se definido
except Exception:
    HF_API_KEY = os.environ.get("HF_API_KEY", "")

HEADERS = {"Accept": "application/json"}
if HF_API_KEY:
    HEADERS["Authorization"] = f"Bearer {HF_API_KEY}"

def _call_hf_inference(url: str, text: str, timeout: int = 30):
    """Chama a HF Inference API e retorna string traduzida ou None em erro."""
    payload = {"inputs": text}
    try:
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
    except requests.RequestException as e:
        print("Erro requests -> HF API:", e)
        return None

    if resp.status_code == 200:
        data = resp.json()
        # normalmente vem lista [{'translation_text': '...'}] ou [{'generated_text': '...'}]
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                for k in ("translation_text", "generated_text", "text"):
                    if k in first and first[k]:
                        return str(first[k]).strip()
            return str(first).strip()
        if isinstance(data, dict):
            for k in ("translation_text", "generated_text", "text"):
                if k in data and data[k]:
                    return str(data[k]).strip()
            return str(data).strip()
        return str(data).strip()
    else:
        print(f"HF API HTTP {resp.status_code}: {resp.text[:300]}")
        return None

def _is_bad(s: str) -> bool:
    if not s or not s.strip():
        return True
    ss = s.strip()
    if all(ch in " \n\t\r.,;:!?-—()[]{}" for ch in ss):
        return True
    letters = sum(1 for ch in ss if ch.isalpha())
    return letters < 2

def translate_pt_to_en(text: str) -> str:
    if not text or not text.strip():
        return ""
    out = _call_hf_inference(HF_API_URL_PT_EN, text)
    if out and not _is_bad(out):
        return out
    # fallback mensagem clara
    if HF_API_KEY == "":
        return "(tradução indisponível sem HF_API_KEY — configure HF_API_KEY no Streamlit secrets ou variável de ambiente)"
    return "(erro na tradução — veja logs)"

def translate_en_to_pt(text: str) -> str:
    if not text or not text.strip():
        return ""
    out = _call_hf_inference(HF_API_URL_EN_PT, text)
    if out and not _is_bad(out):
        return out
    if HF_API_KEY == "":
        return "(tradução indisponível sem HF_API_KEY — configure HF_API_KEY no Streamlit secrets ou variável de ambiente)"
    return "(erro na tradução — veja logs)"

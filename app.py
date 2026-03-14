import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Indic Speech Translator", page_icon="🌏")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

LANG_MAP = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru"
}

st.title("🌏 Indian Multilingual Translator")

col1, col2 = st.columns(2)

with col1:
    src_lang = st.selectbox("Source Language", list(LANG_MAP.keys()))

with col2:
    tgt_lang = st.selectbox("Target Language", list(LANG_MAP.keys()))

text = st.text_area("Enter text to translate")

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter text")

    elif src_lang == tgt_lang:
        st.warning("Choose different languages")

    else:
        tokenizer.src_lang = LANG_MAP[src_lang]

        inputs = tokenizer(text, return_tensors="pt")

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_MAP[tgt_lang]),
            max_length=200
        )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        st.success("Translation")
        st.write(result)
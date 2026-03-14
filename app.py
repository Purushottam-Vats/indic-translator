import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import tempfile

st.set_page_config(page_title="Indic AI Translator", page_icon="🌏", layout="centered")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

LANG_MAP = {
    "English": ("eng_Latn","en"),
    "Hindi": ("hin_Deva","hi"),
    "Kannada": ("kan_Knda","kn"),
    "Tamil": ("tam_Taml","ta"),
    "Telugu": ("tel_Telu","te"),
    "Malayalam": ("mal_Mlym","ml"),
    "Bengali": ("ben_Beng","bn"),
    "Marathi": ("mar_Deva","mr"),
    "Gujarati": ("guj_Gujr","gu"),
    "Punjabi": ("pan_Guru","pa")
}

# -------- UI DESIGN --------
st.markdown(
"""
<style>
.big-title{
font-size:40px;
font-weight:700;
text-align:center;
color:#4CAF50;
}
.subtitle{
text-align:center;
color:gray;
margin-bottom:20px;
}
.result-box{
background:#f1f3f6;
padding:15px;
border-radius:10px;
font-size:18px;
}
</style>
""",
unsafe_allow_html=True
)

st.markdown('<p class="big-title">🌏 Indic AI Translator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Translate Indian languages with AI</p>', unsafe_allow_html=True)

col1,col2 = st.columns(2)

with col1:
    src_lang = st.selectbox("Source Language", list(LANG_MAP.keys()))

with col2:
    tgt_lang = st.selectbox("Target Language", list(LANG_MAP.keys()))

text = st.text_area("Enter text")

if st.button("Translate"):

    if text.strip()=="":
        st.warning("Enter some text")

    elif src_lang==tgt_lang:
        st.warning("Choose different languages")

    else:
        tokenizer.src_lang = LANG_MAP[src_lang][0]

        inputs = tokenizer(text, return_tensors="pt")

        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_MAP[tgt_lang][0]),
            max_length=200
        )

        result = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

        st.markdown('<div class="result-box">'+result+'</div>', unsafe_allow_html=True)

        # ---------- TEXT TO SPEECH ----------
        tts = gTTS(text=result, lang=LANG_MAP[tgt_lang][1])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_file = open(fp.name, "rb")
            audio_bytes = audio_file.read()

        st.audio(audio_bytes, format="audio/mp3")

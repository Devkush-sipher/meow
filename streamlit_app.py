import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load translation model once
@st.cache_resource
def load_model():
    model_name = "ai4bharat/indictrans2-en-indic"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Font map for rendering
LANGUAGE_FONTS = {
    "Telugu": "NotoSansTelugu.ttf",
    "Hindi": "NotoSansDevanagari.ttf",
    "Tamil": "NotoSansTamil.ttf",
    "Kannada": "NotoSansKannada.ttf",
    "Malayalam": "NotoSansMalayalam.ttf",
    "Bengali": "NotoSansBengali.ttf"
}

# Language codes used by the model
LANGUAGE_CODES = {
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn"
}

# UI
st.set_page_config(page_title="Indic Infographic Generator", layout="wide")
st.title("🌐 Indic Language Infographic Generator with Translation")

language = st.selectbox("Choose target language", list(LANGUAGE_FONTS.keys()))
input_text = st.text_area("Enter your message in English:", "Save water, save life")

if st.button("Translate & Generate Infographic"):
    # Translate using Hugging Face model
    target_lang_code = LANGUAGE_CODES[language]
    input_ids = tokenizer(f"<2{target_lang_code}> " + input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Load font
    font_file = LANGUAGE_FONTS.get(language)
    if not os.path.exists(font_file):
        st.error(f"Font for {language} not found: {font_file}")
        st.stop()
    font = ImageFont.truetype(font_file, 28)

    # Create image
    img = Image.new('RGB', (700, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 180), translated_text, font=font, fill=(0, 0, 0))

    # Display and download
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=f"{language} Infographic")
    st.download_button(
        "Download Infographic",
        data=buf.getvalue(),
        file_name=f"{language}_infographic.png",
        mime="image/png"
    )

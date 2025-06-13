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
# IMPORTANT: Ensure these font files are in a 'fonts' subdirectory relative to your script.
LANGUAGE_FONTS = {
    "Telugu": "NotoSansTelugu-Regular.ttf", # Often fonts include a style like '-Regular'
    "Hindi": "NotoSansDevanagari-Regular.ttf",
    "Tamil": "NotoSansTamil-Regular.ttf",
    "Kannada": "NotoSansKannada-Regular.ttf",
    "Malayalam": "NotoSansMalayalam-Regular.ttf",
    "Bengali": "NotoSansBengali-Regular.ttf"
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
    # The tokenizer requires special tokens to specify the target language for translation
    input_ids = tokenizer(f"<2{target_lang_code}> " + input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Construct the full path to the font file
    # This assumes a 'fonts' folder exists in the same directory as your Streamlit script.
    font_file_path = os.path.join("fonts", LANGUAGE_FONTS.get(language))

    # Load font
    if not os.path.exists(font_file_path):
        st.error(f"Font for {language} not found at: {font_file_path}")
        st.warning("Please ensure the required Noto Sans font file is in a 'fonts' subfolder.")
        st.stop()
    
    # You might need to adjust the font size based on the length of translated text
    # and the overall design.
    font_size = 32 # Increased font size for better readability
    font = ImageFont.truetype(font_file_path, font_size)

    # Create image
    # Increased image size to accommodate potentially longer translated text
    img_width = 800
    img_height = 450
    img = Image.new('RGB', (img_width, img_height), color=(240, 248, 255)) # Light blue background

    draw = ImageDraw.Draw(img)

    # Calculate text bounding box to center the text
    # This helps in positioning the text better on the infographic
    bbox = draw.textbbox((0,0), translated_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text
    x_pos = (img_width - text_width) / 2
    y_pos = (img_height - text_height) / 2
    
    # Add a simple border to the image for better visual appeal
    draw.rectangle([(10, 10), (img_width - 10, img_height - 10)], outline=(100, 100, 100), width=3)

    # Draw the translated text
    draw.text((x_pos, y_pos), translated_text, font=font, fill=(0, 0, 0)) # Black text color

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

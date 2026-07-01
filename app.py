import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# 1. Konfigurasi Tampilan Awal (Tech & Clean Look)
st.set_page_config(
    page_title="Fruit Scanner AI",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS untuk mempercantik tampilan ala Futuristic AI
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top, #1e293b, #0f172a);
    }
    h1 {
        color: #10b981 !important;
        font-family: sans-serif;
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
        text-align: center;
    }
    p {
        color: #94a3b8 !important;
        text-align: center;
    }
    .stCamera {
        border: 2px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 16px !important;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header Aplikasi
st.title("🍎 Fruit Scanner AI")
st.write("Powered by CHEESE • UAS Computer Vision")
st.markdown("---")

# 2. Load Model AI dari Google Drive (Otomatis didownload jika belum ada)
@st.cache_resource
def load_fruit_model():
    file_path = 'Fruits.h5'
    
    # Mengecek apakah file model sudah ada
    if not os.path.exists(file_path):
        with st.spinner("Cheese is downloading the AI ​​model from Google Drive... (This only happens once)"):
            # PERHATIAN: Ganti teks di bawah ini dengan ID File Google Drive Anda!
            file_id = '1qagLF58FPRg6oUJliaGwyKlWzGcrkgak' 
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, file_path, quiet=False)
            
    # Mengembalikan model yang sudah di-load
    return tf.keras.models.load_model(file_path)

try:
    model = load_fruit_model()
except Exception as e:
    st.error(f"Failed to load model. Error: {e}")
    st.stop()

# Class names sesuai dengan kode Flask Anda
class_names = [
    'Avocado Green 1',
    'Pear 1',
    'apple_red_1'
]

# 3. Input Gambar (Kamera & File Uploader)
st.subheader("📸 Scanning fruits?")
img_file = st.camera_input("Place the fruit in front of the camera.")

# Fitur opsional jika kamera tidak aktif
uploaded_file = st.file_uploader("Or upload the image file manually.", type=["jpg", "jpeg", "png"])

# Tentukan sumber gambar yang digunakan
input_image = img_file if img_file is not None else uploaded_file

# 4. Proses Prediksi AI
if input_image is not None:
    # Membuka gambar dengan PIL
    img = Image.open(input_image)
    
    # Tampilkan Animasi Loading saat AI memproses
    with st.spinner("AI is analyzing... Extracting image features... Cheese wants you to wait for a while :P"):
        # Preprocessing gambar (sama persis dengan Flask Anda target_size=(224, 224))
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # Jalankan Prediksi
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        result = class_names[predicted_class]

    # 5. Tampilan Hasil Prediksi yang Keren
    st.markdown("---")
    st.success("✅ The object was successfully identified.!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Scanned image", use_container_width=True)
    
    with col2:
        st.markdown(f"### Hasil: <span style='color:#10b981; font-size:28px;'>**{result}**</span>", unsafe_allow_html=True)
        st.write(f"Confidence level (Accuracy): {confidence:.2f}%")
        
        # Progress bar interaktif untuk akurasi
        st.progress(int(confidence))

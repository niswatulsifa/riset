import streamlit as st
import gdown
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from datetime import datetime
import pytz

# Fungsi untuk mendownload model dari Google Drive
@st.cache_resource
def download_model():
    # Gantilah dengan ID file model Google Drive yang benar
    model_url = "https://drive.google.com/uc?id=1GOL7SjYXnzYH_FD4UEksC4ubVboa93JR"  # Update dengan link unduhan langsung
    output_path = "best_model.h5"
    
    if not os.path.exists(output_path):
        st.write("Mengunduh model...")  # Informasi unduhan
        gdown.download(model_url, output_path, quiet=False)
        st.write("Model berhasil diunduh!")  # Konfirmasi setelah unduhan selesai
    else:
        st.write("Model sudah ada di direktori.")
        
    return output_path

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    try:
        # Debugging: Cek lokasi dan keberadaan file
        st.write(f"Current directory: {os.getcwd()}")  # Menampilkan direktori kerja saat ini
        st.write(f"Is model file present? {os.path.exists(model_path)}")  # Mengecek apakah model ada

        # Memastikan file model ada
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model tidak ditemukan di {model_path}")
        
        # Memuat model dengan TensorFlow
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Fungsi untuk memproses gambar
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)  # Resize ke target size
    img_array = np.array(image)  # Convert ke numpy array
    if img_array.shape[-1] != 3:  # Pastikan ada 3 channel (RGB)
        raise ValueError("Gambar harus memiliki 3 channel (RGB).")
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array

# Fungsi untuk mendapatkan tanggal Indonesia (WIB)
def get_current_date():
    indonesia_tz = pytz.timezone('Asia/Jakarta')
    datetime_indonesia = datetime.now(indonesia_tz)
    return datetime_indonesia.strftime("%Y-%m-%d")  # Format: Tahun-Bulan-Hari

# Sidebar untuk navigasi
st.sidebar.title("Selamat Datang!")

# Tampilkan tanggal di sidebar
current_date = get_current_date()
st.sidebar.write(f"Tanggal: **{current_date}**")

# Pilih menu
app_mode = st.sidebar.selectbox("Pilih Menu", ["Klasifikasi", "Penulis"])

if app_mode == "Klasifikasi":
    st.title("PENERAPAN CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM) PADA  ARSITEKTUR DEEP LEARNING VGG16 UNTUK KLASIFIKASI COVID-19")
    st.text("Aplikasi ini menggunakan CBAM-VGG16.")

    # Unduh model
    model_path = download_model()

    # Muat model
    model = load_model(model_path)

    if model is not None:
        # Upload gambar
        uploaded_file = st.file_uploader("Unggah gambar X-Ray Anda", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
            try:
                # Proses gambar
                image = Image.open(uploaded_file).convert("RGB")
                processed_image = preprocess_image(image)

                # Debugging bentuk input
                st.write(f"Bentuk input gambar: {processed_image.shape}")

                # Label yang sesuai dengan kelas yang digunakan saat pelatihan (berurutan secara alfabet)
                labels = ['COVID-19', 'Normal', 'Pneumonia']

                # Prediksi dengan model
                prediction = model.predict(processed_image)

                # Ambil indeks kelas dengan probabilitas tertinggi
                predicted_class_index = np.argmax(prediction)

                # Menampilkan nama kelas sesuai dengan label
                predicted_class = labels[predicted_class_index]
                st.write(f"Prediksi: **{predicted_class}**")
                st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

elif app_mode == "Penulis":
    st.title("Penulis")
    st.write("""
        oleh : <br>
        Niswatul Sifa 210411100145 <br>
        Dosen Pembimbing Riset : Prof. Dr. Rima Tri Wahyuningrum, S.T., MT.
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from datetime import datetime
import os
from dotenv import load_dotenv

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
load_dotenv()
st.set_page_config(page_title="Lawu AI Assistant", page_icon="🏔️", layout="centered")
st.title("🏔️ Asisten Cerdas Gunung Lawu")
st.caption("Didukung oleh Machine Learning & Google Gemini")

# ==========================================
# 2. MEMUAT MACHINE LEARNING MODEL
# ==========================================
@st.cache_resource
def load_ml_model():
    return joblib.load('models/rf_danger_level_model.joblib')

try:
    ml_model = load_ml_model()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file rf_danger_level_model.joblib ada.")
    st.stop()

# ==========================================
# 3. SETUP API KEY 
# ==========================================
api_key = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ API Key tidak ditemukan di .env")
    st.stop()

genai.configure(api_key=api_key)
llm = genai.GenerativeModel('gemini-2.5-flash')

with st.sidebar:
    st.header("⚙️ Status Sistem")
    st.success("🟢 Memori Chat Aktif")
    st.success("🟢 ML Terkoneksi")
    st.write("Coba ngobrol dengan bahasa santai sehari-hari!")

# ==========================================
# 4. FUNGSI DATA MENTAH UNTUK DIANALISIS AI
# ==========================================
def tarik_data_dan_prediksi():
    # Simulasi data sensor
    data_hari_ini = [
        [1915, -7.6639, 111.189, 16.0, 15.0, 85, 0.0, 0.0, 10.0, 180, 15.0, 50, 1010.0, 3, 10.0, 0.0, 0], 
        [1913, -7.6631, 111.188, 12.0, 10.0, 88, 15.0, 15.0, 55.0, 180, 60.0, 90, 1008.0, 63, 10.0, 0.0, 0],
        [1496, -7.5950, 111.155, 18.0, 18.0, 80, 0.0, 0.0, 5.0, 180, 10.0, 20, 1012.0, 1, 10.0, 0.0, 0]   
    ]
    df_input = pd.DataFrame(data_hari_ini, columns=[
        'Elevasi (mdpl)', 'Lat', 'Lon', 'Suhu (C)', 'Suhu Terasa (C)', 'Kelembaban (%)',
        'Curah Hujan (mm)', 'Hujan (mm)', 'Kecepatan Angin (km/h)', 'Arah Angin (derajat)',
        'Angin Kencang (km/h)', 'Tutupan Awan (%)', 'Tekanan Udara (hPa)', 'Kode Cuaca WMO',
        'Jarak_Titik_Api_Terdekat_KM', 'FRP_Terdekat_MW', 'Status_Kebakaran_Sekitar'
    ])
    
    # Tambahkan fitur waktu siklis saat ini
    now = datetime.now()
    hour = now.hour
    month = now.month
    
    df_input['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df_input['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df_input['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df_input['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    prediksi = ml_model.predict(df_input)
    status_label = {0: "AMAN", 1: "WASPADA", 2: "BAHAYA", 3: "DILARANG"}
    
    # Kita hanya kirimkan ANGKA STATISTIK, biarkan AI yang menjabarkan dengan natural
    laporan = f"Waktu Sistem: {now.strftime('%Y-%m-%d %H:%M')}\n"
    laporan += f"- Cemoro Sewu  -> Suhu: 16C, Hujan: 0mm/jam, Angin: 10km/h. ML: {status_label[prediksi[0]]}\n"
    laporan += f"- Cemoro Kandang -> Suhu: 12C, Hujan: 15mm/jam, Angin: 55km/h. ML: {status_label[prediksi[1]]}\n"
    laporan += f"- Candi Cetho  -> Suhu: 18C, Hujan: 0mm/jam, Angin: 5km/h. ML: {status_label[prediksi[2]]}\n"
    return laporan

# ==========================================
# 5. ANTARMUKA CHATBOT YANG PUNYA MEMORI
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Sapaan awal yang jauh lebih natural
    st.session_state.messages.append({"role": "assistant", "content": "Halo! Saya Ranger Virtual Gunung Lawu. Mau ngecek kondisi jalur atau cuaca buat muncak hari ini?"})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tulis pesanmu di sini..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        loading_text = st.empty()
        loading_text.markdown("🔄 Menganalisis kondisi Lawu...")
        
        konteks_sistem = tarik_data_dan_prediksi()
        
        # MEMBANGUN INGATAN (MEMORY) OBROLAN
        chat_memory = ""
        for msg in st.session_state.messages[-6:]: # Mengingat 6 chat terakhir
            pengirim = "User" if msg["role"] == "user" else "Asisten"
            chat_memory += f"{pengirim}: {msg['content']}\n"
            
        # PROMPT BARU: Memaksa LLM agar natural dan tidak kaku
        prompt_ke_llm = f"""
        RIWAYAT OBROLAN KITA:
        {chat_memory}
        
        DATA SENSOR & PREDIKSI MACHINE LEARNING TERKINI:
        {konteks_sistem}
        
        TUGASMU:
        Balas chat terakhir dari "User". Kamu adalah Ranger/Guide Gunung Lawu yang asyik. 
        ATURAN:
        1. Jawab dengan gaya bahasa ngobrol biasa (santai, empati, natural), JANGAN kaku seperti robot/customer service.
        2. JANGAN pernah mengulang sapaan "Halo saya asisten..." di tengah obrolan.
        3. Jelaskan kondisi cuaca berdasarkan ANGKA di data sensor secara natural. Misalnya: "Cemoro kandang lagi bahaya nih soalnya anginnya kenceng banget sampai 55 km/jam dan lagi hujan turun." 
        """
        
        try:
            response = llm.generate_content(prompt_ke_llm)
            loading_text.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            loading_text.markdown(f"⚠️ Terjadi kesalahan jaringan API: {e}")
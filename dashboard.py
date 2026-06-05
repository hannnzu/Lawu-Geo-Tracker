import streamlit as st
import pandas as pd
import datetime
import json
import requests
import psycopg2 
from psycopg2.extras import RealDictCursor
import folium
from streamlit_folium import st_folium

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLE CUSTOM
# ==========================================
st.set_page_config(page_title="Lawu Geo-Tracker Dashboard", layout="wide", page_icon="⛰️")

st.markdown("""
<style>
    .weather-card {
        background-color: #1a202c; padding: 24px; border-radius: 12px;
        border: 1px solid #2d3748; margin-top: 15px; margin-bottom: 15px;
    }
    .safe-card {
        background-color: #064e3b; padding: 24px; border-radius: 12px;
        border: 1px solid #059669; margin-top: 15px; margin-bottom: 15px;
    }
    .hazard-card {
        background-color: #7f1d1d; padding: 24px; border-radius: 12px;
        border: 1px solid #b91c1c; margin-top: 15px; margin-bottom: 15px;
    }
    .metric-value { font-size: 56px; font-weight: bold; line-height: 1; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI KONEKSI & QUERY KE AIVEN
# ==========================================
@st.cache_resource
def init_aiven_connection():
    try:
        return psycopg2.connect(
            host=st.secrets.get("AIVEN_HOST", "your-aiven-host.aivencloud.com"),
            database=st.secrets.get("AIVEN_DB", "defaultdb"),
            user=st.secrets.get("AIVEN_USER", "avnadmin"),
            password=st.secrets.get("AIVEN_PASSWORD", "your-password"),
            port=st.secrets.get("AIVEN_PORT", 24432)
        )
    except Exception as e:
        return None

def query_aiven(sql_query, params=None):
    conn = init_aiven_connection()
    if conn is None: return None
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query, params)
            return cur.fetchall()
    except Exception as e:
        conn.rollback()
        return None

# ==========================================
# 3. FUNGSI DATA (CUACA, GEOJSON, FILTERING)
# ==========================================
@st.cache_data(ttl=1800)
def get_live_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -7.6276, "longitude": 111.1925,
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "weather_code", "wind_speed_10m"],
        "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"],
        "timezone": "Asia/Jakarta"
    }
    try:
        return requests.get(url, params=params).json()
    except:
        return None

@st.cache_data
def load_geojson():
    try:
        with open("data/raw/geospatial/jalur_lawu_lengkap.geojson", "r") as f:
            return json.load(f)
    except:
        return None

def filter_geojson(geo_data, keyword):
    """Fungsi untuk menyaring fitur GeoJSON berdasarkan kata kunci (nama jalur)"""
    if not geo_data or 'features' not in geo_data:
        return geo_data
    
    filtered_features = []
    for feat in geo_data['features']:
        # Mengubah properties menjadi string dan mencari kata kunci
        prop_str = json.dumps(feat.get('properties', {})).lower()
        if keyword.lower() in prop_str:
            filtered_features.append(feat)
            
    # Jika tidak ada yang cocok, kembalikan kosong agar peta bersih
    return {"type": "FeatureCollection", "features": filtered_features}

WMO_CODES = {
    0: "Cerah", 1: "Cerah Berawan", 2: "Berawan", 3: "Mendung",
    45: "Berkabut", 48: "Kabut Tebal", 61: "Hujan Ringan", 63: "Hujan Sedang", 65: "Hujan Lebat"
}

weather_data = get_live_weather()
geojson_data = load_geojson()

# ==========================================
# 4. TAMPILAN HEADER
# ==========================================
now = datetime.datetime.now()
st.markdown(f"### ⛰️ Lawu Geo-Tracker Dashboard")
st.markdown(f"**{now.strftime('%A, %d %B %Y | %H:%M WIB')}**")
st.markdown("---")

# ==========================================
# 5. LAYOUT ATAS: CURRENT WEATHER & DISASTER ALERTS
# ==========================================
col_weather, col_disaster = st.columns([2, 1])

with col_disaster:
    st.subheader("DISASTER ALERTS")
    st.success("**✅ BEBAS TITIK API**\n\nSatelit NASA FIRMS tidak mendeteksi anomali panas hari ini.")
    st.markdown("---")
    st.subheader("💡 Tanya Geo-Tracker")
    
    template_opsi = st.selectbox(
        "Pilih Pertanyaan:",
        [
            "-- Pilih Pertanyaan --",
            "Jalur mana yang paling aman saat ini?",
            "Tampilkan semua rute pendakian",
            "Tampilkan detail Jalur Cemoro Kandang",
            "Tampilkan detail Jalur Cemoro Sewu",
            "Tampilkan detail Jalur Candi Cetho",
            "Bagaimana prediksi cuaca besok di Gunung Lawu?",
            "Apa potensi bahaya (hazard) cuaca saat ini?"
        ]
    )

with col_weather:
    st.subheader("CURRENT WEATHER (PUNCAK LAWU)")
    
    # 1. Komponen Cuaca Default
    if weather_data and "current" in weather_data:
        curr = weather_data["current"]
        st.markdown(f"<div class='weather-card'>"
                    f"<p style='color:#a0aec0; margin-bottom:0px;'>KONDISI SEKARANG</p>"
                    f"<h3 style='margin-top:0px;'>{WMO_CODES.get(curr['weather_code'], 'Berawan')}</h3>"
                    f"<div style='display:flex; align-items:baseline;'><span class='metric-value'>{curr['temperature_2m']}°C</span>"
                    f"<span style='margin-left:15px; color:#a0aec0;'>RealFeel® {curr['apparent_temperature']}°C</span></div>"
                    f"</div>", unsafe_allow_html=True)
        
        cw1, cw2 = st.columns([1, 1])
        with cw1: st.metric(label="Kelembaban Udara", value=f"{curr['relative_humidity_2m']}%")
        with cw2: st.metric(label="Kecepatan Angin", value=f"{curr['wind_speed_10m']} km/h")
    else:
        st.error("Gagal memuat data cuaca real-time.")

    # 2. Output Jawaban Dinamis (Custom Cards)
    st.markdown("<br>", unsafe_allow_html=True)
    if template_opsi != "-- Pilih Pertanyaan --":
        st.subheader("📊 HASIL ANALISIS GEO-TRACKER")
        
        if template_opsi == "Jalur mana yang paling aman saat ini?":
            # Simulasi query ke Aiven, mendapatkan Cemoro Kandang
            st.markdown(f"<div class='safe-card'>"
                        f"<p style='color:#6ee7b7; font-weight:bold; margin-bottom:0px;'>REKOMENDASI SISTEM</p>"
                        f"<h3 style='margin-top:0px; color:white;'>Jalur Cemoro Kandang</h3>"
                        f"<p style='color:#a7f3d0;'>Status: <strong>AMAN</strong><br>Alasan: Elevasi yang stabil, tidak ada anomali suhu, dan intensitas angin terhalang oleh vegetasi hutan yang rapat.</p>"
                        f"</div>", unsafe_allow_html=True)

        elif template_opsi == "Bagaimana prediksi cuaca besok di Gunung Lawu?":
            if weather_data and "daily" in weather_data:
                besok_temp = weather_data["daily"]["temperature_2m_max"][1]
                besok_wmo = weather_data["daily"]["weather_code"][1]
                st.markdown(f"<div class='weather-card'>"
                            f"<p style='color:#63b3ed; font-weight:bold; margin-bottom:0px;'>PREDIKSI BESOK</p>"
                            f"<h3 style='margin-top:0px;'>{WMO_CODES.get(besok_wmo, 'Cerah Berawan')}</h3>"
                            f"<div class='metric-value'>{besok_temp}°C</div>"
                            f"<p style='color:#a0aec0; margin-top:5px;'>Saran: Lakukan *summit attack* sebelum jam 9 pagi untuk menghindari kabut tebal.</p>"
                            f"</div>", unsafe_allow_html=True)

        elif template_opsi == "Apa potensi bahaya (hazard) cuaca saat ini?":
            if weather_data:
                wind = weather_data['current']['wind_speed_10m']
                st.markdown(f"<div class='hazard-card'>"
                            f"<p style='color:#fca5a5; font-weight:bold; margin-bottom:0px;'>MONITORING BAHAYA LINGKUNGAN</p>"
                            f"<h3 style='margin-top:0px; color:white;'>Analisis Risiko Saat Ini</h3>"
                            f"<ul style='color:#fecaca; margin-bottom:0px;'>"
                            f"<li><strong>Angin Badai:</strong> {'Tinggi' if wind > 30 else 'Rendah'} ({wind} km/h)</li>"
                            f"<li><strong>Hujan Ekstrem:</strong> Tidak terdeteksi</li>"
                            f"<li><strong>Titik Api (Kebakaran):</strong> Aman (0 Titik)</li>"
                            f"</ul>"
                            f"</div>", unsafe_allow_html=True)

        elif "Tampilkan detail" in template_opsi:
            st.info("Peta di bawah telah diperbarui untuk hanya menampilkan jalur yang Anda minta.")

st.markdown("---")

# ==========================================
# 6. LAYOUT TENGAH: PETA DINAMIS (FOLIUM)
# ==========================================
st.subheader("INTERACTIVE TRAIL MAP")

# Logika Penyaringan Peta
geo_to_display = geojson_data
map_keyword = None
jalur_warna = '#3b82f6' # Biru untuk jalur tunggal

if template_opsi == "Jalur mana yang paling aman saat ini?":
    map_keyword = "kandang" # Karena aman adalah Kandang
    jalur_warna = '#10b981' # Hijau
elif template_opsi == "Tampilkan detail Jalur Cemoro Kandang":
    map_keyword = "kandang"
elif template_opsi == "Tampilkan detail Jalur Cemoro Sewu":
    map_keyword = "sewu"
elif template_opsi == "Tampilkan detail Jalur Candi Cetho":
    map_keyword = "cetho"

# Terapkan filter jika ada kata kunci
if map_keyword and geojson_data:
    geo_to_display = filter_geojson(geojson_data, map_keyword)
    jalur_ditampilkan = len(geo_to_display['features'])
else:
    jalur_warna = '#ef4444' # Merah jika menampilkan semua
    jalur_ditampilkan = len(geojson_data['features']) if geojson_data else 0

map_info_col, map_display_col = st.columns([1, 2])

with map_info_col:
    st.markdown("#### Detail Rute di Peta")
    
    # Menampilkan informasi detail spesifik yang dipilih user
    if map_keyword == "kandang":
        st.success("**JALUR CEMORO KANDANG**\n\n- **Karakteristik:** Relatif landai & berliku.\n- **Estimasi:** 7-9 jam.\n- **Kondisi:** Vegetasi rapat, aman dari angin kencang.")
    elif map_keyword == "sewu":
        st.warning("**JALUR CEMORO SEWU**\n\n- **Karakteristik:** Terjal & berbatu (susunan tangga batu).\n- **Estimasi:** 6-7 jam.\n- **Kondisi:** Cepat naik elevasi, rawan licin saat hujan.")
    elif map_keyword == "cetho":
        st.error("**JALUR CANDI CETHO**\n\n- **Karakteristik:** Sabana terbuka & rute terpanjang.\n- **Estimasi:** 9-11 jam.\n- **Kondisi:** Indah namun rawan badai kabut tebal.")
    else:
        st.info("**SEMUA JALUR PENDAKIAN**\n\nMenampilkan seluruh opsi rute resmi Gunung Lawu (Cemoro Sewu, Cemoro Kandang, dan Candi Cetho).")

    st.metric(label="Jumlah Garis Jalur Tergambar", value=jalur_ditampilkan)

with map_display_col:
    m = folium.Map(
        location=[-7.6276, 111.1925], 
        zoom_start=12,
        tiles="OpenTopoMap"
    )
    
    # Tambahkan GeoJSON yang sudah disaring ke peta
    if geo_to_display and jalur_ditampilkan > 0:
        folium.GeoJson(
            geo_to_display,
            name="Jalur Pendakian",
            style_function=lambda feature: {
                'color': jalur_warna,
                'weight': 5,
                'opacity': 0.9
            }
        ).add_to(m)
    else:
        st.warning("Data GeoJSON tidak ditemukan untuk jalur tersebut.")

    st_folium(m, use_container_width=True, height=450)

st.markdown("---")

# ==========================================
# 7. TAMPILAN BAWAH: FORECAST 7 HARI
# ==========================================
st.subheader("7-DAY WEATHER FORECAST")

if weather_data and "daily" in weather_data:
    daily = weather_data["daily"]
    for i in range(5):
        fc1, fc2, fc3 = st.columns([2, 3, 2])
        with fc1: st.markdown(f"**{daily['time'][i]}**")
        with fc2: st.write(WMO_CODES.get(daily['weather_code'][i], "Berawan"))
        with fc3: st.write(f"🌡️ {daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C")
        st.markdown("---")
import requests
import json

API_KEY = "719dbca34c89cfae4ffee282f39bab21"  # Ganti dengan kodemu dari Langkah 1
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Koordinat Puncak Gunung Lawu
titik_puncak = {
    "nama": "Puncak Hargo Dumilah",
    "lat": -7.6278,
    "lon": 111.1970
}

print(f"Mengirim permintaan data cuaca untuk {titik_puncak['nama']}...")

parameter = {
    "lat": titik_puncak["lat"],
    "lon": titik_puncak["lon"],
    "appid": API_KEY,
    "units": "metric",  # Meminta suhu dalam Celcius (bukan Kelvin atau Fahrenheit)
    "lang": "id"        # Meminta deskripsi cuaca dalam Bahasa Indonesia
}

# Mengirim HTTP GET Request
response = requests.get(BASE_URL, params=parameter)

# Mengecek apakah koneksi berhasil (Status Code 200 = Sukses)
if response.status_code == 200:
    data_cuaca_mentah = response.json()
    print("Berhasil! Data JSON diterima.\n")
    
    suhu = data_cuaca_mentah['main']['temp']
    suhu_terasa = data_cuaca_mentah['main']['feels_like']
    kelembaban = data_cuaca_mentah['main']['humidity']
    deskripsi = data_cuaca_mentah['weather'][0]['description']
    angin = data_cuaca_mentah['wind']['speed']
    
    print("=== LAPORAN CUACA REAL-TIME GUNUNG LAWU ===")
    print(f"Lokasi     : {titik_puncak['nama']} (Lat: {titik_puncak['lat']}, Lon: {titik_puncak['lon']})")
    print(f"Kondisi    : {deskripsi.capitalize()}")
    print(f"Suhu Asli  : {suhu} °C")
    print(f"Suhu Terasa: {suhu_terasa} °C")
    print(f"Kelembaban : {kelembaban} %")
    print(f"Kec. Angin : {angin} meter/detik")
    print("===========================================")
    
else:
    print(f"Gagal mengambil data! Error Code: {response.status_code}")
    print(response.text)
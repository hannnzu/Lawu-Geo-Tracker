import requests
import csv
import os
from datetime import datetime
import dotenv

dotenv.load_dotenv()

# ==========================================
# 1. KONFIGURASI
# ==========================================
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
NAMA_FILE_CSV = "dataset_cuaca_lawu.csv"

# Daftar Pos/Titik di Gunung Lawu yang ingin dilacak cuacanya
daftar_pos = [
    {"nama_pos": "Basecamp Cemoro Sewu", "lat": -7.663901, "lon": 111.191535, "elevasi_mdpl": 1915},
    {"nama_pos": "Basecamp Candi Ceto", "lat": -7.59508, "lon": 111.157188, "elevasi_mdpl": 1466},
    {"nama_pos": "Basecamp Cemoro Kandang", "lat": -7.663160, "lon": 111.187170, "elevasi_mdpl": 1912.6},
    {"nama_pos": "Pos 3 Cemoro Sewu", "lat": -7.63531, "lon": 111.18449, "elevasi_mdpl": 2887.6},
    {"nama_pos": "Pos 3 Candi Ceto", "lat": -7.602822, "lon": 111.177754, "elevasi_mdpl": 2163.1},
    {"nama_pos": "Pos 3 Cemoro Kandang", "lat": -7.63529, "lon": 111.1845, "elevasi_mdpl": 2889.2},
    {"nama_pos": "Hargo Dumilah", "lat": -7.627324, "lon": 111.194387, "elevasi_mdpl": 3256.2},

]

# ==========================================
# 2. FUNGSI EXTRACT & TRANSFORM
# ==========================================
def ambil_data_cuaca():
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    baris_data_baru = []
    
    print(f"[{waktu_sekarang}] Memulai ekstraksi data cuaca...")
    
    for pos in daftar_pos:
        parameter = {
            "lat": pos["lat"],
            "lon": pos["lon"],
            "appid": API_KEY,
            "units": "metric",
            "lang": "id"
        }
        
        response = requests.get(BASE_URL, params=parameter)
        
        if response.status_code == 200:
            data = response.json()
            
            # Transformasi JSON ke bentuk kolom (flat)
            baris = {
                "timestamp": waktu_sekarang,
                "nama_pos": pos["nama_pos"],
                "lat": pos["lat"],
                "lon": pos["lon"],
                "elevasi_mdpl": pos["elevasi_mdpl"],
                "suhu_celsius": data["main"]["temp"],
                "kelembaban_persen": data["main"]["humidity"],
                "kecepatan_angin_ms": data["wind"]["speed"],
                "kondisi_cuaca": data["weather"][0]["description"]
            }
            baris_data_baru.append(baris)
            print(f"  -> Sukses mengambil data {pos['nama_pos']} ({baris['suhu_celsius']}°C)")
        else:
            print(f"  -> Gagal mengambil data {pos['nama_pos']}. Error: {response.status_code}")
            
    return baris_data_baru

# ==========================================
# 3. LOAD (Menyimpan ke CSV sebagai Dataset)
# ==========================================
def simpan_ke_csv(data_baru):
    if not data_baru:
        return
        
    # Tentukan nama kolom (header)
    kolom = ["timestamp", "nama_pos", "lat", "lon", "elevasi_mdpl", 
             "suhu_celsius", "kelembaban_persen", "kecepatan_angin_ms", "kondisi_cuaca"]
    
    # Cek apakah file CSV sudah ada, jika belum buat header-nya
    file_sudah_ada = os.path.isfile(NAMA_FILE_CSV)
    
    # Mode 'a' (append) berarti menambahkan data di baris paling bawah, bukan menimpa file
    with open(NAMA_FILE_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=kolom)
        
        if not file_sudah_ada:
            writer.writeheader()  # Tulis nama kolom untuk pertama kali
            
        for baris in data_baru:
            writer.writerow(baris)
            
    print(f"Berhasil menyimpan {len(data_baru)} baris baru ke {NAMA_FILE_CSV}\n")

# Eksekusi Utama
if __name__ == "__main__":
    hasil_ekstraksi = ambil_data_cuaca()
    simpan_ke_csv(hasil_ekstraksi)
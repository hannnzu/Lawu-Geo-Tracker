import requests
import json
import csv

# Menggunakan Open-Meteo Historical Weather API karena gratis dan tidak membutuhkan API Key
# untuk mengambil data historis secara spesifik selama setahun penuh.
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Titik koordinat untuk sebuah kecamatan di Kota Madiun (Contoh: Kecamatan Taman)
lokasi = {
    "nama": "Kecamatan Taman, Kota Madiun",
    "lat": -7.6369,
    "lon": 111.5361
}

# Parameter rentang waktu (Awal Januari hingga Akhir Desember)
# Anda bisa mengubah tahun pada variabel berikut sesuai kebutuhan (misal 2024 atau 2025)
tanggal_mulai = "2025-01-01"
tanggal_selesai = "2025-12-31"

print(f"Memulai pengambilan data cuaca historis untuk {lokasi['nama']}...")
print(f"Rentang waktu: {tanggal_mulai} s/d {tanggal_selesai}\n")

parameter = {
    "latitude": lokasi["lat"],
    "longitude": lokasi["lon"],
    "start_date": tanggal_mulai,
    "end_date": tanggal_selesai,
    "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"],
    "timezone": "Asia/Jakarta"
}

# Mengirim HTTP GET Request
response = requests.get(BASE_URL, params=parameter)

# Mengecek apakah koneksi berhasil (Status Code 200 = Sukses)
if response.status_code == 200:
    data_cuaca = response.json()
    print("Berhasil! Data cuaca diterima.\n")
    
    # Memproses data harian
    daily_data = data_cuaca['daily']
    tanggal_list = daily_data['time']
    suhu_max_list = daily_data['temperature_2m_max']
    suhu_min_list = daily_data['temperature_2m_min']
    suhu_rata_list = daily_data['temperature_2m_mean']
    curah_hujan_list = daily_data['precipitation_sum']
    angin_max_list = daily_data['wind_speed_10m_max']
    
    # Menyimpan data ke dalam file CSV untuk analisis lebih lanjut
    nama_file_csv = "data_cuaca_taman_2025.csv"
    print(f"Menyimpan data ke file: {nama_file_csv} ...")
    
    with open(nama_file_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Menulis header
        writer.writerow(["Tanggal", "Suhu Maks (C)", "Suhu Min (C)", "Suhu Rata-rata (C)", "Curah Hujan (mm)", "Kecepatan Angin Maks (km/h)"])
        
        # Menulis baris data
        for i in range(len(tanggal_list)):
            writer.writerow([
                tanggal_list[i],
                suhu_max_list[i],
                suhu_min_list[i],
                suhu_rata_list[i],
                curah_hujan_list[i],
                angin_max_list[i]
            ])
            
    print(f"Pengambilan dan penyimpanan data selesai! File dapat dilihat pada {nama_file_csv}")
    
else:
    print(f"Gagal mengambil data! Error Code: {response.status_code}")
    print(response.text)

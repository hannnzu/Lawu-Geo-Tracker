# 🌋 Lawu Geo Tracker - Data Engineering & ML Pipeline

Sebuah proyek Data Engineering *end-to-end* yang merancang pipeline ETL (*Extract, Transform, Load*) untuk memproses data geospasial, cuaca historis, dan titik api (kebakaran hutan) di Gunung Lawu. Data yang telah diolah kemudian digunakan untuk melatih model *Machine Learning* yang dapat memprediksi Tingkat Bahaya (*Danger Level*) di jalur pendakian.

## 🏗️ Arsitektur Pipeline

Proyek ini menggunakan pendekatan arsitektur **3 Tahap Utama ETL** untuk mempermudah *maintenance* dan eksekusi:

1. **Extract / Ingestion (`pipeline_1_ingestion.py`)**
   Menarik data dari berbagai sumber eksternal:
   * **NASA FIRMS API**: Data historis titik api/kebakaran hutan.
   * **Open-Meteo API**: Data cuaca historis per jam (Suhu, Angin, Curah Hujan).
   * **OpenStreetMap (Overpass API)**: Data spasial jalur pendakian.

2. **Transform (`pipeline_2_transformation.py`)**
   * Membersihkan data mentah (*Data Cleaning*).
   * Melakukan integrasi spatiotemporal (menghitung jarak antara pos pendakian dengan titik api).
   * Memproses data GPX (menghitung elevasi, kemiringan, jarak).
   * Mengkalkulasi metrik dan melabeli target variabel `Danger_Level` ke dalam CSV lokal.

3. **Load (`pipeline_3_loading.py`)**
   * Membangun skema database PostgreSQL secara terprogram.
   * Memuat data dimensi (Pos, Jalur) dan data fakta (Cuaca Terintegrasi) ke dalam *cloud database* Aiven.
   * Melakukan verifikasi integritas data (*Server-side validation*).

## 🤖 Pemodelan Machine Learning
Setelah data dimuat, model **Random Forest Classifier** dilatih untuk memprediksi *Danger Level*.
* **Data Splitting**: Pengacakan otomatis dengan proporsi **80% Data Latih (Training)** dan **20% Data Uji (Testing)** menggunakan Scikit-Learn.
* Model disimpan dalam format `.joblib` untuk keperluan *deployment* pada *dashboard* atau aplikasi.

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman**: Python 3.9+
* **Data Processing**: Pandas, GeoPandas, Scikit-Learn
* **Database**: PostgreSQL (Hosted via Aiven Cloud)
* **Database Connector**: SQLAlchemy, Psycopg2
* **Penyimpanan Spatial**: GeoJSON, GPX

## 📁 Struktur Direktori
```text
📦 Lawu-Geo-Tracker
 ┣ 📂 DATA
 ┃ ┣ 📂 raw          # Data mentah (Ingestion)
 ┃ ┣ 📂 processed    # Data setengah matang
 ┃ ┗ 📂 curated      # Data bersih siap ML (Transform)
 ┣ 📂 models         # File hasil training ML (.joblib)
 ┣ 📂 pipelines      # Script utama ETL (3 Tahap)
 ┣ 📂 scripts        # Script utilitas (termasuk ML Training)
 ┣ 📂 src            # Modul fungsi (ingestion, transformation, loading)
 ┣ 📜 .env.example   # Template environment variables
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
"""
train_model.py
--------------------------------
Script untuk melatih ulang model Machine Learning (Random Forest)
untuk memprediksi Danger Level di jalur pendakian Gunung Lawu.

Data Splitting: 80% Training, 20% Testing (Randomized)
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def run_training():
    print("=" * 60)
    print("MEMULAI TRAINING MODEL ML (80% Train, 20% Test)")
    print("=" * 60)

    # Sesuaikan dengan path dataset hasil pipeline 2 (Transformation)
    # Ubah nama tahun sesuai yang kamu pakai, misal 2021_2025
    data_path = Path("../DATA/curated/dataset_integrated_lawu_2021_2025.csv")
    model_output_path = Path("../models/rf_danger_level_model.joblib")

    if not data_path.exists():
        print(f"[!] Dataset tidak ditemukan di {data_path}")
        print("    Pastikan Pipeline 2 sudah dijalankan!")
        return

    # 1. Memuat Dataset
    print("[1/5] Memuat dataset terintegrasi...")
    df = pd.read_csv(data_path)
    print(f"      Total data awal: {len(df):,} baris.")

    # Pastikan kolom target ada
    target_col = 'Danger_Level'
    if target_col not in df.columns:
        print(f"[!] Kolom '{target_col}' tidak ditemukan! Pastikan ejaan huruf besar/kecilnya benar.")
        return

    # 2. Pre-processing (Hapus kolom yang tidak relevan untuk fitur ML)
    # Sesuaikan list 'cols_to_drop' ini dengan kolom di CSV kamu yang BUKAN angka/fitur metrik
    cols_to_drop = ['timestamp', 'nama_pos', 'waktu', 'tanggal', 'lokasi', 'status_kebakaran_sekitar']
    X = df.drop(columns=[target_col] + [c for c in cols_to_drop if c in df.columns])
    
    # Isi nilai kosong (NaN) dengan angka 0 atau rata-rata jika ada
    X = X.fillna(0) 
    y = df[target_col]

    # 3. Data Splitting (80% Train, 20% Test) secara Random
    print("\n[2/5] Membagi data secara acak (80% Training, 20% Testing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42 # Agar hasil random konsisten
    )

    print(f"      -> Data Training : {len(X_train):,} baris (80%)")
    print(f"      -> Data Testing  : {len(X_test):,} baris (20%)")

    # 4. Model Training
    print("\n[3/5] Memulai proses training model Random Forest Classifier...")
    # n_jobs=-1 agar menggunakan semua core CPU komputer/laptop
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Model Evaluation
    print("\n[4/5] Mengevaluasi akurasi model pada data testing...")
    y_pred = model.predict(X_test)
    
    print("-" * 40)
    print("HASIL EVALUASI MODEL:")
    print("Akurasi Keseluruhan : {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nDetail Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    print("-" * 40)

    # 6. Menyimpan Model
    print("\n[5/5] Menyimpan model...")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"      Model berhasil disimpan di: {model_output_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_training()
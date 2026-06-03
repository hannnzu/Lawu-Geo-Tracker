import pandas as pd
import numpy as np
import os
import time
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("Memuat dataset historis... (mohon tunggu sebentar)")
    file_path = os.path.join('DATA', 'curated', 'dataset_integrated_lawu_2021_2025.csv')
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File {file_path} tidak ditemukan.")
        return

    print("=== DATA PREPROCESSING & FEATURE ENGINEERING ===")
    
    # 1. Konversi Timestamp dan Ekstraksi Fitur Waktu Siklis (Cyclical Time Features)
    print("Mengekstrak fitur temporal siklis (sin/cos)...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour
    df['month'] = df['Timestamp'].dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 2. Pemisahan Data Secara Temporal (Time-Series Splitting)
    # Train: 2021-2024, Test: 2025
    print("Membagi data secara temporal: Train (2021-2024), Test (2025)...")
    train_df = df[df['Timestamp'].dt.year <= 2024]
    test_df = df[df['Timestamp'].dt.year == 2025]
    
    print(f"  Jumlah baris training : {train_df.shape[0]:,}")
    print(f"  Jumlah baris testing  : {test_df.shape[0]:,}")

    # Kolom yang akan dibuang untuk pemodelan (non-fitur atau 100% null)
    cols_to_drop = ['Timestamp', 'Nama Pos', 'Jalur', 'Jarak Pandang (m)', 'Danger_Level', 'hour', 'month']
    
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df['Danger_Level']
    
    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df['Danger_Level']

    print("\nFitur yang digunakan untuk training:")
    for col in X_train.columns:
        print(f" - {col}")

    print("\n=== LANGKAH 3: MELATIH MODEL (TRAINING) ===")
    print("Algoritma: Random Forest Classifier dengan Max Depth Regularization")
    print("Mulai melatih mesin...")
    print("[WAIT] Mohon tunggu sekitar 1-3 menit. Proses ini membutuhkan kinerja CPU yang tinggi...\n")
    
    start_time = time.time()

    # Inisialisasi Model AI dengan batasan kedalaman pohon (mencegah overfit)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,             # Membatasi overfit agar model lebih generalis
        random_state=42, 
        n_jobs=-1,                # Menggunakan seluruh core CPU
        class_weight='balanced'   # Penyeimbang data akibat class imbalance
    )

    # Memulai proses belajar (FIT)
    model.fit(X_train, y_train)

    waktu_training = time.time() - start_time
    print(f"[OK] Training Selesai! Waktu eksekusi: {waktu_training:.2f} detik.\n")

    print("=== LANGKAH 4: EVALUASI MODEL ===")
    print("Menguji model pada data Test Year 2025...")
    
    # Meminta model menebak jawaban (Prediksi)
    y_pred = model.predict(X_test)

    # Menilai hasil
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[ACC] Akurasi Keseluruhan Model: {acc * 100:.2f}%\n")

    print("[REPORT] Laporan Detail (Classification Report):")
    print(classification_report(y_test, y_pred))

    print("=== LANGKAH 5: MENYIMPAN MODEL (EXPORT) ===")
    os.makedirs('models', exist_ok=True)
    
    model_path = os.path.join('models', 'rf_danger_level_model.joblib')
    joblib.dump(model, model_path)
    
    print(f"[OK] Model AI berhasil disimpan di: {model_path}")
    print("Model ini sekarang siap dihubungkan ke chatbot streamlit!")

if __name__ == "__main__":
    main()
import pandas as pd
import os
import time
import joblib 
from sklearn.model_selection import train_test_split
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

    print("=== DATA PREPROCESSING ===")
    # Membuang kolom yang tidak diperlukan
    cols_to_drop = ['Timestamp', 'Nama Pos', 'Jalur', 'Jarak Pandang (m)']
    df_clean = df.drop(columns=cols_to_drop)

    # Memisahkan Fitur (X) dan Target (Y)
    X = df_clean.drop(columns=['Danger_Level'])
    y = df_clean['Danger_Level']

    # Membagi 80% data untuk belajar, 20% data untuk ujian (testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=== LANGKAH 3: MELATIH MODEL (TRAINING) ===")
    print("Algoritma: Random Forest Classifier")
    print(f"Mulai melatih mesin dengan {X_train.shape[0]:,} baris data...")
    print("⏳ Mohon tunggu sekitar 1-3 menit. Proses ini membutuhkan kinerja CPU yang tinggi...\n")
    
    start_time = time.time()

    # Inisialisasi Model AI
    model = RandomForestClassifier(
        n_estimators=50,       # Jumlah "pohon" keputusan
        random_state=42, 
        n_jobs=-1,             # Gunakan seluruh core CPU
        class_weight='balanced'# Penyeimbang data agar kelas 3 (Dilarang) tetap diperhatikan
    )

    # Memulai proses belajar (FIT)
    model.fit(X_train, y_train)

    waktu_training = time.time() - start_time
    print(f"✅ Training Selesai! Waktu eksekusi: {waktu_training:.2f} detik.\n")

    print("=== LANGKAH 4: EVALUASI MODEL ===")
    print("Menguji kepintaran model pada data Testing (yang belum pernah dilihatnya)...")
    
    # Meminta model menebak jawaban (Prediksi)
    y_pred = model.predict(X_test)

    # Menilai hasil tebakan vs kunci jawaban asli
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Akurasi Keseluruhan Model: {acc * 100:.2f}%\n")

    print("📊 Laporan Detail (Classification Report):")
    print(classification_report(y_test, y_pred))

    print("=== LANGKAH 5: MENYIMPAN MODEL (EXPORT) ===")
    
    # Membuat folder 'models' jika belum ada di dalam folder utama
    os.makedirs('models', exist_ok=True)
    
    # Menyimpan model ke dalam folder models
    model_path = os.path.join('models', 'rf_danger_level_model.joblib')
    joblib.dump(model, model_path)
    
    print(f"✅ Model AI berhasil disimpan di: {model_path}")
    print("Model ini sekarang siap dihubungkan ke sistem backend atau API!")

if __name__ == "__main__":
    main()
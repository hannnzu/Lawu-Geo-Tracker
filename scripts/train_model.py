"""
train_model.py
--------------------------------
Script untuk melatih ulang model Machine Learning (Random Forest)
untuk memprediksi Danger Level di jalur pendakian Gunung Lawu.

Data Splitting: 80% Training, 20% Testing (Randomized)
"""

import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

from src.utils.config import Config

def run_training():
    print("=" * 60)
    print("MEMULAI TRAINING MODEL ML (80% Train, 20% Test)")
    print("=" * 60)

    t_w_start = Config.WEATHER_HISTORICAL_START[:4]
    t_w_end = Config.WEATHER_HISTORICAL_END[:4]
    data_path = Config.DATA_CURATED_DIR / f"dataset_integrated_lawu_{t_w_start}_{t_w_end}.csv"
    
    model_output_path = Path("models/rf_danger_level_model.joblib")
    metrics_output_path = Path("models/model_metrics.json") # File baru untuk metrik

    if not data_path.exists():
        print(f"[!] Dataset tidak ditemukan di {data_path}")
        return

    print("[1/5] Memuat dataset terintegrasi...")
    df = pd.read_csv(data_path)
    print(f"      Total data awal: {len(df):,} baris.")

    target_col = 'Danger_Level'
    if target_col not in df.columns:
        print(f"[!] Kolom '{target_col}' tidak ditemukan!")
        return

    print("[2/5] Melakukan pre-processing (Hapus kolom teks & data leakage)...")
    cols_to_drop = [target_col, 'status_kebakaran_sekitar']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    X = X.select_dtypes(include=['number'])
    X = X.fillna(0) 
    y = df[target_col]

    print("\n[3/5] Membagi data secara acak (80% Training, 20% Testing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      -> Data Training : {len(X_train):,} baris (80%)")
    print(f"      -> Data Testing  : {len(X_test):,} baris (20%)")

    print("\n[4/5] Memulai proses training model Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n[5/5] Mengevaluasi akurasi model pada data testing...")
    y_pred = model.predict(X_test)
    
    # Kalkulasi Metrik
    acc = accuracy_score(y_test, y_pred)
    clf_rep_dict = classification_report(y_test, y_pred, output_dict=True)
    clf_rep_str = classification_report(y_test, y_pred)
    
    print("-" * 40)
    print("HASIL EVALUASI MODEL:")
    print("Akurasi Keseluruhan : {:.2f}%".format(acc * 100))
    print("\nDetail Laporan Klasifikasi:")
    print(clf_rep_str)
    print("-" * 40)

    # Simpan Model
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    
    # PERUBAHAN: Simpan Metrik ke File JSON
    metrics_data = {
        "accuracy": acc,
        "classification_report": clf_rep_dict,
        "total_data": len(df),
        "train_data": len(X_train),
        "test_data": len(X_test)
    }
    with open(metrics_output_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"      Model berhasil disimpan di: {model_output_path.absolute()}")
    print(f"      Metrik berhasil disimpan di: {metrics_output_path.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    run_training()
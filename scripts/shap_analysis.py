"""
scripts/shap_analysis.py
--------------------------------
Script untuk melakukan audit interpretabilitas model Machine Learning (LightGBM)
menggunakan SHAP (SHapley Additive exPlanations).

Tahapan:
1. Memuat dataset terintegrasi Lawu.
2. Feature engineering (Cyclic & Interaction) dan pembuangan fitur leakage.
3. Melakukan stratified sampling pada data pengujian (Test Set 2025) sebanyak 2000 sampel.
4. Memuat model LightGBM terlatih.
5. Menghitung SHAP values menggunakan TreeExplainer.
6. Menyimpan visualisasi SHAP:
   - Summary Plot Beeswarm untuk Level 3 (Bahaya/Dilarang).
   - Summary Plot Bar (All Classes) untuk perbandingan kontribusi global.
   - Waterfall Plot lokal untuk mendiagnosis satu sampel Level 3.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap

from src.utils.config import Config

# Set backend matplotlib ke Non-Interactive (Agg) agar tidak crash jika dijalankan tanpa display
import matplotlib
matplotlib.use('Agg')

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan encoding siklis (sin/cos) untuk fitur temporal."""
    df = df.copy()
    datetime_col = None
    for col in ['Timestamp', 'Datetime', 'datetime', 'time', 'date']:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col:
        dt = pd.to_datetime(df[datetime_col], errors='coerce')
        df['bulan_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
        df['bulan_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
        df['jam_sin']   = np.sin(2 * np.pi * dt.dt.hour / 24)
        df['jam_cos']   = np.cos(2 * np.pi * dt.dt.hour / 24)
        df['doy_sin']   = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
        df['doy_cos']   = np.cos(2 * np.pi * dt.dt.dayofyear / 365)
        df['_tahun']    = dt.dt.year
    else:
        df['_tahun'] = 0

    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur interaksi spasial-meteorologi."""
    df = df.copy()
    if 'Suhu Terasa (C)' in df.columns and 'Angin Kencang (km/h)' in df.columns:
        df['compound_cold_wind'] = df['Suhu Terasa (C)'] * df['Angin Kencang (km/h)']
    if 'FRP_Terdekat_MW' in df.columns and 'Jarak_Titik_Api_Terdekat_KM' in df.columns:
        df['fire_proximity_index'] = (
            df['FRP_Terdekat_MW'] / (df['Jarak_Titik_Api_Terdekat_KM'] + 0.1)
        )
    return df

def run_shap_analysis():
    print("=" * 65)
    print("ANALISIS SHAP EXPLAINABILITY & AUDIT MODEL")
    print("=" * 65)

    # 1. Cari dataset terintegrasi
    t_w_start = Config.WEATHER_HISTORICAL_START[:4]
    t_w_end   = Config.WEATHER_HISTORICAL_END[:4]
    filename  = f"dataset_integrated_lawu_{t_w_start}_{t_w_end}.csv"

    possible_paths = [
        Config.DATA_CURATED_DIR / filename,
        Config.ROOT_DIR / "DATA" / "curated" / filename,
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)
    if data_path is None:
        print("[!] Dataset tidak ditemukan.")
        return

    # Path model dan folder output
    model_path = Path("models/lgbm_danger_level_model.joblib")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[!] Model LightGBM tidak ditemukan di {model_path}. Jalankan training terlebih dahulu.")
        return

    # FASE KRITIS: Memuat Dataset dan Preprocessing
    # Kita menggunakan preprocessing yang identik dengan scripts/train_lgbm.py
    # untuk memastikan data input SHAP sesuai dengan fitur training model.
    print(f"\n[1/5] Memuat dataset: {data_path.name}")
    df = pd.read_csv(data_path)
    df = add_cyclic_features(df)
    df = add_interaction_features(df)

    target_col = 'Danger_Level'
    LEAKAGE_COLS  = ['status_kebakaran_sekitar', 'Status_Kebakaran_Sekitar']
    TEMPORAL_COLS = ['_tahun']
    EXCLUDE_COLS  = [target_col] + LEAKAGE_COLS + TEMPORAL_COLS

    # Filter data pengujian (Test Set tahun 2025)
    df_test = df[df['_tahun'] == 2025].reset_index(drop=True)
    if len(df_test) == 0:
        print("[!] Data pengujian (tahun 2025) kosong!")
        return

    # FASE KRITIS: Stratified Sampling untuk Kecepatan Komputasi SHAP
    # Menghitung SHAP values untuk 166.000+ baris test set sangat lambat.
    # Kita mengambil sampel terstratifikasi maksimal 500 baris per kelas
    # (total ~2000 baris) agar representatif namun tetap cepat dihitung.
    print("\n[2/5] Melakukan stratified sampling pada Test Set 2025...")
    samples_per_class = 500
    sampled_dfs = []
    for cl in sorted(df_test[target_col].unique()):
        df_cl = df_test[df_test[target_col] == cl]
        n_samples = min(len(df_cl), samples_per_class)
        sampled_dfs.append(df_cl.sample(n=n_samples, random_state=42))
    df_sample = pd.concat(sampled_dfs).reset_index(drop=True)

    # Pisahkan fitur dan target
    X_sample = df_sample.drop(columns=[c for c in EXCLUDE_COLS if c in df_sample.columns])
    X_sample = X_sample.select_dtypes(include=['number'])
    X_sample = X_sample.fillna(0)
    y_sample = df_sample[target_col]

    print(f"      Total sampel terpilih untuk SHAP: {len(X_sample)} baris")
    print(f"      Jumlah fitur: {X_sample.shape[1]} kolom")
    print("      Distribusi kelas sampel:")
    for lvl, cnt in y_sample.value_counts().sort_index().items():
        print(f"        Level {lvl}: {cnt} baris")

    # 3. Memuat Model
    print(f"\n[3/5] Memuat model LightGBM dari: {model_path}")
    model = joblib.load(model_path)

    # FASE KRITIS: Komputasi SHAP Values dengan TreeExplainer
    # TreeExplainer sangat efisien untuk model berbasis pohon (seperti LightGBM).
    # Untuk klasifikasi multi-kelas, SHAP akan menghasilkan list array numpy,
    # di mana setiap array merepresentasikan SHAP values untuk satu kelas tertentu.
    print("\n[4/5] Menghitung SHAP values menggunakan TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Pada shap >= 0.45.0 untuk LightGBM multiclass, shap_values bisa berupa list
    # berukuran num_classes (masing-masing array shape: [num_samples, num_features]),
    # atau numpy array dengan dimensi [num_samples, num_features, num_classes].
    # Kita pastikan formatnya agar tidak terjadi error saat diakses.
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Ubah ke format list of arrays [class][samples, features]
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        shap_values = shap_values_list

    # FASE KRITIS: Visualisasi SHAP dan Penyimpanan Plot
    # Kita akan menyimpan plot penting untuk kebutuhan audit keselamatan:
    # 1. Summary Plot (Beeswarm) untuk Level 3 (Bahaya/Dilarang) -> Menjelaskan pendorong utama larangan pendakian.
    # 2. Summary Plot (Bar) All Classes -> Perbandingan pengaruh fitur global di seluruh level bahaya.
    # 3. Waterfall Plot Lokal -> Menjelaskan salah satu contoh kasus nyata Level 3.
    print("\n[5/5] Menghasilkan dan menyimpan visualisasi SHAP...")

    # Plot 1: Summary Bar Plot (All Classes)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Global Feature Importance (All Classes)", fontsize=14, pad=15)
    plt.tight_layout()
    plot_all_path = output_dir / "shap_summary_all.png"
    plt.savefig(plot_all_path, dpi=150)
    plt.close()
    print(f"      [OK] Saved: {plot_all_path}")

    # Plot 2: Summary Beeswarm Plot untuk Level 3 (Bahaya/Dilarang)
    # Target index untuk Level 3 adalah 3
    if len(shap_values) > 3:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[3], X_sample, show=False)
        plt.title("SHAP Summary Plot for Danger Level 3 (Dilarang)", fontsize=14, pad=15)
        plt.tight_layout()
        plot_lvl3_path = output_dir / "shap_summary_level3.png"
        plt.savefig(plot_lvl3_path, dpi=150)
        plt.close()
        print(f"      [OK] Saved: {plot_lvl3_path}")
    else:
        print("      [!] Kelas Level 3 tidak tersedia dalam SHAP values.")

    # Plot 3: Waterfall Plot Lokal untuk Contoh Level 3
    # Temukan sampel yang diprediksi Level 3 dan benar-benar berlabel Level 3
    y_pred = model.predict(X_sample)
    idx_level3 = np.where((y_pred == 3) & (y_sample == 3))[0]

    if len(idx_level3) > 0 and len(shap_values) > 3:
        sample_idx = idx_level3[0] # Ambil sampel pertama
        
        # Buat objek Explanation untuk kelas 3
        # base_values dapat berupa array (per sampel) atau skalar.
        base_val = explainer.expected_value[3]
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
            base_val = base_val[sample_idx]
            
        exp = shap.Explanation(
            values=shap_values[3][sample_idx],
            base_values=base_val,
            data=X_sample.iloc[sample_idx].values,
            feature_names=X_sample.columns.tolist()
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(exp, show=False)
        plt.title(f"SHAP Local Explanation (Waterfall Plot) for Sample #{sample_idx} (Level 3)", fontsize=12, pad=15)
        plt.tight_layout()
        plot_waterfall_path = output_dir / "shap_waterfall_level3.png"
        plt.savefig(plot_waterfall_path, dpi=150)
        plt.close()
        print(f"      [OK] Saved: {plot_waterfall_path}")

        # Cetak detail sampel terpilih untuk validasi teks
        print(f"\n      Detail Sampel Terpilih (Indeks #{sample_idx}):")
        print(f"      +--------------------------------+-----------------+")
        print(f"      | Nama Fitur                     | Nilai Fitur     |")
        print(f"      +--------------------------------+-----------------+")
        for col_name in X_sample.columns:
            val_f = X_sample.iloc[sample_idx][col_name]
            print(f"      | {col_name:<30} | {val_f:<15.4f} |")
        print(f"      +--------------------------------+-----------------+")
    else:
        print("      [!] Tidak ada sampel yang memenuhi kriteria prediksi Level 3 untuk Waterfall plot.")

    print("\n[+] Analisis SHAP berhasil diselesaikan.")
    print("=" * 65)

if __name__ == "__main__":
    run_shap_analysis()

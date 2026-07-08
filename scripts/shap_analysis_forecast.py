"""
scripts/shap_analysis_forecast.py
--------------------------------
Script untuk melakukan audit interpretabilitas model Machine Learning Forecasting (LightGBM)
menggunakan SHAP (SHapley Additive exPlanations).

Menghasilkan plot global dan lokal untuk model lgbm_forecast_t3h_v2_model.joblib.
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

# Set backend matplotlib ke Non-Interactive (Agg)
import matplotlib
matplotlib.use('Agg')

def run_shap_analysis_forecast():
    print("=" * 65)
    print("ANALISIS SHAP EXPLAINABILITY & AUDIT MODEL FORECASTING")
    print("=" * 65)

    HORIZON_HOURS = 3
    dataset_filename = f"dataset_forecast_lawu_t{HORIZON_HOURS}h_2021_2025.csv"
    possible = [
        Path(f"data/curated/{dataset_filename}"),
        Path(f"DATA/curated/{dataset_filename}"),
    ]
    data_path = next((p for p in possible if p.exists()), None)
    if not data_path:
        print(f"[!] Dataset forecasting tidak ditemukan: {dataset_filename}")
        return

    model_path = Path(f"models/lgbm_forecast_t{HORIZON_HOURS}h_v2_model.joblib")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[!] Model tidak ditemukan: {model_path}")
        return

    print(f"\n[1/5] Memuat dataset forecasting: {data_path.name}")
    df = pd.read_csv(data_path)

    target_col = f'Danger_Level_t{HORIZON_HOURS}h'
    tahun_series = df['_tahun'].copy() if '_tahun' in df.columns else pd.Series([0] * len(df))

    EXCLUDE_COLS = [target_col, 'Danger_Level', 'status_kebakaran_sekitar',
                    'Status_Kebakaran_Sekitar', '_tahun']
    
    # Filter data pengujian (Test Set tahun 2025)
    df_test = df[tahun_series == 2025].reset_index(drop=True)
    if len(df_test) == 0:
        print("[!] Data pengujian (tahun 2025) kosong!")
        return

    # Stratified Sampling: Ambil sampel terstratifikasi maksimal 300 baris per kelas
    print("\n[2/5] Melakukan stratified sampling pada Test Set 2025...")
    samples_per_class = 300
    sampled_dfs = []
    for cl in sorted(df_test[target_col].unique()):
        df_cl = df_test[df_test[target_col] == cl]
        n_samples = min(len(df_cl), samples_per_class)
        sampled_dfs.append(df_cl.sample(n=n_samples, random_state=42))
    df_sample = pd.concat(sampled_dfs).reset_index(drop=True)

    # Pisahkan fitur dan target
    X_sample = df_sample.drop(columns=[c for c in EXCLUDE_COLS if c in df_sample.columns])
    X_sample = X_sample.select_dtypes(include=['number']).fillna(0)
    y_sample = df_sample[target_col].astype(int)

    print(f"      Total sampel terpilih untuk SHAP: {len(X_sample)} baris")
    print(f"      Jumlah fitur: {X_sample.shape[1]} kolom")
    print("      Distribusi kelas sampel:")
    for lvl, cnt in y_sample.value_counts().sort_index().items():
        print(f"        Level {lvl}: {cnt} baris")

    # 3. Memuat Model
    print(f"\n[3/5] Memuat model LightGBM dari: {model_path}")
    model = joblib.load(model_path)

    # 4. Hitung SHAP values
    print("\n[4/5] Menghitung SHAP values menggunakan TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        shap_values = shap_values_list

    # 5. Visualisasi SHAP
    print("\n[5/5] Menghasilkan dan menyimpan visualisasi SHAP...")

    # Plot 1: Summary Bar Plot (All Classes)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Global Feature Importance (All Classes) - Forecasting Model", fontsize=12, pad=15)
    plt.tight_layout()
    plot_all_path = output_dir / "shap_summary_all_forecast.png"
    plt.savefig(plot_all_path, dpi=150)
    plt.close()
    print(f"      [OK] Saved: {plot_all_path}")

    # Plot 2: Summary Beeswarm Plot untuk Level 3 (Dilarang)
    if len(shap_values) > 3:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[3], X_sample, show=False)
        plt.title("SHAP Summary Plot for Danger Level 3 (Dilarang) - Forecasting Model", fontsize=12, pad=15)
        plt.tight_layout()
        plot_lvl3_path = output_dir / "shap_summary_level3_forecast.png"
        plt.savefig(plot_lvl3_path, dpi=150)
        plt.close()
        print(f"      [OK] Saved: {plot_lvl3_path}")
    else:
        print("      [!] Kelas Level 3 tidak tersedia dalam SHAP values.")

    # Plot 3: Waterfall Plot Lokal untuk Contoh Level 3
    # Temukan sampel yang diprediksi Level 3 dengan safety threshold 0.10
    proba = model.predict_proba(X_sample)
    y_pred = np.argmax(proba, axis=1)
    y_pred[proba[:, 3] >= 0.10] = 3
    
    idx_level3 = np.where((y_pred == 3) & (y_sample == 3))[0]

    if len(idx_level3) > 0 and len(shap_values) > 3:
        sample_idx = idx_level3[0] # Ambil sampel pertama
        base_val = explainer.expected_value[3]
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
            base_val = base_val[sample_idx]
            
        exp = shap.Explanation(
            values=shap_values[3][sample_idx],
            base_values=base_val,
            data=X_sample.iloc[sample_idx].values,
            feature_names=X_sample.columns.tolist()
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(exp, show=False)
        plt.title(f"SHAP Waterfall Plot for Sample #{sample_idx} (Level 3) - Forecasting Model", fontsize=10, pad=15)
        plt.tight_layout()
        plot_waterfall_path = output_dir / "shap_waterfall_level3_forecast.png"
        plt.savefig(plot_waterfall_path, dpi=150)
        plt.close()
        print(f"      [OK] Saved: {plot_waterfall_path}")

        # Tampilkan top 5 kontributor positif/negatif
        print(f"\n      Kontribusi Fitur untuk Sampel #{sample_idx}:")
        sorted_indices = np.argsort(shap_values[3][sample_idx])[::-1]
        for rank, idx in enumerate(sorted_indices[:10]):
            feat_name = X_sample.columns[idx]
            val_f = X_sample.iloc[sample_idx][feat_name]
            shap_val = shap_values[3][sample_idx][idx]
            print(f"        Rank {rank+1}: {feat_name:<35} = {val_f:<12.4f} (SHAP: {shap_val:+.4f})")
    else:
        print("      [!] Tidak ada sampel yang memenuhi kriteria prediksi Level 3 untuk Waterfall plot.")

    print("\n[+] Analisis SHAP berhasil diselesaikan.")
    print("=" * 65)

if __name__ == "__main__":
    run_shap_analysis_forecast()

"""
scripts/shap_analysis_multi_horizon.py
--------------------------------------
Sprint 4 — Tahap 3: Analisis SHAP Explainability untuk Multi-Horizon Models.

Menghasilkan Beeswarm plot dan Bar plot global untuk masing-masing model
forecasting (t+1h, t+3h, t+6h).
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


def run_shap_analysis_multi_horizon():
    print("=" * 65)
    print("ANALISIS SHAP EXPLAINABILITY UNTUK MULTI-HORIZON MODELS")
    print("=" * 65)

    t_w_start = Config.WEATHER_HISTORICAL_START[:4]
    t_w_end   = Config.WEATHER_HISTORICAL_END[:4]
    dataset_filename = f"dataset_forecast_lawu_multi_{t_w_start}_{t_w_end}.csv"

    possible_paths = [
        Config.DATA_CURATED_DIR / dataset_filename,
        Config.ROOT_DIR / "DATA" / "curated" / dataset_filename,
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)
    if data_path is None:
        print(f"[!] Dataset tidak ditemukan: {dataset_filename}")
        return

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Memuat dataset multi-horizon: {data_path.name}")
    df = pd.read_csv(data_path)

    TARGET_COLS = ['Danger_Level_t1h', 'Danger_Level_t3h', 'Danger_Level_t6h']
    EXCLUDE_COLS = TARGET_COLS + ['Danger_Level', 'status_kebakaran_sekitar', 'Status_Kebakaran_Sekitar', '_tahun']

    # Filter data pengujian (Test Set tahun 2025)
    df_test = df[df['_tahun'] == 2025].reset_index(drop=True)
    if len(df_test) == 0:
        print("[!] Data pengujian (tahun 2025) kosong!")
        return

    # Hitung SHAP untuk masing-masing model
    for target in TARGET_COLS:
        horizon_name = target.split('_')[-1] # t1h, t3h, t6h
        model_path = Path(f"models/lgbm_forecast_{horizon_name}_model.joblib")
        if not model_path.exists():
            print(f"[!] Model tidak ditemukan: {model_path}")
            continue

        print(f"\n" + "-" * 50)
        print(f"SHAP AUDIT UNTUK HORIZON: {horizon_name.upper()}")
        print("-" * 50)

        # Stratified sampling
        print("      Melakukan stratified sampling...")
        samples_per_class = 300
        sampled_dfs = []
        for cl in sorted(df_test[target].unique()):
            df_cl = df_test[df_test[target] == cl]
            n_samples = min(len(df_cl), samples_per_class)
            sampled_dfs.append(df_cl.sample(n=n_samples, random_state=42))
        df_sample = pd.concat(sampled_dfs).reset_index(drop=True)

        X_sample = df_sample.drop(columns=[c for c in EXCLUDE_COLS if c in df_sample.columns])
        X_sample = X_sample.select_dtypes(include=['number']).fillna(0)
        y_sample = df_sample[target].astype(int)

        print(f"      Total sampel terpilih: {len(X_sample)}")

        # Memuat model
        model = joblib.load(model_path)

        # Hitung SHAP
        print("      Menghitung SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

        # Plot 1: Summary Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Global Feature Importance (All Classes) - Horizon {horizon_name.upper()}", fontsize=12, pad=15)
        plt.tight_layout()
        plot_all = output_dir / f"shap_summary_all_{horizon_name}.png"
        plt.savefig(plot_all, dpi=150)
        plt.close()
        print(f"      [OK] Saved global plot: {plot_all}")

        # Plot 2: Summary Beeswarm Level 3
        if len(shap_values) > 3:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[3], X_sample, show=False)
            plt.title(f"SHAP Summary Plot for Danger Level 3 (Dilarang) - Horizon {horizon_name.upper()}", fontsize=12, pad=15)
            plt.tight_layout()
            plot_lvl3 = output_dir / f"shap_summary_level3_{horizon_name}.png"
            plt.savefig(plot_lvl3, dpi=150)
            plt.close()
            print(f"      [OK] Saved Level 3 plot: {plot_lvl3}")

            # Dapatkan 5 fitur teratas berdasarkan pengaruh absolut rata-rata untuk Level 3
            mean_abs_shap = np.abs(shap_values[3]).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:5]
            print("      Top 5 Fitur Paling Berpengaruh untuk Level 3:")
            for rank, idx in enumerate(top_indices):
                feat_name = X_sample.columns[idx]
                score = mean_abs_shap[idx]
                print(f"        Rank {rank+1}: {feat_name:<35} (Mean Abs SHAP: {score:.4f})")
        else:
            print("      [!] Kelas Level 3 tidak tersedia.")

    print("\n[+] Analisis SHAP untuk semua horizon selesai.")
    print("=" * 65)


if __name__ == "__main__":
    run_shap_analysis_multi_horizon()

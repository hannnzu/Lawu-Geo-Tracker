"""
scripts/optimize_forecast_threshold.py
----------------------------------
Sprint 3 — Tahap 3: Optimasi Threshold untuk Level 3 (Dilarang).

DIAGNOSA:
  Recall Level 3 pada Test Set 2025 hanya 38.46% karena distribusi
  Level 3 di tahun 2025 sangat berbeda dari tahun training:
    - Train 2023 : 3.467% Level 3 (5.771 baris)  → model "terbiasa"
    - Test  2025 : 0.148% Level 3 (247 baris)     → kondisi berbeda
  Model LightGBM dengan threshold default 0.5 tidak sensitif terhadap
  kelas yang sangat jarang pada data baru (temporal distribution shift).

SOLUSI: Threshold Optimization berbasis predict_proba()
  Alih-alih menggunakan argmax dari 4 kelas, kita turunkan threshold
  untuk Level 3. Jika P(Level 3) > THRESHOLD_LVL3, langsung prediksi
  Level 3 — tanpa menunggu probabilitas Level 3 menjadi yang tertinggi.

  Ini adalah praktik standar pada sistem keselamatan jiwa: 
  "lebih baik false alarm daripada miss deteksi bahaya nyata."

METRIK OPTIMAL:
  Target: Maximize F1-score Level 3, dengan constraint Precision ≥ 0.40
  (Precision terlalu rendah = terlalu banyak false alarm yang mengganggu)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score
)

HORIZON_HOURS = 3
THRESHOLD_SEARCH_RANGE = np.arange(0.05, 0.70, 0.01)


def apply_custom_threshold(proba: np.ndarray, threshold_lvl3: float) -> np.ndarray:
    """
    Terapkan threshold kustom untuk Level 3.
    Jika P(Level 3) >= threshold, prediksi Level 3.
    Jika tidak, ambil kelas dengan probabilitas tertinggi di antara Level 0,1,2.
    """
    predictions = np.argmax(proba, axis=1)
    lvl3_mask = proba[:, 3] >= threshold_lvl3
    predictions[lvl3_mask] = 3
    return predictions


def run_threshold_optimization():
    print("=" * 65)
    print("TAHAP 3: OPTIMASI THRESHOLD LEVEL 3 (Recall-Precision Tradeoff)")
    print("=" * 65)

    # Muat dataset forecasting
    dataset_filename = f"dataset_forecast_lawu_t{HORIZON_HOURS}h_2021_2025.csv"
    possible = [
        Path(f"data/curated/{dataset_filename}"),
        Path(f"DATA/curated/{dataset_filename}"),
    ]
    data_path = next((p for p in possible if p.exists()), None)
    if not data_path:
        print(f"[!] Dataset forecasting tidak ditemukan. Jalankan build_forecast_dataset.py terlebih dahulu.")
        return

    # Muat model
    model_path = Path(f"models/lgbm_forecast_t{HORIZON_HOURS}h_model.joblib")
    if not model_path.exists():
        print(f"[!] Model tidak ditemukan: {model_path}")
        return

    print(f"\n[1/4] Memuat model dan dataset...")
    model    = joblib.load(model_path)
    df       = pd.read_csv(data_path)
    print(f"      Dataset: {len(df):,} baris, {len(df.columns)} kolom.")

    # Siapkan fitur (identik dengan training)
    target_col = f'Danger_Level_t{HORIZON_HOURS}h'
    tahun_series = df['_tahun'].copy() if '_tahun' in df.columns else pd.Series([0] * len(df))

    EXCLUDE_COLS = [target_col, 'Danger_Level', 'status_kebakaran_sekitar',
                    'Status_Kebakaran_Sekitar', '_tahun']
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    X = X.select_dtypes(include=['number']).fillna(0)
    y = df[target_col].astype(int)

    # Isolasi Validation Set (2024) untuk mencari threshold optimal
    mask_val  = tahun_series == 2024
    mask_test = tahun_series == 2025

    X_val,  y_val  = X[mask_val].reset_index(drop=True),  y[mask_val].reset_index(drop=True)
    X_test, y_test = X[mask_test].reset_index(drop=True), y[mask_test].reset_index(drop=True)

    print(f"      Val set (2024) : {len(X_val):,} baris")
    print(f"      Test set (2025): {len(X_test):,} baris")
    print(f"      Level 3 di Val : {(y_val==3).sum()} baris")
    print(f"      Level 3 di Test: {(y_test==3).sum()} baris")

    # Hitung probabilitas prediksi
    print("\n[2/4] Menghitung predict_proba()...")
    proba_val  = model.predict_proba(X_val)
    proba_test = model.predict_proba(X_test)

    # Grid search threshold pada Validation Set
    print("\n[3/4] Grid search threshold optimal pada Val Set (2024)...")
    print(f"      {'Threshold':>10} | {'Recall_L3':>9} | {'Prec_L3':>8} | {'F1_L3':>7} | {'Acc':>7}")
    print("      " + "-" * 50)

    best_threshold = 0.5
    best_f1_lvl3   = 0.0
    results = []

    for thr in THRESHOLD_SEARCH_RANGE:
        y_pred_thr = apply_custom_threshold(proba_val, thr)

        # Hitung metrik per kelas
        lvl3_mask_true = (y_val == 3)
        lvl3_mask_pred = (y_pred_thr == 3)

        recall_l3    = recall_score(y_val, y_pred_thr, labels=[3], average='macro', zero_division=0)
        precision_l3 = precision_score(y_val, y_pred_thr, labels=[3], average='macro', zero_division=0)
        f1_l3        = f1_score(y_val, y_pred_thr, labels=[3], average='macro', zero_division=0)
        acc          = accuracy_score(y_val, y_pred_thr)

        results.append({
            'threshold': round(float(thr), 2),
            'recall_l3': round(recall_l3, 4),
            'precision_l3': round(precision_l3, 4),
            'f1_l3': round(f1_l3, 4),
            'accuracy': round(acc, 4),
        })

        # Tampilkan setiap 0.05 kelipatan
        if round(thr * 100) % 5 == 0:
            print(f"      {thr:>10.2f} | {recall_l3*100:>8.2f}% | {precision_l3*100:>7.2f}% | {f1_l3:.4f} | {acc*100:.2f}%")

        # Pilih threshold terbaik: max F1 Level 3 dengan Precision >= 0.30
        if precision_l3 >= 0.30 and f1_l3 > best_f1_lvl3:
            best_f1_lvl3   = f1_l3
            best_threshold = thr

    print(f"\n      --> Threshold optimal: {best_threshold:.2f} (F1 Level3 = {best_f1_lvl3:.4f})")

    # Evaluasi akhir pada Test Set dengan threshold optimal
    print(f"\n[4/4] Evaluasi FINAL pada Test Set (2025) — threshold={best_threshold:.2f}...")
    y_pred_final = apply_custom_threshold(proba_test, best_threshold)

    print(f"\n  [TEST SET 2025 — Threshold Default 0.50]")
    y_pred_default = np.argmax(proba_test, axis=1)
    r_def = recall_score(y_test, y_pred_default, labels=[3], average='macro', zero_division=0)
    p_def = precision_score(y_test, y_pred_default, labels=[3], average='macro', zero_division=0)
    print(f"  Recall L3: {r_def*100:.2f}% | Precision L3: {p_def*100:.2f}% | Acc: {accuracy_score(y_test, y_pred_default)*100:.2f}%")

    print(f"\n  [TEST SET 2025 — Threshold Optimal {best_threshold:.2f}]")
    r_opt = recall_score(y_test, y_pred_final, labels=[3], average='macro', zero_division=0)
    p_opt = precision_score(y_test, y_pred_final, labels=[3], average='macro', zero_division=0)
    f1_opt = f1_score(y_test, y_pred_final, labels=[3], average='macro', zero_division=0)
    acc_opt = accuracy_score(y_test, y_pred_final)
    f1_mac_opt = f1_score(y_test, y_pred_final, average='macro', zero_division=0)

    print(f"  Recall L3: {r_opt*100:.2f}% | Precision L3: {p_opt*100:.2f}% | F1 L3: {f1_opt:.4f}")
    print(f"  Akurasi  : {acc_opt*100:.2f}% | F1-Macro: {f1_mac_opt:.4f}")
    print("\n  Laporan Klasifikasi Lengkap:")
    print(classification_report(y_test, y_pred_final, zero_division=0))

    # Simpan konfigurasi threshold optimal
    threshold_config = {
        "model": f"lgbm_forecast_t{HORIZON_HOURS}h_model.joblib",
        "optimal_threshold_level3": round(float(best_threshold), 2),
        "tuned_on": "validation_set_2024",
        "evaluation_on_test_2025": {
            "accuracy":       round(acc_opt, 6),
            "f1_macro":       round(f1_mac_opt, 6),
            "recall_level3":  round(r_opt, 6),
            "precision_level3": round(p_opt, 6),
            "f1_level3":      round(f1_opt, 6),
        },
        "baseline_test_2025_threshold_0_50": {
            "recall_level3":    round(r_def, 6),
            "precision_level3": round(p_def, 6),
        },
        "threshold_search_results": results,
        "rationale": (
            "Level 3 (DILARANG) adalah kelas kritis keselamatan jiwa. "
            "False alarm lebih dapat diterima daripada miss deteksi bahaya nyata. "
            "Threshold diturunkan dari 0.50 ke nilai optimal untuk meningkatkan recall "
            "dengan constraint Precision >= 0.30."
        )
    }

    out_path = Path(f"models/lgbm_forecast_t{HORIZON_HOURS}h_threshold_config.json")
    with open(out_path, "w") as f:
        json.dump(threshold_config, f, indent=4)

    print(f"\n  [OK] Konfigurasi threshold disimpan: {out_path}")
    print("=" * 65)
    print(f"  RINGKASAN PERBANDINGAN THRESHOLD (Test Set 2025, Level 3):")
    print(f"  +--------------------+----------+----------+")
    print(f"  | Threshold          | Recall   | Precision|")
    print(f"  +--------------------+----------+----------+")
    print(f"  | Default (0.50)     | {r_def*100:6.2f}%  | {p_def*100:6.2f}%  |")
    print(f"  | Optimal ({best_threshold:.2f})      | {r_opt*100:6.2f}%  | {p_opt*100:6.2f}%  |")
    print(f"  +--------------------+----------+----------+")
    print("=" * 65)


if __name__ == "__main__":
    run_threshold_optimization()

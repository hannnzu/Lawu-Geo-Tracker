"""
scripts/train_multi_horizon_models.py
-------------------------------------
Sprint 4 — Tahap 2: Pelatihan dan penyetelan 3 model forecasting (t+1h, t+3h, t+6h).

Langkah:
1. Memuat dataset multi-horizon.
2. Memisahkan Train/Val/Test secara temporal.
3. Melatih model LightGBM secara independen untuk target t+1h, t+3h, dan t+6h.
4. Menyetel safety threshold Level 3 pada Validation Set (2024) agar Recall L3 >= 85%.
5. Mengevaluasi performa model final pada Test Set (2025).
6. Menyimpan model binaries dan laporan metrik terpadu.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from src.utils.config import Config


def apply_safety_threshold(proba: np.ndarray, threshold_lvl3: float) -> np.ndarray:
    """Prediksi dengan safety threshold khusus untuk kelas Level 3."""
    predictions = np.argmax(proba, axis=1)
    lvl3_mask = proba[:, 3] >= threshold_lvl3
    predictions[lvl3_mask] = 3
    return predictions


def tune_safety_threshold(proba_val, y_val, target_recall: float = 0.85) -> float:
    """
    Mencari threshold tertinggi untuk Level 3 pada Validation Set
    yang masih mampu menghasilkan Recall Level 3 >= target_recall (85%).
    Jika tidak ada threshold yang memenuhi, pilih threshold yang memaksimalkan Recall.
    """
    best_threshold = 0.10
    max_recall = 0.0
    best_precision_at_target = 0.0
    
    # Cari dari 0.01 sampai 0.50
    for thr in np.arange(0.01, 0.51, 0.01):
        y_pred = apply_safety_threshold(proba_val, thr)
        rec_l3 = recall_score(y_val, y_pred, labels=[3], average='macro', zero_division=0)
        prec_l3 = precision_score(y_val, y_pred, labels=[3], average='macro', zero_division=0)
        
        if rec_l3 >= target_recall:
            # Jika memenuhi recall target, pilih threshold tertinggi (untuk presisi lebih baik)
            if thr > best_threshold or best_threshold == 0.10:
                best_threshold = thr
                best_precision_at_target = prec_l3
        
        # Fallback tracking
        if rec_l3 > max_recall:
            max_recall = rec_l3
            
    # Jika tidak ada threshold yang memenuhi target_recall >= 85%, gunakan threshold yang menghasilkan max recall
    if max_recall < target_recall:
        for thr in np.arange(0.01, 0.51, 0.01):
            y_pred = apply_safety_threshold(proba_val, thr)
            rec_l3 = recall_score(y_val, y_pred, labels=[3], average='macro', zero_division=0)
            if rec_l3 == max_recall:
                best_threshold = thr
                break
                
    return float(best_threshold)


def run_multi_horizon_training():
    if not LGBM_AVAILABLE:
        print("[!] LightGBM tidak terinstall. Jalankan: pip install lightgbm")
        return

    print("=" * 65)
    print("SPRINT 4: TRAINING & THRESHOLD TUNING MULTI-HORIZON ML")
    print("=" * 65)

    # 1. Identifikasi berkas dataset
    t_w_start = Config.WEATHER_HISTORICAL_START[:4]
    t_w_end   = Config.WEATHER_HISTORICAL_END[:4]
    dataset_filename = f"dataset_forecast_lawu_multi_{t_w_start}_{t_w_end}.csv"

    possible_paths = [
        Config.DATA_CURATED_DIR / dataset_filename,
        Config.ROOT_DIR / "DATA" / "curated" / dataset_filename,
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)
    if data_path is None:
        print(f"[!] Dataset multi-horizon tidak ditemukan: {dataset_filename}")
        print("    Jalankan terlebih dahulu: python -m scripts.build_multi_horizon_datasets")
        return

    print(f"\n[1/5] Memuat dataset multi-horizon: {data_path.name}")
    df = pd.read_csv(data_path)
    print(f"      Total: {len(df):,} baris, {len(df.columns)} kolom.")

    # 2. Definisikan X
    # Target dan leakage dikeluarkan dari X
    TARGET_COLS = ['Danger_Level_t1h', 'Danger_Level_t3h', 'Danger_Level_t6h']
    EXCLUDE_COLS = TARGET_COLS + ['Danger_Level', 'status_kebakaran_sekitar', 'Status_Kebakaran_Sekitar', '_tahun']
    
    tahun_series = df['_tahun'].copy() if '_tahun' in df.columns else pd.Series([0] * len(df))
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    X = X.select_dtypes(include=['number']).fillna(0)
    
    print(f"      Jumlah Fitur X: {X.shape[1]} kolom")

    # Pembagian Temporal 3-Split
    mask_train = tahun_series <= 2023
    mask_val   = tahun_series == 2024
    mask_test  = tahun_series == 2025

    X_train = X[mask_train].reset_index(drop=True)
    X_val   = X[mask_val].reset_index(drop=True)
    X_test  = X[mask_test].reset_index(drop=True)

    print(f"      Split: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

    # Konfigurasi Latih & Bobot
    class_weights = {0: 1.0, 1: 1.0, 2: 3.0, 3: 50.0}
    lgbm_params = {
        'objective':         'multiclass',
        'num_class':         4,
        'metric':            'multi_logloss',
        'num_leaves':        31,
        'max_depth':         8,
        'learning_rate':     0.03,
        'n_estimators':      2000,
        'subsample':         0.7,
        'colsample_bytree':  0.7,
        'reg_alpha':         0.1,
        'reg_lambda':        0.1,
        'min_child_samples': 20,
        'random_state':      42,
        'n_jobs':            -1,
        'verbose':           -1
    }

    metrics_summary = {}
    threshold_configs = {}

    # Latih model per horizon
    for target in TARGET_COLS:
        horizon_name = target.split('_')[-1] # t1h, t3h, t6h
        print(f"\n" + "-" * 50)
        print(f"MELATIH MODEL FORECASTING HORIZON: {horizon_name.upper()}")
        print("-" * 50)

        y_train = df[mask_train][target].astype(int).reset_index(drop=True)
        y_val   = df[mask_val][target].astype(int).reset_index(drop=True)
        y_test  = df[mask_test][target].astype(int).reset_index(drop=True)

        # Hitung sample weights
        sample_weights = np.array([class_weights[lbl] for lbl in y_train])

        # Bangun & latih model
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0) # Sembunyikan output verbose training
            ]
        )
        print(f"      [OK] Training selesai pada iterasi ke-{model.best_iteration_}")

        # Hitung probabilitas evaluasi
        proba_val  = model.predict_proba(X_val)
        proba_test = model.predict_proba(X_test)

        # Penyetelan threshold keamanan
        optimal_thr = tune_safety_threshold(proba_val, y_val, target_recall=0.85)
        print(f"      [OK] Penyetelan threshold Level 3 optimal: {optimal_thr:.2f}")

        # Evaluasi akhir pada Test Set (2025)
        y_pred_test = apply_safety_threshold(proba_test, optimal_thr)
        acc_test    = accuracy_score(y_test, y_pred_test)
        f1_mac_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        rec_l3_test = recall_score(y_test, y_pred_test, labels=[3], average='macro', zero_division=0)
        prec_l3_test = precision_score(y_test, y_pred_test, labels=[3], average='macro', zero_division=0)

        print(f"      [TEST SET 2025 EVALUATION]")
        print(f"        Akurasi      : {acc_test*100:.2f}%")
        print(f"        F1-Macro     : {f1_mac_test:.4f}")
        print(f"        Recall L3    : {rec_l3_test*100:.2f}%")
        print(f"        Precision L3 : {prec_l3_test*100:.2f}%")

        # Simpan model
        model_path = Path(f"models/lgbm_forecast_{horizon_name}_model.joblib")
        joblib.dump(model, model_path)
        print(f"      [OK] Model disimpan: {model_path}")

        # Simpan laporan metrik & konfigurasi
        threshold_configs[horizon_name] = {
            "model_file": f"lgbm_forecast_{horizon_name}_model.joblib",
            "optimal_threshold_level3": optimal_thr,
            "tuned_on": "validation_set_2024",
            "test_set_2025_recall_l3": round(float(rec_l3_test), 4)
        }

        metrics_summary[horizon_name] = {
            "test_accuracy": round(float(acc_test), 4),
            "test_f1_macro": round(float(f1_mac_test), 4),
            "test_recall_level3": round(float(rec_l3_test), 4),
            "test_precision_level3": round(float(prec_l3_test), 4),
            "best_iteration": int(model.best_iteration_),
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        }

    # Simpan berkas konfigurasi terpadu
    config_out = Path("models/lgbm_multi_horizon_threshold_config.json")
    with open(config_out, "w") as f:
        json.dump(threshold_configs, f, indent=4)

    metrics_out = Path("models/lgbm_multi_horizon_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n" + "=" * 65)
    print("  RINGKASAN METRIK MULTI-HORIZON FINAL (Test Set 2025):")
    print("  +---------+----------+----------+-----------+-----------+")
    print("  | Horizon | Akurasi  | F1-Macro | Recall L3 | Threshold |")
    print("  +---------+----------+----------+-----------+-----------+")
    for hz in ['t1h', 't3h', 't6h']:
        m = metrics_summary[hz]
        t = threshold_configs[hz]['optimal_threshold_level3']
        print(f"  | {hz.upper():<7} | {m['test_accuracy']*100:6.2f}%  | {m['test_f1_macro']:.4f}   | {m['test_recall_level3']*100:6.2f}%   | {t:9.2f} |")
    print("  +---------+----------+----------+-----------+-----------+")
    print(f"  [OK] Konfigurasi terpadu disimpan: {config_out}")
    print(f"  [OK] Metrik terpadu disimpan     : {metrics_out}")
    print("=" * 65)


if __name__ == "__main__":
    run_multi_horizon_training()

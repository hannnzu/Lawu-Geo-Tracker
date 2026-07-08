"""
scripts/train_lgbm_forecast.py
----------------------------------
Sprint 3: Training model LightGBM untuk prediksi Danger_Level t+3 jam ke depan.

Target BUKAN lagi 'Danger_Level' (kondisi saat ini) melainkan
'Danger_Level_t3h' (kondisi 3 jam ke depan).

Karena ada ketidakpastian alami pada prediksi masa depan, akurasi
model ini diharapkan LEBIH RENDAH dari model nowcasting (< 100%).
Jika akurasi masih 100%, ada bug pada feature engineering di Tahap 1.

Hyperparameter lebih ketat (L1/L2 regularisasi) untuk mencegah
overfitting pada fitur lag yang memiliki korelasi tinggi satu sama lain.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[!] LightGBM tidak terinstall. Jalankan: pip install lightgbm")

from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

from src.utils.config import Config


HORIZON_HOURS = 3  # Harus sama dengan yang digunakan di build_forecast_dataset.py


def run_forecast_training():
    if not LGBM_AVAILABLE:
        print("LightGBM tidak tersedia. Training dibatalkan.")
        return

    print("=" * 65)
    print(f"SPRINT 3: TRAINING LGBM FORECASTING (t+{HORIZON_HOURS} jam)")
    print("=" * 65)

    # Cari dataset forecasting yang dibuat Tahap 1
    t_w_start = Config.WEATHER_HISTORICAL_START[:4]
    t_w_end   = Config.WEATHER_HISTORICAL_END[:4]
    forecast_filename = f"dataset_forecast_lawu_t{HORIZON_HOURS}h_{t_w_start}_{t_w_end}.csv"

    possible_paths = [
        Config.DATA_CURATED_DIR / forecast_filename,
        Config.ROOT_DIR / "DATA" / "curated" / forecast_filename,
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)
    if data_path is None:
        print(f"[!] Dataset forecasting tidak ditemukan: {forecast_filename}")
        print("    Jalankan terlebih dahulu: python -m scripts.build_forecast_dataset")
        return

    model_output_path   = Path(f"models/lgbm_forecast_t{HORIZON_HOURS}h_model.joblib")
    metrics_output_path = Path(f"models/lgbm_forecast_t{HORIZON_HOURS}h_metrics.json")

    # ----------------------------------------------------------
    # LANGKAH 1: Muat Dataset Forecasting
    # ----------------------------------------------------------
    print(f"\n[1/5] Memuat dataset forecasting: {data_path.name}")
    df = pd.read_csv(data_path)
    print(f"      Total: {len(df):,} baris, {len(df.columns)} kolom.")

    # ----------------------------------------------------------
    # LANGKAH 2: Definisikan Fitur (X) dan Target (y)
    # FASE KRITIS: Target adalah Danger_Level_t3h (BUKAN Danger_Level).
    # Kolom Danger_Level asli WAJIB dibuang dari X agar model tidak
    # curang dengan "melihat" label saat ini untuk memprediksi t+3.
    # ----------------------------------------------------------
    target_col = f'Danger_Level_t{HORIZON_HOURS}h'
    if target_col not in df.columns:
        print(f"[!] Kolom target '{target_col}' tidak ada. Pastikan build_forecast_dataset.py berhasil.")
        return

    print(f"\n[2/5] Mendefinisikan fitur X dan target y='{target_col}'...")
    EXCLUDE_COLS = [
        target_col,
        'Danger_Level',              # Buang label saat ini — ini adalah "jawaban" masa depan
        'status_kebakaran_sekitar',  # Data leakage deterministik
        'Status_Kebakaran_Sekitar',
        '_tahun',                    # Kolom bantu temporal split
    ]

    tahun_series = df['_tahun'].copy() if '_tahun' in df.columns else pd.Series([0] * len(df))

    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    X = X.select_dtypes(include=['number'])
    X = X.fillna(0)
    y = df[target_col].astype(int)

    print(f"      Total fitur X: {X.shape[1]} kolom")
    print(f"      Distribusi kelas target:")
    for lvl, cnt in y.value_counts().sort_index().items():
        print(f"        Level {lvl}: {cnt:,} ({cnt/len(y)*100:.2f}%)")

    # ----------------------------------------------------------
    # LANGKAH 3: Temporal 3-Split
    # ----------------------------------------------------------
    print("\n[3/5] Temporal 3-Split (Train<=2023, Val=2024, Test=2025)...")
    mask_train = tahun_series <= 2023
    mask_val   = tahun_series == 2024
    mask_test  = tahun_series == 2025

    X_train, y_train = X[mask_train].reset_index(drop=True), y[mask_train].reset_index(drop=True)
    X_val,   y_val   = X[mask_val].reset_index(drop=True),   y[mask_val].reset_index(drop=True)
    X_test,  y_test  = X[mask_test].reset_index(drop=True),  y[mask_test].reset_index(drop=True)

    total = len(X)
    print(f"      Train (2021-2023): {len(X_train):>7,} ({len(X_train)/total*100:.1f}%)")
    print(f"      Val   (2024):      {len(X_val):>7,} ({len(X_val)/total*100:.1f}%)")
    print(f"      Test  (2025):      {len(X_test):>7,} ({len(X_test)/total*100:.1f}%)")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("[!] Salah satu split kosong!")
        return

    # ----------------------------------------------------------
    # LANGKAH 4: Training LightGBM Forecasting
    # FASE KRITIS: Hyperparameter lebih ketat dibanding nowcasting.
    # - num_leaves dikurangi (31 vs 63) untuk menghindari overfitting
    #   pada fitur lag yang berkorelasi tinggi.
    # - reg_alpha (L1) dan reg_lambda (L2) ditambahkan untuk
    #   memaksa model mengabaikan fitur lag yang tidak informatif.
    # - min_child_samples ditingkatkan (50 vs 20) untuk stabilitas
    #   pada distribusi kelas yang lebih tidak seimbang.
    # ----------------------------------------------------------
    print("\n[4/5] Training LightGBM Forecasting (hyperparameter regularisasi ketat)...")
    lgbm_params = {
        'objective':         'multiclass',
        'num_class':         4,
        'metric':            'multi_logloss',
        'num_leaves':        31,          # Lebih rendah dari nowcasting (63) — cegah overfit
        'max_depth':         8,           # Batasan kedalaman eksplisit
        'learning_rate':     0.03,        # Lebih lambat untuk generalisasi lebih baik
        'n_estimators':      2000,        # Max (early stopping akan memotong lebih awal)
        'class_weight':      'balanced',
        'subsample':         0.7,         # Row subsampling lebih ketat
        'colsample_bytree':  0.7,
        'reg_alpha':         0.1,         # L1 regularization — pruning fitur tidak penting
        'reg_lambda':        0.1,         # L2 regularization — penalti bobot besar
        'min_child_samples': 50,          # Stabilitas pada kelas minoritas
        'random_state':      42,
        'n_jobs':            -1,
        'verbose':           -1
    }

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    print(f"      Jumlah pohon yang digunakan: {model.best_iteration_}")

    # ----------------------------------------------------------
    # LANGKAH 5: Evaluasi & Simpan
    # Verifikasi kritis (statistical-analyst): akurasi HARUS < 100%.
    # Jika akurasi masih 100%, ada bug look-ahead bias.
    # ----------------------------------------------------------
    print("\n[5/5] Evaluasi model pada Train / Validation / Test Set...")

    def evaluate_split(name: str, X_s, y_s) -> dict:
        y_pred = model.predict(X_s)
        acc    = accuracy_score(y_s, y_pred)
        f1_mac = f1_score(y_s, y_pred, average='macro', zero_division=0)
        report = classification_report(y_s, y_pred, output_dict=True, zero_division=0)
        report_str = classification_report(y_s, y_pred, zero_division=0)
        print(f"\n  [{name}]")
        print(f"  Akurasi  : {acc*100:.2f}%")
        print(f"  F1-Macro : {f1_mac:.4f}")
        print(f"\n{report_str}")

        # AUDIT STATISTIKAL: Recall Level 3 (kelas kritis keselamatan)
        recall_lvl3 = report.get('3', {}).get('recall', 0.0)
        print(f"  [AUDIT] Recall Level 3 (Dilarang): {recall_lvl3*100:.2f}%")
        if recall_lvl3 < 0.85:
            print("  [!] PERHATIAN: Recall Level 3 < 85%. Model berisiko miss deteksi bahaya kritis.")
        else:
            print("  [OK] Recall Level 3 memenuhi target >= 85%.")

        return {
            "accuracy":      round(acc, 6),
            "f1_macro":      round(f1_mac, 6),
            "recall_level3": round(recall_lvl3, 6),
            "n_samples":     len(y_s),
            "report":        report
        }

    print("-" * 65)
    train_metrics = evaluate_split("TRAIN SET  (2021-2023)", X_train, y_train)
    val_metrics   = evaluate_split("VALIDATION SET (2024)", X_val,   y_val)
    test_metrics  = evaluate_split("TEST SET   (2025)",     X_test,  y_test)
    print("-" * 65)

    # Verifikasi degradasi akurasi (statistical-analyst check)
    print("\n  [AUDIT STATISTIKAL: Verifikasi Degradasi Akurasi]")
    if test_metrics['accuracy'] < 0.99:
        print(f"  [OK] Akurasi test {test_metrics['accuracy']*100:.2f}% < 99% -> Model belajar secara statistik (bukan deterministik).")
    else:
        print(f"  [!!] PERINGATAN: Akurasi masih {test_metrics['accuracy']*100:.2f}%. Kemungkinan ada look-ahead bias.")
        print("       Periksa kembali penggunaan .shift(1) di dalam rolling features.")

    # Simpan model
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    # Feature importance
    feature_importances = dict(sorted(
        {feat: int(imp) for feat, imp in zip(X_train.columns, model.feature_importances_)}.items(),
        key=lambda x: x[1], reverse=True
    ))

    metrics_data = {
        "algorithm":          "LightGBM Forecasting",
        "horizon_hours":      HORIZON_HOURS,
        "split_scheme":       "temporal_3split",
        "best_iteration":     int(model.best_iteration_),
        "lgbm_params":        lgbm_params,
        "train_metrics":      train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics":       test_metrics,
        "feature_importances": feature_importances,
        "top10_features":     list(feature_importances.keys())[:10],
    }

    with open(metrics_output_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    model_size_mb = model_output_path.stat().st_size / (1024 * 1024)
    print(f"\n  [OK] Model  disimpan: {model_output_path} ({model_size_mb:.2f} MB)")
    print(f"  [OK] Metrik disimpan: {metrics_output_path}")
    print(f"\n  RINGKASAN AKHIR (LightGBM Forecast t+{HORIZON_HOURS}h):")
    print(f"  +---------------------+----------+----------+")
    print(f"  | Split               | Akurasi  | F1-Macro |")
    print(f"  +---------------------+----------+----------+")
    print(f"  | Train  (2021-2023)  | {train_metrics['accuracy']*100:6.2f}%  | {train_metrics['f1_macro']:.4f}   |")
    print(f"  | Val    (2024)       | {val_metrics['accuracy']*100:6.2f}%  | {val_metrics['f1_macro']:.4f}   |")
    print(f"  | Test   (2025)       | {test_metrics['accuracy']*100:6.2f}%  | {test_metrics['f1_macro']:.4f}   |")
    print(f"  +---------------------+----------+----------+")
    print(f"  | Recall Level 3 Test | {test_metrics['recall_level3']*100:6.2f}%            |")
    print(f"  +---------------------+----------+----------+")
    print("=" * 65)


if __name__ == "__main__":
    run_forecast_training()

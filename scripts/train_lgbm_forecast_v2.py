"""
scripts/train_lgbm_forecast_v2.py
----------------------------------
Sprint 3 — Retrain v2: Custom Class Weight Ekstrem untuk Level 3.

DIAGNOSA v1:
  - 46.2% sampel Level 3 sesungguhnya diprediksi sebagai Level 2
  - P(Level 3) rata-rata hanya 0.39, median 0.15
  - Model "ragu" memilih Level 3 karena distribusinya sangat jarang di 2025
  - 95/247 (38.5%) sampel punya P(L3) >= 0.30, tapi masih kalah dari P(L2)

SOLUSI v2: Custom Class Weight {0:1, 1:1, 2:3, 3:50}
  - Level 3 diberi bobot 50x lipat vs Level 0/1
  - Level 2 diberi bobot 3x (juga kurang terwakili)
  - Ini memaksa model membayar "penalti" 50x lebih besar saat salah
    mengklasifikasikan Level 3, sehingga model lebih agresif memilihnya
  - Threshold juga diturunkan ke 0.10 sebagai fallback

KONTEKS (statistical-analyst):
  Dalam sistem keselamatan jiwa (SAR/emergency), cost matrix tidak simetris:
    - Miss Level 3 (False Negative) = Pendaki masuk zona larangan = risiko jiwa
    - False Alarm Level 3 (False Positive) = Penutupan jalur sia-sia = kerugian ekonomi kecil
  Oleh karena itu, bias model menuju over-detection Level 3 secara etis dibenarkan.
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

from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import joblib

from src.utils.config import Config

HORIZON_HOURS = 3


def apply_safety_threshold(proba: np.ndarray, threshold_lvl3: float = 0.10) -> np.ndarray:
    """
    Prediksi dengan safety threshold untuk Level 3.
    Jika P(Level 3) >= threshold, langsung prediksi Level 3.
    Threshold 0.10 memberikan keseimbangan recall-precision yang baik.
    """
    predictions = np.argmax(proba, axis=1)
    lvl3_mask = proba[:, 3] >= threshold_lvl3
    predictions[lvl3_mask] = 3
    return predictions


def run_v2_training():
    if not LGBM_AVAILABLE:
        print("LightGBM tidak tersedia.")
        return

    print("=" * 65)
    print(f"SPRINT 3 v2: LGBM FORECASTING — Custom Weight Level 3 (t+{HORIZON_HOURS}h)")
    print("=" * 65)

    # Cari dataset forecasting
    t_w_start = Config.WEATHER_HISTORICAL_START[:4]
    t_w_end   = Config.WEATHER_HISTORICAL_END[:4]
    forecast_filename = f"dataset_forecast_lawu_t{HORIZON_HOURS}h_{t_w_start}_{t_w_end}.csv"

    possible_paths = [
        Config.DATA_CURATED_DIR / forecast_filename,
        Config.ROOT_DIR / "DATA" / "curated" / forecast_filename,
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)
    if not data_path:
        print(f"[!] Dataset tidak ditemukan. Jalankan build_forecast_dataset.py.")
        return

    model_output_path   = Path(f"models/lgbm_forecast_t{HORIZON_HOURS}h_v2_model.joblib")
    metrics_output_path = Path(f"models/lgbm_forecast_t{HORIZON_HOURS}h_v2_metrics.json")

    # Muat data
    print(f"\n[1/5] Memuat dataset: {data_path.name}")
    df = pd.read_csv(data_path)
    print(f"      Total: {len(df):,} baris, {len(df.columns)} kolom.")

    target_col = f'Danger_Level_t{HORIZON_HOURS}h'
    tahun_series = df['_tahun'].copy() if '_tahun' in df.columns else pd.Series([0] * len(df))

    EXCLUDE_COLS = [target_col, 'Danger_Level', 'status_kebakaran_sekitar',
                    'Status_Kebakaran_Sekitar', '_tahun']
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    X = X.select_dtypes(include=['number']).fillna(0)
    y = df[target_col].astype(int)

    print(f"\n[2/5] Temporal 3-Split...")
    mask_train = tahun_series <= 2023
    mask_val   = tahun_series == 2024
    mask_test  = tahun_series == 2025

    X_train, y_train = X[mask_train].reset_index(drop=True), y[mask_train].reset_index(drop=True)
    X_val,   y_val   = X[mask_val].reset_index(drop=True),   y[mask_val].reset_index(drop=True)
    X_test,  y_test  = X[mask_test].reset_index(drop=True),  y[mask_test].reset_index(drop=True)

    print(f"      Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"      Level 3 di Train: {(y_train==3).sum():,} | Val: {(y_val==3).sum():,} | Test: {(y_test==3).sum():,}")

    # ----------------------------------------------------------
    # CUSTOM CLASS WEIGHT — Kunci perbaikan v2
    # ----------------------------------------------------------
    # Bobot berbanding terbalik dengan frekuensi kelas, tapi Level 3
    # diberi bobot ekstra tinggi karena konsekuensi keselamatan jiwa.
    # Rasio: L0=1, L1=1, L2=3x, L3=50x
    # ----------------------------------------------------------
    class_weights_custom = {0: 1.0, 1: 1.0, 2: 3.0, 3: 50.0}
    print(f"\n[3/5] Class weight custom: {class_weights_custom}")

    # Bangun sample_weight array sesuai label setiap baris
    sample_weights_train = np.array([class_weights_custom[lbl] for lbl in y_train])

    print(f"\n[4/5] Training LightGBM v2 (custom weight Level 3 = 50x)...")
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
        'min_child_samples': 20,    # Dikurangi untuk Level 3 yang jarang
        'random_state':      42,
        'n_jobs':            -1,
        'verbose':           -1
    }

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    print(f"      Jumlah pohon: {model.best_iteration_}")

    # ----------------------------------------------------------
    # EVALUASI dengan Safety Threshold 0.10
    # ----------------------------------------------------------
    print("\n[5/5] Evaluasi dengan Safety Threshold P(Level3) >= 0.10...")

    def evaluate(name, X_s, y_s, threshold=0.10):
        proba  = model.predict_proba(X_s)
        y_pred = apply_safety_threshold(proba, threshold)
        acc    = accuracy_score(y_s, y_pred)
        f1_mac = f1_score(y_s, y_pred, average='macro', zero_division=0)
        r_l3   = recall_score(y_s, y_pred, labels=[3], average='macro', zero_division=0)
        p_l3   = precision_score(y_s, y_pred, labels=[3], average='macro', zero_division=0)
        report = classification_report(y_s, y_pred, zero_division=0)
        print(f"\n  [{name}] — threshold={threshold}")
        print(f"  Akurasi  : {acc*100:.2f}%  |  F1-Macro: {f1_mac:.4f}")
        print(f"  Recall L3: {r_l3*100:.2f}%  |  Precision L3: {p_l3*100:.2f}%")
        print(f"\n{report}")
        if r_l3 < 0.85:
            print(f"  [!] Recall Level 3 masih < 85%. Pertimbangkan threshold lebih rendah.")
        else:
            print(f"  [OK] Recall Level 3 memenuhi target >= 85%.")
        return {"accuracy": round(acc,6), "f1_macro": round(f1_mac,6),
                "recall_level3": round(r_l3,6), "precision_level3": round(p_l3,6),
                "threshold_used": threshold, "n_samples": len(y_s)}

    print("-" * 65)
    safety_threshold = 0.10
    train_m = evaluate("TRAIN (2021-2023)",  X_train, y_train, safety_threshold)
    val_m   = evaluate("VALIDATION (2024)",  X_val,   y_val,   safety_threshold)
    test_m  = evaluate("TEST (2025)",        X_test,  y_test,  safety_threshold)
    print("-" * 65)

    print("\n  [AUDIT] Verifikasi degradasi akurasi:")
    if test_m['accuracy'] < 0.99:
        print(f"  [OK] Test = {test_m['accuracy']*100:.2f}% — Paradigma forecasting valid (< 99%).")
    else:
        print(f"  [!!] Akurasi masih {test_m['accuracy']*100:.2f}%. Cek look-ahead bias!")

    # Simpan
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    fi = dict(sorted(
        {f: int(i) for f,i in zip(X_train.columns, model.feature_importances_)}.items(),
        key=lambda x: x[1], reverse=True
    ))
    metrics = {
        "algorithm":          "LightGBM Forecasting v2 (Custom Weight)",
        "version":            "v2",
        "horizon_hours":      HORIZON_HOURS,
        "class_weights":      class_weights_custom,
        "safety_threshold":   safety_threshold,
        "best_iteration":     int(model.best_iteration_),
        "lgbm_params":        lgbm_params,
        "train_metrics":      train_m,
        "validation_metrics": val_m,
        "test_metrics":       test_m,
        "top10_features":     list(fi.keys())[:10],
        "feature_importances": fi,
    }
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    sz = model_output_path.stat().st_size / (1024*1024)
    print(f"\n  [OK] Model  disimpan: {model_output_path} ({sz:.2f} MB)")
    print(f"  [OK] Metrik disimpan: {metrics_output_path}")
    print(f"\n  RINGKASAN v2 (LightGBM Forecast t+{HORIZON_HOURS}h, thr={safety_threshold}):")
    print(f"  +---------------------+----------+----------+-----------+")
    print(f"  | Split               | Akurasi  | F1-Macro | Recall L3 |")
    print(f"  +---------------------+----------+----------+-----------+")
    for lbl, m in [("Train  (2021-2023)", train_m), ("Val    (2024)    ", val_m), ("Test   (2025)    ", test_m)]:
        print(f"  | {lbl} | {m['accuracy']*100:6.2f}%  | {m['f1_macro']:.4f}   | {m['recall_level3']*100:6.2f}%   |")
    print(f"  +---------------------+----------+----------+-----------+")
    print("=" * 65)


if __name__ == "__main__":
    run_v2_training()

"""
train_lgbm.py
--------------------------------
Script eksperimen: Training model LightGBM sebagai alternatif
Random Forest untuk memprediksi Danger Level di Gunung Lawu.

Keunggulan LightGBM vs Random Forest:
- Ukuran model jauh lebih kecil (~2-5 MB vs ~20 MB)
- Training jauh lebih cepat (~30 detik vs ~10 menit)
- Early stopping mencegah overfitting secara otomatis

Data Splitting: Temporal 3-Split (Train: 2021-2023, Val: 2024, Test: 2025)
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


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan encoding siklis (sin/cos) untuk fitur temporal.
    Identik dengan fungsi di train_model.py untuk konsistensi.
    """
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


def run_lgbm_training():
    if not LGBM_AVAILABLE:
        print("LightGBM tidak tersedia. Training dibatalkan.")
        return

    print("=" * 65)
    print("EKSPERIMEN: LIGHTGBM CLASSIFIER (Temporal 3-Split)")
    print("=" * 65)

    # Cari dataset
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

    model_output_path   = Path("models/lgbm_danger_level_model.joblib")
    metrics_output_path = Path("models/lgbm_model_metrics.json")

    print(f"\n[1/6] Memuat dataset: {data_path.name}")
    df = pd.read_csv(data_path)
    print(f"      Total: {len(df):,} baris, {len(df.columns)} kolom.")

    target_col = 'Danger_Level'
    if target_col not in df.columns:
        print(f"[!] Kolom '{target_col}' tidak ditemukan!")
        return

    print("\n[2/6] Feature engineering (Cyclic + Interaction)...")
    df = add_cyclic_features(df)
    df = add_interaction_features(df)

    print("\n[3/6] Pre-processing: Menghapus kolom deterministik...")
    LEAKAGE_COLS  = ['status_kebakaran_sekitar', 'Status_Kebakaran_Sekitar']
    TEMPORAL_COLS = ['_tahun']
    EXCLUDE_COLS  = [target_col] + LEAKAGE_COLS + TEMPORAL_COLS

    tahun_series = df['_tahun'].copy()

    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    X = X.select_dtypes(include=['number'])
    X = X.fillna(0)
    y = df[target_col]

    dropped_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    print(f"      Fitur leakage dibuang: {dropped_leakage if dropped_leakage else 'Tidak ada'}")
    print(f"      Total fitur: {X.shape[1]} kolom")

    print("\n[4/6] Temporal 3-Split...")
    mask_train = tahun_series <= 2023
    mask_val   = tahun_series == 2024
    mask_test  = tahun_series == 2025

    X_train, y_train = X[mask_train].reset_index(drop=True), y[mask_train].reset_index(drop=True)
    X_val,   y_val   = X[mask_val].reset_index(drop=True),   y[mask_val].reset_index(drop=True)
    X_test,  y_test  = X[mask_test].reset_index(drop=True),  y[mask_test].reset_index(drop=True)

    total = len(X)
    print(f"      -> Train (2021-2023): {len(X_train):>7,} ({len(X_train)/total*100:.1f}%)")
    print(f"      -> Val   (2024):      {len(X_val):>7,} ({len(X_val)/total*100:.1f}%)")
    print(f"      -> Test  (2025):      {len(X_test):>7,} ({len(X_test)/total*100:.1f}%)")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("[!] Salah satu split kosong!")
        return

    # ============================================================
    # FASE KRITIS: Training LightGBM dengan Early Stopping
    # Early stopping memantau performa di Validation Set setiap
    # iterasi dan menghentikan training jika tidak ada perbaikan
    # selama N iterasi — mencegah overfitting secara otomatis.
    # ============================================================
    print("\n[5/6] Training LightGBM Classifier (dengan Early Stopping)...")
    lgbm_params = {
        'objective':      'multiclass',
        'num_class':      4,             # Level 0, 1, 2, 3
        'metric':         'multi_logloss',
        'num_leaves':     63,            # Max daun per pohon (kompleksitas)
        'learning_rate':  0.05,          # Learning rate lambat → lebih stabil
        'n_estimators':   1000,          # Max pohon (early stopping akan berhenti lebih awal)
        'class_weight':   'balanced',    # Penanganan imbalance kelas minoritas
        'subsample':      0.8,           # Row subsampling per iterasi (GOSS)
        'colsample_bytree': 0.8,         # Feature subsampling per pohon
        'min_child_samples': 20,         # Regularisasi daun kecil
        'random_state':   42,
        'n_jobs':         -1,
        'verbose':        -1             # Suppress verbose output
    }

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],               # Monitor Validation Set
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),  # Stop jika 50 iter tidak membaik
            lgb.log_evaluation(period=50)          # Log setiap 50 iterasi
        ]
    )
    print(f"      Jumlah pohon yang digunakan: {model.best_iteration_}")

    print("\n[6/6] Evaluasi model pada Train / Validation / Test Set...")

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
        return {"accuracy": round(acc, 6), "f1_macro": round(f1_mac, 6),
                "n_samples": len(y_s), "report": report}

    print("-" * 65)
    train_metrics = evaluate_split("TRAIN SET  (2021-2023)", X_train, y_train)
    val_metrics   = evaluate_split("VALIDATION SET (2024)", X_val,   y_val)
    test_metrics  = evaluate_split("TEST SET   (2025)",     X_test,  y_test)
    print("-" * 65)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    # Feature importance dari LightGBM
    feature_importances = {
        feat: int(imp)
        for feat, imp in zip(X_train.columns, model.feature_importances_)
    }
    feature_importances = dict(
        sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    )

    metrics_data = {
        "algorithm":          "LightGBM",
        "split_scheme":       "temporal_3split",
        "best_iteration":     int(model.best_iteration_),
        "lgbm_params":        lgbm_params,
        "train_metrics":      train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics":       test_metrics,
        "feature_importances": feature_importances,
    }

    with open(metrics_output_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    model_size_mb = model_output_path.stat().st_size / (1024 * 1024)
    print(f"\n  [OK] Model  disimpan di: {model_output_path} ({model_size_mb:.2f} MB)")
    print(f"  [OK] Metrik disimpan di: {metrics_output_path}")
    print(f"\n  RINGKASAN AKHIR (LightGBM):")
    print(f"  +---------------------+----------+----------+")
    print(f"  | Split               | Akurasi  | F1-Macro |")
    print(f"  +---------------------+----------+----------+")
    print(f"  | Train  (2021-2023)  | {train_metrics['accuracy']*100:6.2f}%  | {train_metrics['f1_macro']:.4f}   |")
    print(f"  | Val    (2024)       | {val_metrics['accuracy']*100:6.2f}%  | {val_metrics['f1_macro']:.4f}   |")
    print(f"  | Test   (2025)       | {test_metrics['accuracy']*100:6.2f}%  | {test_metrics['f1_macro']:.4f}   |")
    print(f"  +---------------------+----------+----------+")
    print("=" * 65)


if __name__ == "__main__":
    run_lgbm_training()

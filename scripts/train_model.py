"""
train_model.py
--------------------------------
Script untuk melatih ulang model Machine Learning (Random Forest)
untuk memprediksi Danger Level di jalur pendakian Gunung Lawu.

Data Splitting: Temporal 3-Split (Train: 2021-2023, Val: 2024, Test: 2025)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

from src.utils.config import Config


# ============================================================
# FASE KRITIS: Cyclic Encoding untuk Fitur Temporal
# Bulan/Jam bersifat siklis (Des→Jan bukan lompatan), encoding
# sinusoidal mencegah model memperlakukan ini sebagai linear.
# ============================================================
def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan encoding siklis (sin/cos) untuk fitur temporal agar
    model memahami siklus kalender tanpa bias linear.
    """
    df = df.copy()

    # Pastikan kolom datetime terparse sebagai objek waktu
    datetime_col = None
    for col in ['Timestamp', 'Datetime', 'datetime', 'time', 'date']:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col:
        dt = pd.to_datetime(df[datetime_col], errors='coerce')
        # Encoding siklis untuk bulan (1-12)
        df['bulan_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
        df['bulan_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
        # Encoding siklis untuk jam (0-23)
        df['jam_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df['jam_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
        # Encoding siklis untuk hari dalam tahun (1-365)
        df['doy_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
        df['doy_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear / 365)
        # Kolom bantu untuk temporal split
        df['_tahun'] = dt.dt.year
    else:
        print("      [!] Kolom datetime tidak ditemukan. Fitur siklis dilewati.")
        df['_tahun'] = 0  # Fallback agar split tetap bisa berjalan

    return df


# ============================================================
# FASE KRITIS: Fitur Interaksi Spasial-Meteorologi
# Kombinasi fitur menghasilkan sinyal prediktif lebih kuat dari
# fitur individual — menangkap risiko gabungan (compound risk).
# ============================================================
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur interaksi antara variabel meteorologi dan spasial
    untuk meningkatkan kualitas sinyal prediksi model.
    """
    df = df.copy()

    # Windchill Compound Risk: suhu rendah × angin kencang → risiko hipotermia
    if 'Suhu Terasa (C)' in df.columns and 'Angin Kencang (km/h)' in df.columns:
        df['compound_cold_wind'] = df['Suhu Terasa (C)'] * df['Angin Kencang (km/h)']

    # Fire Proximity Index: intensitas api / (jarak + epsilon) → risiko kebakaran terukur
    if 'FRP_Terdekat_MW' in df.columns and 'Jarak_Titik_Api_Terdekat_KM' in df.columns:
        df['fire_proximity_index'] = (
            df['FRP_Terdekat_MW'] / (df['Jarak_Titik_Api_Terdekat_KM'] + 0.1)
        )

    return df


def run_training():
    print("=" * 65)
    print("MEMULAI TRAINING MODEL ML (Temporal 3-Split: 2021-23 / 2024 / 2025)")
    print("=" * 65)

    # Coba dua kemungkinan path (DATA/ uppercase dan data/ lowercase)
    t_w_start = Config.WEATHER_HISTORICAL_START[:4]
    t_w_end   = Config.WEATHER_HISTORICAL_END[:4]
    filename  = f"dataset_integrated_lawu_{t_w_start}_{t_w_end}.csv"

    # Cek kedua path secara berurutan
    possible_paths = [
        Config.DATA_CURATED_DIR / filename,
        Config.ROOT_DIR / "DATA" / "curated" / filename,
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)

    if data_path is None:
        print(f"[!] Dataset tidak ditemukan. Dicari di:")
        for p in possible_paths:
            print(f"    - {p}")
        return

    model_output_path   = Path("models/rf_danger_level_model.joblib")
    metrics_output_path = Path("models/model_metrics.json")

    # ============================================================
    # FASE KRITIS: Memuat Dataset
    # Dataset berisi ~830 ribu baris data cuaca + kebakaran hutan.
    # ============================================================
    print(f"\n[1/6] Memuat dataset dari: {data_path.name}")
    df = pd.read_csv(data_path)
    print(f"      Total data awal: {len(df):,} baris, {len(df.columns)} kolom.")

    target_col = 'Danger_Level'
    if target_col not in df.columns:
        print(f"[!] Kolom '{target_col}' tidak ditemukan!")
        return

    # ============================================================
    # FASE KRITIS: Feature Engineering — Cyclic + Interaction
    # Dilakukan SEBELUM pemisahan data agar encoding konsisten
    # di seluruh train/val/test set.
    # ============================================================
    print("\n[2/6] Melakukan feature engineering (Cyclic + Interaction)...")
    df = add_cyclic_features(df)
    df = add_interaction_features(df)
    print(f"      Jumlah kolom setelah engineering: {len(df.columns)}")

    # ============================================================
    # FASE KRITIS: Menghapus fitur deterministik yang menyebabkan
    # tautological learning (model reverse-engineers label rules).
    # status_kebakaran_sekitar adalah komponen langsung label Level 3.
    # ============================================================
    print("\n[3/6] Pre-processing: Menghapus kolom deterministik & data leakage...")
    LEAKAGE_COLS = [
        'status_kebakaran_sekitar',  # Deterministic: r6 = 3 jika ini == 1
        'Status_Kebakaran_Sekitar',  # Kemungkinan nama alternatif
    ]
    TEMPORAL_COLS = ['_tahun']  # Kolom bantu (bukan fitur prediktif)
    EXCLUDE_COLS  = [target_col] + LEAKAGE_COLS + TEMPORAL_COLS

    # Simpan kolom tahun sebelum dibuang (untuk temporal split)
    tahun_series = df['_tahun'].copy()

    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    X = X.select_dtypes(include=['number'])
    X = X.fillna(0)
    y = df[target_col]

    dropped_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    print(f"      Fitur leakage yang dibuang: {dropped_leakage if dropped_leakage else 'Tidak ada (sudah bersih)'}")
    print(f"      Total fitur untuk training: {X.shape[1]} kolom")
    print(f"      Distribusi kelas target:")
    for lvl, cnt in y.value_counts().sort_index().items():
        print(f"        Level {lvl}: {cnt:,} baris ({cnt/len(y)*100:.2f}%)")

    # ============================================================
    # FASE KRITIS: Temporal 3-Split
    # Pemisahan berdasarkan tahun BUKAN acak. Ini mencegah
    # temporal leakage (data 2025 bocor ke training set 2021-2023).
    # - Train:  2021-2023 (pola historis)
    # - Val:    2024      (tuning hyperparameter & pemilihan model)
    # - Test:   2025      (evaluasi akhir, hanya disentuh SATU KALI)
    # ============================================================
    print("\n[4/6] Membagi data secara temporal (3-Split)...")
    mask_train = tahun_series <= 2023
    mask_val   = tahun_series == 2024
    mask_test  = tahun_series == 2025

    X_train, y_train = X[mask_train].reset_index(drop=True), y[mask_train].reset_index(drop=True)
    X_val,   y_val   = X[mask_val].reset_index(drop=True),   y[mask_val].reset_index(drop=True)
    X_test,  y_test  = X[mask_test].reset_index(drop=True),  y[mask_test].reset_index(drop=True)

    total = len(X)
    print(f"      -> Train Set  (2021-2023): {len(X_train):>7,} baris ({len(X_train)/total*100:.1f}%)")
    print(f"      -> Val Set    (2024):      {len(X_val):>7,} baris ({len(X_val)/total*100:.1f}%)")
    print(f"      -> Test Set   (2025):      {len(X_test):>7,} baris ({len(X_test)/total*100:.1f}%)")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("[!] Salah satu split kosong! Pastikan dataset mencakup tahun 2021-2025.")
        print(f"    Rentang tahun ditemukan: {tahun_series[tahun_series > 0].unique().tolist()}")
        return

    # ============================================================
    # FASE KRITIS: Training Model Random Forest
    # class_weight='balanced' memberikan penalti lebih besar untuk
    # kesalahan pada kelas minoritas (Level 2 & 3) — kritis untuk
    # menghindari bias menuju kelas mayoritas (Level 0 & 1).
    # ============================================================
    print("\n[5/6] Training Random Forest Classifier (class_weight=balanced)...")
    model = RandomForestClassifier(
        n_estimators=200,          # Lebih banyak pohon untuk dataset besar
        max_depth=20,              # Batasi kedalaman untuk mengurangi overfitting
        min_samples_split=10,      # Minimal sampel per split (regularisasi)
        class_weight='balanced',   # Penanganan ketidakseimbangan kelas
        random_state=42,
        n_jobs=-1                  # Gunakan semua core CPU
    )
    model.fit(X_train, y_train)
    print("      Model selesai dilatih.")

    # ============================================================
    # FASE KRITIS: Evaluasi Terpisah pada 3 Set
    # TIDAK BOLEH menggunakan Test Set untuk keputusan apapun
    # selain pelaporan final. Semua keputusan tuning menggunakan
    # Validation Set (2024).
    # ============================================================
    print("\n[6/6] Evaluasi model pada Train / Validation / Test Set...")

    def evaluate_split(name: str, X_s, y_s) -> dict:
        y_pred = model.predict(X_s)
        acc    = accuracy_score(y_s, y_pred)
        f1_mac = f1_score(y_s, y_pred, average='macro', zero_division=0)
        report = classification_report(y_s, y_pred, output_dict=True, zero_division=0)
        report_str = classification_report(y_s, y_pred, zero_division=0)
        print(f"\n  [{name}]")
        print(f"  Akurasi    : {acc*100:.2f}%")
        print(f"  F1-Macro   : {f1_mac:.4f}")
        print(f"\n{report_str}")
        return {
            "accuracy":    round(acc, 6),
            "f1_macro":    round(f1_mac, 6),
            "n_samples":   len(y_s),
            "report":      report
        }

    print("-" * 65)
    train_metrics = evaluate_split("TRAIN SET  (2021-2023)", X_train, y_train)
    val_metrics   = evaluate_split("VALIDATION SET (2024)", X_val,   y_val)
    test_metrics  = evaluate_split("TEST SET   (2025)",     X_test,  y_test)
    print("-" * 65)

    # Simpan Model
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    # Simpan Metrik Lengkap ke JSON (termasuk semua 3 split)
    feature_importances = {
        feat: round(float(imp), 6)
        for feat, imp in zip(X_train.columns, model.feature_importances_)
    }
    # Urutkan berdasarkan importance tertinggi
    feature_importances = dict(
        sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    )

    metrics_data = {
        "split_scheme":      "temporal_3split",
        "train_years":       "2021-2023",
        "validation_year":   "2024",
        "test_year":         "2025",
        "features_used":     list(X_train.columns),
        "features_dropped":  dropped_leakage + TEMPORAL_COLS,
        "train_metrics":     train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics":      test_metrics,
        "feature_importances": feature_importances,
    }

    with open(metrics_output_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"\n  [OK] Model  disimpan di: {model_output_path.absolute()}")
    print(f"  [OK] Metrik disimpan di: {metrics_output_path.absolute()}")
    print(f"\n  RINGKASAN AKHIR:")
    print(f"  +---------------------+----------+----------+")
    print(f"  | Split               | Akurasi  | F1-Macro |")
    print(f"  +---------------------+----------+----------+")
    print(f"  | Train  (2021-2023)  | {train_metrics['accuracy']*100:6.2f}%  | {train_metrics['f1_macro']:.4f}   |")
    print(f"  | Val    (2024)       | {val_metrics['accuracy']*100:6.2f}%  | {val_metrics['f1_macro']:.4f}   |")
    print(f"  | Test   (2025)       | {test_metrics['accuracy']*100:6.2f}%  | {test_metrics['f1_macro']:.4f}   |")
    print(f"  +---------------------+----------+----------+")
    print("=" * 65)


if __name__ == "__main__":
    run_training()
"""
scripts/build_forecast_dataset.py
----------------------------------
Sprint 3: Membangun dataset forecasting untuk prediksi Danger_Level t+3 jam ke depan.

PARADIGMA BARU (Forecasting):
  INPUT  : Cuaca jam t, t-1, t-2, ... (kondisi yang diketahui SEKARANG)
  OUTPUT : Danger_Level jam t+3 (bahaya 3 jam ke depan — belum diketahui)

Mengapa ini berbeda dari model sebelumnya?
  Model sebelumnya: y = f(X_t)  -> relasi deterministik 100%
  Model ini       : y = f(X_t, X_t-1, X_t-2, ...) -> ada ketidakpastian alami cuaca

URUTAN OPERASI (WAJIB, tidak boleh ditukar):
  1. Urutkan data [Pos, Timestamp]
  2. Buat label target masa depan (.shift(-HORIZON) per grup pos)
  3. Buat lag features (.shift(+N) per grup pos)
  4. Buat rolling features (dengan .shift(1) anti-leakage di dalam rolling)
  5. Hapus NaN
  6. Split Train/Val/Test berdasarkan TAHUN
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import Config


# ============================================================
# KONFIGURASI SPRINT
# Sesuaikan HORIZON untuk mengubah target prediksi.
# HORIZON=3 berarti model memprediksi bahaya 3 jam ke depan.
# ============================================================
HORIZON_HOURS = 3  # Jam ke depan yang ingin diprediksi

# Fitur cuaca utama yang akan dibuat lag dan rolling-nya
WEATHER_KEYS_FOR_LAG = [
    'Angin Kencang (km/h)',
    'Kecepatan Angin (km/h)',
    'Suhu Terasa (C)',
    'Curah Hujan (mm)',
    'Kode Cuaca WMO',
]

# Jendela waktu rolling (dalam jam)
ROLLING_WINDOWS = [3, 6, 12]

# Jendela waktu lag (dalam jam)
LAG_STEPS = [1, 3, 6]


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


def build_forecast_dataset():
    print("=" * 65)
    print(f"MEMBANGUN DATASET FORECASTING (Horizon: t+{HORIZON_HOURS} jam)")
    print("=" * 65)

    # Cari dataset terintegrasi
    t_w_start = Config.WEATHER_HISTORICAL_START[:4]
    t_w_end   = Config.WEATHER_HISTORICAL_END[:4]
    filename  = f"dataset_integrated_lawu_{t_w_start}_{t_w_end}.csv"

    possible_paths = [
        Config.DATA_CURATED_DIR / filename,
        Config.ROOT_DIR / "DATA" / "curated" / filename,
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)
    if data_path is None:
        print("[!] Dataset terintegrasi tidak ditemukan.")
        return

    output_filename = f"dataset_forecast_lawu_t{HORIZON_HOURS}h_{t_w_start}_{t_w_end}.csv"
    output_paths = [
        Config.DATA_CURATED_DIR / output_filename,
        Config.ROOT_DIR / "DATA" / "curated" / output_filename,
    ]
    # Gunakan path yang directory-nya sudah ada
    output_path = next((p for p in output_paths if p.parent.exists()), output_paths[0])

    # ----------------------------------------------------------
    # LANGKAH 1: Muat Dataset
    # ----------------------------------------------------------
    print(f"\n[1/6] Memuat dataset: {data_path.name}")
    df = pd.read_csv(data_path)
    print(f"      Baris awal  : {len(df):,}")
    print(f"      Kolom awal  : {len(df.columns)}")

    # Tambahkan fitur siklis waktu
    df = add_cyclic_features(df)

    # Pastikan Timestamp bisa di-parse
    datetime_col = next((c for c in ['Timestamp', 'Datetime'] if c in df.columns), None)
    if datetime_col is None:
        print("[!] Kolom Timestamp tidak ditemukan. Proses dihentikan.")
        return

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

    # ----------------------------------------------------------
    # LANGKAH 2: Urutkan berdasarkan [Pos, Timestamp]
    # FASE KRITIS: Urutan ini WAJIB dilakukan sebelum shift/rolling.
    # Jika tidak, label dan lag dari pos berbeda akan tercampur.
    # ----------------------------------------------------------
    print("\n[2/6] Mengurutkan data berdasarkan [Lat, Lon, Timestamp]...")
    df = df.sort_values(['Lat', 'Lon', datetime_col]).reset_index(drop=True)

    pos_groups = df.groupby(['Lat', 'Lon'])
    n_pos = pos_groups.ngroups
    print(f"      Jumlah pos unik: {n_pos}")

    # ----------------------------------------------------------
    # LANGKAH 3: Buat Label Target Forecasting
    # FASE KRITIS: .shift(-HORIZON) menggeser nilai Danger_Level
    # dari baris berikutnya ke baris saat ini. Ini berarti:
    #   Baris jam 10.00 -> Danger_Level jam 13.00 (3 jam ke depan)
    # .shift(-N) WAJIB dilakukan per grup pos agar label dari
    # Pos A tidak bocor ke baris terakhir Pos B.
    # ----------------------------------------------------------
    print(f"\n[3/6] Membuat label target t+{HORIZON_HOURS}h...")
    target_col = f'Danger_Level_t{HORIZON_HOURS}h'
    df[target_col] = (
        df.groupby(['Lat', 'Lon'])['Danger_Level']
        .shift(-HORIZON_HOURS)
    )

    # ----------------------------------------------------------
    # LANGKAH 4: Buat Lag Features
    # FASE KRITIS: .shift(+N) mengambil nilai N jam yang LALU.
    # Ini adalah kondisi cuaca yang diketahui model saat prediksi.
    # Dibuat per grup pos agar lag dari pos berbeda tidak tercampur.
    # ----------------------------------------------------------
    print(f"\n[4/6] Membuat lag features {LAG_STEPS}h untuk {len(WEATHER_KEYS_FOR_LAG)} variabel...")
    lag_cols_created = []
    for col in WEATHER_KEYS_FOR_LAG:
        if col not in df.columns:
            print(f"      [!] Kolom '{col}' tidak ada, dilewati.")
            continue
        for lag in LAG_STEPS:
            new_col = f'{col}_lag_{lag}h'
            df[new_col] = df.groupby(['Lat', 'Lon'])[col].shift(lag)
            lag_cols_created.append(new_col)

    print(f"      Total lag features dibuat: {len(lag_cols_created)}")

    # ----------------------------------------------------------
    # LANGKAH 5: Buat Rolling Window Features
    # FASE KRITIS: .shift(1) di dalam rolling adalah penjaga
    # utama anti-look-ahead bias. Tanpa .shift(1), rolling mean
    # pada jam t akan menyertakan nilai cuaca jam t itu sendiri —
    # bukan masalah untuk data saat ini, tapi saat training model
    # ini menciptakan kebocoran implisit terhadap nilai target.
    # Dengan .shift(1): rolling mean jam t = rata-rata jam t-1 sd t-W.
    # ----------------------------------------------------------
    print(f"\n[5/6] Membuat rolling features (windows: {ROLLING_WINDOWS}h)...")
    rolling_cols_created = []
    roll_keys = ['Angin Kencang (km/h)', 'Curah Hujan (mm)', 'Suhu Terasa (C)']
    for col in roll_keys:
        if col not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            # Rolling Mean (rata-rata bergulir — menangkap tren cuaca)
            col_mean = f'{col}_roll_mean_{w}h'
            df[col_mean] = (
                df.groupby(['Lat', 'Lon'])[col]
                .transform(lambda x, ww=w: x.shift(1).rolling(ww, min_periods=1).mean())
            )
            rolling_cols_created.append(col_mean)

            # Rolling Max (puncak tertinggi — penting untuk mendeteksi eskalasi badai)
            col_max = f'{col}_roll_max_{w}h'
            df[col_max] = (
                df.groupby(['Lat', 'Lon'])[col]
                .transform(lambda x, ww=w: x.shift(1).rolling(ww, min_periods=1).max())
            )
            rolling_cols_created.append(col_max)

    # Delta features (laju perubahan — seberapa cepat cuaca berubah)
    print("      Membuat delta features (laju perubahan 3h)...")
    delta_keys = ['Angin Kencang (km/h)', 'Suhu Terasa (C)', 'Curah Hujan (mm)']
    delta_cols_created = []
    for col in delta_keys:
        if col in df.columns and f'{col}_lag_3h' in df.columns:
            delta_col = f'delta_{col.split(" (")[0].replace(" ", "_").lower()}_3h'
            df[delta_col] = df[col] - df[f'{col}_lag_3h']
            delta_cols_created.append(delta_col)

    # Interaction features (identik dengan model sebelumnya)
    if 'Suhu Terasa (C)' in df.columns and 'Angin Kencang (km/h)' in df.columns:
        df['compound_cold_wind'] = df['Suhu Terasa (C)'] * df['Angin Kencang (km/h)']
    if 'FRP_Terdekat_MW' in df.columns and 'Jarak_Titik_Api_Terdekat_KM' in df.columns:
        df['fire_proximity_index'] = (
            df['FRP_Terdekat_MW'] / (df['Jarak_Titik_Api_Terdekat_KM'] + 0.1)
        )

    print(f"      Total rolling features: {len(rolling_cols_created)}")
    print(f"      Total delta features  : {len(delta_cols_created)}")

    # ----------------------------------------------------------
    # LANGKAH 6: Hapus Baris NaN dan Simpan
    # Baris pertama dan terakhir setiap grup akan memiliki NaN
    # akibat proses shift. Ini harus dihapus agar training bersih.
    # ----------------------------------------------------------
    print(f"\n[6/6] Menghapus baris NaN dan menyimpan dataset...")
    n_before = len(df)
    df = df.dropna(subset=[target_col])  # Wajib: baris tanpa label target dibuang
    # Hapus baris dengan NaN di lag features kritis
    critical_lag_cols = [c for c in lag_cols_created if '_lag_3h' in c or '_lag_1h' in c]
    if critical_lag_cols:
        df = df.dropna(subset=critical_lag_cols[:5])  # Cukup cek subset agar efisien
    df[target_col] = df[target_col].astype(int)

    n_after = len(df)
    print(f"      Baris sebelum hapus NaN: {n_before:,}")
    print(f"      Baris setelah hapus NaN : {n_after:,}")
    print(f"      Baris dibuang           : {n_before - n_after:,} ({(n_before-n_after)/n_before*100:.1f}%)")

    # Tampilkan distribusi kelas target
    print(f"\n      Distribusi label target '{target_col}':")
    vc = df[target_col].value_counts().sort_index()
    for lvl, cnt in vc.items():
        print(f"        Level {lvl}: {cnt:,} baris ({cnt/len(df)*100:.2f}%)")

    # Tampilkan ringkasan split temporal
    print("\n      Estimasi jumlah baris per temporal split:")
    for yr_label, mask in [
        ('Train (2021-2023)', df['_tahun'] <= 2023),
        ('Val   (2024)',      df['_tahun'] == 2024),
        ('Test  (2025)',      df['_tahun'] == 2025),
    ]:
        cnt = mask.sum()
        print(f"        {yr_label}: {cnt:,} baris ({cnt/len(df)*100:.1f}%)")

    # Simpan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  [OK] Dataset forecasting disimpan: {output_path}")
    print(f"  [OK] Ukuran file: {size_mb:.1f} MB")
    print(f"  [OK] Total kolom: {len(df.columns)} (termasuk {len(lag_cols_created)} lag, "
          f"{len(rolling_cols_created)} rolling, {len(delta_cols_created)} delta)")
    print("=" * 65)


if __name__ == "__main__":
    build_forecast_dataset()

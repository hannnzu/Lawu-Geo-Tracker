"""
scripts/build_multi_horizon_datasets.py
---------------------------------------
Sprint 4 — Tahap 1: Membangun dataset forecasting untuk multi-horizon (t+1h, t+3h, t+6h).

Targets:
  - Danger_Level_t1h: Prediksi bahaya 1 jam ke depan
  - Danger_Level_t3h: Prediksi bahaya 3 jam ke depan
  - Danger_Level_t6h: Prediksi bahaya 6 jam ke depan
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import Config

# Parameter runtun waktu
WEATHER_KEYS = [
    'Angin Kencang (km/h)',
    'Kecepatan Angin (km/h)',
    'Suhu Terasa (C)',
    'Curah Hujan (mm)',
    'Kode Cuaca WMO',
]

ROLLING_WINDOWS = [3, 6, 12]
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


def build_multi_horizon_dataset():
    print("=" * 65)
    print("MEMBANGUN DATASET FORECASTING MULTI-HORIZON (t+1h, t+3h, t+6h)")
    print("=" * 65)

    # Identifikasi berkas dataset asal
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

    output_filename = f"dataset_forecast_lawu_multi_{t_w_start}_{t_w_end}.csv"
    output_paths = [
        Config.DATA_CURATED_DIR / output_filename,
        Config.ROOT_DIR / "DATA" / "curated" / output_filename,
    ]
    output_path = next((p for p in output_paths if p.parent.exists()), output_paths[0])

    # 1. Muat Dataset
    print(f"\n[1/6] Memuat dataset: {data_path.name}")
    df = pd.read_csv(data_path)
    print(f"      Baris awal: {len(df):,}")

    df = add_cyclic_features(df)

    datetime_col = next((c for c in ['Timestamp', 'Datetime'] if c in df.columns), None)
    if datetime_col is None:
        print("[!] Kolom Timestamp tidak ditemukan.")
        return
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

    # 2. Urutkan berdasarkan [Lat, Lon, Timestamp] per pos (ANTI-LEAKAGE)
    print("\n[2/6] Mengurutkan data berdasarkan [Lat, Lon, Timestamp]...")
    df = df.sort_values(['Lat', 'Lon', datetime_col]).reset_index(drop=True)

    # 3. Buat Label Target Forecasting Multi-Horizon (t+1h, t+3h, t+6h)
    print("\n[3/6] Membuat label target multi-horizon...")
    for h in [1, 3, 6]:
        target_col = f'Danger_Level_t{h}h'
        df[target_col] = df.groupby(['Lat', 'Lon'])['Danger_Level'].shift(-h)
        print(f"      [OK] Target '{target_col}' dibuat.")

    # 4. Buat Lag Features
    print(f"\n[4/6] Membuat lag features {LAG_STEPS}h...")
    lag_cols = []
    for col in WEATHER_KEYS:
        if col not in df.columns:
            continue
        for lag in LAG_STEPS:
            new_col = f'{col}_lag_{lag}h'
            df[new_col] = df.groupby(['Lat', 'Lon'])[col].shift(lag)
            lag_cols.append(new_col)

    # 5. Buat Rolling Window Features (menggunakan .shift(1) agar bebas look-ahead bias)
    print(f"\n[5/6] Membuat rolling window features {ROLLING_WINDOWS}h...")
    roll_keys = ['Angin Kencang (km/h)', 'Curah Hujan (mm)', 'Suhu Terasa (C)']
    roll_cols = []
    for col in roll_keys:
        if col not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            # Mean
            col_mean = f'{col}_roll_mean_{w}h'
            df[col_mean] = df.groupby(['Lat', 'Lon'])[col].transform(lambda x, ww=w: x.shift(1).rolling(ww, min_periods=1).mean())
            roll_cols.append(col_mean)
            # Max
            col_max = f'{col}_roll_max_{w}h'
            df[col_max] = df.groupby(['Lat', 'Lon'])[col].transform(lambda x, ww=w: x.shift(1).rolling(ww, min_periods=1).max())
            roll_cols.append(col_max)

    # Delta & Interaksi
    print("      Membuat delta & interaction features...")
    delta_cols = []
    for col in roll_keys:
        if col in df.columns and f'{col}_lag_3h' in df.columns:
            delta_col = f'delta_{col.split(" (")[0].replace(" ", "_").lower()}_3h'
            df[delta_col] = df[col] - df[f'{col}_lag_3h']
            delta_cols.append(delta_col)

    df['compound_cold_wind'] = df['Suhu Terasa (C)'] * df['Angin Kencang (km/h)']
    df['fire_proximity_index'] = df['FRP_Terdekat_MW'] / (df['Jarak_Titik_Api_Terdekat_KM'] + 0.1)

    # 6. Pembersihan Baris NaN & Simpan
    print("\n[6/6] Menghapus baris NaN untuk target terpanjang (t+6h)...")
    # Jika Danger_Level_t6h bernilai NaN (yaitu 6 jam terakhir per pos), kita buang
    n_before = len(df)
    df = df.dropna(subset=['Danger_Level_t6h'])
    
    # Hapus juga baris yang tidak memiliki lag awal (6 jam pertama per pos)
    if lag_cols:
        df = df.dropna(subset=[c for c in lag_cols if '_lag_6h' in c][:3])

    for h in [1, 3, 6]:
        df[f'Danger_Level_t{h}h'] = df[f'Danger_Level_t{h}h'].astype(int)

    n_after = len(df)
    print(f"      Baris sebelum : {n_before:,}")
    print(f"      Baris setelah : {n_after:,}")
    print(f"      Baris dibuang : {n_before - n_after:,} ({(n_before - n_after)/n_before*100:.2f}%)")

    # Simpan dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  [OK] Dataset Multi-Horizon disimpan: {output_path}")
    print(f"  [OK] Ukuran file: {size_mb:.1f} MB")
    print(f"  [OK] Total kolom: {len(df.columns)}")
    print("=" * 65)


if __name__ == "__main__":
    build_multi_horizon_dataset()

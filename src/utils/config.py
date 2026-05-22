"""
utils/config.py
----------------
Memuat konfigurasi dari environment variables dan file config/.
Open-Meteo Archive API tidak membutuhkan API Key.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Muat .env untuk parameter opsional (WEATHER_START_DATE, WEATHER_END_DATE)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


class Config:
    """Konfigurasi terpusat seluruh pipeline."""

    # --- Direktori (relatif dari root project) ---
    ROOT_DIR:         Path = Path(__file__).resolve().parents[2]
    DATA_RAW_DIR:     Path = ROOT_DIR / "data" / "raw"
    DATA_PROC_DIR:    Path = ROOT_DIR / "data" / "processed"
    DATA_CURATED_DIR: Path = ROOT_DIR / "data" / "curated"
    OUTPUT_DIR:       Path = ROOT_DIR / "output"
    LOGS_DIR:         Path = ROOT_DIR / "logs"
    CACHE_DIR:        Path = ROOT_DIR / "cache"

    # --- Parameter Pipeline Cuaca Historis (Open-Meteo, hourly, 5 tahun) ---
    WEATHER_HISTORICAL_START: str = os.getenv("WEATHER_START_DATE", "2021-01-01")
    WEATHER_HISTORICAL_END:   str = os.getenv("WEATHER_END_DATE",   "2025-12-31")

    # --- Koordinat Referensi Gunung Lawu ---
    LAWU_PUNCAK_LAT: float = -7.627324
    LAWU_PUNCAK_LON: float = 111.194387

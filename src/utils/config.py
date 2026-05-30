"""
utils/config.py
----------------
Memuat konfigurasi dari environment variables dan file config/.
Mencakup parameter untuk:
  - Open-Meteo Archive API (cuaca historis, tanpa API Key)
  - NASA FIRMS API (data kebakaran hutan, butuh MAP_KEY gratis)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Muat .env untuk semua parameter pipeline
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

    # --- Koordinat & Bounding Box Gunung Lawu ---
    LAWU_PUNCAK_LAT: float = -7.627324
    LAWU_PUNCAK_LON: float = 111.194387
    # Bounding box format: "west,south,east,north" (untuk NASA FIRMS)
    LAWU_BBOX:       str   = "111.10,-7.75,111.25,-7.55"

    # --- Parameter Pipeline Bencana Alam (NASA FIRMS) ---
    # MAP_KEY gratis: https://firms.modaps.eosdis.nasa.gov/api/map_key/
    FIRMS_MAP_KEY:    str = os.getenv("FIRMS_MAP_KEY",    "")
    FIRMS_START_DATE: str = os.getenv("FIRMS_START_DATE", "2021-01-01")
    FIRMS_END_DATE:   str = os.getenv("FIRMS_END_DATE",   "2025-12-31")

    # --- Database (Aiven PostgreSQL + TimescaleDB) ---
    AIVEN_DATABASE_URL: str = os.getenv("AIVEN_DATABASE_URL", "")

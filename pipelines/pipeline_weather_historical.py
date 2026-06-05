"""
pipeline_weather_historical.py
-------------------------------
Entry point: Pipeline data cuaca historis per jam (hourly) Gunung Lawu.

Sumber data   : Open-Meteo Archive API (gratis, tanpa API Key)
Granularitas  : Per jam (hourly) - 24 titik waktu per hari
Lokasi        : 19 titik - 3 basecamp, 5 pos per jalur (3 jalur), puncak
Rentang data  : 5 tahun (2021-2025)
Estimasi baris: ~831.600 baris (19 lokasi x 5 tahun x 8.760 jam)
Data output   : data/processed/weather/cuaca_hourly_lawu_<tahun_mulai>_<tahun_selesai>.csv

Fitur:
  - Retry otomatis dengan exponential backoff saat kena rate limit (HTTP 429)
  - Resume: lokasi yang sudah tersimpan di CSV tidak di-request ulang
  - Append mode: data yang sudah ada tidak tertimpa

Cara jalankan:
    python pipeline_weather_historical.py

Jika terputus di tengah, jalankan ulang -- pipeline akan lanjut dari lokasi
yang belum tersimpan secara otomatis.
"""

from src.ingestion.open_meteo import (
    fetch_all_locations,
    save_to_csv,
    get_existing_locations,
    LOKASI_GUNUNG_LAWU,
)
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("pipeline_weather_historical", log_dir=Config.LOGS_DIR)

# ==========================================
# Konfigurasi Rentang Waktu
# ==========================================
START_DATE = Config.WEATHER_HISTORICAL_START  # default: "2021-01-01"
END_DATE   = Config.WEATHER_HISTORICAL_END    # default: "2025-12-31"

_tahun_mulai   = START_DATE[:4]
_tahun_selesai = END_DATE[:4]
OUTPUT_CSV = Config.DATA_PROC_DIR / "weather" / f"cuaca_hourly_lawu_{_tahun_mulai}_{_tahun_selesai}.csv"


def run() -> None:
    logger.info("=" * 60)
    logger.info("PIPELINE CUACA HOURLY GUNUNG LAWU - Open-Meteo Archive")
    logger.info("=" * 60)
    logger.info(f"Periode  : {START_DATE}  s/d  {END_DATE}  (5 tahun)")
    logger.info(f"Lokasi   : {len(LOKASI_GUNUNG_LAWU)} titik "
                "(3 basecamp + 5 pos x 3 jalur + puncak)")
    logger.info(f"Estimasi : ~{len(LOKASI_GUNUNG_LAWU) * 5 * 8760:,} baris")
    logger.info(f"Output   : {OUTPUT_CSV}")
    logger.info("")

    # --- Cek lokasi yang sudah tersimpan (fitur resume) ---
    existing = get_existing_locations(OUTPUT_CSV)
    if existing:
        logger.info(f"[RESUME] {len(existing)} lokasi sudah tersimpan di CSV, akan di-skip:")
        for nama in sorted(existing):
            logger.info(f"         [OK] {nama}")
        logger.info("")

    # 1. EXTRACT + TRANSFORM
    logger.info("[1/2] Extract & Transform - mengambil data hourly lokasi yang belum ada...")
    logger.info("      (retry otomatis jika kena rate limit - max wait 480s per attempt)")
    all_rows, failed = fetch_all_locations(
        start_date=START_DATE,
        end_date=END_DATE,
        delay_sec=3.0,           # 3s antar request (lebih aman untuk free tier)
        skip_existing=existing,
    )

    # 2. LOAD (append ke CSV yang sudah ada)
    if all_rows:
        logger.info(f"\n[2/2] Load - menyimpan {len(all_rows):,} baris ke CSV (append)...")
        save_to_csv(all_rows, OUTPUT_CSV, mode="a")
    else:
        logger.info("\n[2/2] Load - tidak ada baris baru untuk disimpan.")

    # --- Laporan akhir ---
    logger.info("")
    if failed:
        logger.warning(f"[!] {len(failed)} lokasi GAGAL diambil:")
        for nama in failed:
            logger.warning(f"  - {nama}")
        logger.warning("Jalankan ulang pipeline untuk retry lokasi yang gagal.")
    else:
        logger.info("[OK] Semua lokasi berhasil.")

    logger.info("Pipeline selesai.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()

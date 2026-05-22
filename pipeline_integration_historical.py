"""
pipeline_integration_historical.py
-----------------------------------
Entry point: Pipeline Integrasi Data Spatiotemporal Gunung Lawu.
Menggabungkan data cuaca per jam (processed) dengan data titik api NASA FIRMS (raw)
menjadi satu dataset terpadu (curated) untuk pemodelan Machine Learning.

Cara jalankan:
    python pipeline_integration_historical.py
"""

from pathlib import Path
from src.transformation.integration import integrate_weather_and_disaster
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("pipeline_integration_historical", log_dir=Config.LOGS_DIR)

# Ambang batas jarak bahaya kebakaran terdekat (KM)
# Jika ada titik api aktif pada hari yang sama dalam radius ini, pos ditandai berisiko.
DISTANCE_THRESHOLD_KM = 3.0


def run() -> None:
    logger.info("=" * 60)
    logger.info("PIPELINE INTEGRASI DATA SPATIOTEMPORAL GUNUNG LAWU")
    logger.info("=" * 60)

    # 1. Konfigurasi Path Input & Output
    tahun_mulai_weather = Config.WEATHER_HISTORICAL_START[:4]
    tahun_selesai_weather = Config.WEATHER_HISTORICAL_END[:4]
    weather_csv = (
        Config.DATA_PROC_DIR / "weather"
        / f"cuaca_hourly_lawu_{tahun_mulai_weather}_{tahun_selesai_weather}.csv"
    )

    tahun_mulai_firms = Config.FIRMS_START_DATE[:4]
    tahun_selesai_firms = Config.FIRMS_END_DATE[:4]
    disaster_csv = (
        Config.DATA_RAW_DIR / "disaster"
        / f"titik_api_lawu_{tahun_mulai_firms}_{tahun_selesai_firms}.csv"
    )

    output_csv = (
        Config.DATA_CURATED_DIR
        / f"dataset_integrated_lawu_{tahun_mulai_weather}_{tahun_selesai_weather}.csv"
    )

    logger.info(f"Weather Input  : {weather_csv}")
    logger.info(f"Disaster Input : {disaster_csv}")
    logger.info(f"Target Output  : {output_csv}")
    logger.info(f"Radius Bahaya  : {DISTANCE_THRESHOLD_KM} KM")
    logger.info("")

    # 2. Validasi File Input
    if not weather_csv.exists():
        logger.error(f"[!] File cuaca historis tidak ditemukan!")
        logger.error(f"    Pastikan Anda sudah menjalankan: python pipeline_weather_historical.py")
        return

    if not disaster_csv.exists():
        logger.error(f"[!] File titik api bencana tidak ditemukan!")
        logger.error(f"    Pastikan Anda sudah menjalankan: python pipeline_disaster_historical.py")
        return

    # 3. Jalankan Integrasi
    logger.info("[1/2] Memulai integrasi spatiotemporal...")
    logger.info("      (Menghitung jarak Haversine untuk pos pendakian terhadap titik api harian)")
    
    success = integrate_weather_and_disaster(
        weather_csv_path=weather_csv,
        disaster_csv_path=disaster_csv,
        output_csv_path=output_csv,
        distance_threshold_km=DISTANCE_THRESHOLD_KM
    )

    if success:
        logger.info("[2/2] Integrasi selesai dengan sukses!")
        logger.info(f"      File curated tersimpan di: {output_csv}")
    else:
        logger.error("[!] Integrasi GAGAL. Silakan periksa log di atas.")

    logger.info("=" * 60)


if __name__ == "__main__":
    run()

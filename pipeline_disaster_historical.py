"""
pipeline_disaster_historical.py
--------------------------------
Entry point: Pipeline data kebakaran hutan historis Gunung Lawu.

Sumber data  : NASA FIRMS API (VIIRS Suomi-NPP Standard Processing)
Sensor       : VIIRS_SNPP_SP (resolusi 375m, data tervalidasi)
Lokasi       : Bounding box Gunung Lawu (111.10,-7.75,111.25,-7.55)
Rentang data : 5 tahun (2021-2025)
Data output  : data/raw/disaster/titik_api_lawu_<tahun_mulai>_<tahun_selesai>.csv

Fitur:
  - Validasi MAP_KEY sebelum memulai (dengan panduan registrasi jika kosong)
  - Resume otomatis dari tanggal terakhir yang tersimpan di CSV
  - Append mode agar data lama tidak tertimpa
  - Retry otomatis saat kena rate limit (HTTP 429)

Cara jalankan:
    python pipeline_disaster_historical.py

Prasyarat:
    1. Daftar MAP_KEY gratis di: https://firms.modaps.eosdis.nasa.gov/api/map_key/
    2. Tambahkan ke .env: FIRMS_MAP_KEY=<32-karakter-key>
"""

from src.ingestion.nasa_firms import (
    fetch_all_fire_data,
    save_to_csv,
    get_last_saved_date,
)
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("pipeline_disaster_historical", log_dir=Config.LOGS_DIR)

# ==========================================
# Konfigurasi
# ==========================================
MAP_KEY    = Config.FIRMS_MAP_KEY
START_DATE = Config.FIRMS_START_DATE   # default: "2021-01-01"
END_DATE   = Config.FIRMS_END_DATE     # default: "2025-12-31"

_tahun_mulai   = START_DATE[:4]
_tahun_selesai = END_DATE[:4]
OUTPUT_CSV = (
    Config.DATA_RAW_DIR / "disaster"
    / f"titik_api_lawu_{_tahun_mulai}_{_tahun_selesai}.csv"
)


def run() -> None:
    logger.info("=" * 60)
    logger.info("PIPELINE BENCANA ALAM (KEBAKARAN) GUNUNG LAWU - NASA FIRMS")
    logger.info("=" * 60)

    # --- Validasi MAP_KEY ---
    if not MAP_KEY:
        logger.error("[!] FIRMS_MAP_KEY tidak ditemukan di .env!")
        logger.error("    Daftar gratis di: https://firms.modaps.eosdis.nasa.gov/api/map_key/")
        logger.error("    Tambahkan ke .env: FIRMS_MAP_KEY=<32-karakter-key>")
        return

    if len(MAP_KEY) != 32:
        logger.error(f"[!] FIRMS_MAP_KEY tidak valid (panjang {len(MAP_KEY)}, harusnya 32 karakter).")
        return

    logger.info(f"MAP_KEY  : {MAP_KEY[:4]}...{MAP_KEY[-4:]}  (tersembunyi)")
    logger.info(f"Periode  : {START_DATE}  s/d  {END_DATE}  (5 tahun)")
    logger.info(f"Sensor   : VIIRS_SNPP_SP (resolusi 375m)")
    logger.info(f"Area     : {Config.LAWU_BBOX}  (Gunung Lawu)")
    logger.info(f"Output   : {OUTPUT_CSV}")
    logger.info("")

    # --- Cek resume ---
    last_date = get_last_saved_date(OUTPUT_CSV)
    if last_date:
        logger.info(f"[RESUME] Data sudah ada sampai tanggal: {last_date}")
        logger.info(f"         Pipeline akan lanjut dari tanggal berikutnya.")
        logger.info("")

    # --- EXTRACT ---
    logger.info("[1/2] Extract - mengambil data titik api dari NASA FIRMS...")
    logger.info("      (iterasi per 5 hari, ~365 batch untuk 5 tahun)")
    all_rows = fetch_all_fire_data(
        map_key=MAP_KEY,
        start_date=START_DATE,
        end_date=END_DATE,
        resume_from=last_date,
        delay_sec=1.0,
    )

    # --- LOAD ---
    if all_rows:
        logger.info(f"\n[2/2] Load - menyimpan {len(all_rows):,} baris ke CSV...")
        save_to_csv(all_rows, OUTPUT_CSV, mode="a")
    else:
        logger.info("\n[2/2] Load - tidak ada data baru untuk disimpan.")
        logger.info("      (mungkin tidak ada kebakaran terdeteksi, atau semua sudah di-resume)")

    logger.info("")
    logger.info("Pipeline selesai.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()

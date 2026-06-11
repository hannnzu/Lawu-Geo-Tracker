"""
pipeline_3_loading.py
--------------------------------
Fase LOAD: Mengunggah semua data olahan ke Database Aiven.
1. Membuat/Memastikan Schema Database
2. Memuat tabel dimensi (Pos, Titik Api, Jalur GPX)
3. Memuat tabel fakta (Cuaca Integrated)
4. Melakukan update server-side untuk Danger Level
5. Menjalankan verifikasi akhir
"""

import sys
import csv
from sqlalchemy import text
from src.utils.config import Config
from src.utils.logger import get_logger

from src.loading.aiven_loader import (
    get_engine, ensure_schema, load_pos_pendakian, load_titik_api,
    load_cuaca_integrated, update_danger_level_in_db, load_jalur_pendakian
)

logger = get_logger("pipeline_3_loading", log_dir=Config.LOGS_DIR)

def run_verification(engine):
    logger.info("\n--- [6/6] VERIFIKASI DATA DATABASE ---")
    with engine.connect() as conn:
        cnt_pos = conn.execute(text("SELECT COUNT(*) FROM pos_pendakian")).scalar()
        cnt_api = conn.execute(text("SELECT COUNT(*) FROM titik_api")).scalar()
        cnt_jalur = conn.execute(text("SELECT COUNT(*) FROM jalur_pendakian")).scalar()
        cnt_cuaca = conn.execute(text("SELECT COUNT(*) FROM cuaca_integrated")).scalar()
        cnt_kebakaran = conn.execute(text("SELECT COUNT(*) FROM cuaca_integrated WHERE status_kebakaran_sekitar = 1")).scalar()
        
        logger.info(f"Total baris 'pos_pendakian'   : {cnt_pos}")
        logger.info(f"Total baris 'titik_api'       : {cnt_api}")
        logger.info(f"Total baris 'jalur_pendakian' : {cnt_jalur}")
        logger.info(f"Total baris 'cuaca_integrated': {cnt_cuaca:,}")
        logger.info(f"Total baris berisiko api      : {cnt_kebakaran:,}")

        # Cek anomali danger level
        anomali = conn.execute(text("SELECT COUNT(*) FROM cuaca_integrated WHERE status_kebakaran_sekitar = 1 AND danger_level != 3")).scalar()
        if anomali == 0:
            logger.info("Integritas Danger Level: LULUS (Semua kebakaran sekitar = Level 3)")
        else:
            logger.warning(f"Integritas Danger Level: GAGAL ({anomali} anomali ditemukan)")

def run():
    logger.info("=" * 60)
    logger.info("PIPELINE 3: LOADING KE AIVEN (LOAD) DIMULAI")
    logger.info("=" * 60)

    db_url = Config.AIVEN_DATABASE_URL
    if not db_url:
        logger.error("[!] AIVEN_DATABASE_URL tidak ditemukan di file .env!")
        sys.exit(1)

    # Path Input
    t_f_start, t_f_end = Config.FIRMS_START_DATE[:4], Config.FIRMS_END_DATE[:4]
    t_w_start, t_w_end = Config.WEATHER_HISTORICAL_START[:4], Config.WEATHER_HISTORICAL_END[:4]
    
    disaster_csv = Config.DATA_RAW_DIR / "disaster" / f"titik_api_lawu_{t_f_start}_{t_f_end}.csv"
    cuaca_integrated_csv = Config.DATA_CURATED_DIR / f"dataset_integrated_lawu_{t_w_start}_{t_w_end}.csv"
    gpx_csv = Config.DATA_PROC_DIR / "geospatial" / "jalur_pendakian_processed.csv"

    if not cuaca_integrated_csv.exists() or not gpx_csv.exists() or not disaster_csv.exists():
        logger.error("[!] File input tidak lengkap. Pastikan Pipeline 1 dan 2 sudah dijalankan.")
        sys.exit(1)

    try:
        engine = get_engine(db_url)
        
        logger.info("--- [1/6] Memastikan Skema Database ---")
        ensure_schema(engine)

        logger.info("--- [2/6] Memuat Dimensi: Pos Pendakian ---")
        load_pos_pendakian(engine)

        logger.info("--- [3/6] Memuat Dimensi: Titik Api ---")
        load_titik_api(engine, disaster_csv)

        logger.info("--- [4/6] Memuat Dimensi: Jalur Pendakian (GPX) ---")
        with open(gpx_csv, mode='r', encoding='utf-8') as f:
            gpx_data = list(csv.DictReader(f))
            load_jalur_pendakian(engine, gpx_data)

        logger.info("--- [5/6] Memuat Fakta: Cuaca & Update Danger Level ---")
        load_cuaca_integrated(engine, cuaca_integrated_csv)
        updated_count = update_danger_level_in_db(engine)
        logger.info(f"Berhasil update {updated_count:,} baris danger_level di server.")

        run_verification(engine)

        logger.info("=" * 60)
        logger.info("PIPELINE 3 SELESAI DENGAN SUKSES!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"[!] Terjadi kesalahan saat proses Aiven: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run()
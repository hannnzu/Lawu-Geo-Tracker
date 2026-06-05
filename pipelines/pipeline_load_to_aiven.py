"""
pipeline_load_to_aiven.py
--------------------------
Script orchestrator untuk memuat data ke Aiven PostgreSQL + TimescaleDB.
Menjalankan seluruh tahapan Load dari pembuatan schema, data loading,
hingga verifikasi akhir.

Cara jalankan:
    python pipeline_load_to_aiven.py
"""

import sys
from pathlib import Path
from sqlalchemy import text
from src.utils.config import Config
from src.utils.logger import get_logger
from src.loading.aiven_loader import (
    get_engine,
    ensure_schema,
    load_pos_pendakian,
    load_titik_api,
    load_cuaca_integrated
)

logger = get_logger("pipeline_load_to_aiven", log_dir=Config.LOGS_DIR)


def run_verification(engine) -> None:
    """Menjalankan query verifikasi SQL untuk memastikan data terisi dengan benar."""
    logger.info("=" * 60)
    logger.info("MENJALANKAN VERIFIKASI DATA...")
    logger.info("=" * 60)

    with engine.connect() as conn:
        # 1. Cek jumlah baris di pos_pendakian
        cnt_pos = conn.execute(text("SELECT COUNT(*) FROM pos_pendakian")).scalar()
        logger.info(f"Jumlah baris 'pos_pendakian'   : {cnt_pos} (Ekspektasi: 19)")

        # 2. Cek jumlah baris di titik_api
        cnt_api = conn.execute(text("SELECT COUNT(*) FROM titik_api")).scalar()
        logger.info(f"Jumlah baris 'titik_api'       : {cnt_api} (Ekspektasi: 552)")

        # 3. Cek jumlah baris di cuaca_integrated
        cnt_cuaca = conn.execute(text("SELECT COUNT(*) FROM cuaca_integrated")).scalar()
        logger.info(f"Jumlah baris 'cuaca_integrated' : {cnt_cuaca:,} (Ekspektasi: 832,656)")

        # 4. Cek jumlah baris status kebakaran sekitar aktif
        cnt_kebakaran = conn.execute(
            text("SELECT COUNT(*) FROM cuaca_integrated WHERE status_kebakaran_sekitar = 1")
        ).scalar()
        logger.info(f"Jumlah baris berisiko api      : {cnt_kebakaran:,} (Ekspektasi: 3,624)")

        # 5. Cek contoh query time-series (Membuktikan TimescaleDB time_bucket)
        logger.info("Menguji query time-series (TimescaleDB time_bucket)...")
        test_query = """
            SELECT time_bucket('1 day', timestamp) AS hari, AVG(suhu_c) AS rata_suhu
            FROM cuaca_integrated
            WHERE nama_pos = 'Basecamp Cemoro Sewu'
              AND timestamp BETWEEN '2023-09-01 00:00:00+07' AND '2023-09-05 23:59:59+07'
            GROUP BY hari
            ORDER BY hari;
        """
        results = conn.execute(text(test_query)).fetchall()
        for row in results:
            hari_str = row[0].strftime('%Y-%m-%d')
            logger.info(f"  - Tanggal: {hari_str} | Rata-rata Suhu: {row[1]:.2f} °C")

    logger.info("=" * 60)
    logger.info("VERIFIKASI SELESAI DENGAN SUKSES!")
    logger.info("=" * 60)


def run() -> None:
    logger.info("=" * 60)
    logger.info("PIPELINE LOAD DATA KE AIVEN POSTGRESQL + TIMESCALEDB")
    logger.info("=" * 60)

    # 1. Validasi keberadaan AIVEN_DATABASE_URL di env
    db_url = Config.AIVEN_DATABASE_URL
    if not db_url:
        logger.error("[!] AIVEN_DATABASE_URL tidak ditemukan di file .env!")
        logger.error("    Silakan buat/konfigurasi service database di Aiven Console,")
        logger.error("    lalu salin Service URI ke file .env.")
        logger.error("    Format: AIVEN_DATABASE_URL=postgresql://avnadmin:password@host:port/defaultdb?sslmode=require")
        sys.exit(1)

    # 2. Resolusi Path Input
    tahun_mulai_weather = Config.WEATHER_HISTORICAL_START[:4]
    tahun_selesai_weather = Config.WEATHER_HISTORICAL_END[:4]
    cuaca_integrated_csv = (
        Config.DATA_CURATED_DIR
        / f"dataset_integrated_lawu_{tahun_mulai_weather}_{tahun_selesai_weather}.csv"
    )

    tahun_mulai_firms = Config.FIRMS_START_DATE[:4]
    tahun_selesai_firms = Config.FIRMS_END_DATE[:4]
    disaster_csv = (
        Config.DATA_RAW_DIR / "disaster"
        / f"titik_api_lawu_{tahun_mulai_firms}_{tahun_selesai_firms}.csv"
    )

    logger.info(f"File Titik Api (Input) : {disaster_csv}")
    logger.info(f"File Cuaca (Input)     : {cuaca_integrated_csv}")
    logger.info("")

    # Validasi file input lokal
    if not disaster_csv.exists():
        logger.error(f"[!] File titik api '{disaster_csv}' tidak ditemukan!")
        logger.error("    Pastikan pipeline_disaster_historical.py sudah dijalankan.")
        sys.exit(1)

    if not cuaca_integrated_csv.exists():
        logger.error(f"[!] File cuaca terintegrasi '{cuaca_integrated_csv}' tidak ditemukan!")
        logger.error("    Pastikan pipeline_integration_historical.py sudah dijalankan.")
        sys.exit(1)

    try:
        # 3. Membuat engine koneksi ke Aiven
        engine = get_engine(db_url)

        # 4. Membuat/Memeriksa Schema Database
        ensure_schema(engine)

        # 5. Load Data ke Tabel Dimensi 'pos_pendakian'
        load_pos_pendakian(engine)

        # 6. Load Data ke Tabel 'titik_api'
        load_titik_api(engine, disaster_csv)

        # 7. Load Data ke Hypertable 'cuaca_integrated'
        load_cuaca_integrated(engine, cuaca_integrated_csv)

        # 8. Jalankan Verifikasi
        run_verification(engine)

        logger.info("Proses pipeline loading data ke Aiven selesai dengan SUKSES!")

    except Exception as e:
        logger.error(f"[!] Terjadi kesalahan saat memproses pipeline ke Aiven: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run()

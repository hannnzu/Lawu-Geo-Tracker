"""
pipeline_jalur_pendakian.py
----------------------------
Entry point utama untuk ETL data jalur pendakian Gunung Lawu.
Alur:
1. Membaca file GPX mentah.
2. Memproses metrik spasial (jarak, gain, kemiringan, tipe medan).
3. Menyimpan hasil olahan ke CSV lokal.
4. Membuat skema database di Aiven PostgreSQL.
5. Memuat data olahan ke tabel 'jalur_pendakian'.
6. Memverifikasi hasil pemuatan.

Cara menjalankan:
    python pipeline_jalur_pendakian.py
"""

import csv
from pathlib import Path
from dotenv import load_dotenv

from src.utils.config import Config
from src.utils.logger import get_logger
from src.ingestion.gpx_reader import parse_gpx
from src.transformation.gpx_processor import process_gpx_points
from src.loading.aiven_loader import get_engine, ensure_schema, load_jalur_pendakian

# Load environment variables dari .env
load_dotenv()

logger = get_logger("pipeline_jalur_pendakian")


def run():
    logger.info("============================================================")
    logger.info("PIPELINE ETL JALUR PENDAKIAN GUNUNG LAWU")
    logger.info("============================================================")

    # 1. Definisikan file input dan output
    gpx_dir = Config.DATA_RAW_DIR / "gpx"
    processed_dir = Config.DATA_PROC_DIR / "geospatial"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    local_csv_path = processed_dir / "jalur_pendakian_processed.csv"

    # Daftar GPX yang akan diproses beserta penamaannya
    gpx_files_config = [
        {
            "filename": "mount-lawu-via-cemoro-sewu.gpx",
            "nama_jalur": "Cemoro Sewu"
        },
        {
            "filename": "jalur-pendakian-gunung-lawu-via-cemoro-kandang.gpx",
            "nama_jalur": "Cemoro Kandang"
        },
        {
            "filename": "gunung-lawu-via-candi-cetho-karanganyar-jawa-tengah-agustus-.gpx",
            "nama_jalur": "Candi Cetho"
        }
    ]

    # 2. Proses file GPX
    all_processed_points = []

    for item in gpx_files_config:
        file_path = gpx_dir / item["filename"]
        if not file_path.exists():
            logger.warning(f"File GPX tidak ditemukan di: {file_path}. Melewati jalur {item['nama_jalur']}.")
            continue
            
        logger.info(f"Membaca & memproses jalur: {item['nama_jalur']} dari {item['filename']}...")
        try:
            raw_points = parse_gpx(file_path)
            processed_points = process_gpx_points(raw_points, item["nama_jalur"])
            all_processed_points.extend(processed_points)
        except Exception as e:
            logger.error(f"Gagal memproses file {item['filename']}: {e}", exc_info=True)

    if not all_processed_points:
        logger.error("Tidak ada titik GPX yang berhasil diproses. Menghentikan pipeline.")
        return

    # 3. Simpan ke CSV lokal
    logger.info(f"Menyimpan hasil pemrosesan jalur ke CSV lokal: {local_csv_path}...")
    try:
        headers = [
            "nama_jalur", "urutan_titik", "lat", "lon", "elevasi_mdpl",
            "kemiringan_pct", "jarak_dari_basecamp_km", "akumulasi_gain_m",
            "sumber_file", "terrain_type"
        ]
        with open(local_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_processed_points)
        logger.info(f"Berhasil menyimpan {len(all_processed_points)} baris ke CSV lokal.")
    except Exception as e:
        logger.error(f"Gagal menyimpan ke CSV lokal: {e}")

    # 4. Hubungkan ke database dan lakukan loading
    db_url = Config.AIVEN_DATABASE_URL
    if not db_url:
        logger.error("AIVEN_DATABASE_URL tidak diset di .env!")
        return

    try:
        engine = get_engine(db_url)
        
        # Pastikan skema tabel dibuat/diperbarui
        ensure_schema(engine)
        
        # Load ke database
        loaded_count = load_jalur_pendakian(engine, all_processed_points)
        logger.info(f"Pipeline Database: Muat {loaded_count} titik sukses.")
        
        # Verifikasi database
        from sqlalchemy import text
        with engine.connect() as conn:
            # Hitung jumlah titik per jalur di DB
            res = conn.execute(text("""
                SELECT nama_jalur, COUNT(*), MIN(elevasi_mdpl), MAX(elevasi_mdpl)
                FROM jalur_pendakian
                GROUP BY nama_jalur
                ORDER BY nama_jalur;
            """)).fetchall()
            
            logger.info("--- HASIL VERIFIKASI TABEL JALUR_PENDAKIAN ---")
            for row in res:
                logger.info(f"Jalur: {row[0]:<15} | Jumlah Titik: {row[1]:>5} | Elevasi: {row[2]:>4}m - {row[3]:>4}m")
            logger.info("-----------------------------------------------")
            
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat memuat ke database Aiven: {e}", exc_info=True)

    logger.info("=== PIPELINE JALUR PENDAKIAN SELESAI ===")


if __name__ == "__main__":
    run()

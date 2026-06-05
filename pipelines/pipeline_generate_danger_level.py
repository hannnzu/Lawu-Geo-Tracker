"""
pipeline_generate_danger_level.py
----------------------------------
Entry point utama untuk menghitung dan memperbarui tingkat bahaya (danger_level).
Alur:
1. Membaca CSV lokal curated.
2. Menghitung 'Danger_Level' (skala 0-3) baris demi baris secara efisien.
3. Menyimpan hasil perbaruan ke CSV lokal curated.
4. Menambahkan kolom 'danger_level' di tabel database Aiven jika belum ada.
5. Melakukan update batch server-side pada tabel 'cuaca_integrated' di database Aiven.
6. Melakukan verifikasi distribusi kelas danger level.

Cara menjalankan:
    python pipeline_generate_danger_level.py
"""

import csv
import time
from pathlib import Path
from dotenv import load_dotenv

from src.utils.config import Config
from src.utils.logger import get_logger
from src.transformation.danger_labeler import hitung_danger_level_dari_row_csv
from src.loading.aiven_loader import get_engine, ensure_schema, update_danger_level_in_db

# Load environment variables dari .env
load_dotenv()

logger = get_logger("pipeline_generate_danger_level")


def run():
    logger.info("============================================================")
    logger.info("PIPELINE GENERATE DANGER LEVEL DATA LAWU")
    logger.info("============================================================")

    # 1. Definisikan file input/output
    csv_path = Config.DATA_CURATED_DIR / "dataset_integrated_lawu_2021_2025.csv"
    temp_csv_path = csv_path.parent / "dataset_integrated_lawu_2021_2025_temp.csv"

    # Self-healing: jika file utama terhapus tapi file temp ada dari proses sebelumnya, pulihkan
    if not csv_path.exists() and temp_csv_path.exists():
        logger.info("Mendeteksi file temp sisa dari proses sebelumnya. Memulihkan file asli...")
        try:
            temp_csv_path.rename(csv_path)
            logger.info("File asli berhasil dipulihkan.")
        except Exception as e:
            logger.error(f"Gagal memulihkan file asli: {e}")
            return

    if not csv_path.exists():
        logger.error(f"Dataset terintegrasi tidak ditemukan di {csv_path}!")
        return

    # 2. Proses CSV lokal
    logger.info(f"Memproses CSV lokal: {csv_path}...")
    start_time = time.time()
    processed_count = 0
    
    # Distribusi kelas untuk verifikasi lokal
    class_distribution = {0: 0, 1: 0, 2: 0, 3: 0}

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f_in, \
             open(temp_csv_path, mode="w", newline="", encoding="utf-8") as f_out:
            
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames or []
            
            # Tambahkan kolom baru 'Danger_Level' jika belum ada
            if "Danger_Level" not in fieldnames:
                extended_fieldnames = fieldnames + ["Danger_Level"]
            else:
                extended_fieldnames = fieldnames
                
            writer = csv.DictWriter(f_out, fieldnames=extended_fieldnames)
            writer.writeheader()
            
            for row in reader:
                danger_val = hitung_danger_level_dari_row_csv(row)
                row["Danger_Level"] = danger_val
                
                # Catat distribusi kelas
                class_distribution[danger_val] += 1
                
                writer.writerow(row)
                processed_count += 1
                
                if processed_count % 200000 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"  Sudah memproses {processed_count:,} baris... ({elapsed:.1f} detik)")

        # Berikan jeda sejenak agar OS Windows melepas lock file
        time.sleep(1.0)

        # Ganti file lama dengan yang baru (dengan retry untuk mengantisipasi race condition di Windows)
        success = False
        for attempt in range(5):
            try:
                if csv_path.exists():
                    csv_path.unlink()
                temp_csv_path.rename(csv_path)
                success = True
                break
            except PermissionError:
                logger.warning(f"File sedang dikunci oleh proses lain. Mencoba kembali dalam 1 detik... (Percobaan {attempt+1}/5)")
                time.sleep(1.0)
                
        if not success:
            raise PermissionError("Gagal mengganti file CSV karena dikunci secara permanen oleh proses lain.")
        
        duration = time.time() - start_time
        logger.info(f"Selesai memperbarui CSV lokal. Total: {processed_count:,} baris dalam {duration:.2f} detik.")
        logger.info("Distribusi kelas Danger Level di CSV lokal:")
        for k, v in class_distribution.items():
            pct = (v / processed_count) * 100 if processed_count > 0 else 0.0
            logger.info(f"  Level {k}: {v:>8,} baris ({pct:.2f}%)")

    except Exception as e:
        logger.error(f"Gagal memperbarui file CSV lokal: {e}", exc_info=True)
        time.sleep(0.5)
        if temp_csv_path.exists():
            try:
                temp_csv_path.unlink()
            except PermissionError:
                pass
        return

    # 3. Update database Aiven
    db_url = Config.AIVEN_DATABASE_URL
    if not db_url:
        logger.error("AIVEN_DATABASE_URL tidak diset di .env!")
        return

    try:
        engine = get_engine(db_url)
        
        # Pastikan kolom 'danger_level' ada di database schema
        ensure_schema(engine)
        
        # Jalankan update server-side untuk tabel cuaca_integrated
        updated_count = update_danger_level_in_db(engine)
        logger.info(f"Database: Berhasil memperbarui {updated_count:,} baris.")

        # 4. Verifikasi Database
        from sqlalchemy import text
        with engine.connect() as conn:
            # Uji 1: Cek distribusi danger_level di DB
            res = conn.execute(text("""
                SELECT danger_level, COUNT(*), ROUND(COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER() * 100, 2) AS persentase
                FROM cuaca_integrated
                GROUP BY danger_level
                ORDER BY danger_level;
            """)).fetchall()
            
            logger.info("--- HASIL VERIFIKASI DANGER LEVEL DI DB ---")
            for row in res:
                logger.info(f"Level {row[0]}: {row[1]:>8,} baris ({row[2]}%)")
            
            # Uji 2: Cek anomali (apakah ada status_kebakaran_sekitar = 1 tapi danger_level != 3)
            anomali = conn.execute(text("""
                SELECT COUNT(*) 
                FROM cuaca_integrated 
                WHERE status_kebakaran_sekitar = 1 AND danger_level != 3;
            """)).fetchone()[0]
            
            if anomali == 0:
                logger.info("Verifikasi Integritas: LULUS (Semua data kebakaran sekitar terlabeli Level 3).")
            else:
                logger.warning(f"Verifikasi Integritas: GAGAL ({anomali} baris anomali terdeteksi).")
            logger.info("-------------------------------------------")

    except Exception as e:
        logger.error(f"Terjadi kesalahan saat memproses database Aiven: {e}", exc_info=True)

    logger.info("=== PIPELINE GENERATE DANGER LEVEL SELESAI ===")


if __name__ == "__main__":
    run()

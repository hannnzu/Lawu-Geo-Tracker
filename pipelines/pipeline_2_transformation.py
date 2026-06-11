"""
pipeline_2_transformation.py
--------------------------------
Fase TRANSFORM: Membersihkan, mengintegrasikan, dan melabeli data lokal.
1. Integrasi Spatiotemporal Cuaca & Titik Api
2. Generate Danger Level (ke dalam CSV lokal)
3. Ekstraksi dan kalkulasi metrik file GPX
"""

import csv
import time
from pathlib import Path
from src.utils.config import Config
from src.utils.logger import get_logger

from src.transformation.integration import integrate_weather_and_disaster
from src.transformation.danger_labeler import hitung_danger_level_dari_row_csv
from src.ingestion.gpx_reader import parse_gpx
from src.transformation.gpx_processor import process_gpx_points

logger = get_logger("pipeline_2_transformation", log_dir=Config.LOGS_DIR)

def transform_integration_and_danger_level():
    logger.info("--- [1/2] TRANSFORM: INTEGRASI & DANGER LEVEL ---")
    
    # Setup Paths
    t_w_start, t_w_end = Config.WEATHER_HISTORICAL_START[:4], Config.WEATHER_HISTORICAL_END[:4]
    t_f_start, t_f_end = Config.FIRMS_START_DATE[:4], Config.FIRMS_END_DATE[:4]
    
    weather_csv = Config.DATA_PROC_DIR / "weather" / f"cuaca_hourly_lawu_{t_w_start}_{t_w_end}.csv"
    disaster_csv = Config.DATA_RAW_DIR / "disaster" / f"titik_api_lawu_{t_f_start}_{t_f_end}.csv"
    curated_csv = Config.DATA_CURATED_DIR / f"dataset_integrated_lawu_{t_w_start}_{t_w_end}.csv"
    temp_csv_path = curated_csv.parent / (curated_csv.name.replace('.csv', '_temp.csv'))

    if not weather_csv.exists() or not disaster_csv.exists():
        logger.error("[!] File input cuaca atau disaster tidak ditemukan. Jalankan Pipeline 1 dahulu.")
        return

    # 1. Integrasi Cuaca & Bencana
    logger.info("Melakukan integrasi spatiotemporal (Jarak Threshold: 3.0 KM)...")
    success = integrate_weather_and_disaster(weather_csv, disaster_csv, curated_csv, distance_threshold_km=3.0)
    
    if not success or not curated_csv.exists():
        logger.error("[!] Integrasi GAGAL.")
        return

    # 2. Kalkulasi Danger Level ke CSV
    logger.info("Menghitung Danger Level dan memperbarui CSV Curated...")
    processed_count = 0
    class_distribution = {0: 0, 1: 0, 2: 0, 3: 0}

    with open(curated_csv, mode="r", encoding="utf-8") as f_in, \
         open(temp_csv_path, mode="w", newline="", encoding="utf-8") as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        if "Danger_Level" not in fieldnames:
            fieldnames.append("Danger_Level")
            
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            danger_val = hitung_danger_level_dari_row_csv(row)
            row["Danger_Level"] = danger_val
            class_distribution[danger_val] += 1
            writer.writerow(row)
            processed_count += 1

    time.sleep(1.0)
    curated_csv.unlink()
    temp_csv_path.rename(curated_csv)
    
    logger.info(f"Selesai! {processed_count:,} baris diproses.")
    for k, v in class_distribution.items():
        logger.info(f"  Level {k}: {v:>8,} baris")

def transform_gpx():
    logger.info("\n--- [2/2] TRANSFORM: PEMROSESAN GPX ---")
    gpx_dir = Config.DATA_RAW_DIR / "gpx"
    processed_dir = Config.DATA_PROC_DIR / "geospatial"
    processed_dir.mkdir(parents=True, exist_ok=True)
    local_csv_path = processed_dir / "jalur_pendakian_processed.csv"

    gpx_files = [
        {"filename": "mount-lawu-via-cemoro-sewu.gpx", "nama_jalur": "Cemoro Sewu"},
        {"filename": "jalur-pendakian-gunung-lawu-via-cemoro-kandang.gpx", "nama_jalur": "Cemoro Kandang"},
        {"filename": "gunung-lawu-via-candi-cetho-karanganyar-jawa-tengah-agustus-.gpx", "nama_jalur": "Candi Cetho"}
    ]

    all_points = []
    for item in gpx_files:
        file_path = gpx_dir / item["filename"]
        if file_path.exists():
            raw_points = parse_gpx(file_path)
            all_points.extend(process_gpx_points(raw_points, item["nama_jalur"]))

    if all_points:
        headers = ["nama_jalur", "urutan_titik", "lat", "lon", "elevasi_mdpl", "kemiringan_pct", "jarak_dari_basecamp_km", "akumulasi_gain_m", "sumber_file", "terrain_type"]
        with open(local_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_points)
        logger.info(f"Berhasil menyimpan {len(all_points)} titik jalur pendakian ke {local_csv_path.name}")

def run():
    logger.info("=" * 60)
    logger.info("PIPELINE 2: TRANSFORMATION (TRANSFORM) DIMULAI")
    logger.info("=" * 60)
    transform_integration_and_danger_level()
    transform_gpx()
    logger.info("=" * 60)
    logger.info("PIPELINE 2 SELESAI")

if __name__ == "__main__":
    run()
"""
pipeline_osm_trails.py
-----------------------
Entry point: Pipeline data jalur pendakian dari OpenStreetMap.
Menjalankan ETL: Extract (Overpass API) -> Transform (GeoJSON) -> Load ke file.

Cara jalankan:
    python pipeline_osm_trails.py
"""

from src.ingestion.osm_extractor import fetch_trails_osm, osm_to_geojson, save_geojson
from src.utils.config import Config
from src.utils.logger import get_logger
import json

logger = get_logger("pipeline_osm_trails")

OUTPUT_GEOJSON = Config.DATA_PROC_DIR / "geospatial" / "jalur_lawu_osm.geojson"


def run():
    logger.info("=== PIPELINE OSM TRAILS DIMULAI ===")

    # 1. EXTRACT
    logger.info("[Extract] Mengambil jalur dari Overpass API...")
    osm_data = fetch_trails_osm()

    # 2. TRANSFORM
    logger.info("[Transform] Konversi ke GeoJSON...")
    geojson = osm_to_geojson(osm_data)
    logger.info(f"  {len(geojson['features'])} feature berhasil dikonversi.")

    # 3. LOAD
    logger.info(f"[Load] Menyimpan ke {OUTPUT_GEOJSON}...")
    save_geojson(geojson, OUTPUT_GEOJSON)

    logger.info("=== PIPELINE SELESAI ===")


if __name__ == "__main__":
    run()

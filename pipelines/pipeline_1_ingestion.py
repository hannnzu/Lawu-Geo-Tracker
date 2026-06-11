"""
pipeline_1_ingestion.py
--------------------------------
Fase EXTRACT: Menarik semua data mentah dari API eksternal.
1. NASA FIRMS (Titik Api)
2. Open-Meteo (Cuaca Historis)
3. OpenStreetMap (Jalur OSM)
"""

from src.utils.config import Config
from src.utils.logger import get_logger

from src.ingestion.nasa_firms import fetch_all_fire_data, save_to_csv as save_firms, get_last_saved_date
from src.ingestion.open_meteo import fetch_all_locations, save_to_csv as save_meteo, get_existing_locations, LOKASI_GUNUNG_LAWU
from src.ingestion.osm_extractor import fetch_trails_osm, osm_to_geojson, save_geojson

logger = get_logger("pipeline_1_ingestion", log_dir=Config.LOGS_DIR)

def ingest_disaster() -> None:
    """
    Menjalankan ekstraksi data titik api dari NASA FIRMS API.
    
    Membaca API key dan range tanggal pencarian dari konfigurasi sistem.
    Memeriksa tanggal data terakhir yang tersimpan di lokal untuk mendukung fitur resume.
    Melakukan request API NASA FIRMS secara batch untuk periode tanggal yang tersisa.
    Menyimpan data titik api yang diperoleh secara append ke file CSV lokal.
    
    Returns:
        None
    """
    logger.info("--- [1/3] EXTRACT: NASA FIRMS (TITIK API) ---")
    MAP_KEY = Config.FIRMS_MAP_KEY
    if not MAP_KEY or len(MAP_KEY) != 32:
        logger.error("[!] FIRMS_MAP_KEY tidak valid/tidak ditemukan di .env!")
        return

    START_DATE, END_DATE = Config.FIRMS_START_DATE, Config.FIRMS_END_DATE
    OUTPUT_CSV = Config.DATA_RAW_DIR / "disaster" / f"titik_api_lawu_{START_DATE[:4]}_{END_DATE[:4]}.csv"
    
    last_date = get_last_saved_date(OUTPUT_CSV)
    all_rows = fetch_all_fire_data(MAP_KEY, START_DATE, END_DATE, resume_from=last_date, delay_sec=1.0)
    
    if all_rows:
        save_firms(all_rows, OUTPUT_CSV, mode="a")
        logger.info(f"Berhasil menyimpan {len(all_rows)} baris data titik api ke {OUTPUT_CSV.name}")
    else:
        logger.info("Tidak ada data titik api baru untuk disimpan.")

def ingest_weather() -> None:
    """
    Menjalankan ekstraksi data cuaca per jam dari Open-Meteo Archive API.
    
    Membaca konfigurasi tanggal dan memeriksa pos pendakian yang sudah ada di lokal (fitur resume).
    Melakukan request API Open-Meteo hanya untuk lokasi pos pendakian yang tersisa.
    Menerapkan jeda waktu 3.0 detik antar lokasi stasiun cuaca untuk menghindari batas limit API.
    Menyimpan data hasil unduhan baru secara append (mode='a') ke file CSV lokal.
    
    Returns:
        None
    """
    logger.info("\n--- [2/3] EXTRACT: OPEN-METEO (CUACA HOURLY) ---")
    START_DATE, END_DATE = Config.WEATHER_HISTORICAL_START, Config.WEATHER_HISTORICAL_END
    OUTPUT_CSV = Config.DATA_PROC_DIR / "weather" / f"cuaca_hourly_lawu_{START_DATE[:4]}_{END_DATE[:4]}.csv"

    existing = get_existing_locations(OUTPUT_CSV)
    all_rows, failed = fetch_all_locations(START_DATE, END_DATE, delay_sec=3.0, skip_existing=existing)
    
    if all_rows:
        save_meteo(all_rows, OUTPUT_CSV, mode="a")
        logger.info(f"Berhasil menyimpan {len(all_rows):,} baris cuaca ke {OUTPUT_CSV.name}")
    else:
        logger.info("Tidak ada data cuaca baru untuk disimpan.")
        
    if failed:
        logger.warning(f"Lokasi gagal diambil: {failed}")

def ingest_osm() -> None:
    """
    Menjalankan ekstraksi data rute jalur pendakian dari OpenStreetMap (OSM).
    
    Mengirimkan query Overpass API untuk mengambil jalan setapak (highway=path) di wilayah Gunung Lawu.
    Mengonversi data struktur nodes dan ways dari OSM menjadi format standar GeoJSON.
    Menyimpan file GeoJSON hasil ekstraksi secara lokal untuk pemetaan spasial.
    
    Returns:
        None
    """
    logger.info("\n--- [3/3] EXTRACT: OPENSTREETMAP (JALUR OSM) ---")
    OUTPUT_GEOJSON = Config.DATA_PROC_DIR / "geospatial" / "jalur_lawu_osm.geojson"
    osm_data = fetch_trails_osm()
    geojson = osm_to_geojson(osm_data)
    save_geojson(geojson, OUTPUT_GEOJSON)
    logger.info(f"Berhasil menyimpan {len(geojson['features'])} feature GeoJSON ke {OUTPUT_GEOJSON.name}")

def run() -> None:
    """
    Menjalankan seluruh rangkaian Pipeline Ingestion (Extract) secara berurutan.
    
    Mencatat log penanda dimulainya pipeline ekstraksi data.
    Memanggil fungsi ekstraksi titik api (NASA FIRMS), cuaca per jam (Open-Meteo), dan jalur (OSM).
    Mencatat log akhir penanda bahwa seluruh proses ekstraksi telah selesai.
    
    Returns:
        None
    """
    logger.info("=" * 60)
    logger.info("PIPELINE 1: INGESTION (EXTRACT) DIMULAI")
    logger.info("=" * 60)
    ingest_disaster()
    ingest_weather()
    ingest_osm()
    logger.info("=" * 60)
    logger.info("PIPELINE 1 SELESAI")

if __name__ == "__main__":
    run()
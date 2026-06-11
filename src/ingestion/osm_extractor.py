"""
ingestion/osm_extractor.py
--------------------------
Extract data jalur pendakian dari OpenStreetMap via Overpass API.
Dipindahkan dari: cemoro_sewu.py
"""

import requests
import json
from pathlib import Path

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# Bounding box area Gunung Lawu (south, west, north, east)
LAWU_BBOX = (-7.700, 111.140, -7.580, 111.230)


def build_overpass_query(bbox: tuple, name_filter: str = "cemoro|lawu") -> str:
    """
    Membangun query Overpass QL untuk mengambil jalur pendakian.

    Args:
        bbox: Tuple (south, west, north, east).
        name_filter: Regex filter nama jalur (case-insensitive).

    Returns:
        str: Overpass QL query string.
    """
    s, w, n, e = bbox
    return f"""
[out:json][timeout:30];
(
  way["highway"~"path|footway"]["name"~"{name_filter}",i]({s},{w},{n},{e});
);
out geom;
"""


def fetch_trails_osm(bbox: tuple = LAWU_BBOX,
                     name_filter: str = "cemoro|lawu") -> dict:
    """
    Mengambil data jalur pendakian dari Overpass API.

    Args:
        bbox: Bounding box area pencarian.
        name_filter: Filter nama jalur.

    Returns:
        dict: Data OSM JSON mentah.
    """
    query = build_overpass_query(bbox, name_filter)
    print("Mengambil data jalur dari OpenStreetMap (Overpass API)...")
    
    # 1. Tambahkan identitas User-Agent agar tidak dianggap bot spam oleh server
    headers = {
        "User-Agent": "LawuGeoTracker/1.0 (Data Engineering Project)"
    }
    
    # 2. Sisipkan parameter headers ke dalam requests.post
    response = requests.post(
        OVERPASS_URL, 
        data={"data": query}, 
        headers=headers, 
        timeout=60
    )
    
    response.raise_for_status()
    data = response.json()
    print(f"Berhasil! Ditemukan {len(data['elements'])} segmen jalur.")
    return data


def osm_to_geojson(osm_data: dict) -> dict:
    """
    Mengonversi respons OSM JSON ke format GeoJSON FeatureCollection.

    Args:
        osm_data: Data mentah dari Overpass API.

    Returns:
        dict: GeoJSON FeatureCollection.
    """
    features = []
    for elemen in osm_data.get("elements", []):
        if elemen["type"] == "way":
            koordinat = [
                [titik["lon"], titik["lat"]]
                for titik in elemen.get("geometry", [])
            ]
            features.append({
                "type": "Feature",
                "properties": {
                    "osm_id":     elemen.get("id"),
                    "nama_jalur": elemen.get("tags", {}).get("name", "Tidak diketahui"),
                    "jenis":      elemen.get("tags", {}).get("highway", "path"),
                    "sumber":     "OpenStreetMap via Overpass API",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": koordinat,
                },
            })

    return {"type": "FeatureCollection", "features": features}


def save_geojson(geojson: dict, output_path: str | Path) -> None:
    """Simpan GeoJSON ke file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    print(f"GeoJSON tersimpan: {output_path}")

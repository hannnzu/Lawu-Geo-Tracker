"""
transformation/geo_transform.py
---------------------------------
Transformasi data geospasial: GPX -> GeoJSON, normalisasi elevasi, dll.
"""

from pathlib import Path
import json


def gpx_points_to_geojson(points: list[dict], route_name: str = "Unknown") -> dict:
    """
    Mengonversi list titik GPX menjadi GeoJSON LineString Feature.

    Args:
        points: List titik dari parse_gpx() [lat, lon, ele, time].
        route_name: Nama jalur/rute.

    Returns:
        dict: GeoJSON Feature dengan geometry LineString.
    """
    # Menyusun ulang pasangan koordinat menjadi format [longitude, latitude, elevasi] standar spesifikasi GeoJSON.
    coordinates = [
        [p["lon"], p["lat"], p.get("ele", 0)]
        for p in points
    ]

    elevasi_values = [p.get("ele", 0) for p in points if p.get("ele") is not None]

    return {
        "type": "Feature",
        "properties": {
            "nama_jalur":    route_name,
            "jumlah_titik":  len(points),
            "elevasi_min":   round(min(elevasi_values), 1) if elevasi_values else None,
            "elevasi_max":   round(max(elevasi_values), 1) if elevasi_values else None,
            "elevasi_rata":  round(sum(elevasi_values) / len(elevasi_values), 1) if elevasi_values else None,
            "waktu_mulai":   points[0].get("time") if points else None,
            "waktu_selesai": points[-1].get("time") if points else None,
            "sumber":        points[0].get("source_file") if points else None,
        },
        "geometry": {
            "type":        "LineString",
            "coordinates": coordinates,
        },
    }


def merge_geojson_features(features: list[dict]) -> dict:
    """
    Menggabungkan beberapa GeoJSON Feature menjadi satu FeatureCollection.

    Args:
        features: List GeoJSON Feature dict.

    Returns:
        dict: GeoJSON FeatureCollection.
    """
    return {
        "type":     "FeatureCollection",
        "features": features,
    }

"""
ingestion/gpx_reader.py
-----------------------
Membaca dan mem-parse file GPX (GPS Exchange Format) dari folder data/raw/gpx/.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator


GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def parse_gpx(file_path: str | Path) -> list[dict]:
    """
    Mem-parse file GPX dan mengembalikan list titik koordinat.

    Args:
        file_path: Path ke file .gpx.

    Returns:
        list[dict]: List titik dengan field: lat, lon, ele, time.
    """
    file_path = Path(file_path)
    # Membaca dan mem-parse struktur XML dari file GPX mentah.
    tree = ET.parse(file_path)
    root = tree.getroot()

    titik_list = []
    for trkpt in root.findall(".//gpx:trkpt", GPX_NS):
        titik = {
            "lat": float(trkpt.get("lat")),
            "lon": float(trkpt.get("lon")),
            "ele": float(trkpt.findtext("gpx:ele", default="0", namespaces=GPX_NS)),
            "time": trkpt.findtext("gpx:time", default=None, namespaces=GPX_NS),
            "source_file": file_path.name,
        }
        titik_list.append(titik)

    print(f"  -> Parsed {len(titik_list)} titik dari {file_path.name}")
    return titik_list


def load_all_gpx(gpx_dir: str | Path) -> list[dict]:
    """
    Membaca semua file .gpx dalam sebuah direktori.

    Args:
        gpx_dir: Path ke direktori yang berisi file-file .gpx.

    Returns:
        list[dict]: Gabungan semua titik dari semua file GPX.
    """
    gpx_dir = Path(gpx_dir)
    all_points = []
    gpx_files = list(gpx_dir.glob("*.gpx"))

    print(f"Ditemukan {len(gpx_files)} file GPX di {gpx_dir}")

    for gpx_file in gpx_files:
        points = parse_gpx(gpx_file)
        all_points.extend(points)

    print(f"Total: {len(all_points)} titik GPX berhasil dimuat.")
    return all_points

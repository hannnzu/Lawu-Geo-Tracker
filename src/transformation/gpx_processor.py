"""
src/transformation/gpx_processor.py
------------------------------------
Modul untuk memproses data titik GPX, menghitung metrik spasial:
- Akumulasi jarak dari basecamp (KM)
- Akumulasi elevation gain (meter)
- Kemiringan / gradient (%)
- Klasifikasi jenis medan (terrain type)
- Penanganan otomatis track terbalik (summit -> basecamp)
"""

import math
from typing import List, Dict
from src.utils.logger import get_logger

logger = get_logger("gpx_processor")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Menghitung jarak antara dua koordinat menggunakan formula Haversine (dalam kilometer).
    """
    R = 6371.0  # Radius bumi dalam KM
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    # Menghitung koefisien a dari rumus Haversine menggunakan selisih latitude dan longitude dalam radian.
    a = math.sin(dlat / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def klasifikasi_medan(kemiringan_pct: float) -> str:
    """
    Mengklasifikasikan tipe medan berdasarkan persentase kemiringan.
    """
    kemiringan_abs = abs(kemiringan_pct)
    if kemiringan_abs < 8.0:
        return "Datar/Landai"
    elif kemiringan_abs < 16.0:
        return "Sedang"
    elif kemiringan_abs < 30.0:
        return "Curam"
    else:
        return "Sangat Curam"


def process_gpx_points(points: list[dict], nama_jalur: str) -> list[dict]:
    """
    Memproses daftar titik koordinat GPX dan menghitung metrik akumulasi.
    Mendeteksi secara otomatis apabila urutan trek terbalik (dari puncak ke basecamp).

    Args:
        points: List titik hasil parse_gpx().
        nama_jalur: Nama jalur pendakian (misal: "Cemoro Sewu").

    Returns:
        list[dict]: List titik yang telah diperkaya dengan kolom-kolom spasial.
    """
    if not points:
        logger.warning(f"Daftar titik GPX untuk jalur '{nama_jalur}' kosong.")
        return []

    # Mengecek apakah elevasi awal jauh lebih tinggi dibanding elevasi akhir (indikasi perekaman dari puncak ke basecamp).
    ele_awal = points[0].get("ele", 0.0)
    ele_akhir = points[-1].get("ele", 0.0)
    if ele_awal > ele_akhir + 500.0:
        logger.info(f"Mendeteksi trek terbalik pada '{nama_jalur}' (Awal: {ele_awal}m, Akhir: {ele_akhir}m). Membalikkan urutan titik...")
        # Membalikkan urutan array koordinat agar berurutan dari basecamp naik ke atas puncak.
        points = list(reversed(points))

    processed_points = []
    kumulatif_jarak = 0.0
    kumulatif_gain = 0.0
    
    for i, pt in enumerate(points):
        urutan = i + 1
        lat = pt["lat"]
        lon = pt["lon"]
        ele = pt.get("ele", 0.0)
        sumber = pt.get("source_file", "")

        if i == 0:
            # Titik pertama (Basecamp)
            jarak_segmen = 0.0
            gain_segmen = 0.0
            kemiringan = 0.0
        else:
            prev_pt = points[i - 1]
            prev_lat = prev_pt["lat"]
            prev_lon = prev_pt["lon"]
            prev_ele = prev_pt.get("ele", 0.0)

            # Menghitung jarak garis lurus di permukaan bumi antar koordinat berurutan.
            jarak_segmen = haversine_distance(prev_lat, prev_lon, lat, lon)
            kumulatif_jarak += jarak_segmen

            # Hitung elevasi gain segmen (meter)
            delta_ele = ele - prev_ele
            if delta_ele > 0:
                kumulatif_gain += delta_ele

            # Hitung kemiringan (%)
            jarak_m = jarak_segmen * 1000.0
            if jarak_m > 0.1:  # Hindari pembagian dengan nol atau angka terlampau kecil akibat GPS noise
                # Menghitung persentase kemiringan lereng berdasarkan beda tinggi dibagi jarak horizontal.
                kemiringan = (delta_ele / jarak_m) * 100.0
            else:
                kemiringan = 0.0

        terrain = klasifikasi_medan(kemiringan)

        processed_points.append({
            "nama_jalur": nama_jalur,
            "urutan_titik": urutan,
            "lat": lat,
            "lon": lon,
            "elevasi_mdpl": int(round(ele)),
            "kemiringan_pct": round(kemiringan, 2),
            "jarak_dari_basecamp_km": round(kumulatif_jarak, 4),
            "akumulasi_gain_m": round(kumulatif_gain, 1),
            "sumber_file": sumber,
            "terrain_type": terrain
        })

    logger.info(f"Selesai memproses jalur '{nama_jalur}': {len(processed_points)} titik, Jarak: {kumulatif_jarak:.2f} KM, Gain: {kumulatif_gain:.1f}m")
    return processed_points

"""
transformation/integration.py
------------------------------
Modul transformasi untuk mengintegrasikan data cuaca per jam (hourly)
dengan data titik api (kebakaran hutan) berdasarkan kedekatan spasial
dan temporal.
"""

import csv
import math
from pathlib import Path
from datetime import datetime


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Menghitung jarak lingkaran besar (great-circle distance) antara dua titik
    di permukaan bumi menggunakan formula Haversine.

    Args:
        lat1, lon1: Koordinat titik pertama (derajat).
        lat2, lon2: Koordinat titik kedua (derajat).

    Returns:
        float: Jarak dalam kilometer (KM).
    """
    # Radius bumi dalam kilometer
    R = 6371.0

    # Ubah derajat ke radian
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Perbedaan koordinat
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Menghitung koefisien a dari rumus Haversine berdasarkan selisih sudut radian antar koordinat.
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def load_fire_data_by_date(disaster_csv_path: Path) -> dict[str, list[dict]]:
    """
    Membaca data titik api dan mengelompokkannya berdasarkan tanggal (YYYY-MM-DD).

    Args:
        disaster_csv_path: Path berkas CSV titik api.

    Returns:
        dict: Dictionary dengan key tanggal dan value berupa list data titik api pada tanggal tsb.
    """
    fire_by_date = {}

    if not disaster_csv_path.exists():
        print(f"[!] Warning: File titik api tidak ditemukan di {disaster_csv_path}")
        return fire_by_date

    with open(disaster_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tanggal = row.get("Tanggal", "")
            if not tanggal:
                continue

            try:
                lat = float(row.get("Lat", 0.0))
                lon = float(row.get("Lon", 0.0))
                frp = float(row.get("FRP (MW)", 0.0))
            except ValueError:
                # Lewati baris jika koordinat atau FRP tidak valid
                continue

            detection = {
                "lat": lat,
                "lon": lon,
                "frp": frp,
                "keyakinan": row.get("Keyakinan", "n"),
                "siang_malam": row.get("Siang/Malam", "D")
            }

            if tanggal not in fire_by_date:
                fire_by_date[tanggal] = []
            fire_by_date[tanggal].append(detection)

    total_dates = len(fire_by_date)
    total_points = sum(len(v) for v in fire_by_date.values())
    print(f"Loaded {total_points} titik api dari {total_dates} tanggal unik.")
    return fire_by_date


def integrate_weather_and_disaster(
    weather_csv_path: Path,
    disaster_csv_path: Path,
    output_csv_path: Path,
    distance_threshold_km: float = 3.0
) -> bool:
    """
    Mengintegrasikan data cuaca per jam dengan data titik api harian.
    Untuk setiap baris cuaca pada tanggal D dan lokasi pos P, dihitung jarak ke titik api
    terdekat pada hari tersebut.

    Args:
        weather_csv_path: Path berkas CSV cuaca.
        disaster_csv_path: Path berkas CSV titik api.
        output_csv_path: Path berkas CSV output terpadu.
        distance_threshold_km: Batas jarak aman dalam KM. Default 3.0 KM.

    Returns:
        bool: True jika sukses, False jika gagal.
    """
    # 1. Load data titik api terkelompok tanggal
    fire_by_date = load_fire_data_by_date(disaster_csv_path)

    # 2. Buka berkas cuaca dan siapkan berkas output
    if not weather_csv_path.exists():
        print(f"[!] Error: File cuaca tidak ditemukan di {weather_csv_path}")
        return False

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("Memproses integrasi data baris demi baris...")
    start_time = datetime.now()

    with open(weather_csv_path, mode="r", encoding="utf-8") as f_in, \
         open(output_csv_path, mode="w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []

        # Tambahkan kolom baru hasil integrasi
        new_columns = [
            "Jarak_Titik_Api_Terdekat_KM",
            "FRP_Terdekat_MW",
            "Status_Kebakaran_Sekitar"
        ]
        extended_fieldnames = fieldnames + new_columns

        writer = csv.DictWriter(f_out, fieldnames=extended_fieldnames)
        writer.writeheader()

        processed_rows = 0
        fire_relations_count = 0

        for row in reader:
            processed_rows += 1
            timestamp = row.get("Timestamp", "")
            # Ambil tanggal saja (YYYY-MM-DD)
            tanggal = timestamp[:10]

            lat_pos = float(row.get("Lat", 0.0))
            lon_pos = float(row.get("Lon", 0.0))

            # Default jika tidak ada titik api pada tanggal tersebut
            min_distance = 999.0  # Sentinel value untuk jarak aman/tidak ada api
            closest_frp = 0.0
            status_kebakaran = 0

            # Cari titik api pada tanggal tersebut
            detections = fire_by_date.get(tanggal, [])
            if detections:
                # Mengiterasi seluruh hotspot yang terjadi pada hari yang sama untuk mencari jarak spasial terdekat.
                for det in detections:
                    dist = haversine_distance(lat_pos, lon_pos, det["lat"], det["lon"])
                    if dist < min_distance:
                        min_distance = dist
                        closest_frp = det["frp"]

                # Menentukan status ancaman kebakaran aktif apabila jarak terdekat berada di bawah threshold (3 km).
                if min_distance <= distance_threshold_km:
                    status_kebakaran = 1
                    fire_relations_count += 1

            # Pembulatan nilai untuk efisiensi penyimpanan & estetika
            row["Jarak_Titik_Api_Terdekat_KM"] = round(min_distance, 3) if min_distance != 999.0 else 999.0
            row["FRP_Terdekat_MW"] = round(closest_frp, 2)
            row["Status_Kebakaran_Sekitar"] = status_kebakaran

            writer.writerow(row)

            # Logging periodik setiap 100.000 baris
            if processed_rows % 200000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"  Processed {processed_rows:,} baris... ({elapsed:.1f}s)")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nSelesai! Berhasil mengintegrasikan {processed_rows:,} baris dalam {duration:.2f} detik.")
    print(f"Total baris berisiko kebakaran (jarak <= {distance_threshold_km} KM): {fire_relations_count:,} baris.")
    print(f"Dataset terpadu disimpan ke: {output_csv_path}")

    return True

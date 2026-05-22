"""
ingestion/open_meteo.py
------------------------
Extract data cuaca historis per jam dari Open-Meteo Archive API.
Sumber: https://archive-api.open-meteo.com
- Gratis, tanpa API Key.
- Data per jam: suhu, kelembaban, hujan, angin, awan, jarak pandang, dll.
- Multi-lokasi: 3 basecamp, seluruh pos di 3 jalur, dan puncak Gunung Lawu.
- Rentang waktu: 5 tahun (2021-2025) -> ~831.600 baris total.

Diadaptasi dari: madiun_cuaca_tahunan.py
"""

import csv
import time
from pathlib import Path

import requests


BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variabel per jam yang diambil dari Open-Meteo
# Dipilih berdasarkan relevansi untuk 4 task ML:
# (1) Prediksi kondisi cuaca, (2) Risiko/bahaya,
# (3) Rekomendasi jalur, (4) Estimasi waktu tempuh
HOURLY_VARIABLES = [
    "temperature_2m",          # Suhu udara aktual (°C)
    "apparent_temperature",    # Suhu terasa / windchill (°C)
    "relative_humidity_2m",    # Kelembaban udara (%)
    "precipitation",           # Curah hujan (mm)
    "rain",                    # Hujan murni, bukan salju (mm)
    "wind_speed_10m",          # Kecepatan angin (km/h)
    "wind_direction_10m",      # Arah angin (°)
    "wind_gusts_10m",          # Angin kencang sesaat (km/h)
    "cloud_cover",             # Tutupan awan (%)
    "visibility",              # Jarak pandang (m)
    "surface_pressure",        # Tekanan udara (hPa)
    "weather_code",            # Kode kondisi cuaca WMO
]

CSV_HEADER = [
    "Timestamp",
    "Nama Pos",
    "Jalur",
    "Elevasi (mdpl)",
    "Lat",
    "Lon",
    "Suhu (C)",
    "Suhu Terasa (C)",
    "Kelembaban (%)",
    "Curah Hujan (mm)",
    "Hujan (mm)",
    "Kecepatan Angin (km/h)",
    "Arah Angin (derajat)",
    "Angin Kencang (km/h)",
    "Tutupan Awan (%)",
    "Jarak Pandang (m)",
    "Tekanan Udara (hPa)",
    "Kode Cuaca WMO",
]

# =============================================================================
# Daftar Lokasi Gunung Lawu
# Terdiri dari: 3 Basecamp + Pos per jalur + Puncak Hargo Dumilah
# Tiga jalur: Cemoro Sewu (CS), Cemoro Kandang (CK), Candi Cetho (CC)
# =============================================================================
LOKASI_GUNUNG_LAWU: list[dict] = [
    # -----------------------------------------------------------------------
    # JALUR CEMORO SEWU (via Magetan, Jawa Timur)
    # -----------------------------------------------------------------------
    {
        "nama_pos": "Basecamp Cemoro Sewu",
        "jalur":    "Cemoro Sewu",
        "lat":      -7.663901,
        "lon":      111.191535,
        "elevasi":  1915,
    },
    {
        "nama_pos": "Pos 1 Cemoro Sewu",
        "jalur":    "Cemoro Sewu",
        "lat":      -7.656200,
        "lon":      111.190200,
        "elevasi":  2100,
    },
    {
        "nama_pos": "Pos 2 Cemoro Sewu",
        "jalur":    "Cemoro Sewu",
        "lat":      -7.648500,
        "lon":      111.188500,
        "elevasi":  2360,
    },
    {
        "nama_pos": "Pos 3 Cemoro Sewu",
        "jalur":    "Cemoro Sewu",
        "lat":      -7.635310,
        "lon":      111.184490,
        "elevasi":  2888,
    },
    {
        "nama_pos": "Pos 4 Cemoro Sewu",
        "jalur":    "Cemoro Sewu",
        "lat":      -7.630100,
        "lon":      111.187200,
        "elevasi":  3050,
    },
    {
        "nama_pos": "Pos 5 Cemoro Sewu",
        "jalur":    "Cemoro Sewu",
        "lat":      -7.627800,
        "lon":      111.191000,
        "elevasi":  3150,
    },

    # -----------------------------------------------------------------------
    # JALUR CEMORO KANDANG (via Tawangmangu, Karanganyar, Jawa Tengah)
    # -----------------------------------------------------------------------
    {
        "nama_pos": "Basecamp Cemoro Kandang",
        "jalur":    "Cemoro Kandang",
        "lat":      -7.663160,
        "lon":      111.187170,
        "elevasi":  1913,
    },
    {
        "nama_pos": "Pos 1 Cemoro Kandang",
        "jalur":    "Cemoro Kandang",
        "lat":      -7.656800,
        "lon":      111.186500,
        "elevasi":  2090,
    },
    {
        "nama_pos": "Pos 2 Cemoro Kandang",
        "jalur":    "Cemoro Kandang",
        "lat":      -7.648000,
        "lon":      111.185800,
        "elevasi":  2350,
    },
    {
        "nama_pos": "Pos 3 Cemoro Kandang",
        "jalur":    "Cemoro Kandang",
        "lat":      -7.635290,
        "lon":      111.184500,
        "elevasi":  2889,
    },
    {
        "nama_pos": "Pos 4 Cemoro Kandang",
        "jalur":    "Cemoro Kandang",
        "lat":      -7.629500,
        "lon":      111.187800,
        "elevasi":  3045,
    },
    {
        "nama_pos": "Pos 5 Cemoro Kandang",
        "jalur":    "Cemoro Kandang",
        "lat":      -7.627500,
        "lon":      111.190500,
        "elevasi":  3148,
    },

    # -----------------------------------------------------------------------
    # JALUR CANDI CETHO (via Jenawi, Karanganyar, Jawa Tengah)
    # -----------------------------------------------------------------------
    {
        "nama_pos": "Basecamp Candi Cetho",
        "jalur":    "Candi Cetho",
        "lat":      -7.595080,
        "lon":      111.157188,
        "elevasi":  1466,
    },
    {
        "nama_pos": "Pos 1 Candi Cetho",
        "jalur":    "Candi Cetho",
        "lat":      -7.599200,
        "lon":      111.161500,
        "elevasi":  1680,
    },
    {
        "nama_pos": "Pos 2 Candi Cetho",
        "jalur":    "Candi Cetho",
        "lat":      -7.601500,
        "lon":      111.168800,
        "elevasi":  1940,
    },
    {
        "nama_pos": "Pos 3 Candi Cetho",
        "jalur":    "Candi Cetho",
        "lat":      -7.602822,
        "lon":      111.177754,
        "elevasi":  2163,
    },
    {
        "nama_pos": "Pos 4 Candi Cetho",
        "jalur":    "Candi Cetho",
        "lat":      -7.609000,
        "lon":      111.181200,
        "elevasi":  2450,
    },
    {
        "nama_pos": "Pos 5 Candi Cetho",
        "jalur":    "Candi Cetho",
        "lat":      -7.617500,
        "lon":      111.184800,
        "elevasi":  2730,
    },

    # -----------------------------------------------------------------------
    # PUNCAK
    # -----------------------------------------------------------------------
    {
        "nama_pos": "Hargo Dumilah (Puncak)",
        "jalur":    "Puncak",
        "lat":      -7.627324,
        "lon":      111.194387,
        "elevasi":  3265,
    },
]


def fetch_historical_weather(lat: float, lon: float,
                              start_date: str, end_date: str,
                              max_retries: int = 5) -> dict:
    """
    Mengambil data cuaca historis per jam dari Open-Meteo Archive API
    untuk satu titik koordinat, dengan retry + exponential backoff
    otomatis saat terkena rate limit (HTTP 429).

    Args:
        lat:         Latitude lokasi.
        lon:         Longitude lokasi.
        start_date:  Tanggal mulai (format: YYYY-MM-DD).
        end_date:    Tanggal selesai (format: YYYY-MM-DD).
        max_retries: Jumlah maksimum percobaan ulang (default: 5).

    Returns:
        dict: Respons JSON mentah dari API (field: 'hourly').

    Raises:
        requests.HTTPError: Jika semua retry gagal.
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     HOURLY_VARIABLES,
        "timezone":   "Asia/Jakarta",
    }

    for attempt in range(1, max_retries + 1):
        response = requests.get(BASE_URL, params=params, timeout=60)

        if response.status_code == 429:
            # Exponential backoff: 30s, 60s, 120s, 240s, 480s
            wait = 30 * (2 ** (attempt - 1))
            print(f"         [429] Rate limit. Menunggu {wait}s sebelum retry "
                  f"({attempt}/{max_retries})...")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    # Semua retry habis
    raise requests.HTTPError(
        f"Gagal setelah {max_retries} percobaan (rate limit persisten)."
    )


def parse_hourly_rows(raw: dict, lokasi: dict) -> list[dict]:
    """
    Mengurai respons JSON Open-Meteo (hourly) menjadi list baris per jam
    dengan menyertakan informasi lokasi.

    Args:
        raw:    Respons JSON dari fetch_historical_weather().
        lokasi: Dict lokasi dari LOKASI_GUNUNG_LAWU
                (field: nama_pos, jalur, lat, lon, elevasi).

    Returns:
        list[dict]: List baris per jam, siap ditulis ke CSV.
    """
    hourly = raw.get("hourly", {})

    def col(key: str) -> list:
        return hourly.get(key, [])

    timestamps      = col("time")
    suhu            = col("temperature_2m")
    suhu_terasa     = col("apparent_temperature")
    kelembaban      = col("relative_humidity_2m")
    curah_hujan     = col("precipitation")
    hujan           = col("rain")
    angin_kec       = col("wind_speed_10m")
    angin_arah      = col("wind_direction_10m")
    angin_kencang   = col("wind_gusts_10m")
    tutupan_awan    = col("cloud_cover")
    jarak_pandang   = col("visibility")
    tekanan_udara   = col("surface_pressure")
    kode_cuaca      = col("weather_code")

    def safe(lst: list, i: int):
        return lst[i] if i < len(lst) else None

    rows = []
    for i, ts in enumerate(timestamps):
        rows.append({
            "Timestamp":              ts,
            "Nama Pos":              lokasi["nama_pos"],
            "Jalur":                 lokasi["jalur"],
            "Elevasi (mdpl)":        lokasi["elevasi"],
            "Lat":                   lokasi["lat"],
            "Lon":                   lokasi["lon"],
            "Suhu (C)":              safe(suhu, i),
            "Suhu Terasa (C)":       safe(suhu_terasa, i),
            "Kelembaban (%)": safe(kelembaban, i),
            "Curah Hujan (mm)":      safe(curah_hujan, i),
            "Hujan (mm)":            safe(hujan, i),
            "Kecepatan Angin (km/h)":safe(angin_kec, i),
            "Arah Angin (derajat)": safe(angin_arah, i),
            "Angin Kencang (km/h)":  safe(angin_kencang, i),
            "Tutupan Awan (%)": safe(tutupan_awan, i),
            "Jarak Pandang (m)":     safe(jarak_pandang, i),
            "Tekanan Udara (hPa)":   safe(tekanan_udara, i),
            "Kode Cuaca WMO":        safe(kode_cuaca, i),
        })

    return rows


def fetch_all_locations(start_date: str, end_date: str,
                        delay_sec: float = 3.0,
                        skip_existing: set | None = None) -> tuple[list[dict], list[str]]:
    """
    Mengambil data cuaca historis per jam untuk semua lokasi
    di LOKASI_GUNUNG_LAWU.

    Args:
        start_date:     Tanggal mulai (format: YYYY-MM-DD).
        end_date:       Tanggal selesai (format: YYYY-MM-DD).
        delay_sec:      Jeda antar request ke API (detik). Default 3.0s.
        skip_existing:  Set nama_pos yang sudah berhasil diambil
                        (untuk resume setelah partial failure).

    Returns:
        tuple:
          - list[dict]: Gabungan semua baris dari lokasi yang berhasil.
          - list[str]:  List nama_pos yang gagal diambil.
    """
    all_rows: list[dict] = []
    failed:   list[str]  = []
    total = len(LOKASI_GUNUNG_LAWU)
    skip_existing = skip_existing or set()

    for idx, lokasi in enumerate(LOKASI_GUNUNG_LAWU, start=1):
        nama = lokasi["nama_pos"]

        if nama in skip_existing:
            print(f"  [{idx:02d}/{total}] SKIP (sudah ada): {nama}")
            continue

        print(f"  [{idx:02d}/{total}] Mengambil: {nama} "
              f"({lokasi['elevasi']} mdpl)...")
        try:
            raw  = fetch_historical_weather(lokasi["lat"], lokasi["lon"],
                                            start_date, end_date)
            rows = parse_hourly_rows(raw, lokasi)
            all_rows.extend(rows)
            print(f"         OK - {len(rows):,} jam")
        except requests.HTTPError as e:
            print(f"         GAGAL PERMANEN - {e}")
            failed.append(nama)

        if idx < total:
            time.sleep(delay_sec)

    sukses = total - len(failed) - len(skip_existing)
    print(f"\nSelesai: {sukses} lokasi OK, "
          f"{len(failed)} gagal, {len(skip_existing)} di-skip.")
    if failed:
        print(f"Lokasi gagal: {failed}")
    print(f"Total baris baru: {len(all_rows):,}")
    return all_rows, failed


def save_to_csv(rows: list[dict], output_path: str | Path,
                mode: str = "a") -> None:
    """
    Menyimpan list baris hourly ke file CSV.
    Default mode 'a' (append) agar bisa resume tanpa kehilangan data
    yang sudah tersimpan sebelumnya.

    Args:
        rows:        List baris dari fetch_all_locations() atau parse_hourly_rows().
        output_path: Path file CSV tujuan.
        mode:        'w' = tulis ulang dari awal, 'a' = append (default).
    """
    if not rows:
        print("Tidak ada baris untuk disimpan.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_baru = not output_path.exists() or mode == "w"

    with open(output_path, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if file_baru:
            writer.writeheader()
        writer.writerows(rows)

    aksi = "Ditulis" if file_baru else "Ditambahkan"
    print(f"{aksi}: {len(rows):,} baris -> {output_path}")


def get_existing_locations(output_path: str | Path) -> set[str]:
    """
    Membaca CSV yang sudah ada dan mengembalikan set nama_pos
    yang sudah berhasil tersimpan (untuk fitur resume).

    Args:
        output_path: Path file CSV yang sudah ada.

    Returns:
        set[str]: Set nama_pos yang sudah ada di CSV.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return set()

    existing: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(row.get("Nama Pos", ""))

    return existing

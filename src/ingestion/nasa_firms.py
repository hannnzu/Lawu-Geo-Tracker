"""
ingestion/nasa_firms.py
------------------------
Extract data titik api (kebakaran hutan) historis dari NASA FIRMS API
untuk area Gunung Lawu.

Sumber  : https://firms.modaps.eosdis.nasa.gov/api/area/
Sensor  : VIIRS Suomi-NPP Standard Processing (VIIRS_SNPP_SP)
Resolusi: 375 meter per pixel
Akses   : Gratis, butuh MAP_KEY (daftar di https://firms.modaps.eosdis.nasa.gov/api/map_key/)
Output  : CSV dengan kolom titik api per deteksi satelit

Endpoint:
  GET /api/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{DAY_RANGE}/{DATE}
  - BBOX format : west,south,east,north
  - DAY_RANGE   : 1-5 hari per request
  - DATE        : YYYY-MM-DD (tanggal mulai batch)
"""

import csv
import io
import time
from datetime import date, timedelta
from pathlib import Path

import requests


# =============================================================================
# Konstanta
# =============================================================================

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Sensor VIIRS Suomi-NPP Standard Processing (resolusi 375m, divalidasi)
FIRMS_SOURCE   = "VIIRS_SNPP_SP"

# Jumlah hari per request (max 5 per dokumentasi API)
DAY_RANGE      = 5

# Bounding box area Gunung Lawu: west,south,east,north
LAWU_BBOX      = "111.10,-7.75,111.25,-7.55"

# Kolom output CSV yang disimpan (subset dari kolom VIIRS asli)
CSV_HEADER = [
    "Tanggal",
    "Waktu (UTC)",
    "Lat",
    "Lon",
    "Kecerahan Ti4 (K)",
    "Kecerahan Ti5 (K)",
    "FRP (MW)",
    "Keyakinan",
    "Siang/Malam",
    "Satelit",
    "Versi",
]

# Mapping dari nama kolom API FIRMS ke nama kolom output kita
COLUMN_MAP = {
    "acq_date":   "Tanggal",
    "acq_time":   "Waktu (UTC)",
    "latitude":   "Lat",
    "longitude":  "Lon",
    "bright_ti4": "Kecerahan Ti4 (K)",
    "bright_ti5": "Kecerahan Ti5 (K)",
    "frp":        "FRP (MW)",
    "confidence": "Keyakinan",
    "daynight":   "Siang/Malam",
    "satellite":  "Satelit",
    "version":    "Versi",
}


# =============================================================================
# Fungsi Pengambilan Data
# =============================================================================

def fetch_fire_batch(map_key: str, start_date: str,
                     day_range: int = DAY_RANGE,
                     max_retries: int = 3) -> list[dict]:
    """
    Mengambil data titik api dari NASA FIRMS untuk satu batch tanggal.

    Args:
        map_key:     MAP_KEY NASA FIRMS (32 karakter).
        start_date:  Tanggal mulai batch (format: YYYY-MM-DD).
        day_range:   Jumlah hari per request (1-5). Default: 5.
        max_retries: Jumlah percobaan ulang saat error. Default: 3.

    Returns:
        list[dict]: List baris titik api yang sudah di-rename kolomnya.
                    Kosong jika tidak ada deteksi pada periode tersebut.

    Raises:
        requests.HTTPError: Jika semua retry gagal.
    """
    url = f"{FIRMS_BASE_URL}/{map_key}/{FIRMS_SOURCE}/{LAWU_BBOX}/{day_range}/{start_date}"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=30)

            # 429 Rate limit
            if resp.status_code == 429:
                wait = 60 * attempt
                print(f"         [429] Rate limit. Tunggu {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            text = resp.text.strip()

            # Respons kosong atau hanya header = tidak ada deteksi
            if not text or text.count("\n") == 0:
                return []

            # Parse CSV teks
            reader = csv.DictReader(io.StringIO(text))
            rows = []
            for raw_row in reader:
                row = {}
                for api_col, out_col in COLUMN_MAP.items():
                    row[out_col] = raw_row.get(api_col, "")
                rows.append(row)
            return rows

        except requests.ConnectionError as e:
            if attempt < max_retries:
                print(f"         [ERR] Koneksi gagal. Retry {attempt}/{max_retries}...")
                time.sleep(10 * attempt)
            else:
                raise requests.HTTPError(str(e))

    raise requests.HTTPError(
        f"Gagal setelah {max_retries} percobaan."
    )


def fetch_all_fire_data(map_key: str, start_date: str, end_date: str,
                        resume_from: str | None = None,
                        delay_sec: float = 1.0) -> list[dict]:
    """
    Mengambil semua data titik api dari NASA FIRMS untuk rentang tanggal penuh,
    dengan iterasi per DAY_RANGE hari.

    Args:
        map_key:     MAP_KEY NASA FIRMS.
        start_date:  Tanggal mulai (format: YYYY-MM-DD).
        end_date:    Tanggal selesai (format: YYYY-MM-DD).
        resume_from: Tanggal resume (lewati batch sebelum tanggal ini).
                     Gunakan get_last_saved_date() untuk mendapatkan nilainya.
        delay_sec:   Jeda antar request (detik). Default: 1.0s.

    Returns:
        list[dict]: Semua baris titik api yang ditemukan.
    """
    start  = date.fromisoformat(start_date)
    end    = date.fromisoformat(end_date)
    resume = date.fromisoformat(resume_from) if resume_from else start

    all_rows: list[dict] = []
    current  = start
    total_batches = ((end - start).days // DAY_RANGE) + 1
    batch_num = 0

    while current <= end:
        batch_num += 1
        batch_end = min(current + timedelta(days=DAY_RANGE - 1), end)

        # Skip batch yang sudah diambil sebelumnya
        if current < resume:
            current += timedelta(days=DAY_RANGE)
            continue

        pct = int(batch_num / total_batches * 100)
        print(f"  [{pct:3d}%] Batch {current} s/d {batch_end}...", end=" ")

        try:
            rows = fetch_fire_batch(map_key, current.isoformat())
            if rows:
                all_rows.extend(rows)
                print(f"{len(rows)} deteksi")
            else:
                print("0 deteksi (tidak ada kebakaran)")
        except requests.HTTPError as e:
            print(f"GAGAL - {e}")

        current += timedelta(days=DAY_RANGE)
        time.sleep(delay_sec)

    print(f"\nTotal deteksi titik api: {len(all_rows):,}")
    return all_rows


# =============================================================================
# Fungsi I/O
# =============================================================================

def save_to_csv(rows: list[dict], output_path: str | Path,
                mode: str = "a") -> None:
    """
    Menyimpan list baris titik api ke file CSV.

    Args:
        rows:        List baris dari fetch_all_fire_data().
        output_path: Path file CSV tujuan.
        mode:        'w' = tulis ulang, 'a' = append (default).
    """
    if not rows:
        print("Tidak ada baris untuk disimpan.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_baru = not output_path.exists() or mode == "w"

    with open(output_path, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        if file_baru:
            writer.writeheader()
        writer.writerows(rows)

    aksi = "Ditulis" if file_baru else "Ditambahkan"
    print(f"{aksi}: {len(rows):,} baris -> {output_path}")


def get_last_saved_date(output_path: str | Path) -> str | None:
    """
    Membaca CSV yang sudah ada dan mengembalikan tanggal deteksi terakhir
    (kolom 'Tanggal') untuk fitur resume.

    Args:
        output_path: Path file CSV yang sudah ada.

    Returns:
        str | None: Tanggal terakhir (YYYY-MM-DD) atau None jika file belum ada.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return None

    last_date = None
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("Tanggal", "")
            if d and (last_date is None or d > last_date):
                last_date = d

    return last_date

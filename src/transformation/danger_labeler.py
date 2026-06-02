"""
src/transformation/danger_labeler.py
------------------------------------
Modul untuk menghitung tingkat bahaya (danger_level) skala 0-3 secara rule-based:
- 0 = AMAN (Normal / Safe)
- 1 = WASPADA (Caution)
- 2 = BAHAYA (Danger)
- 3 = DILARANG (Forbidden / Closed)
"""

from typing import Any


def hitung_danger_level(
    suhu_terasa_c: float,
    angin_kencang_kmh: float,
    curah_hujan_mm: float,
    jarak_pandang_m: float | None,
    kode_cuaca_wmo: int,
    status_kebakaran_sekitar: int,
    jarak_titik_api_terdekat_km: float
) -> int:
    """
    Menghitung tingkat bahaya (0-3) berdasarkan beberapa aturan cuaca dan kebencanaan.
    """
    # 1. Aturan Suhu Terasa (Windchill)
    # Suhu sangat dingin (< 0 C) berbahaya akibat hipotermia, (< 5 C) perlu waspada.
    r1 = 2 if suhu_terasa_c < 0.0 else (1 if suhu_terasa_c < 5.0 else 0)

    # 2. Aturan Angin Kencang (Gusts)
    # Angin badai sangat ekstrem (> 100 km/h) dilarang, (> 80 km/h) bahaya, (> 50 km/h) waspada.
    r2 = 3 if angin_kencang_kmh > 100.0 else (2 if angin_kencang_kmh > 80.0 else (1 if angin_kencang_kmh > 50.0 else 0))

    # 3. Aturan Curah Hujan
    # Hujan deras (> 20mm/jam) bahaya banjir/longsor, (> 10mm/jam) waspada.
    r3 = 2 if curah_hujan_mm > 20.0 else (1 if curah_hujan_mm > 10.0 else 0)

    # 4. Aturan Jarak Pandang (Skenario C - Proxy WMO jika NULL)
    # Jika bernilai NULL/None, diestimasi dari kode WMO:
    # WMO 45, 48 = Kabut (Level 2)
    # WMO 51-67, 80-82 = Hujan/Gerimis/Showers (Level 1)
    if jarak_pandang_m is None:
        if kode_cuaca_wmo in (45, 48):
            r4 = 2
        elif kode_cuaca_wmo in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
            r4 = 1
        else:
            r4 = 0
    else:
        r4 = 2 if jarak_pandang_m < 200.0 else (1 if jarak_pandang_m < 500.0 else 0)

    # 5. Aturan Kode Cuaca WMO (Badai Petir)
    # WMO 95, 96, 99 adalah badai petir / thunderstorm yang sangat berbahaya di gunung.
    r5 = 3 if kode_cuaca_wmo in (95, 96, 99) else 0

    # 6. Aturan Status Kebakaran Sekitar
    # Jika status kebakaran di radius terdekat aktif, status DILARANG.
    r6 = 3 if status_kebakaran_sekitar == 1 else 0

    # 7. Aturan Jarak Titik Api Terdekat
    # Jika ada titik api aktif dalam jarak kurang dari 1.0 KM, status DILARANG.
    r7 = 3 if jarak_titik_api_terdekat_km < 1.0 else 0

    # Ambil nilai maksimum dari seluruh aturan bahaya
    return max(r1, r2, r3, r4, r5, r6, r7)


def hitung_danger_level_dari_row_csv(row: dict[str, str]) -> int:
    """
    Helper untuk menghitung danger level langsung dari baris CSV mentah (berupa string).
    """
    try:
        suhu_terasa = float(row.get("Suhu Terasa (C)", 0.0) or 0.0)
        angin_kencang = float(row.get("Angin Kencang (km/h)", 0.0) or 0.0)
        curah_hujan = float(row.get("Curah Hujan (mm)", 0.0) or 0.0)
        
        jp_str = row.get("Jarak Pandang (m)", "").strip()
        jarak_pandang = float(jp_str) if jp_str else None
        
        wmo = int(float(row.get("Kode Cuaca WMO", 0) or 0))
        status_kebakaran = int(row.get("Status_Kebakaran_Sekitar", 0) or 0)
        jarak_api = float(row.get("Jarak_Titik_Api_Terdekat_KM", 999.0) or 999.0)
        
        return hitung_danger_level(
            suhu_terasa_c=suhu_terasa,
            angin_kencang_kmh=angin_kencang,
            curah_hujan_mm=curah_hujan,
            jarak_pandang_m=jarak_pandang,
            kode_cuaca_wmo=wmo,
            status_kebakaran_sekitar=status_kebakaran,
            jarak_titik_api_terdekat_km=jarak_api
        )
    except Exception as e:
        # Jika ada error parsing, default ke 1 (Waspada/Caution) demi keselamatan
        return 1

import requests
import json

print("Memulai proses Extract data dari OpenStreetMap...")

API_URL = "http://overpass-api.de/api/interpreter"

kueri_osm = """
[out:json][timeout:25];
(
  /* Penjelasan Perubahan:
     1. "highway"~"path|footway" -> Mengambil tipe 'path' ATAU 'footway'
     2. "name"~"cemoro|lawu",i -> Mencari nama yang mengandung 'cemoro' ATAU 'lawu'.
        Huruf 'i' di belakang berarti Case-Insensitive (mengabaikan huruf besar/kecil).
  */
  way["highway"~"path|footway"]["name"~"cemoro|lawu",i](-7.660, 111.150, -7.600, 111.220);
  
  /* Opsi Backup: Jika ternyata relawan OSM tidak memberi tag 'name' sama sekali pada jalur tersebut,
     kita ambil saja SEMUA jalan setapak di kotak area yang lebih sempit (fokus di lereng Lawu).
     Hapus tanda '//' di baris bawah ini jika hasil di atas masih 0.
  */
  // way["highway"~"path|footway"](-7.640, 111.180, -7.610, 111.210);
);
out geom;
"""

response = requests.post(API_URL, data={'data': kueri_osm})
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
data_mentah = response.json()

print(f"Berhasil Extract! Ditemukan {len(data_mentah['elements'])} segmen jalur.")

print("Melakukan Transformasi data mentah menjadi GeoJSON...")

fitur_geojson = []

for elemen in data_mentah['elements']:
    if elemen['type'] == 'way':
        koordinat = [[titik['lon'], titik['lat']] for titik in elemen['geometry']]
        
        fitur = {
            "type": "Feature",
            "properties": {
                "nama_jalur": elemen['tags'].get('name', 'Tidak diketahui'),
                "jenis": elemen['tags'].get('highway', 'path'),
                "sumber": "OpenStreetMap via Python"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": koordinat
            }
        }
        fitur_geojson.append(fitur)
geojson_akhir = {
    "type": "FeatureCollection",
    "features": fitur_geojson
}
print("Load: Menyimpan file GeoJSON...")
nama_file = "jalur_cemoro_sewu.geojson"
with open(nama_file, 'w') as f:
    json.dump(geojson_akhir, f, indent=4)
print(f"Selesai! File berhasil disimpan dengan nama: {nama_file}")
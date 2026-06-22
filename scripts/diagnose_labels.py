import pandas as pd

df = pd.read_csv('DATA/curated/dataset_integrated_lawu_2021_2025.csv', nrows=100000)

print('=== LEVEL 3: Penyebab sebenarnya ===')
lvl3 = df[df['Danger_Level'] == 3]
print(f'Total Level 3: {len(lvl3)}')
print(f'Angin > 100 km/h: {(lvl3["Angin Kencang (km/h)"] > 100).sum()}')
print(f'WMO 95/96/99:     {lvl3["Kode Cuaca WMO"].isin([95,96,99]).sum()}')
print(f'Api < 1KM:        {(lvl3["Jarak_Titik_Api_Terdekat_KM"] < 1).sum()}')
print(f'Suhu < 0C:        {(lvl3["Suhu Terasa (C)"] < 0).sum()}')
print(f'Max angin di Level 3: {lvl3["Angin Kencang (km/h)"].max():.1f} km/h')
print(f'Min angin di Level 3: {lvl3["Angin Kencang (km/h)"].min():.1f} km/h')

print()
print('=== LEVEL 2: Penyebab sebenarnya ===')
lvl2 = df[df['Danger_Level'] == 2]
print(f'Total Level 2: {len(lvl2)}')
print(f'Angin > 80 km/h:  {(lvl2["Angin Kencang (km/h)"] > 80).sum()}')
print(f'Hujan > 20mm:     {(lvl2["Curah Hujan (mm)"] > 20).sum()}')
print(f'Suhu < 0C:        {(lvl2["Suhu Terasa (C)"] < 0).sum()}')
print(f'WMO 45/48:        {lvl2["Kode Cuaca WMO"].isin([45,48]).sum()}')

print()
print('=== Apakah fitur MENDETERMINASI label? ===')
# Cek apakah Angin > 100 selalu menghasilkan Level 3
mask_angin100 = df['Angin Kencang (km/h)'] > 100
print(f'Baris angin > 100 km/h: {mask_angin100.sum()}')
print(f'  -> Level 3: {(df[mask_angin100]["Danger_Level"] == 3).sum()}')
print(f'  -> Level lain: {(df[mask_angin100]["Danger_Level"] != 3).sum()}')

print()
print('=== Verifikasi: apakah Angin_Kencang 100% menjadi penentu? ===')
mask_lvl3 = df['Danger_Level'] == 3
pct_karena_angin = (df[mask_lvl3]['Angin Kencang (km/h)'] > 100).mean() * 100
print(f'Level 3 yang disebabkan angin > 100: {pct_karena_angin:.1f}%')

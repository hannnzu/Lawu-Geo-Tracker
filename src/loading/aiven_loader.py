"""
src/loading/aiven_loader.py
----------------------------
Modul untuk membuat schema database dan melakukan pemuatan data (load)
ke Aiven PostgreSQL + TimescaleDB secara efisien.
"""

import csv
import io
import time
from pathlib import Path
from sqlalchemy import create_engine, text
from src.utils.logger import get_logger
from src.ingestion.open_meteo import LOKASI_GUNUNG_LAWU

logger = get_logger("aiven_loader")


def get_engine(database_url: str):
    """
    Membuat engine SQLAlchemy untuk koneksi ke database Aiven PostgreSQL.
    Mendukung auto-correct untuk prefix postgres:// menjadi postgresql://.
    """
    if not database_url:
        raise ValueError("Database URL kosong. Pastikan AIVEN_DATABASE_URL sudah diatur di file .env")

    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    logger.info("Membuat engine database SQLAlchemy...")
    # Menambahkan pool_pre_ping untuk mendeteksi koneksi terputus secara otomatis
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require"}
    )


def ensure_schema(engine) -> None:
    """
    Membuat tabel-tabel yang diperlukan dan mengaktifkan ekstensi TimescaleDB
    serta mendefinisikan hypertable jika belum ada.
    """
    logger.info("Memulai pembuatan schema database...")
    
    with engine.connect() as conn:
        # 1. Pastikan ekstensi TimescaleDB aktif
        logger.info("Memastikan ekstensi timescaledb telah aktif...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        conn.commit()

        # 2. Buat tabel pos_pendakian
        logger.info("Membuat tabel 'pos_pendakian' jika belum ada...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pos_pendakian (
                nama_pos     VARCHAR(100) PRIMARY KEY,
                jalur        VARCHAR(50)  NOT NULL,
                elevasi_mdpl INTEGER      NOT NULL,
                lat          FLOAT        NOT NULL,
                lon          FLOAT        NOT NULL
            );
        """))
        conn.commit()

        # 3. Buat tabel titik_api
        logger.info("Membuat tabel 'titik_api' jika belum ada...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS titik_api (
                id            SERIAL PRIMARY KEY,
                tanggal       DATE         NOT NULL,
                waktu_utc     VARCHAR(10),
                lat           FLOAT        NOT NULL,
                lon           FLOAT        NOT NULL,
                kecerahan_ti4 FLOAT,
                kecerahan_ti5 FLOAT,
                frp_mw        FLOAT,
                keyakinan     VARCHAR(5),
                siang_malam   VARCHAR(5),
                satelit       VARCHAR(10),
                versi         VARCHAR(10)
            );
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_titik_api_tanggal ON titik_api (tanggal);"))
        conn.commit()

        # 4. Buat tabel jalur_pendakian
        logger.info("Membuat tabel 'jalur_pendakian' jika belum ada...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS jalur_pendakian (
                id              SERIAL PRIMARY KEY,
                nama_jalur      VARCHAR(100) NOT NULL,
                urutan_titik    INTEGER NOT NULL,
                lat             FLOAT NOT NULL,
                lon             FLOAT NOT NULL,
                elevasi_mdpl    INTEGER,
                kemiringan_pct  FLOAT,
                jarak_dari_basecamp_km FLOAT,
                akumulasi_gain_m FLOAT,
                sumber_file     VARCHAR(200),
                terrain_type    VARCHAR(50)
            );
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_jalur_nama ON jalur_pendakian (nama_jalur);"))
        conn.commit()

        # 5. Buat tabel cuaca_integrated
        logger.info("Membuat tabel 'cuaca_integrated' jika belum ada...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cuaca_integrated (
                timestamp                     TIMESTAMPTZ     NOT NULL,
                nama_pos                      VARCHAR(100)    NOT NULL REFERENCES pos_pendakian(nama_pos),
                jalur                         VARCHAR(50),
                elevasi_mdpl                  INTEGER,
                lat                           FLOAT,
                lon                           FLOAT,
                suhu_c                        FLOAT,
                suhu_terasa_c                 FLOAT,
                kelembaban_pct                FLOAT,
                curah_hujan_mm                FLOAT,
                hujan_mm                      FLOAT,
                kecepatan_angin_kmh           FLOAT,
                arah_angin_derajat            FLOAT,
                angin_kencang_kmh             FLOAT,
                tutupan_awan_pct              FLOAT,
                jarak_pandang_m               FLOAT,
                tekanan_udara_hpa             FLOAT,
                kode_cuaca_wmo                SMALLINT,
                jarak_titik_api_terdekat_km   FLOAT,
                frp_terdekat_mw               FLOAT,
                status_kebakaran_sekitar      SMALLINT,
                danger_level                  SMALLINT DEFAULT 0
            );
        """))
        conn.commit()

        # 6. Memastikan kolom 'danger_level' ada (untuk database yang sudah terisi sebelumnya)
        logger.info("Memastikan kolom 'danger_level' ada di tabel 'cuaca_integrated'...")
        conn.execute(text("ALTER TABLE cuaca_integrated ADD COLUMN IF NOT EXISTS danger_level SMALLINT DEFAULT 0;"))
        conn.commit()

        # 7. Konfigurasi TimescaleDB Hypertable
        logger.info("Memeriksa status hypertable untuk 'cuaca_integrated'...")
        res = conn.execute(text(
            "SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'cuaca_integrated'"
        )).fetchone()

        if not res:
            logger.info("Mengubah 'cuaca_integrated' menjadi TimescaleDB hypertable (interval 1 bulan)...")
            conn.execute(text(
                "SELECT create_hypertable('cuaca_integrated', 'timestamp', chunk_time_interval => INTERVAL '1 month');"
            ))
            conn.commit()
        else:
            logger.info("Tabel 'cuaca_integrated' sudah berupa hypertable.")

        # 8. Buat indeks tambahan untuk cuaca_integrated
        logger.info("Membuat indeks tambahan untuk optimasi query...")
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cuaca_nama_pos_ts ON cuaca_integrated (nama_pos, timestamp DESC);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cuaca_jalur_ts ON cuaca_integrated (jalur, timestamp DESC);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cuaca_danger_level ON cuaca_integrated (danger_level);"))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_cuaca_status_api 
            ON cuaca_integrated (status_kebakaran_sekitar) 
            WHERE status_kebakaran_sekitar = 1;
        """))
        conn.commit()

    logger.info("Pembuatan schema database selesai dengan sukses.")


def load_pos_pendakian(engine) -> int:
    """
    Memuat data statis pos pendakian dari list LOKASI_GUNUNG_LAWU.
    Menggunakan ON CONFLICT DO UPDATE untuk menghindari duplikasi dan
    memperbarui nilai jika ada perubahan koordinat.
    """
    logger.info("Memuat data 'pos_pendakian'...")
    rows_inserted = 0
    
    query = text("""
        INSERT INTO pos_pendakian (nama_pos, jalur, elevasi_mdpl, lat, lon)
        VALUES (:nama_pos, :jalur, :elevasi, :lat, :lon)
        ON CONFLICT (nama_pos) 
        DO UPDATE SET 
            jalur = EXCLUDED.jalur,
            elevasi_mdpl = EXCLUDED.elevasi_mdpl,
            lat = EXCLUDED.lat,
            lon = EXCLUDED.lon;
    """)

    with engine.connect() as conn:
        for lokasi in LOKASI_GUNUNG_LAWU:
            conn.execute(query, {
                "nama_pos": lokasi["nama_pos"],
                "jalur": lokasi["jalur"],
                "elevasi": lokasi["elevasi"],
                "lat": lokasi["lat"],
                "lon": lokasi["lon"]
            })
            rows_inserted += 1
        conn.commit()

    logger.info(f"Berhasil memuat/memperbarui {rows_inserted} baris ke 'pos_pendakian'.")
    return rows_inserted


def load_titik_api(engine, csv_path: Path) -> int:
    """
    Memuat data deteksi titik api dari CSV ke Aiven menggunakan PostgreSQL COPY.
    Sebelum memuat, tabel akan dikosongkan terlebih dahulu (TRUNCATE) untuk
    memastikan data bersih tanpa duplikasi.
    """
    logger.info(f"Membaca file titik api dari: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"File titik api tidak ditemukan di {csv_path}")

    # Ambil baris data untuk pelaporan
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    logger.info(f"Menghapus data lama di 'titik_api'...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE titik_api RESTART IDENTITY CASCADE;"))
        conn.commit()

    logger.info(f"Memulai bulk load {row_count} baris ke 'titik_api' via COPY...")
    
    start_time = time.time()
    
    # Ambil objek raw connection dari DBAPI psycopg2 untuk copy_expert
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # Gunakan COPY SQL dengan opsi NULL
            # Kolom di mapping eksplisit karena tabel memiliki SERIAL id
            copy_sql = """
                COPY titik_api (
                    tanggal, waktu_utc, lat, lon, kecerahan_ti4, 
                    kecerahan_ti5, frp_mw, keyakinan, siang_malam, satelit, versi
                ) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '');
            """
            with open(csv_path, mode="r", encoding="utf-8") as f:
                cur.copy_expert(sql=copy_sql, file=f)
        raw_conn.commit()
    except Exception as e:
        raw_conn.rollback()
        logger.error(f"Gagal memuat titik_api: {e}")
        raise e
    finally:
        raw_conn.close()

    duration = time.time() - start_time
    logger.info(f"Berhasil memuat {row_count} baris ke 'titik_api' dalam {duration:.2f} detik.")
    return row_count


def load_cuaca_integrated(engine, csv_path: Path) -> int:
    """
    Memuat dataset cuaca terpadu skala besar (~830rb baris) dari CSV ke Aiven 
    menggunakan PostgreSQL COPY.
    Sebelum memuat, tabel akan dikosongkan terlebih dahulu (TRUNCATE) untuk
    memastikan data bersih tanpa duplikasi.
    """
    logger.info(f"Membaca file cuaca terintegrasi dari: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"File cuaca terintegrasi tidak ditemukan di {csv_path}")

    # Menghitung baris secara efisien (menggunakan buffer generator)
    logger.info("Menghitung jumlah baris CSV...")
    with open(csv_path, mode="r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1 # Kurangi header

    logger.info(f"Menghapus data lama di 'cuaca_integrated'...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE cuaca_integrated CASCADE;"))
        conn.commit()

    logger.info(f"Memulai bulk load {row_count:,} baris ke 'cuaca_integrated' via COPY...")
    
    start_time = time.time()
    
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # Karena skema kolom di DB sama persis urutannya dengan header CSV, 
            # kita tidak wajib mendefinisikan kolom, tapi baik dilakukan agar aman.
            copy_sql = """
                COPY cuaca_integrated (
                    timestamp, nama_pos, jalur, elevasi_mdpl, lat, lon,
                    suhu_c, suhu_terasa_c, kelembaban_pct, curah_hujan_mm,
                    hujan_mm, kecepatan_angin_kmh, arah_angin_derajat,
                    angin_kencang_kmh, tutupan_awan_pct, jarak_pandang_m,
                    tekanan_udara_hpa, kode_cuaca_wmo, jarak_titik_api_terdekat_km,
                    frp_terdekat_mw, status_kebakaran_sekitar
                ) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '');
            """
            with open(csv_path, mode="r", encoding="utf-8") as f:
                cur.copy_expert(sql=copy_sql, file=f)
        raw_conn.commit()
    except Exception as e:
        raw_conn.rollback()
        logger.error(f"Gagal memuat cuaca_integrated: {e}")
        raise e
    finally:
        raw_conn.close()

    duration = time.time() - start_time
    logger.info(f"Berhasil memuat {row_count:,} baris ke 'cuaca_integrated' dalam {duration:.2f} detik.")
    return row_count


def load_jalur_pendakian(engine, points: list[dict]) -> int:
    """
    Memuat data jalur pendakian ke database Aiven PostgreSQL.
    Sebelum memuat, data lama akan dihapus (TRUNCATE) untuk menghindari duplikasi.
    """
    logger.info("Menghapus data lama di 'jalur_pendakian'...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE jalur_pendakian RESTART IDENTITY CASCADE;"))
        conn.commit()

    if not points:
        logger.warning("Tidak ada data titik jalur yang akan dimuat.")
        return 0

    logger.info(f"Memulai pemuatan {len(points)} titik jalur ke 'jalur_pendakian'...")
    query = text("""
        INSERT INTO jalur_pendakian (
            nama_jalur, urutan_titik, lat, lon, elevasi_mdpl,
            kemiringan_pct, jarak_dari_basecamp_km, akumulasi_gain_m,
            sumber_file, terrain_type
        ) VALUES (
            :nama_jalur, :urutan_titik, :lat, :lon, :elevasi_mdpl,
            :kemiringan_pct, :jarak_dari_basecamp_km, :akumulasi_gain_m,
            :sumber_file, :terrain_type
        );
    """)
    
    start_time = time.time()
    with engine.connect() as conn:
        conn.execute(query, points)
        conn.commit()
        
    duration = time.time() - start_time
    logger.info(f"Berhasil memuat {len(points)} titik ke 'jalur_pendakian' dalam {duration:.2f} detik.")
    return len(points)


def update_danger_level_in_db(engine) -> int:
    """
    Memperbarui kolom danger_level di tabel cuaca_integrated berdasarkan rule-based logic
    secara langsung di server database untuk efisiensi maksimal.
    """
    logger.info("Memperbarui kolom 'danger_level' di tabel 'cuaca_integrated' via SQL server-side...")
    start_time = time.time()
    
    query = text("""
        UPDATE cuaca_integrated
        SET danger_level = GREATEST(
            -- Rule 1: Suhu Terasa
            CASE 
                WHEN suhu_terasa_c < 0 THEN 2 
                WHEN suhu_terasa_c < 5 THEN 1 
                ELSE 0 
            END,
            -- Rule 2: Angin Kencang
            CASE 
                WHEN angin_kencang_kmh > 100 THEN 3 
                WHEN angin_kencang_kmh > 80 THEN 2 
                WHEN angin_kencang_kmh > 50 THEN 1 
                ELSE 0 
            END,
            -- Rule 3: Curah Hujan
            CASE 
                WHEN curah_hujan_mm > 20 THEN 2 
                WHEN curah_hujan_mm > 10 THEN 1 
                ELSE 0 
            END,
            -- Rule 4: Jarak Pandang (NULL dianggap kabut/danger level 1)
            CASE 
                WHEN jarak_pandang_m IS NULL THEN 1
                WHEN jarak_pandang_m < 200 THEN 2 
                WHEN jarak_pandang_m < 500 THEN 1 
                ELSE 0 
            END,
            -- Rule 5: Kode Cuaca WMO
            CASE 
                WHEN kode_cuaca_wmo IN (95, 96, 99) THEN 3 
                ELSE 0 
            END,
            -- Rule 6: Status Kebakaran Sekitar
            CASE 
                WHEN status_kebakaran_sekitar = 1 THEN 3 
                ELSE 0 
            END,
            -- Rule 7: Jarak Titik Api Terdekat
            CASE 
                WHEN jarak_titik_api_terdekat_km < 1.0 THEN 3 
                ELSE 0 
            END
        );
    """)
    
    with engine.connect() as conn:
        res = conn.execute(query)
        conn.commit()
        row_count = res.rowcount
        
    duration = time.time() - start_time
    logger.info(f"Berhasil memperbarui {row_count:,} baris 'danger_level' di database dalam {duration:.2f} detik.")
    return row_count

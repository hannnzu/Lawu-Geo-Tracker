"""
loading/csv_writer.py
----------------------
Menulis data ke file CSV (append-mode untuk streaming/micro-batch).
"""

import csv
import os
from pathlib import Path


def write_csv(rows: list[dict], output_path: str | Path,
              mode: str = "a") -> None:
    """
    Menulis list dict ke CSV. Otomatis menulis header jika file baru.

    Args:
        rows: List baris data (list of dict).
        output_path: Path file CSV tujuan.
        mode: 'w' = overwrite, 'a' = append (default).
    """
    if not rows:
        print("Tidak ada data untuk ditulis.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kolom = list(rows[0].keys())
    file_baru = not output_path.exists() or mode == "w"

    with open(output_path, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=kolom)
        if file_baru:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Berhasil menulis {len(rows)} baris ke {output_path}")


def write_csv_overwrite(rows: list[dict], output_path: str | Path) -> None:
    """Shortcut: tulis/timpa CSV dari awal."""
    write_csv(rows, output_path, mode="w")


def write_csv_append(rows: list[dict], output_path: str | Path) -> None:
    """Shortcut: tambahkan baris baru ke CSV yang sudah ada."""
    write_csv(rows, output_path, mode="a")

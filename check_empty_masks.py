"""Diagnostico: conta quantas mascaras em annotations/ estao vazias."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="lakes")
    parser.add_argument("--delete-empty", action="store_true",
                        help="Apagar as mascaras vazias encontradas")
    parser.add_argument("--move-to-negatives", action="store_true",
                        help="Mover tiles vazios para lista de negativos")
    args = parser.parse_args()

    total = 0
    empty = 0
    empty_paths = []

    for year in Config.YEARS:
        d = Config.MASKS_DIR / str(year) / "annotations" / args.feature
        if not d.exists():
            continue
        for p in sorted(d.glob(f"tile_*_{args.feature}.png")):
            total += 1
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None or (m > 127).sum() == 0:
                empty += 1
                empty_paths.append(p)
                print(f"  VAZIO: {year}/{p.name}")

    print(f"\nTotal: {total} | Vazias: {empty} | Com lago: {total - empty}")

    if args.delete_empty and empty > 0:
        confirm = input(f"\nApagar {empty} mascaras vazias? (s/N): ")
        if confirm.lower() == "s":
            for p in empty_paths:
                p.unlink()
            print(f"  {len(empty_paths)} arquivos apagados.")


if __name__ == "__main__":
    main()

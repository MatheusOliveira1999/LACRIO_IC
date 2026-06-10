"""
00_normalize_resolution.py - Padroniza mosaicos RGB para resolucao comum

Mosaicos do Schiaparelli tem resolucoes nativas DIFERENTES (5.42 a 8.23 cm/px)
porque foram processados pelo Agisoft com configs distintas. Para comparacao
cross-year justa, este script reamostra os RGBs para uma resolucao comum
(default 8.0 cm/px) ANTES do tiling e inferencia.

Gera:
  Schiaparelli_glacier/normalized_8cm/Schiaparelli_mosaic_{ano}_8cm.tif

Depois desta etapa, rode:
  python 01_create_tiles.py --source-dir Schiaparelli_glacier/normalized_8cm \\
                            --output-dir tiles_8cm
  for Y in 2016 2017 2018 2019 2020; do
      python 04_inference_unet.py --feature crevasses --year $Y \\
          --threshold 0.5 --no-feature-filter --tiles-dir tiles_8cm
      python 05_reconstruct_mosaic.py --feature crevasses --year $Y \\
          --tiles-dir tiles_8cm --output-suffix _8cm
  done
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config import Config


YEARS_DEFAULT = [2016, 2017, 2018, 2019, 2020]


def find_rgb(year):
    base = Config.DATA_SOURCE_DIR
    for prefix in ("Schiaparelli_mosaic_", "schiaparelli_mosaic_"):
        p = base / f"{prefix}{year}.tif"
        if p.exists():
            return p
    return None


def resample_rgb(src_path, dst_path, target_res_m, resampling=Resampling.average):
    """Reamostra um raster RGB para resolucao alvo, preservando CRS e bounds."""
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise RuntimeError(f"{src_path}: sem CRS")

        left, bottom, right, top = src.bounds
        width_m = right - left
        height_m = top - bottom

        new_width = max(1, int(round(width_m / target_res_m)))
        new_height = max(1, int(round(height_m / target_res_m)))

        new_transform = rasterio.transform.from_bounds(
            left, bottom, right, top, new_width, new_height
        )

        profile = src.profile.copy()
        profile.update({
            "driver": "GTiff",
            "transform": new_transform,
            "width": new_width,
            "height": new_height,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        })

        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                buf = np.zeros((new_height, new_width), dtype=src.dtypes[b-1])
                reproject(
                    source=rasterio.band(src, b),
                    destination=buf,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=resampling,
                )
                dst.write(buf, b)

        return new_width, new_height


def main():
    parser = argparse.ArgumentParser(
        description="Normaliza resolucao dos mosaicos RGB para grade comum")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT)
    parser.add_argument("--target-cm", type=float, default=8.0,
                        help="Resolucao alvo em cm/px (default 8.0)")
    parser.add_argument("--out-dir", type=Path,
                        default=Config.DATA_SOURCE_DIR / "normalized_8cm")
    args = parser.parse_args()

    target_res = args.target_cm / 100.0  # metros
    suffix = f"{int(round(args.target_cm))}cm"

    print("=" * 60)
    print(f"NORMALIZACAO DE RESOLUCAO RGB -> {args.target_cm:.2f} cm/px")
    print("=" * 60)
    print(f"Saida: {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        rgb = find_rgb(year)
        if rgb is None:
            print(f"  {year}: RGB ausente")
            continue

        with rasterio.open(rgb) as src:
            orig_res = abs(src.transform.a) * 100
            orig_w, orig_h = src.width, src.height

        out = args.out_dir / f"{rgb.stem}_{suffix}.tif"

        # Skip se ja existe e nao for parcial (>= 100 MB indica completo)
        if out.exists() and out.stat().st_size > 100 * 1024 * 1024:
            print(f"  {year}: {orig_res:.2f}cm -> ja existe ({out.stat().st_size/1e9:.2f} GB), pulando")
            continue

        print(f"  {year}: {orig_res:.2f}cm ({orig_w}x{orig_h}) -> {args.target_cm:.2f}cm ",
              end="", flush=True)
        try:
            w, h = resample_rgb(rgb, out, target_res)
            print(f"OK ({w}x{h}) -> {out.name}")
        except Exception as e:
            print(f"ERRO: {e}")

    print(f"\nProxima etapa: tiling em 8cm")
    print(f"  python 01_create_tiles.py --source-dir {args.out_dir} "
          f"--output-dir {Config.PROJECT_DIR.name}/tiles_{suffix}")


if __name__ == "__main__":
    main()

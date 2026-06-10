"""
09_glaciological_analysis.py - Camada 3 da analise: glaciologia

Gera:
  - results/crevasse_geometries_{ano}.csv      (geometria individual por fenda)
  - results/figures/glaciological/altitude_dist.png
  - results/figures/glaciological/slope_dist.png
  - results/figures/glaciological/density_map_{ano}.png
  - results/figures/glaciological/persistence_map.png
  - results/figures/glaciological/dh_{ano1}_minus_{ano0}.png

Uso:
  python 09_glaciological_analysis.py
  python 09_glaciological_analysis.py --skip-dh   # sem analise de dH
  python 09_glaciological_analysis.py --skip-density --skip-persistence
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform

from config import Config
from shadow_utils import compute_slope_map


YEARS_DEFAULT = [2016, 2017, 2018, 2019]


# ============================================================================
# Helpers
# ============================================================================

USE_NORMALIZED = False  # setado pelo --normalized no main


def find_pred_mosaic(year):
    base = Config.RESULTS_DIR / str(year)
    if USE_NORMALIZED:
        normalized = sorted(base.glob(f"crevasses_mask_{year}_*cm.tif"))
        if normalized:
            return normalized[0]
    canonical = base / f"crevasses_mask_{year}.tif"
    if canonical.exists():
        return canonical
    variants = sorted(base.glob(f"crevasses_mask_{year}.tif"))
    variants += sorted(base.glob(f"crevasses_mask_{year}_[0-9].tif"))
    return variants[0] if variants else None


def find_dem(year):
    base = Config.DATA_SOURCE_DIR
    for prefix in ("Schiaparelli_DEM_", "schiaparelli_DEM_"):
        p = base / f"{prefix}{year}.tif"
        if p.exists():
            return p
    return None


def read_raster_full(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.transform, src.crs, src.shape, src.res


def read_downsampled(tif_path, max_dim, resampling=Resampling.bilinear, band=1):
    """Le um banda com downsampling."""
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        scale = max(1.0, max(h, w) / max_dim)
        new_h = int(h / scale)
        new_w = int(w / scale)
        arr = src.read(band, out_shape=(new_h, new_w), resampling=resampling)
        return arr, scale


def sample_dem_at_mask(dem_path, mask_path, max_samples=200_000):
    """Sampleia valores do DEM nos pixels onde mask > 127.

    Resample mask para grid do DEM (mais grosso) e usa onde mask >= 0.5.
    Retorna array 1D de valores DEM.
    """
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype(np.float32)
        if dem_src.nodata is not None:
            dem[dem == dem_src.nodata] = np.nan
        dem_h, dem_w = dem.shape
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs
        res_x = abs(dem_src.res[0])
        res_y = abs(dem_src.res[1])

    # Reprojeta a mascara para grid do DEM
    with rasterio.open(mask_path) as mask_src:
        if mask_src.crs != dem_crs:
            print(f"    [AVISO] CRS diferentes: mask={mask_src.crs} dem={dem_crs}")
        mask_resampled = np.empty((dem_h, dem_w), dtype=np.uint8)
        reproject(
            source=rasterio.band(mask_src, 1),
            destination=mask_resampled,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.max,  # preserva positivo
        )

    valid = (mask_resampled > 127) & np.isfinite(dem)
    values = dem[valid]
    if values.size > max_samples:
        rng = np.random.default_rng(42)
        values = rng.choice(values, max_samples, replace=False)
    return values, (res_x, res_y)


def sample_slope_at_mask(dem_path, mask_path, max_samples=200_000):
    """Slope (graus) nos pixels de fenda."""
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype(np.float32)
        if dem_src.nodata is not None:
            dem[dem == dem_src.nodata] = np.nan
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs
        res_x = abs(dem_src.res[0])
        res_y = abs(dem_src.res[1])
        dem_h, dem_w = dem.shape

    slope = compute_slope_map(dem, res_x, res_y)

    with rasterio.open(mask_path) as mask_src:
        mask_resampled = np.empty((dem_h, dem_w), dtype=np.uint8)
        reproject(
            source=rasterio.band(mask_src, 1),
            destination=mask_resampled,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.max,
        )

    valid = (mask_resampled > 127) & np.isfinite(slope) & (slope > 0)
    values = slope[valid]
    if values.size > max_samples:
        rng = np.random.default_rng(42)
        values = rng.choice(values, max_samples, replace=False)
    return values


# ============================================================================
# Geometria individual (CSV)
# ============================================================================

def per_crevasse_geometry(year, out_csv, min_area_px=10):
    pred_path = find_pred_mosaic(year)
    if pred_path is None:
        return None
    with rasterio.open(pred_path) as src:
        mask = src.read(1)
        transform = src.transform
        res_x = abs(src.res[0])
        res_y = abs(src.res[1])

    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    pixel_area_m2 = res_x * res_y

    rows = []
    for lid in range(1, num_labels):
        area_px = int(stats[lid, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        x = int(stats[lid, cv2.CC_STAT_LEFT])
        y = int(stats[lid, cv2.CC_STAT_TOP])
        w = int(stats[lid, cv2.CC_STAT_WIDTH])
        h = int(stats[lid, cv2.CC_STAT_HEIGHT])

        # Slice local via bounding box (evita criar mascara global de centenas de MB)
        comp = (labels[y:y+h, x:x+w] == lid).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = cnts[0]
        rect = cv2.minAreaRect(cnt)
        wr, hr = rect[1]
        major_px = max(wr, hr)
        minor_px = min(wr, hr)
        angle = rect[2]
        if wr < hr:
            angle += 90.0
        azimuth = (90.0 - angle) % 180.0  # 0=N

        # centroid em UTM
        cx_px = x + w / 2.0
        cy_px = y + h / 2.0
        cx_utm = transform.c + cx_px * transform.a + cy_px * transform.b
        cy_utm = transform.f + cx_px * transform.d + cy_px * transform.e

        rows.append({
            "year": year,
            "id": lid,
            "area_px": area_px,
            "area_m2": round(area_px * pixel_area_m2, 4),
            "length_m": round(major_px * res_x, 3),
            "width_m": round(minor_px * res_x, 3),
            "aspect_ratio": round(major_px / max(minor_px, 1e-6), 2),
            "azimuth_deg": round(azimuth, 2),
            "centroid_utm_x": round(cx_utm, 2),
            "centroid_utm_y": round(cy_utm, 2),
        })

    if not rows:
        return None
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out_csv, len(rows)


# ============================================================================
# Distribuicoes altitude/slope
# ============================================================================

def plot_altitude_distribution(years, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = ["#2980b9", "#16a085", "#d35400", "#c0392b"]
    summary = {}

    for color, year in zip(palette, years):
        dem = find_dem(year)
        mask = find_pred_mosaic(year)
        if dem is None or mask is None:
            continue
        print(f"  Sampleando DEM {year}...", end="", flush=True)
        values, _ = sample_dem_at_mask(dem, mask)
        if values.size == 0:
            print(" vazio")
            continue
        ax.hist(values, bins=60, alpha=0.45, color=color,
                label=f"{year}  (n={values.size:,})", density=True)
        summary[year] = (float(values.mean()), float(np.median(values)),
                         float(values.min()), float(values.max()))
        print(f" mean={values.mean():.0f}m  med={np.median(values):.0f}m")

    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Densidade de pixels de fenda")
    ax.set_title("Distribuição altitudinal das fendas (DEM)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return summary


def plot_slope_distribution(years, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = ["#2980b9", "#16a085", "#d35400", "#c0392b"]
    summary = {}

    for color, year in zip(palette, years):
        dem = find_dem(year)
        mask = find_pred_mosaic(year)
        if dem is None or mask is None:
            continue
        print(f"  Slope {year}...", end="", flush=True)
        values = sample_slope_at_mask(dem, mask)
        if values.size == 0:
            print(" vazio")
            continue
        ax.hist(values, bins=60, range=(0, 60), alpha=0.45, color=color,
                label=f"{year}  (n={values.size:,})", density=True)
        summary[year] = (float(values.mean()), float(np.median(values)))
        print(f" mean={values.mean():.1f}°  med={np.median(values):.1f}°")

    ax.set_xlabel("Slope (graus)")
    ax.set_ylabel("Densidade de pixels de fenda")
    ax.set_title("Distribuição de slope sob fendas")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return summary


# ============================================================================
# Mapa de densidade (kernel density agregado)
# ============================================================================

def density_map(year, out_path, kernel_m=50, max_dim=2000):
    """Densidade de fendas via convolucao com kernel circular de raio kernel_m."""
    pred_path = find_pred_mosaic(year)
    if pred_path is None:
        return None
    arr, scale = read_downsampled(pred_path, max_dim, Resampling.average)

    with rasterio.open(pred_path) as src:
        original_res = abs(src.res[0])
    effective_res = original_res * scale
    kernel_px = max(3, int(round(kernel_m / effective_res)))
    if kernel_px % 2 == 0:
        kernel_px += 1

    binary = (arr > 64).astype(np.float32)
    blurred = cv2.GaussianBlur(binary, (kernel_px, kernel_px), kernel_px / 3.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(blurred, cmap="hot", interpolation="bilinear")
    ax.set_title(f"Densidade de fendas {year}  "
                 f"(kernel ~{kernel_m}m · downsample {scale:.1f}×)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, label="densidade local")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


# ============================================================================
# Persistencia cross-year (reprojeta todos para grid comum)
# ============================================================================

def persistence_map(years, out_path, min_years=3, max_dim=2500):
    """Pixel = positivo se aparece em >= min_years.

    Estrategia: usa grid do menor mosaico (em pixels) como referencia
    e reprojeta todos pra ele.
    """
    paths = [(y, find_pred_mosaic(y)) for y in years]
    paths = [(y, p) for y, p in paths if p is not None]
    if len(paths) < min_years:
        print(f"  [PULADO] persistencia: precisa de >= {min_years} anos com mosaico")
        return None

    # Referencia: o mosaico com MENOR area (ou pegamos o 2018 como referencia padrao)
    ref_year, ref_path = paths[0]
    for y, p in paths:
        with rasterio.open(p) as s:
            if y == 2018:  # 2018 e do meio, boa referencia
                ref_year, ref_path = y, p
                break
    print(f"  Grid de referencia: {ref_year}")

    with rasterio.open(ref_path) as src:
        ref_h, ref_w = src.height, src.width
        ref_transform = src.transform
        ref_crs = src.crs
        scale = max(1.0, max(ref_h, ref_w) / max_dim)
        new_h = int(ref_h / scale)
        new_w = int(ref_w / scale)
        # transform para grid downsampled
        new_transform = src.transform * src.transform.scale(scale, scale)

    print(f"  Grid: {new_h}x{new_w} px (downsample {scale:.1f}×)")

    accum = np.zeros((new_h, new_w), dtype=np.int16)
    n_anos_validos = 0
    for y, p in paths:
        with rasterio.open(p) as src:
            dst = np.zeros((new_h, new_w), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=ref_crs,
                dst_shape=(new_h, new_w),
                resampling=Resampling.max,
            )
        accum += (dst > 127).astype(np.int16)
        n_anos_validos += 1
        print(f"    {y}: somado")

    persistent = (accum >= min_years).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    im0 = axes[0].imshow(accum, cmap="viridis", vmin=0, vmax=n_anos_validos)
    axes[0].set_title(f"Soma de anos com fenda (0–{n_anos_validos})")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.04, label="anos com fenda")

    axes[1].imshow(persistent, cmap="Reds")
    axes[1].set_title(f"Persistente em ≥ {min_years} anos")
    axes[1].axis("off")

    plt.suptitle(f"Persistência cross-year das fendas ({len(paths)} anos analisados)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


# ============================================================================
# dH temporal (diferenca DEMs)
# ============================================================================

def dh_temporal(years, out_dir, max_dim=2000):
    """Calcula dH entre anos consecutivos (DEM_t - DEM_{t-1})."""
    years = sorted(years)
    pairs = list(zip(years[:-1], years[1:]))
    outputs = []

    for y0, y1 in pairs:
        d0 = find_dem(y0)
        d1 = find_dem(y1)
        if d0 is None or d1 is None:
            print(f"  {y0}->{y1}: DEM ausente")
            continue

        # Reprojeta d1 para grid de d0 (downsampled)
        with rasterio.open(d0) as src:
            scale = max(1.0, max(src.height, src.width) / max_dim)
            new_h = int(src.height / scale)
            new_w = int(src.width / scale)
            new_transform = src.transform * src.transform.scale(scale, scale)
            d0_arr = src.read(1, out_shape=(new_h, new_w),
                              resampling=Resampling.bilinear).astype(np.float32)
            if src.nodata is not None:
                d0_arr[d0_arr == src.nodata] = np.nan
            ref_crs = src.crs

        with rasterio.open(d1) as src:
            d1_arr = np.empty((new_h, new_w), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=d1_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=ref_crs,
                dst_shape=(new_h, new_w),
                resampling=Resampling.bilinear,
            )
            if src.nodata is not None:
                d1_arr[d1_arr == src.nodata] = np.nan

        dh = d1_arr - d0_arr
        valid = np.isfinite(dh)
        if not valid.any():
            continue

        # cap em percentil 1-99 para visualizacao
        vals = dh[valid]
        vmin, vmax = np.percentile(vals, [2, 98])
        cap = max(abs(vmin), abs(vmax))

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(dh, cmap="RdBu_r", vmin=-cap, vmax=cap)
        ax.set_title(f"dH = DEM({y1}) − DEM({y0})    "
                     f"mediana = {np.nanmedian(vals):+.2f} m")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.04, label="Δ elevação (m)")
        plt.tight_layout()
        out = out_dir / f"dh_{y1}_minus_{y0}.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        outputs.append(out)
        print(f"  {y0}->{y1}: OK  mediana={np.nanmedian(vals):+.2f} m  ({out.name})")
    return outputs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Camada 3 — analise glaciologica")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT)
    parser.add_argument("--out-dir", type=Path,
                        default=Config.RESULTS_DIR / "figures" / "glaciological")
    parser.add_argument("--skip-geometry", action="store_true",
                        help="Pula CSV de geometria por fenda")
    parser.add_argument("--skip-altitude", action="store_true")
    parser.add_argument("--skip-slope", action="store_true")
    parser.add_argument("--skip-density", action="store_true")
    parser.add_argument("--skip-persistence", action="store_true")
    parser.add_argument("--skip-dh", action="store_true")
    parser.add_argument("--persistence-min-years", type=int, default=3)
    parser.add_argument("--normalized", action="store_true",
                        help="Usa mosaicos padronizados (sufixo _8cm).")
    args = parser.parse_args()

    global USE_NORMALIZED
    USE_NORMALIZED = args.normalized

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CAMADA 3 — ANALISE GLACIOLOGICA")
    print("=" * 60)
    print(f"Anos: {args.years}")

    if not args.skip_geometry:
        print("\n[1/6] Geometria individual (CSV)")
        for year in args.years:
            out = Config.RESULTS_DIR / f"crevasse_geometries_{year}.csv"
            print(f"  {year}: ", end="", flush=True)
            r = per_crevasse_geometry(year, out)
            if r:
                print(f"{r[1]:,} fendas -> {out.name}")
            else:
                print("PULADO")

    if not args.skip_altitude:
        print("\n[2/6] Distribuicao altitudinal")
        plot_altitude_distribution(args.years, args.out_dir / "altitude_dist.png")
        print(f"  OK -> altitude_dist.png")

    if not args.skip_slope:
        print("\n[3/6] Distribuicao de slope")
        plot_slope_distribution(args.years, args.out_dir / "slope_dist.png")
        print(f"  OK -> slope_dist.png")

    if not args.skip_density:
        print("\n[4/6] Mapas de densidade")
        for year in args.years:
            out = args.out_dir / f"density_map_{year}.png"
            print(f"  {year}: ", end="", flush=True)
            r = density_map(year, out)
            print(f"OK -> {out.name}" if r else "PULADO")

    if not args.skip_persistence:
        print("\n[5/6] Persistencia cross-year")
        out = args.out_dir / f"persistence_{args.persistence_min_years}of{len(args.years)}.png"
        r = persistence_map(args.years, out, min_years=args.persistence_min_years)
        print(f"  OK -> {out.name}" if r else "  PULADO")

    if not args.skip_dh:
        print("\n[6/6] dH temporal entre DEMs")
        dh_temporal(args.years, args.out_dir)

    print("\nCAMADA 3 CONCLUIDA.")


if __name__ == "__main__":
    main()

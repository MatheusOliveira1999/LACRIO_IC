"""
08_qualitative_overlays.py - Camada 2 da analise: visualizacoes qualitativas

Gera:
  - results/figures/qualitative/overlay_{ano}.png      (overlay RGB+pred por ano)
  - results/figures/qualitative/comparative_panel.png  (4 anos lado a lado)
  - results/figures/qualitative/best_worst/{ano}_{best,worst}_{i}_{tile}.png
  - results/figures/qualitative/orientation_rose_{ano}.png

Uso:
  python 08_qualitative_overlays.py
  python 08_qualitative_overlays.py --years 2016 2017
  python 08_qualitative_overlays.py --skip-overlay --skip-bestworst   # so rosetas
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

from config import Config


YEARS_DEFAULT = [2016, 2017, 2018, 2019]
MASK_COLOR_RGB = (220, 40, 40)   # vermelho para fenda
ALPHA = 0.55                      # transparencia do overlay


# ============================================================================
# Helpers
# ============================================================================

USE_NORMALIZED = False  # setado pelo --normalized no main


def find_rgb_mosaic(year):
    """Encontra mosaico RGB do ano. Se USE_NORMALIZED, prefere versao _8cm."""
    if USE_NORMALIZED:
        norm_dir = Config.DATA_SOURCE_DIR / "normalized_8cm"
        if norm_dir.exists():
            for prefix in ("Schiaparelli_mosaic_", "schiaparelli_mosaic_"):
                hits = sorted(norm_dir.glob(f"{prefix}{year}_*cm.tif"))
                if hits:
                    return hits[0]
    base = Config.DATA_SOURCE_DIR
    for prefix in ("Schiaparelli_mosaic_", "schiaparelli_mosaic_"):
        p = base / f"{prefix}{year}.tif"
        if p.exists():
            return p
    return None


def find_pred_mosaic(year):
    """Encontra mosaico de predicao do ano. Se USE_NORMALIZED, prefere versao _8cm."""
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


def read_downsampled(tif_path, max_dim, resampling=Resampling.bilinear):
    """Le raster com downsampling para max_dim no eixo maior."""
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        if max(h, w) <= max_dim:
            scale = 1.0
            new_h, new_w = h, w
        else:
            scale = max(h, w) / max_dim
            new_h = int(h / scale)
            new_w = int(w / scale)
        arr = src.read(out_shape=(src.count, new_h, new_w), resampling=resampling)
        if arr.shape[0] == 1:
            return arr[0], scale
        return np.transpose(arr, (1, 2, 0)), scale


def to_uint8_rgb(arr):
    """Normaliza array para uint8 RGB."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        # auto-escala se for float ou outro int
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if finite.size == 0:
            return np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
        lo, hi = np.percentile(finite, [2, 98])
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def make_overlay(rgb_arr, pred_arr, color=MASK_COLOR_RGB, alpha=ALPHA):
    """Cria overlay colorido a partir de RGB + mascara binaria."""
    rgb = to_uint8_rgb(rgb_arr)
    if pred_arr.shape[:2] != rgb.shape[:2]:
        pred_arr = cv2.resize(pred_arr, (rgb.shape[1], rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    binary = (pred_arr > 127).astype(np.float32)
    overlay = rgb.astype(np.float32)
    color = np.asarray(color, dtype=np.float32)
    for c in range(3):
        overlay[..., c] = overlay[..., c] * (1 - alpha * binary) + color[c] * (alpha * binary)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ============================================================================
# Overlays
# ============================================================================

def overlay_full_mosaic(year, out_path, max_dim=4000):
    rgb_path = find_rgb_mosaic(year)
    pred_path = find_pred_mosaic(year)
    if rgb_path is None or pred_path is None:
        return None

    rgb, scale_rgb = read_downsampled(rgb_path, max_dim, Resampling.bilinear)
    pred, _ = read_downsampled(pred_path, max_dim, Resampling.nearest)
    overlay = make_overlay(rgb, pred)

    n_fendas = int(((pred > 127).sum()))
    cov_pct = 100.0 * n_fendas / pred.size if pred.size else 0.0

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.imshow(overlay)
    ax.set_title(f"Crevasses {year}  —  overlay  "
                 f"(downsample {scale_rgb:.1f}× · cobertura {cov_pct:.2f}%)",
                 fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def comparative_panel(years, out_path, max_dim=2500):
    n = len(years)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 7.5 * rows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, year in zip(axes_flat, years):
        rgb_path = find_rgb_mosaic(year)
        pred_path = find_pred_mosaic(year)
        if rgb_path is None or pred_path is None:
            ax.set_visible(False)
            continue
        rgb, _ = read_downsampled(rgb_path, max_dim, Resampling.bilinear)
        pred, _ = read_downsampled(pred_path, max_dim, Resampling.nearest)
        overlay = make_overlay(rgb, pred, alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(f"{year}", fontsize=14)
        ax.axis("off")

    for ax in axes_flat[len(years):]:
        ax.set_visible(False)

    plt.suptitle("Crevasses do Glaciar Schiaparelli — comparação multi-ano",
                 fontsize=15, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ============================================================================
# Best/worst tiles
# ============================================================================

def best_worst_tiles(year, n=3, out_dir=None):
    val_path = Config.RESULTS_DIR / f"unet_validation_crevasses_{year}.json"
    if not val_path.exists():
        return []
    with open(val_path) as f:
        data = json.load(f)
    per_tile = data.get("per_tile", [])
    valid = [t for t in per_tile if t.get("gt_area", 0) > 0]
    if not valid:
        return []
    valid.sort(key=lambda t: -t["f1"])
    selected = [("best", valid[:n]), ("worst", valid[-n:][::-1])]

    out_files = []
    for label, tiles in selected:
        for i, t in enumerate(tiles, start=1):
            tile_id = t["tile"]
            tp = Config.TILES_DIR / str(year) / f"{tile_id}.png"
            gt_p = Config.MASKS_DIR / str(year) / "annotations" / "crevasses" / f"{tile_id}_crevasses.png"
            pr_p = Config.MASKS_DIR / str(year) / "crevasses" / f"{tile_id}_crevasses.png"
            if not tp.exists() or not gt_p.exists():
                continue
            tile_img = cv2.cvtColor(cv2.imread(str(tp)), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(pr_p), cv2.IMREAD_GRAYSCALE) if pr_p.exists() else np.zeros_like(gt)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
            axes[0].imshow(tile_img); axes[0].set_title("RGB"); axes[0].axis("off")

            axes[1].imshow(tile_img)
            mask_gt = np.zeros((*gt.shape, 4))
            mask_gt[gt > 127] = [0.1, 0.8, 0.1, 0.55]   # verde
            axes[1].imshow(mask_gt)
            axes[1].set_title(f"GT  ({t['gt_area']} px)"); axes[1].axis("off")

            axes[2].imshow(tile_img)
            mask_pr = np.zeros((*pred.shape, 4))
            mask_pr[pred > 127] = [0.9, 0.2, 0.2, 0.55]  # vermelho
            axes[2].imshow(mask_pr)
            axes[2].set_title(f"Pred ({t['pred_area']} px) · F1={t['f1']:.3f}")
            axes[2].axis("off")

            plt.suptitle(
                f"{year} — {label.upper()} #{i} — {tile_id}    "
                f"F1={t['f1']:.3f} · P={t['precision']:.3f} · R={t['recall']:.3f}",
                fontsize=12,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out = out_dir / f"{year}_{label}_{i}_{tile_id}.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            out_files.append(out)
    return out_files


# ============================================================================
# Roseta de orientacao
# ============================================================================

def orientation_rose(year, out_path, n_bins=36, min_area_px=20):
    pred_path = find_pred_mosaic(year)
    if pred_path is None:
        return None
    with rasterio.open(pred_path) as src:
        mask = src.read(1)
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    azimuths, weights = [], []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        # Slice local do componente via bounding box (em vez de alocar mascara global)
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w_bb = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h_bb = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        comp_local = (labels[y:y+h_bb, x:x+w_bb] == label_id).astype(np.uint8)
        cnts, _ = cv2.findContours(comp_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        rect = cv2.minAreaRect(cnts[0])
        w, h = rect[1]
        angle = rect[2]
        if w < h:
            angle = angle + 90.0
        # azimute medido em sentido horario do Norte (linha sem direcao -> [0,180))
        azimuth = (90.0 - angle) % 180.0
        azimuths.append(azimuth)
        weights.append(area)

    if not azimuths:
        return None

    azimuths = np.asarray(azimuths)
    weights = np.asarray(weights, dtype=np.float64)

    bins = np.linspace(0, 180, n_bins + 1)
    hist, _ = np.histogram(azimuths, bins=bins, weights=weights)
    # espelhar para 0-360 (linha = duas direcoes opostas)
    hist_full = np.concatenate([hist, hist])
    angles_rad = np.deg2rad(np.linspace(0, 360, 2 * n_bins, endpoint=False))
    width_rad = 2 * np.pi / (2 * n_bins)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(angles_rad + width_rad / 2, hist_full, width=width_rad,
           color="#c0392b", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Orientação das fendas — {year}\n"
                 f"n={len(azimuths):,} fendas · ponderado por área", pad=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Camada 2 — visualizacoes qualitativas")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT)
    parser.add_argument("--out-dir", type=Path,
                        default=Config.RESULTS_DIR / "figures" / "qualitative")
    parser.add_argument("--max-dim", type=int, default=4000,
                        help="Dimensao maxima dos overlays (downsampling)")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--skip-panel", action="store_true")
    parser.add_argument("--skip-bestworst", action="store_true")
    parser.add_argument("--skip-rose", action="store_true")
    parser.add_argument("--normalized", action="store_true",
                        help="Usa mosaicos padronizados (sufixo _8cm).")
    args = parser.parse_args()

    global USE_NORMALIZED
    USE_NORMALIZED = args.normalized

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "best_worst").mkdir(exist_ok=True)

    print("=" * 60)
    print("CAMADA 2 — VISUALIZACOES QUALITATIVAS")
    print("=" * 60)
    print(f"Anos: {args.years}  |  Output: {args.out_dir}")

    if not args.skip_overlay:
        print("\n[1/4] Overlays full-mosaic")
        for year in args.years:
            out = args.out_dir / f"overlay_{year}.png"
            print(f"  {year}: ", end="", flush=True)
            r = overlay_full_mosaic(year, out, max_dim=args.max_dim)
            print(f"OK -> {out.name}" if r else "PULADO (arquivos ausentes)")

    if not args.skip_panel:
        print("\n[2/4] Painel comparativo")
        out = args.out_dir / "comparative_panel.png"
        r = comparative_panel(args.years, out, max_dim=max(2500, args.max_dim // 2))
        print(f"  OK -> {out.name}" if r else "  FALHOU")

    if not args.skip_bestworst:
        print("\n[3/4] Best/worst tiles (3 best + 3 worst por ano)")
        total = 0
        for year in args.years:
            files = best_worst_tiles(year, n=3, out_dir=args.out_dir / "best_worst")
            print(f"  {year}: {len(files)} figuras")
            total += len(files)
        print(f"  Total: {total} figuras")

    if not args.skip_rose:
        print("\n[4/4] Rosetas de orientacao")
        for year in args.years:
            out = args.out_dir / f"orientation_rose_{year}.png"
            print(f"  {year}: ", end="", flush=True)
            r = orientation_rose(year, out)
            print(f"OK -> {out.name}" if r else "PULADO")

    print("\nCAMADA 2 CONCLUIDA.")


if __name__ == "__main__":
    main()

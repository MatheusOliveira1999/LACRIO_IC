"""
04_inference_unet.py - Inferencia semantica com U-Net

Projeto: LACRIO IC - Extracao de Feicoes Supraglaciais
Modelo de producao do pipeline (SAM descontinuado em abr/2026).

Velocidade: ~0.05s/tile

Uso:
    python 04_inference_unet.py --feature lakes --year 2016
    python 04_inference_unet.py --feature lakes --year 2016 --annotated-only
    python 04_inference_unet.py --feature lakes --year 2016 --annotated-only --validate
    python 04_inference_unet.py --threshold 0.5
    python 04_inference_unet.py --tta   # Test-Time Augmentation
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config

# Import do modelo U-Net
import importlib
_unet_train = importlib.import_module("03_train_unet")
UNetResNet34 = _unet_train.UNetResNet34

from shadow_utils import (
    precompute_year_shadows, get_shadow_mask_for_tile, filter_by_texture,
    precompute_year_relief, get_relief_for_tile, filter_by_dem_relief,
    precompute_year_dem_features, get_dem_features_for_tile, normalize_dem_features,
)


def apply_feature_filter(mask, image, feature,
                         use_feature_filter=True,
                         lake_blue_ratio=1.2,
                         lake_dark_brightness=80.0,
                         lake_max_brightness=200.0,
                         crevasses_max_brightness=150.0,
                         crevasses_min_aspect=3.0,
                         channels_min_aspect=5.0):
    """Filtros por feicao para reduzir FPs. Critérios:
    - Lakes: razao B/R > 1.3 ou agua escura azulada
    - Crevasses: baixa luminosidade (<150) + aspect ratio > 3
    - Channels: aspect ratio > 5
    """
    if mask.max() == 0 or not use_feature_filter:
        return mask

    filtered = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    feature_cfg = Config.FEATURES[feature]
    min_area = feature_cfg["min_area"]
    max_area = feature_cfg["max_area"]

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        component_mask = (labels == label_id).astype(np.uint8)

        if feature == "lakes":
            region_pixels = image[component_mask > 0]
            if len(region_pixels) > 0:
                mean_r = region_pixels[:, 0].mean()
                mean_b = region_pixels[:, 2].mean()
                brightness = region_pixels.mean()

                is_blue = mean_r > 0 and mean_b / mean_r > lake_blue_ratio
                is_dark_water = brightness < lake_dark_brightness and mean_b > mean_r
                is_too_bright = brightness > lake_max_brightness

                if is_too_bright or not (is_blue or is_dark_water):
                    continue

        elif feature == "crevasses":
            region_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_brightness = region_gray[component_mask > 0].mean()
            if mean_brightness >= crevasses_max_brightness:
                continue

            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                rect = cv2.minAreaRect(contours[0])
                w_rect, h_rect = rect[1]
                if min(w_rect, h_rect) > 0:
                    aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
                    if aspect < crevasses_min_aspect:
                        continue

        elif feature == "channels":
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                rect = cv2.minAreaRect(contours[0])
                w_rect, h_rect = rect[1]
                if min(w_rect, h_rect) > 0:
                    aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
                    if aspect < channels_min_aspect:
                        continue

        filtered[component_mask > 0] = 255

    return filtered


def load_unet(feature: str):
    """Carrega U-Net treinada. Detecta in_channels e DEM features do checkpoint."""
    checkpoint_path = Config.MODELS_DIR / f"unet_{feature}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Modelo U-Net nao encontrado: {checkpoint_path}\n"
            f"Execute primeiro: python 03_train_unet.py --feature {feature}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    in_channels = checkpoint.get("in_channels", 3)
    dem_features = checkpoint.get("dem_features")
    dem_window = checkpoint.get("dem_window_meters", 3.0)

    model = UNetResNet34(pretrained=False, in_channels=in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(Config.DEVICE)
    model.eval()

    img_size = checkpoint.get("img_size", 512)

    print(f"  U-Net carregada: {checkpoint_path.name}")
    print(f"  Epoca {checkpoint['epoch']} | Val Dice: {checkpoint['val_dice']:.4f} | "
          f"Val F1: {checkpoint.get('val_f1', 'N/A')}")
    print(f"  Input: {img_size}x{img_size} | in_channels: {in_channels}")
    if dem_features:
        print(f"  DEM channels: {','.join(dem_features)} (window {dem_window}m)")

    return model, img_size, in_channels, dem_features, dem_window


def preprocess_image(image, img_size=512, dem_features_arr=None):
    """Preprocessa imagem para U-Net.

    Args:
        image: RGB uint8 (H, W, 3).
        img_size: Tamanho de redimensionamento.
        dem_features_arr: (H, W, n_dem) float32 ja normalizado, ou None.
                          Se fornecido, concatenado apos RGB.
    """
    image_resized = cv2.resize(image, (img_size, img_size),
                                interpolation=cv2.INTER_LINEAR)
    image_norm = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std

    if dem_features_arr is not None:
        if dem_features_arr.shape[:2] != (img_size, img_size):
            dem_features_arr = cv2.resize(
                dem_features_arr, (img_size, img_size),
                interpolation=cv2.INTER_LINEAR,
            )
            if dem_features_arr.ndim == 2:
                dem_features_arr = dem_features_arr[..., None]
        full = np.concatenate([image_norm, dem_features_arr.astype(np.float32)], axis=-1)
    else:
        full = image_norm

    tensor = torch.from_numpy(full).permute(2, 0, 1).float()
    return tensor.unsqueeze(0)


def predict_tile_unet(model, image, img_size=512, threshold=0.5,
                      dem_features_arr=None):
    """Predicao U-Net para um tile. Forward pass unico."""
    h, w = image.shape[:2]

    tensor = preprocess_image(image, img_size, dem_features_arr).to(Config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits[0, 0]).cpu().numpy()

    # Resize para tamanho original
    probs_full = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (probs_full > threshold).astype(np.uint8) * 255

    return mask


def predict_tile_unet_tta(model, image, img_size=512, threshold=0.5,
                          dem_features_arr=None):
    """Predicao U-Net com TTA (flips). ~4x mais lento, +1-3% IoU."""
    h, w = image.shape[:2]

    transforms = [
        ("original", lambda x: x, lambda x: x),
        ("hflip", lambda x: np.fliplr(x).copy(), lambda x: np.fliplr(x)),
        ("vflip", lambda x: np.flipud(x).copy(), lambda x: np.flipud(x)),
        ("rot180", lambda x: np.rot90(x, 2).copy(), lambda x: np.rot90(x, -2)),
    ]

    probs_sum = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for name, fwd, inv in transforms:
            aug_img = fwd(image)
            aug_dem = fwd(dem_features_arr) if dem_features_arr is not None else None
            tensor = preprocess_image(aug_img, img_size, aug_dem).to(Config.DEVICE)

            logits = model(tensor)
            probs = torch.sigmoid(logits[0, 0]).cpu().numpy()
            probs_full = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)

            probs_back = inv(probs_full)
            probs_sum += probs_back

    probs_avg = probs_sum / len(transforms)
    mask = (probs_avg > threshold).astype(np.uint8) * 255

    return mask


def compute_metrics(pred_mask, gt_mask):
    """Calcula metricas de segmentacao."""
    pred = (pred_mask > 127).astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "gt_area": int(gt.sum()), "pred_area": int(pred.sum())}


def run_inference(feature: str, year: int, threshold: float = 0.5,
                  annotated_only: bool = False, validate: bool = False,
                  use_tta: bool = False, use_feature_filter: bool = True,
                  use_shadow: bool = True,
                  use_dem_filter: bool = False,
                  dem_min_depth: float = -0.08,
                  dem_window_meters: float = 3.0,
                  dem_min_frac_below: float = 0.05,
                  output_subdir_suffix: str = ""):
    """Executa inferencia U-Net em todos os tiles de um ano."""

    print(f"\n--- Inferencia U-Net: {feature} | Ano: {year} ---")

    tiles_index_path = Config.TILES_DIR / str(year) / "tiles_index.json"
    if not tiles_index_path.exists():
        print(f"  [ERRO] Indice de tiles nao encontrado: {tiles_index_path}")
        return None

    with open(tiles_index_path) as f:
        tiles_index = json.load(f)

    tiles = tiles_index["tiles"]
    total_tiles = len(tiles)

    if annotated_only:
        gt_dir = Config.MASKS_DIR / str(year) / "annotations" / feature
        gt_files = sorted(gt_dir.glob(f"tile_*_{feature}.png"))
        annotated_filenames = {
            f"{gt_file.stem.removesuffix(f'_{feature}')}.png" for gt_file in gt_files
        }
        tiles = [t for t in tiles if t.get("filename") in annotated_filenames]
        print(f"  Modo rapido: {len(tiles)}/{total_tiles} tiles")
    else:
        print(f"  Total de tiles: {total_tiles}")

    # Sombra DEM
    shadow_mask_full = None
    dem_transform = None
    if use_shadow and feature == "lakes":
        shadow_mask_full, dem_transform = precompute_year_shadows(year)

    # Relief DEM (discrimina fenda real vs detrito escuro)
    relief_map = None
    relief_transform = None
    if use_dem_filter:
        print(f"  [DEM FILTER] min_depth={dem_min_depth}m | "
              f"window={dem_window_meters}m | min_frac_below={dem_min_frac_below}")
        relief_map, relief_transform = precompute_year_relief(year, dem_window_meters)
        if relief_map is None:
            print(f"  [AVISO] DEM filter solicitado mas relief nao carregou. Continuando sem.")

    # Carregar modelo (detecta in_channels e DEM features pelo checkpoint)
    model, img_size, in_channels, ckpt_dem_features, ckpt_dem_window = load_unet(feature)

    # Pre-computar DEM features se o checkpoint usa canais extras
    dem_features_stack = None
    dem_features_transform = None
    dem_features_stats = None
    if in_channels > 3 and ckpt_dem_features:
        dem_features_stack, dem_features_transform, _, dem_features_stats = \
            precompute_year_dem_features(
                year, window_meters=ckpt_dem_window,
                features=tuple(ckpt_dem_features),
            )
        if dem_features_stack is None:
            raise RuntimeError(
                f"Modelo treinado com DEM channels mas DEM {year} indisponivel."
            )

    def _dem_feats_for_tile(tile_info):
        if dem_features_stack is None:
            return None
        feats = get_dem_features_for_tile(
            tile_info, dem_features_stack, dem_features_transform
        )
        return normalize_dem_features(feats, dem_features_stats, ckpt_dem_features)

    # Diretorio de saida
    output_dir = Config.MASKS_DIR / str(year) / f"{feature}{output_subdir_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(tiles), "processed": 0, "with_features": 0,
             "skipped": 0, "shadow_removed": 0, "dem_removed": 0}

    # Validacao
    val_tp, val_fp, val_fn = 0, 0, 0
    per_tile_metrics = []

    predict_fn = predict_tile_unet_tta if use_tta else predict_tile_unet

    for tile_info in tqdm(tiles, desc=f"  {feature}/{year}"):
        tile_file = Config.TILES_DIR / str(year) / tile_info["filename"]

        if not tile_file.exists():
            stats["skipped"] += 1
            continue

        image = cv2.imread(str(tile_file))
        if image is None:
            stats["skipped"] += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # DEM features (se modelo usa)
        dem_feats = _dem_feats_for_tile(tile_info)

        # Predicao
        mask = predict_fn(model, image, img_size=img_size, threshold=threshold,
                          dem_features_arr=dem_feats)

        # Filtros pos-processamento
        if use_feature_filter:
            mask = apply_feature_filter(mask, image, feature)

        # Filtro topografico por DEM (fenda = depressao vertical, detrito = superficie)
        if relief_map is not None and mask.max() > 0:
            had_pixels = True
            tile_relief = get_relief_for_tile(
                tile_info, relief_map, relief_transform
            )
            mask = filter_by_dem_relief(
                mask, tile_relief,
                min_depth=dem_min_depth,
                min_frac_below=dem_min_frac_below,
            )
            if had_pixels and mask.max() == 0:
                stats["dem_removed"] += 1

        if feature == "lakes" and shadow_mask_full is not None:
            had_pixels = mask.max() > 0
            tile_shadow = get_shadow_mask_for_tile(
                tile_info, shadow_mask_full, dem_transform
            )
            mask[tile_shadow > 127] = 0
            if had_pixels and mask.max() == 0:
                stats["shadow_removed"] += 1

        if feature == "lakes" and mask.max() > 0:
            mask = filter_by_texture(mask, image)

        stats["processed"] += 1

        tile_id = tile_info["id"]
        output_path = output_dir / f"tile_{tile_id:06d}_{feature}.png"

        if mask.max() > 0:
            cv2.imwrite(str(output_path), mask)
            stats["with_features"] += 1
        elif output_path.exists():
            output_path.unlink()

        # Validacao inline
        if validate:
            gt_path = (Config.MASKS_DIR / str(year) / "annotations" / feature /
                       f"tile_{tile_id:06d}_{feature}.png")
            if gt_path.exists():
                gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                m = compute_metrics(mask, gt_mask)
                m["tile"] = f"tile_{tile_id:06d}"
                per_tile_metrics.append(m)
                val_tp += m["tp"]
                val_fp += m["fp"]
                val_fn += m["fn"]

    # Liberar GPU
    del model
    torch.cuda.empty_cache()

    # Salvar stats
    stats_path = output_dir / "inference_stats_unet.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Processados: {stats['processed']} | "
          f"Com feicoes: {stats['with_features']} | "
          f"Shadow removed: {stats['shadow_removed']} | "
          f"DEM removed: {stats['dem_removed']}")

    # Mostrar validacao
    if validate and per_tile_metrics:
        p = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        r = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        iou = val_tp / (val_tp + val_fp + val_fn) if (val_tp + val_fp + val_fn) > 0 else 0.0

        print(f"\n  VALIDACAO ({len(per_tile_metrics)} tiles com GT):")
        print(f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}")

        n_overseg = sum(1 for t in per_tile_metrics
                        if t["pred_area"] > 5 * t["gt_area"] and t["gt_area"] > 0)
        n_zero = sum(1 for t in per_tile_metrics
                     if t["tp"] == 0 and t["gt_area"] > 0)
        print(f"  Overseg (>5x): {n_overseg} | Zero overlap: {n_zero}")

        print(f"\n  {'Tile':<20} {'P':>7} {'R':>7} {'F1':>7} "
              f"{'GT':>7} {'Pred':>7}")
        print(f"  {'-'*55}")
        for t in per_tile_metrics:
            print(f"  {t['tile']:<20} {t['precision']:>7.3f} {t['recall']:>7.3f} "
                  f"{t['f1']:>7.3f} {t['gt_area']:>7} {t['pred_area']:>7}")

        # Salvar resultados
        results_path = Config.RESULTS_DIR / f"unet_validation_{feature}_{year}.json"
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({
                "micro": {"precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f1, 4), "iou": round(iou, 4)},
                "per_tile": per_tile_metrics,
            }, f, indent=2)
        print(f"\n  Resultados: {results_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia semantica com U-Net"
    )
    parser.add_argument("--feature", type=str, default="lakes",
                        choices=list(Config.FEATURES.keys()))
    parser.add_argument("--year", type=int, default=None, choices=Config.YEARS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--annotated-only", action="store_true")
    parser.add_argument("--validate", action="store_true",
                        help="Validar contra GT durante inferencia")
    parser.add_argument("--tta", action="store_true",
                        help="Test-Time Augmentation (4x mais lento)")
    parser.add_argument("--no-feature-filter", action="store_true")
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--dem-filter", action="store_true",
                        help="Filtra predicoes por relief topografico do DEM "
                             "(fendas sao depressoes verticais; detritos nao).")
    parser.add_argument("--dem-min-depth", type=float, default=-0.08,
                        help="Profundidade minima (m, negativa) para manter componente. "
                             "Default: -0.08m (calibrado em 2016, AUC 0.785 com window=3m). "
                             "Mais negativo = mais conservador.")
    parser.add_argument("--dem-window", type=float, default=3.0,
                        help="Janela em metros para media local de elevacao. "
                             "Default: 3.0m (sweet spot vs DEM 31 cm/px Schiaparelli 2016).")
    parser.add_argument("--dem-min-frac-below", type=float, default=0.05,
                        help="Fracao minima de pixels do componente abaixo do threshold. "
                             "Default: 0.05 (basta 5%% do componente ter pixels profundos).")
    parser.add_argument("--tiles-dir", type=Path, default=None,
                        help="Diretorio alternativo de tiles (default: Config.TILES_DIR). "
                             "Use junto com --output-mask-suffix para nao sobrescrever predicoes.")
    parser.add_argument("--output-mask-suffix", type=str, default="",
                        help="Sufixo na pasta de saida das mascaras (ex.: '_8cm' -> "
                             "masks/{ano}/crevasses_8cm/). Default: sem sufixo.")
    args = parser.parse_args()

    if args.tiles_dir is not None:
        Config.TILES_DIR = args.tiles_dir.resolve()
        print(f"📁 Tiles vindos de: {Config.TILES_DIR}")

    print("=" * 60)
    print("INFERENCIA U-NET")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Threshold: {args.threshold}")
    print(f"TTA: {'SIM' if args.tta else 'NAO'}")

    start = time.time()

    years = [args.year] if args.year else Config.YEARS

    for year in years:
        run_inference(
            args.feature, year, args.threshold,
            annotated_only=args.annotated_only,
            validate=args.validate,
            use_tta=args.tta,
            use_feature_filter=not args.no_feature_filter,
            use_shadow=not args.no_shadow,
            use_dem_filter=args.dem_filter,
            dem_min_depth=args.dem_min_depth,
            dem_window_meters=args.dem_window,
            dem_min_frac_below=args.dem_min_frac_below,
            output_subdir_suffix=args.output_mask_suffix,
        )

    elapsed = time.time() - start
    print(f"\nTempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()

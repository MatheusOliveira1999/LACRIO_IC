"""
04b_inference_unet.py - Inferencia semantica com U-Net

Projeto: LACRIO IC - Extracao de Feicoes Supraglaciais
Alternativa ao 04_inference.py (SAM): forward pass unico, sem prompt.

Velocidade: ~0.05s/tile (vs ~3s/tile com SAM)

Uso:
    python 04b_inference_unet.py --feature lakes --year 2016
    python 04b_inference_unet.py --feature lakes --year 2016 --annotated-only
    python 04b_inference_unet.py --feature lakes --year 2016 --annotated-only --validate
    python 04b_inference_unet.py --threshold 0.5
    python 04b_inference_unet.py --tta   # Test-Time Augmentation
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
_unet_train = importlib.import_module("03b_train_unet")
UNetResNet34 = _unet_train.UNetResNet34

# Import dos filtros de pos-processamento existentes
_inf = importlib.import_module("04_inference")
apply_feature_filter = _inf.apply_feature_filter

from shadow_utils import precompute_year_shadows, get_shadow_mask_for_tile, filter_by_texture


def load_unet(feature: str):
    """Carrega U-Net treinada."""
    checkpoint_path = Config.MODELS_DIR / f"unet_{feature}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Modelo U-Net nao encontrado: {checkpoint_path}\n"
            f"Execute primeiro: python 03b_train_unet.py --feature {feature}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = UNetResNet34(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(Config.DEVICE)
    model.eval()

    img_size = checkpoint.get("img_size", 512)

    print(f"  U-Net carregada: {checkpoint_path.name}")
    print(f"  Epoca {checkpoint['epoch']} | Val Dice: {checkpoint['val_dice']:.4f} | "
          f"Val F1: {checkpoint.get('val_f1', 'N/A')}")
    print(f"  Input: {img_size}x{img_size}")

    return model, img_size


def preprocess_image(image, img_size=512):
    """Preprocessa imagem para U-Net (ImageNet normalization)."""
    image_resized = cv2.resize(image, (img_size, img_size),
                                interpolation=cv2.INTER_LINEAR)
    image_norm = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std

    tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()
    return tensor.unsqueeze(0)


def predict_tile_unet(model, image, img_size=512, threshold=0.5):
    """Predicao U-Net para um tile. Forward pass unico."""
    h, w = image.shape[:2]

    tensor = preprocess_image(image, img_size).to(Config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits[0, 0]).cpu().numpy()

    # Resize para tamanho original
    probs_full = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (probs_full > threshold).astype(np.uint8) * 255

    return mask


def predict_tile_unet_tta(model, image, img_size=512, threshold=0.5):
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
            tensor = preprocess_image(aug_img, img_size).to(Config.DEVICE)

            logits = model(tensor)
            probs = torch.sigmoid(logits[0, 0]).cpu().numpy()
            probs_full = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)

            # Desfazer transformacao
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
                  use_shadow: bool = True):
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

    # Carregar modelo
    model, img_size = load_unet(feature)

    # Diretorio de saida
    output_dir = Config.MASKS_DIR / str(year) / feature
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(tiles), "processed": 0, "with_features": 0,
             "skipped": 0, "shadow_removed": 0}

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

        # Predicao
        mask = predict_fn(model, image, img_size=img_size, threshold=threshold)

        # Filtros pos-processamento
        if use_feature_filter:
            mask = apply_feature_filter(mask, image, feature)

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
          f"Shadow removed: {stats['shadow_removed']}")

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
    args = parser.parse_args()

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
        )

    elapsed = time.time() - start
    print(f"\nTempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()

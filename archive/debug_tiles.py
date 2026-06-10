"""Debug rapido: mostra o que o detector espectral ve vs GT em tiles com F1=0."""
import cv2
import numpy as np
import importlib
from config import Config

_inf = importlib.import_module("04_inference")
find_lake_candidates = _inf.find_lake_candidates
apply_feature_filter = _inf.apply_feature_filter

# Tiles com F1=0 no propose-refine
bad_tiles = ["tile_000506", "tile_000507", "tile_000523",
             "tile_000524", "tile_000570", "tile_000579"]

year = 2016
feature = "lakes"

for tile_id in bad_tiles:
    tile_path = Config.TILES_DIR / str(year) / f"{tile_id}.png"
    gt_path = Config.MASKS_DIR / str(year) / "annotations" / feature / f"{tile_id}_{feature}.png"

    image = cv2.imread(str(tile_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    gt_area = (gt > 127).sum()
    gt_ys, gt_xs = np.where(gt > 127)

    # O que o detector espectral encontra
    candidates, spectral_mask = find_lake_candidates(image, min_area=20)
    spectral_area = (spectral_mask > 0).sum()

    # Analisar cor na regiao do GT
    if len(gt_xs) > 0:
        gt_pixels = image[gt > 127]
        mean_r = gt_pixels[:, 0].mean()
        mean_g = gt_pixels[:, 1].mean()
        mean_b = gt_pixels[:, 2].mean()
        brightness = (mean_r + mean_g + mean_b) / 3
        br_ratio = mean_b / mean_r if mean_r > 0 else 0
        ndwi = (mean_g - mean_r) / (mean_g + mean_r) if (mean_g + mean_r) > 0 else 0

        # O espectral cobre o GT?
        gt_binary = (gt > 127).astype(np.uint8)
        spectral_binary = (spectral_mask > 0).astype(np.uint8)
        overlap = (gt_binary & spectral_binary).sum()
        coverage = overlap / gt_area if gt_area > 0 else 0

        # Candidatos perto do GT?
        gt_cx, gt_cy = gt_xs.mean(), gt_ys.mean()
        nearest_dist = float('inf')
        for (bbox, area) in candidates:
            bx = (bbox[0] + bbox[2]) / 2
            by = (bbox[1] + bbox[3]) / 2
            d = np.sqrt((bx - gt_cx)**2 + (by - gt_cy)**2)
            if d < nearest_dist:
                nearest_dist = d

        print(f"\n{tile_id}: GT={gt_area}px  Spectral={spectral_area}px  "
              f"Candidatos={len(candidates)}")
        print(f"  Cor GT: R={mean_r:.0f} G={mean_g:.0f} B={mean_b:.0f} "
              f"Bri={brightness:.0f} B/R={br_ratio:.2f} NDWI={ndwi:.3f}")
        print(f"  Cobertura espectral do GT: {coverage:.1%}")
        print(f"  Candidato mais proximo do GT: {nearest_dist:.0f}px")

        # Verificar se o filtro de feicao removeria
        # Simular mascara perfeita sobre GT
        perfect_mask = gt.copy()
        filtered = apply_feature_filter(perfect_mask, image, feature)
        filtered_area = (filtered > 127).sum()
        print(f"  Filtro feicao no GT perfeito: {gt_area} -> {filtered_area}px "
              f"({'REMOVIDO' if filtered_area == 0 else 'OK'})")
    else:
        print(f"\n{tile_id}: GT vazio!")

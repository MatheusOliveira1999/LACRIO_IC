"""
test_strategies.py - Teste rapido de estrategias de inferencia

Projeto: LACRIO IC - Extracao de Feicoes Supraglaciais
Objetivo: Rodar inferencia nos tiles anotados e comparar metricas
          entre diferentes estrategias (grid cego, propose-refine, etc.)

Uso:
    python test_strategies.py                            # Todas as estrategias
    python test_strategies.py --feature lakes --year 2016
    python test_strategies.py --strategy propose-refine  # Estrategia especifica
    python test_strategies.py --max-tiles 10             # Teste rapido (10 tiles)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import Config

# Import das funcoes de inferencia (04_inference.py)
import importlib
_inf = importlib.import_module("04_inference")
load_finetuned_sam = _inf.load_finetuned_sam
predict_tile = _inf.predict_tile
predict_tile_propose_refine = _inf.predict_tile_propose_refine
predict_with_tta = _inf.predict_with_tta
apply_feature_filter = _inf.apply_feature_filter


def compute_metrics(pred_mask, gt_mask):
    """Calcula metricas de segmentacao pixel-a-pixel."""
    pred = (pred_mask > 127).astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    gt_area = int(gt.sum())
    pred_area = int(pred.sum())
    ratio = pred_area / gt_area if gt_area > 0 else float('inf')

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": f1, "iou": iou,
        "gt_area": gt_area, "pred_area": pred_area,
        "area_ratio": ratio,
    }


def get_annotated_tiles(feature, year):
    """Retorna lista de (tile_path, gt_path) para tiles anotados."""
    gt_dir = Config.MASKS_DIR / str(year) / "annotations" / feature
    if not gt_dir.exists():
        return []

    pairs = []
    for gt_file in sorted(gt_dir.glob(f"tile_*_{feature}.png")):
        tile_id = gt_file.stem.replace(f"_{feature}", "")
        tile_file = Config.TILES_DIR / str(year) / f"{tile_id}.png"
        if tile_file.exists():
            pairs.append((tile_file, gt_file))
    return pairs


def run_strategy(sam, image, strategy, feature="lakes", threshold=0.3,
                 pred_iou_threshold=0.5):
    """Roda uma estrategia de inferencia e retorna mascara."""

    if strategy == "grid-vote-2":
        return predict_tile(sam, image, threshold=threshold,
                            pred_iou_threshold=pred_iou_threshold,
                            combine_mode="vote", vote_threshold=2)

    elif strategy == "grid-vote-3":
        return predict_tile(sam, image, threshold=threshold,
                            pred_iou_threshold=pred_iou_threshold,
                            combine_mode="vote", vote_threshold=3)

    elif strategy == "grid-max":
        return predict_tile(sam, image, threshold=threshold,
                            pred_iou_threshold=pred_iou_threshold,
                            combine_mode="max")

    elif strategy == "grid-max-0.5":
        return predict_tile(sam, image, threshold=0.5,
                            pred_iou_threshold=pred_iou_threshold,
                            combine_mode="max")

    elif strategy == "propose-refine":
        return predict_tile_propose_refine(
            sam, image, threshold=threshold,
            pred_iou_threshold=pred_iou_threshold,
            max_area_ratio=5.0,
        )

    elif strategy == "propose-refine-strict":
        return predict_tile_propose_refine(
            sam, image, threshold=threshold,
            pred_iou_threshold=pred_iou_threshold,
            max_area_ratio=3.0,
        )

    elif strategy == "hybrid":
        # Tenta propose-refine primeiro; se nao encontra nada, fallback grid-max
        mask_pr = predict_tile_propose_refine(
            sam, image, threshold=threshold,
            pred_iou_threshold=pred_iou_threshold,
            max_area_ratio=5.0,
        )
        if mask_pr.max() > 0:
            return mask_pr
        return predict_tile(sam, image, threshold=0.5,
                            pred_iou_threshold=0.7,
                            combine_mode="max")

    else:
        raise ValueError(f"Estrategia desconhecida: {strategy}")


ALL_STRATEGIES = [
    "grid-max",
    "grid-max-0.5",
    "propose-refine",
    "propose-refine-strict",
    "hybrid",
]


def main():
    parser = argparse.ArgumentParser(
        description="Teste rapido de estrategias de inferencia com validacao"
    )
    parser.add_argument("--feature", type=str, default="lakes",
                        choices=list(Config.FEATURES.keys()))
    parser.add_argument("--year", type=int, default=2016, choices=Config.YEARS)
    parser.add_argument("--strategy", type=str, default=None,
                        choices=ALL_STRATEGIES,
                        help="Estrategia especifica (default: todas)")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Limitar numero de tiles (para teste rapido)")
    parser.add_argument("--no-feature-filter", action="store_true",
                        help="Desativar filtro de feicao pos-SAM")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else ALL_STRATEGIES

    # Coletar tiles anotados
    pairs = get_annotated_tiles(args.feature, args.year)
    if not pairs:
        print(f"Nenhum tile anotado para {args.feature}/{args.year}")
        return

    if args.max_tiles:
        pairs = pairs[:args.max_tiles]

    print("=" * 70)
    print("TESTE DE ESTRATEGIAS DE INFERENCIA")
    print("=" * 70)
    print(f"Feature: {args.feature} | Ano: {args.year}")
    print(f"Tiles anotados: {len(pairs)}")
    print(f"Estrategias: {strategies}")
    print(f"Threshold: {args.threshold} | Pred IoU: {args.pred_iou_threshold}")
    print(f"Filtro de feicao: {'NAO' if args.no_feature_filter else 'SIM'}")

    # Carregar modelo
    print(f"\nCarregando modelo...")
    sam = load_finetuned_sam(args.feature)

    # Resultados por estrategia
    results = {s: {"per_tile": [], "total_tp": 0, "total_fp": 0, "total_fn": 0}
               for s in strategies}

    start = time.time()

    for tile_path, gt_path in tqdm(pairs, desc="Processando tiles"):
        image = cv2.imread(str(tile_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        for strategy in strategies:
            mask = run_strategy(sam, image, strategy, args.feature,
                                args.threshold, args.pred_iou_threshold)

            # Aplicar filtro de feicao (se habilitado)
            if not args.no_feature_filter:
                mask = apply_feature_filter(mask, image, args.feature)

            m = compute_metrics(mask, gt_mask)
            m["tile"] = tile_path.stem
            results[strategy]["per_tile"].append(m)
            results[strategy]["total_tp"] += m["tp"]
            results[strategy]["total_fp"] += m["fp"]
            results[strategy]["total_fn"] += m["fn"]

        # Limpar VRAM entre tiles
        torch.cuda.empty_cache()

    elapsed = time.time() - start

    # Imprimir resultados comparativos
    print(f"\n{'=' * 70}")
    print(f"RESULTADOS ({args.feature}/{args.year}, {len(pairs)} tiles, {elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"{'Estrategia':<25} {'P':>7} {'R':>7} {'F1':>7} {'IoU':>7} "
          f"{'AreaRatio':>10} {'Overseg':>8} {'Zero':>6}")
    print("-" * 80)

    best_f1 = 0
    best_strategy = ""

    for strategy in strategies:
        r = results[strategy]
        tp, fp, fn = r["total_tp"], r["total_fp"], r["total_fn"]

        # Micro-average
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        # Diagnosticos
        tiles = r["per_tile"]
        finite_ratios = [t["area_ratio"] for t in tiles
                         if t["area_ratio"] < float('inf')]
        mean_ratio = np.mean(finite_ratios) if finite_ratios else 0.0
        n_overseg = sum(1 for t in tiles if t["area_ratio"] > 5.0)
        n_zero = sum(1 for t in tiles if t["tp"] == 0 and t["gt_area"] > 0)

        print(f"{strategy:<25} {p:>7.3f} {rec:>7.3f} {f1:>7.3f} {iou:>7.3f} "
              f"{mean_ratio:>10.1f}x {n_overseg:>8} {n_zero:>6}")

        if f1 > best_f1:
            best_f1 = f1
            best_strategy = strategy

    print(f"\nMelhor estrategia: {best_strategy} (F1={best_f1:.4f})")

    # Detalhes por tile da melhor estrategia
    print(f"\n--- Detalhes por tile ({best_strategy}) ---")
    print(f"{'Tile':<20} {'P':>7} {'R':>7} {'F1':>7} {'GT':>7} {'Pred':>7} {'Ratio':>8}")
    print("-" * 65)
    for t in results[best_strategy]["per_tile"]:
        ratio_str = f"{t['area_ratio']:.1f}x" if t['area_ratio'] < float('inf') else "inf"
        print(f"{t['tile']:<20} {t['precision']:>7.3f} {t['recall']:>7.3f} "
              f"{t['f1']:>7.3f} {t['gt_area']:>7} {t['pred_area']:>7} {ratio_str:>8}")

    # Salvar resultados detalhados
    output_dir = Config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"strategy_comparison_{args.feature}_{args.year}.json"

    summary = {}
    for strategy in strategies:
        r = results[strategy]
        tp, fp, fn = r["total_tp"], r["total_fp"], r["total_fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        summary[strategy] = {
            "precision": round(p, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "iou": round(iou, 4),
            "per_tile": r["per_tile"],
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResultados salvos: {output_path}")

    # Limpar
    del sam
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

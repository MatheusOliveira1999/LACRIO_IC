"""
active_learning.py - Selecao de tiles para anotacao

Projeto: LACRIO IC - Extracao de Feicoes Supraglaciais

Estrategias disponíveis (--strategy):
  uncertainty  Tiles onde o modelo tem maior incerteza (default para lakes).
               Requer modelo com sinal no ano alvo.
  random       Amostragem aleatoria dentro do glaciar (default para crevasses
               em anos sem dados de treino). Filtra vegetacao e nodata.

Uso:
    python active_learning.py --feature lakes --year 2017
    python active_learning.py --feature lakes --top 30
    python active_learning.py --feature crevasses --year 2018 --strategy random --skip-vegetation
    python active_learning.py --feature crevasses --detect-threshold 0.3
"""

import argparse
import json
import csv
import random as _random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import Config

import importlib
_unet = importlib.import_module("03b_train_unet")
_inf = importlib.import_module("04b_inference_unet")
UNetResNet34 = _unet.UNetResNet34
load_unet = _inf.load_unet
preprocess_image = _inf.preprocess_image


def tile_uncertainty(prob_map, detect_threshold=0.5):
    """Calcula incerteza de um tile a partir do mapa de probabilidades.

    So considera tiles onde o modelo realmente detectou algo (prob > 0.5).
    Tiles sem deteccao sao ignorados — pouco valor anotar negativos puros.

    Criterio de prioridade:
      1. Tiles com deteccao (pred_area > 0) e baixa confianca (max_prob < 0.9)
      2. Score = pred_area * (1 - max_prob): area detectada ponderada por incerteza

    Args:
        prob_map: Array (H, W) float32 com probabilidades [0, 1].
        detect_threshold: Prob minima para pixel contar como deteccao (0.5).

    Returns:
        dict com metricas, ou None se tile sem deteccao.
    """
    pred_area = int((prob_map > detect_threshold).sum())

    # Ignorar tiles sem nenhuma deteccao
    if pred_area == 0:
        return None

    max_prob = float(prob_map.max())

    # Entropia apenas nos pixels detectados (regiao de interesse)
    detected_probs = prob_map[prob_map > detect_threshold]
    eps = 1e-7
    p = np.clip(detected_probs, eps, 1 - eps)
    entropy_detected = float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).mean())

    # Score de incerteza: area detectada * (1 - confianca maxima)
    # Maior score = mais pixels detectados com menor confianca = mais util anotar
    uncertainty_score = pred_area * (1.0 - max_prob)

    return {
        "uncertainty_score": uncertainty_score,
        "pred_area": pred_area,
        "max_prob": max_prob,
        "entropy_detected": entropy_detected,
    }


def is_vegetation_tile(image_rgb, green_blue_margin=15):
    """Retorna True se o tile parece ser vegetacao (nao glaciar).

    Heuristica: vegetacao tem canal verde dominante sobre azul (G >> B).
    Gelo, neve e rocha glaciar sempre tem B >= G.

    Args:
        image_rgb: Array (H, W, 3) uint8 em RGB.
        green_blue_margin: G - B minimo para classificar como vegetacao.
    """
    mean_g = image_rgb[:, :, 1].mean()
    mean_b = image_rgb[:, :, 2].mean()
    return (mean_g - mean_b) > green_blue_margin


def rank_tiles_by_uncertainty(feature, year, top_n=None, detect_threshold=0.5,
                              skip_vegetation=False):
    """Roda o modelo em todos os tiles de um ano e rankeia por incerteza.

    Args:
        feature: Nome da feicao.
        year: Ano para processar.
        top_n: Retornar apenas os top N tiles mais incertos (None = todos).
        detect_threshold: Prob minima para considerar tile com deteccao.
        skip_vegetation: Se True, ignora tiles com dominancia verde (G >> B).

    Returns:
        Lista de dicts com tile_id, metricas de incerteza, ordenados.
    """
    print(f"\n--- Active Learning: {feature} | Ano: {year} ---")

    tiles_index_path = Config.TILES_DIR / str(year) / "tiles_index.json"
    if not tiles_index_path.exists():
        print(f"  [ERRO] tiles_index.json nao encontrado")
        return []

    with open(tiles_index_path) as f:
        tiles = json.load(f)["tiles"]

    # Excluir tiles ja anotados
    gt_dir = Config.MASKS_DIR / str(year) / "annotations" / feature
    annotated = set()
    if gt_dir.exists():
        for f_path in gt_dir.glob(f"tile_*_{feature}.png"):
            annotated.add(f_path.stem.replace(f"_{feature}", ""))
    print(f"  Total tiles: {len(tiles)} | Ja anotados: {len(annotated)} (excluidos)")

    model, img_size = load_unet(feature)

    ranked = []
    skipped = 0

    for tile_info in tqdm(tiles, desc=f"  Computando incerteza"):
        tile_id = f"tile_{tile_info['id']:06d}"

        # Pular anotados
        if tile_id in annotated:
            continue

        tile_file = Config.TILES_DIR / str(year) / tile_info["filename"]
        if not tile_file.exists():
            skipped += 1
            continue

        image = cv2.imread(str(tile_file))
        if image is None:
            skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if skip_vegetation and is_vegetation_tile(image):
            skipped += 1
            continue

        tensor = preprocess_image(image, img_size).to(Config.DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits[0, 0]).cpu().numpy()

        metrics = tile_uncertainty(prob, detect_threshold=detect_threshold)
        if metrics is None:
            continue  # Modelo nao detectou nada, pouco valor anotar

        ranked.append({
            "tile_id": tile_id,
            "year": year,
            "tile_path": str(tile_file),
            **metrics,
        })

    del model
    torch.cuda.empty_cache()

    # Ordenar por score de incerteza (maior = mais util anotar)
    ranked.sort(key=lambda x: x["uncertainty_score"], reverse=True)

    if top_n:
        ranked = ranked[:top_n]

    print(f"  Tiles com deteccao: {len(ranked) + skipped} | "
          f"Selecionados: {len(ranked)} | Skipped: {skipped}")

    return ranked


def save_results(all_ranked, feature, output_dir):
    """Salva resultados em JSON e CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON completo
    json_path = output_dir / f"active_learning_{feature}.json"
    with open(json_path, "w") as f:
        json.dump(all_ranked, f, indent=2)

    # CSV para abrir no Excel/LibreOffice
    csv_path = output_dir / f"active_learning_{feature}.csv"
    if all_ranked:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_ranked[0].keys())
            writer.writeheader()
            writer.writerows(all_ranked)

    return json_path, csv_path


def rank_tiles_random(feature, year, top_n=None, skip_vegetation=False,
                      min_valid_ratio=0.5, seed=42):
    """Seleciona tiles aleatoriamente dentro do glaciar, sem rodar o modelo.

    Util quando o modelo nao tem sinal no ano alvo (ex: crevasses treinado em
    2016 aplicado em 2018 sem anotacoes). Filtra vegetacao e tiles vazios.

    Args:
        feature: Nome da feicao (usado so para excluir ja anotados).
        year: Ano para processar.
        top_n: Quantos tiles retornar (None = todos validos).
        skip_vegetation: Se True, exclui tiles com G >> B.
        min_valid_ratio: Fracao minima de pixels nao-pretos para aceitar tile.
        seed: Semente para reproducibilidade.

    Returns:
        Lista de dicts com tile_id e metricas basicas, em ordem aleatoria.
    """
    print(f"\n--- Random Sampling: {feature} | Ano: {year} ---")

    tiles_index_path = Config.TILES_DIR / str(year) / "tiles_index.json"
    if not tiles_index_path.exists():
        print(f"  [ERRO] tiles_index.json nao encontrado")
        return []

    with open(tiles_index_path) as f:
        tiles = json.load(f)["tiles"]

    gt_dir = Config.MASKS_DIR / str(year) / "annotations" / feature
    annotated = set()
    if gt_dir.exists():
        for f_path in gt_dir.glob(f"tile_*_{feature}.png"):
            annotated.add(f_path.stem.replace(f"_{feature}", ""))
    print(f"  Total tiles: {len(tiles)} | Ja anotados: {len(annotated)} (excluidos)")

    candidates = []
    skipped_veg = 0
    skipped_nodata = 0

    for tile_info in tqdm(tiles, desc="  Filtrando tiles"):
        tile_id = f"tile_{tile_info['id']:06d}"
        if tile_id in annotated:
            continue

        tile_file = Config.TILES_DIR / str(year) / tile_info["filename"]
        if not tile_file.exists():
            continue

        image = cv2.imread(str(tile_file))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Filtrar tiles com muitos pixels pretos (nodata/borda)
        valid_ratio = (image.max(axis=2) > 10).mean()
        if valid_ratio < min_valid_ratio:
            skipped_nodata += 1
            continue

        # Filtrar vegetacao
        if skip_vegetation and is_vegetation_tile(image):
            skipped_veg += 1
            continue

        mean_r = float(image[:, :, 0].mean())
        mean_g = float(image[:, :, 1].mean())
        mean_b = float(image[:, :, 2].mean())

        candidates.append({
            "tile_id": tile_id,
            "year": year,
            "tile_path": str(tile_file),
            "uncertainty_score": 0.0,
            "pred_area": 0,
            "max_prob": 0.0,
            "entropy_detected": 0.0,
            "mean_r": round(mean_r, 1),
            "mean_g": round(mean_g, 1),
            "mean_b": round(mean_b, 1),
            "valid_ratio": round(valid_ratio, 3),
        })

    print(f"  Validos: {len(candidates)} | "
          f"Nodata excluidos: {skipped_nodata} | "
          f"Vegetacao excluida: {skipped_veg}")

    _random.seed(seed)
    _random.shuffle(candidates)

    if top_n:
        candidates = candidates[:top_n]

    return candidates


def print_summary(ranked, year, top_n=20):
    """Imprime tabela com os tiles mais incertos."""
    print(f"\n  TOP {min(top_n, len(ranked))} tiles para anotar (ano {year}):")
    print(f"  {'Tile':<20} {'Score':>10} {'AreaPred':>10} "
          f"{'MaxProb':>9} {'Entropia':>10}")
    print(f"  {'-'*65}")
    for t in ranked[:top_n]:
        print(f"  {t['tile_id']:<20} {t['uncertainty_score']:>10.1f} "
              f"{t['pred_area']:>10} {t['max_prob']:>9.3f} "
              f"{t['entropy_detected']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Selecao de tiles para anotacao por incerteza"
    )
    parser.add_argument("--feature", type=str, default="lakes",
                        choices=list(Config.FEATURES.keys()))
    parser.add_argument("--year", type=int, default=None, choices=Config.YEARS,
                        help="Ano especifico (default: anos sem GT)")
    parser.add_argument("--top", type=int, default=30,
                        help="Quantos tiles retornar por ano (default: 30)")
    parser.add_argument("--detect-threshold", type=float, default=0.5,
                        help="Prob minima para tile contar como deteccao (default: 0.5; "
                             "use 0.3 para features finas como crevasses)")
    parser.add_argument("--skip-vegetation", action="store_true",
                        help="Ignora tiles com dominancia verde (G >> B) — filtra entorno do glaciar")
    parser.add_argument("--strategy", type=str, default="uncertainty",
                        choices=["uncertainty", "random"],
                        help="uncertainty: rankeia por incerteza do modelo (default). "
                             "random: amostragem aleatoria — use quando o modelo nao tem "
                             "sinal no ano alvo (ex: crevasses em ano sem anotacoes)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente para reproducibilidade do random sampling (default: 42)")
    args = parser.parse_args()

    # Anos sem ground truth (candidatos para anotacao)
    if args.year:
        years_to_process = [args.year]
    else:
        years_to_process = []
        for year in Config.YEARS:
            gt_dir = Config.MASKS_DIR / str(year) / "annotations" / args.feature
            has_gt = gt_dir.exists() and any(gt_dir.glob(f"tile_*_{args.feature}.png"))
            if not has_gt:
                years_to_process.append(year)
        print(f"Anos sem GT: {years_to_process}")

    print("=" * 60)
    print("ACTIVE LEARNING - SELECAO DE TILES PARA ANOTACAO")
    print("=" * 60)
    print(f"Feature: {args.feature} | Estrategia: {args.strategy} | "
          f"Top {args.top} por ano | "
          f"Skip vegetacao: {'SIM' if args.skip_vegetation else 'NAO'}")

    all_ranked = []

    for year in years_to_process:
        if args.strategy == "random":
            ranked = rank_tiles_random(args.feature, year, top_n=args.top,
                                       skip_vegetation=args.skip_vegetation,
                                       seed=args.seed)
        else:
            ranked = rank_tiles_by_uncertainty(args.feature, year, top_n=args.top,
                                               detect_threshold=args.detect_threshold,
                                               skip_vegetation=args.skip_vegetation)
        if ranked:
            print_summary(ranked, year, top_n=args.top)
            all_ranked.extend(ranked)

    # Salvar resultados
    if all_ranked:
        output_dir = Config.RESULTS_DIR
        json_path, csv_path = save_results(all_ranked, args.feature, output_dir)
        print(f"\n{'='*60}")
        print(f"RESULTADOS SALVOS")
        print(f"{'='*60}")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        print(f"\n  Total de tiles sugeridos: {len(all_ranked)}")
        print(f"\n  Proximo passo:")
        print(f"  1. Revise o CSV para entender a distribuicao dos tiles sugeridos")
        print(f"  2. Abra o anotador com o CSV gerado:")
        print(f"       python annotate.py --feature {args.feature} --year ANO \\")
        print(f"           --csv {csv_path}")
        print(f"     Desenhe a mascara (ou pressione R para GT=0 se nao houver feicao)")
        print(f"  3. Repita ate ter ~15-20 novos tiles anotados")
        print(f"  4. Re-treine: python 03b_train_unet.py --feature {args.feature}")
    else:
        print("\n  Nenhum tile encontrado para sugerir.")


if __name__ == "__main__":
    main()

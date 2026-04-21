"""
06_validate.py - Validação das predições com métricas de segmentação

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Fase 6: Calcular F1-Score, IoU, Precision e Recall

Uso:
    python 06_validate.py                        # Todas as feições
    python 06_validate.py --feature lakes        # Feição específica
    python 06_validate.py --year 2016            # Ano específico
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config


def compute_metrics(pred_mask, gt_mask):
    """Calcula métricas de segmentação entre predição e ground truth.

    Args:
        pred_mask: Máscara predita (H, W) uint8.
        gt_mask: Máscara ground truth (H, W) uint8.

    Returns:
        dict com precision, recall, f1, iou, dice.
    """
    pred = (pred_mask > 127).astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0.0

    dice = 2 * intersection / (2 * intersection + fp + fn) if (2 * intersection + fp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
    }


def validate_feature(feature: str, year: int):
    """Valida predições contra ground truth para uma feição e ano.

    Compara as máscaras em masks/{year}/{feature}/ (preditas)
    contra masks/{year}/annotations/{feature}/ (ground truth).

    Args:
        feature: Nome da feição.
        year: Ano para validar.

    Returns:
        Dicionário com métricas agregadas e por tile.
    """
    print(f"\n--- Validação: {feature} | Ano: {year} ---")

    # Diretórios
    gt_dir = Config.MASKS_DIR / str(year) / "annotations" / feature
    pred_dir = Config.MASKS_DIR / str(year) / feature

    if not gt_dir.exists():
        print(f"  [AVISO] Sem ground truth em: {gt_dir}")
        return None

    if not pred_dir.exists():
        print(f"  [AVISO] Sem predições em: {pred_dir}")
        return None

    # Encontrar tiles com ground truth
    gt_files = sorted(gt_dir.glob(f"tile_*_{feature}.png"))
    if not gt_files:
        print(f"  [AVISO] Nenhuma anotação encontrada para {feature}/{year}")
        return None

    print(f"  Tiles com ground truth: {len(gt_files)}")

    per_tile_metrics = []
    all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0

    for gt_file in tqdm(gt_files, desc=f"  Validando {feature}/{year}"):
        # Carregar ground truth
        gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        # Encontrar predição correspondente
        pred_file = pred_dir / gt_file.name
        if pred_file.exists():
            pred_mask = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
            if pred_mask is None:
                pred_mask = np.zeros_like(gt_mask)
        else:
            # Sem predição = tudo zero (falso negativo total)
            pred_mask = np.zeros_like(gt_mask)

        # Calcular métricas
        m = compute_metrics(pred_mask, gt_mask)
        m["tile"] = gt_file.stem

        per_tile_metrics.append(m)
        all_tp += m["tp"]
        all_fp += m["fp"]
        all_fn += m["fn"]
        all_tn += m["tn"]

    if not per_tile_metrics:
        print("  Nenhuma comparação possível.")
        return None

    # Métricas agregadas (micro-average)
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = all_tp / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0.0

    # Métricas macro-average
    macro_f1 = np.mean([m["f1"] for m in per_tile_metrics])
    macro_iou = np.mean([m["iou"] for m in per_tile_metrics])

    result = {
        "feature": feature,
        "year": year,
        "n_tiles": len(per_tile_metrics),
        "micro": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "iou": round(iou, 4),
        },
        "macro": {
            "f1_mean": round(float(macro_f1), 4),
            "f1_std": round(float(np.std([m["f1"] for m in per_tile_metrics])), 4),
            "iou_mean": round(float(macro_iou), 4),
            "iou_std": round(float(np.std([m["iou"] for m in per_tile_metrics])), 4),
        },
        "per_tile": per_tile_metrics,
    }

    print(f"\n  RESULTADOS ({feature} / {year}):")
    print(f"  {'Métrica':<15} {'Micro':>10} {'Macro':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Precision':<15} {precision:>10.4f} {'---':>10}")
    print(f"  {'Recall':<15} {recall:>10.4f} {'---':>10}")
    print(f"  {'F1-Score':<15} {f1:>10.4f} {macro_f1:>10.4f}")
    print(f"  {'IoU':<15} {iou:>10.4f} {macro_iou:>10.4f}")

    return result


def plot_results(all_results):
    """Gera gráficos de barras com as métricas por feição.

    Args:
        all_results: Lista de dicionários com resultados da validação.
    """
    if not all_results:
        return

    output_dir = Config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    features = [r["feature"] for r in all_results]
    f1_scores = [r["micro"]["f1"] for r in all_results]
    iou_scores = [r["micro"]["iou"] for r in all_results]
    precision = [r["micro"]["precision"] for r in all_results]
    recall = [r["micro"]["recall"] for r in all_results]

    x = np.arange(len(features))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, precision, width, label="Precision", color="#2196F3")
    ax.bar(x - 0.5*width, recall, width, label="Recall", color="#FF9800")
    ax.bar(x + 0.5*width, f1_scores, width, label="F1-Score", color="#4CAF50")
    ax.bar(x + 1.5*width, iou_scores, width, label="IoU", color="#9C27B0")

    ax.set_ylabel("Score")
    ax.set_title("Métricas de Validação por Feição Supraglacial")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in features])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    plot_path = output_dir / "validation_metrics.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Gráfico salvo: {plot_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validação das predições com métricas de segmentação"
    )
    parser.add_argument(
        "--feature", type=str, default=None,
        choices=list(Config.FEATURES.keys()),
        help="Feição específica. Se omitido, valida todas."
    )
    parser.add_argument(
        "--year", type=int, default=None,
        choices=Config.YEARS,
        help="Ano específico. Se omitido, valida todos com GT."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FASE 6 - VALIDAÇÃO DAS PREDIÇÕES")
    print("=" * 60)

    start = time.time()

    features = [args.feature] if args.feature else list(Config.FEATURES.keys())
    years = [args.year] if args.year else Config.YEARS

    all_results = []
    for feature in features:
        for year in years:
            result = validate_feature(feature, year)
            if result:
                all_results.append(result)

    # Salvar resultados completos
    if all_results:
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = Config.RESULTS_DIR / "validation_results.json"

        # Remover per_tile para o JSON principal (muito grande)
        summary = []
        for r in all_results:
            s = {k: v for k, v in r.items() if k != "per_tile"}
            summary.append(s)

        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n  Resultados salvos: {results_path}")

        # Gerar gráfico
        plot_results(all_results)

    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO DA VALIDAÇÃO")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['feature']}/{r['year']}: "
              f"F1={r['micro']['f1']:.4f} | "
              f"IoU={r['micro']['iou']:.4f} | "
              f"P={r['micro']['precision']:.4f} | "
              f"R={r['micro']['recall']:.4f}")

    elapsed = time.time() - start
    print(f"\nTempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()

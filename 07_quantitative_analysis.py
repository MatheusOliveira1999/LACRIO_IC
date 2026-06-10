"""
07_quantitative_analysis.py - Camada 1 da analise de fendas

Gera:
  - results/crevasses_stats_per_year.csv  (uma linha por ano)
  - results/figures/quantitative/temporal_evolution.png
  - results/figures/quantitative/size_distribution.png
  - results/figures/quantitative/validation_metrics.png

Uso:
  python 07_quantitative_analysis.py                     # todos os anos
  python 07_quantitative_analysis.py --years 2016 2017   # subset
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config


YEARS_DEFAULT = [2016, 2017, 2018, 2019]   # 2020 tem mosaico mas nao tem GT
ANNOTATED_YEARS = [2016, 2017, 2018, 2019]


# Skeleton: tenta skimage, depois cv2.ximgproc, depois fallback
def _make_skeleton_fn():
    try:
        from skimage.morphology import skeletonize
        def fn(binary):
            return skeletonize(binary > 0).astype(np.uint8)
        return fn, "skimage"
    except ImportError:
        pass
    try:
        thin = cv2.ximgproc.thinning
        def fn(binary):
            return (thin((binary > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
        return fn, "cv2.ximgproc"
    except (AttributeError, ImportError):
        pass
    return None, None


_skel_fn, _skel_lib = _make_skeleton_fn()


USE_NORMALIZED = False  # setado pelo --normalized no main


def find_mosaic_path(year):
    """Encontra mosaico do ano.

    Se USE_NORMALIZED estiver True, busca primeiro versao com sufixo _8cm.
    Senao, prefere nome canonico.
    """
    base_dir = Config.RESULTS_DIR / str(year)
    if USE_NORMALIZED:
        normalized = sorted(base_dir.glob(f"crevasses_mask_{year}_*cm.tif"))
        if normalized:
            return normalized[0]
    canonical = base_dir / f"crevasses_mask_{year}.tif"
    if canonical.exists():
        return canonical
    variants = sorted(base_dir.glob(f"crevasses_mask_{year}.tif"))
    variants += sorted(base_dir.glob(f"crevasses_mask_{year}_[0-9].tif"))
    return variants[0] if variants else None


def analyze_year(year):
    """Calcula estatisticas geometricas para um ano."""
    path = find_mosaic_path(year)
    if path is None:
        print(f"  [PULADO] {year}: mosaico nao encontrado")
        return None, None

    print(f"\n=== {year} ===")
    print(f"  Arquivo: {path.name}")

    with rasterio.open(path) as src:
        mask = src.read(1)
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        crs = src.crs

    print(f"  CRS: {crs} | Resolucao: {res_x*100:.2f} x {res_y*100:.2f} cm/px")
    print(f"  Dimensoes: {mask.shape[1]} x {mask.shape[0]} px")

    binary = (mask > 127).astype(np.uint8)
    total_pixels = int(binary.size)
    feature_pixels = int(binary.sum())
    pixel_area_m2 = res_x * res_y

    # Componentes conectados
    num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas_px = cc_stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])

    # Comprimento total
    if _skel_fn is not None:
        skeleton = _skel_fn(binary)
        skeleton_pixels = int(skeleton.sum())
        # cada pixel do skeleton conta como (res_x + res_y)/2 (aproximacao isotropica)
        pixel_len_m = (res_x + res_y) / 2.0
        total_length_m = skeleton_pixels * pixel_len_m
        length_method = _skel_lib
    else:
        # Fallback: somar lado maior do minAreaRect de cada componente
        # Adequado para fendas retas/lineares (definicao estrita)
        total_length_m = 0.0
        for label_id in range(1, num_labels):
            comp = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            rect = cv2.minAreaRect(contours[0])
            w_px, h_px = rect[1]
            major_axis_m = max(w_px, h_px) * res_x  # assumindo res_x ~= res_y
            total_length_m += major_axis_m
        length_method = "minAreaRect_fallback"

    stats_dict = {
        "year": year,
        "file": path.name,
        "resolution_cm_per_px": round(res_x * 100, 3),
        "total_pixels": total_pixels,
        "feature_pixels": feature_pixels,
        "coverage_percent": round(100.0 * feature_pixels / total_pixels, 4),
        "n_components": int(num_labels - 1),
        "total_area_m2": round(feature_pixels * pixel_area_m2, 2),
        "total_area_km2": round(feature_pixels * pixel_area_m2 / 1e6, 6),
        "mean_area_m2": round(float(component_areas_px.mean() * pixel_area_m2), 4) if len(component_areas_px) else 0.0,
        "median_area_m2": round(float(np.median(component_areas_px) * pixel_area_m2), 4) if len(component_areas_px) else 0.0,
        "p95_area_m2": round(float(np.percentile(component_areas_px, 95) * pixel_area_m2), 4) if len(component_areas_px) else 0.0,
        "max_area_m2": round(float(component_areas_px.max() * pixel_area_m2), 2) if len(component_areas_px) else 0.0,
        "total_length_m": round(total_length_m, 2),
        "length_method": length_method,
    }

    print(f"  Componentes (fendas): {stats_dict['n_components']:,}")
    print(f"  Area total:           {stats_dict['total_area_km2']:.4f} km²   ({stats_dict['total_area_m2']:,.0f} m²)")
    print(f"  Tamanho de fenda:     mediana {stats_dict['median_area_m2']:.2f} m²  p95 {stats_dict['p95_area_m2']:.2f} m²")
    print(f"  Comprimento total:    {stats_dict['total_length_m']:,.0f} m  ({length_method})")
    print(f"  Cobertura do mosaico: {stats_dict['coverage_percent']:.3f}%")

    component_areas_m2 = component_areas_px * pixel_area_m2
    return stats_dict, component_areas_m2


def load_validation_metrics(years):
    """Carrega per-tile e global de cada validation json."""
    metrics = []
    for year in years:
        path = Config.RESULTS_DIR / f"unet_validation_crevasses_{year}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        global_m = data.get("micro", data.get("global", {}))
        per_tile = data.get("per_tile", [])
        if not global_m:
            continue
        metrics.append({"year": year, "global": global_m, "per_tile": per_tile})
    return metrics


def plot_temporal_evolution(stats_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(stats_df["year"], stats_df["total_area_km2"], marker="o",
                 linewidth=2, markersize=10, color="#c0392b")
    axes[0].set_xlabel("Ano")
    axes[0].set_ylabel("Área total de fendas (km²)")
    axes[0].set_title("Evolução temporal — área total")
    axes[0].grid(alpha=0.3)
    for x, y in zip(stats_df["year"], stats_df["total_area_km2"]):
        axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    axes[1].plot(stats_df["year"], stats_df["n_components"], marker="s",
                 linewidth=2, markersize=10, color="#2980b9")
    axes[1].set_xlabel("Ano")
    axes[1].set_ylabel("Número de fendas (componentes)")
    axes[1].set_title("Evolução temporal — contagem")
    axes[1].grid(alpha=0.3)
    for x, y in zip(stats_df["year"], stats_df["n_components"]):
        axes[1].annotate(f"{y:,}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    axes[2].plot(stats_df["year"], stats_df["total_length_m"] / 1000, marker="^",
                 linewidth=2, markersize=10, color="#27ae60")
    axes[2].set_xlabel("Ano")
    axes[2].set_ylabel("Comprimento total (km)")
    axes[2].set_title("Evolução temporal — comprimento")
    axes[2].grid(alpha=0.3)
    for x, y in zip(stats_df["year"], stats_df["total_length_m"] / 1000):
        axes[2].annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_size_distribution(areas_by_year, out_path):
    n_years = len(areas_by_year)
    fig, axes = plt.subplots(1, n_years, figsize=(3.5 * n_years, 4), sharey=True)
    if n_years == 1:
        axes = [axes]

    # bin edges comuns (log) baseados no range global
    all_areas = np.concatenate([a for a in areas_by_year.values() if len(a) > 0])
    if len(all_areas) == 0:
        return
    lo = max(all_areas.min(), 0.1)
    hi = all_areas.max() * 1.1
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)

    for ax, (year, areas) in zip(axes, sorted(areas_by_year.items())):
        if len(areas) == 0:
            ax.set_title(f"{year} — vazio")
            continue
        ax.hist(areas, bins=bins, log=True, color="#34495e",
                alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Área da fenda (m²)")
        ax.set_title(f"{year} (n={len(areas):,})")
        ax.grid(alpha=0.3)
        ax.axvline(np.median(areas), color="#e74c3c", linestyle="--",
                   linewidth=1.2, alpha=0.8)
    axes[0].set_ylabel("Frequência (log)")

    plt.suptitle("Distribuição de tamanho de fendas — escala log/log    "
                 "(linha tracejada = mediana)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_validation_metrics(val_metrics, out_path):
    if not val_metrics:
        print("  [AVISO] Sem metricas de validacao para plotar.")
        return

    years = [m["year"] for m in val_metrics]
    f1 = [m["global"].get("f1", np.nan) for m in val_metrics]
    p = [m["global"].get("precision", np.nan) for m in val_metrics]
    r = [m["global"].get("recall", np.nan) for m in val_metrics]
    iou = [m["global"].get("iou", np.nan) for m in val_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(years))
    width = 0.2
    axes[0].bar(x - 1.5 * width, f1, width, label="F1", color="#c0392b")
    axes[0].bar(x - 0.5 * width, iou, width, label="IoU", color="#e67e22")
    axes[0].bar(x + 0.5 * width, p, width, label="Precision", color="#2980b9")
    axes[0].bar(x + 1.5 * width, r, width, label="Recall", color="#27ae60")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(years)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Ano")
    axes[0].set_ylabel("Métrica (micro)")
    axes[0].set_title("Validação — métricas globais por ano")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)
    for i, (v_f1, v_iou) in enumerate(zip(f1, iou)):
        axes[0].annotate(f"{v_f1:.3f}", (i - 1.5 * width, v_f1),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=8)
        axes[0].annotate(f"{v_iou:.3f}", (i - 0.5 * width, v_iou),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=8)

    f1_per_tile_list = [[t["f1"] for t in m["per_tile"]] for m in val_metrics]
    bp = axes[1].boxplot(f1_per_tile_list, tick_labels=years, showmeans=True,
                         meanprops=dict(marker="D", markerfacecolor="#c0392b",
                                        markeredgecolor="#c0392b", markersize=7),
                         medianprops=dict(color="#2c3e50", linewidth=1.5),
                         flierprops=dict(marker="o", markersize=4, alpha=0.5))
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Ano")
    axes[1].set_ylabel("F1 por tile")
    axes[1].set_title("Distribuição F1 per-tile (caixa=quartis, ◇=média)")
    axes[1].grid(axis="y", alpha=0.3)
    for i, vals in enumerate(f1_per_tile_list):
        axes[1].annotate(f"n={len(vals)}", (i + 1, -0.02),
                         ha="center", fontsize=8, color="#7f8c8d")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Camada 1 — analise quantitativa de fendas")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT)
    parser.add_argument("--out-dir", type=Path,
                        default=Config.RESULTS_DIR / "figures" / "quantitative")
    parser.add_argument("--normalized", action="store_true",
                        help="Usa mosaicos padronizados (sufixo _8cm). "
                             "Rode 00_normalize_resolution.py primeiro.")
    args = parser.parse_args()

    global USE_NORMALIZED
    USE_NORMALIZED = args.normalized

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CAMADA 1 - ANALISE QUANTITATIVA DE FENDAS")
    print("=" * 60)
    print(f"Anos: {args.years}")
    print(f"Skeleton lib: {_skel_lib or 'nao disponivel (fallback minAreaRect)'}")

    all_stats = []
    areas_by_year = {}
    for year in args.years:
        result = analyze_year(year)
        if result == (None, None):
            continue
        stats_dict, areas = result
        all_stats.append(stats_dict)
        areas_by_year[year] = areas

    if not all_stats:
        print("\nNenhum ano processado.")
        return

    stats_df = pd.DataFrame(all_stats)
    csv_path = Config.RESULTS_DIR / "crevasses_stats_per_year.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"CSV salvo: {csv_path}")
    print(f"{'=' * 60}")
    cols_display = ["year", "n_components", "total_area_km2",
                    "median_area_m2", "p95_area_m2", "total_length_m",
                    "coverage_percent"]
    print(stats_df[cols_display].to_string(index=False))

    val_metrics = load_validation_metrics(ANNOTATED_YEARS)
    print(f"\nMetricas de validacao encontradas: {len(val_metrics)} anos")

    print(f"\nGerando figuras em {args.out_dir}/")
    plot_temporal_evolution(stats_df, args.out_dir / "temporal_evolution.png")
    print(f"  OK temporal_evolution.png")
    plot_size_distribution(areas_by_year, args.out_dir / "size_distribution.png")
    print(f"  OK size_distribution.png")
    plot_validation_metrics(val_metrics, args.out_dir / "validation_metrics.png")
    print(f"  OK validation_metrics.png")

    print("\nCAMADA 1 CONCLUIDA.")


if __name__ == "__main__":
    main()

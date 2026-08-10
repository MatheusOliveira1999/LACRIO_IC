"""
12_hd95_metric.py - HD95 (Hausdorff 95%) de borda entre predicao e GT, por ano.

Definicao identica a Duan et al. 2025 (IJAEOG 144:104881, Eq. 3-12):
  HD95(A,B) = max( P95_{a em A} min_b |a-b| , P95_{b em B} min_a |a-b| )
com A e B = conjuntos de pixels de BORDA da predicao e do GT, por tile;
agregacao = media e mediana entre tiles (tiles com pred e GT nao vazios).

Referencia de comparacao: Duan et al. 2025 reportam HD95 = 6.51 px a 3 cm/px
(~19.5 cm) no teste in-domain deles.

Uso: python 12_hd95_metric.py   (usa as predicoes atuais em masks/{ano}/crevasses/)
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion

from config import Config

YEARS = [2016, 2017, 2018, 2019]
GSD_CM = {2016: 8.23, 2017: 7.85, 2018: 8.23, 2019: 5.42}  # nativo aproximado por ano


def boundary(mask):
    m = mask > 127
    return m & ~binary_erosion(m)


def hd95_pair(pred, gt):
    bp, bg = boundary(pred), boundary(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return None
    dt_g = distance_transform_edt(~bg)
    dt_p = distance_transform_edt(~bp)
    d_pg = dt_g[bp]   # pred -> GT
    d_gp = dt_p[bg]   # GT -> pred
    return max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))


def matched_only(pred, gt):
    """Mantem so componentes com alguma sobreposicao no outro conjunto —
    separa LOCALIZACAO de borda (isto) de COMPLETUDE de deteccao (HD95 bruto)."""
    p = (pred > 127).astype(np.uint8)
    g = (gt > 127).astype(np.uint8)
    np_, lp = cv2.connectedComponents(p)
    ng_, lg = cv2.connectedComponents(g)
    keep_p = {i for i in range(1, np_) if (g[lp == i]).any()}
    keep_g = {i for i in range(1, ng_) if (p[lg == i]).any()}
    pm = np.isin(lp, list(keep_p)).astype(np.uint8) * 255 if keep_p else np.zeros_like(pred)
    gm = np.isin(lg, list(keep_g)).astype(np.uint8) * 255 if keep_g else np.zeros_like(gt)
    return pm, gm


def main():
    print(f"{'Ano':<6}{'n tiles':>8}{'HD95 med (px)':>15}{'HD95 média (px)':>17}"
          f"{'HD95 med (cm)':>15}{'excl.':>7}")
    print("-" * 68)
    for year in YEARS:
        gt_dir = Config.MASKS_DIR / str(year) / "annotations" / "crevasses"
        pred_dir = Config.MASKS_DIR / str(year) / "crevasses"
        vals, vals_m, skipped = [], [], 0
        for gt_path in sorted(gt_dir.glob("tile_*_crevasses.png")):
            pred_path = pred_dir / gt_path.name
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            pred = (cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                    if pred_path.exists()
                    else np.zeros_like(gt))
            h = hd95_pair(pred, gt)
            if h is None:
                skipped += 1
                continue
            vals.append(h)
            hm = hd95_pair(*matched_only(pred, gt))
            if hm is not None:
                vals_m.append(hm)
        v, vm = np.array(vals), np.array(vals_m)
        cm = np.median(v) * GSD_CM[year]
        cm_m = np.median(vm) * GSD_CM[year] if len(vm) else float("nan")
        print(f"{year:<6}{len(v):>8}{np.median(v):>15.2f}{v.mean():>17.2f}"
              f"{cm:>15.1f}{skipped:>7}"
              f"   | casados: {np.median(vm):>6.2f} px = {cm_m:>6.1f} cm (n={len(vm)})")
    print("\nReferencia Duan et al. 2025 (in-domain, gelo seco, 3 cm/px): "
          "HD95 = 6.51 px = ~19.5 cm")
    print("(excl. = tiles sem borda em pred ou GT — HD95 indefinido, "
          "convencao igual a de metricas por imagem)")


if __name__ == "__main__":
    main()

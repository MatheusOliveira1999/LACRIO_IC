"""
14_crevasse_depth.py - Profundidade APARENTE das fendas por plano de referencia local.

Adaptacao do workflow de Duan et al. 2025 (IJAEOG 144:104881, secao 3.4.3) do
LiDAR para o nosso MDE fotogrametrico (~31 cm):
  1. Por componente de fenda (mascara 8 cm, area >= 1 m2): anel de vizinhanca
     (dilatacao ate ~2 m) EXCLUINDO qualquer pixel de fenda.
  2. Plano de referencia local z = ax+by+c ajustado ao anel (minimos quadrados),
     com controle de qualidade: RMSE do plano < 0.15 m e n_anel >= 200 px.
  3. Profundidade dos pixels internos = plano - MDE; profundidade da fenda =
     Q0.75 da distribuicao positiva (conservador, como Duan), exigindo >= 30%
     dos pixels internos abaixo do plano.

LIMITACOES DECLARADAS: fotogrametria nao enxerga o fundo de fendas estreitas/
sombreadas (o proprio Duan admite subestimacao ATE com LiDAR) e o MDE de 31 cm
so resolve fendas com largura >~ 2 px de MDE (~0.6 m). Os valores sao
PROFUNDIDADE APARENTE (limite inferior), validos para a cauda larga da
populacao.

Saidas: results/crevasse_depth_{ano}.csv + resumo no terminal.
"""

import csv
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import label as ndlabel, find_objects, binary_dilation

from config import Config

YEARS = [2016, 2017, 2018, 2019]
MIN_AREA_PX = 157        # 1 m2 a 8 cm
RING_DIL = 25            # ~2 m de anel
PAD = 60                 # ~4.8 m de janela extra
RMSE_MAX = 0.15          # QC do plano (m)
MIN_RING = 200
MIN_FRAC_BELOW = 0.30


def process_year(year):
    mpath = Config.RESULTS_DIR / str(year) / f"crevasses_mask_{year}_8cm.tif"
    msrc = rasterio.open(mpath)
    mask = msrc.read(1) > 0
    dsrc = rasterio.open(Config.get_dem_path(year))
    vrt = WarpedVRT(dsrc, crs=msrc.crs, transform=msrc.transform,
                    width=msrc.width, height=msrc.height,
                    resampling=Resampling.bilinear, nodata=np.nan)

    lab, n = ndlabel(mask)
    slices = find_objects(lab)
    px_area = abs(msrc.transform.a * msrc.transform.e)
    rows, stats = [], {"total": 0, "qc_plane": 0, "qc_below": 0, "ok": 0}

    for cid, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        comp_full = (lab[sl] == cid)
        area_px = int(comp_full.sum())
        if area_px < MIN_AREA_PX:
            continue
        stats["total"] += 1

        r0 = max(0, sl[0].start - PAD); r1 = min(msrc.height, sl[0].stop + PAD)
        c0 = max(0, sl[1].start - PAD); c1 = min(msrc.width, sl[1].stop + PAD)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        z = vrt.read(1, window=win)
        sub_lab = lab[r0:r1, c0:c1]
        comp = (sub_lab == cid)
        any_crev = mask[r0:r1, c0:c1]

        ring = binary_dilation(comp, iterations=RING_DIL) & ~any_crev & np.isfinite(z)
        if ring.sum() < MIN_RING:
            continue
        ry, rx = np.nonzero(ring)
        A = np.column_stack([rx * 0.08, ry * 0.08, np.ones(len(rx))])
        coef, *_ = np.linalg.lstsq(A, z[ring], rcond=None)
        rmse = float(np.sqrt(np.mean((A @ coef - z[ring]) ** 2)))
        if rmse > RMSE_MAX:
            stats["qc_plane"] += 1
            continue

        iy, ix = np.nonzero(comp & np.isfinite(z))
        if len(iy) < 10:
            continue
        plane = coef[0] * (ix * 0.08) + coef[1] * (iy * 0.08) + coef[2]
        d = plane - z[iy, ix]
        below = d[d > 0]
        if len(below) < MIN_FRAC_BELOW * len(d):
            stats["qc_below"] += 1
            continue
        depth = float(np.percentile(below, 75))
        cy = (r0 + iy.mean()) ; cx = (c0 + ix.mean())
        X, Y = msrc.transform * (cx, cy)
        rows.append([cid, round(area_px * px_area, 2), round(depth, 3),
                     round(float(below.max()), 3), round(rmse, 3),
                     int(ring.sum()), round(X, 1), round(Y, 1)])
        stats["ok"] += 1

    out = Config.RESULTS_DIR / f"crevasse_depth_{year}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comp_id", "area_m2", "depth_q75_m", "depth_max_m",
                    "plane_rmse_m", "n_ring", "utm_x", "utm_y"])
        w.writerows(rows)

    d = np.array([r[2] for r in rows]) if rows else np.array([np.nan])
    print(f"{year}: candidatas(>=1 m2)={stats['total']} | com profundidade={stats['ok']} "
          f"| reprovadas QC plano={stats['qc_plane']} / QC interior={stats['qc_below']}")
    if rows:
        print(f"   profundidade aparente: mediana={np.median(d):.2f} m | "
              f"media={d.mean():.2f} ± {d.std():.2f} m | max={d.max():.2f} m")
    vrt.close(); dsrc.close(); msrc.close()
    return rows


def main():
    print("PROFUNDIDADE APARENTE (Q0.75 abaixo do plano local; limite inferior fotogrametrico)")
    print("=" * 84)
    for y in YEARS:
        process_year(y)
    print("\nReferencia Duan et al. 2025 (UAV-LiDAR, >500 pts/m2): media 3.06 ± 3.91 m, "
          "max 26.69 m\n(comparacao indicativa — LiDAR penetra mais que fotogrametria)")


if __name__ == "__main__":
    main()

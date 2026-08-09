"""
fig07_dh_mapas.py - Fig. 7 do relatorio final IC.

Mapas de dH (DEM differencing UAV) para os pares 2016-17, 2017-18 e 2018-19,
sobre hillshade do DEM 2016, com contornos de elevacao e colorbar unica.

Metodo (identico ao 11_density_dh_analysis.py):
  - DEMs (~31 cm) reamostrados bilinearmente (WarpedVRT) para grade comum de
    2 m na malha da mascara 8 cm de 2016 (EPSG:32719);
  - dominio "gelo" por proxy de afinamento < -0.5 m em intervalos
    COMPLEMENTARES ao par analisado (evita circularidade);
  - pixels do dominio gelo sem DEM valido em um dos anos do par: cinza;
  - colormap divergente RdBu com pivo no zero (TwoSlopeNorm), vermelho = perda,
    limites IDENTICOS e assimetricos (perda domina), como em Hugonnet et al. 2021;
  - dH mediano do dominio exibido anotado dentro de cada painel.

Saida: results/figures/relatorio/fig07_dh_mapas.png (300 dpi)
Uso:   python report_figs/fig07_dh_mapas.py  (cwd = raiz do repo)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "relatorio" / "fig07_dh_mapas.png"

YEARS = [2016, 2017, 2018, 2019]
PAIRS = [(2016, 2017), (2017, 2018), (2018, 2019)]
AGG = 25
ICE_THR = -0.5
VMIN, VMAX = -12.0, 6.0   # p2~-13 / p98~+6-9 m; perda domina -> assimetrico
GRAY = "#b8b8b8"
CONTOUR_COLOR = "#4a6b8a"  # cinza-azulado

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
})


def load_dems():
    ref_path = ROOT / "results" / "2016" / "crevasses_mask_2016_8cm.tif"
    with rasterio.open(ref_path) as ref:
        h2, w2 = ref.height // AGG, ref.width // AGG
        transform2m = ref.transform * ref.transform.scale(AGG, AGG)
        crs = ref.crs
    dem = {}
    for y in YEARS:
        with rasterio.open(ROOT / "Schiaparelli_glacier" / f"Schiaparelli_DEM_{y}.tif") as src:
            with WarpedVRT(src, crs=crs, transform=transform2m, width=w2, height=h2,
                           resampling=Resampling.bilinear, nodata=np.nan) as vrt:
                z = vrt.read(1).astype(np.float64)
        z[z < -1000] = np.nan
        dem[y] = z
        print(f"  DEM {y} -> grade 2 m ok")
    return dem, transform2m, (h2, w2)


def hillshade(z, res=2.0, az=315.0, alt=45.0):
    zs = gaussian_filter(np.where(np.isfinite(z), z, np.nanmean(z)), 1.5)
    gy, gx = np.gradient(zs, res)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azr, altr = np.radians(360 - az + 90), np.radians(alt)
    hs = np.sin(altr) * np.sin(slope) + np.cos(altr) * np.cos(slope) * np.cos(azr - aspect)
    hs = (hs - np.nanmin(hs)) / (np.nanmax(hs) - np.nanmin(hs))
    hs[~np.isfinite(z)] = np.nan
    return hs


def km_fmt(v, _):
    return f"{v / 1000:.1f}"


def main():
    print("Fig. 7 - mapas de dH por par de anos")
    dem, T, (h2, w2) = load_dems()

    # proxy de gelo por afinamento em intervalos complementares (11_density_dh)
    ice_proxy = {
        (2016, 2017): (dem[2019] - dem[2017]) < ICE_THR,
        (2017, 2018): ((dem[2017] - dem[2016]) < ICE_THR)
                      | ((dem[2019] - dem[2018]) < ICE_THR),
        (2018, 2019): (dem[2018] - dem[2016]) < ICE_THR,
    }

    left, top = T.c, T.f
    ext = (left, left + T.a * w2, top + T.e * h2, top)
    hs = hillshade(dem[2016])
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)

    fig = plt.figure(figsize=(7.48, 3.25))
    gs = fig.add_gridspec(1, 3, wspace=0.06,
                          left=0.075, right=0.995, top=0.91, bottom=0.335)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]

    letters = ["a", "b", "c"]
    im = None
    for i, (t1, t2) in enumerate(PAIRS):
        ax = axes[i]
        ice = ice_proxy[(t1, t2)]
        both = np.isfinite(dem[t1]) & np.isfinite(dem[t2])
        dh = np.where(ice & both, dem[t2] - dem[t1], np.nan)
        med = float(np.nanmedian(dh))

        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext, origin="upper",
                  interpolation="bilinear", alpha=0.85, zorder=0)
        # gelo sem DEM valido em um dos anos: cinza
        gap = np.where(ice & ~both, 1.0, np.nan)
        ax.imshow(gap, cmap=ListedColormap([GRAY]), extent=ext, origin="upper",
                  interpolation="nearest", zorder=1)
        im = ax.imshow(dh, cmap="RdBu", norm=norm, extent=ext, origin="upper",
                       interpolation="nearest", zorder=2)

        # contorno do gelo
        outline = gaussian_filter(ice.astype(float), 4)
        ax.contour(outline, levels=[0.5], colors="k", linewidths=0.4,
                   extent=(ext[0], ext[1], ext[3], ext[2]), zorder=4)

        # contornos de elevacao (DEM do ano inicial), rotulados a cada 100 m
        zsm = gaussian_filter(np.where(np.isfinite(dem[t1]), dem[t1],
                                       np.nanmean(dem[t1])), 8)
        zsm[~np.isfinite(dem[t1])] = np.nan
        cs = ax.contour(np.flipud(zsm), levels=[50, 100, 150, 200], extent=ext,
                        colors=CONTOUR_COLOR, linewidths=0.45, alpha=0.9, zorder=3)
        ax.clabel(cs, levels=[100, 200], fmt="%.0f m", fontsize=8,
                  inline=True, inline_spacing=2)

        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(km_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(km_fmt))
        ax.tick_params(labelsize=8, length=2, pad=1)
        if i > 0:
            ax.set_yticklabels([])
        ax.set_title(f"({letters[i]}) {t1} → {t2}", fontsize=9, pad=2)
        ax.text(0.03, 0.05, f"dH mediano = {med:+.2f} m".replace(".", ","),
                transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.25"), zorder=5)
        print(f"  {t1}->{t2}: dH mediano exibido = {med:+.2f} m")

    # scale bar (canto superior esquerdo, fora do gelo) + norte no painel (a)
    ax = axes[0]
    x0, y0 = ext[0] + 130, ext[3] - 300
    ax.plot([x0, x0 + 500], [y0, y0], color="k", lw=2, solid_capstyle="butt", zorder=6)
    ax.text(x0 + 250, y0 - 60, "500 m", ha="center", va="top", fontsize=8, zorder=6)
    xn, yn = ext[1] - 170, ext[3] - 430
    ax.annotate("N", xy=(xn, yn + 260), xytext=(xn, yn), fontsize=9,
                fontweight="bold", ha="center", va="center", zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))

    fig.text(0.012, 0.65, "UTM N (km)", rotation=90, va="center", fontsize=8)
    fig.text(0.535, 0.235, "UTM E (km)", ha="center", fontsize=8)

    # colorbar horizontal unica
    cax = fig.add_axes([0.28, 0.135, 0.38, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both",
                      ticks=[-12, -9, -6, -3, 0, 3, 6])
    cb.set_label("dH (m) — vermelho = perda de elevação", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    fig.legend(handles=[Patch(facecolor=GRAY, edgecolor="none",
                              label="gelo sem DEM\nválido no par")],
               loc="center left", bbox_to_anchor=(0.70, 0.10), fontsize=8,
               frameon=False, handlelength=1.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  salvo: {OUT}")


if __name__ == "__main__":
    main()

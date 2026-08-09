"""
fig06_densidade_persistencia.py - Fig. 6 do relatorio final IC.

Densidade de fendas em grade de 100 m (2016-2019), persistencia interanual e
variacao de densidade 2016->2019 sobre hillshade do DEM UAV 2016.

Metodo (identico ao 11_density_dh_analysis.py):
  - grade comum de 2 m na malha da mascara 8 cm de 2016 (media 25x25 px);
  - blocos de 100 m = 50x50 celulas de 2 m;
  - dominio "gelo" = afinamento (DEM2019 - DEM2016) < -0.5 m;
  - densidade do bloco = media da fracao de fenda nas celulas de gelo (%);
  - persistencia = n de anos (1-4) com densidade > 0.5% no bloco;
  - blocos com cobertura de DEM < 50% em area majoritariamente glacial: cinza.

Saida: results/figures/relatorio/fig06_densidade_persistencia.png (300 dpi)
Uso:   python report_figs/fig06_densidade_persistencia.py  (cwd = raiz do repo)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "relatorio" / "fig06_densidade_persistencia.png"

YEARS = [2016, 2017, 2018, 2019]
YEAR_COLORS = {2016: "#0072B2", 2017: "#E69F00", 2018: "#009E73", 2019: "#D55E00"}
AGG = 25      # 25 px de 8 cm -> celula de 2 m
BLOCK = 50    # 50 celulas de 2 m -> bloco de 100 m
ICE_THR = -0.5      # m de afinamento 2016->2019 que define gelo
PERSIST_THR = 0.5   # % de densidade para "fenda presente" no bloco
DENS_VMAX = 10.0    # % (p98 dos blocos ~9.2%)
GRAY = "#b8b8b8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
})


def load_grids():
    ref_path = ROOT / "results" / "2016" / "crevasses_mask_2016_8cm.tif"
    with rasterio.open(ref_path) as ref:
        h2, w2 = ref.height // AGG, ref.width // AGG
        transform2m = ref.transform * ref.transform.scale(AGG, AGG)
        crs = ref.crs

    dens = {}
    for y in YEARS:
        mpath = ROOT / "results" / str(y) / f"crevasses_mask_{y}_8cm.tif"
        with rasterio.open(mpath) as src:
            a = src.read(1, out_shape=(h2 * AGG, w2 * AGG))[: h2 * AGG, : w2 * AGG]
        dens[y] = (a > 0).astype(np.float32).reshape(h2, AGG, w2, AGG).mean(axis=(1, 3))
        del a
        print(f"  densidade 2 m {y}: ok")

    dem = {}
    for y in (2016, 2019):
        with rasterio.open(ROOT / "Schiaparelli_glacier" / f"Schiaparelli_DEM_{y}.tif") as src:
            with WarpedVRT(src, crs=crs, transform=transform2m, width=w2, height=h2,
                           resampling=Resampling.bilinear, nodata=np.nan) as vrt:
                z = vrt.read(1).astype(np.float64)
        z[z < -1000] = np.nan
        dem[y] = z
    return dens, dem, transform2m, (h2, w2)


def blocks(a, f=np.nanmean, B=BLOCK):
    H, W = a.shape
    h, w = H // B * B, W // B * B
    return f(a[:h, :w].reshape(h // B, B, w // B, B), axis=(1, 3))


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
    print("Fig. 6 - densidade / persistencia / delta-densidade")
    dens, dem, T, (h2, w2) = load_grids()

    valid = np.isfinite(dem[2016]) & np.isfinite(dem[2019])
    ice = valid & ((dem[2019] - dem[2016]) < ICE_THR)

    ice_f = blocks(ice.astype(float))
    valid_f = blocks(valid.astype(float))
    ice_of_valid = np.divide(blocks((ice & valid).astype(float)),
                             np.maximum(valid_f, 1e-9))
    on_glacier = ice_f >= 0.5
    # cinza: cobertura de DEM <50% mas o que ha de dado e majoritariamente gelo
    gray_b = (~on_glacier) & (valid_f < 0.5) & (valid_f > 0.05) & (ice_of_valid > 0.5)

    dens_b = {}
    for y in YEARS:
        d = blocks(np.where(ice, dens[y], np.nan)) * 100.0
        d[~on_glacier] = np.nan
        dens_b[y] = d

    persist = np.zeros_like(dens_b[2016])
    for y in YEARS:
        persist += (dens_b[y] > PERSIST_THR).astype(float)
    persist[~on_glacier] = np.nan

    delta = dens_b[2019] - dens_b[2016]
    dlim = float(np.ceil(np.nanpercentile(np.abs(delta), 99)))
    print(f"  blocos no gelo: {int(on_glacier.sum())} | delta p99 = {dlim:.0f} p.p.")

    # extents UTM
    left, top = T.c, T.f
    ext2m = (left, left + T.a * w2, top + T.e * h2, top)
    nbr, nbc = dens_b[2016].shape
    extb = (left, left + T.a * BLOCK * nbc, top + T.e * BLOCK * nbr, top)

    hs = hillshade(dem[2016])
    outline = gaussian_filter(ice.astype(float), 4)

    fig = plt.figure(figsize=(7.48, 5.15))
    gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.08,
                          left=0.08, right=0.99, top=0.95, bottom=0.205)
    axes = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in range(3)]

    dens_cmap = plt.get_cmap("Reds").copy()
    gray_masked = np.where(gray_b, 1.0, np.nan)

    def base(ax):
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext2m, origin="upper",
                  interpolation="bilinear", alpha=0.85, zorder=0)
        ax.imshow(gray_masked, cmap=ListedColormap([GRAY]), extent=extb,
                  origin="upper", interpolation="nearest", zorder=1)
        ax.contour(outline, levels=[0.5], colors="k", linewidths=0.4,
                   extent=(ext2m[0], ext2m[1], ext2m[3], ext2m[2]), zorder=3)
        ax.set_xlim(ext2m[0], ext2m[1])
        ax.set_ylim(ext2m[2], ext2m[3])
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(km_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(km_fmt))
        ax.tick_params(labelsize=8, length=2, pad=1)

    ims = {}
    letters = ["a", "b", "c", "d", "e", "f"]
    for i, y in enumerate(YEARS):
        ax = axes[i]
        base(ax)
        ims["dens"] = ax.imshow(dens_b[y], cmap=dens_cmap, vmin=0, vmax=DENS_VMAX,
                                extent=extb, origin="upper", interpolation="nearest",
                                zorder=2)
        ax.set_title(f"({letters[i]}) {y}", fontsize=9, color=YEAR_COLORS[y],
                     fontweight="bold", pad=2)

    # (e) persistencia
    ax = axes[4]
    base(ax)
    pers_cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.1, 0.92, 4)))
    pers_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4)
    pshow = np.where(persist >= 1, persist, np.nan)
    ims["pers"] = ax.imshow(pshow, cmap=pers_cmap, norm=pers_norm, extent=extb,
                            origin="upper", interpolation="nearest", zorder=2)
    ax.set_title("(e) Persistência 2016–2019", fontsize=9, pad=2)

    # (f) delta densidade
    ax = axes[5]
    base(ax)
    ims["delta"] = ax.imshow(delta, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                             extent=extb, origin="upper", interpolation="nearest",
                             zorder=2)
    ax.set_title("(f) Δ densidade 2016→2019", fontsize=9, pad=2)

    # ticks apenas nas bordas externas
    for i, ax in enumerate(axes):
        if i % 3 != 0:
            ax.set_yticklabels([])
        if i < 3:
            ax.set_xticklabels([])
    fig.text(0.012, 0.60, "UTM N (km)", rotation=90, va="center", fontsize=8)
    fig.text(0.545, 0.162, "UTM E (km)", ha="center", fontsize=8)

    # scale bar + norte no painel (a)
    ax = axes[0]
    x0, y0 = ext2m[0] + 130, ext2m[2] + 150
    ax.plot([x0, x0 + 500], [y0, y0], color="k", lw=2, solid_capstyle="butt", zorder=5)
    ax.text(x0 + 250, y0 + 45, "500 m", ha="center", va="bottom", fontsize=8, zorder=5)
    xn, yn = ext2m[1] - 170, ext2m[3] - 420
    ax.annotate("N", xy=(xn, yn + 260), xytext=(xn, yn), fontsize=9,
                fontweight="bold", ha="center", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))

    # colorbars alinhadas as colunas dos mapas
    fig.canvas.draw()
    cb_y, cb_h = 0.085, 0.022
    specs = [
        (axes[3], ims["dens"], "Densidade de fendas (%) — (a–d)",
         dict(extend="max")),
        (axes[4], ims["pers"], "Persistência (anos c/ dens. > 0,5%)",
         dict(ticks=[1, 2, 3, 4])),
        (axes[5], ims["delta"], "Δ densidade 2016→2019 (p.p.)",
         dict(extend="both")),
    ]
    for ax_ref, im, lab, kw in specs:
        p = ax_ref.get_position()
        cax = fig.add_axes([p.x0 + 0.01, cb_y, p.width - 0.02, cb_h])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", **kw)
        cb.set_label(lab, fontsize=8)
        cb.ax.tick_params(labelsize=8)

    axes[4].legend(handles=[Patch(facecolor=GRAY, edgecolor="none",
                                  label="DEM < 50% (sem dado)")],
                   loc="upper left", fontsize=8, frameon=True, framealpha=0.85,
                   borderpad=0.3, handlelength=1.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  salvo: {OUT}")


if __name__ == "__main__":
    main()

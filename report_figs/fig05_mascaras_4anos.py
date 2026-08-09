"""Fig5 — Máscaras de fendas sobre o ortomosaico — 4 anos (2016–2019).

Grade 2×2 (um painel por ano, mesma extensão UTM) com as fendas segmentadas
(U-Net ep.107, TTA, thr 0,6, blend) em azul saturado opaco sobre o RGB.
Dois recortes fixos (Zoom 1 e Zoom 2), idênticos nos 4 anos, na parte
inferior, evidenciando persistência e fragmentação no detalhe de 8 cm.

Reprodutível: rodar a partir da raiz do repo (LACRIO_IC).
Saída: results/figures/relatorio/fig05_mascaras_4anos.png (300 dpi, ~19 cm).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
import numpy as np
import rasterio
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/figures/relatorio/fig05_mascaras_4anos.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

YEARS = [2016, 2017, 2018, 2019]
YEAR_COLORS = {2016: "#0072B2", 2017: "#E69F00", 2018: "#009E73",
               2019: "#D55E00"}                      # paleta fixa por ano
MASK_COLOR = "#2A6FDB"                               # predição (azul saturado)
PX = 0.08                                            # m por pixel
DEC = 16                                             # decimação da visão geral

# n de fendas e cobertura (results/crevasses_stats_per_year.csv)
STATS = {2016: (11551, 1.40), 2017: (13297, 1.22),
         2018: (14885, 1.48), 2019: (17500, 1.18)}

# Zooms fixos: centro em px nativos da grade 2016 (convertidos p/ UTM)
ZOOMS = [("1", 8960, 11040), ("2", 17600, 7680)]
ZHALF = 312                                          # 625 px -> 50 m de lado


def mosaic_path(year):
    return REPO / f"Schiaparelli_glacier/normalized_8cm/Schiaparelli_mosaic_{year}_8cm.tif"


def mask_path(year):
    return REPO / f"results/{year}/crevasses_mask_{year}_8cm.tif"


def pooled_mask(path, dec):
    """Max-pool da máscara binária (preserva fendas finas na decimação)."""
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        Hs, Ws = H // dec, W // dec
        out = np.zeros((Hs, Ws), bool)
        step = 256 * dec
        for r0 in range(0, Hs * dec, step):
            rows = min(step, Hs * dec - r0)
            a = src.read(1, window=((r0, r0 + rows), (0, Ws * dec)))
            out[r0 // dec:(r0 + rows) // dec] = (
                a.reshape(rows // dec, dec, Ws, dec).max(axis=(1, 3)) > 0)
    return out


# centros dos zooms em UTM (grade 2016)
with rasterio.open(mosaic_path(2016)) as s16:
    ZOOM_UTM = [s16.xy(cy, cx) for _, cx, cy in ZOOMS]

# ------------------------------------------------------------- figura --------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
FIGW = 7.48
fig = plt.figure(figsize=(FIGW, 9.55), dpi=300)
gs = fig.add_gridspec(2, 1, height_ratios=[5.6, 3.55], hspace=0.1,
                      left=0.075, right=0.985, top=0.99, bottom=0.035)
gs_top = gs[0].subgridspec(2, 2, hspace=0.035, wspace=0.035)
gs_bot = gs[1].subgridspec(2, 4, hspace=0.06, wspace=0.05)

mask_rgba_lut = np.zeros((2, 4), np.float32)
mask_rgba_lut[1] = matplotlib.colors.to_rgba(MASK_COLOR)

axes_year = {}
for i, yr in enumerate(YEARS):
    ax = fig.add_subplot(gs_top[i // 2, i % 2])
    axes_year[yr] = ax
    with rasterio.open(mosaic_path(yr)) as src:
        left, bottom, right, top = src.bounds
        rgba = src.read(out_shape=(4, src.height // DEC, src.width // DEC))
    rgb = np.moveaxis(rgba[:3], 0, -1).astype(np.uint8).copy()
    rgb[rgba[3] < 128] = 255
    pm = pooled_mask(mask_path(yr), DEC)
    ax.imshow(rgb, extent=[left, right, bottom, top], zorder=1,
              interpolation="bilinear")
    ax.imshow(mask_rgba_lut[pm.astype(int)], extent=[left, right, bottom, top],
              zorder=2, interpolation="nearest")
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    # rótulo do ano + n
    n, cov = STATS[yr]
    ax.text(0.015, 0.975, str(yr), transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color="w",
            bbox=dict(boxstyle="round,pad=0.25", fc=YEAR_COLORS[yr],
                      ec="none"))
    ax.text(0.015, 0.885, f"n = {n:,} fendas".replace(",", " ")
            + f"\ncobertura {cov:.2f}%".replace(".", ","),
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            color="k",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5",
                      lw=0.5, alpha=0.9))
    # caixas dos zooms (idênticas nos 4 painéis)
    for (znum, _, _), (ux, uy) in zip(ZOOMS, ZOOM_UTM):
        half_m = ZHALF * PX
        ax.add_patch(Rectangle((ux - half_m, uy - half_m), 2 * half_m,
                               2 * half_m, facecolor="none", edgecolor="k",
                               linewidth=1.0, zorder=5))
        ax.annotate(znum, (ux + half_m, uy + half_m), xytext=(2, 2),
                    textcoords="offset points", fontsize=7.5,
                    fontweight="bold", ha="left", va="bottom", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="k",
                              lw=0.5, alpha=0.9))
    # coordenadas apenas nas molduras externas
    ax.tick_params(labelsize=6.5)
    ax.xaxis.set_major_formatter(lambda v, p: f"{v/1000:.1f}")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v/1000:.1f}")
    if i % 2 == 1:
        ax.set_yticklabels([])
    if i // 2 == 0:
        ax.set_xticklabels([])
    if i == 2:
        ax.set_xlabel("Leste UTM 19S (×10³ m) — EPSG:32719", fontsize=7.5)
        ax.set_ylabel("Norte UTM 19S (×10³ m)", fontsize=7.5)

# scale bar compartilhada (painel 2019) + seta norte (painel 2017) ----------
ax = axes_year[2019]
with rasterio.open(mosaic_path(2019)) as src:
    left, bottom, right, top = src.bounds
sb_x0, sb_y = right - 1000 - 160, bottom + 150
ax.add_patch(Rectangle((sb_x0 - 70, sb_y - 60), 1000 + 160, 240,
                       facecolor="white", edgecolor="0.4", linewidth=0.5,
                       alpha=0.88, zorder=7))
ax.plot([sb_x0, sb_x0 + 1000], [sb_y, sb_y], color="k", lw=3.0,
        solid_capstyle="butt", zorder=8)
ax.plot([sb_x0 + 500, sb_x0 + 1000], [sb_y, sb_y], color="w", lw=1.8,
        solid_capstyle="butt", zorder=9)
for dx, t in [(0, "0"), (500, "0,5"), (1000, "1 km")]:
    ax.text(sb_x0 + dx, sb_y + 40, t, ha="center", va="bottom", fontsize=6,
            color="k", zorder=9)
ax2 = axes_year[2017]
na_x, na_y = right - 130, top - 320
ax2.annotate("N", (na_x, na_y + 165), ha="center", va="bottom", fontsize=8.5,
             fontweight="bold", zorder=8)
ax2.add_patch(FancyArrow(na_x, na_y - 60, 0, 160, width=16, head_width=58,
                         head_length=70, fc="k", ec="k", zorder=8))

# ------------------------------------------------------------- zooms ---------
for zi, ((znum, _, _), (ux, uy)) in enumerate(zip(ZOOMS, ZOOM_UTM)):
    for yi, yr in enumerate(YEARS):
        axz = fig.add_subplot(gs_bot[zi, yi])
        with rasterio.open(mosaic_path(yr)) as src:
            r, c = src.index(ux, uy)
            win = ((r - ZHALF, r + ZHALF), (c - ZHALF, c + ZHALF))
            img = np.moveaxis(src.read([1, 2, 3], window=win), 0, -1)
        with rasterio.open(mask_path(yr)) as ms:
            r, c = ms.index(ux, uy)
            m = ms.read(1, window=((r - ZHALF, r + ZHALF),
                                   (c - ZHALF, c + ZHALF))) > 0
        axz.imshow(img, interpolation="bilinear")
        axz.imshow(mask_rgba_lut[m.astype(int)], interpolation="nearest")
        axz.set_xticks([])
        axz.set_yticks([])
        for s in axz.spines.values():
            s.set_edgecolor(YEAR_COLORS[yr])
            s.set_linewidth(1.8)
        if zi == 0:
            axz.text(0.04, 0.96, str(yr), transform=axz.transAxes,
                     fontsize=7.5, fontweight="bold", va="top", ha="left",
                     color="w",
                     bbox=dict(boxstyle="round,pad=0.22",
                               fc=YEAR_COLORS[yr], ec="none"))
        if yi == 0:
            axz.set_ylabel(f"Zoom {znum}", fontsize=8, fontweight="bold")
        # scale bar própria (20 m = 250 px; janela de 50 m)
        bar_px = 20 / PX
        x0, y0 = img.shape[1] - bar_px - 30, img.shape[0] - 42
        axz.plot([x0, x0 + bar_px], [y0, y0], color="w", lw=2.4,
                 solid_capstyle="butt")
        axz.plot([x0, x0 + bar_px], [y0, y0], color="k", lw=1.0,
                 solid_capstyle="butt")
        axz.text(x0 + bar_px / 2, y0 - 18, "20 m", color="w", fontsize=5.5,
                 ha="center", va="bottom",
                 path_effects=[pe.withStroke(linewidth=1.4, foreground="k")])

fig.savefig(OUT, dpi=300)
print(f"salvo: {OUT}")

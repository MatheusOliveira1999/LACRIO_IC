"""Fig1 — Área de estudo e desenho da validação cruzada espacial.

Mapa do ortomosaico UAV 2016 (Glaciar Schiaparelli) com os 5 folds espaciais
do CV (blocos de 2048 px) coloridos; inset esquemático de localização; 5 zooms
(a–e) com morfologias de fendas, um por fold.

Reprodutível: rodar a partir da raiz do repo (LACRIO_IC).
Saída: results/figures/relatorio/fig01_area_estudo_cv.png (300 dpi, ~19 cm).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch, Rectangle, FancyArrow
import numpy as np
import rasterio
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MOSAIC = REPO / "Schiaparelli_glacier/normalized_8cm/Schiaparelli_mosaic_2016_8cm.tif"
OUT = REPO / "results/figures/relatorio/fig01_area_estudo_cv.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- folds ------
# Blocos 2048 px (bx, by) por fold — derivados deterministicamente do CV
# (10_crossval_curves.py, seed 42; ver scratchpad/derive_folds.py).
FOLD_BLOCKS = {
    1: [(1, 4), (1, 5), (2, 3), (2, 5), (2, 7), (3, 7), (4, 7), (4, 8), (5, 2),
        (5, 3), (6, 2), (6, 3), (6, 9), (7, 7), (8, 1), (9, 10), (10, 5),
        (10, 9), (11, 9), (12, 3), (12, 4), (13, 7)],
    2: [(0, 6), (1, 6), (1, 7), (2, 6), (2, 9), (3, 3), (4, 4), (4, 10), (5, 7),
        (5, 10), (6, 5), (6, 7), (7, 1), (7, 4), (7, 8), (8, 4), (8, 5), (8, 8),
        (10, 8), (11, 3), (12, 6), (13, 9)],
    3: [(0, 7), (3, 4), (3, 6), (5, 6), (6, 1), (6, 4), (6, 8), (6, 10), (7, 6),
        (7, 10), (9, 2), (9, 4), (9, 9), (10, 2), (10, 3), (11, 4), (11, 6),
        (11, 7), (11, 8), (12, 5), (13, 6), (13, 8)],
    4: [(3, 5), (3, 8), (4, 9), (5, 5), (5, 9), (7, 3), (7, 5), (8, 3), (8, 6),
        (8, 9), (9, 1), (9, 5), (9, 6), (10, 0), (10, 1), (10, 7), (11, 0),
        (11, 5), (12, 2), (12, 8), (12, 9)],
    5: [(1, 8), (2, 4), (2, 8), (3, 9), (3, 10), (4, 3), (4, 5), (4, 6), (5, 8),
        (6, 6), (8, 2), (8, 7), (8, 10), (9, 0), (9, 3), (9, 7), (10, 6),
        (11, 1), (12, 1), (12, 7), (14, 9)],
}
FOLD_NTILES = {1: 56, 2: 57, 3: 57, 4: 56, 5: 56}
# Paleta categórica validada (CVD-safe, all-pairs PASS)
FOLD_COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#882255",
               5: "#56B4E9"}
BLOCK_PX = 2048           # tamanho do bloco em px nativos (8 cm)
PX = 0.08                 # m por pixel

# ------------------------------------------------------------- zooms ---------
# (letra, cx, cy em px nativos 2016, descrição, fold)
ZOOMS = [
    ("a", 5000, 13000, "Fendas abertas c/ detritos\n(porção terminal) — fold 2", 2),
    ("b", 6900, 15000, "Zona fraturada sombreada\n(margem sul) — fold 1", 1),
    ("c", 8960, 11040, "Fraturas lineares paralelas\n(campo central) — fold 5", 5),
    ("d", 18400, 6400, "Fendas abertas em neve\n(porção superior) — fold 4", 4),
    ("e", 23500, 9200, "Fraturas finas incipientes\n(porção leste) — fold 3", 3),
]
ZHALF = 400               # meia-janela do zoom (px nativos) -> 64 m de lado

# ------------------------------------------------------------ leitura --------
DEC = 8                   # decimação do mosaico para o mapa principal
with rasterio.open(MOSAIC) as src:
    left, bottom, right, top = src.bounds
    H, W = src.height, src.width
    rgba = src.read(out_shape=(4, H // DEC, W // DEC))
    zoom_imgs = []
    for _, cx, cy, _, _ in ZOOMS:
        win = ((cy - ZHALF, cy + ZHALF), (cx - ZHALF, cx + ZHALF))
        zoom_imgs.append(np.moveaxis(src.read([1, 2, 3], window=win), 0, -1))

rgb = np.moveaxis(rgba[:3], 0, -1).astype(np.uint8).copy()
nodata = rgba[3] < 128
rgb[nodata] = 255         # fundo branco fora do footprint

def px2utm(x_px, y_px):
    return left + x_px * PX, top - y_px * PX

# ------------------------------------------------------------- figura --------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
FIGW = 7.48               # ~19 cm
fig = plt.figure(figsize=(FIGW, 8.3), dpi=300)
gs = fig.add_gridspec(2, 5, height_ratios=[5.45, 1.55], hspace=0.16,
                      wspace=0.06, left=0.085, right=0.98, top=0.985,
                      bottom=0.075)
ax = fig.add_subplot(gs[0, :])
ax.imshow(rgb, extent=[left, right, bottom, top], zorder=1,
          interpolation="bilinear")

# folds --------------------------------------------------------------------
for f, blocks in FOLD_BLOCKS.items():
    for bx, by in blocks:
        x0, y1 = px2utm(bx * BLOCK_PX, by * BLOCK_PX)          # canto sup-esq
        x1, y0 = px2utm(min((bx + 1) * BLOCK_PX, W), min((by + 1) * BLOCK_PX, H))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor=FOLD_COLORS[f], alpha=0.35,
                               edgecolor=FOLD_COLORS[f], linewidth=0.7,
                               zorder=3))

# caixas dos zooms + letras ------------------------------------------------
for letter, cx, cy, _, _ in ZOOMS:
    x0, y1 = px2utm(cx - ZHALF, cy - ZHALF)
    x1, y0 = px2utm(cx + ZHALF, cy + ZHALF)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor="k", linewidth=1.1, zorder=6))
    ax.annotate(letter, (x0, y1), xytext=(-1, 3), textcoords="offset points",
                fontsize=9, fontweight="bold", color="k", ha="right",
                va="bottom", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="k",
                          lw=0.6, alpha=0.9))

# eixos em UTM (×10³ m) ----------------------------------------------------
ax.set_xlim(left, right)
ax.set_ylim(bottom, top)
ax.set_aspect("equal")
ax.xaxis.set_major_formatter(lambda v, p: f"{v/1000:.1f}")
ax.yaxis.set_major_formatter(lambda v, p: f"{v/1000:.1f}")
ax.set_xlabel("Leste UTM 19S (×10³ m) — EPSG:32719", fontsize=8)
ax.set_ylabel("Norte UTM 19S (×10³ m)", fontsize=8)
ax.tick_params(labelsize=7)

# scale bar 1 km + seta norte ---------------------------------------------
sb_x0 = right - 1000 - 150
sb_y = bottom + 140
ax.add_patch(Rectangle((sb_x0 - 70, sb_y - 55), 1000 + 160, 190,
                       facecolor="white", edgecolor="0.4", linewidth=0.5,
                       alpha=0.88, zorder=7))
ax.plot([sb_x0, sb_x0 + 1000], [sb_y, sb_y], color="k", lw=3.0,
        solid_capstyle="butt", zorder=8)
ax.plot([sb_x0 + 250, sb_x0 + 500], [sb_y, sb_y], color="w", lw=1.8,
        solid_capstyle="butt", zorder=9)
ax.plot([sb_x0 + 750, sb_x0 + 1000], [sb_y, sb_y], color="w", lw=1.8,
        solid_capstyle="butt", zorder=9)
for dx, t in [(0, "0"), (500, "0,5"), (1000, "1 km")]:
    ax.text(sb_x0 + dx, sb_y + 32, t, ha="center", va="bottom", fontsize=6.5,
            color="k", zorder=9)
na_x, na_y = right - 90, top - 260
ax.annotate("N", (na_x, na_y + 145), ha="center", va="bottom", fontsize=9,
            fontweight="bold", zorder=8)
ax.add_patch(FancyArrow(na_x, na_y - 60, 0, 150, width=14, head_width=52,
                        head_length=60, fc="k", ec="k", zorder=8))

# legenda dos folds (canto inferior esquerdo, sobre o lago) ----------------
handles = [mpatches.Patch(facecolor=FOLD_COLORS[f], alpha=0.55,
                          edgecolor=FOLD_COLORS[f],
                          label=f"Fold {f} ({FOLD_NTILES[f]} tiles)")
           for f in range(1, 6)]
leg = ax.legend(handles=handles, loc="lower left", fontsize=7,
                title="Folds espaciais do CV (blocos de 164 m)",
                title_fontsize=7.5, framealpha=0.92, borderpad=0.5,
                bbox_to_anchor=(0.005, 0.005))
leg.set_zorder(10)

# inset de localização (esquemático, canto superior esquerdo) --------------
axi = ax.inset_axes([0.012, 0.66, 0.24, 0.325])
axi.set_facecolor("#f2f0eb")
axi.set_xlim(-75, -65)
axi.set_ylim(-56.5, -51.5)
axi.grid(color="#c9c5bd", lw=0.5)
axi.plot(-70.9, -54.4, marker="s", ms=6, mfc="#D55E00", mec="k", mew=0.6)
axi.annotate("Glaciar Schiaparelli\nCordilheira Darwin, Chile\n(~54,4°S, 70,9°W)",
             (-70.9, -54.4), xytext=(0, 8), textcoords="offset points",
             ha="center", va="bottom", fontsize=6.2)
axi.set_xticks([-74, -70, -66])
axi.set_yticks([-56, -54, -52])
axi.set_xticklabels(["74°W", "70°W", "66°W"], fontsize=5.5)
axi.set_yticklabels(["56°S", "54°S", "52°S"], fontsize=5.5)
axi.tick_params(length=2, pad=1)
for s in axi.spines.values():
    s.set_linewidth(0.6)

# ------------------------------------------------------------- zooms ---------
for i, ((letter, cx, cy, desc, foldn), img) in enumerate(zip(ZOOMS, zoom_imgs)):
    axz = fig.add_subplot(gs[1, i])
    axz.imshow(img, interpolation="bilinear")
    axz.set_xticks([])
    axz.set_yticks([])
    for s in axz.spines.values():
        s.set_edgecolor(FOLD_COLORS[foldn])
        s.set_linewidth(2.2)
    axz.text(0.03, 0.97, letter, transform=axz.transAxes, fontsize=9,
             fontweight="bold", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="k", lw=0.6,
                       alpha=0.9))
    axz.text(0.5, -0.05, desc, transform=axz.transAxes, fontsize=5.8,
             ha="center", va="top", linespacing=1.25)
    # scale bar 20 m = 250 px do zoom (janela de 800 px = 64 m)
    bar_px = 20 / PX
    x0 = img.shape[1] - bar_px - 40
    y0 = img.shape[0] - 55
    axz.plot([x0, x0 + bar_px], [y0, y0], color="w", lw=2.6,
             solid_capstyle="butt")
    axz.plot([x0, x0 + bar_px], [y0, y0], color="k", lw=1.2,
             solid_capstyle="butt")
    axz.text(x0 + bar_px / 2, y0 - 22, "20 m", color="w", fontsize=6,
             ha="center", va="bottom",
             path_effects=[pe.withStroke(linewidth=1.5, foreground="k")])
    # linha de conexão zoom -> caixa no mapa
    ux, uy = px2utm(cx, cy + ZHALF)
    con = ConnectionPatch(xyA=(0.5, 1.0), coordsA=axz.transAxes,
                          xyB=(ux, uy), coordsB=ax.transData,
                          color="0.45", lw=0.5, alpha=0.75, zorder=2)
    fig.add_artist(con)

fig.savefig(OUT, dpi=300)
print(f"salvo: {OUT}")

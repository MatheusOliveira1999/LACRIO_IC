#!/usr/bin/env python
"""Fig. 4 — Galeria de falsos positivos tipicos.

Grade 2x3 de recortes do ortomosaico 2016 (tiles de validacao) em que o
modelo prediz fendas em areas sem fendas reais (GT ausente ou quase nula).
Predicao em azul #2A6FDB (mesma cor da Fig. 3). Tiles escolhidos por
inspecao de results/unet_validation_crevasses_2016.json (fp alto, gt baixo).

Uso: python report_figs/fig04_falsos_positivos.py  (cwd = raiz do repo)
Saida: results/figures/relatorio/fig04_falsos_positivos.png (300 dpi, ~19 cm)
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from scipy import ndimage as ndi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results/figures/relatorio/fig04_falsos_positivos.png")

PR_COLOR = (42, 111, 219)   # #2A6FDB
PX_M = 0.08

# (tile, titulo curto da causa, lado do recorte em px; 384 px = 30,7 m)
PANELS = [
    ("tile_001191", "Sulcos e lineações na neve", 384),
    ("tile_001641", "Depressões alongadas\n(fendas fechadas sob neve)", 384),
    ("tile_000476", "Canal/depressão\nsupraglacial sombreado", 384),
    ("tile_000338", "Sombra de crista de neve", 384),
    ("tile_000623", "Morena / rocha exposta", 256),
    ("tile_000415", "Margem rochosa sombreada", 256),
]


def load(tile):
    rgb = np.array(Image.open(f"{REPO}/tiles/2016/{tile}.png").convert("RGB"))
    gtp = f"{REPO}/masks/2016/annotations/crevasses/{tile}_crevasses.png"
    prp = f"{REPO}/masks/2016/crevasses/{tile}_crevasses.png"
    gt = (np.array(Image.open(gtp).convert("L")) > 127) if os.path.exists(gtp) \
        else np.zeros(rgb.shape[:2], bool)
    pr = (np.array(Image.open(prp).convert("L")) > 127) if os.path.exists(prp) \
        else np.zeros(rgb.shape[:2], bool)
    return rgb, gt, pr


def crop_on_fp(rgb, gt, pr, size):
    """Recorte centrado no maior aglomerado de FP (pred longe do GT).

    Retorna o recorte RGB e apenas os pixels FALSO-POSITIVOS (predicao a
    mais de ~0,6 m de qualquer pixel do GT), para nao pintar de azul
    acertos verdadeiros dentro do recorte.
    """
    fp = pr & ~ndi.binary_dilation(gt, iterations=8)
    if fp.sum() == 0:
        fp = pr
    lab, n = ndi.label(ndi.binary_dilation(fp, iterations=6))
    sizes = ndi.sum(fp, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == k)
    cy, cx = int(ys.mean()), int(xs.mean())
    H, W = pr.shape
    y0 = min(max(cy - size // 2, 0), H - size)
    x0 = min(max(cx - size // 2, 0), W - size)
    sl = np.s_[y0:y0 + size, x0:x0 + size]
    return rgb[sl], fp[sl]


def scalebar(ax, H, meters=10, px_m=PX_M):
    L = meters / px_m
    x0, y0 = H * 0.036, H - H * 0.052
    ax.plot([x0, x0 + L], [y0, y0], color="white", lw=3, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=5, foreground="black")])
    ax.text(x0 + L / 2, y0 - H * 0.02, f"{meters} m", color="white", fontsize=8,
            ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(7.48, 5.55))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.905, bottom=0.055,
                        wspace=0.025, hspace=0.24)

    letters = "abcdef"
    for i, (tile, causa, size) in enumerate(PANELS):
        ax = axes.flat[i]
        rgb, gt, pr = load(tile)
        img, fpc = crop_on_fp(rgb, gt, pr, size)
        over = img.copy()
        over[fpc] = PR_COLOR
        ax.imshow(over)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"({letters[i]}) {causa}", fontsize=9, pad=3)
        scalebar(ax, size)

    handles = [mpatches.Patch(color="#2A6FDB",
                              label="Predição U-Net (thr 0,6) — falso positivo "
                                    "(sem fenda na referência manual)")]
    fig.legend(handles=handles, loc="lower center", fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print("salvo:", OUT)


if __name__ == "__main__":
    main()

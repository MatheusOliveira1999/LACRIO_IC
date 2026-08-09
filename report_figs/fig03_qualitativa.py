#!/usr/bin/env python
"""Fig. 3 — Avaliacao qualitativa: ortofoto | GT | predicao | sobreposicao.

Grade 4 colunas x 5 linhas com cenas de teste representativas de 2016
(validacao per-tile: results/unet_validation_crevasses_2016.json).
Convencao de cores fixa do relatorio: GT = turquesa #00C2A8,
predicao = azul #2A6FDB (uma cor por fonte; nunca esquema TP/FP/FN).

Uso: python report_figs/fig03_qualitativa.py  (cwd = raiz do repo)
Saida: results/figures/relatorio/fig03_qualitativa.png (300 dpi, ~19 cm)
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results/figures/relatorio/fig03_qualitativa.png")

GT_COLOR = (0, 194, 168)      # #00C2A8 turquesa
PR_COLOR = (42, 111, 219)     # #2A6FDB azul
PX_M = 0.08                   # 8 cm/pixel -> tile 512 px = 41 m

# (tile, rotulo curto da cena, metrica anotada)
SCENES = [
    ("tile_001343", "fendas largas\ngelo c/ detritos", "F1 = 0,91"),
    ("tile_001591", "fraturas finas\ngelo limpo",      "F1 = 0,90"),
    ("tile_000927", "campo denso\nde fraturas",        "F1 = 0,70"),
    ("tile_000646", "fendas sob neve\n(FN difícil)",   "F1 = 0,47 | R = 0,33"),
    ("tile_002073", "seracs e sombra",                 "F1 = 0,50 | R = 0,35"),
]

# elipses tracejadas (linha_idx, col_idx=3 combinado, cx, cy, w, h, cor, nota)
# coordenadas em pixels do tile 512x512
ANNOT = [
    # FP sistematico no campo denso (linha c): sulco sem GT (cluster medido)
    (2, 165, 52, 150, 85, "yellow"),
    # lacunas (FN) nas cenas de neve/sombra (clusters medidos)
    (3, 290, 300, 280, 260, "white"),
    (4, 225, 180, 190, 165, "white"),
    (4, 400, 75, 130, 110, "white"),
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


def paint(rgb, mask, color):
    out = rgb.copy()
    out[mask] = color
    return out


def scalebar(ax, meters=20, px_m=PX_M, H=512):
    L = meters / px_m
    x0, y0 = 18, H - 26
    ax.plot([x0, x0 + L], [y0, y0], color="white", lw=3, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=5, foreground="black")])
    ax.text(x0 + L / 2, y0 - 10, f"{meters} m", color="white", fontsize=8,
            ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])


def main():
    n = len(SCENES)
    fig, axes = plt.subplots(n, 4, figsize=(7.48, 9.55))
    plt.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.035,
                        wspace=0.02, hspace=0.02)

    col_titles = ["Ortofoto", "Referência (GT)", "Predição U-Net",
                  "Sobreposição"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9.5, pad=4)

    letters = "abcde"
    for i, (tile, desc, metric) in enumerate(SCENES):
        rgb, gt, pr = load(tile)
        panels = [rgb,
                  paint(rgb, gt, GT_COLOR),
                  paint(rgb, pr, PR_COLOR),
                  paint(paint(rgb, gt, GT_COLOR), pr, PR_COLOR)]
        for j, img in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        scalebar(axes[i, 0])
        # metrica anotada no painel combinado
        axes[i, 3].text(0.97, 0.035, metric, transform=axes[i, 3].transAxes,
                        fontsize=8, color="white", ha="right", va="bottom",
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="black")])
        # rotulo de linha a esquerda
        axes[i, 0].set_ylabel(f"({letters[i]}) {desc}", fontsize=8.5,
                              rotation=90, labelpad=6, va="center")

    for (row, cx, cy, w, h, color) in ANNOT:
        ax = axes[row, 3]
        ell = mpatches.Ellipse((cx, cy), w, h, fill=False, ls=(0, (4, 3)),
                               lw=1.4, edgecolor=color,
                               path_effects=[pe.withStroke(linewidth=2.6,
                                                           foreground="black")])
        ax.add_patch(ell)

    # legenda de cores no rodape via handles
    handles = [mpatches.Patch(color="#00C2A8", label="Referência manual (GT)"),
               mpatches.Patch(color="#2A6FDB", label="Predição U-Net (thr 0,6)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.535, 0.0))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print("salvo:", OUT)


if __name__ == "__main__":
    main()

"""Figura 2 do relatório: divisão espacial da validação cruzada (gabarito final, set/2026).

Substitui o mapa QGIS `mapa_folds_cv_qgis.png`, que (a) tinha a legenda cortada no
fold 5 e (b) pintava cada bloco de 2 048 px inteiro, inclusive a parte fora do
ortomosaico. Aqui só aparecem os 218 recortes anotados, na cor do grupo de teste
(cheio = com fenda, vazado = sem fenda); a cor de fundo dos blocos foi retirada
porque dava a impressão de que a área inteira tinha fenda.

Fontes:
- grupos: results/oof/crevasses_fp0.7_fn0.3_lab2/fold_of_tile.json (CV de referência, semente 42)
- posição: tiles/2016/tiles_index.json (x, y em px do mosaico nativo de 2016)
- com/sem fenda: masks/2016/annotations/crevasses/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parent.parent
MOSAICO = ROOT / "Schiaparelli_glacier/Schiaparelli_mosaic_2016.tif"
FOLDS = ROOT / "results/oof/crevasses_fp0.7_fn0.3_lab2/fold_of_tile.json"
INDEX = ROOT / "tiles/2016/tiles_index.json"
MASKS = ROOT / "masks/2016/annotations/crevasses"
TILES = ROOT / "tiles/2016"
OUTLINE = ROOT / "results/qgis/glacier_outline.geojson"
SAIDA = ROOT / "results/figures/relatorio/mapa_folds_cv.png"

BLOCO = 2048          # px, igual ao --block-size do 10_crossval_curves.py
TILE = 512
DEC = 12              # fator de redução do mosaico para o fundo
# Okabe-Ito, mesma ordem do mapa anterior (validado para daltonismo)
CORES = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#D55E00", 5: "#CC79A7"}


def main():
    fold_of = {k: int(v) for k, v in json.load(open(FOLDS)).items()}
    idx = {t["filename"]: t for t in json.load(open(INDEX))["tiles"]}
    com_fenda = {t: bool((np.array(Image.open(MASKS / t.replace(".png", "_crevasses.png"))) > 0).any())
                 for t in fold_of}

    with rasterio.open(MOSAICO) as src:
        h, w = src.height // DEC, src.width // DEC
        img = src.read(out_shape=(src.count, h, w), resampling=Resampling.average)
        T = src.transform
    res = T.a * DEC
    rgb = np.moveaxis(img[:3], 0, -1).astype(float) / 255
    valido = img[3] > 0 if img.shape[0] == 4 else rgb.sum(-1) > 0

    # recorte da área útil (imagem válida) com margem
    ys, xs = np.where(valido)
    pad = 20
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)

    def px2m(x, y):  # px do mosaico nativo -> metros UTM
        return T.c + x * T.a, T.f + y * T.e

    x0m, y0m = px2m(c0 * DEC, r0 * DEC)
    x1m, y1m = px2m(c1 * DEC, r1 * DEC)
    ext = (x0m, x1m, y1m, y0m)

    # blocos de ~160 m (só para a contagem no subtítulo)
    blocos = {}
    for t, f in fold_of.items():
        blocos[(int(idx[t]["x"] // BLOCO), int(idx[t]["y"] // BLOCO))] = f
    fundo = np.where(valido[..., None], rgb, 1.0)

    fig = plt.figure(figsize=(11.69, 8.27), dpi=300)
    ax = fig.add_axes([0.03, 0.10, 0.68, 0.78])
    ax.imshow(fundo[r0:r1, c0:c1], extent=ext, interpolation="bilinear")

    # contorno do glaciar
    try:
        g = json.load(open(OUTLINE))
        for feat in g["features"]:
            geom = feat["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                for anel in poly:
                    a = np.array(anel)
                    ax.plot(a[:, 0], a[:, 1], color="black", lw=1.0)
    except FileNotFoundError:
        pass

    # recortes anotados
    lado = TILE * T.a
    for t, f in fold_of.items():
        xm, ym = px2m(idx[t]["x"], idx[t]["y"])
        cheio = com_fenda[t]
        if not cheio:
            ax.add_patch(Rectangle((xm, ym - lado), lado, lado, fc="none", ec="white",
                                   lw=2.6, alpha=0.9, zorder=3))
        ax.add_patch(Rectangle((xm, ym - lado), lado, lado,
                               fc=CORES[f] if cheio else "none", ec="white" if cheio else CORES[f],
                               lw=0.5 if cheio else 1.4, alpha=1.0, zorder=3))

    # exemplos: um recorte com fenda por grupo (mediana da fração de fenda)
    exemplos = {}
    for f in range(1, 6):
        cand = []
        for t, ff in fold_of.items():
            if ff == f and com_fenda[t]:
                m = np.array(Image.open(MASKS / t.replace(".png", "_crevasses.png"))) > 0
                cand.append((m.mean(), t))
        cand.sort()
        exemplos[f] = cand[int(len(cand) * 0.7)][1]
    for f, t in exemplos.items():
        xm, ym = px2m(idx[t]["x"], idx[t]["y"])
        ax.annotate(str(f), (xm + lado / 2, ym - lado / 2), xytext=(xm + lado / 2 + 70, ym + 70),
                    fontsize=9, fontweight="bold", color="black", zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=CORES[f], lw=1.5),
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8))

    ax.set_xlim(x0m, x1m)
    ax.set_ylim(y1m, y0m)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)

    # escala e norte
    L = 500
    xs0 = x0m + 0.06 * (x1m - x0m); ys0 = y1m + 0.05 * (y0m - y1m)
    ax.add_patch(Rectangle((xs0, ys0), L / 2, 12, fc="black", ec="black", zorder=6))
    ax.add_patch(Rectangle((xs0 + L / 2, ys0), L / 2, 12, fc="white", ec="black", zorder=6))
    for d, lab in [(0, "0"), (L / 2, "250"), (L, "500 m")]:
        ax.text(xs0 + d, ys0 - 18, lab, ha="center", va="top", fontsize=8, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.86), xycoords="axes fraction",
                ha="center", fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

    fig.text(0.03, 0.955, "Divisão do glaciar para o teste do modelo (validação cruzada espacial)",
             fontsize=15, fontweight="bold")
    n_pos = sum(com_fenda.values())
    fig.text(0.03, 0.895,
             f"{len(fold_of)} recortes anotados de 2016: {n_pos} com fenda (quadrado cheio) e "
             f"{len(fold_of) - n_pos} sem fenda (vazado). A cor indica o grupo de teste.\n"
             f"Cada grupo reúne blocos de ~160 m ({len(blocos)} no total) e é testado por um modelo treinado sem ele. "
             "Ortomosaico de 2016 · UTM 19S",
             fontsize=8.5, color="#333333")

    # legenda
    cont = {f: (sum(1 for t in fold_of if fold_of[t] == f),
                sum(1 for t in fold_of if fold_of[t] == f and com_fenda[t])) for f in CORES}
    hand = [Patch(fc="#555555", ec="white", label="recorte com fenda (cheio)"),
            Patch(fc="none", ec="#555555", lw=1.4, label="recorte sem fenda, conferido (vazado)"),
            Line2D([], [], color="black", lw=1.0, label="limite do glaciar")]
    hand += [Patch(fc=CORES[f], ec=CORES[f],
                   label=f"Grupo {f}  ({cont[f][0]} recortes, {cont[f][1]} com fenda)") for f in CORES]
    fig.legend(handles=hand, loc="lower left", bbox_to_anchor=(0.03, 0.005), ncol=3,
               fontsize=8.5, frameon=False, handlelength=1.6, columnspacing=1.5)

    # painéis de exemplo
    x_p, larg = 0.735, 0.14
    alt = 0.78 / 5
    for i, f in enumerate(range(1, 6)):
        t = exemplos[f]
        y = 0.88 - (i + 1) * alt + 0.008
        axp = fig.add_axes([x_p, y, larg, alt - 0.016])
        im = np.array(Image.open(TILES / t).convert("RGB")).astype(float) / 255
        m = np.array(Image.open(MASKS / t.replace(".png", "_crevasses.png"))) > 0
        im[m] = 0.45 * im[m] + 0.55 * np.array([0.9, 0.05, 0.05])
        axp.imshow(im)
        axp.set_xticks([]); axp.set_yticks([])
        for s in axp.spines.values():
            s.set_edgecolor(CORES[f]); s.set_linewidth(2.5)
        fig.text(x_p + larg + 0.008, y + (alt - 0.016) / 2,
                 f"Exemplo do\ngrupo {f}\n({t.replace('.png', '')})",
                 fontsize=8, va="center", color="#222222")
    fig.text(x_p, 0.885, "Exemplos (gabarito em vermelho, ~41 × 41 m)", fontsize=8.5, color="#333333")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=300, facecolor="white")
    print(SAIDA, {f: cont[f] for f in CORES}, "blocos", len(blocos))


if __name__ == "__main__":
    main()

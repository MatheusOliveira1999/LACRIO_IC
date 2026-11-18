"""Figura 13 do relatório: densidade de fendas em blocos de 100 m, persistência e variação.

Substitui o mapa QGIS `mapa_densidade_qgis.png`, que (a) deixava transparentes os
blocos com persistência 0 (o 0 estava como "sem dado"), sem essa classe na legenda,
e (b) pintava de rosa o quadro inteiro nos painéis de densidade, inclusive fora do
glaciar. Aqui só os blocos do glaciar (≥ 50% da área dentro do limite) são pintados
e a persistência tem a classe "0 anos".

Fontes (mesmos rasters do projeto QGIS, results/qgis/):
- dens100_{ano}.tif  densidade de fendas por bloco de 100 m (%)
- persistencia.tif   nº de anos (0–4) com densidade > 0,5%
- delta_dens.tif     densidade 2019 − 2016 (pontos percentuais)
- hillshade_2016.tif relevo sombreado de fundo
- glacier_outline.geojson
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from rasterio.enums import Resampling
from shapely.geometry import box, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "results/qgis"
SAIDA = ROOT / "results/figures/relatorio/mapa_densidade.png"
ANOS = [2016, 2017, 2018, 2019]
DENS_MAX = 10.0      # %
DELTA_MAX = 6.0      # pontos percentuais
FRAC_MIN = 0.5       # fração mínima do bloco dentro do glaciar
COR_ZERO = "#d9d9d9"


def ler(nome):
    with rasterio.open(Q / f"{nome}.tif") as r:
        return r.read(1).astype(float), r.transform, r.shape


def main():
    dens = {a: ler(f"dens100_{a}")[0] for a in ANOS}
    pers, T, (nr, nc) = ler("persistencia")
    delta = ler("delta_dens")[0]

    geleira = unary_union([shape(f["geometry"])
                           for f in json.load(open(Q / "glacier_outline.geojson"))["features"]])
    frac = np.zeros((nr, nc))
    for i in range(nr):
        for j in range(nc):
            x0, y0 = T * (j, i)
            x1, y1 = T * (j + 1, i + 1)
            b = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            frac[i, j] = b.intersection(geleira).area / b.area
    no_gelo = frac >= FRAC_MIN

    with rasterio.open(Q / "hillshade_2016.tif") as r:
        f = 6
        hs = r.read(1, out_shape=(r.height // f, r.width // f), resampling=Resampling.average).astype(float)
        hs[hs == 0] = np.nan
        hb = r.bounds
    ext_hs = (hb.left, hb.right, hb.bottom, hb.top)
    x0, y0 = T * (0, 0)
    x1, y1 = T * (nc, nr)
    ext_b = (x0, x1, y1, y0)

    def mascara(a):
        return np.where(no_gelo, a, np.nan)

    fig = plt.figure(figsize=(11.69, 8.27), dpi=300)
    fig.text(0.02, 0.965, "Densidade de fendas em blocos de 100 m, persistência e variação 2016→2019",
             fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, left=0.02, right=0.98, top=0.93, bottom=0.21, wspace=0.04, hspace=0.12)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    contornos = [np.array(anel) for g in getattr(geleira, "geoms", [geleira])
                 for anel in [g.exterior.coords, *[i.coords for i in g.interiors]]]

    def base(ax, titulo):
        ax.imshow(hs, cmap="gray", extent=ext_hs, vmin=0, vmax=255, alpha=0.7, interpolation="bilinear")
        for a in contornos:
            ax.plot(a[:, 0], a[:, 1], color="black", lw=0.9, zorder=4)
        ax.set_xlim(hb.left, hb.right)
        ax.set_ylim(hb.bottom, hb.top)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titulo, loc="left", fontsize=10, fontweight="bold", pad=3)

    cmap_d = plt.get_cmap("Reds")
    for k, a in enumerate(ANOS):
        base(axes[k], f"({'abcd'[k]}) Densidade {a}")
        im_d = axes[k].imshow(mascara(dens[a]), cmap=cmap_d, vmin=0, vmax=DENS_MAX, extent=ext_b,
                              interpolation="nearest", alpha=0.9, zorder=2)

    # (e) persistência: 0 em cinza, 1–4 em roxo (uma cor, claro -> escuro)
    roxos = plt.get_cmap("Purples")(np.linspace(0.45, 0.95, 4))
    cmap_p = ListedColormap([COR_ZERO, *roxos])
    norm_p = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5)
    base(axes[4], "(e) Persistência: anos com fenda")
    axes[4].imshow(mascara(pers), cmap=cmap_p, norm=norm_p, extent=ext_b,
                   interpolation="nearest", alpha=0.95, zorder=2)

    for i, j in zip(*np.where(no_gelo & (pers == 0))):
        xa, ya = T * (j, i + 1)
        axes[4].add_patch(Rectangle((xa, ya), T.a, -T.e, fc="none", ec="#8c8c8c", hatch="////",
                                    lw=0, zorder=3))
    base(axes[5], "(f) Variação da densidade 2016→2019")
    im_v = axes[5].imshow(mascara(delta), cmap="RdBu_r", vmin=-DELTA_MAX, vmax=DELTA_MAX, extent=ext_b,
                          interpolation="nearest", alpha=0.9, zorder=2)

    # escala no painel (d)
    ax = axes[3]
    L = 500
    xs = hb.left + 0.06 * (hb.right - hb.left); ys = hb.bottom + 0.07 * (hb.top - hb.bottom)
    ax.add_patch(Rectangle((xs, ys), L / 2, 20, fc="black", ec="black", zorder=6))
    ax.add_patch(Rectangle((xs + L / 2, ys), L / 2, 20, fc="white", ec="black", zorder=6))
    for d, lab in [(0, "0"), (L, "500 m")]:
        ax.text(xs + d, ys + 45, lab, ha="center", va="bottom", fontsize=7.5, zorder=6)

    # legendas
    cax = fig.add_axes([0.05, 0.11, 0.26, 0.022])
    cb = fig.colorbar(im_d, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Densidade de fendas no bloco (% da área), painéis (a)–(d)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    n_cls = {c: int(np.sum(no_gelo & (pers == c))) for c in range(5)}
    rot = {0: "0 anos: pouca ou nenhuma fenda\n(< 0,5% da área em todos os anos)",
           1: "1 ano", 2: "2 anos", 3: "3 anos", 4: "4 anos (fenda em todos os anos)"}
    hand = [Patch(fc=cmap_p(c), ec="#8c8c8c" if c == 0 else "#555555", lw=0.4, hatch="////" if c == 0 else None,
                  label=f"{rot[c]}  [{n_cls[c]} blocos]") for c in range(5)]
    fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.37, 0.175), fontsize=8,
               frameon=False, title="Persistência, painel (e)", title_fontsize=8.5,
               alignment="left", handlelength=1.4)

    cax = fig.add_axes([0.70, 0.11, 0.26, 0.022])
    cb = fig.colorbar(im_v, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Variação da densidade 2016→2019 (pontos percentuais), painel (f)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.70, 0.142, "← menos fendas em 2019", fontsize=8, color="#333333")
    fig.text(0.96, 0.142, "mais fendas em 2019 →", fontsize=8, color="#333333", ha="right")

    fig.text(0.02, 0.015,
             f"Só aparecem os {int(no_gelo.sum())} blocos com pelo menos metade da área dentro do limite do glaciar "
             "(linha preta). Fundo: relevo sombreado do MDE de 2016. UTM 19S.",
             fontsize=8, color="#333333")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=300, facecolor="white")
    print(SAIDA, "blocos no gelo:", int(no_gelo.sum()), "classes:", n_cls)


if __name__ == "__main__":
    main()

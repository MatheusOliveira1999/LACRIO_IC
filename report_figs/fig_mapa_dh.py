"""Figura 12 do relatório: quanto a superfície baixou (dH) em cada par de anos.

Substitui o mapa QGIS `mapa_dh_qgis.png`, cuja legenda de cores era um quadradinho
com só "6" e "-12". Aqui: barra de cores horizontal longa (como na Figura 13),
escala simétrica ±12 m com zero em branco, textos sem jargão.

Fontes (results/qgis/): dh_{par}.tif (grade de 2 m, só gelo), hillshade_2016.tif,
contours_50m.gpkg, glacier_outline.geojson.
"""
import json
from pathlib import Path

import fiona
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "results/qgis"
SAIDA = ROOT / "results/figures/relatorio/mapa_dh.png"
PARES = [(2016, 2017), (2017, 2018), (2018, 2019)]
LIM = 12.0          # escala simétrica: mesmo espaçamento para perda e ganho


def main():
    with rasterio.open(Q / "hillshade_2016.tif") as r:
        f = 6
        hs = r.read(1, out_shape=(r.height // f, r.width // f), resampling=Resampling.average).astype(float)
        hs[hs == 0] = np.nan
        hb = r.bounds
    ext_hs = (hb.left, hb.right, hb.bottom, hb.top)

    contorno = json.load(open(Q / "glacier_outline.geojson"))
    aneis = []
    for feat in contorno["features"]:
        g = feat["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            aneis += [np.array(a) for a in poly]

    curvas = {}
    with fiona.open(Q / "contours_50m.gpkg") as src:
        for feat in src:
            z = feat["properties"]["ELEV"]
            if z < 50:
                continue
            g = feat["geometry"]
            linhas = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
            curvas.setdefault(z, []).extend(np.array(l) for l in linhas)

    fig = plt.figure(figsize=(11.69, 5.3), dpi=300)
    fig.text(0.015, 0.955, "Quanto a superfície do glaciar baixou a cada ano (diferença entre mapas de altura)",
             fontsize=14, fontweight="bold")
    fig.text(0.015, 0.905, "Vermelho = perdeu altura (derretimento); azul = ganhou altura. "
             "Linhas azul-acinzentadas: curvas de nível a cada 50 m.", fontsize=9, color="#333333")
    gs = fig.add_gridspec(1, 3, left=0.015, right=0.985, top=0.84, bottom=0.28, wspace=0.03)

    for k, (a, b) in enumerate(PARES):
        ax = fig.add_subplot(gs[k])
        with rasterio.open(Q / f"dh_{a}_{b}.tif") as r:
            dh = r.read(1).astype(float)
            dh[dh == r.nodata] = np.nan
            rb = r.bounds
        med = np.nanmedian(dh)
        ax.imshow(hs, cmap="gray", extent=ext_hs, vmin=0, vmax=255, alpha=0.8, interpolation="bilinear")
        im = ax.imshow(dh, cmap="RdBu", vmin=-LIM, vmax=LIM, extent=(rb.left, rb.right, rb.bottom, rb.top),
                       interpolation="nearest", zorder=2)
        for z, ls in curvas.items():
            for l in ls:
                ax.plot(l[:, 0], l[:, 1], color="#5b7897", lw=0.5, zorder=3)
            maior = max(ls, key=len)
            if len(maior) > 20:
                x, y = maior[len(maior) // 2]
                if hb.left < x < hb.right and hb.bottom < y < hb.top:
                    ax.text(x, y, f"{z:.0f} m", fontsize=6, color="#2f4a66", ha="center", va="center",
                            zorder=5, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.3))
        for anel in aneis:
            ax.plot(anel[:, 0], anel[:, 1], color="black", lw=0.9, zorder=4)
        ax.set_xlim(hb.left, hb.right)
        ax.set_ylim(hb.bottom, hb.top)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({'abc'[k]}) {a}→{b}: baixou {abs(med):.2f} m (mediana)".replace(".", ","),
                     loc="left", fontsize=10, fontweight="bold", pad=3)
        if k == 0:
            L = 500
            xs = hb.left + 0.05 * (hb.right - hb.left); ys = hb.bottom + 0.05 * (hb.top - hb.bottom)
            ax.add_patch(Rectangle((xs, ys), L / 2, 22, fc="black", ec="black", zorder=6))
            ax.add_patch(Rectangle((xs + L / 2, ys), L / 2, 22, fc="white", ec="black", zorder=6))
            for d, lab in [(0, "0"), (L, "500 m")]:
                ax.text(xs + d, ys + 45, lab, ha="center", va="bottom", fontsize=7.5, zorder=6)

    cax = fig.add_axes([0.20, 0.155, 0.60, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both",
                      ticks=np.arange(-12, 13, 2))
    cb.set_label("Mudança da altura da superfície entre os dois anos (m)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    fig.text(0.20, 0.205, "← perdeu altura (derreteu)", fontsize=8.5, color="#333333")
    fig.text(0.80, 0.205, "ganhou altura →", fontsize=8.5, color="#333333", ha="right")
    fig.text(0.015, 0.012,
             "Cinza (relevo sombreado) = fora da área de gelo analisada ou sem mapa de altura válido. "
             "Em rocha, que não deveria mudar,\na diferença foi de no máximo 4 cm, o que confirma que os mapas estão "
             "bem alinhados. Grade de 2 m. UTM 19S.", fontsize=7.5, color="#333333")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=300, facecolor="white")
    print(SAIDA)


if __name__ == "__main__":
    main()

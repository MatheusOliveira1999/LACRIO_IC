# -*- coding: utf-8 -*-
"""
Fig. 11 — Orientação das fendas: rosetas axiais por ano (small multiples).
Dados axiais (0–180°) espelhados a 180°, bins de 10°, escala de área igual
(raio proporcional a sqrt da frequência), norte no topo. Mediana axial marcada
como linha radial; mediana + MAD + n anotados sob cada painel.

Dados: results/crevasse_attrs_all.csv (azimuth_deg, por contagem de fendas).
Saída: results/figures/relatorio/fig11_rosetas_orientacao.png (300 dpi, ~19 cm).
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "figures", "relatorio")
os.makedirs(OUT, exist_ok=True)

YEAR_COLORS = {
    "2016": "#0072B2",
    "2017": "#E69F00",
    "2018": "#009E73",
    "2019": "#D55E00",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "text.color": "#222222",
})

# ------------------------------------------------------------------- dados
az = {y: [] for y in YEAR_COLORS}
with open(os.path.join(REPO, "results", "crevasse_attrs_all.csv")) as f:
    for r in csv.DictReader(f):
        if r["azimuth_deg"]:
            az[r["year"]].append(float(r["azimuth_deg"]) % 180.0)
az = {y: np.array(v) for y, v in az.items()}


def axial_median_mad(a):
    """Mediana axial (grade 0.5 graus) e MAD axial, distancia axial <=90."""
    grid = np.arange(0.0, 180.0, 0.5)
    d = np.abs(grid[:, None] - a[None, :])
    d = np.minimum(d, 180.0 - d)
    med = grid[d.sum(axis=1).argmin()]
    dm = np.abs(med - a)
    dm = np.minimum(dm, 180.0 - dm)
    return med, np.median(dm)


# ------------------------------------------------------------------- figura
BIN = 10  # graus
edges = np.arange(0, 181, BIN)
RINGS = [2, 5, 10, 15]  # % (raio = sqrt do valor)

fig, axes = plt.subplots(
    1, 4, figsize=(19 / 2.54, 6.6 / 2.54),
    subplot_kw={"projection": "polar"},
)
fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.16, wspace=0.45)

for ax, y in zip(axes, ["2016", "2017", "2018", "2019"]):
    a = az[y]
    cnt, _ = np.histogram(a, bins=edges)
    freq = cnt / cnt.sum() * 100.0  # % por bin de 10 graus
    r = np.sqrt(freq)
    theta = np.deg2rad(edges[:-1] + BIN / 2.0)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # barras espelhadas (dado axial): hemisferio 0-180 e copia a +180
    for t0 in (theta, theta + np.pi):
        ax.bar(t0, r, width=np.deg2rad(BIN) * 0.97, bottom=0.0,
               color=YEAR_COLORS[y], edgecolor="white", linewidth=0.3,
               alpha=0.9, zorder=3)

    # mediana axial: linha radial atravessando a roseta
    med, mad = axial_median_mad(a)
    rmax = np.sqrt(RINGS[-1]) * 1.02
    for m in (med, med + 180.0):
        ax.plot([np.deg2rad(m)] * 2, [0, rmax], color="#333333", lw=1.1,
                ls="--", zorder=4)

    # grade e rotulos
    ax.set_rlim(0, rmax)
    ax.set_rgrids(np.sqrt(RINGS), labels=[f"{v}%" for v in RINGS],
                  angle=171, fontsize=5.8, color="#777777")
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      labels=["N", "", "L", "", "S", "", "O", ""],
                      fontsize=7.5, color="#555555")
    ax.tick_params(pad=-2)
    ax.grid(color="#cccccc", lw=0.5, alpha=0.8)
    ax.spines["polar"].set_color("#aaaaaa")
    ax.spines["polar"].set_linewidth(0.6)

    ax.set_title(y, color=YEAR_COLORS[y], fontweight="bold", fontsize=10,
                 pad=13)
    n = f"{len(a):,}".replace(",", " ")
    ax.text(0.5, -0.24,
            f"mediana {med:.0f}° | MAD {mad:.0f}°\nn = {n}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7,
            color="#444444", linespacing=1.35)

out = os.path.join(OUT, "fig11_rosetas_orientacao.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", out)

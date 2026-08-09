#!/usr/bin/env python
"""Fig10 — Morfometria e fragmentacao das fendas 2016-2019.

Grade 2x2:
(a) histograma de area (bins log, step, densidade) com medianas como ticks no topo
(b) idem para comprimento
(c) ECDF de area (eixo x log)
(d) boxplot de largura por ano (whiskers 5-95)

Dados: results/crevasse_attrs_all.csv (inventario de 09/08).
Saida: results/figures/relatorio/fig10_morfometria.png (300 dpi, ~19 cm).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YEAR_COLORS = {
    2016: "#0072B2",
    2017: "#E69F00",
    2018: "#009E73",
    2019: "#D55E00",
}
YEARS = [2016, 2017, 2018, 2019]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    }
)

df = pd.read_csv(os.path.join(REPO, "results", "crevasse_attrs_all.csv"))
# largura 0 ou negativa nao e fisica (artefato de esqueletizacao) -> descarta so no painel d
groups = {y: df[df["year"] == y] for y in YEARS}
n_by_year = {y: len(groups[y]) for y in YEARS}

fig, axes = plt.subplots(2, 2, figsize=(19 / 2.54, 15 / 2.54))
(ax_a, ax_b), (ax_c, ax_d) = axes


def hist_panel(ax, col, bins, xlabel):
    """Histograma step com bins geometricos + tick de mediana no topo."""
    for y in YEARS:
        vals = groups[y][col].values
        vals = vals[vals > 0]
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.5,
            color=YEAR_COLORS[y],
            label=f"{y} (n={n_by_year[y]:,})".replace(",", "."),
        )
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidade de probabilidade")
    # ticks de mediana no topo (eixo em coordenadas de dados x, fracao y)
    for i, y in enumerate(YEARS):
        vals = groups[y][col].values
        med = np.median(vals[vals > 0])
        ax.axvline(med, ymin=0.94, ymax=1.0, color=YEAR_COLORS[y], lw=2.0)
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)


# (a) area
bins_area = np.logspace(np.log10(0.06), np.log10(700), 40)
hist_panel(ax_a, "area_m2", bins_area, "Área da fenda (m²)")
ax_a.set_title("(a) Distribuição de área", loc="left")

# (b) comprimento
bins_len = np.logspace(np.log10(0.2), np.log10(200), 40)
hist_panel(ax_b, "length_m", bins_len, "Comprimento da fenda (m)")
ax_b.set_title("(b) Distribuição de comprimento", loc="left")

# (c) ECDF de area
for y in YEARS:
    vals = np.sort(groups[y]["area_m2"].values)
    vals = vals[vals > 0]
    ecdf = np.arange(1, len(vals) + 1) / len(vals)
    ax_c.plot(vals, ecdf, color=YEAR_COLORS[y], lw=1.5, label=str(y))
    med = np.median(vals)
    ax_c.plot(med, 0.5, "o", ms=4, color=YEAR_COLORS[y], zorder=5)
ax_c.axhline(0.5, color="0.6", lw=0.6, ls=":", zorder=1)
ax_c.set_xscale("log")
ax_c.set_xlabel("Área da fenda (m²)")
ax_c.set_ylabel("Fração acumulada")
ax_c.set_ylim(0, 1.02)
ax_c.set_title("(c) ECDF de área", loc="left")
# rotulos diretos: mediana de cada ano
ax_c.legend(frameon=False, loc="lower right", handlelength=1.2)
ax_c.annotate(
    "pontos = medianas",
    xy=(0.02, 0.96),
    xycoords="axes fraction",
    ha="left",
    va="top",
    fontsize=7.5,
    color="0.3",
)

# (d) boxplot de largura, whiskers 5-95
data_w = [groups[y].loc[groups[y]["width_m"] > 0, "width_m"].values for y in YEARS]
bp = ax_d.boxplot(
    data_w,
    positions=range(len(YEARS)),
    whis=(5, 95),
    showfliers=False,
    widths=0.55,
    patch_artist=True,
    medianprops=dict(color="black", lw=1.4),
)
for patch, y in zip(bp["boxes"], YEARS):
    patch.set_facecolor(YEAR_COLORS[y])
    patch.set_alpha(0.75)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.8)
for part in ("whiskers", "caps"):
    for line in bp[part]:
        line.set_color("black")
        line.set_linewidth(0.8)
ax_d.set_xticks(range(len(YEARS)))
ax_d.set_xticklabels([str(y) for y in YEARS])
ax_d.set_ylabel("Largura da fenda (m)")
ax_d.set_xlabel("Ano")
ax_d.set_title("(d) Largura por ano (whiskers 5–95%)", loc="left")
# anota mediana ao lado da linha central de cada caixa
for i, y in enumerate(YEARS):
    med = np.median(data_w[i])
    ax_d.annotate(
        f"{med:.2f}",
        xy=(i + 0.30, med),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.5,
    )
ax_d.set_xlim(-0.6, len(YEARS) - 0.25)
ax_d.yaxis.grid(True, lw=0.4, color="0.85")
ax_d.set_axisbelow(True)

for ax in (ax_a, ax_b, ax_c):
    ax.grid(True, which="major", lw=0.4, color="0.9")
    ax.set_axisbelow(True)

fig.tight_layout(w_pad=2.0, h_pad=2.0)
out = os.path.join(REPO, "results", "figures", "relatorio", "fig10_morfometria.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300)
print("saved", out)

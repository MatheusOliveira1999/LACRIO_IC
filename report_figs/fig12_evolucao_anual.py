#!/usr/bin/env python
"""Fig12 — Evolucao 2016-2019: contagem, area total, cobertura e altitude mediana.

4 paineis empilhados (sharex, eixo categorico com os 4 anos):
(a) n de fendas (n_components)
(b) area total (10^3 m2)
(c) cobertura do glaciar (%)
(d) altitude mediana das fendas (m), ponto + IC bootstrap 95%

Barras de erro (a-c): aproximacao pelos limites de precision/recall da
validacao cruzada espacial (P=0.6131, R=0.7020): [valor*P, valor/R].
Dados: results/crevasses_stats_per_year.csv (paineis a-c, mascaras finais 8 cm)
       results/crevasse_attrs_all.csv (painel d, inventario de 09/08).
Saida: results/figures/relatorio/fig12_evolucao_anual.png (300 dpi, ~9 cm).
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
P_CV = 0.6131  # precisao media (CV espacial 5-fold)
R_CV = 0.7020  # recall medio (CV espacial 5-fold)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
    }
)

stats = pd.read_csv(os.path.join(REPO, "results", "crevasses_stats_per_year.csv"))
stats = stats.set_index("year").loc[YEARS]
attrs = pd.read_csv(os.path.join(REPO, "results", "crevasse_attrs_all.csv"))

n_comp = stats["n_components"].values.astype(float)
area_k = stats["total_area_m2"].values / 1e3  # 10^3 m2
cover = stats["coverage_percent"].values

# altitude mediana + IC bootstrap 95% da mediana
rng = np.random.default_rng(42)
elev_med, elev_lo, elev_hi = [], [], []
for y in YEARS:
    v = attrs.loc[attrs["year"] == y, "elev_m"].dropna().values
    elev_med.append(np.median(v))
    boots = np.median(
        rng.choice(v, size=(1000, len(v)), replace=True), axis=1
    )
    elev_lo.append(np.percentile(boots, 2.5))
    elev_hi.append(np.percentile(boots, 97.5))
elev_med, elev_lo, elev_hi = map(np.array, (elev_med, elev_lo, elev_hi))

x = np.arange(len(YEARS))
colors = [YEAR_COLORS[y] for y in YEARS]

fig, axes = plt.subplots(
    4, 1, figsize=(9 / 2.54, 17.5 / 2.54), sharex=True,
    gridspec_kw={"hspace": 0.42},
)


def bar_panel(ax, vals, ylabel, title, fmt):
    lo = vals * P_CV          # limite inferior: so o que e precisao
    hi = vals / R_CV          # limite superior: corrige recall
    yerr = np.vstack([vals - lo, hi - vals])
    ax.bar(x, vals, width=0.6, color=colors, edgecolor="black", lw=0.6, zorder=3)
    ax.errorbar(
        x, vals, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0,
        capsize=3, capthick=1.0, zorder=4,
    )
    for xi, v, h in zip(x, vals, hi):
        ax.annotate(
            fmt.format(v),
            xy=(xi, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=3)
    ax.set_ylim(0, max(hi) * 1.22)
    ax.yaxis.grid(True, lw=0.4, color="0.88")
    ax.set_axisbelow(True)


bar_panel(
    axes[0], n_comp, "Nº de fendas (×10³)", "(a) Número de fendas",
    "{:,.0f}",
)
axes[0].set_yticks([0, 10000, 20000])
axes[0].set_yticklabels(["0", "10", "20"])
# corrige formato dos rotulos (milhar com ponto)
for t in axes[0].texts:
    t.set_text(t.get_text().replace(",", "."))

bar_panel(
    axes[1], area_k, "Área total (10³ m²)", "(b) Área total de fendas",
    "{:.1f}",
)
bar_panel(
    axes[2], cover, "Cobertura (%)", "(c) Cobertura do glaciar",
    "{:.2f}",
)

# (d) altitude mediana: ponto + haste (IC bootstrap 95%)
ax = axes[3]
ax.errorbar(
    x, elev_med, yerr=[elev_med - elev_lo, elev_hi - elev_med],
    fmt="none", ecolor="black", elinewidth=1.0, capsize=3, capthick=1.0, zorder=3,
)
ax.scatter(x, elev_med, s=34, c=colors, edgecolor="black", lw=0.6, zorder=4)
for xi, v, h in zip(x, elev_med, elev_hi):
    ax.annotate(
        f"{v:.1f}",
        xy=(xi, h),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
ax.set_ylabel("Altitude mediana (m)")
ax.set_title("(d) Altitude mediana das fendas", loc="left", pad=3)
pad = (elev_hi.max() - elev_lo.min()) * 0.35
ax.set_ylim(elev_lo.min() - pad, elev_hi.max() + pad)
ax.yaxis.grid(True, lw=0.4, color="0.88")
ax.set_axisbelow(True)

ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEARS])
ax.set_xlabel("Ano")
ax.set_xlim(-0.6, len(YEARS) - 0.4)

fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.08)
out = os.path.join(REPO, "results", "figures", "relatorio", "fig12_evolucao_anual.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
print("saved", out)

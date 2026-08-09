# -*- coding: utf-8 -*-
"""
Fig. 8 — Controle hipsométrico: dH e densidade de fendas por banda de elevação.
Painéis: (a) dH mediano por banda de 50 m por par de anos; (b) densidade de
fendas (% da área com gelo da banda) por ano + n de células (apoio, cinza);
(c) distribuição de elevação das fendas por ano (KDE normalizada pico=1).

Dados: results/density_dh_bands.csv, results/crevasse_attrs_all.csv.
Saída: results/figures/relatorio/fig08_hipsometria.png (300 dpi, ~19 cm).
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "figures", "relatorio")
os.makedirs(OUT, exist_ok=True)

YEAR_COLORS = {
    "2016": "#0072B2",
    "2017": "#E69F00",
    "2018": "#009E73",
    "2019": "#D55E00",
}
COREG_ERR = 0.04  # m, terreno estável (co-registro)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#555555",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
})


def fmt_br(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


# ---------------------------------------------------------------- dados: bandas
bands = {}  # pair -> list of dict
with open(os.path.join(REPO, "results", "density_dh_bands.csv")) as f:
    for r in csv.DictReader(f):
        bands.setdefault(r["pair"], []).append(
            {k: float(v) if k != "pair" else v for k, v in r.items()}
        )
pairs = ["2016-2017", "2017-2018", "2018-2019"]
centers = {p: np.array([(b["band_lo_m"] + b["band_hi_m"]) / 2 for b in bands[p]])
           for p in pairs}

# densidade por ANO: dens_t1_pct é do ano inicial do par; 2019 = 2018 + ddens
dens_year = {
    "2016": np.array([b["dens_t1_pct"] for b in bands["2016-2017"]]),
    "2017": np.array([b["dens_t1_pct"] for b in bands["2017-2018"]]),
    "2018": np.array([b["dens_t1_pct"] for b in bands["2018-2019"]]),
    "2019": np.array([b["dens_t1_pct"] + b["ddens_pp"] for b in bands["2018-2019"]]),
}
ncells = np.array([b["n_cells"] for b in bands["2018-2019"]])  # apoio

# ------------------------------------------------------------- dados: fendas
elev = {y: [] for y in YEAR_COLORS}
with open(os.path.join(REPO, "results", "crevasse_attrs_all.csv")) as f:
    for r in csv.DictReader(f):
        if r["elev_m"]:
            elev[r["year"]].append(float(r["elev_m"]))
elev = {y: np.array(v) for y, v in elev.items()}

# ------------------------------------------------------------------- figura
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    3, 1, figsize=(19 / 2.54, 17 / 2.54), sharex=True,
    gridspec_kw={"hspace": 0.14, "height_ratios": [1.0, 1.0, 0.9]},
)
XLIM = (15, 300)

# ---- (a) dH por banda, por par --------------------------------------------
ax_a.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
for p in pairs:
    y0 = p.split("-")[0]
    c = YEAR_COLORS[y0]
    dh = np.array([b["dh_median_m"] for b in bands[p]])
    ax_a.errorbar(centers[p], dh, yerr=COREG_ERR, color=c, marker="o", ms=4,
                  lw=1.6, capsize=2, zorder=3)
    ax_a.annotate(p, xy=(centers[p][-1], dh[-1]), xytext=(6, 0),
                  textcoords="offset points", va="center", color=c,
                  fontsize=8, fontweight="bold")
ax_a.set_ylabel("dH mediano (m)")
ax_a.set_ylim(-7.6, 0.9)
ax_a.text(0.985, 0.06,
          f"barras de erro: co-registro ±{fmt_br(COREG_ERR, 2)} m "
          "(menores que os marcadores)",
          transform=ax_a.transAxes, ha="right", va="bottom",
          fontsize=7.2, color="#777777")
ax_a.text(0.012, 0.94, "(a)", transform=ax_a.transAxes, fontsize=10,
          fontweight="bold", va="top")

# ---- (b) densidade por ano + n de células (apoio) -------------------------
# rótulos diretos em pontos distintos de cada curva (evita colisão à direita)
lab_anchor = {"2016": (0, (0, 9)), "2017": (0, (0, -11)),
              "2018": (1, (10, 8)), "2019": (1, (0, -12))}
for y in ["2016", "2017", "2018", "2019"]:
    cx = centers[pairs[0]]
    ax_b.plot(cx, dens_year[y], color=YEAR_COLORS[y], marker="o", ms=4,
              lw=1.6, zorder=3)
    i, xy_off = lab_anchor[y]
    ax_b.annotate(y, xy=(cx[i], dens_year[y][i]), xytext=xy_off,
                  textcoords="offset points", ha="center", va="center",
                  color=YEAR_COLORS[y], fontsize=8, fontweight="bold")
ax_b.set_ylabel("Densidade de fendas\n(% da área da banda)")
ax_b.set_ylim(0, 5.2)
ax_b.text(0.012, 0.94, "(b)", transform=ax_b.transAxes, fontsize=10,
          fontweight="bold", va="top")

ax_n = ax_b.twinx()  # linha fina de apoio: n de células válidas por banda
ax_n.plot(centers[pairs[0]], ncells / 1000.0, color="#aaaaaa", lw=1.0,
          ls=":", zorder=2)
ax_n.set_ylabel("n células (×10³)", color="#888888", fontsize=7.5)
ax_n.tick_params(axis="y", colors="#999999", labelsize=7)
ax_n.set_ylim(0, 260)
ax_n.spines["top"].set_visible(False)
ax_n.spines["right"].set_color("#bbbbbb")
ax_n.annotate("n células", xy=(centers[pairs[0]][2], ncells[2] / 1000.0),
              xytext=(0, 5), textcoords="offset points", ha="center",
              color="#999999", fontsize=7)

# ---- (c) distribuição de elevação das fendas ------------------------------
grid = np.linspace(XLIM[0], XLIM[1], 400)
for y in ["2016", "2017", "2018", "2019"]:
    e = elev[y]
    kde = gaussian_kde(e)
    d = kde(grid)
    d = d / d.max()  # pico = 1
    ax_c.plot(grid, d, color=YEAR_COLORS[y], lw=1.6, zorder=3)
    med = np.median(e)
    ax_c.axvline(med, color=YEAR_COLORS[y], lw=1.0, ls="--", alpha=0.75,
                 ymax=0.92, zorder=2)
ax_c.set_ylim(0, 1.12)
ax_c.set_yticks([0, 0.5, 1])
ax_c.set_yticklabels(["0", "0,5", "1"])
ax_c.set_ylabel("Densidade\n(normalizada, pico = 1)")
ax_c.set_xlabel("Elevação (m, bandas de 50 m)")
ax_c.set_xlim(*XLIM)
ax_c.text(0.012, 0.94, "(c)", transform=ax_c.transAxes, fontsize=10,
          fontweight="bold", va="top")

# painel de texto à direita: n e mediana por ano (cores por ano)
ax_c.text(0.985, 0.95, "mediana | n", transform=ax_c.transAxes, ha="right",
          va="top", fontsize=7.2, color="#555555", fontweight="bold")
for k, y in enumerate(["2016", "2017", "2018", "2019"]):
    m = np.median(elev[y])
    n = f"{len(elev[y]):,}".replace(",", " ")
    ax_c.text(0.985, 0.84 - 0.115 * k,
              f"{y}: {fmt_br(m, 0)} m | n = {n}",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=7.2,
              color=YEAR_COLORS[y], fontweight="bold")

fig.align_ylabels([ax_a, ax_b, ax_c])
out = os.path.join(OUT, "fig08_hipsometria.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", out)

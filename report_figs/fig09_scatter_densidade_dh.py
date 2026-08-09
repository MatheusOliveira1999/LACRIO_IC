#!/usr/bin/env python
"""Fig. 9 — Densidade de fendas vs dH em blocos de 100 m (3 pares interanuais).

Scatter 1x3 (2016-17, 2017-18, 2018-19) de dH mediano do bloco (m) contra
densidade de fendas do bloco (%), pontos coloridos por elevacao (viridis,
colorbar unica). rho de Spearman, p e n anotados em cada painel (valores de
results/density_dh_summary.json). Linha de tendencia pontilhada apenas no par
2016-2017 (unico com acoplamento negativo significativo); os resultados nulos
sao mostrados com o mesmo desenho.

Uso: python report_figs/fig09_scatter_densidade_dh.py
Saida: results/figures/relatorio/fig09_scatter_densidade_dh.png (300 dpi)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "density_dh_blocks.csv")
SUMMARY = os.path.join(REPO, "results", "density_dh_summary.json")
OUT = os.path.join(REPO, "results", "figures", "relatorio", "fig09_scatter_densidade_dh.png")

PAIRS = ["2016-2017", "2017-2018", "2018-2019"]
TREND_PAIRS = {"2016-2017", "2017-2018"}  # linha pontilhada onde p<0.05 (regra do guia); 2017-18 e fraco-positivo


def fmt_num(v, dec=2):
    return f"{v:.{dec}f}".replace(".", ",").replace("-", "−")


def fmt_p(p):
    if p < 0.001:
        return "p < 0,001"
    return ("p = " + f"{p:.3f}").replace(".", ",")


def main():
    df = pd.read_csv(CSV)
    df = df[df["ice"] == 1]
    summ = json.load(open(SUMMARY))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(19 / 2.54, 7.2 / 2.54), sharex=True, sharey=True
    )

    vmin, vmax = df["elev_m"].min(), df["elev_m"].max()
    sc = None
    for ax, pair in zip(axes, PAIRS):
        sub = df[df["pair"] == pair]
        s = summ[pair]

        ax.axhline(0, color="0.55", lw=0.7, zorder=1)
        sc = ax.scatter(
            sub["dens_pct"], sub["dh_median_m"], c=sub["elev_m"],
            cmap="viridis", vmin=vmin, vmax=vmax,
            s=10, alpha=0.65, linewidths=0, zorder=3,
        )

        if pair in TREND_PAIRS:
            b1, b0 = np.polyfit(sub["dens_pct"], sub["dh_median_m"], 1)
            xs = np.linspace(0, sub["dens_pct"].max(), 50)
            ax.plot(xs, b0 + b1 * xs, ls=":", lw=1.4, color="0.15", zorder=4)

        rho_txt = "ρ = " + fmt_num(s["rho"])
        box = f"{rho_txt}\n{fmt_p(s['p'])}\nn = {s['n_blocks']} blocos"
        ax.text(0.965, 0.965, box, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", lw=0.6))

        ax.set_title(pair.replace("-", "–"), fontsize=9, pad=4)
        ax.set_xlim(-0.4, 12.8)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, ls=":", lw=0.4, color="0.85")
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5)

    fmt = FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "−"))
    for ax in axes:
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

    axes[0].set_ylabel("dH mediano do bloco (m)")
    axes[1].set_xlabel("Densidade de fendas do bloco (%)")

    fig.subplots_adjust(left=0.065, right=0.90, top=0.90, bottom=0.17, wspace=0.08)
    cax = fig.add_axes([0.915, 0.17, 0.016, 0.73])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Elevação (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()

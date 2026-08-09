#!/usr/bin/env python
"""Fig. 2 — Curvas Precision/Recall/F1 vs threshold de binarizacao (CV espacial 5-fold).

Deriva as curvas exatas dos histogramas de probabilidade OOF acumulados por fold
(results/crossval/crossval_crevasses_fp0.7_fn0.3_curves.npz):
  pos_ge[i] = soma(hist_pos[i:])  -> TP com threshold = i/N
  neg_ge[i] = soma(hist_neg[i:])  -> FP
Linha = media entre os 5 folds; banda = +-1 desvio-padrao entre folds.
Pontos calculados a cada 0.05 ligados por linha (sem suavizacao).

Uso: python report_figs/fig02_curvas_prf1_threshold.py
Saida: results/figures/relatorio/fig02_curvas_prf1_threshold.png (300 dpi)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(REPO, "results", "crossval", "crossval_crevasses_fp0.7_fn0.3_curves.npz")
OUT = os.path.join(REPO, "results", "figures", "relatorio", "fig02_curvas_prf1_threshold.png")

THR_OP = 0.6           # threshold operacional usado nos produtos finais
F1_CV_TXT = "F1 (CV) = 0,654 ± 0,014\n(5 folds espaciais)"

# Cores CVD-safe (Okabe-Ito): P azul, R laranja, F1 preto
C_P = "#0072B2"
C_R = "#E69F00"
C_F1 = "#000000"


def prf_at(hist_pos, hist_neg, thr):
    """P, R, F1 no threshold thr (pred positivo se prob >= thr)."""
    n = len(hist_pos)
    pos_ge = np.cumsum(hist_pos[::-1])[::-1]
    neg_ge = np.cumsum(hist_neg[::-1])[::-1]
    idx = np.clip(np.round(np.asarray(thr) * n).astype(int), 0, n - 1)
    tp = pos_ge[idx]
    fp = neg_ge[idx]
    fn = hist_pos.sum() - tp
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return p, r, f1


def main():
    d = np.load(NPZ)
    hp_f = d["hist_pos_folds"].astype(np.float64)  # (5, 2000)
    hn_f = d["hist_neg_folds"].astype(np.float64)

    thr = np.arange(0.05, 0.96, 0.05)
    curves = np.array([prf_at(hp_f[k], hn_f[k], thr) for k in range(hp_f.shape[0])])
    # curves: (5 folds, 3 metricas, n_thr)
    mean = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=1)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(9 / 2.54, 7.4 / 2.54))

    labels = ["Precisão", "Recall", "F1"]
    colors = [C_P, C_R, C_F1]
    for m, (lab, c) in enumerate(zip(labels, colors)):
        ax.fill_between(thr, mean[m] - sd[m], mean[m] + sd[m], color=c, alpha=0.18, lw=0)
        lw = 1.6 if lab == "F1" else 1.2
        ax.plot(thr, mean[m], "-o", color=c, lw=lw, ms=2.6, label=lab, zorder=3)

    # threshold operacional
    ax.axvline(THR_OP, color="0.35", ls="--", lw=0.9, zorder=2)
    ax.text(THR_OP + 0.012, 0.03, "thr. operacional = 0,6", rotation=90,
            va="bottom", ha="left", fontsize=7, color="0.25")

    # losango no F1 maximo da curva media
    i_max = int(np.argmax(mean[2]))
    ax.plot(thr[i_max], mean[2, i_max], marker="D", ms=5.5, mfc="white",
            mec=C_F1, mew=1.2, zorder=5)
    ax.annotate(f"F1 máx = {mean[2, i_max]:.3f}".replace(".", ",") +
                f" (thr {thr[i_max]:.2f})".replace(".", ","),
                xy=(thr[i_max], mean[2, i_max]),
                xytext=(thr[i_max] - 0.16, mean[2, i_max] - 0.20),
                ha="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))

    # caixa com o F1 do CV
    ax.text(0.03, 0.055, F1_CV_TXT, transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", lw=0.6))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold de binarização")
    ax.set_ylabel("Métrica")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    fmt = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(True, ls=":", lw=0.4, color="0.85")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="0.7", fontsize=7.5, borderpad=0.4)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=300)
    print("OK ->", OUT)
    print("F1 medio @0.6:", mean[2][np.argmin(np.abs(thr - 0.6))])
    print("F1 max:", mean[2, i_max], "@", thr[i_max])


if __name__ == "__main__":
    main()

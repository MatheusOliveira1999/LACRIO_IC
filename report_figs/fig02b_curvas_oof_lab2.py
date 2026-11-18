"""Figura 3 (revisao set/2026): P/R/F1 x limiar a partir dos mapas OOF do CV com GT corrigido (lab2),
banda = +-1 dp entre folds; single (linha cheia) e TTA (tracejada). Tambem recomputa AUC-PR/ROC."""
import json, numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from config import Config
TAG = "crevasses_fp0.7_fn0.3_lab2"; base = Config.RESULTS_DIR / "oof" / TAG
fold_of = json.load(open(base / "fold_of_tile.json"))
thr = np.round(np.arange(0.05, 0.96, 0.05), 2)
def counts(variant):
    per_fold = {k: np.zeros((len(thr), 3)) for k in sorted(set(fold_of.values()))}
    hp = np.zeros(2000); hn = np.zeros(2000)
    for f in sorted((base / f"probs_{variant}").glob("*.npy")):
        p = np.load(f).astype(np.float32); gt = cv2.imread(str(Config.MASKS_DIR / "2016" / "annotations" / "crevasses" / (f.stem + "_crevasses.png")), 0) > 127
        k = fold_of[f.stem + ".png"]
        for i, t in enumerate(thr):
            pr = p > t; per_fold[k][i] += [(pr & gt).sum(), (pr & ~gt).sum(), (~pr & gt).sum()]
        hp += np.histogram(p[gt], bins=2000, range=(0, 1))[0]; hn += np.histogram(p[~gt], bins=2000, range=(0, 1))[0]
    return per_fold, hp, hn
def prf(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    P = tp / np.maximum(tp + fp, 1); R = tp / np.maximum(tp + fn, 1); F = 2 * P * R / np.maximum(P + R, 1e-9); return P, R, F
def auc(hp, hn):
    tp = np.cumsum(hp[::-1])[::-1]; fp = np.cumsum(hn[::-1])[::-1]
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0); rec = tp / hp.sum(); fpr = fp / hn.sum()
    o = np.argsort(rec); o2 = np.argsort(fpr)
    return float(np.trapz(prec[o], rec[o])), float(np.trapz(rec[o2], fpr[o2]))
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
COL = {"P": "#2a78d6", "R": "#eb6834", "F1": "#1baf7a"}; out = {}
for variant, ls, lab in (("single", "-", "modelo único"), ("tta", "--", "TTA")):
    pf, hp, hn = counts(variant); arr = np.stack([np.stack(prf(pf[k]), axis=1) for k in pf])  # folds x thr x 3
    tot = sum(pf.values()); Pt, Rt, Ft = prf(tot)
    for j, (name, tot_curve) in enumerate((("P", Pt), ("R", Rt), ("F1", Ft))):
        m = tot_curve; s = arr[:, :, j].std(axis=0)
        ax.plot(thr, m, ls, color=COL[name], lw=2, label=f"{ {'P':'Precisão','R':'Revocação','F1':'F1'}[name]} — {lab}")
        if variant == "single": ax.fill_between(thr, m - s, m + s, color=COL[name], alpha=0.12, lw=0)
    out[variant] = {"auc_pr": auc(hp, hn)[0], "auc_roc": auc(hp, hn)[1], "F1@0.5": float(Ft[np.argmin(abs(thr - 0.5))]), "F1@0.6": float(Ft[np.argmin(abs(thr - 0.6))]), "best_thr": float(thr[np.argmax(Ft)]), "best_F1": float(Ft.max())}
ax.axvline(0.6, color="#777", lw=1, ls=":"); ax.text(0.605, 0.31, "limiar de produção (0,6)", fontsize=8, color="#555")
ax.set_xlabel("limiar de binarização"); ax.set_ylabel("métrica (out-of-fold, 218 tiles)"); ax.set_ylim(0.3, 0.95); ax.set_xlim(0.05, 0.95)
ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left")
fig.tight_layout(); outp = Config.RESULTS_DIR / "figures" / "relatorio" / "fig02_curvas_prf1_threshold.png"; fig.savefig(outp, bbox_inches="tight")
print(json.dumps(out, indent=1)); print("salvo:", outp)

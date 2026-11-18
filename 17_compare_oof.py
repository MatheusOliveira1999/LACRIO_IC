"""
17_compare_oof.py - Comparacao PAREADA de dois runs OOF (mapas de probabilidade salvos
pelo 16_oof_ensemble_eval.py em results/oof/<tag>/probs_<variante>/).

Para cada tile anotado de 2016: metricas @threshold de A e de B no MESMO tile ->
  * micro P/R/F1 agregado de cada run e a diferenca B-A;
  * IC 95% bootstrap (re-amostrando TILES, 2000 reps) da diferenca micro-F1 / micro-P;
  * teste de Wilcoxon pareado sobre F1 por tile (tiles com GT>0 ou pred>0);
  * tabela por fold (mesmo split, checkpoints dos folds guardam val_tiles).

Uso:
  python 17_compare_oof.py --a crevasses_fp0.7_fn0.3_ft25ctrl --b crevasses_fp0.7_fn0.3_clce1.0_it3_ft25
  python 17_compare_oof.py --a crevasses_fp0.7_fn0.3 --b crevasses_fp0.7_fn0.3 --va single --vb tta
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon

from config import Config


def load_run(tag, variant):
    d = Config.RESULTS_DIR / "oof" / tag / f"probs_{variant}"
    files = sorted(d.glob("tile_*.npy"))
    if not files:
        raise FileNotFoundError(d)
    fold_of = json.load(open(Config.RESULTS_DIR / "oof" / tag / "fold_of_tile.json"))
    return {f.stem + ".png": f for f in files}, fold_of


def gt_mask(fname, feature="crevasses"):
    p = Config.MASKS_DIR / "2016" / "annotations" / feature / fname.replace(".png", f"_{feature}.png")
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (m > 127)


def counts(prob_path, gt, thr):
    pred = np.load(prob_path).astype(np.float32) > thr
    return np.array([(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()], dtype=np.float64)


def prf(c):
    tp, fp, fn = c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="tag do run A (referencia)")
    ap.add_argument("--b", required=True, help="tag do run B (candidato)")
    ap.add_argument("--va", default="single"); ap.add_argument("--vb", default="single")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, fold_a = load_run(args.a, args.va)
    B, fold_b = load_run(args.b, args.vb)
    common = sorted(set(A) & set(B))
    print(f"A={args.a}/{args.va}  B={args.b}/{args.vb}  tiles em comum: {len(common)}")
    if fold_a != fold_b:
        print("  [AVISO] atribuicao de folds difere entre os runs — tabela por fold usa a de A")
    gts = {f: gt_mask(f) for f in common}
    rng = np.random.default_rng(args.seed)
    out = {"a": args.a, "b": args.b, "va": args.va, "vb": args.vb, "n_tiles": len(common), "thresholds": {}}

    for thr in args.thr:
        ca = np.stack([counts(A[f], gts[f], thr) for f in common])
        cb = np.stack([counts(B[f], gts[f], thr) for f in common])
        pa, ra, fa = prf(ca.sum(0)); pb, rb, fb = prf(cb.sum(0))
        # bootstrap por tile da diferenca micro
        idx = np.arange(len(common))
        dF, dP = [], []
        for _ in range(args.reps):
            s = rng.choice(idx, size=len(idx), replace=True)
            _, _, f1a = prf(ca[s].sum(0)); p1a = prf(ca[s].sum(0))[0]
            _, _, f1b = prf(cb[s].sum(0)); p1b = prf(cb[s].sum(0))[0]
            dF.append(f1b - f1a); dP.append(p1b - p1a)
        dF, dP = np.array(dF), np.array(dP)
        # F1 por tile (so tiles onde ha algo a medir)
        fa_t = np.array([prf(c)[2] for c in ca]); fb_t = np.array([prf(c)[2] for c in cb])
        valid = np.array([(c[0] + c[1] + c[2]) > 0 for c in ca]) | np.array([(c[0] + c[1] + c[2]) > 0 for c in cb])
        d_t = (fb_t - fa_t)[valid]
        try:
            w = wilcoxon(d_t, zero_method="zsplit")
            wp = float(w.pvalue)
        except ValueError:
            wp = float("nan")
        print(f"\n@thr={thr:.2f}")
        print(f"  A: P={pa:.4f} R={ra:.4f} F1={fa:.4f}")
        print(f"  B: P={pb:.4f} R={rb:.4f} F1={fb:.4f}")
        print(f"  ΔF1 (B−A) = {fb - fa:+.4f}  IC95 bootstrap [{np.percentile(dF, 2.5):+.4f}, {np.percentile(dF, 97.5):+.4f}]")
        print(f"  ΔP  (B−A) = {pb - pa:+.4f}  IC95 bootstrap [{np.percentile(dP, 2.5):+.4f}, {np.percentile(dP, 97.5):+.4f}]")
        print(f"  F1 por tile: mediana Δ = {np.median(d_t):+.4f} | B>A em {int((d_t > 0).sum())}/{len(d_t)} tiles | Wilcoxon p={wp:.3g}")
        per_fold = []
        for k in sorted(set(fold_a.values())):
            sel = [i for i, f in enumerate(common) if fold_a.get(f) == k]
            if not sel:
                continue
            _, _, f1a_k = prf(ca[sel].sum(0)); _, _, f1b_k = prf(cb[sel].sum(0))
            p_a_k = prf(ca[sel].sum(0))[0]; p_b_k = prf(cb[sel].sum(0))[0]
            per_fold.append({"fold": k, "f1_a": f1a_k, "f1_b": f1b_k, "p_a": p_a_k, "p_b": p_b_k})
        dfk = np.array([r["f1_b"] - r["f1_a"] for r in per_fold])
        print("  por fold: " + " | ".join(f"f{r['fold']}: {r['f1_a']:.4f}→{r['f1_b']:.4f} ({r['f1_b']-r['f1_a']:+.4f})" for r in per_fold))
        print(f"  ΔF1 médio por fold = {dfk.mean():+.4f} ± {dfk.std(ddof=1) if len(dfk) > 1 else 0:.4f} | B>A em {(dfk > 0).sum()}/{len(dfk)} folds")
        out["thresholds"][str(thr)] = {
            "A": {"p": pa, "r": ra, "f1": fa}, "B": {"p": pb, "r": rb, "f1": fb},
            "dF1": fb - fa, "dF1_ci95": [float(np.percentile(dF, 2.5)), float(np.percentile(dF, 97.5))],
            "dP": pb - pa, "dP_ci95": [float(np.percentile(dP, 2.5)), float(np.percentile(dP, 97.5))],
            "wilcoxon_p_tile_f1": wp, "n_tiles_valid": int(valid.sum()),
            "tiles_b_better": int((d_t > 0).sum()), "per_fold": per_fold,
        }
    out_dir = Config.RESULTS_DIR / "oof" / "comparacoes"; out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.a}_{args.va}__vs__{args.b}_{args.vb}.json"
    with open(out_dir / name, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  JSON: {out_dir / name}")


if __name__ == "__main__":
    main()

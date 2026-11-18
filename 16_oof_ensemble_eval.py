"""
16_oof_ensemble_eval.py - Avaliacao OOF por tile + ensemble dos folds + metricas de
esqueleto (clDice/clIoU com tolerancia) + sweep de pos-processamento.

Usa os checkpoints de fold salvos pelo 10_crossval_curves.py
(models/crossval_folds/<tag>/fold{k}.pth, que guardam `val_tiles`), logo:

  --mode oof       cada tile 2016 e predito SOMENTE pelo modelo do fold em que ele
                   foi validacao -> mapas de probabilidade honestos (out-of-fold),
                   com e sem TTA. Salva em results/oof/<tag>/probs{,_tta}/*.npy.
                   Reporta P/R/F1 pixel + clDice/clIoU (tau = 0/2/4 px) e
                   sweep threshold x area minima x histerese.

  --mode ensemble  media das probabilidades dos 5 modelos de fold (+ opcional
                   ep.107 de producao) nos tiles ANOTADOS de --years; compara com
                   o modelo de producao sozinho. Em 2016 os folds sao parcialmente
                   in-sample (cada modelo viu 80% dos tiles) -> so cross-year e
                   comparacao honesta.

Metricas de esqueleto (Shit et al. 2021; OmniCrack30k 2024; Wiedemann 1998):
  Tprec(tau) = |skel(P) dentro de dilate(G, tau)| / |skel(P)|     (correctness)
  Tsens(tau) = |skel(G) dentro de dilate(P, tau)| / |skel(G)|     (completeness)
  clDice = 2*Tprec*Tsens/(Tprec+Tsens) ; clIoU = 1/(1/Tprec + 1/Tsens - 1)
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from config import Config
from shadow_utils import (precompute_year_dem_features, get_dem_features_for_tile,
                          normalize_dem_features)


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_unet", Config.PROJECT_DIR / "03_train_unet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TU = _load_train_module()
DEVICE = Config.DEVICE
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def load_fold_models(tag):
    """tag: nome de uma pasta em models/crossval_folds/ ou varios nomes separados por '+'
    (ensemble de mais de um CV, ex. seeds/splits diferentes)."""
    paths = []
    for t in str(tag).split("+"):
        ckpt_dir = Config.MODELS_DIR / "crossval_folds" / t
        found = sorted(ckpt_dir.glob("fold*.pth"))
        if not found:
            raise FileNotFoundError(f"Nenhum fold em {ckpt_dir}")
        paths += found
    ckpt_dir = Config.MODELS_DIR / "crossval_folds" / str(tag).split("+")[0]
    models, meta = [], None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = TU.UNetResNet34(pretrained=False, in_channels=ck["in_channels"])
        m.load_state_dict(ck["state_dict"])
        m.to(DEVICE).eval()
        models.append({"model": m, "fold": ck["fold"], "best_epoch": ck["best_epoch"],
                       "val_f1": ck["val_f1"],
                       "val_tiles": [Path(t).name for t in ck["val_tiles"]]})
        meta = {"in_channels": ck["in_channels"], "dem_features": ck["dem_features"],
                "dem_window": ck.get("dem_window_meters", 3.0)}
    print(f"  {len(models)} modelos de fold carregados de {tag} "
          f"(best epochs: {[m['best_epoch'] for m in models]})")
    return models, meta


def load_production_model(feature="crevasses", name=None):
    path = Config.MODELS_DIR / (name or f"unet_{feature}_best.pth")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TU.UNetResNet34(pretrained=False, in_channels=ck["in_channels"])
    m.load_state_dict(ck["model_state_dict"])
    m.to(DEVICE).eval()
    print(f"  Modelo de producao: {path.name} (ep.{ck['epoch']}, val_f1={ck.get('val_f1', float('nan')):.4f})")
    return m, {"in_channels": ck["in_channels"], "dem_features": ck["dem_features"],
               "dem_window": ck.get("dem_window_meters", 3.0)}


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
class YearData:
    """Tiles anotados de um ano + provider de DEM features normalizadas."""

    def __init__(self, year, feature, dem_features, dem_window):
        self.year = year
        self.feature = feature
        idx_path = Config.TILES_DIR / str(year) / "tiles_index.json"
        with open(idx_path) as f:
            index = json.load(f)
        self.tile_info = {t["filename"]: t for t in index["tiles"]}
        gt_dir = Config.MASKS_DIR / str(year) / "annotations" / feature
        self.gt = {}
        for gt_path in sorted(gt_dir.glob(f"tile_*_{feature}.png")):
            fname = gt_path.stem.replace(f"_{feature}", "") + ".png"
            if fname in self.tile_info and (Config.TILES_DIR / str(year) / fname).exists():
                self.gt[fname] = gt_path
        print(f"  {year}: {len(self.gt)} tiles anotados")
        self.dem_stack, self.dem_transform, self.dem_stats = None, None, None
        self.dem_features = dem_features
        if dem_features:
            self.dem_stack, self.dem_transform, _, self.dem_stats = \
                precompute_year_dem_features(year, window_meters=dem_window,
                                             features=tuple(dem_features))
            if self.dem_stack is None:
                raise RuntimeError(f"DEM {year} indisponivel")

    def filenames(self):
        return sorted(self.gt.keys())

    def load(self, fname):
        img = cv2.imread(str(Config.TILES_DIR / str(self.year) / fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(self.gt[fname]), cv2.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(np.uint8)
        dem = None
        if self.dem_stack is not None:
            feats = get_dem_features_for_tile(self.tile_info[fname], self.dem_stack,
                                              self.dem_transform)
            dem = normalize_dem_features(feats, self.dem_stats, self.dem_features)
        return img, gt, dem


def to_tensor(image, dem, img_size=512):
    img = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
    if dem is not None:
        if dem.shape[:2] != (img_size, img_size):
            dem = cv2.resize(dem, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            if dem.ndim == 2:
                dem = dem[..., None]
        img = np.concatenate([img, dem.astype(np.float32)], axis=-1)
    return torch.from_numpy(img).permute(2, 0, 1).float()


TTA_OPS = [  # (forward, inverse) em arrays HxW(xC) — flips (= producao, 04_inference)
    (lambda x: x, lambda x: x),
    (lambda x: np.ascontiguousarray(np.fliplr(x)), lambda x: np.fliplr(x)),
    (lambda x: np.ascontiguousarray(np.flipud(x)), lambda x: np.flipud(x)),
    (lambda x: np.ascontiguousarray(np.rot90(x, 2)), lambda x: np.rot90(x, -2)),
]
# Grupo diedral completo D4 (8 simetrias do quadrado): flips + rot90/270 (+ transposta)
TTA_OPS_D4 = TTA_OPS + [
    (lambda x: np.ascontiguousarray(np.rot90(x, 1)), lambda x: np.rot90(x, -1)),
    (lambda x: np.ascontiguousarray(np.rot90(x, 3)), lambda x: np.rot90(x, -3)),
    (lambda x: np.ascontiguousarray(np.swapaxes(x, 0, 1)), lambda x: np.swapaxes(x, 0, 1)),
    (lambda x: np.ascontiguousarray(np.rot90(np.swapaxes(x, 0, 1), 2)),
     lambda x: np.swapaxes(np.rot90(x, -2), 0, 1)),
]
TTA_MODE = "flips"


@torch.no_grad()
def predict_probs(models, image, dem, tta=False, img_size=512):
    """Media das probabilidades de `models` (lista) para um tile; TTA em batch."""
    h, w = image.shape[:2]
    ops = (TTA_OPS_D4 if TTA_MODE == "d4" else TTA_OPS) if tta else TTA_OPS[:1]
    batch = torch.stack([to_tensor(fwd(image), fwd(dem) if dem is not None else None,
                                   img_size) for fwd, _ in ops]).to(DEVICE)
    acc = np.zeros((h, w), dtype=np.float32)
    for m in models:
        probs = torch.sigmoid(m(batch))[:, 0].cpu().numpy()
        for (_, inv), p in zip(ops, probs):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
            acc += inv(p)
    return acc / (len(models) * len(ops))


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def pixel_counts(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    return (int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum()))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": p, "recall": r, "f1": f, "iou": iou}


def skeleton_counts(pred, gt, taus=(0, 2, 4)):
    """Contagens por tile para 3 familias de metricas tolerantes (tau em px):
    (1) clDice tolerante (esqueleto vs MASCARA dilatada): a=|skel(P) em dil(G,tau)|, b=|skel(P)|,
        c=|skel(G) em dil(P,tau)|, d=|skel(G)|;
    (2) clIoU OmniCrack30k (Benz & Rodehorst 2024; esqueleto vs ESQUELETO dilatado, Eq. 1):
        tp=|skel(G) em dil(skel(P),tau)|, fp=|skel(P) fora de dil(skel(G),tau)|, fn=|skel(G)| - tp;
    (3) F1 relaxado (Mnih 2013, rho=tau; mascara vs mascara dilatada):
        tpP=|P em dil(G,tau)|, nP=|P|, tpG=|G em dil(P,tau)|, nG=|G|.
    Retorna dict tau -> tupla de 12 contagens."""
    pred = pred.astype(bool); gt = gt.astype(bool)
    sp = skeletonize(pred) if pred.any() else np.zeros_like(pred)
    sg = skeletonize(gt) if gt.any() else np.zeros_like(gt)
    inf = np.full(gt.shape, np.inf)
    dt_g = distance_transform_edt(~gt) if gt.any() else inf
    dt_p = distance_transform_edt(~pred) if pred.any() else inf
    dt_sg = distance_transform_edt(~sg) if sg.any() else inf
    dt_sp = distance_transform_edt(~sp) if sp.any() else inf
    out = {}
    for tau in taus:
        a = int((sp & (dt_g <= tau)).sum()); b = int(sp.sum())
        c = int((sg & (dt_p <= tau)).sum()); d = int(sg.sum())
        tp = int((sg & (dt_sp <= tau)).sum()); fp = int((sp & ~(dt_sg <= tau)).sum()); fn = d - tp
        tpP = int((pred & (dt_g <= tau)).sum()); nP = int(pred.sum())
        tpG = int((gt & (dt_p <= tau)).sum()); nG = int(gt.sum())
        out[tau] = (a, b, c, d, tp, fp, fn, tpP, nP, tpG, nG, 0)
    return out


def skeleton_metrics(acc):
    """acc: dict tau->12 contagens agregadas (micro)."""
    res = {}
    for tau, v in acc.items():
        a, b, c, d, tp, fp, fn, tpP, nP, tpG, nG, _ = v
        tprec = a / b if b else 0.0
        tsens = c / d if d else 0.0
        cld = 2 * tprec * tsens / (tprec + tsens) if tprec + tsens else 0.0
        cliou = 1.0 / (1.0 / tprec + 1.0 / tsens - 1.0) if tprec and tsens else 0.0
        cliou_omni = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        p_rel = tpP / nP if nP else 0.0
        r_rel = tpG / nG if nG else 0.0
        f1_rel = 2 * p_rel * r_rel / (p_rel + r_rel) if p_rel + r_rel else 0.0
        res[f"tau{tau}"] = {"tprec": tprec, "tsens": tsens, "cldice": cld, "cliou": cliou,
                            "cliou_omnicrack": cliou_omni,
                            "p_relaxed": p_rel, "r_relaxed": r_rel, "f1_relaxed": f1_rel}
    return res


def remove_small(mask, min_area):
    if min_area <= 0 or not mask.any():
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def hysteresis(probs, low, high):
    """Mantem componentes de (probs>low) que contem ao menos um pixel > high."""
    lo = probs > low
    if not lo.any():
        return lo
    hi = probs > high
    n, lab = cv2.connectedComponents(lo.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(lab[hi])] = True
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------------------
# Avaliacao agregada sobre um conjunto {fname: (probs, gt)}
# ---------------------------------------------------------------------------
def evaluate_set(items, thresholds, min_areas=(0,), hyst_lows=(None,), taus=(0, 2, 4),
                 skeleton=True, label=""):
    """items: iteravel de (fname, probs float32 HxW, gt uint8 HxW).
    Retorna lista de dicts (uma linha por config de pos-processamento)."""
    configs = [(t, a, lo) for t in thresholds for a in min_areas for lo in hyst_lows
               if lo is None or lo < t]
    counts = {c: [0, 0, 0] for c in configs}
    skel = {c: {tau: [0] * 12 for tau in taus} for c in configs}
    n = 0
    for fname, probs, gt in items:
        n += 1
        for c in configs:
            t, a, lo = c
            m = hysteresis(probs, lo, t) if lo is not None else probs > t
            m = remove_small(m, a)
            tp, fp, fn = pixel_counts(m, gt)
            counts[c][0] += tp; counts[c][1] += fp; counts[c][2] += fn
            if skeleton:
                for tau, vals in skeleton_counts(m, gt, taus).items():
                    for i in range(12):
                        skel[c][tau][i] += vals[i]
    rows = []
    for c in configs:
        t, a, lo = c
        row = {"threshold": t, "min_area": a, "hyst_low": lo, "n_tiles": n}
        row.update(prf(*counts[c]))
        row.update({"tp": counts[c][0], "fp": counts[c][1], "fn": counts[c][2]})
        if skeleton:
            row["skeleton"] = skeleton_metrics({tau: tuple(v) for tau, v in skel[c].items()})
        rows.append(row)
    return rows


def fmt_row(r):
    hy = "-" if r["hyst_low"] is None else f"{r['hyst_low']:.2f}"
    s = (f"thr={r['threshold']:.2f} area>={r['min_area']:>4} hyst={hy:>4} | "
         f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} IoU={r['iou']:.4f}")
    if "skeleton" in r:
        sk = r["skeleton"]
        for tau in sorted(sk, key=lambda k: int(k[3:])):
            s += (f" | {tau}: clDice={sk[tau]['cldice']:.3f} clIoU={sk[tau]['cliou']:.3f}"
                  f" clIoUomni={sk[tau]['cliou_omnicrack']:.3f} F1rel={sk[tau]['f1_relaxed']:.3f}")
    return s


# ---------------------------------------------------------------------------
# Modos
# ---------------------------------------------------------------------------
def mode_oof(args):
    models, meta = load_fold_models(args.tag)
    data = YearData(2016, args.feature, meta["dem_features"], meta["dem_window"])
    out_dir = Config.RESULTS_DIR / "oof" / (args.tag + args.out_suffix)
    variants = ["single", "tta"] if not args.no_tta else ["single"]
    if args.tta_mode == "d4":
        variants = ["tta"]  # so a variante nova; 'single' ja existe no run padrao
    for v in variants:
        (out_dir / f"probs_{v}").mkdir(parents=True, exist_ok=True)

    # Cobertura: cada tile anotado deve estar em exatamente 1 fold
    owner = {}
    for m in models:
        for f in m["val_tiles"]:
            owner.setdefault(f, []).append(m["fold"])
    fnames = data.filenames()
    missing = [f for f in fnames if f not in owner]
    dup = [f for f, o in owner.items() if len(o) > 1]
    print(f"  cobertura OOF: {len(fnames) - len(missing)}/{len(fnames)} tiles "
          f"(sem fold: {len(missing)}, em >1 fold: {len(dup)})")
    if missing:
        print(f"    [AVISO] tiles anotados fora dos folds (ignorados): {missing[:8]}...")
    by_fold = {m["fold"]: m for m in models}

    t0 = time.time()
    store = {v: {} for v in variants}
    fold_of = {}
    for fname in fnames:
        if fname not in owner:
            continue
        m = by_fold[owner[fname][0]]
        image, gt, dem = data.load(fname)
        fold_of[fname] = m["fold"]
        for v in variants:
            p = predict_probs([m["model"]], image, dem, tta=(v == "tta"))
            np.save(out_dir / f"probs_{v}" / fname.replace(".png", ".npy"), p.astype(np.float16))
            store[v][fname] = (p, gt)
    print(f"  inferencia OOF: {len(fold_of)} tiles x {len(variants)} variantes em "
          f"{time.time() - t0:.0f}s")
    with open(out_dir / "fold_of_tile.json", "w") as f:
        json.dump(fold_of, f, indent=1)

    report = {"tag": args.tag, "n_tiles": len(fold_of), "variants": {}}
    for v in variants:
        items = [(f, p, g) for f, (p, g) in store[v].items()]
        print(f"\n=== OOF {v.upper()} — pixel + esqueleto @0.5 e thresholds ===")
        rows = evaluate_set(items, thresholds=args.thresholds, label=v)
        for r in rows:
            print("  " + fmt_row(r))
        # por fold @0.5 (mesmo protocolo do 10)
        per_fold = []
        for k in sorted(by_fold):
            sub = [(f, p, g) for f, p, g in items if fold_of[f] == k]
            r = evaluate_set(sub, thresholds=[0.5], skeleton=True)[0]
            r["fold"] = k
            per_fold.append(r)
        f1s = [r["f1"] for r in per_fold]; ps = [r["precision"] for r in per_fold]
        cl4 = [r["skeleton"]["tau4"]["cliou"] for r in per_fold]
        print(f"  por fold @0.5: F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
              f"P {np.mean(ps):.4f} ± {np.std(ps):.4f} | clIoU@4 {np.mean(cl4):.4f} ± {np.std(cl4):.4f}")
        if args.no_sweep:
            report["variants"][v] = {"thresholds": rows, "per_fold_at_0.5": per_fold}
            continue
        # sweep pos-processamento
        print(f"\n=== OOF {v.upper()} — sweep threshold x area minima x histerese ===")
        sweep = evaluate_set(items, thresholds=args.sweep_thresholds,
                             min_areas=args.min_areas, hyst_lows=[None] + list(args.hyst_lows),
                             skeleton=True)
        sweep.sort(key=lambda r: -r["f1"])
        base = [r for r in rows if abs(r["threshold"] - 0.5) < 1e-9][0]
        print(f"  baseline thr0.5 sem pos-proc: F1={base['f1']:.4f} P={base['precision']:.4f} "
              f"clIoU@4={base['skeleton']['tau4']['cliou']:.4f}")
        print("  top-10 por F1 pixel:")
        for r in sweep[:10]:
            print("    " + fmt_row(r))
        best_cl = sorted(sweep, key=lambda r: -r["skeleton"]["tau4"]["cliou"])[:5]
        print("  top-5 por clIoU@4px:")
        for r in best_cl:
            print("    " + fmt_row(r))
        report["variants"][v] = {"thresholds": rows, "per_fold_at_0.5": per_fold,
                                 "sweep_top_f1": sweep[:25], "sweep_top_cliou4": best_cl,
                                 "sweep_all": sweep}
    with open(out_dir / "oof_report.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\n  relatorio: {out_dir / 'oof_report.json'}")


def mode_ensemble(args):
    models, meta = load_fold_models(args.tag)
    prod, pmeta = load_production_model(args.feature, args.prod_name)
    assert pmeta["in_channels"] == meta["in_channels"]
    fold_models = [m["model"] for m in models]
    variants = {
        "prod": ([prod], False), "prod_tta": ([prod], True),
        "folds5": (fold_models, False), "folds5_tta": (fold_models, True),
        "folds5+prod_tta": (fold_models + [prod], True),
    }
    if args.no_tta:
        variants = {k: v for k, v in variants.items() if not v[1]}
    out_dir = Config.RESULTS_DIR / "ensemble" / (args.tag + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"tag": args.tag, "years": {}}
    for year in args.years:
        data = YearData(year, args.feature, meta["dem_features"], meta["dem_window"])
        fnames = data.filenames()
        t0 = time.time()
        loaded = {f: data.load(f) for f in fnames}
        store = {v: [] for v in variants}
        for f in fnames:
            image, gt, dem = loaded[f]
            for v, (ms, tta) in variants.items():
                store[v].append((f, predict_probs(ms, image, dem, tta=tta), gt))
        print(f"  {year}: {len(fnames)} tiles x {len(variants)} variantes em {time.time() - t0:.0f}s")
        note = " (folds parcialmente IN-SAMPLE — nao e numero honesto)" if year == 2016 else ""
        print(f"\n=== {year}{note} ===")
        yr = {}
        for v in variants:
            rows = evaluate_set(store[v], thresholds=args.thresholds, skeleton=True)
            yr[v] = rows
            r5 = [r for r in rows if abs(r["threshold"] - 0.5) < 1e-9][0]
            rb = max(rows, key=lambda r: r["f1"])
            print(f"  {v:<16} @0.5: P={r5['precision']:.4f} R={r5['recall']:.4f} F1={r5['f1']:.4f} "
                  f"clIoU@4={r5['skeleton']['tau4']['cliou']:.4f} | "
                  f"melhor thr={rb['threshold']:.2f}: F1={rb['f1']:.4f} P={rb['precision']:.4f}")
            if args.save_probs and v in ("folds5_tta", "prod_tta"):
                d = out_dir / f"probs_{year}_{v}"; d.mkdir(exist_ok=True)
                for f, p, _ in store[v]:
                    np.save(d / f.replace(".png", ".npy"), p.astype(np.float16))
        report["years"][year] = yr
    with open(out_dir / "ensemble_report.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\n  relatorio: {out_dir / 'ensemble_report.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oof", "ensemble"], required=True)
    ap.add_argument("--tag", default="crevasses_fp0.7_fn0.3",
                    help="subpasta de models/crossval_folds/")
    ap.add_argument("--feature", default="crevasses")
    ap.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019, 2016])
    ap.add_argument("--prod-name", default=None, help="checkpoint de producao (models/)")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--tta-mode", choices=["flips", "d4"], default="flips",
                    help="flips = 4 transforms (producao); d4 = 8 simetrias do quadrado")
    ap.add_argument("--out-suffix", default="", help="sufixo da pasta de saida (ex.: _d4)")
    ap.add_argument("--save-probs", action="store_true")
    ap.add_argument("--no-sweep", action="store_true", help="pula o sweep de pos-processamento")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    ap.add_argument("--sweep-thresholds", type=float, nargs="+",
                    default=[0.4, 0.5, 0.55, 0.6, 0.65, 0.7])
    ap.add_argument("--min-areas", type=int, nargs="+", default=[0, 30, 60, 100, 200, 400])
    ap.add_argument("--hyst-lows", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    global TTA_MODE
    TTA_MODE = args.tta_mode
    print(f"Device: {DEVICE} | TTA: {TTA_MODE}")
    if args.mode == "oof":
        mode_oof(args)
    else:
        mode_ensemble(args)


if __name__ == "__main__":
    main()

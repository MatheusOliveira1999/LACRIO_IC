"""
18_selftrain_pilot.py - Piloto de self-training offline em UM fold do CV espacial, com
controle pareado (receita da nota "Pesquisa - Caminhos de melhoria (2026-09)", acao 4).

Protocolo (fold k de um CV ja rodado, ex. --tag crevasses_fp0.7_fn0.3_lab2 --fold 1):
  1. Reconstroi o split do fold (mesma logica/seed do 10_crossval_curves.py) e confere que
     os val_tiles batem com o checkpoint models/crossval_folds/<tag>/fold{k}.pth.
  2. Teacher = modelo do fold k + TTA flips. Pool nao rotulado = tiles 2016 sem anotacao
     cujo footprint fica a >= --buffer-px de QUALQUER tile de validacao (tampao espacial).
  3. Pseudo-rotulos DUROS: p >= tau_pos -> 1 ; p <= tau_neg -> 0 ; entre -> ignorar (128).
     Mineracao: pool_pos = 0.5% <= area(p>=tau_pos) <= 15% ; pool_neg = area(p>tau_neg) <= 1%.
     Composicao ~55% pos / 45% neg.
  4. Treina do ImageNet, real + pseudo (razao pseudo:real por passo = --ratio), aug forte nos
     pseudo (fotometrica RGB, ganho/offset/ruido no DEM, CutMix consistente), loss pseudo =
     BCE + soft-Dice so nos pixels validos. Real: 0.3 BCE + 0.7 Tversky(0.7/0.3) (producao).
  5. CONTROLE pareado: mesmo pipeline, mesma aug forte e CutMix nos rotulados, mesma semente,
     mesmo numero de passos, sem ramo pseudo (lambda_u = 0).
  6. Avaliacao no fold de validacao (single + TTA): pixel P/R/F1 e metricas de esqueleto;
     salva mapas em results/oof/<tag>_pilot{k}_<arm>/probs_* para o 17_compare_oof.py.

Uso:
  python 18_selftrain_pilot.py --tag crevasses_fp0.7_fn0.3_lab2 --fold 1 --arm semi
  python 18_selftrain_pilot.py --tag crevasses_fp0.7_fn0.3_lab2 --fold 1 --arm ctrl
  (--stage pseudo so gera os pseudo-rotulos; --stage train so treina; default: ambos)
"""

import argparse
import copy
import importlib.util
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import Config


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, Config.PROJECT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TU = _load("train_unet", "03_train_unet.py")
CV = _load("crossval", "10_crossval_curves.py")
EV = _load("oofeval", "16_oof_ensemble_eval.py")
DEVICE = Config.DEVICE
IGNORE = 128


# ---------------------------------------------------------------------------
# 1. Split do fold (identico ao 10) + checagem contra o checkpoint
# ---------------------------------------------------------------------------
def rebuild_fold(args):
    tile_paths, mask_paths = TU.collect_pairs(args.feature, years=[2016],
                                              neg_ratio=args.neg_ratio, shadow_neg_ratio=0.5)
    is_pos = [1 if m is not None else 0 for m in mask_paths]
    coords = CV.load_tile_coords([2016])
    groups, xy, _ = CV.assign_spatial_groups(tile_paths, coords, args.block_size)
    folds = CV.grouped_kfold_balanced(groups, is_pos, args.folds, seed=args.seed)
    val_idx = folds[args.fold - 1]
    all_idx = set(range(len(tile_paths)))
    train_idx = sorted(all_idx - set(val_idx))
    train_idx, n_removed = CV.buffered_train_idx(train_idx, val_idx, xy, Config.TILE_SIZE)
    ck_path = Config.MODELS_DIR / "crossval_folds" / args.tag / f"fold{args.fold}.pth"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ck_val = sorted(Path(t).name for t in ck["val_tiles"])
    my_val = sorted(tile_paths[i].name for i in val_idx)
    if ck_val != my_val:
        msg = "val_tiles do checkpoint != split reconstruido (seed/neg-ratio/GT mudou?)"
        if not args.no_split_check:
            raise AssertionError(msg)
        print(f"  [AVISO] {msg} — seguindo com o split do CHECKPOINT (val) e o restante como treino")
        val_set = set(ck_val)
        val_idx = [i for i, t in enumerate(tile_paths) if t.name in val_set]
        train_idx = sorted(all_idx - set(val_idx))
        train_idx, n_removed = CV.buffered_train_idx(train_idx, val_idx, xy, Config.TILE_SIZE)
    print(f"  fold {args.fold}: treino={len(train_idx)} ({sum(is_pos[i] for i in train_idx)} pos) "
          f"| val={len(val_idx)} ({sum(is_pos[i] for i in val_idx)} pos) | buffer removeu {n_removed}"
          f" | split confere com {ck_path.name} (best ep {ck['best_epoch']}, val_f1 {ck['val_f1']:.4f})")
    train = [(tile_paths[i], mask_paths[i]) for i in train_idx]
    val = [(tile_paths[i], mask_paths[i]) for i in val_idx]
    return train, val, ck


# ---------------------------------------------------------------------------
# 2-3. Pool nao rotulado com tampao espacial + pseudo-rotulos do teacher
# ---------------------------------------------------------------------------
def unlabeled_pool(val_tiles, buffer_px, feature):
    idx = json.load(open(Config.TILES_DIR / "2016" / "tiles_index.json"))["tiles"]
    info = {t["filename"]: t for t in idx}
    ann_dir = Config.MASKS_DIR / "2016" / "annotations" / feature
    annotated = {p.stem.replace(f"_{feature}", "") + ".png" for p in ann_dir.glob("tile_*.png")}
    val_boxes = np.array([[info[Path(v).name]["x"], info[Path(v).name]["y"],
                           info[Path(v).name]["width"], info[Path(v).name]["height"]] for v in val_tiles])
    pool, excluded = [], 0
    for t in idx:
        if t["filename"] in annotated:
            continue
        x, y, w, h = t["x"], t["y"], t["width"], t["height"]
        # distancia entre retangulos (0 se sobrepoem)
        dx = np.maximum(0, np.maximum(val_boxes[:, 0] - (x + w), x - (val_boxes[:, 0] + val_boxes[:, 2])))
        dy = np.maximum(0, np.maximum(val_boxes[:, 1] - (y + h), y - (val_boxes[:, 1] + val_boxes[:, 3])))
        if (np.sqrt(dx ** 2 + dy ** 2) < buffer_px).any():
            excluded += 1
            continue
        p = Config.TILES_DIR / "2016" / t["filename"]
        if p.exists():
            pool.append((p, t))
    print(f"  pool nao rotulado: {len(pool)} tiles (excluidos por tampao de {buffer_px}px: {excluded}; "
          f"anotados: {len(annotated)})")
    return pool, info


@torch.no_grad()
def make_pseudo(args, teacher, pool, data16, out_dir):
    """Teacher + TTA -> pseudo-rotulo duro (0/255) com faixa de ignorar (128). Salva PNG + stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    t0 = time.time()
    for i, (p, tinfo) in enumerate(pool):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        dem = data16.dem_for(tinfo)
        prob = EV.predict_probs([teacher], img, dem, tta=True)
        pos = prob >= args.tau_pos
        neg = prob <= args.tau_neg
        pseudo = np.full(prob.shape, IGNORE, dtype=np.uint8)
        pseudo[neg] = 0
        pseudo[pos] = 255
        fpos, fabove = float(pos.mean()), float((prob > args.tau_neg).mean())
        stats.append({"tile": p.name, "frac_pos": fpos, "frac_above_neg": fabove,
                      "frac_ignore": float(1 - pos.mean() - neg.mean()), "max_prob": float(prob.max())})
        cv2.imwrite(str(out_dir / p.name), pseudo)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(pool)} ({time.time()-t0:.0f}s)")
    json.dump(stats, open(out_dir / "stats.json", "w"), indent=1)
    return stats


def select_pseudo(stats, args):
    pos = [s for s in stats if args.min_pos_frac <= s["frac_pos"] <= args.max_pos_frac]
    neg = [s for s in stats if s["frac_above_neg"] <= args.max_neg_frac and s["frac_pos"] == 0]
    rng = random.Random(args.seed)
    rng.shuffle(pos); rng.shuffle(neg)
    pos = pos[:args.max_pseudo_pos]
    n_neg = min(len(neg), int(round(len(pos) * 0.45 / 0.55)))
    neg = neg[:n_neg]
    print(f"  pseudo selecionados: {len(pos)} pos (0.5-15% de area) + {len(neg)} neg (<=1% acima de tau_neg)"
          f" = {len(pos)+len(neg)} | ignorados em media {np.mean([s['frac_ignore'] for s in pos])*100:.1f}% dos px nos pos")
    return [s["tile"] for s in pos] + [s["tile"] for s in neg]


# ---------------------------------------------------------------------------
# Dados / aug forte
# ---------------------------------------------------------------------------
class Year2016:
    def __init__(self, dem_features, dem_window):
        idx = json.load(open(Config.TILES_DIR / "2016" / "tiles_index.json"))["tiles"]
        self.info = {t["filename"]: t for t in idx}
        self.dem_features = dem_features
        self.stack, self.transform, _, self.stats = EV.precompute_year_dem_features(
            2016, window_meters=dem_window, features=tuple(dem_features))

    def dem_for(self, tinfo):
        feats = EV.get_dem_features_for_tile(tinfo, self.stack, self.transform)
        return EV.normalize_dem_features(feats, self.stats, self.dem_features)

    def dem_for_name(self, fname):
        return self.dem_for(self.info[fname])


def geom_aug(rng, arrays):
    """Flip/rot90 aleatorios aplicados igualmente a todos os arrays (H,W[,C])."""
    k = rng.randint(0, 3); fl = rng.random() < 0.5; fu = rng.random() < 0.5
    out = []
    for a in arrays:
        b = np.rot90(a, k)
        if fl:
            b = np.fliplr(b)
        if fu:
            b = np.flipud(b)
        out.append(np.ascontiguousarray(b))
    return out


def strong_photometric(rng, img_u8):
    """Brilho/contraste/gama no RGB (componente mais forte segundo DPMS)."""
    img = img_u8.astype(np.float32) / 255.0
    c = rng.uniform(0.7, 1.3); b = rng.uniform(-0.15, 0.15); g = rng.uniform(0.7, 1.4)
    img = np.clip((img - 0.5) * c + 0.5 + b, 0, 1) ** g
    if rng.random() < 0.3:  # dessaturacao leve
        gray = img.mean(axis=2, keepdims=True); a = rng.uniform(0.5, 1.0)
        img = a * img + (1 - a) * gray
    return (img * 255).astype(np.uint8)


def dem_aug(rng, dem):
    gain = rng.uniform(0.8, 1.2); off = rng.gauss(0, 0.1)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 0.05, dem.shape).astype(np.float32)
    return (dem * gain + off + noise).astype(np.float32)


def cutmix(rng, a, b):
    """Caixa ~50% da area de b colada em a; aplicado a TODOS os canais e mascaras."""
    H, W = a[0].shape[:2]
    lam = rng.uniform(0.35, 0.65)
    bh, bw = int(H * np.sqrt(lam)), int(W * np.sqrt(lam))
    y0, x0 = rng.randint(0, H - bh), rng.randint(0, W - bw)
    out = []
    for xa, xb in zip(a, b):
        xa = xa.copy(); xa[y0:y0 + bh, x0:x0 + bw] = xb[y0:y0 + bh, x0:x0 + bw]
        out.append(xa)
    return out


class TileSet(Dataset):
    """Rotulados (mask 0/1, sem ignorar) ou pseudo (0/1 + 128=ignorar). Devolve
    (x[6,H,W], y[1,H,W] float 0/1, valid[1,H,W] float)."""

    def __init__(self, items, data16, strong=True, cutmix_p=0.5, seed=0, tile_size=512):
        self.items = items  # lista de (tile_path, mask_path|None, is_pseudo)
        self.d = data16; self.strong = strong; self.cutmix_p = cutmix_p
        self.rng = random.Random(seed); self.tile_size = tile_size

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        tp, mp, pseudo = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(tp)), cv2.COLOR_BGR2RGB)
        dem = self.d.dem_for_name(tp.name)
        if mp is None:
            m = np.zeros(img.shape[:2], np.uint8)
        else:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if pseudo:
            valid = (m != IGNORE).astype(np.float32); y = (m > 200).astype(np.float32)
        else:
            valid = np.ones(img.shape[:2], np.float32); y = (m > 127).astype(np.float32)
        if dem.shape[:2] != img.shape[:2]:
            dem = cv2.resize(dem, (img.shape[1], img.shape[0]))
        return img, dem.astype(np.float32), y, valid

    def __getitem__(self, i):
        img, dem, y, valid = self._load(i)
        if self.strong:
            img = strong_photometric(self.rng, img); dem = dem_aug(self.rng, dem)
        img, dem, y, valid = geom_aug(self.rng, [img, dem, y, valid])
        if self.strong and self.rng.random() < self.cutmix_p:
            j = self.rng.randrange(len(self.items))
            img2, dem2, y2, valid2 = self._load(j)
            img2 = strong_photometric(self.rng, img2); dem2 = dem_aug(self.rng, dem2)
            img2, dem2, y2, valid2 = geom_aug(self.rng, [img2, dem2, y2, valid2])
            img, dem, y, valid = cutmix(self.rng, [img, dem, y, valid], [img2, dem2, y2, valid2])
        x = EV.to_tensor(img, dem, self.tile_size)
        return x, torch.from_numpy(y)[None], torch.from_numpy(valid)[None]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def pseudo_loss(logits, y, valid):
    """BCE + soft-Dice so em pixels validos."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid; t = y * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return 0.5 * bce + 0.5 * dice


# ---------------------------------------------------------------------------
# Treino de um braco
# ---------------------------------------------------------------------------
def train_arm(args, train, val, pseudo_items, data16, in_channels):
    CV.set_seed(args.seed)
    if args.labeled_aug == "standard":
        # Pipeline de PRODUCAO para os rotulados (identico ao 10_crossval_curves.build_loaders):
        # albumentations geom + cromatica calibrada + copy-paste; sem CutMix. Assim o controle
        # pareado e o proprio fold do CV (mesmo split, mesma seed, mesma aug).
        from shadow_utils import build_dem_provider
        dem_provider, dem_names = build_dem_provider([2016], feature_names=tuple(data16.dem_features),
                                                     window_meters=3.0)
        geom_t, chrom_t = TU.get_train_transforms_split(n_dem_channels=len(dem_names))
        lab_ds = TU.GlacierDataset([t for t, _ in train], [m for _, m in train], transform=geom_t,
                                   img_size=512, copy_paste=True, copy_paste_prob=0.5,
                                   dem_provider=dem_provider, dem_chrom_transform=chrom_t)
        lab_loader = DataLoader(lab_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True, pin_memory=True)
    else:
        lab = TileSet([(t, m, False) for t, m in train], data16, strong=True, cutmix_p=args.cutmix_p, seed=args.seed)
        lab_loader = DataLoader(lab, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                drop_last=True, pin_memory=True)
    pseudo_loader = None
    if args.arm == "semi":
        ps = TileSet(pseudo_items, data16, strong=True, cutmix_p=args.cutmix_p, seed=args.seed + 1)
        pseudo_loader = DataLoader(ps, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   drop_last=True, pin_memory=True)
    model, optimizer, scheduler, warmup = CV.build_model_and_optim(in_channels, args.lr, args.epochs)
    crit_sup = TU.BCETverskyLoss(bce_weight=0.3, tversky_weight=0.7, fp_weight=0.7, fn_weight=0.3)
    val_items = [(str(t), m) for t, m in val]

    def eval_val(m):
        m.eval(); tp = fp = fn = 0
        with torch.no_grad():
            for t, mp in val_items:
                img = cv2.cvtColor(cv2.imread(t), cv2.COLOR_BGR2RGB); dem = data16.dem_for_name(Path(t).name)
                gt = np.zeros(img.shape[:2], bool) if mp is None else cv2.imread(str(mp), 0) > 127
                pred = EV.predict_probs([m], img, dem, tta=False) > 0.5
                tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        return (2 * p * r / (p + r) if p + r else 0), p, r

    best = (-1, 0, None); patience = 0; t0 = time.time()
    pseudo_iter = iter(pseudo_loader) if pseudo_loader else None
    pos_frac_log = []
    start_epoch = 1
    resume_path = Config.RESULTS_DIR / "selftrain" / f"{args.tag_out}.resume.pth"
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        best = (st["best_f1"], st["best_epoch"], st["best_state"]); patience = st["patience"]
        start_epoch = st["epoch"] + 1
        print(f"  [RESUME] retomando da epoca {start_epoch} (best ep {best[1]}, f1={best[0]:.4f}, patience={patience})")
        if st.get("finished"):
            print("  [RESUME] treino ja concluido anteriormente"); model.load_state_dict(best[2]); return model, best[0], best[1]
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= warmup:
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = args.lr * epoch / warmup * (0.1 if i == 0 else 1.0)
        else:
            scheduler.step()
        model.train(); n = 0; ls = lu = 0.0
        for batch in lab_loader:
            x, y = batch[0], batch[1]
            # gradientes ACUMULADOS batch a batch (memoria de 1 batch; GTX 1650 4 GB)
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit_sup(model(x), y); ls += loss.item()
            ok = bool(torch.isfinite(loss))
            if ok:
                loss.backward()
            if pseudo_loader is not None:
                for _ in range(args.ratio):
                    try:
                        xu, yu, vu = next(pseudo_iter)
                    except StopIteration:
                        pseudo_iter = iter(pseudo_loader); xu, yu, vu = next(pseudo_iter)
                    xu, yu, vu = xu.to(DEVICE), yu.to(DEVICE), vu.to(DEVICE)
                    lp = args.lambda_u * pseudo_loss(model(xu), yu, vu) / args.ratio
                    if torch.isfinite(lp):
                        lp.backward(); lu += lp.item()
                    pos_frac_log.append(float(((yu > 0.5) & (vu > 0.5)).float().mean()))
                    del xu, yu, vu, lp
            if ok:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            n += 1
        f1, p, r = eval_val(model)
        if epoch % args.log_every == 0 or epoch == 1:
            extra = f" | Lu={lu/max(n,1):.4f} pos_frac_pseudo={np.mean(pos_frac_log[-n*args.ratio:]) if pos_frac_log else 0:.4f}" if pseudo_loader else ""
            print(f"  ep {epoch:03d}: val_f1={f1:.4f} P={p:.3f} R={r:.3f} | Ls={ls/max(n,1):.4f}{extra} | {(time.time()-t0)/60:.1f} min")
        stop = False
        if f1 > best[0] + Config.EARLY_STOPPING_MIN_DELTA:
            best = (f1, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop ep {epoch} (best ep {best[1]}, f1={best[0]:.4f})"); stop = True
        # estado de retomada (janelas curtas de GPU: o processo pode ser morto a qualquer momento)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_f1": best[0],
                    "best_epoch": best[1], "best_state": best[2], "patience": patience,
                    "finished": stop or epoch == args.epochs}, resume_path)
        if stop:
            break
    model.load_state_dict(best[2])
    return model, best[0], best[1]


def save_val_probs(args, model, val, data16, tag_out):
    out = Config.RESULTS_DIR / "oof" / tag_out
    for v in ("single", "tta"):
        (out / f"probs_{v}").mkdir(parents=True, exist_ok=True)
    fold_of = {}
    items = {"single": [], "tta": []}
    for t, mp in val:
        img = cv2.cvtColor(cv2.imread(str(t)), cv2.COLOR_BGR2RGB); dem = data16.dem_for_name(t.name)
        gt = (np.zeros(img.shape[:2], np.uint8) if mp is None else (cv2.imread(str(mp), 0) > 127).astype(np.uint8))
        for v in ("single", "tta"):
            p = EV.predict_probs([model], img, dem, tta=(v == "tta"))
            np.save(out / f"probs_{v}" / t.name.replace(".png", ".npy"), p.astype(np.float16))
            items[v].append((t.name, p, gt))
        fold_of[t.name] = args.fold
    json.dump(fold_of, open(out / "fold_of_tile.json", "w"), indent=1)
    res = {}
    for v in ("single", "tta"):
        rows = EV.evaluate_set(items[v], thresholds=[0.4, 0.5, 0.6], skeleton=True)
        res[v] = rows
        r5 = [r for r in rows if abs(r["threshold"] - 0.5) < 1e-9][0]
        print(f"  VAL fold {args.fold} [{v}] @0.5: P={r5['precision']:.4f} R={r5['recall']:.4f} F1={r5['f1']:.4f} "
              f"clIoU@4={r5['skeleton']['tau4']['cliou']:.4f} clIoUomni@4={r5['skeleton']['tau4']['cliou_omnicrack']:.4f} "
              f"F1rel@4={r5['skeleton']['tau4']['f1_relaxed']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True); ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--arm", choices=["semi", "ctrl"], default="semi")
    ap.add_argument("--stage", choices=["pseudo", "train", "both"], default="both")
    ap.add_argument("--feature", default="crevasses")
    ap.add_argument("--neg-ratio", type=float, default=0.515); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=2048); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer-px", type=int, default=1024)
    ap.add_argument("--tau-pos", type=float, default=0.7); ap.add_argument("--tau-neg", type=float, default=0.3)
    ap.add_argument("--min-pos-frac", type=float, default=0.005); ap.add_argument("--max-pos-frac", type=float, default=0.15)
    ap.add_argument("--max-neg-frac", type=float, default=0.01); ap.add_argument("--max-pseudo-pos", type=int, default=1500)
    ap.add_argument("--ratio", type=int, default=3, help="batches pseudo por batch rotulado")
    ap.add_argument("--lambda-u", type=float, default=1.0); ap.add_argument("--cutmix-p", type=float, default=0.5)
    ap.add_argument("--labeled-aug", choices=["strong", "standard"], default="strong",
                    help="strong = aug forte + CutMix tambem nos rotulados (controle = arm ctrl); "
                         "standard = pipeline de producao nos rotulados (controle = o proprio fold do CV)")
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2); ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-split-check", action="store_true",
                    help="nao aborta se o split reconstruido diferir do checkpoint (usa val do ckpt)")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    if args.smoke:
        args.epochs, args.patience, args.max_pseudo_pos, args.log_every = 2, 2, 12, 1

    tag_out = (f"{args.tag}_pilot{args.fold}_{args.arm}" + ("_std" if args.labeled_aug == "standard" else "")
               + ("_smoke" if args.smoke else ""))
    args.tag_out = tag_out
    print(f"=== PILOTO SELF-TRAINING — {tag_out} ===")
    train, val, ck = rebuild_fold(args)
    dem_features = ck["dem_features"]; in_channels = ck["in_channels"]
    data16 = Year2016(dem_features, ck.get("dem_window_meters", 3.0))

    pseudo_dir = Config.RESULTS_DIR / "selftrain" / f"{args.tag}_fold{args.fold}" / "pseudo"
    pseudo_items = []
    if args.arm == "semi":
        if args.stage in ("pseudo", "both") or not (pseudo_dir / "stats.json").exists():
            teacher = TU.UNetResNet34(pretrained=False, in_channels=in_channels)
            teacher.load_state_dict(ck["state_dict"]); teacher.to(DEVICE).eval()
            pool, _ = unlabeled_pool([t for t, _ in val], args.buffer_px, args.feature)
            if args.smoke:
                pool = pool[:60]
            print(f"  gerando pseudo-rotulos (teacher fold {args.fold} + TTA, tau_pos={args.tau_pos}, tau_neg={args.tau_neg})...")
            stats = make_pseudo(args, teacher, pool, data16, pseudo_dir)
            del teacher; torch.cuda.empty_cache()
        else:
            stats = json.load(open(pseudo_dir / "stats.json"))
            print(f"  pseudo-rotulos existentes: {len(stats)} tiles em {pseudo_dir}")
        if args.stage == "pseudo":
            select_pseudo(stats, args); return
        names = select_pseudo(stats, args)
        pseudo_items = [(Config.TILES_DIR / "2016" / n, pseudo_dir / n, True) for n in names]

    t0 = time.time()
    model, best_f1, best_ep = train_arm(args, train, val, pseudo_items, data16, in_channels)
    print(f"  >> {args.arm}: best ep {best_ep} val_f1={best_f1:.4f} ({(time.time()-t0)/60:.1f} min)")
    ck_dir = Config.MODELS_DIR / "selftrain"; ck_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_channels": in_channels, "dem_features": dem_features,
                "fold": args.fold, "arm": args.arm, "best_epoch": best_ep, "val_f1": best_f1,
                "val_tiles": [str(t) for t, _ in val], "args": vars(args)}, ck_dir / f"{tag_out}.pth")
    res = save_val_probs(args, model, val, data16, tag_out)
    json.dump({"tag_out": tag_out, "args": vars(args), "best_epoch": best_ep, "val_f1": best_f1,
               "n_train": len(train), "n_val": len(val), "n_pseudo": len(pseudo_items), "val_metrics": res},
              open(Config.RESULTS_DIR / "selftrain" / f"{tag_out}.json", "w"), indent=1, default=float)
    print(f"  salvo: models/selftrain/{tag_out}.pth | results/selftrain/{tag_out}.json | results/oof/{tag_out}/")
    (Config.RESULTS_DIR / "selftrain" / f"{tag_out}.DONE").write_text(time.strftime("%F %T"))


if __name__ == "__main__":
    main()

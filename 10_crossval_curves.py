"""
10_crossval_curves.py - Cross-validation espacial + curvas PR/ROC/F1 para crevasses 2016

Motivacao (orientacao do coorientador, jun/2026):
  1. "Achar o melhor modelo treinando com 2016" de forma estatisticamente robusta
     -> validacao cruzada (K-fold) em vez de um unico split 80/20 com val set minusculo.
  2. "Curva ROC / cross-validation" -> ROC + Precision-Recall + F1 x threshold (OOF).
  3. "Verificar se o modelo esta balanceado" -> diagnostico P vs R e razao FN/FP.
  4. "Penalizar fendas nao achadas (subestimacao)" -> comparar Tversky fp/fn = 0.7/0.3
     (favorece precisao, padrao atual) vs 0.3/0.7 (favorece recall = pega mais fenda).

Notas metodologicas:
  - Segmentacao binaria desbalanceada (~1% de pixels positivos): a metrica de selecao
    correta NAO e RMSE (premia subestimacao do fundo). Usa-se F1/IoU + AUC-PR.
  - RMSE faz sentido na etapa de ablacao/dH (regressao em metros), nao aqui.
  - CV ESPACIAL EM BLOCOS COM BUFFER (Roberts et al. 2017): tiles vizinhos tem overlap
    de 64 px e vem do mesmo mosaico -> sao quase-duplicatas. K-fold aleatorio vazaria
    informacao (tile de treino quase igual a tile de val) e inflaria a metrica.
    Agrupamos tiles em blocos espaciais grandes (px); um bloco nunca aparece em treino e
    validacao ao mesmo tempo. Como blocos vizinhos ainda podem ter tiles que se sobrepoem
    na fronteira, aplicamos um BUFFER: todo tile de treino cujo bounding box (512 px)
    sobrepoe qualquer tile de validacao e REMOVIDO do treino daquele fold. Isso elimina
    (nao so reduz) o vazamento por overlap. O nº de tiles removidos por fold e logado.

Reutiliza TODO o pipeline de 03_train_unet.py (collect_pairs, UNetResNet34,
GlacierDataset, losses, transforms, build_dem_provider) via importlib.

Uso:
    # CV espacial 5-fold, config de producao (109 pos, 6-canal, Tversky 0.7/0.3)
    python 10_crossval_curves.py --feature crevasses --years 2016 \
        --use-dem-channels --loss bce_tversky --fp-weight 0.7 --fn-weight 0.3 \
        --folds 5 --epochs 200 --patience 60

    # Experimento de balanceamento: favorecer recall (penaliza fenda perdida)
    python 10_crossval_curves.py --feature crevasses --years 2016 \
        --use-dem-channels --loss bce_tversky --fp-weight 0.3 --fn-weight 0.7 \
        --folds 5 --epochs 200 --patience 60

    # Rodar as duas e comparar automaticamente
    python 10_crossval_curves.py --feature crevasses --years 2016 \
        --use-dem-channels --folds 5 --epochs 200 --patience 60 --compare-weights

    # Teste rapido end-to-end (valida o script em ~2 min)
    python 10_crossval_curves.py --feature crevasses --years 2016 --use-dem-channels --smoke
"""

import argparse
import copy
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config


# ---------------------------------------------------------------------------
# Carregar 03_train_unet.py como modulo (nome comeca com digito -> importlib)
# ---------------------------------------------------------------------------
def _load_train_module():
    path = Path(__file__).resolve().parent / "03_train_unet.py"
    spec = importlib.util.spec_from_file_location("train_unet", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TU = _load_train_module()


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Coordenadas espaciais dos tiles (do tiles_index.json) para blocagem
# ---------------------------------------------------------------------------
def load_tile_coords(years):
    """Mapeia (ano, filename) -> (x, y) em pixels do mosaico original."""
    coords = {}
    for year in years:
        idx = Config.TILES_DIR / str(year) / "tiles_index.json"
        if not idx.exists():
            continue
        with open(idx) as f:
            d = json.load(f)
        for t in d.get("tiles", []):
            coords[(int(year), t["filename"])] = (float(t["x"]), float(t["y"]))
    return coords


def assign_spatial_groups(tile_paths, coords, block_size):
    """Cada tile -> (id de bloco espacial, coord (x,y)). Tiles sem coord ganham grupo proprio.

    Returns:
        groups: lista de ids de bloco (gx, gy) ou ("u", n) para tiles sem coord.
        xy: lista de (x, y) em px ou None (para o buffer de overlap).
        n_unknown: nº de tiles sem coordenada.
    """
    groups, xy, unknown = [], [], 0
    for p in tile_paths:
        try:
            key = (int(p.parent.name), p.name)
        except ValueError:
            key = (None, p.name)
        if key in coords:
            x, y = coords[key]
            groups.append((int(x // block_size), int(y // block_size)))
            xy.append((x, y))
        else:
            groups.append(("u", unknown))
            xy.append(None)
            unknown += 1
    return groups, xy, unknown


def buffered_train_idx(train_idx, val_idx, xy, tile_size):
    """Remove do TREINO tiles cujo bbox (tile_size px) sobrepoe qualquer tile de val.

    Dois tiles (x1,y1),(x2,y2) de lado tile_size se sobrepoem sse
    |x1-x2| < tile_size E |y1-y2| < tile_size. Elimina o vazamento por overlap de 64 px
    na fronteira entre blocos de folds diferentes (buffered block CV, Roberts et al. 2017).
    """
    val_pts = np.array([xy[i] for i in val_idx if xy[i] is not None], dtype=float)
    if len(val_pts) == 0:
        return list(train_idx), 0
    keep, removed = [], 0
    for i in train_idx:
        p = xy[i]
        if p is None:
            keep.append(i)
            continue
        overlap = (np.abs(val_pts[:, 0] - p[0]) < tile_size) & \
                  (np.abs(val_pts[:, 1] - p[1]) < tile_size)
        if overlap.any():
            removed += 1
        else:
            keep.append(i)
    return keep, removed


def grouped_kfold_balanced(groups, is_pos, n_folds, seed=42):
    """K-fold por grupo (bloco) em DUAS PASSADAS: balanceia POSITIVOS e TAMANHO.

    Garante que um mesmo bloco nunca aparece em treino e validacao (sem vazamento
    espacial). Passada 1: distribui blocos COM positivos minimizando positivos por fold.
    Passada 2: distribui blocos puramente negativos minimizando o tamanho total por fold
    (evita despejar todos os negativos num unico fold).
    """
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    def g_pos(g):
        return sum(is_pos[i] for i in members[g])

    def g_size(g):
        return len(members[g])

    glist = list(members.keys())
    rng = random.Random(seed)
    rng.shuffle(glist)  # quebra empates de forma deterministica

    pos_blocks = sorted([g for g in glist if g_pos(g) > 0],
                        key=lambda g: (g_pos(g), g_size(g)), reverse=True)
    neg_blocks = sorted([g for g in glist if g_pos(g) == 0],
                        key=g_size, reverse=True)

    fold_pos = [0] * n_folds
    fold_size = [0] * n_folds
    fold_groups = [[] for _ in range(n_folds)]

    # Passada 1: blocos com positivos -> equaliza positivos (desempate por tamanho)
    for g in pos_blocks:
        f = min(range(n_folds), key=lambda k: (fold_pos[k], fold_size[k]))
        fold_groups[f].append(g)
        fold_pos[f] += g_pos(g)
        fold_size[f] += g_size(g)

    # Passada 2: blocos so-negativos -> equaliza tamanho total (desempate por positivos)
    for g in neg_blocks:
        f = min(range(n_folds), key=lambda k: (fold_size[k], fold_pos[k]))
        fold_groups[f].append(g)
        fold_size[f] += g_size(g)

    return [sorted(i for g in fold_groups[f] for i in members[g])
            for f in range(n_folds)]


# ---------------------------------------------------------------------------
# Construcao de loaders/modelo/otimizador (replicando 03_train_unet.py)
# ---------------------------------------------------------------------------
def build_loaders(train_tiles, train_masks, val_tiles, val_masks, args,
                  dem_provider, dem_feature_names):
    use_dem = dem_provider is not None
    if not args.no_augment:
        if use_dem:
            geom_t, chrom_t = TU.get_train_transforms_split(
                n_dem_channels=len(dem_feature_names))
        else:
            geom_t, chrom_t = TU.get_train_transform(), None
    else:
        geom_t, chrom_t = None, None

    train_ds = TU.GlacierDataset(
        train_tiles, train_masks, transform=geom_t, img_size=args.img_size,
        copy_paste=not args.no_augment, copy_paste_prob=0.5,
        dem_provider=dem_provider, dem_chrom_transform=chrom_t)
    val_ds = TU.GlacierDataset(
        val_tiles, val_masks, transform=None, img_size=args.img_size,
        copy_paste=False, dem_provider=dem_provider, dem_chrom_transform=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


def build_model_and_optim(in_channels, lr, epochs):
    model = TU.UNetResNet34(pretrained=True, in_channels=in_channels).to(Config.DEVICE)
    enc_pref = ("enc0.", "enc1.", "enc2.", "enc3.", "enc4.", "pool0.")
    enc_params = [p for n, p in model.named_parameters()
                  if any(n.startswith(pf) for pf in enc_pref) and p.requires_grad]
    dec_params = [p for n, p in model.named_parameters()
                  if not any(n.startswith(pf) for pf in enc_pref) and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr * 0.1}, {"params": dec_params, "lr": lr}],
        weight_decay=1e-4)
    warmup = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup), eta_min=lr * 0.01)
    return model, optimizer, scheduler, warmup


def make_criterion(args):
    if args.loss == "bce_tversky":
        return TU.BCETverskyLoss(bce_weight=0.3, tversky_weight=0.7,
                                 fp_weight=args.fp_weight, fn_weight=args.fn_weight)
    return TU.BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)


# ---------------------------------------------------------------------------
# Treino de um fold + acumulo de histograma OOF
# ---------------------------------------------------------------------------
N_BINS = 2000  # resolucao do histograma de probabilidades para as curvas


@torch.no_grad()
def eval_fold(model, val_loader):
    """Roda a validacao acumulando: tp/fp/fn @0.5 (micro) + histogramas pos/neg."""
    model.eval()
    device = Config.DEVICE
    hist_pos = torch.zeros(N_BINS, device=device)
    hist_neg = torch.zeros(N_BINS, device=device)
    tp = fp = fn = 0.0
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        probs = torch.sigmoid(model(images))
        preds = (probs > 0.5).float()
        tp += (preds * masks).sum().item()
        fp += (preds * (1 - masks)).sum().item()
        fn += ((1 - preds) * masks).sum().item()
        pos_p = probs[masks > 0.5]
        neg_p = probs[masks <= 0.5]
        if pos_p.numel():
            hist_pos += torch.histc(pos_p, bins=N_BINS, min=0.0, max=1.0)
        if neg_p.numel():
            hist_neg += torch.histc(neg_p, bins=N_BINS, min=0.0, max=1.0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    metrics = {"precision": prec, "recall": rec, "f1": f1, "iou": iou,
               "tp": tp, "fp": fp, "fn": fn}
    return metrics, hist_pos.cpu().numpy(), hist_neg.cpu().numpy()


def train_one_fold(fold_idx, n_folds, train_tiles, train_masks, val_tiles, val_masks,
                   args, dem_provider, dem_feature_names, in_channels):
    set_seed(args.seed + fold_idx)
    train_loader, val_loader = build_loaders(
        train_tiles, train_masks, val_tiles, val_masks, args,
        dem_provider, dem_feature_names)
    model, optimizer, scheduler, warmup = build_model_and_optim(
        in_channels, args.lr, args.epochs)
    criterion = make_criterion(args)

    n_train_pos = sum(1 for m in train_masks if m is not None)
    n_val_pos = sum(1 for m in val_masks if m is not None)
    print(f"\n{'-'*60}\nFOLD {fold_idx+1}/{n_folds}  "
          f"treino={len(train_tiles)} ({n_train_pos} pos) | "
          f"val={len(val_tiles)} ({n_val_pos} pos)\n{'-'*60}")

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    device = Config.DEVICE

    for epoch in range(1, args.epochs + 1):
        if epoch <= warmup:
            wlr = args.lr * epoch / warmup
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = wlr * (0.1 if i == 0 else 1.0)
        else:
            scheduler.step()

        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Validacao @0.5 para selecao da melhor epoca (mesmo criterio da producao)
        metrics, _, _ = eval_fold(model, val_loader)
        if (epoch % args.log_every == 0) or epoch == 1:
            print(f"  ep {epoch:03d}: val_f1={metrics['f1']:.4f} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"IoU={metrics['iou']:.4f}")

        if metrics["f1"] > best_f1 + Config.EARLY_STOPPING_MIN_DELTA:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  early stop ep {epoch} (best ep {best_epoch}, f1={best_f1:.4f})")
                break

    # Reavalia com o MELHOR estado para preencher histograma OOF + metricas finais
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, hist_pos, hist_neg = eval_fold(model, val_loader)
    metrics.update({"fold": fold_idx + 1, "best_epoch": best_epoch,
                    "n_train": len(train_tiles), "n_val": len(val_tiles),
                    "n_val_pos": n_val_pos})
    print(f"  >> FOLD {fold_idx+1} (best ep {best_epoch}): "
          f"F1={metrics['f1']:.4f} IoU={metrics['iou']:.4f} "
          f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, hist_pos, hist_neg


# ---------------------------------------------------------------------------
# Curvas OOF a partir dos histogramas agregados
# ---------------------------------------------------------------------------
def curves_from_hist(hist_pos, hist_neg):
    """Calcula PR, ROC e F1xthreshold exatos a partir dos histogramas de prob.

    bin i cobre [i/N, (i+1)/N). Predicao positiva se prob > threshold.
    """
    N = len(hist_pos)
    centers = (np.arange(N) + 0.5) / N
    total_pos = hist_pos.sum()
    total_neg = hist_neg.sum()

    # cumsum reverso: contagem de bins com indice >= i (i.e., prob acima do limiar)
    pos_ge = np.cumsum(hist_pos[::-1])[::-1]   # pos_ge[i] = sum(hist_pos[i:])
    neg_ge = np.cumsum(hist_neg[::-1])[::-1]

    # thresholds = bordas inferiores dos bins; pred positivo se prob >= edge[i]
    thr = np.arange(N) / N
    tp = pos_ge
    fp = neg_ge
    fn = total_pos - tp
    tn = total_neg - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 1.0)
        recall = np.where(total_pos > 0, tp / total_pos, 0.0)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
        tpr = recall
        fpr = np.where(total_neg > 0, fp / total_neg, 0.0)

    # AUC com ANCORAGEM dos extremos (senao a regiao de alta confianca prob->1, que
    # satura no bin do topo, fica fora da integral e subestima a AUC-PR de classe rara).
    # thr cresce -> recall/fpr decrescem; invertendo, ficam crescentes em [r_min..1].
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renomeou
    rec_asc, prec_asc = recall[::-1], precision[::-1]
    fpr_asc, tpr_asc = fpr[::-1], tpr[::-1]
    # PR: ancorar (recall=0, precision=1) -> recupera o canto de alta precisao
    rec_pr = np.concatenate([[0.0], rec_asc])
    prec_pr = np.concatenate([[1.0], prec_asc])
    auc_pr = float(_trapz(prec_pr, rec_pr))
    # ROC: ancorar (0,0) e (1,1)
    fpr_roc = np.concatenate([[0.0], fpr_asc, [1.0]])
    tpr_roc = np.concatenate([[0.0], tpr_asc, [1.0]])
    auc_roc = float(_trapz(tpr_roc, fpr_roc))

    best_i = int(np.argmax(f1))
    best = {
        "threshold": float(thr[best_i]),
        "f1": float(f1[best_i]),
        "precision": float(precision[best_i]),
        "recall": float(recall[best_i]),
    }
    # metricas no threshold padrao 0.5
    i50 = int(np.argmin(np.abs(thr - 0.5)))
    at_05 = {"threshold": 0.5, "f1": float(f1[i50]),
             "precision": float(precision[i50]), "recall": float(recall[i50])}

    return {
        "thr": thr, "precision": precision, "recall": recall, "f1": f1,
        "tpr": tpr, "fpr": fpr, "auc_roc": auc_roc, "auc_pr": auc_pr,
        "best": best, "at_0.5": at_05,
        "total_pos": float(total_pos), "total_neg": float(total_neg),
    }


def balance_diagnosis(per_fold, curves):
    """Diagnostico de balanceamento do modelo e dos dados."""
    pos_frac = curves["total_pos"] / (curves["total_pos"] + curves["total_neg"] + 1e-9)
    p = curves["at_0.5"]["precision"]
    r = curves["at_0.5"]["recall"]
    # razao FN/FP agregada @0.5 (>1 = perde mais do que inventa = subestima)
    sum_fp = sum(f["fp"] for f in per_fold)
    sum_fn = sum(f["fn"] for f in per_fold)
    fn_fp = sum_fn / sum_fp if sum_fp > 0 else float("inf")
    if r > p + 0.03:
        verdict = ("Enviesado para RECALL: super-segmenta (mais falso-positivo que "
                   "falso-negativo). NAO esta subestimando no agregado.")
    elif p > r + 0.03:
        verdict = ("Enviesado para PRECISAO: subestima (perde fendas, mais "
                   "falso-negativo que falso-positivo).")
    else:
        verdict = "Balanceado (P ~ R)."
    return {
        "pixel_pos_fraction": float(pos_frac),
        "precision_at_0.5": float(p),
        "recall_at_0.5": float(r),
        "fn_over_fp_at_0.5": float(fn_fp),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_curves(curves, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # PR
    axes[0].plot(curves["recall"], curves["precision"], color="#1f77b4")
    axes[0].axhline(curves["total_pos"] / (curves["total_pos"] + curves["total_neg"]),
                    ls="--", color="grey", lw=1, label="baseline (prevalencia)")
    axes[0].scatter([curves["best"]["recall"]], [curves["best"]["precision"]],
                    color="red", zorder=5,
                    label=f"melhor F1={curves['best']['f1']:.3f}@thr={curves['best']['threshold']:.2f}")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall (AUC-PR={curves['auc_pr']:.3f})")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(alpha=0.3)

    # ROC
    axes[1].plot(curves["fpr"], curves["tpr"], color="#2ca02c")
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR (Recall)")
    axes[1].set_title(f"ROC (AUC={curves['auc_roc']:.3f})\n[otimista p/ classe rara]")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3)

    # F1 x threshold
    axes[2].plot(curves["thr"], curves["f1"], color="#d62728", label="F1")
    axes[2].plot(curves["thr"], curves["precision"], color="#1f77b4", lw=1, alpha=0.7, label="P")
    axes[2].plot(curves["thr"], curves["recall"], color="#ff7f0e", lw=1, alpha=0.7, label="R")
    axes[2].axvline(curves["best"]["threshold"], ls="--", color="red", lw=1)
    axes[2].axvline(0.5, ls=":", color="grey", lw=1, label="thr=0.5")
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("Metrica")
    axes[2].set_title("F1 / P / R vs Threshold")
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    fig.suptitle(f"Cross-validation OOF — {tag}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / f"crossval_curves_{tag}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver de uma configuracao (um conjunto de pesos fp/fn)
# ---------------------------------------------------------------------------
def run_crossval(args):
    tag = f"{args.feature}_fp{args.fp_weight}_fn{args.fn_weight}"
    print(f"\n{'='*70}\nCROSS-VALIDATION ESPACIAL — {tag}\n{'='*70}")
    print(f"Device={Config.DEVICE} | folds={args.folds} | epochs={args.epochs} "
          f"| patience={args.patience} | block_size={args.block_size}px | "
          f"loss={args.loss} (fp={args.fp_weight}, fn={args.fn_weight})")

    set_seed(args.seed)
    years = args.years if args.years else [2016]

    # 1. Coleta de pares (reutiliza a logica exata de producao)
    tile_paths, mask_paths = TU.collect_pairs(
        args.feature, years=years, neg_ratio=args.neg_ratio, shadow_neg_ratio=0.5)
    if len(tile_paths) == 0:
        print("Nenhum dado encontrado."); return None

    # smoke / max-tiles: subamostra mantendo todos os positivos possiveis
    if args.max_tiles and len(tile_paths) > args.max_tiles:
        pos_idx = [i for i, m in enumerate(mask_paths) if m is not None]
        neg_idx = [i for i, m in enumerate(mask_paths) if m is None]
        rng = random.Random(args.seed)
        keep_pos = pos_idx[:args.max_tiles] if len(pos_idx) >= args.max_tiles else pos_idx
        n_neg = max(0, args.max_tiles - len(keep_pos))
        keep_neg = rng.sample(neg_idx, min(n_neg, len(neg_idx)))
        keep = sorted(keep_pos + keep_neg)
        tile_paths = [tile_paths[i] for i in keep]
        mask_paths = [mask_paths[i] for i in keep]
        print(f"  [max-tiles] subamostrado para {len(tile_paths)} tiles")

    is_pos = [1 if m is not None else 0 for m in mask_paths]
    n_pos = sum(is_pos)
    print(f"  Total: {len(tile_paths)} tiles ({n_pos} positivos, {len(tile_paths)-n_pos} negativos)")

    # 2. Grupos espaciais + folds
    coords = load_tile_coords(years)
    groups, xy, n_unknown = assign_spatial_groups(tile_paths, coords, args.block_size)
    n_blocks = len(set(groups))
    if n_unknown:
        print(f"  [AVISO] {n_unknown} tiles sem coordenada espacial -> sem protecao de "
              f"vazamento para eles (verifique tiles_index.json dos anos {years}).")
    folds = grouped_kfold_balanced(groups, is_pos, args.folds, seed=args.seed)

    fold_sizes = [len(fv) for fv in folds]
    fold_pos_counts = [sum(is_pos[i] for i in fv) for fv in folds]
    print(f"  Blocos espaciais: {n_blocks} | tiles por fold: {fold_sizes} | "
          f"positivos por fold: {fold_pos_counts}")

    # Diagnosticos de saude dos folds
    n_blocks_with_pos = sum(1 for g in set(groups)
                            if any(is_pos[i] for i, gg in enumerate(groups) if gg == g))
    if n_blocks_with_pos < args.folds:
        print(f"  [AVISO] so {n_blocks_with_pos} blocos tem positivos para {args.folds} "
              f"folds -> algum fold ficara sem positivos (recall indefinido).")
    empty_pos_folds = [f + 1 for f, c in enumerate(fold_pos_counts) if c == 0]
    if empty_pos_folds:
        print(f"  [AVISO] folds sem nenhum positivo de validacao: {empty_pos_folds} "
              f"(serao excluidos da media de F1/recall).")
    nonempty = [s for s in fold_sizes if s > 0]
    if nonempty and max(nonempty) / max(1, min(nonempty)) > 1.5:
        print(f"  [AVISO] folds desbalanceados em tamanho "
              f"(max/min = {max(nonempty)/min(nonempty):.1f}x).")

    if args.dry_run:
        all_idx = set(range(len(tile_paths)))
        for f in range(args.folds):
            val_idx = folds[f]
            if not val_idx:
                continue
            tr = sorted(all_idx - set(val_idx))
            _, n_removed = buffered_train_idx(tr, val_idx, xy, Config.TILE_SIZE)
            print(f"    fold {f+1}: val={len(val_idx)} ({fold_pos_counts[f]} pos) | "
                  f"treino={len(tr)-n_removed} (buffer removeu {n_removed} por overlap)")
        print("  [DRY-RUN] composicao dos folds OK; nenhum treino executado.")
        return None

    # 3. DEM provider (uma vez, compartilhado entre folds)
    dem_provider, dem_feature_names, in_channels = None, None, 3
    if args.use_dem_channels:
        from shadow_utils import build_dem_provider
        dem_provider, dem_feature_names = build_dem_provider(
            years, feature_names=tuple(args.dem_features), window_meters=args.dem_window)
        if dem_provider is None:
            print("  [DEM] indisponivel -> RGB-only"); args.use_dem_channels = False
        else:
            in_channels = 3 + len(dem_feature_names)
            print(f"  [DEM] in_channels={in_channels} ({','.join(dem_feature_names)})")

    # 4. Loop de folds
    t0 = time.time()
    per_fold = []
    hist_pos_total = np.zeros(N_BINS)
    hist_neg_total = np.zeros(N_BINS)
    all_idx = set(range(len(tile_paths)))
    removed_total = 0
    for f in range(args.folds):
        val_idx = folds[f]
        if not val_idx:
            print(f"  [fold {f+1}] vazio, pulado")
            continue
        if sum(is_pos[i] for i in val_idx) == 0:
            print(f"  [fold {f+1}] sem positivos de validacao, pulado")
            continue
        train_idx = sorted(all_idx - set(val_idx))
        # BUFFER: remove do treino tiles que se sobrepoem a tiles de val (anti-vazamento)
        train_idx, n_removed = buffered_train_idx(train_idx, val_idx, xy, Config.TILE_SIZE)
        removed_total += n_removed
        if n_removed:
            print(f"  [fold {f+1}] buffer: {n_removed} tiles de treino removidos "
                  f"(overlap com val)")
        m, hp, hn = train_one_fold(
            f, args.folds,
            [tile_paths[i] for i in train_idx], [mask_paths[i] for i in train_idx],
            [tile_paths[i] for i in val_idx], [mask_paths[i] for i in val_idx],
            args, dem_provider, dem_feature_names, in_channels)
        per_fold.append(m)
        hist_pos_total += hp
        hist_neg_total += hn

    if not per_fold:
        print("  [ERRO] nenhum fold valido (todos vazios ou sem positivos)."); return None
    n_effective_folds = len(per_fold)
    if n_effective_folds < args.folds:
        print(f"  [AVISO] CV efetiva: {n_effective_folds}/{args.folds} folds "
              f"(demais vazios/sem positivos).")

    # 5. Agregacao
    def agg(key):
        vals = [f[key] for f in per_fold]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))}

    cv_summary = {k: agg(k) for k in ("f1", "iou", "precision", "recall")}
    curves = curves_from_hist(hist_pos_total, hist_neg_total)
    balance = balance_diagnosis(per_fold, curves)
    elapsed = (time.time() - t0) / 60

    out_dir = Config.RESULTS_DIR / "crossval"
    fig_dir = Config.RESULTS_DIR / "figures" / "crossval"
    fig_path = plot_curves(curves, tag, fig_dir)

    result = {
        "tag": tag,
        "config": {
            "feature": args.feature, "years": years, "folds": args.folds,
            "epochs": args.epochs, "patience": args.patience,
            "block_size_px": args.block_size, "loss": args.loss,
            "fp_weight": args.fp_weight, "fn_weight": args.fn_weight,
            "neg_ratio": args.neg_ratio, "use_dem_channels": args.use_dem_channels,
            "in_channels": in_channels, "n_tiles": len(tile_paths), "n_positives": n_pos,
            "n_blocks": n_blocks, "n_effective_folds": n_effective_folds,
            "fold_sizes": fold_sizes, "fold_pos_counts": fold_pos_counts,
            "buffer_tiles_removed": removed_total, "n_unknown_coord": n_unknown,
            "elapsed_min": round(elapsed, 1),
        },
        "per_fold": [{k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in m.items()} for m in per_fold],
        "cv_summary_at_0.5": cv_summary,
        "oof_curves": {
            "auc_pr": curves["auc_pr"], "auc_roc": curves["auc_roc"],
            "best_threshold": curves["best"], "at_0.5": curves["at_0.5"],
        },
        "balance": balance,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"crossval_{tag}.json"
    with open(json_path, "w") as fp:
        json.dump(result, fp, indent=2)

    # 6. Resumo no terminal
    print(f"\n{'='*70}\nRESULTADO CV — {tag}  ({elapsed:.1f} min)\n{'='*70}")
    print(f"  F1   (CV @0.5): {cv_summary['f1']['mean']:.4f} ± {cv_summary['f1']['std']:.4f}  "
          f"[{cv_summary['f1']['min']:.3f}–{cv_summary['f1']['max']:.3f}]")
    print(f"  IoU  (CV @0.5): {cv_summary['iou']['mean']:.4f} ± {cv_summary['iou']['std']:.4f}")
    print(f"  P    (CV @0.5): {cv_summary['precision']['mean']:.4f} ± {cv_summary['precision']['std']:.4f}")
    print(f"  R    (CV @0.5): {cv_summary['recall']['mean']:.4f} ± {cv_summary['recall']['std']:.4f}")
    print(f"  AUC-PR (OOF): {curves['auc_pr']:.4f} | AUC-ROC (OOF): {curves['auc_roc']:.4f}")
    print(f"  Melhor threshold (max F1 OOF): {curves['best']['threshold']:.2f} "
          f"-> F1={curves['best']['f1']:.4f} (P={curves['best']['precision']:.3f} "
          f"R={curves['best']['recall']:.3f})  | hoje usa 0.5")
    print(f"  Balanceamento: {balance['verdict']}")
    print(f"    pixels positivos={balance['pixel_pos_fraction']*100:.2f}% | "
          f"FN/FP@0.5={balance['fn_over_fp_at_0.5']:.2f} "
          f"(>1 = perde mais do que inventa)")
    print(f"  JSON: {json_path}\n  Figura: {fig_path}")
    return result


def compare_results(res_a, res_b):
    if not res_a or not res_b:
        return
    print(f"\n{'#'*70}\nCOMPARACAO DE BALANCEAMENTO (pesos Tversky)\n{'#'*70}")
    print(f"{'metrica':<22}{res_a['tag']:<28}{res_b['tag']:<28}")
    for k, label in [("f1", "F1 (CV @0.5)"), ("recall", "Recall (CV @0.5)"),
                     ("precision", "Precision (CV @0.5)"), ("iou", "IoU (CV @0.5)")]:
        a = res_a["cv_summary_at_0.5"][k]
        b = res_b["cv_summary_at_0.5"][k]
        print(f"{label:<22}{a['mean']:.4f} ± {a['std']:.4f}        "
              f"{b['mean']:.4f} ± {b['std']:.4f}")
    print(f"{'AUC-PR (OOF)':<22}{res_a['oof_curves']['auc_pr']:<28.4f}"
          f"{res_b['oof_curves']['auc_pr']:<28.4f}")
    print(f"{'best thr (F1)':<22}"
          f"{res_a['oof_curves']['best_threshold']['threshold']:<28.2f}"
          f"{res_b['oof_curves']['best_threshold']['threshold']:<28.2f}")
    print(f"\n  Leitura: pesos fn>fp devem SUBIR o recall (pega mais fenda) e DESCER a "
          f"precisao.\n  A escolha certa e a que maximiza o F1 (CV) ou atinge o recall-alvo "
          f"desejado\n  sem colapsar a precisao.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Cross-validation espacial + curvas PR/ROC/F1")
    ap.add_argument("--feature", default="crevasses", choices=list(Config.FEATURES.keys()))
    ap.add_argument("--years", type=int, nargs="+", default=[2016])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--block-size", type=int, default=2048,
                    help="Tamanho do bloco espacial em px (CV em blocos). Grande o "
                         "bastante p/ separar tiles vizinhos com overlap. Default 2048.")
    ap.add_argument("--loss", default="bce_tversky", choices=["bce_dice", "bce_tversky"])
    ap.add_argument("--fp-weight", type=float, default=0.7)
    ap.add_argument("--fn-weight", type=float, default=0.3)
    ap.add_argument("--use-dem-channels", action="store_true")
    ap.add_argument("--dem-features", nargs="+", default=["relief", "slope", "curvature"],
                    choices=["relief", "slope", "curvature"])
    ap.add_argument("--dem-window", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Limita o nº de tiles (debug/smoke).")
    ap.add_argument("--dry-run", action="store_true",
                    help="So monta os folds e imprime a composicao (nao treina). "
                         "Use para inspecionar tamanhos/positivos/buffer antes da rodada longa.")
    ap.add_argument("--compare-weights", action="store_true",
                    help="Roda CV para fp/fn=0.7/0.3 E 0.3/0.7 e compara (experimento "
                         "de balanceamento). Ignora --fp-weight/--fn-weight.")
    ap.add_argument("--smoke", action="store_true",
                    help="Teste rapido: folds=2, epochs=2, patience=2, max-tiles=40.")
    args = ap.parse_args()

    if args.smoke:
        args.folds = 2
        args.epochs = 8          # > warmup(5) para exercitar o scheduler cosseno
        args.patience = 8
        args.log_every = 1
        args.max_tiles = args.max_tiles or 60
        print("[SMOKE] folds=2 epochs=8 patience=8 max-tiles=60 (exercita scheduler)")

    if args.compare_weights:
        cfg_a = copy.deepcopy(args); cfg_a.fp_weight, cfg_a.fn_weight = 0.7, 0.3
        cfg_b = copy.deepcopy(args); cfg_b.fp_weight, cfg_b.fn_weight = 0.3, 0.7
        res_a = run_crossval(cfg_a)
        res_b = run_crossval(cfg_b)
        compare_results(res_a, res_b)
    else:
        run_crossval(args)


if __name__ == "__main__":
    main()

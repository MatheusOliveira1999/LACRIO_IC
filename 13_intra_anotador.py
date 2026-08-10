"""
13_intra_anotador.py - Consistencia intra-anotador (auto-concordancia) do GT.

Protocolo: re-anotacao AS CEGAS de 15 tiles de 2016 ja anotados ha >=2 meses
(12 positivos estratificados por tercil de area GT + 3 GT=0; SEM os 32 tiles
da auditoria de 09/08, que estao frescos na memoria). As mascaras originais
sao movidas para um cofre durante a sessao (o annotate.py entao apresenta os
tiles em branco) e restauradas ao final; as re-anotacoes ficam arquivadas a
parte. Metricas: P/R/F1/IoU estritas e com tolerancia de 4 px (~32 cm),
originais como referencia.

Uso:
  python 13_intra_anotador.py prepare   # sorteia, move originais, gera fila
  # (sessao: python annotate.py --feature crevasses --no-model \
  #          --csv results/intra_anotador/fila.csv)
  python 13_intra_anotador.py finish    # arquiva re-anotacoes, RESTAURA GT,
                                        # e imprime a comparacao
ATENCAO: entre 'prepare' e 'finish' o GT de 2016 fica temporariamente com
203/218 tiles — nao treinar nem validar nesse intervalo.
"""

import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from config import Config

GT_DIR = Config.MASKS_DIR / "2016" / "annotations" / "crevasses"
BASE = Config.RESULTS_DIR / "intra_anotador"
COFRE = BASE / "originais"
REANO = BASE / "reanotadas"
FILA = BASE / "fila.csv"
N_POS, N_NEG, SEED = 12, 3, 42
CUTOFF_MTIME = 1786244400  # 2026-08-09 00:00 -03 — exclui anotacoes da auditoria


def tile_info():
    """(tile, area_gt, mtime) para cada anotacao de 2016."""
    out = []
    for p in sorted(GT_DIR.glob("tile_*_crevasses.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        out.append((p.stem.removesuffix("_crevasses"), int((m > 127).sum()),
                    p.stat().st_mtime))
    return out


def prepare():
    if COFRE.exists() and any(COFRE.iterdir()):
        sys.exit("Cofre nao vazio — sessao anterior nao finalizada? Rode 'finish'.")
    info = [t for t in tile_info() if t[2] < CUTOFF_MTIME]
    pos = sorted([t for t in info if t[1] > 0], key=lambda t: t[1])
    neg = [t for t in info if t[1] == 0]
    rng = random.Random(SEED)
    terco = len(pos) // 3
    sample = (rng.sample(pos[:terco], N_POS // 3)
              + rng.sample(pos[terco:2 * terco], N_POS // 3)
              + rng.sample(pos[2 * terco:], N_POS - 2 * (N_POS // 3))
              + rng.sample(neg, N_NEG))
    rng.shuffle(sample)

    COFRE.mkdir(parents=True, exist_ok=True)
    REANO.mkdir(parents=True, exist_ok=True)
    rows = []
    for tile, area, _ in sample:
        src = GT_DIR / f"{tile}_crevasses.png"
        shutil.move(str(src), str(COFRE / src.name))
        img_path = (Config.TILES_DIR / "2016" / f"{tile}.png").resolve()
        rows.append([tile, 2016, str(img_path), 1.0, 0, 0.0, 0.0, 0, 0, 0, 1.0])
    with open(FILA, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "year", "tile_path", "uncertainty_score",
                    "pred_area", "max_prob", "entropy_detected",
                    "mean_r", "mean_g", "mean_b", "valid_ratio"])
        w.writerows(rows)
    print(f"{len(sample)} tiles na fila (originais no cofre). Sessao:\n"
          f"  python annotate.py --feature crevasses --no-model --csv {FILA}\n"
          f"Depois: python 13_intra_anotador.py finish")


def _tol_pr(a, b, tol_px=4):
    """Fracao dos pixels de a dentro de tol_px de b (precision relaxada)."""
    if a.sum() == 0:
        return None
    if b.sum() == 0:
        return 0.0
    dt = distance_transform_edt(~b)
    return float((dt[a] <= tol_px).mean())


def finish():
    fila = list(csv.DictReader(open(FILA)))
    resultados, faltando = [], []
    for r in fila:
        tile = r["tile_id"]
        nova = GT_DIR / f"{tile}_crevasses.png"
        if not nova.exists():
            faltando.append(tile)
            continue
        shutil.move(str(nova), str(REANO / nova.name))
    for p in sorted(COFRE.glob("*.png")):
        shutil.move(str(p), str(GT_DIR / p.name))
    n_gt = len(list(GT_DIR.glob("tile_*_crevasses.png")))
    print(f"GT restaurado: {n_gt} tiles em {GT_DIR}")
    if faltando:
        print(f"AVISO: {len(faltando)} tiles nao re-anotados: {faltando}")

    print(f"\n{'tile':<14}{'F1':>7}{'IoU':>7}{'P':>7}{'R':>7}"
          f"{'F1@4px':>9}  (referencia = original)")
    tp = fp = fn = 0
    tps4 = []
    for r in fila:
        tile = r["tile_id"]
        o_p, n_p = COFRE.parent / "originais", REANO
        orig_f = REANO.parent / "originais" / f"{tile}_crevasses.png"
        # apos restauracao, original esta no GT_DIR:
        orig = cv2.imread(str(GT_DIR / f"{tile}_crevasses.png"),
                          cv2.IMREAD_GRAYSCALE) > 127
        re_f = REANO / f"{tile}_crevasses.png"
        if not re_f.exists():
            continue
        re = cv2.imread(str(re_f), cv2.IMREAD_GRAYSCALE) > 127
        _tp = int((orig & re).sum())
        _fp = int((re & ~orig).sum())
        _fn = int((orig & ~re).sum())
        tp, fp, fn = tp + _tp, fp + _fp, fn + _fn
        p = _tp / (_tp + _fp) if _tp + _fp else (1.0 if _fn == 0 else 0.0)
        rr = _tp / (_tp + _fn) if _tp + _fn else (1.0 if _fp == 0 else 0.0)
        f1 = 2 * p * rr / (p + rr) if p + rr else (1.0 if _fp + _fn == 0 else 0.0)
        iou = _tp / (_tp + _fp + _fn) if _tp + _fp + _fn else 1.0
        pt = _tol_pr(re, orig)
        rt = _tol_pr(orig, re)
        if pt is not None and rt is not None and (pt + rt) > 0:
            f1t = 2 * pt * rt / (pt + rt)
            tps4.append((pt, rt))
        elif _fp + _fn == 0:
            f1t = 1.0
        else:
            f1t = 0.0
        print(f"{tile:<14}{f1:>7.3f}{iou:>7.3f}{p:>7.3f}{rr:>7.3f}{f1t:>9.3f}")

    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F1 = 2 * P * R / (P + R) if P + R else 0
    IoU = tp / (tp + fp + fn) if tp + fp + fn else 0
    print("-" * 58)
    print(f"{'MICRO':<14}{F1:>7.3f}{IoU:>7.3f}{P:>7.3f}{R:>7.3f}")
    if tps4:
        pt = np.mean([x[0] for x in tps4])
        rt = np.mean([x[1] for x in tps4])
        print(f"Tolerancia 4px (medias por tile): P={pt:.3f} R={rt:.3f} "
              f"F1={2*pt*rt/(pt+rt):.3f}")
    print("\nInterpretacao: este F1 e o TETO de auto-consistencia do GT — "
          "compare com o F1 do modelo (CV 0.654; in-sample 0.72).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare()
    elif cmd == "finish":
        finish()
    else:
        sys.exit(__doc__)

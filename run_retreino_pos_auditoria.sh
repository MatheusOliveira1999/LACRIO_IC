#!/usr/bin/env bash
# Retreino + re-CV apos a auditoria dos 65 negativos aleatorios (09/08/2026).
# Dataset 2016 corrigido: 218 tiles = 141 positivos + 77 GT=0 explicitos.
#
# --neg-ratio 0.55 CONGELA o pool de negativos: int(141*0.55)=77 = exatamente os
# GT=0 explicitos, ZERO negativos aleatorios nao verificados (a causa do label
# noise que esta auditoria corrigiu). NAO voltar ao default 1.0 sem nova auditoria.
#
# Protocolo identico ao ep.125 (loss/LR/epocas) de proposito: este run mede
# APENAS o efeito da correcao de rotulos. Melhorias de treino ficam para depois.

set -o pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/retreino_pos_auditoria_${TS}"
mkdir -p "$RUN_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_glaciar

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_DIR/main.log"; }

log "==> RUN: $RUN_DIR"

# ---- 0. Preservar o ep.125 e as validacoes dele ----
if [ ! -f models/unet_crevasses_ep125_F1_0.7235_20260610.pth ]; then
    cp models/unet_crevasses_best.pth models/unet_crevasses_ep125_F1_0.7235_20260610.pth
    log "==> ep.125 arquivado em models/unet_crevasses_ep125_F1_0.7235_20260610.pth"
fi
mkdir -p results/archive_ep125
for Y in 2016 2017 2018 2019; do
    cp -n "results/unet_validation_crevasses_${Y}.json" \
          "results/archive_ep125/unet_validation_crevasses_${Y}_ep125.json" 2>/dev/null
done
log "==> Validacoes do ep.125 preservadas em results/archive_ep125/"

# ---- 1. Treino (dataset corrigido, pool congelado) ----
log "==> [1/3] Treino: 141 pos + 77 GT=0, --neg-ratio 0.55 (zero aleatorios)"
python -u 03_train_unet.py \
    --feature crevasses --years 2016 \
    --loss bce_tversky --fp-weight 0.7 --fn-weight 0.3 \
    --epochs 200 --patience 100 \
    --use-dem-channels --neg-ratio 0.55 \
    > "$RUN_DIR/1_train.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "    ERRO treino (rc=$rc). Ver $RUN_DIR/1_train.log"; exit 1
fi
comp=$(grep -E "Amostras positivas" "$RUN_DIR/1_train.log" | tail -1)
best=$(grep -oE "Melhor epoca: [0-9]+" "$RUN_DIR/1_train.log" | tail -1)
f1=$(grep -oE "Melhor Val F1: [0-9.]+" "$RUN_DIR/1_train.log" | tail -1)
log "    $comp"
log "    $best | $f1"
if ! grep -qE "Negativas aleatorias: 0" "$RUN_DIR/1_train.log"; then
    log "    AVISO: pool NAO congelou (negativas aleatorias != 0) — conferir!"
fi

# ---- 2. Validacoes annotated-only (2016 agora com 218 tiles; cross-year p/ leitura) ----
for Y in 2016 2017 2018 2019; do
    log "==> [2/3] Validacao $Y"
    python -u 04_inference_unet.py --feature crevasses --year "$Y" \
        --annotated-only --threshold 0.5 --no-feature-filter --validate \
        > "$RUN_DIR/2_validate_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "    ERRO validacao $Y (rc=$rc)"; continue; fi
    met=$(grep -E "^  (Precision|Recall|F1|IoU)" "$RUN_DIR/2_validate_${Y}.log" | tr '\n' ' ')
    log "    $Y: $met"
done

# ---- 3. Cross-validation espacial (config de producao; salva folds + curvas) ----
log "==> [3/3] CV espacial 5-fold (fp0.7/fn0.3, neg-ratio 0.55, salvando checkpoints)"
python -u 10_crossval_curves.py \
    --feature crevasses --years 2016 --use-dem-channels \
    --loss bce_tversky --fp-weight 0.7 --fn-weight 0.3 \
    --folds 5 --epochs 200 --patience 60 --block-size 2048 \
    --neg-ratio 0.55 \
    > "$RUN_DIR/3_crossval.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "    ERRO CV (rc=$rc). Ver $RUN_DIR/3_crossval.log"; exit 1
fi
grep -E "RESULTADO CV|F1   \(CV|P    \(CV|R    \(CV|AUC-PR|Melhor threshold" \
    "$RUN_DIR/3_crossval.log" | tee -a "$RUN_DIR/main.log"

log "==> CONCLUIDO. Comparar com baseline ep.125: CV F1 0.6193±0.030, P 0.5627, R 0.6964."
log "    Checkpoints de fold: models/crossval_folds/ | Curvas: results/crossval/*_curves.npz"

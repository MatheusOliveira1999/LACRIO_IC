#!/usr/bin/env bash
# Cross-validation espacial + curvas PR/ROC/F1 para crevasses 2016.
# Experimento de balanceamento: compara Tversky fp/fn = 0.7/0.3 (favorece precisao,
# config atual) vs 0.3/0.7 (favorece recall = penaliza fenda perdida).
# Saidas: results/crossval/*.json + results/figures/crossval/*.png

set -o pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/crossval_${TS}"
mkdir -p "$RUN_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_glaciar

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_DIR/main.log"; }

# Config (sobrescrevivel por variavel de ambiente)
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-60}"
BLOCK="${BLOCK:-2048}"

log "==> RUN: $RUN_DIR"
log "==> folds=$FOLDS epochs=$EPOCHS patience=$PATIENCE block_size=${BLOCK}px"
log "==> Comparando Tversky 0.7/0.3 (precisao) vs 0.3/0.7 (recall)"

python -u 10_crossval_curves.py \
    --feature crevasses --years 2016 --use-dem-channels \
    --folds "$FOLDS" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --block-size "$BLOCK" --compare-weights \
    > "$RUN_DIR/crossval.log" 2>&1
rc=$?

if [ $rc -ne 0 ]; then
    log "    ERRO (rc=$rc). Ver $RUN_DIR/crossval.log"
    exit 1
fi

# Eco do resumo final
grep -E "RESULTADO CV|F1   \(CV|AUC-PR|Melhor threshold|Balanceamento|COMPARACAO|^[A-Za-z].*±" \
    "$RUN_DIR/crossval.log" | tee -a "$RUN_DIR/main.log"

log "==> CONCLUIDO. JSONs em results/crossval/ | figuras em results/figures/crossval/"

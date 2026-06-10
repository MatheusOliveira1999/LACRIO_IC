#!/usr/bin/env bash
# Retreino crevasses 2016 com anotacoes revisadas + validacao + inferencia + mosaico.
# Modelo antigo ja salvo em models/unet_crevasses_PRE_RETREINO_*.pth

set -o pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/retreino_2016_${TS}"
mkdir -p "$RUN_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_glaciar

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_DIR/main.log"; }

log "==> RUN: $RUN_DIR"
log "==> Modelo antigo preservado em models/unet_crevasses_PRE_RETREINO_*.pth"

# ---- 1. Treino ----
log "==> [1/4] Treino (109 pos + 44 GT=0, --use-dem-channels)"
python -u 03_train_unet.py \
    --feature crevasses --years 2016 \
    --loss bce_tversky --fp-weight 0.7 --fn-weight 0.3 \
    --epochs 200 --patience 100 \
    --use-dem-channels \
    > "$RUN_DIR/1_train.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "    ERRO treino (rc=$rc). Ver $RUN_DIR/1_train.log"
    exit 1
fi
best=$(grep -oE "Melhor epoca: [0-9]+" "$RUN_DIR/1_train.log" | tail -1)
f1=$(grep -oE "Melhor Val F1: [0-9.]+" "$RUN_DIR/1_train.log" | tail -1)
tempo=$(grep -oE "Tempo: [0-9.]+ minutos" "$RUN_DIR/1_train.log" | tail -1)
log "    $best | $f1 | $tempo"

# ---- 2. Validacao annotated-only ----
log "==> [2/4] Validacao em tiles GT 2016"
python -u 04_inference_unet.py --feature crevasses --year 2016 \
    --annotated-only --threshold 0.5 --no-feature-filter --validate \
    > "$RUN_DIR/2_validate.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "    ERRO validacao (rc=$rc)"
else
    metrics=$(grep -E "^  Precision" "$RUN_DIR/2_validate.log" | tail -1)
    log "    $metrics"
fi

# ---- 3. Inferencia completa 2016 ----
log "==> [3/4] Inferencia completa em todos os tiles de 2016"
python -u 04_inference_unet.py --feature crevasses --year 2016 \
    --threshold 0.5 --no-feature-filter \
    > "$RUN_DIR/3_inference.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "    ERRO inferencia (rc=$rc)"
else
    proc=$(grep -oE "Processados: [0-9]+ \| Com feicoes: [0-9]+" "$RUN_DIR/3_inference.log" | tail -1)
    log "    $proc"
fi

# ---- 4. Reconstrucao mosaico ----
log "==> [4/4] Reconstruindo mosaico 2016"
python -u 05_reconstruct_mosaic.py --feature crevasses --year 2016 \
    > "$RUN_DIR/4_reconstruct.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "    ERRO mosaico (rc=$rc)"
else
    cov=$(grep -oE "[0-9.]+% cobertura" "$RUN_DIR/4_reconstruct.log" | tail -1)
    log "    $cov"
fi

log "==> CONCLUIDO"

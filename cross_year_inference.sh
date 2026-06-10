#!/usr/bin/env bash
# Inferencia cross-year com modelo 6-ch (F1=0.6356 em 2016).
# Roda em sequencia: inferencia + reconstrucao para 2017, 2018, 2019, 2020.

set -o pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/cross_year_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_glaciar

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_DIR/main.log"; }

log "==> Restaurando melhor checkpoint (6-ch F1=0.6356)"
cp models/unet_crevasses_6ch_F1_0.5837_20260527.pth models/unet_crevasses_best.pth

declare -A SUMMARY

for YEAR in 2017 2018 2019 2020; do
    log "==> Ano $YEAR | Inferencia"
    python 04_inference_unet.py --feature crevasses --year "$YEAR" \
        --threshold 0.5 --no-feature-filter \
        > "$RUN_DIR/inference_${YEAR}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO inferencia ${YEAR} (rc=$rc)"; continue
    fi
    proc=$(grep -oE "Processados: [0-9]+ \| Com feicoes: [0-9]+" "$RUN_DIR/inference_${YEAR}.log" | tail -1)
    log "    $proc"

    log "==> Ano $YEAR | Reconstruindo mosaico"
    python 05_reconstruct_mosaic.py --feature crevasses --year "$YEAR" \
        > "$RUN_DIR/reconstruct_${YEAR}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO mosaico ${YEAR} (rc=$rc)"; continue
    fi
    cov=$(grep -oE "[0-9]+\.[0-9]+% cobertura" "$RUN_DIR/reconstruct_${YEAR}.log" | tail -1)
    log "    $cov"
    SUMMARY[$YEAR]="$cov | $proc"
done

log "==> CONCLUIDO"
echo ""
echo "============================================"
echo "  RESUMO CROSS-YEAR"
echo "============================================"
for YEAR in 2017 2018 2019 2020; do
    echo "  $YEAR: ${SUMMARY[$YEAR]:-FALHOU}"
done | tee -a "$RUN_DIR/main.log"
echo ""
echo "  Logs: $RUN_DIR/"
echo "  Mosaicos: results/{2017,2018,2019,2020}/crevasses_mask_*.tif"

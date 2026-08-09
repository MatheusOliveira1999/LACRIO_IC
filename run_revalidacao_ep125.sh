#!/usr/bin/env bash
# Re-validacao + re-inferencia cross-year com o modelo ep.125 (retreino 10/06, 109 pos).
# Motivacao: metricas cross-year 2017-2019 e mascaras 8cm ainda eram do modelo antigo
# (checkpoint 27/05 na validacao; ep.98 nas mascaras 8cm de 03/06). Ver vault Progresso.md.
#
# Fases:
#   A) Validacao annotated-only 2017/2018/2019 (tiles nativos, protocolo de 28/05)
#   B) Inferencia 8cm + mosaicos 2016-2019 (tiles_8cm, substitui mascaras do ep.98)
#   C) Analises Camada 1 (07) e Camada 3 (09) sobre as novas mascaras 8cm
#      (Camada 2 / 08 PULADA: falhou com rc=143 apos 9h30 em 03/06 - rodar a parte)
#   D) Inferencia nativa + mosaicos 2017/2018/2019/2020 (2016 ja feito em 10/06)
#
# Artefatos antigos preservados em results/archive_pre_ep125/

set -o pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/revalidacao_ep125_${TS}"
mkdir -p "$RUN_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_glaciar

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_DIR/main.log"; }

log "==> RUN: $RUN_DIR"

# ---- Guarda: so roda com o checkpoint ep.125 ----
EPOCH=$(python -c "import torch; print(torch.load('models/unet_crevasses_best.pth', map_location='cpu', weights_only=True).get('epoch'))")
if [ "$EPOCH" != "125" ]; then
    log "ABORT: models/unet_crevasses_best.pth nao e o ep.125 (epoca=$EPOCH)."
    log "       NAO rodar cross_year_inference.sh antes deste script (ele restaura o checkpoint antigo)."
    exit 1
fi
log "==> Checkpoint confirmado: ep.125 (retreino 10/06, 109 positivos, 6 canais)"

# =============================================================================
# Fase A: Validacao cross-year (tiles nativos, mesmo protocolo de 28/05)
# =============================================================================
for Y in 2017 2018 2019; do
    log "==> [A] Validacao $Y (annotated-only, thr=0.5, no-feature-filter)"
    python -u 04_inference_unet.py --feature crevasses --year "$Y" \
        --annotated-only --threshold 0.5 --no-feature-filter --validate \
        > "$RUN_DIR/A_validate_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO validacao $Y (rc=$rc)"; continue
    fi
    met=$(grep -E "^  (Precision|Recall|F1|IoU)" "$RUN_DIR/A_validate_${Y}.log" | tr '\n' ' ')
    log "    $Y: $met"
done

# =============================================================================
# Fase B: Inferencia 8cm + reconstrucao 2016-2019
# =============================================================================
for Y in 2016 2017 2018 2019; do
    log "==> [B] Inferencia 8cm $Y"
    python -u 04_inference_unet.py --feature crevasses --year "$Y" \
        --threshold 0.5 --no-feature-filter \
        --tiles-dir tiles_8cm --output-mask-suffix _8cm \
        > "$RUN_DIR/B_inference8cm_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO inferencia 8cm $Y (rc=$rc)"; continue
    fi
    proc=$(grep -oE "Processados: [0-9]+ \| Com feicoes: [0-9]+" "$RUN_DIR/B_inference8cm_${Y}.log" | tail -1)
    log "    $Y: $proc"

    log "==> [B] Mosaico 8cm $Y"
    python -u 05_reconstruct_mosaic.py --feature crevasses --year "$Y" \
        --tiles-dir tiles_8cm --masks-subdir-suffix _8cm --output-suffix _8cm \
        > "$RUN_DIR/B_reconstruct8cm_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO mosaico 8cm $Y (rc=$rc)"; continue
    fi
    cov=$(grep -oE "[0-9]+\.[0-9]+% cobertura" "$RUN_DIR/B_reconstruct8cm_${Y}.log" | tail -1)
    log "    $Y: $cov"
done

# =============================================================================
# Fase C: Analises sobre as novas mascaras 8cm (Camadas 1 e 3; 08 pulada)
# =============================================================================
log "==> [C] Camada 1 quantitativa (07 --normalized)"
python -u 07_quantitative_analysis.py --normalized > "$RUN_DIR/C_camada1.log" 2>&1
log "    rc=$?"

log "==> [C] Camada 3 glaciologica (09 --normalized)"
python -u 09_glaciological_analysis.py --normalized > "$RUN_DIR/C_camada3.log" 2>&1
log "    rc=$?"

log "==> [C] Camada 2 (08) PULADA — rodar a parte (falhou rc=143/9h30 em 03/06)"

# =============================================================================
# Fase D: Inferencia nativa + mosaicos 2017-2020 (2016 ja e ep.125 desde 10/06)
# =============================================================================
for Y in 2017 2018 2019 2020; do
    log "==> [D] Inferencia nativa $Y"
    python -u 04_inference_unet.py --feature crevasses --year "$Y" \
        --threshold 0.5 --no-feature-filter \
        > "$RUN_DIR/D_inference_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO inferencia nativa $Y (rc=$rc)"; continue
    fi
    proc=$(grep -oE "Processados: [0-9]+ \| Com feicoes: [0-9]+" "$RUN_DIR/D_inference_${Y}.log" | tail -1)
    log "    $Y: $proc"

    log "==> [D] Mosaico nativo $Y"
    python -u 05_reconstruct_mosaic.py --feature crevasses --year "$Y" \
        > "$RUN_DIR/D_reconstruct_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO mosaico nativo $Y (rc=$rc)"; continue
    fi
    cov=$(grep -oE "[0-9]+\.[0-9]+% cobertura" "$RUN_DIR/D_reconstruct_${Y}.log" | tail -1)
    log "    $Y: $cov"
done

log "==> CONCLUIDO. Validacoes: results/unet_validation_crevasses_{2017,2018,2019}.json"
log "    Mascaras 8cm: results/{ano}/crevasses_mask_{ano}_8cm.tif | Analises: results/*.csv + figures/"
log "    Antigos preservados em results/archive_pre_ep125/"

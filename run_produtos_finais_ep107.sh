#!/usr/bin/env bash
# Produtos finais com o modelo ep.107 (dataset auditado): TTA + threshold 0.6 +
# merge probabilistico (Hann) — receita validada em 09/08 (TTA +0.9pp; blend
# +1.7pp P no mosaico; thr 0.6 = centro do plato OOF, P 0.624/R 0.687).
#
# Blend probabilistico: 8cm 2016-2019 e nativo 2016-2018.
# Nativo 2019/2020: max-merge (canvases de 11.6/16.9 GB nao cabem na RAM de 23GB;
# sao produtos de visualizacao — as analises usam os 8cm).
# Disco a ~91%: mapas de probabilidade sao APAGADOS apos cada blend.

set -o pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/produtos_finais_ep107_${TS}"
mkdir -p "$RUN_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_glaciar

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_DIR/main.log"; }

log "==> RUN: $RUN_DIR"

EPOCH=$(python -c "import torch; print(torch.load('models/unet_crevasses_best.pth', map_location='cpu', weights_only=True).get('epoch'))")
if [ "$EPOCH" != "107" ]; then
    log "ABORT: best.pth nao e o ep.107 (epoca=$EPOCH)"; exit 1
fi
log "==> Checkpoint confirmado: ep.107 | receita: TTA + thr 0.6 + blend Hann"

# ---- 0. Arquivar produtos do ep.125 e validacoes ep.107@0.5 ----
mkdir -p results/archive_ep125 results/archive_ep107_thr05
for Y in 2016 2017 2018 2019; do
    cp -n "results/$Y/crevasses_mask_${Y}_8cm.tif" results/archive_ep125/ 2>/dev/null
done
for Y in 2016 2017 2018 2019 2020; do
    cp -n "results/$Y/crevasses_mask_${Y}.tif" "results/archive_ep125/crevasses_mask_${Y}_nativo.tif" 2>/dev/null
done
cp -n results/crevasses_stats_per_year.csv results/crevasse_geometries_*.csv results/archive_ep125/ 2>/dev/null
for Y in 2016 2017 2018 2019; do
    cp -n "results/unet_validation_crevasses_${Y}.json" results/archive_ep107_thr05/ 2>/dev/null
done
log "==> Produtos ep.125 e validacoes ep.107@0.5 arquivados"

# ---- C. Validacoes com a receita final (annotated-only, TTA, thr 0.6) ----
for Y in 2016 2017 2018 2019; do
    log "==> [C] Validacao $Y (TTA, thr 0.6)"
    python -u 04_inference_unet.py --feature crevasses --year "$Y" \
        --annotated-only --threshold 0.6 --tta --no-feature-filter --validate \
        > "$RUN_DIR/C_validate_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "    ERRO validacao $Y (rc=$rc)"; continue; fi
    met=$(grep -E "^  (Precision|Recall|F1|IoU)" "$RUN_DIR/C_validate_${Y}.log" | tr '\n' ' ')
    log "    $Y: $met"
done

# ---- A. Produtos 8cm (base das analises): TTA + probs -> blend 0.6 -> limpa ----
for Y in 2016 2017 2018 2019; do
    log "==> [A] Inferencia 8cm $Y (TTA + probs)"
    python -u 04_inference_unet.py --feature crevasses --year "$Y" \
        --threshold 0.6 --tta --save-probs --no-feature-filter \
        --tiles-dir tiles_8cm --output-mask-suffix _8cm \
        > "$RUN_DIR/A_inf8cm_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "    ERRO inferencia 8cm $Y (rc=$rc)"; continue; fi
    log "    $(grep -oE 'Processados: [0-9]+ \| Com feicoes: [0-9]+' "$RUN_DIR/A_inf8cm_${Y}.log" | tail -1)"

    log "==> [A] Blend 8cm $Y (thr 0.6)"
    python -u 05_reconstruct_mosaic.py --feature crevasses --year "$Y" \
        --tiles-dir tiles_8cm --masks-subdir-suffix _8cm --output-suffix _8cm \
        --from-probs --prob-threshold 0.6 \
        > "$RUN_DIR/A_blend8cm_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "    ERRO blend 8cm $Y (rc=$rc)"; continue; fi
    log "    $(grep -oE '[0-9.]+% cobertura' "$RUN_DIR/A_blend8cm_${Y}.log" | tail -1)"
    rm -rf "masks/$Y/crevasses_8cm_probs"
done

# ---- D. Analises sobre os produtos finais 8cm ----
log "==> [D] Camada 1 (07 --normalized)"
python -u 07_quantitative_analysis.py --normalized > "$RUN_DIR/D_camada1.log" 2>&1
log "    rc=$?"
log "==> [D] Camada 3 (09 --normalized)"
python -u 09_glaciological_analysis.py --normalized > "$RUN_DIR/D_camada3.log" 2>&1
log "    rc=$?"
log "==> [D] Camada 2 (08 --normalized)"
python -u 08_qualitative_overlays.py --normalized > "$RUN_DIR/D_camada2.log" 2>&1
log "    rc=$?"

# ---- B. Produtos nativos (visualizacao) ----
for Y in 2016 2017 2018; do
    log "==> [B] Inferencia nativa $Y (TTA + probs)"
    python -u 04_inference_unet.py --feature crevasses --year "$Y" \
        --threshold 0.6 --tta --save-probs --no-feature-filter \
        > "$RUN_DIR/B_inf_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "    ERRO inferencia $Y (rc=$rc)"; continue; fi
    log "==> [B] Blend nativo $Y (thr 0.6)"
    python -u 05_reconstruct_mosaic.py --feature crevasses --year "$Y" \
        --from-probs --prob-threshold 0.6 \
        > "$RUN_DIR/B_blend_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "    ERRO blend nativo $Y (rc=$rc)"; fi
    rm -rf "masks/$Y/crevasses_probs"
done
for Y in 2019 2020; do
    log "==> [B] Inferencia nativa $Y (TTA, max-merge — canvas grande demais p/ blend)"
    python -u 04_inference_unet.py --feature crevasses --year "$Y" \
        --threshold 0.6 --tta --no-feature-filter \
        > "$RUN_DIR/B_inf_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "    ERRO inferencia $Y (rc=$rc)"; continue; fi
    python -u 05_reconstruct_mosaic.py --feature crevasses --year "$Y" \
        > "$RUN_DIR/B_rec_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "    ERRO mosaico $Y (rc=$rc)"; fi
    log "    $(grep -oE '[0-9.]+% cobertura' "$RUN_DIR/B_rec_${Y}.log" | tail -1)"
done

log "==> CONCLUIDO. Produtos finais ep.107 (TTA + thr 0.6; blend onde couber)."
log "    Antigos: results/archive_ep125/ | Validacoes @0.5: results/archive_ep107_thr05/"

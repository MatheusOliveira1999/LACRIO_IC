#!/usr/bin/env bash
# Pipeline completo: padronizar resolucao -> tilear -> inferir -> reconstruir -> analisar
# Roda em sequencia, parando se algum passo falhar.

set -o pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/normalized_pipeline_${TS}"
mkdir -p "$RUN_DIR"
SUMMARY="$RUN_DIR/RESUMO.md"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_glaciar

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_DIR/main.log"; }

log "==> RUN dir: $RUN_DIR"
log "==> Checkpoint sera usado: models/unet_crevasses_best.pth"
python -c "
import torch
ck = torch.load('models/unet_crevasses_best.pth', map_location='cpu', weights_only=True)
print(f'    Epoca: {ck.get(\"epoch\")}, Val F1 interno: {ck.get(\"val_f1\"):.4f}, in_channels: {ck.get(\"in_channels\", 3)}')
" | tee -a "$RUN_DIR/main.log"

YEARS=(2016 2017 2018 2019)   # 2020 excluido (sem GT, mosaico muito grande)
T0=$(date +%s)

# -----------------------------------------------------------------------------
# Fase 1: Normalizacao de RGB
# -----------------------------------------------------------------------------
log "==> [1/5] Normalizando RGBs para 8 cm/px (anos: ${YEARS[*]})"
python -u 00_normalize_resolution.py --years "${YEARS[@]}" > "$RUN_DIR/01_normalize.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "    ERRO normalize (rc=$rc). Ver $RUN_DIR/01_normalize.log"
    exit 1
fi
log "    OK"

# -----------------------------------------------------------------------------
# Fase 2: Tiling
# -----------------------------------------------------------------------------
log "==> [2/5] Tilando mosaicos normalizados em tiles_8cm/"
python -u 01_create_tiles.py \
    --source-dir Schiaparelli_glacier/normalized_8cm \
    --output-dir tiles_8cm \
    > "$RUN_DIR/02_create_tiles.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "    ERRO tiling (rc=$rc). Ver $RUN_DIR/02_create_tiles.log"
    exit 1
fi
log "    OK"

# -----------------------------------------------------------------------------
# Fase 3: Inferencia cross-year
# -----------------------------------------------------------------------------
log "==> [3/5] Inferencia cross-year (5 anos)"
for Y in "${YEARS[@]}"; do
    log "    inferindo $Y..."
    python -u 04_inference_unet.py \
        --feature crevasses --year "$Y" \
        --threshold 0.5 --no-feature-filter \
        --tiles-dir tiles_8cm \
        --output-mask-suffix _8cm \
        > "$RUN_DIR/03_inference_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO inferencia $Y (rc=$rc)"
        continue
    fi
    proc=$(grep -oE "Processados: [0-9]+ \| Com feicoes: [0-9]+" "$RUN_DIR/03_inference_${Y}.log" | tail -1)
    log "      $Y: $proc"
done

# -----------------------------------------------------------------------------
# Fase 4: Reconstrucao
# -----------------------------------------------------------------------------
log "==> [4/5] Reconstruindo mosaicos a 8 cm"
for Y in "${YEARS[@]}"; do
    log "    reconstruindo $Y..."
    python -u 05_reconstruct_mosaic.py \
        --feature crevasses --year "$Y" \
        --tiles-dir tiles_8cm \
        --masks-subdir-suffix _8cm \
        --output-suffix _8cm \
        > "$RUN_DIR/04_reconstruct_${Y}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "    ERRO reconstruct $Y (rc=$rc)"
        continue
    fi
    cov=$(grep -oE "[0-9]+\.[0-9]+% cobertura" "$RUN_DIR/04_reconstruct_${Y}.log" | tail -1)
    log "      $Y: $cov"
done

# -----------------------------------------------------------------------------
# Fase 5: Analises 07 / 08 / 09
# -----------------------------------------------------------------------------
log "==> [5/5] Analises (Camadas 1, 2, 3)"

log "    [Camada 1] quantitativa..."
python -u 07_quantitative_analysis.py --normalized > "$RUN_DIR/05_camada1.log" 2>&1
log "      rc=$?"

log "    [Camada 2] qualitativa..."
python -u 08_qualitative_overlays.py --normalized > "$RUN_DIR/05_camada2.log" 2>&1
log "      rc=$?"

log "    [Camada 3] glaciologica..."
python -u 09_glaciological_analysis.py --normalized > "$RUN_DIR/05_camada3.log" 2>&1
log "      rc=$?"

# -----------------------------------------------------------------------------
# Resumo
# -----------------------------------------------------------------------------
T1=$(date +%s)
ELAPSED=$((T1 - T0))

log "==> CONCLUIDO em $((ELAPSED/60))min$((ELAPSED%60))s"

# Gerar RESUMO.md
{
    echo "# Pipeline normalizado 8 cm — RUN $TS"
    echo ""
    echo "**Tempo total:** $((ELAPSED/60))m $((ELAPSED%60))s"
    echo ""
    echo "## Cobertura por ano (mosaicos 8 cm)"
    echo ""
    echo "| Ano | Processados | Com feição | Cobertura |"
    echo "|-----|------------:|-----------:|----------:|"
    for Y in "${YEARS[@]}"; do
        proc=$(grep -oE "Processados: [0-9]+" "$RUN_DIR/03_inference_${Y}.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo "?")
        cf=$(grep -oE "Com feicoes: [0-9]+" "$RUN_DIR/03_inference_${Y}.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo "?")
        cov=$(grep -oE "[0-9]+\.[0-9]+% cobertura" "$RUN_DIR/04_reconstruct_${Y}.log" 2>/dev/null | tail -1 || echo "?")
        echo "| $Y | $proc | $cf | $cov |"
    done
    echo ""
    echo "## Arquivos gerados"
    echo ""
    echo "- Mosaicos 8 cm: results/{ano}/crevasses_mask_{ano}_8cm.tif"
    echo "- Tiles 8 cm: tiles_8cm/{ano}/"
    echo "- Predicoes 8 cm: masks/{ano}/crevasses_8cm/"
    echo "- Figuras: results/figures/{quantitative,qualitative,glaciological}/"
    echo "- CSVs: results/crevasses_stats_per_year.csv"
    echo "- Geometria por fenda: results/crevasse_geometries_{ano}.csv"
    echo ""
    echo "## Logs detalhados"
    echo "Ver $RUN_DIR/{01..05}_*.log"
} > "$SUMMARY"

log "==> Ver resumo: $SUMMARY"
echo ""
echo "============================================"
echo "  PIPELINE CONCLUIDO"
echo "  Resumo: $SUMMARY"
echo "============================================"

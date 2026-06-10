#!/usr/bin/env bash
# Treino U-Net crevasses (so 2016, nova augmentation) + validacao cross-year.
# Executar com nohup: nohup bash overnight_run.sh > logs/overnight_TS/launch.log 2>&1 &

set -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/overnight_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
SUMMARY="$RUN_DIR/RESUMO.md"

# Ativar conda env (sem -u para nao quebrar nos scripts de deactivate)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_glaciar

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_DIR/main.log"; }

log "==> Run dir: $RUN_DIR"
log "==> Python: $(which python)"
log "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>&1 | head -1)"

# -----------------------------------------------------------------------------
# 1. Backup explicito do checkpoint atual
# -----------------------------------------------------------------------------
log "==> Backup do checkpoint atual"
cp models/unet_crevasses_best.pth "$RUN_DIR/unet_crevasses_PRE_RUN.pth"
cp models/unet_crevasses_history.json "$RUN_DIR/unet_crevasses_PRE_RUN_history.json" 2>/dev/null || true
log "    Backup salvo em $RUN_DIR/unet_crevasses_PRE_RUN.pth"

# -----------------------------------------------------------------------------
# 2. Treino: U-Net crevasses, SOMENTE 2016, nova augmentation
# -----------------------------------------------------------------------------
log "==> Iniciando treino (--years 2016, bce_tversky fp=0.7 fn=0.3, 200 epocas, patience=100)"
TRAIN_LOG="$RUN_DIR/train.log"

python -u 03_train_unet.py \
    --feature crevasses \
    --years 2016 \
    --loss bce_tversky \
    --fp-weight 0.7 \
    --fn-weight 0.3 \
    --epochs 200 \
    --patience 100 \
    > "$TRAIN_LOG" 2>&1

TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
    log "ERRO: Treino falhou (rc=$TRAIN_RC). Veja $TRAIN_LOG"
    {
        echo "# Run $TIMESTAMP - FALHOU"
        echo ""
        echo "Treino falhou (rc=$TRAIN_RC). Ver log: $TRAIN_LOG"
    } > "$SUMMARY"
    exit 1
fi
log "    Treino concluido OK"

# Salvar copia do novo checkpoint com nome unico
cp models/unet_crevasses_best.pth "$RUN_DIR/unet_crevasses_2016only_${TIMESTAMP}.pth"
log "    Novo checkpoint: $RUN_DIR/unet_crevasses_2016only_${TIMESTAMP}.pth"

# -----------------------------------------------------------------------------
# 3. Validacao cross-year (2016-2020)
# -----------------------------------------------------------------------------
log "==> Iniciando validacao cross-year"
declare -A VAL_RC

for YEAR in 2016 2017 2018 2019 2020; do
    VAL_LOG="$RUN_DIR/val_${YEAR}.log"
    log "    Validando ano $YEAR..."

    python -u 04_inference_unet.py \
        --feature crevasses \
        --year "$YEAR" \
        --annotated-only \
        --threshold 0.5 \
        --no-feature-filter \
        --validate \
        > "$VAL_LOG" 2>&1
    VAL_RC[$YEAR]=$?

    if [ "${VAL_RC[$YEAR]}" -eq 0 ]; then
        log "    Ano $YEAR OK"
    else
        log "    Ano $YEAR FALHOU (rc=${VAL_RC[$YEAR]}) - ver $VAL_LOG"
    fi
done

# -----------------------------------------------------------------------------
# 4. RESUMO consolidado
# -----------------------------------------------------------------------------
log "==> Gerando RESUMO.md"

python -u - <<PYEOF >> "$RUN_DIR/main.log" 2>&1
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
results_dir = Path("results")
summary_path = run_dir / "RESUMO.md"

# Comparar com baseline anterior (vault diz: ep.95 04/05/2026)
baseline = {
    2016: {"F1": None, "IoU": None, "P": None, "R": None, "note": "nao validado"},
    2017: {"F1": 0.7296, "IoU": 0.5743, "P": 0.7237, "R": 0.7356},
    2018: {"F1": 0.7135, "IoU": 0.5546, "P": 0.6680, "R": 0.7655},
    2019: {"F1": 0.7835, "IoU": 0.6441, "P": 0.8490, "R": 0.7275},
    2020: {"F1": None, "IoU": None, "P": None, "R": None, "note": "nao validado"},
}

lines = []
lines.append("# Resumo - Treino overnight $TIMESTAMP")
lines.append("")
lines.append("**Setup:** U-Net crevasses, treino SOMENTE em 2016, nova augmentation (gradient brightness + HSV + RGBShift).")
lines.append("**Loss:** bce_tversky (fp=0.7, fn=0.3)  |  **Epocas:** 200 (patience=100)  |  **Threshold inferencia:** 0.5")
lines.append("")
lines.append("---")
lines.append("")

# Tabela comparativa
lines.append("## Validacao cross-year (modelo treinado SO em 2016)")
lines.append("")
lines.append("| Ano | F1 novo | F1 baseline (4 anos, ep.95) | Delta F1 | IoU novo | P novo | R novo | Tiles |")
lines.append("|-----|--------:|----------------------------:|---------:|---------:|-------:|-------:|------:|")

for year in [2016, 2017, 2018, 2019, 2020]:
    val_path = results_dir / f"unet_validation_crevasses_{year}.json"
    if not val_path.exists():
        lines.append(f"| {year} | (sem json) | - | - | - | - | - | - |")
        continue
    try:
        data = json.loads(val_path.read_text())
        m = data.get("global", data)
        f1 = m.get("f1", float('nan'))
        iou = m.get("iou", float('nan'))
        p = m.get("precision", m.get("p", float('nan')))
        r = m.get("recall", m.get("r", float('nan')))
        n_tiles = len(data.get("per_tile", []))
        b = baseline.get(year, {})
        f1_b = b.get("F1")
        delta = f"{(f1 - f1_b)*100:+.1f}pp" if f1_b is not None else "—"
        f1_b_str = f"{f1_b:.4f}" if f1_b is not None else "—"
        lines.append(f"| {year} | **{f1:.4f}** | {f1_b_str} | {delta} | {iou:.4f} | {p:.4f} | {r:.4f} | {n_tiles} |")
    except Exception as e:
        lines.append(f"| {year} | ERRO ao ler JSON: {e} | - | - | - | - | - | - |")

lines.append("")
lines.append("---")
lines.append("")
lines.append("## Como interpretar")
lines.append("")
lines.append("- **Delta F1 positivo** em 2017/2018/2019 = nova augmentation generaliza melhor que baseline (que viu 4 anos).")
lines.append("- **Delta F1 negativo grande** = augmentation exagerada OU regressao por usar so 2016. Diagnostico em train.log.")
lines.append("- 2016 e 2020 nao tem baseline (nao foram validados naquela rodada).")
lines.append("")
lines.append("## Arquivos")
lines.append("")
lines.append(f"- Checkpoint NOVO (so 2016): unet_crevasses_2016only_$TIMESTAMP.pth")
lines.append(f"- Checkpoint anterior (backup): unet_crevasses_PRE_RUN.pth")
lines.append(f"- Log de treino: train.log")
lines.append(f"- Logs de validacao: val_YYYY.log")
lines.append(f"- JSONs detalhados: results/unet_validation_crevasses_YYYY.json (com per-tile metrics)")
lines.append("")
lines.append("## Para restaurar o checkpoint anterior")
lines.append("")
lines.append(f"\`\`\`bash")
lines.append(f"cp {run_dir}/unet_crevasses_PRE_RUN.pth models/unet_crevasses_best.pth")
lines.append(f"\`\`\`")

summary_path.write_text("\n".join(lines))
print(f"Resumo salvo em {summary_path}")
PYEOF

log "==> CONCLUIDO. Ver: $SUMMARY"
echo ""
echo "============================================"
echo "  CONCLUIDO em $(date '+%F %T')"
echo "  Resumo: $SUMMARY"
echo "============================================"

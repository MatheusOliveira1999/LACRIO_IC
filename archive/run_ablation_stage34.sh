#!/usr/bin/env bash
# Ablation Stage 3/4 (HQ vs HQ+Aug vs HQ+Aug+LoRA)
# Usage:
#   conda activate sam_glaciar
#   bash run_ablation_stage34.sh
# Optional env vars:
#   EPOCHS=30 FORCE_RETRAIN=0 FORCE_REEVAL=0 INSTALL_MISSING=0 bash run_ablation_stage34.sh

set -u -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

EPOCHS="${EPOCHS:-30}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"
INSTALL_MISSING="${INSTALL_MISSING:-0}"

ABL_DIR="results/ablation_stage34"
SNAP_DIR="models/ablation_stage34"
LOG_DIR="logs"
FEATURES=(lakes crevasses channels)
VARIANTS=(hq_only hq_aug hq_aug_lora)

mkdir -p "$ABL_DIR" "$LOG_DIR" \
  "$SNAP_DIR"/hq_only "$SNAP_DIR"/hq_aug "$SNAP_DIR"/hq_aug_lora "$SNAP_DIR"/_active_backup

_restored=0

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_cmd() {
  local label="$1"
  shift
  log "$label"
  "$@" &
  local pid=$!

  # Heartbeat: evita sensacao de travamento quando output esta bufferizado.
  while kill -0 "$pid" 2>/dev/null; do
    sleep 45
    if kill -0 "$pid" 2>/dev/null; then
      log "Ainda em execucao: $label (pid=$pid)"
    fi
  done

  wait "$pid"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "ERRO ($rc): $label"
    return "$rc"
  fi
  return 0
}

variant_ckpt_path() {
  local variant="$1"
  local feature="$2"
  echo "$SNAP_DIR/$variant/sam_finetuned_${feature}_best.pth"
}

variant_ready() {
  local variant="$1"
  local feature
  for feature in "${FEATURES[@]}"; do
    if [[ ! -f "$(variant_ckpt_path "$variant" "$feature")" ]]; then
      return 1
    fi
  done
  return 0
}

result_ready() {
  local variant="$1"
  [[ -f "$ABL_DIR/${variant}.json" ]]
}

backup_active_models() {
  log "Backup dos checkpoints ativos"
  local feature src dst
  for feature in "${FEATURES[@]}"; do
    src="models/sam_finetuned_${feature}_best.pth"
    dst="$SNAP_DIR/_active_backup/sam_finetuned_${feature}_best.pth"
    if [[ -f "$src" ]]; then
      cp "$src" "$dst"
    fi
  done
}

restore_active_models() {
  if [[ "$_restored" -eq 1 ]]; then
    return 0
  fi
  _restored=1

  log "Restaurando checkpoints ativos (se backup existir)"
  local feature src dst
  for feature in "${FEATURES[@]}"; do
    src="$SNAP_DIR/_active_backup/sam_finetuned_${feature}_best.pth"
    dst="models/sam_finetuned_${feature}_best.pth"
    if [[ -f "$src" ]]; then
      cp "$src" "$dst"
    fi
  done
}

cleanup() {
  restore_active_models
}

trap cleanup EXIT INT TERM

have_python_module() {
  local module_name="$1"
  python - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("${module_name}") else 1)
PY
}

ensure_dependencies_for_variant() {
  local variant="$1"

  # hq_aug and hq_aug_lora need albumentations
  if [[ "$variant" == "hq_aug" || "$variant" == "hq_aug_lora" ]]; then
    if have_python_module "albumentations"; then
      return 0
    fi

    if [[ "$INSTALL_MISSING" == "1" ]]; then
      run_cmd "Instalando dependencia ausente: albumentations" \
        python -m pip install -U albumentations || return 1
      have_python_module "albumentations" || return 1
      return 0
    fi

    log "ERRO: modulo 'albumentations' nao encontrado no ambiente atual."
    log "Instale e rode novamente: python -m pip install -U albumentations"
    log "Ou rode com instalacao automatica: INSTALL_MISSING=1 bash run_ablation_stage34.sh"
    return 1
  fi

  return 0
}

train_variant() {
  local variant="$1"
  shift || true
  local extra_flags=("$@")

  log "TREINO variante=$variant | epochs=$EPOCHS | flags=${extra_flags[*]:-(nenhuma)}"

  local feature src dst
  for feature in "${FEATURES[@]}"; do
    run_cmd "Treinando $feature ($variant)" \
      python -u 03_finetune_sam.py --feature "$feature" --epochs "$EPOCHS" "${extra_flags[@]}" || return 1

    src="models/sam_finetuned_${feature}_best.pth"
    dst="$(variant_ckpt_path "$variant" "$feature")"

    if [[ ! -f "$src" ]]; then
      log "ERRO: checkpoint nao encontrado apos treino: $src"
      return 1
    fi

    cp "$src" "$dst"
    log "Checkpoint salvo: $dst"
  done

  return 0
}

train_all_variants() {
  local variant
  for variant in "${VARIANTS[@]}"; do
    if [[ "$FORCE_RETRAIN" != "1" ]] && variant_ready "$variant"; then
      log "Treino pulado para $variant (checkpoints ja existem)."
      continue
    fi

    ensure_dependencies_for_variant "$variant" || return 1

    case "$variant" in
      hq_only)
        train_variant "$variant" || return 1
        ;;
      hq_aug)
        train_variant "$variant" --augment || return 1
        ;;
      hq_aug_lora)
        train_variant "$variant" --augment --lora || return 1
        ;;
      *)
        log "ERRO: variante desconhecida: $variant"
        return 1
        ;;
    esac
  done

  return 0
}

activate_variant_checkpoints() {
  local variant="$1"
  local feature src dst

  for feature in "${FEATURES[@]}"; do
    src="$(variant_ckpt_path "$variant" "$feature")"
    dst="models/sam_finetuned_${feature}_best.pth"

    if [[ ! -f "$src" ]]; then
      log "ERRO: checkpoint da variante ausente: $src"
      return 1
    fi

    cp "$src" "$dst"
  done

  return 0
}

eval_variant() {
  local variant="$1"

  log "INFERENCIA + VALIDACAO variante=$variant"
  activate_variant_checkpoints "$variant" || return 1

  # Mesmos parametros para todas as variantes
  run_cmd "Infer lakes ($variant)" \
    python -u 04_inference.py --feature lakes --year 2016 --annotated-only \
      --combine-mode max --pred-iou-threshold 0.60 --threshold 0.60 \
      --lakes-blue-ratio 1.25 --lakes-dark-brightness 75 --lakes-max-brightness 195 || return 1

  run_cmd "Infer crevasses ($variant)" \
    python -u 04_inference.py --feature crevasses --year 2016 --annotated-only \
      --combine-mode max --pred-iou-threshold 0.60 --threshold 0.55 \
      --crevasses-max-brightness 150 --crevasses-min-aspect 3.5 || return 1

  run_cmd "Infer channels ($variant)" \
    python -u 04_inference.py --feature channels --year 2016 --annotated-only \
      --combine-mode max --pred-iou-threshold 0.60 --threshold 0.50 \
      --channels-min-aspect 5.0 || return 1

  run_cmd "Validate all ($variant)" python -u 06_validate.py --year 2016 || return 1

  cp results/validation_results.json "$ABL_DIR/${variant}.json"
  log "Resultado salvo: $ABL_DIR/${variant}.json"

  return 0
}

eval_all_variants() {
  local variant
  for variant in "${VARIANTS[@]}"; do
    if [[ "$FORCE_REEVAL" != "1" ]] && result_ready "$variant"; then
      log "Avaliacao pulada para $variant (JSON ja existe)."
      continue
    fi

    if ! variant_ready "$variant"; then
      log "ERRO: variante $variant sem checkpoints completos para avaliar."
      return 1
    fi

    eval_variant "$variant" || return 1
  done

  return 0
}

sanity_check_lora() {
  if ! variant_ready "hq_aug_lora"; then
    log "Sanity check LoRA pulado (hq_aug_lora incompleto)."
    return 0
  fi

  log "Sanity check LoRA na variante hq_aug_lora"
  python -u - <<'PY'
import glob
import torch

paths = sorted(glob.glob("models/ablation_stage34/hq_aug_lora/*.pth"))
if not paths:
    raise SystemExit("Nenhum checkpoint encontrado em models/ablation_stage34/hq_aug_lora")

for p in paths:
    ck = torch.load(p, map_location="cpu")
    lora_state = ck.get("lora_state_dict", {})
    print(p, "| use_lora=", ck.get("use_lora"), "| lora_state_dict keys=", len(lora_state))
PY
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "ERRO: sanity check LoRA falhou"
    return "$rc"
  fi
  return 0
}

compare_results() {
  log "Comparacao final HQ vs HQ+Aug vs HQ+Aug+LoRA"
  python -u - <<'PY'
import json
from pathlib import Path

root = Path("results/ablation_stage34")
exp = {
    "HQ": root / "hq_only.json",
    "HQ+Aug": root / "hq_aug.json",
    "HQ+Aug+LoRA": root / "hq_aug_lora.json",
}

for _, p in exp.items():
    if not p.exists():
        raise SystemExit(f"Arquivo ausente: {p}")

data = {}
for k, p in exp.items():
    arr = json.load(open(p))
    data[k] = {r["feature"]: r["micro"] for r in arr}

features = ["lakes", "crevasses", "channels"]
print("feature | HQ_F1 -> Aug_F1 (d) -> AugLoRA_F1 (d) | HQ_IoU -> Aug_IoU (d) -> AugLoRA_IoU (d)")
for f in features:
    hq, aug, lora = data["HQ"][f], data["HQ+Aug"][f], data["HQ+Aug+LoRA"][f]
    print(
        f"{f:10s} | "
        f"{hq['f1']:.4f} -> {aug['f1']:.4f} ({aug['f1']-hq['f1']:+.4f}) -> {lora['f1']:.4f} ({lora['f1']-hq['f1']:+.4f}) | "
        f"{hq['iou']:.4f} -> {aug['iou']:.4f} ({aug['iou']-hq['iou']:+.4f}) -> {lora['iou']:.4f} ({lora['iou']-hq['iou']:+.4f})"
    )
PY
}

main() {
  log "Inicio da ablation Stage 3/4"
  log "Config: EPOCHS=$EPOCHS FORCE_RETRAIN=$FORCE_RETRAIN FORCE_REEVAL=$FORCE_REEVAL INSTALL_MISSING=$INSTALL_MISSING"

  backup_active_models

  train_all_variants || return 1
  sanity_check_lora || return 1
  eval_all_variants || return 1
  compare_results || return 1

  log "Ablation concluida com sucesso"
  return 0
}

if main; then
  rc=0
  log "Finalizado com sucesso."
else
  rc=$?
  log "Finalizado com erro (exit=$rc). Veja logs acima."
fi

exit "$rc"

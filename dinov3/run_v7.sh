#!/usr/bin/env bash
# Treina DINOv3-Small + LoRA nos 5 folds do v7 (mesma receita do v6) e agrega.
cd "$(dirname "$0")"
export CBEB_DATA_ROOT=~/Repos/CLAUCIA-ML/datasets/dataset_wagner_v7_balanced CBEB_RESULTS=$PWD/resultados_v7 CBEB_E1_ROOT=none
PY=${PY:-~/.cache/cbeb-train/bin/python}
for fold in 0 1 2 3 4; do
  [ -f resultados_v7/small_lora/fold$fold/summary.json ] && continue
  $PY train_dinov3.py --size small --fold $fold --adaptation lora > logs/v7_small_lora_fold$fold.log 2>&1 \
    && echo "[$(date '+%d/%m %H:%M')] OK fold$fold" || echo "[$(date '+%d/%m %H:%M')] FALHOU fold$fold"
done
$PY aggregate_results.py > logs/aggregate_v7.log 2>&1 && echo "[$(date '+%d/%m %H:%M')] V7 CONCLUIDO"

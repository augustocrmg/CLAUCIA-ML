#!/usr/bin/env bash
# Ablação de adaptação no v7: só a cabeça (backbone congelado) e fine-tuning completo.
cd "$(dirname "$0")"
export CBEB_DATA_ROOT=~/Repos/CLAUCIA-ML/datasets/dataset_wagner_v7_balanced CBEB_RESULTS=$PWD/resultados_v7 CBEB_E1_ROOT=none
PY=${PY:-~/.cache/cbeb-train/bin/python}
for adapt in head full; do for fold in 0 1 2 3 4; do
  [ -f resultados_v7/small_$adapt/fold$fold/summary.json ] && continue
  $PY train_dinov3.py --size small --fold $fold --adaptation $adapt > logs/v7_small_${adapt}_fold$fold.log 2>&1 \
    && echo "[$(date '+%d/%m %H:%M')] OK $adapt fold$fold" || echo "[$(date '+%d/%m %H:%M')] FALHOU $adapt fold$fold"
done; done
$PY aggregate_results.py > logs/aggregate_v7.log 2>&1 && echo "[$(date '+%d/%m %H:%M')] ABLACAO V7 CONCLUIDA"

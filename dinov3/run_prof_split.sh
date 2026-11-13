#!/usr/bin/env bash
# (1) Predições dos modelos do professor na validação do v4 (para temperature scaling, A5).
# (2) Ablação de adaptação (cabeça / LoRA / completo) no split original do professor (A6).
cd "$(dirname "$0")"
PY=${PY:-~/.cache/cbeb-train/bin/python}
V=~/Repos/CLAUCIA-ML/datasets/perfect_balanced_dataset_wagner_v4_1000
for s in small base; do
  [ -f resultados_prof_val_$s.csv ] || $PY eval_prof.py --size $s --root $V/validacao --out resultados_prof_val_$s.csv > logs/prof_val_$s.log 2>&1
  echo "[$(date '+%d/%m %H:%M')] val $s ok"
done
export CBEB_DATA_ROOT=~/Repos/CLAUCIA-ML/datasets/v4_split_prof CBEB_RESULTS=$PWD/resultados_prof_split CBEB_E1_ROOT=none
for adapt in head lora full; do
  [ -f resultados_prof_split/small_$adapt/fold0/summary.json ] && continue
  $PY train_dinov3.py --size small --fold 0 --adaptation $adapt > logs/profsplit_small_$adapt.log 2>&1 \
    && echo "[$(date '+%d/%m %H:%M')] OK $adapt" || echo "[$(date '+%d/%m %H:%M')] FALHOU $adapt"
done
echo "[$(date '+%d/%m %H:%M')] PROF SPLIT CONCLUIDO"

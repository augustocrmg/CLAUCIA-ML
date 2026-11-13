#!/usr/bin/env bash
cd "$(dirname "$0")"
export CBEB_DATA_ROOT=~/Repos/CLAUCIA-ML/datasets/dataset_wagner_v7_balanced CBEB_RESULTS=$PWD/resultados_v7 CBEB_TFLITE=$PWD/tflite_v7 CBEB_E2=none CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
for fold in 0 1 2 3 4; do
  ${PY_EXPORT:-~/.cache/cbeb-export/bin/python} export_tflite.py --config small_lora --fold $fold >> logs/export_v7.log 2>&1 \
    && echo "[$(date '+%d/%m %H:%M')] export fold$fold OK" || echo "[$(date '+%d/%m %H:%M')] export fold$fold FALHOU"
  rm -f tflite_v7/small_lora/fold$fold/*.tflite
done
echo "[$(date '+%d/%m %H:%M')] EXPORT V7 CONCLUIDO"

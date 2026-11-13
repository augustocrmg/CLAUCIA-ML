#!/usr/bin/env bash
# Exporta os checkpoints do professor (sem retreino) para LiteRT e mede acurácia de cada variante no teste dele.
cd "$(dirname "$0")"
while [ ! -f resultados_prof_val_base.csv ]; do sleep 60; done   # espera a avaliação de validação liberar a CPU
export CBEB_DATA_ROOT=~/Repos/CLAUCIA-ML/datasets/v4_split_prof CBEB_RESULTS=$PWD/resultados_prof_ckpt CBEB_TFLITE=$PWD/tflite_prof CBEB_E2=none CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
for cfg in small_lora base_lora; do
  ${PY_EXPORT:-~/.cache/cbeb-export/bin/python} export_tflite.py --config $cfg --fold 0 >> logs/export_prof.log 2>&1 \
    && echo "[$(date '+%d/%m %H:%M')] export $cfg OK" || echo "[$(date '+%d/%m %H:%M')] export $cfg FALHOU"
done
echo "[$(date '+%d/%m %H:%M')] EXPORT PROF CONCLUIDO"

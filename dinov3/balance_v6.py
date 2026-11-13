"""Balanceia o dataset_wagner_v6 por fold com o pipeline original do CLAUCIA-ML.

Usa as mesmas 10 transformações e a mesma divisão de imagens por técnica de
utils/balanceamento_dataset_script_v3.py (balance_split), com duas diferenças:
  - a augmentation entra só no treino; validação e teste ficam com originais;
  - a semente é fixa por fold.

Para o fold k: teste = fold k, validação = fold (k+1) % 5, treino = demais.
Saída: ~/Repos/CLAUCIA-ML/datasets/dataset_wagner_v6_balanced/fold{k}/{treino,validacao,teste}/<classe>/
Uso: python balance_v6.py [--target 700] [--folds 0 1 2 3 4]
"""
import argparse
import os
import random
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cnn_baselines", "utils"))
from balanceamento_dataset_script_v3 import balance_split, count_images_in_dir  # noqa: E402

V6 = os.path.expanduser(os.environ.get("CBEB_SRC", "~/Repos/CLAUCIA-ML/datasets/dataset_wagner_v6"))
OUT = os.path.expanduser(os.environ.get("CBEB_OUT", "~/Repos/CLAUCIA-ML/datasets/dataset_wagner_v6_balanced"))
N_FOLDS = 5
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=int, default=700, help="imagens por classe no treino (artigo: 700)")
parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
args = parser.parse_args()

manifest = pd.read_csv(os.path.join(V6, "manifest.csv"))
cv = manifest[manifest.split == "cv"]

for k in args.folds:
    random.seed(SEED + k)
    np.random.seed(SEED + k)
    val_fold = (k + 1) % N_FOLDS
    roles = {"teste": cv.fold == k, "validacao": cv.fold == val_fold,
             "treino": ~cv.fold.isin([k, val_fold])}

    fold_dir = os.path.join(OUT, f"fold{k}")
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    for split, mask in roles.items():
        for row in cv[mask].itertuples():
            target = os.path.join(fold_dir, split, row.label, os.path.basename(row.file))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(V6, row.file), target)

    # balance_split aumenta in-place a pasta do split, como no pipeline original.
    balance_split(os.path.join(fold_dir, "treino"), args.target)

    summary = {split: {c: count_images_in_dir(os.path.join(fold_dir, split, c))
                       for c in sorted(os.listdir(os.path.join(fold_dir, split)))}
               for split in roles}
    print(f"\n### fold {k} (teste={k}, validação={val_fold})")
    print(pd.DataFrame(summary).to_string())

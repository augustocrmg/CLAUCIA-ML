"""Monta o dataset_wagner_v7: tudo o que temos com rótulo Wagner compatível.

Composição (uma imagem por cluster de duplicatas, sem clusters com rótulo conflitante):
  - v4: recortes 224x224 (Graus 0-3, Normal) e fotos (Graus 4-5) rotulados pela
    profissional de saúde (base do artigo);
  - Archive: fotos inteiras 640x480, Graus 0-5;
  - Mendeley: pés saudáveis inteiros (CC BY 4.0), na classe Normal.
Assim cada classe mistura recortes e fotos inteiras, o que reduz o atalho de formato.

Agrupamento para os folds: cluster de duplicatas (md5 + dHash invariante à
orientação) e, para os recortes do v4, a foto original do Laith de onde saíram
(patch_to_original.csv, por SIFT). Recortes da mesma foto nunca caem em folds
diferentes. Fontes do Kaggle com rótulo de outra convenção (ADPM, Stages,
Roboflow: concordância com v4 de 0,12-0,16) ficam de fora.

Uso: python build_v7.py  ->  ~/Repos/CLAUCIA-ML/datasets/dataset_wagner_v7/{cv/fold*/<classe>, manifest.csv}
"""
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.expanduser("~/Repos/CLAUCIA-ML/datasets")
OUT = os.path.join(DS, "dataset_wagner_v7")
SEED = 42
N_FOLDS = 5
N_MENDELEY = 150
LABEL_MAP = {f"W{i}": f"Grade {i}" for i in range(6)} | {"Normal_foot": "Normal", "Normal_patch": "Normal"}

df = pd.read_parquet(os.path.join(HERE, "index_labeled.parquet"))
curated = df[df.source.isin(["v4", "archive"])]
labels = curated.groupby("cluster").label.agg(lambda s: frozenset(s))

# Foto original do Laith para cada recorte do v4 (via manifest do v6 + SIFT).
man6 = pd.read_csv(os.path.join(DS, "dataset_wagner_v6", "manifest.csv"))
p2o = pd.read_csv(os.path.join(HERE, "patch_to_original.csv")).dropna(subset=["original"])
e1 = man6[man6.split == "external_e1"].merge(p2o, on="file")
photo_of_v4 = dict(zip(e1.origin_path, e1.original))
cluster_of_path = dict(zip(df.path, df.cluster))


def representative(group):
    group = group.assign(area=group.w * group.h).sort_values(["area", "path"], ascending=[False, True])
    return group.iloc[0]


rows, conflicts = [], 0
for cluster, group in curated.groupby("cluster"):
    if len(labels[cluster]) != 1:
        conflicts += 1
        continue
    rep = representative(group)
    photos = {photo_of_v4.get(p) for p in group.path} - {None}
    rows.append(dict(cluster=cluster, path=rep.path, source=rep.source, w=rep.w, h=rep.h,
                     label=LABEL_MAP[next(iter(labels[cluster]))],
                     photo=next(iter(sorted(photos)), None)))
core = pd.DataFrame(rows)

mendeley = df[(df.source == "mendeley") & (df.label == "Normal_foot")]
mc = mendeley.cluster.unique()
np.random.default_rng(SEED).shuffle(mc)
for c in mc[:N_MENDELEY]:
    r = representative(mendeley[mendeley.cluster == c])
    core.loc[len(core)] = dict(cluster=c, path=r.path, source="mendeley", w=r.w, h=r.h, label="Normal", photo=None)

# Grupo = cluster da foto de origem (quando houver) ou o próprio cluster.
# Union-find para juntar clusters que compartilham a mesma foto.
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for r in core.itertuples():
    find(r.cluster)
    if isinstance(r.photo, str):
        a, b = find(r.cluster), find(("photo", cluster_of_path.get(r.photo, r.photo)))
        parent[a] = b
core["group"] = [str(find(c)) for c in core.cluster]

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
core["fold"] = -1
for fold, (_, idx) in enumerate(sgkf.split(core, core.label, core.group)):
    core.loc[core.index[idx], "fold"] = fold
core["split"] = "cv"

if os.path.exists(OUT):
    shutil.rmtree(OUT)
dest = []
for r in core.itertuples():
    target = os.path.join(OUT, f"cv/fold{r.fold}", r.label, f"{r.source}_{r.cluster}{os.path.splitext(r.path)[1].lower()}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(r.path, target)
    dest.append(os.path.relpath(target, OUT))
core["file"] = dest
core.rename(columns={"path": "origin_path"}).to_csv(os.path.join(OUT, "manifest.csv"), index=False)

assert core.groupby("group").fold.nunique().max() == 1
print("clusters descartados por conflito:", conflicts)
print("recortes do v4 com foto de origem conhecida:", core.photo.notna().sum(),
      "| grupos com >1 imagem:", (core.group.value_counts() > 1).sum())
print(pd.crosstab([core.label], [core.source]).to_string())
print(pd.crosstab(core.label, core.fold).to_string())
print("Total:", len(core))

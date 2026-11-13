"""Monta o dataset_wagner_v6 a partir do índice deduplicado (index_labeled.parquet).

Uso: rodar index_all.py -> cluster.py -> labels.py -> build_v6.py

Composição:
  - Núcleo (CV): fotos inteiras da pasta Archive (Wagner 0-5, 640x480), uma
    imagem por cluster de duplicatas, sem clusters com rótulo conflitante.
  - Normal: pés saudáveis inteiros do Mendeley (CC BY 4.0, 1.379 participantes),
    amostrados para ficar na escala das demais classes.
  - Externo E1: recortes 224x224 do v4 (Graus 0-3 e Normal) rotulados pela
    profissional de saúde, de outra fonte e sem cluster em comum com o núcleo.
Folds: 5 folds estratificados por classe; cada cluster é uma linha, então
duplicatas nunca caem em folds diferentes.
"""
import os, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("~/Repos/CLAUCIA-ML/datasets/dataset_wagner_v6")
SEED = 42
N_NORMAL_CORE = 150
N_FOLDS = 5
CLASSES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Normal"]
LABEL_MAP = {f"W{i}": f"Grade {i}" for i in range(6)} | {"Normal_foot": "Normal", "Normal_patch": "Normal"}

df = pd.read_parquet(os.path.join(HERE, "index_labeled.parquet"))
rng = np.random.default_rng(SEED)

# Rótulos dos conjuntos curados pela equipe (v4 + Archive), por cluster.
curated = df[df.source.isin(["v4", "archive"])]
curated_labels = curated.groupby("cluster").label.agg(lambda s: frozenset(s))


def representative(group: pd.DataFrame) -> pd.Series:
    # Maior resolução; empate resolvido pelo caminho, para ser determinístico.
    group = group.assign(area=group.w * group.h).sort_values(["area", "path"], ascending=[False, True])
    return group.iloc[0]


# Núcleo: Archive com rótulo único também entre v4 e Archive.
archive = df[df.source == "archive"]
core_rows = []
dropped_conflict = 0
for cluster, group in archive.groupby("cluster"):
    labels = curated_labels[cluster]
    if len(labels) != 1:
        dropped_conflict += 1
        continue
    rep = representative(group)
    core_rows.append(dict(cluster=cluster, path=rep.path, source="archive", w=rep.w, h=rep.h,
                          label=LABEL_MAP[next(iter(labels))]))
core = pd.DataFrame(core_rows)

# Normal: Mendeley, amostrado.
mendeley = df[(df.source == "mendeley") & (df.label == "Normal_foot")]
normal_clusters = mendeley.cluster.unique()
rng.shuffle(normal_clusters)
normal_core = [representative(mendeley[mendeley.cluster == c]) for c in normal_clusters[:N_NORMAL_CORE]]
core = pd.concat([core, pd.DataFrame([dict(cluster=r.cluster, path=r.path, source="mendeley", w=r.w, h=r.h,
                                            label="Normal") for r in normal_core])], ignore_index=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
core["fold"] = -1
for fold, (_, test_idx) in enumerate(skf.split(core, core.label)):
    core.loc[test_idx, "fold"] = fold
core["split"] = "cv"

# Externo E1: recortes do v4, sem cluster no núcleo e sem conflito de rótulo.
core_clusters = set(core.cluster)
v4 = df[(df.source == "v4") & df.label.isin(["W0", "W1", "W2", "W3", "Normal_patch"])]
ext_rows = []
for cluster, group in v4.groupby("cluster"):
    if cluster in core_clusters or len(curated_labels[cluster]) != 1:
        continue
    rep = representative(group)
    ext_rows.append(dict(cluster=cluster, path=rep.path, source="v4_patch", w=rep.w, h=rep.h,
                         label=LABEL_MAP[rep.label], fold=-1, split="external_e1"))
manifest = pd.concat([core, pd.DataFrame(ext_rows)], ignore_index=True)

# Cópia física, com nomes estáveis por cluster.
if os.path.exists(OUT):
    shutil.rmtree(OUT)
dest = []
for row in manifest.itertuples():
    sub = f"cv/fold{row.fold}" if row.split == "cv" else row.split
    name = f"{row.source}_{row.cluster}{os.path.splitext(row.path)[1].lower()}"
    target = os.path.join(OUT, sub, row.label, name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(row.path, target)
    dest.append(os.path.relpath(target, OUT))
manifest["file"] = dest
manifest = manifest.rename(columns={"path": "origin_path"})
manifest.to_csv(os.path.join(OUT, "manifest.csv"), index=False)

print("clusters do Archive descartados por conflito:", dropped_conflict)
print(pd.crosstab(manifest.label, [manifest.split, manifest.fold]).to_string())
print("\nTotal:", len(manifest), "| CV:", (manifest.split == "cv").sum(), "| E1:", (manifest.split == "external_e1").sum())

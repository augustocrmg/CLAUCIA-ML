"""Localiza cada recorte 224x224 do externo E1 na foto original do dataset do
Laith (DFU_QUTNet, "Original Images") de onde ele foi extraído.

Método: SIFT nas fotos originais (lado maior reduzido a 1024 px) e nos recortes;
para cada recorte, votação por correspondências (teste de razão de Lowe) e
verificação geométrica por homografia com RANSAC. Aceita a foto com mais
inliers se tiver >= MIN_INLIERS e folga sobre a segunda colocada.
Saída: patch_to_original.csv (recorte, foto, inliers, caixa na foto).
"""
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORIG_DIR = Path.home() / "Repos/CLAUCIA-ML/datasets/externos/laithjj_diabetic-foot-ulcer-dfu/DFU/Original Images"
V6 = Path.home() / "Repos/CLAUCIA-ML/datasets/dataset_wagner_v6"
MAX_SIDE = 1024
MIN_INLIERS = 12
TOP_CANDIDATES = 5


def sift_image(path, max_side=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    scale = 1.0
    if max_side and max(img.shape) > max_side:
        scale = max_side / max(img.shape)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kp, des = cv2.SIFT_create(nfeatures=4000).detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kp]) / scale if kp else np.zeros((0, 2), np.float32)
    return str(path), pts, des, img.shape


def original_features():
    paths = sorted(p for p in ORIG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    with ProcessPoolExecutor(16) as ex:
        return list(ex.map(sift_image, paths, [MAX_SIDE] * len(paths), chunksize=8))


def main():
    originals = [o for o in original_features() if o[2] is not None]
    owner = np.concatenate([np.full(len(o[2]), i) for i, o in enumerate(originals)])
    all_des = np.concatenate([o[2] for o in originals]).astype(np.float32)
    all_pts = np.concatenate([o[1] for o in originals])
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    flann.add([all_des]); flann.train()

    manifest = pd.read_csv(V6 / "manifest.csv")
    e1 = manifest[manifest.split == "external_e1"]
    rows = []
    for r in e1.itertuples():
        _, ppts, pdes, pshape = sift_image(V6 / r.file)
        row = dict(file=r.file, label=r.label, original=None, inliers=0, second=0, box=None)
        if pdes is not None and len(pdes) >= 8:
            good = [m for m, n in flann.knnMatch(pdes.astype(np.float32), k=2) if m.distance < 0.75 * n.distance]
            votes = Counter(owner[m.trainIdx] for m in good)
            scored = []
            for cand, _ in votes.most_common(TOP_CANDIDATES):
                ms = [m for m in good if owner[m.trainIdx] == cand]
                if len(ms) < 4:
                    continue
                src = np.float32([ppts[m.queryIdx] for m in ms])
                dst = np.float32([all_pts[m.trainIdx] for m in ms])
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
                if H is None:
                    continue
                corners = cv2.perspectiveTransform(np.float32([[0, 0], [pshape[1], 0], [pshape[1], pshape[0]], [0, pshape[0]]]).reshape(-1, 1, 2), H)
                scored.append((int(mask.sum()), cand, corners.reshape(-1, 2)))
            scored.sort(key=lambda t: -t[0])
            if scored:
                row.update(inliers=scored[0][0], second=scored[1][0] if len(scored) > 1 else 0)
                if scored[0][0] >= MIN_INLIERS and scored[0][0] >= 2 * row["second"]:
                    x0, y0 = scored[0][2].min(0); x1, y1 = scored[0][2].max(0)
                    row.update(original=originals[scored[0][1]][0], box=f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "patch_to_original.csv", index=False)
    print("recortes localizados:", out.original.notna().sum(), "de", len(out))
    print(out.groupby("label").original.apply(lambda s: f"{s.notna().sum()}/{len(s)}").to_string())
    found = out.dropna(subset=["original"])
    per_photo = found.groupby("original").label.agg(lambda s: "/".join(sorted(set(s))))
    print("\nfotos originais distintas:", len(per_photo))
    print(per_photo.value_counts().to_string())


if __name__ == "__main__":
    main()

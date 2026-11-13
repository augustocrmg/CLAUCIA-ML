"""Carrega os modelos treinados pelo professor (ultimos_treinamentos/) e avalia
em pastas de imagens no formato <raiz>/<classe>/<arquivo>.

Uso: python eval_prof.py --size small --root <pasta> --out <csv>
Sem --root: reproduz as predições do teste original (checagem do carregamento).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_dinov3 import (LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET_MODULES, MODEL_IDS,  # noqa: E402
                          DINOv3Classifier, eval_transform)

PROF = Path(os.environ.get("CLAUCIA_PROF_RESULTS", HERE / "results" / "divisao_original"))
EXP = "lora_qv_r8_warmup_v3_stable_large"
PROF_CLASSES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Normal(Healthy skin)"]
V4_SPLIT = Path(os.environ.get("CLAUCIA_V4_SPLIT", HERE.parent / "datasets" / "perfect_balanced_dataset_wagner_v4_1000"))


def load_prof(size: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(MODEL_IDS[size], dtype=torch.float32, attn_implementation="eager")
    backbone = get_peft_model(backbone, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                   target_modules=LORA_TARGET_MODULES, bias="none"))
    model = DINOv3Classifier(backbone, backbone.config.hidden_size)
    sd = torch.load(PROF / size / EXP / "best_trainable_checkpoint.pt", map_location="cpu")["state_dict"]
    own = model.state_dict()
    extra = [k for k in sd if k not in own]
    assert not extra, extra[:5]
    missing_trainable = [k for k in own if ('lora_' in k or k.startswith('classifier')) and k not in sd]
    assert not missing_trainable, missing_trainable[:5]
    model.load_state_dict(sd, strict=False)
    return model.eval()


@torch.inference_mode()
def predict(model, files):
    from PIL import Image
    probs = []
    for i in range(0, len(files), 32):
        x = torch.stack([eval_transform(Image.open(f).convert("RGB")) for f in files[i:i + 32]])
        probs.append(torch.softmax(model(x), -1).numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(16)
    model = load_prof(args.size)
    if args.root is None:  # checagem: compara com test_predictions.csv do professor
        ref = pd.read_csv(PROF / args.size / EXP / "test_predictions.csv").head(args.limit or 64)
        files = [str(V4_SPLIT) + a.split("perfect_balanced_dataset_wagner_v4_1000")[1] for a in ref.arquivo]
        p = predict(model, files)
        refp = ref[[c for c in ref.columns if c.startswith("prob_")]].values
        print("max |Δprob| vs. professor:", float(np.abs(p - refp).max()),
              "| concordância argmax:", float((p.argmax(1) == refp.argmax(1)).mean()))
        return
    root = Path(args.root)
    files = sorted(str(f) for f in root.glob("*/*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    p = predict(model, files)
    df = pd.DataFrame({"arquivo": files, "classe_real": [Path(f).parent.name for f in files]})
    for i, c in enumerate(PROF_CLASSES):
        df[f"prob_{c}"] = p[:, i]
    df.to_csv(args.out, index=False)
    print("salvo", args.out, len(df))


if __name__ == "__main__":
    main()

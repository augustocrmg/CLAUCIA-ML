"""Treino DINOv3 + LoRA por fold no dataset_wagner_v6_balanced.

Reproduz o notebook treinamento_dinov3_lora_pe_diabetico_v4.ipynb (mesmos
hiperparâmetros, aquecimento da cabeça, ReduceLROnPlateau, early stopping pelo
F1 macro de validação, bf16 autocast). Diferenças: roda local (ROCm), sem
torchvision (transformações equivalentes em PIL) e avalia também o externo E1.

--adaptation define a ablação (A6):
  lora : adaptadores LoRA + cabeça (configuração do artigo)
  head : backbone congelado, só a cabeça (linear probe)
  full : fine-tuning completo do backbone + cabeça

Uso: python train_dinov3.py --size small --fold 0 [--adaptation lora]
"""
import argparse
import json
import os

# Cópias DMA assíncronas (SDMA) também são fonte conhecida de page fault no ROCm.
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pesos LVD-1689M convertidos do timm por convert_timm_dinov3.py (o repositório
# facebook/dinov3-* exige aprovação de acesso no Hugging Face).
HF_LOCAL = Path(__file__).resolve().parent / "modelos_hf"
MODEL_IDS = {
    "small": str(HF_LOCAL / "dinov3-vits16"),
    "base": str(HF_LOCAL / "dinov3-vitb16"),
}
CLASSES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Normal"]
NUM_CLASSES = len(CLASSES)
# Caminhos sobrescrevíveis por variável de ambiente (v7 usa CBEB_DATA_ROOT/CBEB_RESULTS;
# CBEB_E1_ROOT=none desliga o externo, que então repete o teste do fold).
DATA_ROOT = Path(os.environ.get("CBEB_DATA_ROOT", Path.home() / "Repos/CLAUCIA-ML/datasets/dataset_wagner_v6_balanced"))
E1_ROOT = os.environ.get("CBEB_E1_ROOT", str(Path.home() / "Repos/CLAUCIA-ML/datasets/dataset_wagner_v6/external_e1"))
RESULTS_ROOT = Path(os.environ.get("CBEB_RESULTS", Path(__file__).resolve().parent / "resultados"))

# Hiperparâmetros do notebook.
IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
MIN_DELTA = 1e-4
HEAD_WARMUP_EPOCHS = 3
LR_LORA = {"small": 5e-5, "base": 5e-5, "large": 1e-5}
LR_HEAD = 5e-5
LR_FULL = 1e-5
WEIGHT_DECAY = 5e-4
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
MIN_LR = 1e-6
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.20
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
CLASSIFIER_DROPOUT = 0.30
LABEL_SMOOTHING = 0.05
MAX_GRAD_NORM = 1.0
SEED = 42
# O notebook treinava com autocast bf16; no ROCm (RX 9070 XT) o backward em bf16
# gera page fault de GPU após ~850 passos, de forma reproduzível. Em fp32 não ocorre.
USE_BF16 = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Equivalentes de torchvision.transforms: Resize((224,224)) bilinear com
# antialias, RandomHorizontalFlip(0.5), RandomRotation(10) (nearest, fill 0),
# ToTensor e Normalize.
def to_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (x - mean) / std


def eval_transform(img: Image.Image) -> torch.Tensor:
    return to_tensor(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))


def train_transform(img: Image.Image) -> torch.Tensor:
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(random.uniform(-10, 10), resample=Image.NEAREST, expand=False, fillcolor=(0, 0, 0))
    return to_tensor(img)


class FolderDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.samples = [(p, CLASSES.index(c.name)) for c in sorted(root.iterdir()) if c.is_dir()
                        for p in sorted(c.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


class DINOv3Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(CLASSIFIER_DROPOUT), nn.Linear(hidden_size, NUM_CLASSES))

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        with torch.autocast(device_type=feats.device.type, enabled=False):
            return self.classifier(feats.float())


def binary_auc(y, s):
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    p, r, _ = precision_recall_curve(y, s)
    return float(roc_auc_score(y, s)), float(auc(r, p))


def macro_metrics(y_true, y_pred, y_prob):
    labels = list(range(NUM_CLASSES))
    present = sorted(set(y_true))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="macro", zero_division=0)
    yb = label_binarize(y_true, classes=labels)
    aucs = [binary_auc(yb[:, c], y_prob[:, c]) for c in labels]
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "auc_roc": float(np.nanmean([a for a, _ in aucs])), "auc_pr": float(np.nanmean([b for _, b in aucs]))}


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    targets, probs = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("loss não finita")
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # foreach=False: o kernel fundido _foreach_norm gerou acesso ilegal à memória no ROCm (gfx12).
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, foreach=False)
            if torch.isfinite(grad_norm):
                optimizer.step()
        targets.extend(y.cpu().tolist())
        probs.append(torch.softmax(logits.float(), 1).detach().cpu().numpy())
    probs = np.concatenate(probs)
    return macro_metrics(targets, probs.argmax(1).tolist(), probs), targets, probs


def predictions_frame(dataset, targets, probs):
    df = pd.DataFrame({"arquivo": [str(p) for p, _ in dataset.samples],
                       "classe_real": [CLASSES[t] for t in targets],
                       "classe_predita": [CLASSES[i] for i in probs.argmax(1)]})
    for i, c in enumerate(CLASSES):
        df[f"prob_{c}"] = probs[:, i]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=MODEL_IDS, default="small")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--adaptation", choices=["lora", "head", "full"], default="lora")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    seed_everything(SEED + args.fold)
    device = torch.device("cuda")
    out_dir = RESULTS_ROOT / f"{args.size}_{args.adaptation}" / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_root = DATA_ROOT / f"fold{args.fold}"

    datasets = {"train": FolderDataset(fold_root / "treino", train_transform),
                "val": FolderDataset(fold_root / "validacao", eval_transform),
                "test": FolderDataset(fold_root / "teste", eval_transform),
                "e1": FolderDataset(fold_root / "teste" if E1_ROOT == "none" else Path(E1_ROOT), eval_transform)}
    g = torch.Generator().manual_seed(SEED + args.fold)
    loaders = {k: DataLoader(d, batch_size=BATCH_SIZE, shuffle=(k == "train"), generator=g if k == "train" else None,
                             num_workers=args.workers, pin_memory=False, persistent_workers=True)
               for k, d in datasets.items()}

    # Atenção eager: os kernels SDPA (aotriton) causaram page fault na RX 9070 XT.
    backbone = AutoModel.from_pretrained(MODEL_IDS[args.size], dtype=torch.float32, attn_implementation="eager")
    hidden = backbone.config.hidden_size
    if args.adaptation == "lora":
        backbone = get_peft_model(backbone, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                       target_modules=LORA_TARGET_MODULES, bias="none"))
    model = DINOv3Classifier(backbone, hidden).to(device)

    head_params = list(model.classifier.parameters())
    if args.adaptation == "lora":
        bb_params = [p for n, p in model.named_parameters() if "lora_" in n]
        bb_lr = LR_LORA[args.size]
    elif args.adaptation == "full":
        bb_params = list(model.backbone.parameters())
        bb_lr = LR_FULL
    else:
        bb_params, bb_lr = [], 0.0
        for p in model.backbone.parameters():
            p.requires_grad = False
    groups = [{"params": bb_params, "lr": bb_lr, "weight_decay": WEIGHT_DECAY}] if bb_params else []
    groups.append({"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY})
    optimizer = torch.optim.AdamW(groups, foreach=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=LR_SCHEDULER_FACTOR,
                                                           patience=LR_SCHEDULER_PATIENCE, threshold=MIN_DELTA,
                                                           min_lr=MIN_LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    def set_backbone_trainable(flag):
        for p in bb_params:
            p.requires_grad = flag

    set_backbone_trainable(False)
    trainable_names = lambda n: "lora_" in n or n.startswith("classifier.") or args.adaptation == "full"
    n_trainable = sum(p.numel() for p in head_params) + sum(p.numel() for p in bb_params)
    n_total = sum(p.numel() for p in model.parameters())

    history, best_f1, stale, best_state = [], -math.inf, 0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        warmup = epoch <= HEAD_WARMUP_EPOCHS
        if epoch == HEAD_WARMUP_EPOCHS + 1 and bb_params:
            set_backbone_trainable(True)
            stale = 0
        te = time.time()
        tr, _, _ = run_epoch(model, loaders["train"], device, criterion, optimizer)
        with torch.inference_mode():
            va, _, _ = run_epoch(model, loaders["val"], device, criterion)
        history.append({"epoch": epoch, "phase": "warmup" if warmup else "adapt",
                        "lr_backbone": optimizer.param_groups[0]["lr"], "seconds": time.time() - te,
                        **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        improved = va["f1"] > best_f1 + MIN_DELTA
        if improved:
            best_f1, stale = va["f1"], 0
            best_state = {"epoch": epoch, "val": va,
                          "state_dict": {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                                         if trainable_names(n)}}
        else:
            stale += 1
        print(f"[{args.size}/{args.adaptation}/fold{args.fold}] ép {epoch:02d} {'W' if warmup else 'A'} "
              f"train_f1={tr['f1']:.3f} val_f1={va['f1']:.3f} val_acc={va['accuracy']:.3f} "
              f"{'*' if improved else ''} ({time.time() - te:.0f}s)", flush=True)
        if not warmup:
            scheduler.step(va["f1"])
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state["state_dict"], strict=False)
    torch.save(best_state, out_dir / "best_checkpoint.pt")
    summary = {"size": args.size, "adaptation": args.adaptation, "fold": args.fold, "model_id": MODEL_IDS[args.size],
               "best_epoch": best_state["epoch"], "best_val": best_state["val"], "epochs_run": len(history),
               "train_minutes": (time.time() - t0) / 60, "params_total": n_total, "params_trainable": n_trainable}
    with torch.inference_mode():
        for split in ["test", "e1"]:
            metrics, targets, probs = run_epoch(model, loaders[split], device, criterion)
            predictions_frame(datasets[split], targets, probs).to_csv(out_dir / f"predictions_{split}.csv", index=False)
            summary[split] = metrics
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["best_epoch", "train_minutes", "test", "e1"]}, indent=1))


if __name__ == "__main__":
    main()

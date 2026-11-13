"""Exporta um modelo treinado (resultados/<config>/fold<k>) para TFLite/LiteRT
em várias precisões e mede a acurácia de cada artefato no teste do fold.

Variantes:
  fp32        conversão direta
  fp16        pesos em float16 (flags do conversor)
  int8_wo     pesos int8, ativações fp32 (análogo ao torchao Int8WeightOnly)
  int8_dyn    pesos int8 e ativações quantizadas dinamicamente (XNNPACK)
  int8_static pesos e ativações int8, calibrado com originais do treino
O grafo recebe NCHW normalizado (ImageNet) e devolve probabilidades (softmax).

Ambiente: ~/.cache/cbeb-export (CPU, litert-torch).
Uso: python export_tflite.py --config small_lora --fold 0 [--variants fp32 int8_dyn]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_dinov3 import (CLASSES, DATA_ROOT, LORA_ALPHA, LORA_DROPOUT, LORA_RANK,  # noqa: E402
                          LORA_TARGET_MODULES, MODEL_IDS, RESULTS_ROOT, DINOv3Classifier, FolderDataset,
                          eval_transform)

VARIANTS = ["fp32", "fp16", "int8_wo", "int8_dyn", "int8_static"]
N_CALIB = 100
# CBEB_E2=none desliga o E2 (no v7 ele se sobrepõe ao núcleo, pelas fotos do Laith).
E2_ROOT = None if os.environ.get("CBEB_E2") == "none" else Path.home() / "Repos/CLAUCIA-ML/datasets/dataset_wagner_v6/external_e2"
TFLITE_ROOT = Path(os.environ.get("CBEB_TFLITE", HERE / "tflite"))


class Exportable(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=-1)


def load_trained(config: str, fold: int) -> nn.Module:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel
    size, adaptation = config.split("_")
    backbone = AutoModel.from_pretrained(MODEL_IDS[size], dtype=torch.float32, attn_implementation="eager")
    if adaptation == "lora":
        backbone = get_peft_model(backbone, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                       target_modules=LORA_TARGET_MODULES, bias="none"))
    model = DINOv3Classifier(backbone, backbone.config.hidden_size)
    ckpt = torch.load(RESULTS_ROOT / config / f"fold{fold}" / "best_checkpoint.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not unexpected, unexpected
    if adaptation == "lora":
        model.backbone = model.backbone.merge_and_unload()
    return Exportable(model.eval())


def calibration_batches(fold: int):
    ds = FolderDataset(DATA_ROOT / f"fold{fold}" / "treino", eval_transform)
    originals = [i for i, (p, _) in enumerate(ds.samples)
                 if not any(t in p.name for t in ("_scaled", "_rotated", "_flipped", "_brightened", "_darkened",
                                                  "_shifted", "_salt", "_gaussian"))]
    rng = np.random.default_rng(0)
    pick = rng.choice(originals, min(N_CALIB, len(originals)), replace=False)
    return [ds[i][0].unsqueeze(0).numpy() for i in pick]


def quantize(fp32_path: Path, variant: str, fold: int) -> bytes:
    from ai_edge_quantizer import quantizer, recipe
    fp16 = [{"regex": ".*", "operation": "*", "algorithm_key": "float_casting",
             "op_config": {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"},
                           "compute_precision": "FLOAT", "explicit_dequantize": True}}]
    rec = {"fp16": fp16, "int8_wo": recipe.weight_only_wi8_afp32(), "int8_dyn": recipe.dynamic_wi8_afp32(),
           "int8_static": recipe.static_wi8_ai8()}[variant]
    q = quantizer.Quantizer(str(fp32_path), rec)
    calib = None
    if q.need_calibration:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=str(fp32_path))
        runner = interp.get_signature_runner()
        input_name = next(iter(runner.get_input_details()))
        calib = q.calibrate({"serving_default": [{input_name: x} for x in calibration_batches(fold)]})
    return q.quantize(calib).quantized_model


def evaluate(tflite_path: Path, fold: int, root: Path = None):
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(tflite_path), num_threads=8)
    runner = interp.get_signature_runner()
    input_name, in_det = next(iter(runner.get_input_details().items()))
    out_det = next(iter(runner.get_output_details().values()))
    ds = FolderDataset(root or DATA_ROOT / f"fold{fold}" / "teste", eval_transform)
    probs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        x = x.unsqueeze(0).numpy()
        if in_det["dtype"] != np.float32:  # modelo inteiro: quantiza a entrada
            scale, zero = in_det["quantization"]
            info = np.iinfo(in_det["dtype"])
            x = np.clip(np.round(x / scale + zero), info.min, info.max).astype(in_det["dtype"])
        out = next(iter(runner(**{input_name: x}).values()))[0]
        if out_det["dtype"] != np.float32:
            scale, zero = out_det["quantization"]
            out = (out.astype(np.float32) - zero) * scale
        probs.append(out); ys.append(y)
    probs, ys = np.array(probs), np.array(ys)
    from sklearn.metrics import accuracy_score, f1_score
    present = sorted(set(ys.tolist()))
    return {"accuracy": float(accuracy_score(ys, probs.argmax(1))),
            "f1": float(f1_score(ys, probs.argmax(1), labels=present, average="macro")), "n": int(len(ys))}, (ds, ys, probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small_lora")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    args = ap.parse_args()

    import litert_torch
    out_dir = TFLITE_ROOT / args.config / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    module = load_trained(args.config, args.fold)
    sample = (torch.randn(1, 3, 224, 224),)
    with torch.no_grad():
        torch_probs = module(*sample).numpy()

    fp32_path = out_dir / "fp32.tflite"
    for variant in args.variants:
        t0 = time.time()
        path = out_dir / f"{variant}.tflite"
        try:
            if variant == "fp32":
                litert_torch.convert(module, sample).export(str(path))
            else:
                path.write_bytes(quantize(fp32_path, variant, args.fold))
            metrics, _ = evaluate(path, args.fold)
            report[variant] = {"size_mb": path.stat().st_size / 2**20, "seconds": time.time() - t0, **metrics}
            if variant in ("fp32", "int8_dyn") and E2_ROOT is not None:  # externo E2 (fotos completas, rótulo provisório)
                e2_metrics, (ds, ys, probs) = evaluate(path, args.fold, E2_ROOT)
                report[variant]["e2"] = e2_metrics
                import pandas as pd
                df = pd.DataFrame({"arquivo": [str(p) for p, _ in ds.samples], "classe_real": [CLASSES[y] for y in ys]})
                for i, c in enumerate(CLASSES):
                    df[f"prob_{c}"] = probs[:, i]
                df.to_csv(out_dir / f"predictions_e2_{variant}.csv", index=False)
            if variant == "fp32":
                from ai_edge_litert.interpreter import Interpreter
                runner = Interpreter(model_path=str(path)).get_signature_runner()
                name = next(iter(runner.get_input_details()))
                tfl = next(iter(runner(**{name: sample[0].numpy()}).values()))
                report[variant]["max_abs_diff_vs_torch"] = float(np.abs(tfl - torch_probs).max())
        except Exception as e:  # registra e segue para a próxima variante
            report[variant] = {"error": f"{type(e).__name__}: {e}"[:500]}
        print(variant, json.dumps(report[variant]), flush=True)
        report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

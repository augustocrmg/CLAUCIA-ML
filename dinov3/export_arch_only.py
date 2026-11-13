"""Exporta arquiteturas para medir latência no dispositivo (os pesos não alteram
o custo de inferência): DINOv3-Small/Base com LoRA já incorporada e
MobileNet-V2 (referência convolucional do artigo), nas mesmas variantes de
export_tflite.py. Saída: tflite/arch/<modelo>/<variante>.tflite
Ambiente: ~/.cache/cbeb-export.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_tflite import Exportable, quantize  # noqa: E402
from train_dinov3 import MODEL_IDS, NUM_CLASSES, DINOv3Classifier  # noqa: E402


def dinov3(size):
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(MODEL_IDS[size], dtype=torch.float32, attn_implementation="eager")
    return Exportable(DINOv3Classifier(backbone, backbone.config.hidden_size).eval())


def mobilenet_v2():
    m = torchvision.models.mobilenet_v2(num_classes=NUM_CLASSES)
    return Exportable(m.eval())


def main():
    import litert_torch
    sample = (torch.randn(1, 3, 224, 224),)
    for name, build in [("dinov3_small", lambda: dinov3("small")), ("dinov3_base", lambda: dinov3("base")),
                        ("mobilenet_v2", mobilenet_v2)]:
        out = HERE / "tflite" / "arch" / name
        out.mkdir(parents=True, exist_ok=True)
        fp32 = out / "fp32.tflite"
        if not fp32.exists():
            litert_torch.convert(build(), sample).export(str(fp32))
        for variant in ["fp16", "int8_wo", "int8_dyn", "int8_static"]:
            path = out / f"{variant}.tflite"
            if path.exists():
                continue
            try:
                # Calibração do int8_static usa originais do treino do fold 0.
                path.write_bytes(quantize(fp32, variant, 0))
            except Exception as e:
                print(name, variant, "ERRO", type(e).__name__, str(e)[:200])
        print(name, {p.name: round(p.stat().st_size / 2**20, 1) for p in sorted(out.glob("*.tflite"))}, flush=True)


if __name__ == "__main__":
    main()

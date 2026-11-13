"""Converte os pesos DINOv3 do timm (sem restrição de acesso) para o formato
DINOv3ViTModel do transformers, o mesmo usado no artigo (facebook/dinov3-*).

O repositório facebook/dinov3-* exige aprovação no Hugging Face; o timm publica
os mesmos pesos pré-treinados LVD-1689M. A conversão só renomeia tensores e
separa a projeção qkv fundida em q_proj, k_proj e v_proj. A validação compara a
saída do modelo convertido com a do timm na mesma entrada.

Uso: python convert_timm_dinov3.py  (gera modelos_hf/dinov3-vit{s,b}16)
"""
from pathlib import Path

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

OUT = Path(__file__).resolve().parent / "modelos_hf"
VARIANTS = {
    "dinov3-vits16": ("vit_small_patch16_dinov3.lvd1689m", dict(hidden_size=384, intermediate_size=1536, num_attention_heads=6)),
    "dinov3-vitb16": ("vit_base_patch16_dinov3.lvd1689m", dict(hidden_size=768, intermediate_size=3072, num_attention_heads=12)),
}


def convert(timm_sd: dict, n_layers: int, dim: int) -> dict:
    hf = {
        "embeddings.cls_token": timm_sd["cls_token"],
        "embeddings.register_tokens": timm_sd["reg_token"],
        "embeddings.patch_embeddings.weight": timm_sd["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.bias": timm_sd["patch_embed.proj.bias"],
        "norm.weight": timm_sd["norm.weight"],
        "norm.bias": timm_sd["norm.bias"],
    }
    for i in range(n_layers):
        t, h = f"blocks.{i}.", f"model.layer.{i}."
        q, k, v = timm_sd[t + "attn.qkv.weight"].split(dim, dim=0)
        qkv_bias = timm_sd.get(t + "attn.qkv.bias", torch.zeros(3 * dim))
        qb, _, vb = qkv_bias.split(dim)
        hf |= {
            h + "norm1.weight": timm_sd[t + "norm1.weight"], h + "norm1.bias": timm_sd[t + "norm1.bias"],
            h + "attention.q_proj.weight": q, h + "attention.q_proj.bias": qb,
            h + "attention.k_proj.weight": k,
            h + "attention.v_proj.weight": v, h + "attention.v_proj.bias": vb,
            h + "attention.o_proj.weight": timm_sd[t + "attn.proj.weight"],
            h + "attention.o_proj.bias": timm_sd[t + "attn.proj.bias"],
            h + "layer_scale1.lambda1": timm_sd[t + "gamma_1"],
            h + "norm2.weight": timm_sd[t + "norm2.weight"], h + "norm2.bias": timm_sd[t + "norm2.bias"],
            h + "mlp.up_proj.weight": timm_sd[t + "mlp.fc1.weight"], h + "mlp.up_proj.bias": timm_sd[t + "mlp.fc1.bias"],
            h + "mlp.down_proj.weight": timm_sd[t + "mlp.fc2.weight"], h + "mlp.down_proj.bias": timm_sd[t + "mlp.fc2.bias"],
            h + "layer_scale2.lambda1": timm_sd[t + "gamma_2"],
        }
    return hf


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    for name, (timm_name, dims) in VARIANTS.items():
        timm_sd = load_file(hf_hub_download(f"timm/{timm_name}", "model.safetensors"))
        config = DINOv3ViTConfig(num_register_tokens=timm_sd["reg_token"].shape[1], **dims)
        model = DINOv3ViTModel(config).eval()
        missing, unexpected = model.load_state_dict(convert(timm_sd, config.num_hidden_layers, dims["hidden_size"]), strict=False)
        assert set(missing) <= {"embeddings.mask_token"} and not unexpected, (missing, unexpected)

        reference = timm.create_model(timm_name, pretrained=True, num_classes=0).eval()
        with torch.no_grad():
            ref_tokens = reference.forward_features(x)
            hf_tokens = model(pixel_values=x).last_hidden_state
        diff = (ref_tokens - hf_tokens).abs().max().item()
        print(f"{name}: max |timm - hf| = {diff:.2e} (tokens {tuple(hf_tokens.shape)})")
        assert diff < 1e-4, "a conversão não reproduz o timm"
        model.save_pretrained(OUT / name)


if __name__ == "__main__":
    main()

# EfficientNet-B7 Memory Optimization Guide

## Problema Identificado

Seu erro de "Out of Memory" ocorreu devido a uma **combinação de limitação de hardware e configurações muito agressivas**:

### Hardware Detectado:
- **GPU**: NVIDIA GeForce RTX 2070 SUPER 
- **Memória GPU**: ~5.6 GB disponível

### Configurações Problemáticas Originais:
- **Imagem**: 600x600 pixels (muito grande)
- **Batch size**: 32 (muito alto para essa GPU)
- **Modelo**: EfficientNet-B7 (o maior da família EfficientNet)

### Cálculo do Uso de Memória Original:
- **Batch de imagens**: 32 × 600 × 600 × 3 × 4 bytes = ~1.3 GB
- **EfficientNet-B7**: ~244 MB de parâmetros
- **Ativações intermediárias**: 3-4 GB com essas dimensões
- **Gradientes e otimizador**: ~500 MB
- **Total estimado**: ~6-7 GB (excede sua GPU de 5.6 GB)

## Otimizações Implementadas

### 1. Configurações Ajustadas:
```python
IMAGE_SIZE = (380, 380)  # Reduzido de 600x600
BATCH_SIZE = 8           # Reduzido de 32
```

### 2. Mixed Precision Training:
```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```
- **Benefício**: Reduz uso de memória em ~40%
- **Impacto**: Mínima perda de precisão, treinamento mais rápido

### 3. XLA Compilation:
```python
tf.config.optimizer.set_jit(True)
```
- **Benefício**: Otimizações automáticas de memória e velocidade

### 4. Memory Growth:
```python
tf.config.experimental.set_memory_growth(gpu, True)
```
- **Benefício**: Aloca memória conforme necessário

### 5. Variáveis de Ambiente:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
```

## Estimativa de Memória Otimizada

Com as novas configurações:
- **Batch de imagens**: 8 × 380 × 380 × 3 × 2 bytes (FP16) = ~0.4 GB
- **EfficientNet-B7**: ~122 MB (FP16 para ativações)
- **Ativações intermediárias**: ~1-1.5 GB
- **Gradientes e otimizador**: ~300 MB
- **Total estimado**: ~2-2.5 GB ✅ (cabe na sua GPU)

## Como Usar

### Opção 1: Script Otimizado (Recomendado)
```bash
cd /home/augusto/Repos/CLAUCIA-ML/models/efficientnet_b7_600/
python run_optimized.py
```

### Opção 2: Script Original Modificado
```bash
cd /home/augusto/Repos/CLAUCIA-ML/models/efficientnet_b7_600/
python efficientnet_b7_600_mlflow.py
```

### Opção 3: Configuração Manual
Se ainda tiver problemas, execute com flags adicionais:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
python efficientnet_b7_600_mlflow.py
```

## Alternativas para GPUs Menores

Se sua GPU tiver menos de 4GB, considere:

### 1. EfficientNet-B3 ou B4:
```python
# Troque por um modelo menor
base_model = keras.applications.EfficientNetB3(...)  # B3 ou B4
IMAGE_SIZE = (300, 300)  # Tamanho adequado para B3/B4
BATCH_SIZE = 4
```

### 2. Gradient Accumulation:
```python
# Simula batch maior acumulando gradientes
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4  # Equivale a batch_size = 8
```

### 3. CPU Offloading:
```python
# Para casos extremos
with tf.device('/CPU:0'):
    # Processa alguns layers na CPU
```

## Comparação de Performance Esperada

| Configuração | Memória GPU | Tempo/Época | Qualidade |
|--------------|-------------|-------------|-----------|
| Original (600px, batch=32) | ~7GB ❌ | - | - |
| Otimizada (380px, batch=8) | ~2.5GB ✅ | +30% | -5% acc |
| B4 (320px, batch=8) | ~1.8GB ✅ | -20% | -8% acc |
| B3 (300px, batch=4) | ~1.2GB ✅ | -40% | -12% acc |

## Diagnóstico de Problemas

### Se ainda der OOM:
1. **Reduza batch size para 4**:
   ```python
   BATCH_SIZE = 4
   ```

2. **Reduza tamanho da imagem**:
   ```python
   IMAGE_SIZE = (320, 320)
   ```

3. **Use EfficientNet-B5**:
   ```python
   base_model = keras.applications.EfficientNetB5(...)
   ```

### Monitoramento de Memória:
```bash
# Durante o treinamento, monitore em outro terminal:
watch -n 1 nvidia-smi
```

### Logs Importantes:
- ✅ `Mixed precision training enabled`
- ✅ `GPU memory growth enabled`
- ❌ `RESOURCE_EXHAUSTED: Out of memory`
- ❌ `Allocator (GPU_0_bfc) ran out of memory`

## Conclusão

**O problema é principalmente de hardware limitado**, mas as otimizações implementadas devem permitir que você treine o EfficientNet-B7 na sua RTX 2070 SUPER com performance aceitável. 

A redução de 600px para 380px e batch de 32 para 8 são essenciais para sua GPU. O mixed precision training compensa parcialmente a perda de batch size, mantendo boa qualidade de treinamento. 
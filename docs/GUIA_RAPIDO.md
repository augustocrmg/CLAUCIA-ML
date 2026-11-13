# 🚀 GUIA RÁPIDO - GPU AMD com TensorFlow

## ✅ Setup Completo

```bash
# 1. Baixar imagem Docker (só uma vez)
docker pull rocm/tensorflow:latest

# 2. Testar GPU
./cnn_baselines/tools/run_tensorflow_docker.sh python test_gpu.py

# 3. Treinar seu modelo
./cnn_baselines/tools/run_tensorflow_docker_full.sh python seu_modelo.py
```

## 📂 Acesso aos Arquivos

**Todos os arquivos do projeto são acessíveis automaticamente!**

```python
# ✅ Funciona perfeitamente
dataset_path = "datasets/dataset_wagner"
model.save("models_trained/modelo.keras")
```

## �� Scripts Disponíveis

| Script | Uso | Quando usar |
|--------|-----|-------------|
| `run_tensorflow_docker.sh` | Básico | Código simples, sem deps extras |
| `run_tensorflow_docker_full.sh` | Completo | Carrega imagens, MLflow, OpenCV |

## 📖 Documentação Completa

- **Setup GPU**: `README_GPU_AMD.md`
- **Como treinar**: `COMO_TREINAR_NO_DOCKER.md`
- **Exemplo prático**: `exemplo_treino_docker.py`

## ⚡ Exemplo Mínimo

```python
import tensorflow as tf

# GPU detectada automaticamente
model = tf.keras.Sequential([...])
model.compile(...)

# Carregar dados (caminho relativo)
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/dataset_wagner",
    image_size=(224, 224)
)

# Treinar (usa GPU automaticamente)
model.fit(train_data, epochs=10)

# Salvar (aparece no host)
model.save("models_trained/modelo.keras")
```

## 🎉 É isso!

Seus modelos treinados aparecerão automaticamente no diretório do projeto.

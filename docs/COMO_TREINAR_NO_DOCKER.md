# Como Treinar Modelos no Docker com GPU AMD

## 📋 Resumo Rápido

Seus arquivos do projeto (datasets, modelos, scripts) são **automaticamente montados** dentro do Docker em `/workspace`.

## 🚀 3 Formas de Usar

### 1. Script Básico (Rápido)
```bash
./cnn_baselines/tools/run_tensorflow_docker.sh python seu_script.py
```

### 2. Script Full (Com dependências extras)
```bash
./cnn_baselines/tools/run_tensorflow_docker_full.sh python seu_script.py
```
Instala automaticamente: Pillow, OpenCV, MLflow

### 3. Bash Interativo
```bash
./cnn_baselines/tools/run_tensorflow_docker_full.sh
# Dentro do container:
python seu_script.py
```

---

## 📂 Estrutura de Arquivos

### No seu sistema (host):
```
/home/augusto/Repos/CLAUCIA-ML/
├── datasets/
│   ├── dataset_wagner/
│   ├── dataset_wagner_v4/
│   └── ...
├── models/
│   ├── efficientnet_b0_224/
│   └── ...
└── seu_script.py
```

### Dentro do Docker:
```
/workspace/
├── datasets/          <- Mesmo conteúdo!
│   ├── dataset_wagner/
│   ├── dataset_wagner_v4/
│   └── ...
├── models/            <- Mesmo conteúdo!
│   ├── efficientnet_b0_224/
│   └── ...
└── seu_script.py     <- Mesmo arquivo!
```

**✅ Todos os arquivos são compartilhados em tempo real!**
- Modificações no Docker aparecem no host
- Modelos salvos no Docker ficam no host
- Datasets são lidos diretamente

---

## 💻 Como Adaptar Seus Scripts

### Exemplo 1: Caminho Absoluto vs Relativo

**❌ Evite caminhos absolutos:**
```python
dataset_path = "/home/augusto/Repos/CLAUCIA-ML/datasets/dataset_wagner"
```

**✅ Use caminhos relativos:**
```python
from pathlib import Path

# Funciona tanto no Docker quanto fora dele
dataset_path = Path("datasets/dataset_wagner")
# ou
dataset_path = "datasets/dataset_wagner"
```

### Exemplo 2: Script de Treinamento

```python
import tensorflow as tf
from pathlib import Path

# GPU será detectada automaticamente
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Caminhos relativos funcionam perfeitamente
DATASET_PATH = "datasets/dataset_wagner_v4"
MODEL_SAVE_PATH = "models_trained/meu_modelo.keras"

# Carregar dados
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    subset='validation'
)

# Criar e treinar modelo
model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Adicionar camadas
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar (automaticamente usa GPU)
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Salvar (arquivo aparece no host)
model.save(MODEL_SAVE_PATH)
print(f"Modelo salvo em: {MODEL_SAVE_PATH}")
```

---

## 🎯 Treinar Seus Modelos Existentes

### EfficientNet B0
```bash
./cnn_baselines/tools/run_tensorflow_docker_full.sh python cnn_baselines/models/efficientnet_b0_224/efficientnet_b0_224_mlflow.py
```

### EfficientNet B1
```bash
./cnn_baselines/tools/run_tensorflow_docker_full.sh python cnn_baselines/models/efficientnet_b1_240/efficientnet_b1_240_mlflow.py
```

### ResNet50
```bash
./cnn_baselines/tools/run_tensorflow_docker_full.sh python cnn_baselines/models/resnet50_224/resnet50_224_mlflow.py
```

---

## 📝 Exemplo Completo

Um exemplo funcional está em `exemplo_treino_docker.py`:

```bash
./cnn_baselines/tools/run_tensorflow_docker_full.sh python exemplo_treino_docker.py
```

Esse exemplo demonstra:
- ✅ Detecção da GPU
- ✅ Acesso aos datasets
- ✅ Carregamento de imagens
- ✅ Criação de modelo
- ✅ Treinamento na GPU
- ✅ Salvamento do modelo

---

## 🔧 Modificar Scripts Existentes

### Antes (pode não funcionar no Docker):
```python
import os
os.environ['PATH'] = '/opt/rocm-6.4.2/bin:' + os.environ.get('PATH', '')
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

DATASET_PATH = "/home/augusto/Repos/CLAUCIA-ML/datasets/dataset_wagner"
```

### Depois (funciona em qualquer lugar):
```python
# Não precisa configurar ROCm manualmente no Docker!

from pathlib import Path

# Use caminhos relativos
DATASET_PATH = "datasets/dataset_wagner"
# ou
DATASET_PATH = Path("datasets") / "dataset_wagner"
```

---

## 💾 Onde os Arquivos São Salvos?

### Modelos treinados
```python
# No script:
model.save("models_trained/meu_modelo.keras")

# Aparece em:
# /home/augusto/Repos/CLAUCIA-ML/models_trained/meu_modelo.keras
```

### Logs do MLflow
```python
# No script:
mlflow.log_artifact("model.h5")

# Aparece em:
# /home/augusto/Repos/CLAUCIA-ML/mlruns/...
```

### Checkpoints
```python
# No script:
checkpoint = ModelCheckpoint("checkpoints/model_{epoch}.keras")

# Aparece em:
# /home/augusto/Repos/CLAUCIA-ML/checkpoints/
```

**✅ Tudo é salvo diretamente no seu sistema!**

---

## 🐛 Troubleshooting

### "Arquivo não encontrado"

**Problema:** Script procura arquivo em caminho absoluto

**Solução:** Use caminhos relativos
```python
# ❌ Errado
path = "/home/user/projeto/datasets"

# ✅ Correto
path = "datasets"
```

### "Permission denied" ao salvar

**Problema:** Permissões de arquivo

**Solução:** Crie o diretório antes:
```python
from pathlib import Path
Path("models_trained").mkdir(exist_ok=True)
model.save("models_trained/model.keras")
```

### "Module not found"

**Problema:** Dependência faltando

**Solução:** Use `run_tensorflow_docker_full.sh` ou instale manualmente:
```bash
./cnn_baselines/tools/run_tensorflow_docker_full.sh
# Dentro do container:
pip install nome-do-pacote
```

### GPU não detectada

**Problema:** Usando script errado

**Solução:** Use sempre os scripts `run_tensorflow_docker*.sh`:
```bash
# ✅ Correto
./cnn_baselines/tools/run_tensorflow_docker.sh python script.py

# ❌ Errado (perde configuração da GPU)
docker run rocm/tensorflow python script.py
```

---

## 📊 Performance

### Primeira execução:
- Mais lenta (instala dependências se usar `_full.sh`)
- ~10-30 segundos extras

### Execuções seguintes:
- Instantânea
- Mesma velocidade do ambiente nativo

### Dicas de Performance:
1. **Use batch size maior** - GPU tem 16GB
2. **Use mixed precision** quando suportado
3. **Prefetch data** para não travar a GPU
4. **Use `_full.sh`** para evitar reinstalar dependências

---

## 🎓 Referências

- **Tutorial oficial AMD**: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html
- **Docker Hub**: https://hub.docker.com/r/rocm/tensorflow
- **Documentação do projeto**: `README_GPU_AMD.md`

---

## ✅ Checklist

Antes de treinar, verifique:

- [ ] Docker instalado e funcionando
- [ ] Imagem baixada: `docker pull rocm/tensorflow:latest`
- [ ] Scripts executáveis: `chmod +x run_tensorflow_docker*.sh`
- [ ] Datasets no diretório correto
- [ ] Scripts usando caminhos relativos
- [ ] Testado com: `./cnn_baselines/tools/run_tensorflow_docker_full.sh python test_gpu.py`

---

**🎉 Pronto! Agora você pode treinar todos os seus modelos com a GPU AMD!**


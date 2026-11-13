# TensorFlow com GPU AMD RX 9070 XT

## ✅ SOLUÇÃO FUNCIONANDO

Sua GPU AMD RX 9070 XT **FUNCIONA PERFEITAMENTE** com TensorFlow usando **Docker**.

### Especificações Confirmadas:
- **GPU**: AMD Radeon RX 9070 XT (gfx1201 - RDNA 3)
- **Memória GPU**: 15024 MB disponível para TensorFlow
- **TensorFlow**: 2.19.1 com ROCm
- **Aceleração**: XLA compilado e funcional

---

## 🚀 Como Usar

### Método 1: Script Helper (Recomendado)

Execute seus scripts Python diretamente:

```bash
./cnn_baselines/tools/run_tensorflow_docker.sh python seu_script.py
```

### Método 2: Bash Interativo

Abra um terminal dentro do container:

```bash
./cnn_baselines/tools/run_tensorflow_docker.sh
```

Dentro do container, execute:
```bash
python seu_script.py
```

### Método 3: Comando Docker Manual

```bash
docker run -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$PWD:/workspace" \
    -w "/workspace" \
    rocm/tensorflow:latest \
    python seu_script.py
```

---

## 📦 Instalação

### 1. Instalar Docker (se necessário)

```bash
# Verificar se Docker está instalado
docker --version

# Se não estiver, instalar:
sudo apt update
sudo apt install docker.io
sudo usermod -aG docker $USER
# Faça logout e login novamente
```

### 2. Baixar Imagem TensorFlow ROCm

```bash
docker pull rocm/tensorflow:latest
```

Pronto! Não precisa instalar TensorFlow localmente.

---

## 🧪 Testar Instalação

Execute o script de teste:

```bash
./cnn_baselines/tools/run_tensorflow_docker.sh python test_gpu.py
```

**Saída esperada:**
```
GPUs detectadas: 1
✓ GPU AMD funcionando!
  GPU 0: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
✓ GPU AMD funcionando perfeitamente com TensorFlow!
```

---

## 💻 Usar em Seus Modelos

### Estrutura Básica

```python
import tensorflow as tf

# Detectar e configurar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ Usando GPU: {gpus[0]}")
else:
    print("⚠ GPU não detectada, usando CPU")

# Seu código de treinamento aqui
model = tf.keras.Sequential([...])
model.compile(...)
model.fit(...)  # Automaticamente usa a GPU
```

### Executar Seus Modelos Existentes

Para treinar seus modelos do projeto:

```bash
# Exemplo: treinar EfficientNet
./cnn_baselines/tools/run_tensorflow_docker.sh python cnn_baselines/models/efficientnet_b0_224/efficientnet_b0_224_mlflow.py

# Exemplo: executar notebook (dentro do Docker)
./cnn_baselines/tools/run_tensorflow_docker.sh
jupyter notebook --ip=0.0.0.0 --allow-root
```

---

## 📊 Performance

### Benefícios do Docker:
- ✅ **Plug & Play**: Funciona imediatamente
- ✅ **Pré-otimizado**: Todas bibliotecas ROCm incluídas
- ✅ **Isolado**: Não interfere com seu sistema
- ✅ **Testado**: Validado pela AMD

### Memória GPU:
- Total: ~16GB VRAM
- Disponível para TensorFlow: 15024 MB
- Crescimento dinâmico: Habilitado

---

## ⚙️ Arquivos do Projeto

- `run_tensorflow_docker.sh` - Script helper para executar Docker
- `test_gpu.py` - Script de teste da GPU
- `README_GPU_AMD.md` - Este arquivo

---

## 🔧 Troubleshooting

### GPU não detectada?

1. **Verifique se está dentro do Docker:**
   ```bash
   ./cnn_baselines/tools/run_tensorflow_docker.sh python test_gpu.py
   ```

2. **Verifique grupos do usuário:**
   ```bash
   groups  # Deve incluir 'video' e 'render'
   ```

3. **Verifique dispositivos:**
   ```bash
   ls -la /dev/kfd /dev/dri
   ```

### Instalaçã local (NÃO RECOMENDADO)

A instalação local do TensorFlow ROCm **NÃO FUNCIONA** adequadamente com GPUs consumer gfx1100 devido a limitações de suporte. Use Docker.

---

## 📚 Referências

- [Tutorial oficial AMD ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html)
- [TensorFlow ROCm Docker Hub](https://hub.docker.com/r/rocm/tensorflow)
- [ROCm Documentation](https://rocm.docs.amd.com/)

---

## ✨ Status

| Componente | Status | Versão |
|-----------|--------|---------|
| GPU AMD | ✅ Funcionando | RX 9070 XT (gfx1201) |
| ROCm | ✅ Funcionando | 7.0 (via Docker) |
| TensorFlow | ✅ Funcionando | 2.19.1 |
| XLA | ✅ Ativo | Compilado |
| Memória GPU | ✅ 15GB | Crescimento dinâmico |

---

**🎉 Sua GPU AMD está pronta para treinar modelos com TensorFlow!**



#!/bin/bash
# Script para executar TensorFlow com GPU AMD usando Docker
# Versão FULL: Instala dependências extras (Pillow, OpenCV, etc)

IMAGE="rocm/tensorflow:latest"
WORKSPACE="/workspace"
PROJECT_DIR="$PWD"

echo "========================================="
echo "TensorFlow ROCm Docker - FULL MODE"
echo "========================================="
echo "Projeto: $PROJECT_DIR"
echo "Imagem: $IMAGE"
echo ""

# Se houver argumento, executa o comando. Caso contrário, abre bash interativo
if [ $# -eq 0 ]; then
    echo "Iniciando bash interativo com dependências extras..."
    echo ""
    docker run -it \
        --network=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --ipc=host \
        --shm-size 16G \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "$PROJECT_DIR:$WORKSPACE" \
        -w "$WORKSPACE" \
        $IMAGE \
        bash -c "pip install --quiet Pillow opencv-python mlflow && /bin/bash"
else
    echo "Instalando dependências extras e executando: $@"
    echo ""
    docker run -it \
        --network=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --ipc=host \
        --shm-size 16G \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "$PROJECT_DIR:$WORKSPACE" \
        -w "$WORKSPACE" \
        $IMAGE \
        bash -c "pip install --quiet Pillow opencv-python mlflow && $*"
fi


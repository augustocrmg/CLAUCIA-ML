#!/bin/bash
# Script para executar TensorFlow com GPU AMD usando Docker
# Baseado no tutorial oficial: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html

# Configurações do container
IMAGE="rocm/tensorflow:latest"
WORKSPACE="/workspace"
PROJECT_DIR="$PWD"

echo "========================================="
echo "TensorFlow ROCm Docker - GPU AMD"
echo "========================================="
echo "Projeto: $PROJECT_DIR"
echo "Imagem: $IMAGE"
echo ""

# Se houver argumento, executa o comando. Caso contrário, abre bash interativo
if [ $# -eq 0 ]; then
    echo "Iniciando bash interativo..."
    echo "Para treinar seus modelos, use: python seu_script.py"
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
        /bin/bash
else
    echo "Executando: $@"
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
        "$@"
fi



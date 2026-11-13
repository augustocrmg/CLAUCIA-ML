#!/usr/bin/env python3
"""
Script de teste para TensorFlow com GPU AMD RX 9070 XT

SOLUÇÃO: Use Docker com a imagem oficial rocm/tensorflow
Execute: ./run_tensorflow_docker.sh python test_gpu.py
"""
import tensorflow as tf

print("="*70)
print("TESTE: TENSORFLOW COM AMD GPU")
print("="*70)
print(f"TensorFlow: {tf.__version__}")
print(f"ROCm compilado: {tf.test.is_built_with_rocm()}")

# Detectar GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs detectadas: {len(gpus)}")

if gpus:
    print("✓ GPU AMD funcionando!")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("\n" + "="*70)
    print("TESTE DE TREINAMENTO")
    print("="*70)
    
    # Teste com MNIST
    print("Carregando MNIST...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Subset para teste rápido
    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:200], y_test[:200]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("\nTreinando na GPU...")
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=1)
    
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia: {acc:.4f}")
    print("\n✓ GPU AMD funcionando perfeitamente com TensorFlow!")
    
else:
    print("✗ GPU não detectada")
    print("\nVocê está executando fora do Docker?")
    print("Use: ./run_tensorflow_docker.sh python test_gpu.py")
    print("\nOu instale a imagem Docker:")
    print("  docker pull rocm/tensorflow:latest")


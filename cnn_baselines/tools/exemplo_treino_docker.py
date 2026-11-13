#!/usr/bin/env python3
"""
Exemplo: Como treinar modelos dentro do Docker com acesso aos datasets

Execute com:
  ./run_tensorflow_docker.sh python exemplo_treino_docker.py
"""

import tensorflow as tf
import os
from pathlib import Path

print("="*70)
print("EXEMPLO: Treinamento com Acesso aos Datasets")
print("="*70)

# 1. Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\n1. GPU Status:")
print(f"   GPUs detectadas: {len(gpus)}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"   ✓ Usando GPU: {gpus[0]}")
else:
    print("   ⚠ Usando CPU")

# 2. Configurar caminhos dos datasets
print(f"\n2. Configuração de Caminhos:")
base_dir = Path.cwd()  # /workspace no Docker
dataset_path = base_dir / "datasets" / "dataset_wagner"

print(f"   Diretório base: {base_dir}")
print(f"   Dataset: {dataset_path}")
print(f"   Existe? {dataset_path.exists()}")

# 3. Listar classes disponíveis
if dataset_path.exists():
    classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    print(f"\n3. Classes encontradas: {len(classes)}")
    for cls in sorted(classes):
        img_count = len(list((dataset_path / cls).glob("*.*")))
        print(f"   - {cls}: {img_count} imagens")

# 4. Criar data generator para carregar imagens
print(f"\n4. Criando Image Data Generator...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurar data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Carregar dados
img_size = 224
batch_size = 32

try:
    train_generator = train_datagen.flow_from_directory(
        str(dataset_path),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        str(dataset_path),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    print(f"   ✓ Training samples: {train_generator.samples}")
    print(f"   ✓ Validation samples: {validation_generator.samples}")
    print(f"   ✓ Classes: {list(train_generator.class_indices.keys())}")

    # 5. Criar modelo simples
    print(f"\n5. Criando modelo...")
    
    num_classes = len(train_generator.class_indices)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                               input_shape=(img_size, img_size, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"   ✓ Modelo criado com {num_classes} classes")
    print(f"   ✓ Total de parâmetros: {model.count_params():,}")

    # 6. Treinar modelo (apenas 2 epochs para demonstração)
    print(f"\n6. Treinando modelo...")
    print(f"   (Apenas 2 epochs para demonstração)")
    
    history = model.fit(
        train_generator,
        epochs=2,
        validation_data=validation_generator,
        verbose=1
    )

    print(f"\n✅ TREINAMENTO CONCLUÍDO!")
    print(f"   Acurácia final (treino): {history.history['accuracy'][-1]:.4f}")
    print(f"   Acurácia final (validação): {history.history['val_accuracy'][-1]:.4f}")

    # 7. Salvar modelo
    output_dir = base_dir / "models_trained"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "exemplo_modelo.keras"
    model.save(model_path)
    print(f"\n7. Modelo salvo em: {model_path}")
    
except Exception as e:
    print(f"\n❌ Erro: {e}")
    print("\nSe o erro for relacionado a imagens corrompidas,")
    print("você pode adicionar validação no carregamento.")

print("\n" + "="*70)
print("✅ EXEMPLO COMPLETO!")
print("="*70)
print("\nPara treinar seus modelos reais:")
print("  ./run_tensorflow_docker.sh python models/efficientnet_b0_224/efficientnet_b0_224_mlflow.py")


# %% [markdown]
# # Integrantes:
# - Augusto Durães Camargo - 190084731
# - Eduardo Rodrigues de Farias - 190086521
# - João Vitor Ferreira Alves - 160127912
# 
# # Resolução: classificação de mudas de plantas
# 
# Para a resolução do problema foi utilizado a arquitetura de rede neural ResNet50v2, note que a rede utilizada não tinha nenhum parâmetro treinado, utilizamos apenas a arquitetura pré definida. Além deste notebook será disponibilizado no repositório um script que faz data augmentation do dataset v1 do kaggle. O dataset também está disponibilizado no zip, no entanto não está no repositório dada as limitações de tamanho de arquivos/diretório impostas pelo github.
# 
# link do repositório: https://github.com/augustocrmg/t1-topicos-deep-learning (solicitar o a acesso).

# %%
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# %%
!unzip drive/MyDrive/dataset_balanceado.zip

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils import image_dataset_from_directory
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns

# %%
image_size = (224, 224)
batch_size = 8

train_ds, val_ds = image_dataset_from_directory(
    "dataset_balanceado",
    validation_split=0.3,
    subset="both",
    seed=3,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)

classes = train_ds.class_names

# %%
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(6):
        image = images[i].numpy().astype("uint8")
        label = np.argmax(labels[i])

        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title(f"{classes[label]} ({label})")
        plt.axis("off")

# %%
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# %%
model = ResNet50V2(
    include_top=True,
    weights=None,
    classes=len(classes),
)

# %%
model.summary()

# %%
model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# %%
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[
        ModelCheckpoint("model.keras", save_best_only=True),
        EarlyStopping(patience=10)
    ],
    epochs=100,
)

# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)

if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
else:
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')

plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# %%
!cp model.keras drive/MyDrive/model.keras

# %%
model = load_model("model.keras")

# %%
predictions = model.predict(val_ds)
predictions = np.argmax(predictions, axis=1)

# Gera a matriz de confusão
y_test = np.concatenate([y for x, y in val_ds], axis=0)
confusion_mtx = confusion_matrix(np.argmax(y_test, axis=1), predictions)

# Plota a matriz de confusão
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt="d", ax=ax, cmap="viridis")
ax.set_xlabel("Etiqueta prevista")
ax.set_ylabel("Etiqueta verdadeira")
ax.set_title("Matriz de Confusão");



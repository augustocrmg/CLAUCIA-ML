import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Enable GPU
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('GPU memory growth enabled')
    except RuntimeError as e:
        print(e)

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNING = 10
DATASET_PATH = "datasets/dataset_wagner_balanceado_3"

# Get class names from directory structure
class_names = sorted([d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])
NUM_CLASSES = len(class_names)

print(f"Classes: {class_names}")
print(f"Number of classes: {NUM_CLASSES}")

# Data loading and preparation
def prepare_dataset():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    
    # Use buffered prefetching to load images from disk without I/O blocking
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# Visualize some samples
def visualize_samples(dataset):
    plt.figure(figsize=(12, 12))
    for images, labels in dataset.take(1):
        for i in range(min(9, images.shape[0])):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    
    sample_path = os.path.join(SCRIPT_DIR, 'sample_images.png')
    plt.savefig(sample_path)
    print(f"Sample images saved to '{sample_path}'")

# Build the model with EfficientNet
def build_model():
    # Load the EfficientNetB0 model with pre-trained ImageNet weights
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="max"
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    # Preprocessing specific to EfficientNet
    x = keras.applications.efficientnet.preprocess_input(inputs)
    # The base model contains batchnorm layers that could be updated
    # during training even if trainable = False. To disable this,
    # we would need to set training = False
    x = base_model(x, training=False)
    # Add a classification head
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model, base_model

# Plot training history
def plot_history(history, filename='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='train accuracy')
    ax1.plot(history.history['val_accuracy'], label='val accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='train loss')
    ax2.plot(history.history['val_loss'], label='val loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plot_path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(plot_path)
    print(f"Training history saved to '{plot_path}'")

# Unfreeze and fine-tune model
def unfreeze_model(model, base_model):
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Keep BatchNormalization layers frozen as recommended
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Create a function to predict on new images
def predict_image(model, image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")
    
    # Show the image with predicted class
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]} ({confidence:.2f})")
    plt.axis("off")
    
    pred_path = os.path.join(SCRIPT_DIR, 'prediction_result.png')
    plt.savefig(pred_path)
    print(f"Prediction visualization saved to '{pred_path}'")
    
    return class_names[predicted_class], confidence, {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

# Save the model
def save_model(model, filename="efficientnet_b0_244"):
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    print(f"Model saved as {model_path}")

# Main execution
def main():
    start_time = time.time()
    
    # Prepare dataset
    print("Preparing dataset...")
    train_ds, val_ds = prepare_dataset()
    
    # Display some sample images
    print("Visualizing sample images...")
    visualize_samples(train_ds)
    
    # Build the model
    print("Building model...")
    model, base_model = build_model()
    
    # Summary of the model
    model.summary()
    
    # Train the model with frozen base
    print("\nInitial training with frozen base model...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS_INITIAL,
        validation_data=val_ds
    )
    
    # Plot training history
    plot_history(history, 'initial_training_history.png')
    
    # Unfreeze and fine-tune
    print("\nFine-tuning the model...")
    model = unfreeze_model(model, base_model)
    
    # Fine-tune the model
    fine_tune_history = model.fit(
        train_ds,
        epochs=EPOCHS_FINE_TUNING,
        validation_data=val_ds
    )
    
    # Plot fine-tuning history
    plot_history(fine_tune_history, 'fine_tuning_history.png')
    
    # Save the model
    save_model(model, "efficientnet_b0_244")
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Find a test image to predict
    test_images = []
    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.isdir(class_path):
            files = os.listdir(class_path)
            if files:
                # Get the first image from each class
                test_images.append(os.path.join(class_path, files[0]))
    
    # Make predictions on test images
    if test_images:
        print("\nTesting model on sample images...")
        for img_path in test_images[:3]:  # Test on up to 3 images
            print(f"\nPredicting image: {img_path}")
            predict_image(model, img_path)
    
    return model

# Execute main function when script is run
if __name__ == "__main__":
    model = main() 
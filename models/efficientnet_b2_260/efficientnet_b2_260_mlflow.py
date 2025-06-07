import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import mlflow
import mlflow.keras
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

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
IMAGE_SIZE = (260, 260)  # EfficientNetB2 optimized input size
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNING = 10
DATASET_PATH = "datasets/dataset_wagner_balanceado_2"
EXPERIMENT_NAME = "EfficientNetB2-SkinCondition"

# Get class names from directory structure
class_names = sorted([d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])
NUM_CLASSES = len(class_names)

print(f"Classes: {class_names}")
print(f"Number of classes: {NUM_CLASSES}")

# Helper function to sanitize metric names for MLflow
def sanitize_mlflow_metric_name(name):
    """
    Sanitize a string to be used as a valid MLflow metric name.
    MLflow metric names may only contain alphanumerics, underscores, dashes, periods, spaces, colons and slashes.
    """
    return name.replace("(", "_").replace(")", "_").replace(",", "_").replace("&", "_").replace(";", "_")

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
    
    sample_img_path = os.path.join(SCRIPT_DIR, 'sample_images.png')
    plt.savefig(sample_img_path)
    print(f"Sample images saved to '{sample_img_path}'")
    
    # Log artifact to MLflow
    mlflow.log_artifact(sample_img_path)

# Build the model with EfficientNetB2
def build_model():
    # Load the EfficientNetB2 model with pre-trained ImageNet weights
    base_model = keras.applications.EfficientNetB2(
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

# MLflow callback for logging metrics during training
class MLflowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)

# Plot training history
def plot_history(history, filename='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Check if history has data before plotting
    if 'accuracy' in history.history:
        ax1.plot(history.history['accuracy'], label='train accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='val accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    if 'loss' in history.history:
        ax2.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='val loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plot_path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(plot_path)
    print(f"Training history saved to '{plot_path}'")
    
    # Log the plot as an artifact
    mlflow.log_artifact(plot_path)

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
    
    pred_img_path = os.path.join(SCRIPT_DIR, f'prediction_{os.path.basename(image_path)}.png')
    plt.savefig(pred_img_path)
    print(f"Prediction visualization saved to '{pred_img_path}'")

# Save the trained model
def save_model(model, filename="efficientnet_b2_260"):
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    print(f"Model saved to '{model_path}'")
    
    # Log the model to MLflow
    mlflow.keras.log_model(model, "model")

def main():
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():
        print("Loading and preparing data...")
        train_ds, val_ds = prepare_dataset()
        
        # Log parameters
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs_initial", EPOCHS_INITIAL)
        mlflow.log_param("epochs_fine_tuning", EPOCHS_FINE_TUNING)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("model_type", "EfficientNetB2")
        
        # Log dataset information
        mlflow.log_param("dataset_path", DATASET_PATH)
        mlflow.log_param("class_names", str(class_names))
        
        print("Visualizing sample images...")
        visualize_samples(train_ds)
        
        print("Building model...")
        model, base_model = build_model()
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Setup callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )
        
        mlflow_callback = MLflowCallback()
        
        callbacks = [early_stopping, reduce_lr, mlflow_callback]
        
        # Initial training with frozen base
        print("\n=== Phase 1: Training with frozen base ===")
        start_time = time.time()
        history_initial = model.fit(
            train_ds,
            epochs=EPOCHS_INITIAL,
            validation_data=val_ds,
            callbacks=callbacks
        )
        initial_training_time = time.time() - start_time
        
        # Plot initial training history
        plot_history(history_initial, 'initial_training_history.png')
        
        # Unfreeze and fine-tune
        print("\n=== Phase 2: Fine-tuning with unfrozen layers ===")
        model = unfreeze_model(model, base_model)
        
        start_time = time.time()
        history_fine_tune = model.fit(
            train_ds,
            epochs=EPOCHS_FINE_TUNING,
            validation_data=val_ds,
            callbacks=callbacks,
            initial_epoch=len(history_initial.history['loss'])
        )
        fine_tuning_time = time.time() - start_time
        
        # Plot fine-tuning history
        plot_history(history_fine_tune, 'fine_tuning_history.png')
        
        # Combined training times
        total_training_time = initial_training_time + fine_tuning_time
        mlflow.log_metric("total_training_time_seconds", total_training_time)
        mlflow.log_metric("initial_training_time_seconds", initial_training_time)
        mlflow.log_metric("fine_tuning_time_seconds", fine_tuning_time)
        
        print(f"\nTotal training time: {total_training_time/60:.2f} minutes")
        
        # Final evaluation
        print("\n=== Final Evaluation ===")
        test_loss, test_accuracy = model.evaluate(val_ds, verbose=0)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Log final metrics
        mlflow.log_metric("final_test_accuracy", test_accuracy)
        mlflow.log_metric("final_test_loss", test_loss)
        
        # Detailed evaluation with confusion matrix
        evaluate_model_with_confusion_matrix(model, val_ds, class_names)
        
        # Save the model
        save_model(model)
        
        print("Training completed successfully!")

def evaluate_model_with_confusion_matrix(model, val_ds, class_names):
    """
    Evaluate the model and generate confusion matrix and classification report
    """
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - EfficientNetB2')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to '{cm_path}'")
    
    # Log confusion matrix as artifact
    mlflow.log_artifact(cm_path)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Pretty print the report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Log individual class metrics to MLflow
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            sanitized_class_name = sanitize_mlflow_metric_name(class_name)
            for metric_name, value in metrics.items():
                if metric_name in ['precision', 'recall', 'f1-score', 'support']:
                    mlflow.log_metric(f"{sanitized_class_name}_{metric_name}", value)
    
    # Log overall metrics
    if 'accuracy' in report:
        mlflow.log_metric("overall_accuracy", report['accuracy'])
    if 'macro avg' in report:
        mlflow.log_metric("macro_avg_f1_score", report['macro avg']['f1-score'])
        mlflow.log_metric("macro_avg_precision", report['macro avg']['precision'])
        mlflow.log_metric("macro_avg_recall", report['macro avg']['recall'])
    if 'weighted avg' in report:
        mlflow.log_metric("weighted_avg_f1_score", report['weighted avg']['f1-score'])
        mlflow.log_metric("weighted_avg_precision", report['weighted avg']['precision'])
        mlflow.log_metric("weighted_avg_recall", report['weighted avg']['recall'])
    
    # Save classification report as JSON
    report_path = os.path.join(SCRIPT_DIR, 'classification_report.json')
    
    # Custom JSON encoder to handle numpy types
    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
    
    print(f"Classification report saved to '{report_path}'")
    mlflow.log_artifact(report_path)

if __name__ == "__main__":
    main() 
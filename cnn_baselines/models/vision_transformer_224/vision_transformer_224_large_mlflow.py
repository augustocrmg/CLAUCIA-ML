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
import glob
import random
from collections import defaultdict
from datetime import datetime

# Import the model logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_logger import save_training_results, extract_final_metrics, convert_model_to_tflite

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
        
        # Enable mixed precision for memory efficiency
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled')
    except RuntimeError as e:
        print(e)

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced from 32 for GPU memory
EPOCHS_INITIAL = 80
EPOCHS_FINE_TUNING = 10
DATASET_PATH = os.environ.get("CLAUCIA_CNN_DATASET", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))), "datasets", "dataset_wagner_v4_balanceado"))
EXPERIMENT_NAME = "VisionTransformer-SkinCondition"

# Vision Transformer hyperparameters - Optimized for 6GB GPU
PATCH_SIZE = 16  # Size of patches
NUM_PATCHES = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2
PROJECTION_DIM = 384  # Reduced from 768
NUM_HEADS = 6  # Reduced from 12
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,  # 768
    PROJECTION_DIM,      # 384
]
TRANSFORMER_LAYERS = 6  # Reduced from 12
MLP_HEAD_UNITS = [1024, 512]  # Reduced from [2048, 1024]

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
    """
    Prepare dataset with balanced split: 70% train, 15% validation, 15% test
    Ensures each class has the same proportion in each split
    """
    # First, analyze the dataset structure
    print("Analyzing dataset structure...")
    class_files = defaultdict(list)
    total_files = 0
    
    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.isdir(class_path):
            # Get all image files for this class
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(class_path, ext)))
                files.extend(glob.glob(os.path.join(class_path, ext.upper())))
            
            class_files[class_name] = files
            total_files += len(files)
            print(f"Class {class_name}: {len(files)} images")
    
    print(f"\nTotal images: {total_files}")
    
    # Check if dataset is balanced
    class_counts = [len(files) for files in class_files.values()]
    min_count = min(class_counts)
    max_count = max(class_counts)
    
    print(f"Min images per class: {min_count}")
    print(f"Max images per class: {max_count}")
    
    if max_count - min_count > min_count * 0.1:  # If difference > 10% of min
        print("WARNING: Dataset appears unbalanced!")
        for class_name, files in class_files.items():
            print(f"  {class_name}: {len(files)} images ({len(files)/total_files*100:.1f}%)")
    else:
        print("Dataset appears balanced ✓")
    
    # Create balanced splits for each class
    train_files = []
    val_files = []
    test_files = []
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("\nCreating balanced splits...")
    for class_name, files in class_files.items():
        # Shuffle files for this class
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)
        
        n_files = len(files_shuffled)
        n_train = int(n_files * 0.70)
        n_val = int(n_files * 0.10)
        n_test = n_files - n_train - n_val  # Remaining files go to test
        
        # Split files
        class_train = files_shuffled[:n_train]
        class_val = files_shuffled[n_train:n_train + n_val]
        class_test = files_shuffled[n_train + n_val:]
        
        train_files.extend([(f, class_name) for f in class_train])
        val_files.extend([(f, class_name) for f in class_val])
        test_files.extend([(f, class_name) for f in class_test])
        
        print(f"  {class_name}: {len(class_train)} train, {len(class_val)} val, {len(class_test)} test")
    
    # Shuffle the final splits
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    
    print(f"\nFinal splits:")
    print(f"  Train: {len(train_files)} images ({len(train_files)/total_files*100:.1f}%)")
    print(f"  Validation: {len(val_files)} images ({len(val_files)/total_files*100:.1f}%)")
    print(f"  Test: {len(test_files)} images ({len(test_files)/total_files*100:.1f}%)")
    
    # Create class name to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Helper function to create dataset from file list
    def create_dataset_from_files(file_list, batch_size=BATCH_SIZE):
        def generator():
            for file_path, class_name in file_list:
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=IMAGE_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.cast(img_array, tf.float32)
                
                # Get class index
                class_idx = class_to_idx[class_name]
                
                yield img_array, class_idx
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    # Create datasets
    print("\nCreating TensorFlow datasets...")
    train_ds = create_dataset_from_files(train_files)
    val_ds = create_dataset_from_files(val_files)
    test_ds = create_dataset_from_files(test_files)
    
    # Verify class distribution in each split
    print("\nVerifying class distribution:")
    
    def count_classes_in_split(file_list, split_name):
        class_counts = defaultdict(int)
        for _, class_name in file_list:
            class_counts[class_name] += 1
        
        print(f"\n{split_name} split distribution:")
        for class_name in class_names:
            count = class_counts[class_name]
            percentage = count / len(file_list) * 100 if file_list else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    count_classes_in_split(train_files, "Train")
    count_classes_in_split(val_files, "Validation")  
    count_classes_in_split(test_files, "Test")
    
    return train_ds, val_ds, test_ds

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

# Vision Transformer Components
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Data preprocessing (minimal since data is already augmented)
def create_data_augmentation():
    # Minimal preprocessing since data is already augmented
    data_augmentation = keras.Sequential([
        layers.Normalization(),
        layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        # Removed additional augmentations since data is already processed
    ], name="data_augmentation")
    return data_augmentation

# Build Vision Transformer model
def build_vision_transformer():
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    
    # Data preprocessing (minimal since data is already processed)
    data_augmentation = create_data_augmentation()
    augmented = data_augmentation(inputs)
    
    # Create patches
    patches = Patches(PATCH_SIZE)(augmented)
    
    # Encode patches
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    # Create multiple layers of the Transformer block
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Add MLP
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    
    # Classify outputs - ensure float32 for mixed precision
    logits = layers.Dense(NUM_CLASSES, dtype='float32')(features)
    outputs = layers.Activation('softmax', dtype='float32')(logits)
    
    # Create the Keras model
    model = keras.Model(inputs, outputs)
    
    return model

# MLflow callback for logging metrics during training
class MLflowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)

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
    
    # Log the plot as an artifact
    mlflow.log_artifact(plot_path)

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
    
    # Log prediction as artifact
    mlflow.log_artifact(pred_img_path)
    
    # Log metrics for the prediction
    prediction_metrics = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    
    for class_name, prob in prediction_metrics.items():
        # Use sanitized class name for metric name
        safe_class_name = sanitize_mlflow_metric_name(class_name)
        mlflow.log_metric(f"prediction_prob_{safe_class_name}", prob)
    
    mlflow.log_metric("prediction_confidence", confidence)
    
    return class_names[predicted_class], confidence, prediction_metrics

# Save the model
def save_model(model, filename="vision_transformer_224"):
    # Save H5 model
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    print(f"Model saved as {model_path}")
    
    # Save TensorFlow Lite model
    tflite_path = os.path.join(SCRIPT_DIR, f"{filename}.tflite")
    convert_model_to_tflite(model, tflite_path)
    
    # Log model to MLflow
    mlflow.log_artifact(model_path)

# Generate confusion matrix and classification report
def evaluate_model_with_confusion_matrix(model, test_ds, class_names):
    """
    Generate confusion matrix and detailed classification metrics
    """
    print("\nGenerating confusion matrix and classification report...")
    
    # Get predictions for the entire test set
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy per class
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(12, 10))
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Plot confusion matrix with percentages
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Confusion Matrix - Absolute Values (Vision Transformer)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_vision_transformer.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to '{cm_path}'")
    mlflow.log_artifact(cm_path)
    plt.close()
    
    # Create normalized confusion matrix (percentages)
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    sns.heatmap(cm_norm_df, annot=True, fmt='.2%', cmap='Blues', 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Confusion Matrix - Normalized (Vision Transformer)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_norm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_vision_transformer.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix saved to '{cm_norm_path}'")
    mlflow.log_artifact(cm_norm_path)
    plt.close()
    
    # Create detailed metrics visualization
    plt.figure(figsize=(14, 8))
    
    # Extract metrics from classification report
    precision_values = [report[class_name]['precision'] for class_name in class_names]
    recall_values = [report[class_name]['recall'] for class_name in class_names]
    f1_values = [report[class_name]['f1-score'] for class_name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, precision_values, width, label='Precision', alpha=0.8)
    plt.bar(x, recall_values, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_values, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Classification Metrics by Class', fontsize=16, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    metrics_path = os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_vision_transformer.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"Classification metrics plot saved to '{metrics_path}'")
    mlflow.log_artifact(metrics_path)
    plt.close()
    
    # Log detailed metrics to MLflow
    overall_accuracy = np.sum(y_true == y_pred) / len(y_true)
    mlflow.log_metric("confusion_matrix_accuracy", overall_accuracy)
    mlflow.log_metric("macro_avg_precision", report['macro avg']['precision'])
    mlflow.log_metric("macro_avg_recall", report['macro avg']['recall'])
    mlflow.log_metric("macro_avg_f1", report['macro avg']['f1-score'])
    mlflow.log_metric("weighted_avg_precision", report['weighted avg']['precision'])
    mlflow.log_metric("weighted_avg_recall", report['weighted avg']['recall'])
    mlflow.log_metric("weighted_avg_f1", report['weighted avg']['f1-score'])
    
    # Log per-class metrics
    for i, class_name in enumerate(class_names):
        safe_class_name = sanitize_mlflow_metric_name(class_name)
        mlflow.log_metric(f"precision_{safe_class_name}", report[class_name]['precision'])
        mlflow.log_metric(f"recall_{safe_class_name}", report[class_name]['recall'])
        mlflow.log_metric(f"f1_score_{safe_class_name}", report[class_name]['f1-score'])
        mlflow.log_metric(f"support_{safe_class_name}", report[class_name]['support'])
        mlflow.log_metric(f"accuracy_{safe_class_name}", class_accuracies[i])
    
    # Save detailed report as JSON
    report_path = os.path.join(SCRIPT_DIR, "classification_report_vision_transformer.json")
    
    # Create a comprehensive report
    detailed_report = {
        "model_name": "Vision Transformer",
        "classification_report": report,
        "class_accuracies": {class_names[i]: float(class_accuracies[i]) for i in range(len(class_names))},
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_normalized.tolist()
    }
    
    with open(report_path, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    mlflow.log_artifact(report_path)
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Average F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    print("\nPer-Class Performance:")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        accuracy = class_accuracies[i]
        
        print(f"{class_name:20} | Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f} | Support: {support}")
    
    print("\n" + "="*60)
    
    return cm, report, class_accuracies

# Generate confusion matrix and classification report for validation set
def evaluate_model_with_confusion_matrix_validation(model, val_ds, class_names):
    """
    Generate confusion matrix and detailed classification metrics for validation set
    """
    print("\nGenerating confusion matrix and classification report for validation set...")
    
    # Get predictions for the entire validation set
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy per class
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Log detailed metrics to MLflow with 'val_' prefix
    overall_accuracy = np.sum(y_true == y_pred) / len(y_true)
    mlflow.log_metric("val_confusion_matrix_accuracy", overall_accuracy)
    mlflow.log_metric("val_macro_avg_precision", report['macro avg']['precision'])
    mlflow.log_metric("val_macro_avg_recall", report['macro avg']['recall'])
    mlflow.log_metric("val_macro_avg_f1", report['macro avg']['f1-score'])
    mlflow.log_metric("val_weighted_avg_precision", report['weighted avg']['precision'])
    mlflow.log_metric("val_weighted_avg_recall", report['weighted avg']['recall'])
    mlflow.log_metric("val_weighted_avg_f1", report['weighted avg']['f1-score'])
    
    # Log per-class metrics with 'val_' prefix
    for i, class_name in enumerate(class_names):
        safe_class_name = sanitize_mlflow_metric_name(class_name)
        mlflow.log_metric(f"val_precision_{safe_class_name}", report[class_name]['precision'])
        mlflow.log_metric(f"val_recall_{safe_class_name}", report[class_name]['recall'])
        mlflow.log_metric(f"val_f1_score_{safe_class_name}", report[class_name]['f1-score'])
        mlflow.log_metric(f"val_support_{safe_class_name}", report[class_name]['support'])
        mlflow.log_metric(f"val_accuracy_{safe_class_name}", class_accuracies[i])
    
    # Print summary for validation set
    print(f"\nValidation Set Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Validation Set Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Validation Set Weighted Average F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    return cm, report, class_accuracies

# Main execution
def main():
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    start_time = datetime.now()
    
    with mlflow.start_run(run_name=f"VisionTransformer_{start_time.strftime('%Y%m%d-%H%M%S')}"):
        
        # Log parameters
        params = {
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            "batch_size": BATCH_SIZE,
            "epochs_initial": EPOCHS_INITIAL,
            "epochs_fine_tuning": EPOCHS_FINE_TUNING,
            "num_classes": NUM_CLASSES,
            "classes": str(class_names),
            "dataset_path": DATASET_PATH,
            "model_architecture": "Vision Transformer",
            "patch_size": PATCH_SIZE,
            "num_patches": NUM_PATCHES,
            "projection_dim": PROJECTION_DIM,
            "num_heads": NUM_HEADS,
            "transformer_layers": TRANSFORMER_LAYERS,
            "train_split": "70%",
            "validation_split": "10%",
            "test_split": "20%",
            "balanced_splits": "True",
            "split_strategy": "per_class_balanced",
            "data_augmentation": "minimal_preprocessing_only"
        }
        mlflow.log_params(params)
        
        # Prepare dataset
        print("Preparing dataset...")
        train_ds, val_ds, test_ds = prepare_dataset()
        
        # Log dataset size
        train_size = sum(1 for _ in train_ds)
        val_size = sum(1 for _ in val_ds)
        test_size = sum(1 for _ in test_ds)
        mlflow.log_metric("train_batches", train_size)
        mlflow.log_metric("val_batches", val_size)
        mlflow.log_metric("test_batches", test_size)
        
        # Display some sample images
        print("Visualizing sample images...")
        visualize_samples(train_ds)
        
        # Build the model
        print("Building Vision Transformer model...")
        model = build_vision_transformer()
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Summary of the model
        model.summary()
        
        # Create MLflow callback for logging during training
        mlflow_callback = MLflowCallback()
        
        # Train the model
        print("\nTraining Vision Transformer model...")
        mlflow.log_metric("training_phase", 1)  # Training phase
        
        history = model.fit(
            train_ds,
            epochs=EPOCHS_INITIAL,
            validation_data=val_ds,
            callbacks=[mlflow_callback]
        )
        
        # Plot training history
        plot_history(history, 'training_history.png')
        
        # Track best validation metrics from training
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # Evaluate model with confusion matrix and detailed metrics
        print("\n" + "="*60)
        print("EVALUATING MODEL PERFORMANCE ON TEST SET")
        print("="*60)
        cm, classification_report_dict, class_accuracies = evaluate_model_with_confusion_matrix(model, test_ds, class_names)
        
        # Also evaluate on validation set for comparison
        print("\n" + "="*60)
        print("EVALUATING MODEL PERFORMANCE ON VALIDATION SET (for comparison)")
        print("="*60)
        cm_val, classification_report_dict_val, class_accuracies_val = evaluate_model_with_confusion_matrix_validation(model, val_ds, class_names)
        
        # Capture end time
        end_time = datetime.now()
        
        # Count model parameters
        model_params = model.count_params()
        
        # Save the model
        save_model(model, "vision_transformer_224")
        
        # ============================================
        # SAVE TRAINING RESULTS TO JSON
        # ============================================
        print("\n" + "="*60)
        print("SAVING TRAINING RESULTS TO JSON")
        print("="*60)
        
        # Extract final metrics from training history
        train_acc = history.history['accuracy'][-1]
        valid_acc = history.history['val_accuracy'][-1]
        
        # Extract dataset name from path
        dataset_name = os.path.basename(DATASET_PATH)
        
        # Save training results to JSON
        save_training_results(
            dataset_name=dataset_name,
            model_name="Vision Transformer",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_INITIAL,
            fine_tuning_epochs=0,  # No fine-tuning for ViT
            train_acc=train_acc,
            valid_acc=valid_acc,
            train_fine_tuning_acc=train_acc,  # Same as train_acc since no fine-tuning
            valid_finetuning_acc=valid_acc,   # Same as valid_acc since no fine-tuning
            output_dir=os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "models_results"),
            model_params=model_params,
            training_start_time=start_time,
            training_end_time=end_time
        )
        
        # Calculate total time
        total_time = datetime.now() - start_time
        hours, remainder = divmod(total_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        print(f"Total training time: {training_time_str}")
        
        # Log training time
        mlflow.log_metric("training_time_seconds", total_time.total_seconds())
        mlflow.log_param("training_time", training_time_str)
        
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
            prediction_results = {}
            
            for img_path in test_images[:3]:  # Test on up to 3 images
                print(f"\nPredicting image: {img_path}")
                class_name, confidence, class_probs = predict_image(model, img_path)
                
                # Store prediction results
                img_basename = os.path.basename(img_path)
                true_class = os.path.basename(os.path.dirname(img_path))
                is_correct = class_name == true_class
                
                prediction_results[img_basename] = {
                    "predicted_class": class_name,
                    "true_class": true_class,
                    "correct": is_correct,
                    "confidence": float(confidence)  # Convert float32 to Python float
                }
                
                # Log prediction correctness - sanitize filename for MLflow metric name
                safe_img_name = sanitize_mlflow_metric_name(img_basename)
                mlflow.log_metric(f"prediction_correct_{safe_img_name}", 1 if is_correct else 0)
            
            # Log prediction results as a JSON artifact
            class NumpyJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
            
            results_path = os.path.join(SCRIPT_DIR, "prediction_results.json")
            with open(results_path, "w") as f:
                json.dump(prediction_results, f, indent=2, cls=NumpyJSONEncoder)
            
            mlflow.log_artifact(results_path)
    
    return model

# Execute main function when script is run
if __name__ == "__main__":
    model = main() 
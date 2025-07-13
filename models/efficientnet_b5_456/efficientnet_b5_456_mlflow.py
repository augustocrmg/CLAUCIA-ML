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
    except RuntimeError as e:
        print(e)

# Configuration
IMAGE_SIZE = (456, 456)  # EfficientNetB5 optimized input size
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNING = 10
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "datasets", "dataset_wagner_v4_balanceado")
EXPERIMENT_NAME = "EfficientNetB5-SkinCondition"

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
    Prepare dataset with balanced split: 70% train, 10% validation, 20% test
    Ensures each class has the same proportion in each split
    
    This function implements a more controlled approach than tensorflow's 
    image_dataset_from_directory to guarantee balanced splits:
    
    1. Analyzes the dataset structure and counts images per class
    2. Creates balanced splits by dividing each class individually:
       - 70% of each class goes to training
       - 10% of each class goes to validation  
       - 20% of each class goes to test
    3. Shuffles files within each class before splitting (reproducible with seed=42)
    4. Creates TensorFlow datasets with proper batching and prefetching
    5. Provides detailed statistics about the final split distribution
    
    Returns:
        train_ds: Training dataset (70% of total data)
        val_ds: Validation dataset (10% of total data) 
        test_ds: Test dataset (20% of total data)
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
    
    print(f"Total images: {total_files}")
    
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
    
    print("Creating balanced splits...")
    for class_name, files in class_files.items():
        # Shuffle files for this class
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)
        
        n_files = len(files_shuffled)
        n_train = int(n_files * 0.70)
        n_val = int(n_files * 0.10)
        #n_val = int(n_files * 0.15)
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
    
    print(f"Final splits:")
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
    print("Creating TensorFlow datasets...")
    train_ds = create_dataset_from_files(train_files)
    val_ds = create_dataset_from_files(val_files)
    test_ds = create_dataset_from_files(test_files)
    
    # Verify class distribution in each split
    print("Verifying class distribution:")
    
    def count_classes_in_split(file_list, split_name):
        class_counts = defaultdict(int)
        for _, class_name in file_list:
            class_counts[class_name] += 1
        
        print(f"{split_name} split distribution:")
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

# Build the model with EfficientNetB5
def build_model():
    # Load the EfficientNetB5 model with pre-trained ImageNet weights
    base_model = keras.applications.EfficientNetB5(
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
def save_model(model, filename="efficientnet_b5_456"):
    # Save H5 model
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    print(f"Model saved to '{model_path}'")
    
    # Save TensorFlow Lite model
    tflite_path = os.path.join(SCRIPT_DIR, f"{filename}.tflite")
    convert_model_to_tflite(model, tflite_path)
    
    # Log the model to MLflow
    mlflow.keras.log_model(model, "model")

def main():
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():
        print("Loading and preparing data...")
        train_ds, val_ds, test_ds = prepare_dataset()
        
        # Log dataset size
        train_size = sum(1 for _ in train_ds)
        val_size = sum(1 for _ in val_ds)
        test_size = sum(1 for _ in test_ds)
        mlflow.log_metric("train_batches", train_size)
        mlflow.log_metric("val_batches", val_size)
        mlflow.log_metric("test_batches", test_size)

        # Log parameters
        params = {
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            "batch_size": BATCH_SIZE,
            "epochs_initial": EPOCHS_INITIAL,
            "epochs_fine_tuning": EPOCHS_FINE_TUNING,
            "num_classes": NUM_CLASSES,
            "classes": str(class_names),
            "dataset_path": DATASET_PATH,
            "model_architecture": "EfficientNetB5",
            "train_split": "70%",
            "validation_split": "10%",
            "test_split": "20%",
            "balanced_splits": "True",
            "split_strategy": "per_class_balanced"
        }
        mlflow.log_params(params)

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
        
        # Track best validation metrics from initial training
        best_val_acc = max(history_initial.history['val_accuracy'])
        best_val_loss = min(history_initial.history['val_loss'])
        mlflow.log_metric("initial_best_val_accuracy", best_val_acc)
        mlflow.log_metric("initial_best_val_loss", best_val_loss)
        
        # Unfreeze and fine-tune
        print("\n=== Phase 2: Fine-tuning with unfrozen layers ===")
        model = unfreeze_model(model, base_model)
        
        # Log new phase
        mlflow.log_metric("training_phase", 2)  # Fine-tuning phase
        
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
        
        # Track best validation metrics from fine-tuning
        if 'val_accuracy' in history_fine_tune.history and history_fine_tune.history['val_accuracy']:
            best_val_acc_ft = max(history_fine_tune.history['val_accuracy'])
            best_val_loss_ft = min(history_fine_tune.history['val_loss'])
            mlflow.log_metric("fine_tuning_best_val_accuracy", best_val_acc_ft)
            mlflow.log_metric("fine_tuning_best_val_loss", best_val_loss_ft)
            
            # Log improvement metrics
            mlflow.log_metric("accuracy_improvement", best_val_acc_ft - best_val_acc)
            mlflow.log_metric("loss_improvement", best_val_loss - best_val_loss_ft)
        else:
            # If no fine-tuning epochs were run, use initial values
            best_val_acc_ft = best_val_acc
            best_val_loss_ft = best_val_loss
            mlflow.log_metric("fine_tuning_best_val_accuracy", best_val_acc_ft)
            mlflow.log_metric("fine_tuning_best_val_loss", best_val_loss_ft)
            mlflow.log_metric("accuracy_improvement", 0.0)
            mlflow.log_metric("loss_improvement", 0.0)
        
        # Combined training times
        total_training_time = initial_training_time + fine_tuning_time
        mlflow.log_metric("total_training_time_seconds", total_training_time)
        mlflow.log_metric("initial_training_time_seconds", initial_training_time)
        mlflow.log_metric("fine_tuning_time_seconds", fine_tuning_time)
        
        print(f"\nTotal training time: {total_training_time/60:.2f} minutes")
        
        # Final evaluation
        print("\n=== Final Evaluation ===")
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Log final metrics
        mlflow.log_metric("final_test_accuracy", test_accuracy)
        mlflow.log_metric("final_test_loss", test_loss)
        
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
        
        # Save the model
        save_model(model)
        
        # ============================================
        # SAVE TRAINING RESULTS TO JSON
        # ============================================
        print("\n" + "="*60)
        print("SAVING TRAINING RESULTS TO JSON")
        print("="*60)
        
        # Extract final metrics from training histories
        train_acc, valid_acc, train_fine_tuning_acc, valid_finetuning_acc = extract_final_metrics(history_initial, history_fine_tune)
        
        # Extract dataset name from path
        dataset_name = os.path.basename(DATASET_PATH)
        
        # Save training results to JSON
        save_training_results(
            dataset_name=dataset_name,
            model_name="EfficientNet-B5",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_INITIAL,
            fine_tuning_epochs=EPOCHS_FINE_TUNING,
            train_acc=train_acc,
            valid_acc=valid_acc,
            train_fine_tuning_acc=train_fine_tuning_acc,
            valid_finetuning_acc=valid_finetuning_acc,
            output_dir=os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "models_results")
        )
        
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
    plt.title('Confusion Matrix - Absolute Values (EfficientNet-B5)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_efficientnet_b5.png')
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
    plt.title('Confusion Matrix - Normalized (EfficientNet-B5)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_norm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_efficientnet_b5.png')
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
    plt.title('Classification Metrics by Class (EfficientNet-B5)', fontsize=16, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    metrics_path = os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_efficientnet_b5.png')
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
    report_path = os.path.join(SCRIPT_DIR, "classification_report_efficientnet_b5.json")
    
    # Create a comprehensive report
    detailed_report = {
        "model_name": "EfficientNet-B5",
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
    
    # Identify most confused classes
    print("\nMost Confused Class Pairs:")
    print("-" * 40)
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((class_names[i], class_names[j], cm[i, j], cm[i, j] / cm[i].sum()))
    
    # Sort by absolute number of confusions
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for true_class, pred_class, count, percentage in confusion_pairs[:10]:  # Top 10 confusions
        print(f"{true_class:20} -> {pred_class:20} | Count: {count:3d} | {percentage:.1%}")
    
    # Create visualization for most confused class pairs
    if confusion_pairs:
        plt.figure(figsize=(14, 10))
        
        # Get top 15 confusions for better visualization
        top_confusions = confusion_pairs[:15]
        
        if top_confusions:
            confusion_labels = [f"{true_class} → {pred_class}" for true_class, pred_class, count, percentage in top_confusions]
            confusion_counts = [count for true_class, pred_class, count, percentage in top_confusions]
            confusion_percentages = [percentage * 100 for true_class, pred_class, count, percentage in top_confusions]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(confusion_labels))
            bars = plt.barh(y_pos, confusion_counts, alpha=0.7)
            
            # Color bars based on confusion severity
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(confusion_counts)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add percentage labels on bars
            for i, (count, percentage) in enumerate(zip(confusion_counts, confusion_percentages)):
                plt.text(count + max(confusion_counts) * 0.01, i, f'{count} ({percentage:.1f}%)', 
                        va='center', fontsize=10, fontweight='bold')
            
            plt.yticks(y_pos, confusion_labels, fontsize=10)
            plt.xlabel('Number of Confusions', fontsize=12, fontweight='bold')
            plt.title('Top Class Confusions (True Class → Predicted Class)', fontsize=16, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Invert y-axis to show highest confusions at top
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            confusion_plot_path = os.path.join(SCRIPT_DIR, 'class_confusions_plot_efficientnet_b5.png')
            plt.savefig(confusion_plot_path, dpi=300, bbox_inches='tight')
            print(f"Class confusions plot saved to '{confusion_plot_path}'")
            mlflow.log_artifact(confusion_plot_path)
            plt.close()
    
    # Create a comprehensive error analysis plot
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Per-class accuracy
    plt.subplot(2, 2, 1)
    class_acc_sorted = sorted(zip(class_names, class_accuracies), key=lambda x: x[1])
    sorted_classes, sorted_accuracies = zip(*class_acc_sorted)
    
    colors_acc = ['red' if acc < 0.7 else 'orange' if acc < 0.85 else 'green' for acc in sorted_accuracies]
    bars = plt.barh(range(len(sorted_classes)), sorted_accuracies, color=colors_acc, alpha=0.7)
    
    plt.yticks(range(len(sorted_classes)), sorted_classes, fontsize=10)
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy (Sorted)', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy labels
    for i, acc in enumerate(sorted_accuracies):
        plt.text(acc + 0.02, i, f'{acc:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Subplot 2: Support (number of samples) per class
    plt.subplot(2, 2, 2)
    support_values = [report[class_name]['support'] for class_name in class_names]
    class_support_sorted = sorted(zip(class_names, support_values), key=lambda x: x[1])
    sorted_classes_sup, sorted_support = zip(*class_support_sorted)
    
    plt.barh(range(len(sorted_classes_sup)), sorted_support, color='skyblue', alpha=0.7)
    plt.yticks(range(len(sorted_classes_sup)), sorted_classes_sup, fontsize=10)
    plt.xlabel('Number of Samples', fontsize=12)
    plt.title('Support per Class (Sorted)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add support labels
    for i, sup in enumerate(sorted_support):
        plt.text(sup + max(sorted_support) * 0.02, i, str(sup), va='center', fontsize=9, fontweight='bold')
    
    # Subplot 3: F1-Score vs Support scatter plot
    plt.subplot(2, 2, 3)
    f1_scores = [report[class_name]['f1-score'] for class_name in class_names]
    
    scatter = plt.scatter(support_values, f1_scores, c=class_accuracies, cmap='RdYlGn', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Accuracy')
    plt.xlabel('Support (Number of Samples)', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score vs Support (colored by Accuracy)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add class labels to points
    for i, class_name in enumerate(class_names):
        plt.annotate(class_name, (support_values[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Subplot 4: Precision vs Recall scatter plot
    plt.subplot(2, 2, 4)
    precision_values = [report[class_name]['precision'] for class_name in class_names]
    recall_values = [report[class_name]['recall'] for class_name in class_names]
    
    scatter2 = plt.scatter(recall_values, precision_values, c=f1_scores, cmap='viridis', 
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter2, label='F1-Score')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision vs Recall (colored by F1-Score)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    # Add diagonal line (perfect precision-recall balance)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Balance')
    plt.legend()
    
    # Add class labels to points
    for i, class_name in enumerate(class_names):
        plt.annotate(class_name, (recall_values[i], precision_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    comprehensive_plot_path = os.path.join(SCRIPT_DIR, 'comprehensive_error_analysis_efficientnet_b5.png')
    plt.savefig(comprehensive_plot_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive error analysis plot saved to '{comprehensive_plot_path}'")
    mlflow.log_artifact(comprehensive_plot_path)
    plt.close()
    
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

if __name__ == "__main__":
    main() 
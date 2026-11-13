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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
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
        
        # Enable mixed precision for better performance on modern GPUs
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled')
        
    except RuntimeError as e:
        print(e)

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 100  # Max epochs with early stopping
EPOCHS_FINE_TUNING = 20  # Max epochs for fine-tuning with early stopping
DATASET_PATH = os.environ.get("CLAUCIA_CNN_DATASET", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))), "datasets", "perfect_balanced_dataset_wagner_v4_1000"))
EXPERIMENT_NAME = "EfficientNetB0-SkinCondition-PerfectBalanced"

# Training options
USE_FINE_TUNING = False  # Set to False to skip fine-tuning phase (only initial training)

# Early stopping configuration
EARLY_STOP_PATIENCE = 5  # Patience for initial training
EARLY_STOP_PATIENCE_FINE_TUNING = 5  # Patience for fine-tuning

# Get class names from directory structure (from treino folder)
train_path = os.path.join(DATASET_PATH, "treino")
class_names = sorted([d for d in os.listdir(train_path) 
                     if os.path.isdir(os.path.join(train_path, d))])
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

# Extract comprehensive metrics for JSON logging
def extract_comprehensive_metrics(classification_report_dict, class_accuracies, auc_score, class_names):
    """
    Extract all metrics in a structured format for JSON logging.
    
    Args:
        classification_report_dict: Classification report from sklearn
        class_accuracies: Array of per-class accuracies
        auc_score: Overall AUC score
        class_names: List of class names
    
    Returns:
        test_metrics: Overall test metrics
        per_class_metrics: Per-class metrics for all classes
    """
    # Overall test metrics
    test_metrics = {
        "accuracy": float(classification_report_dict['accuracy']),
        "precision_macro": float(classification_report_dict['macro avg']['precision']),
        "recall_macro": float(classification_report_dict['macro avg']['recall']),
        "f1_score_macro": float(classification_report_dict['macro avg']['f1-score']),
        "precision_weighted": float(classification_report_dict['weighted avg']['precision']),
        "recall_weighted": float(classification_report_dict['weighted avg']['recall']),
        "f1_score_weighted": float(classification_report_dict['weighted avg']['f1-score']),
        "auc": float(auc_score)
    }
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "accuracy": float(class_accuracies[i]),
            "precision": float(classification_report_dict[class_name]['precision']),
            "recall": float(classification_report_dict[class_name]['recall']),
            "f1_score": float(classification_report_dict[class_name]['f1-score']),
            "support": int(classification_report_dict[class_name]['support'])
        }
    
    return test_metrics, per_class_metrics

# Data loading and preparation for pre-split dataset
def prepare_dataset():
    """
    Prepare dataset from pre-split structure: treino, teste, validacao folders
    
    This function loads a dataset that is already split into:
    - treino/ (training data)
    - teste/ (test data)
    - validacao/ (validation data)
    
    Each folder contains subdirectories for each class.
    
    Returns:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
    """
    print("Loading pre-split dataset...")
    
    # Define paths for each split
    train_path = os.path.join(DATASET_PATH, "treino")
    val_path = os.path.join(DATASET_PATH, "validacao")
    test_path = os.path.join(DATASET_PATH, "teste")
    
    # Verify all paths exist
    for split_name, split_path in [("Train", train_path), ("Validation", val_path), ("Test", test_path)]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_name} directory not found: {split_path}")
        print(f"{split_name} path: {split_path}")
    
    # Count images in each split
    def count_images_in_split(split_path):
        total = 0
        class_counts = {}
        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
                files = []
                for ext in extensions:
                    files.extend(glob.glob(os.path.join(class_path, ext)))
                    files.extend(glob.glob(os.path.join(class_path, ext.upper())))
                count = len(files)
                class_counts[class_name] = count
                total += count
        return total, class_counts
    
    print("\nAnalyzing dataset structure...")
    train_total, train_counts = count_images_in_split(train_path)
    val_total, val_counts = count_images_in_split(val_path)
    test_total, test_counts = count_images_in_split(test_path)
    
    print(f"\nTotal images:")
    print(f"  Train: {train_total} images")
    print(f"  Validation: {val_total} images")
    print(f"  Test: {test_total} images")
    print(f"  Total: {train_total + val_total + test_total} images")
    
    # Display per-class distribution
    print("\nPer-class distribution:")
    print("-" * 60)
    print(f"{'Class':<25} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-" * 60)
    for class_name in class_names:
        train_c = train_counts.get(class_name, 0)
        val_c = val_counts.get(class_name, 0)
        test_c = test_counts.get(class_name, 0)
        total_c = train_c + val_c + test_c
        print(f"{class_name:<25} {train_c:>10} {val_c:>10} {test_c:>10} {total_c:>10}")
    print("-" * 60)
    
    # Helper function to create dataset from directory
    def create_dataset_from_directory(directory_path, batch_size=BATCH_SIZE):
        print(f"\n[DEBUG] Creating dataset from {directory_path}...")
        
        # Collect all files
        file_list = []
        for class_name in class_names:
            class_path = os.path.join(directory_path, class_name)
            if os.path.isdir(class_path):
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
                files = []
                for ext in extensions:
                    files.extend(glob.glob(os.path.join(class_path, ext)))
                    files.extend(glob.glob(os.path.join(class_path, ext.upper())))
                
                for file_path in files:
                    file_list.append((file_path, class_name))
        
        print(f"[DEBUG] Found {len(file_list)} files")
        
        # Create class name to index mapping
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        
        # Shuffle file list
        random.shuffle(file_list)
        
        # Create file paths and labels tensors
        file_paths = [file_path for file_path, _ in file_list]
        labels = [class_to_idx[class_name] for _, class_name in file_list]
        
        # Create dataset from file paths
        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((path_ds, label_ds))
        
        # Optimized image loading function
        def load_and_preprocess_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32)
            return image, label
        
        # Apply transformations with optimizations
        dataset = dataset.map(
            load_and_preprocess_image, 
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        
        # Optimize pipeline
        dataset = dataset.shuffle(buffer_size=min(1000, len(file_list)), seed=42)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Only cache smaller datasets to avoid memory issues
        if len(file_list) < 1000:
            dataset = dataset.cache()
            print(f"[DEBUG] Dataset cached in memory (small dataset)")
        else:
            print(f"[DEBUG] Dataset not cached (large dataset - avoiding memory issues)")
        
        print(f"[DEBUG] Dataset created successfully with batch size {batch_size}")
        return dataset
    
    # Create datasets
    print("\nCreating TensorFlow datasets...")
    train_ds = create_dataset_from_directory(train_path)
    val_ds = create_dataset_from_directory(val_path)
    test_ds = create_dataset_from_directory(test_path)
    
    print("\n[DEBUG] All datasets created successfully!")
    
    # Test dataset iteration
    print("[DEBUG] Testing dataset iteration...")
    try:
        for batch in train_ds.take(1):
            images, labels = batch
            print(f"[DEBUG] Successfully loaded first batch: {len(images)} images")
            print(f"[DEBUG] Image shape: {images[0].shape}")
            break
        print("[DEBUG] Dataset iteration test completed successfully!")
    except Exception as e:
        print(f"[DEBUG] Error during dataset iteration test: {e}")
        raise e
    
    return train_ds, val_ds, test_ds

# Visualize some samples
def visualize_samples(dataset):
    print("[DEBUG] Starting sample visualization...")
    try:
        plt.figure(figsize=(12, 12))
        
        # Take only first batch and limit to avoid hanging
        sample_batch = None
        for batch in dataset.take(1):
            sample_batch = batch
            break
            
        if sample_batch is None:
            print("[DEBUG] No samples found in dataset")
            return
            
        images, labels = sample_batch
        print(f"[DEBUG] Got batch with {len(images)} images")
        
        # Limit to 9 images maximum
        num_samples = min(9, len(images))
        
        for i in range(num_samples):
            ax = plt.subplot(3, 3, i + 1)
            
            # Ensure image is in correct format
            img = images[i].numpy()
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            plt.title(class_names[labels[i]])
            plt.axis("off")
            
            print(f"[DEBUG] Plotted sample {i+1}/{num_samples}")
    
        sample_img_path = os.path.join(SCRIPT_DIR, 'sample_images_perfect_balanced.png')
        plt.savefig(sample_img_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Sample images saved to '{sample_img_path}'")
        
        # Log artifact to MLflow
        try:
            mlflow.log_artifact(sample_img_path)
            print("[DEBUG] Sample images logged to MLflow")
        except Exception as e:
            print(f"[DEBUG] Error logging sample images to MLflow: {e}")
            
    except Exception as e:
        print(f"[DEBUG] Error in visualize_samples: {e}")
        print("[DEBUG] Skipping sample visualization...")
        plt.close('all')  # Close any open figures

# Build the model with EfficientNet
def build_model():
    print("[DEBUG] Starting model building...")
    
    try:
        # Load the EfficientNetB0 model with pre-trained ImageNet weights
        print("[DEBUG] Loading EfficientNetB0 base model...")
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3),
            pooling="max"
        )
        print("[DEBUG] Base model loaded successfully!")
        
        # Freeze the base model
        print("[DEBUG] Freezing base model...")
        base_model.trainable = False
        print(f"[DEBUG] Base model frozen. Total layers: {len(base_model.layers)}")
        
        # Create new model on top
        print("[DEBUG] Building classification head...")
        inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
        
        # Preprocessing specific to EfficientNet
        x = keras.applications.efficientnet.preprocess_input(inputs)
        
        # The base model contains batchnorm layers that could be updated
        # during training even if trainable = False. To disable this,
        # we would need to set training = False
        x = base_model(x, training=False)
        
        # Add a classification head
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype='float32')(x)  # Keep output in float32 for mixed precision
        model = keras.Model(inputs, outputs)
        
        print("[DEBUG] Model architecture built successfully!")
        print(f"[DEBUG] Output classes: {NUM_CLASSES}")
        
        # Compile the model
        print("[DEBUG] Compiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("[DEBUG] Model compiled successfully!")
        
        return model, base_model
        
    except Exception as e:
        print(f"[DEBUG] Error in build_model: {e}")
        print("[DEBUG] This might be due to internet connectivity issues downloading ImageNet weights")
        print("[DEBUG] Retrying with local weights if available...")
        
        # Fallback: try without pre-trained weights if download fails
        try:
            print("[DEBUG] Attempting to build model without pre-trained weights...")
            base_model = keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,  # No pre-trained weights
                input_shape=(*IMAGE_SIZE, 3),
                pooling="max"
            )
            
            base_model.trainable = False
            
            inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
            x = keras.applications.efficientnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype='float32')(x)  # Keep output in float32 for mixed precision
            model = keras.Model(inputs, outputs)
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            print("[DEBUG] Model built successfully without pre-trained weights!")
            print("[DEBUG] WARNING: Model performance may be reduced without ImageNet weights")
            
            return model, base_model
            
        except Exception as e2:
            print(f"[DEBUG] Fallback also failed: {e2}")
            raise e2

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
def save_model(model, filename="efficientnet_b0_244_perfect_balanced"):
    # Save H5 model
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    print(f"Model saved as {model_path}")
    
    # Save TensorFlow Lite model
    tflite_path = os.path.join(SCRIPT_DIR, f"{filename}.tflite")
    convert_model_to_tflite(model, tflite_path)
    
    # Log model to MLflow
    mlflow.log_artifact(model_path)

# Main execution
def main():
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    start_time = datetime.now()
    
    with mlflow.start_run(run_name=f"EfficientNetB0_PerfectBalanced_{start_time.strftime('%Y%m%d-%H%M%S')}"):
        
        # Log parameters
        params = {
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            "batch_size": BATCH_SIZE,
            "epochs_initial_max": EPOCHS_INITIAL,
            "epochs_fine_tuning_max": EPOCHS_FINE_TUNING if USE_FINE_TUNING else 0,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_patience_fine_tuning": EARLY_STOP_PATIENCE_FINE_TUNING,
            "num_classes": NUM_CLASSES,
            "classes": str(class_names),
            "dataset_path": DATASET_PATH,
            "model_architecture": "EfficientNetB0",
            "dataset_type": "pre_split_perfect_balanced",
            "split_strategy": "pre_split_folders",
            "early_stopping_enabled": "True",
            "use_fine_tuning": str(USE_FINE_TUNING)
        }
        mlflow.log_params(params)
        
        # Prepare dataset
        print("Preparing dataset...")
        train_ds, val_ds, test_ds = prepare_dataset()
        
        print("[DEBUG] Dataset preparation completed successfully!")
        
        # Display some sample images
        print("Visualizing sample images...")
        try:
            visualize_samples(train_ds)
            print("[DEBUG] Sample visualization completed!")
        except Exception as e:
            print(f"[DEBUG] Error in sample visualization: {e}")
            print("[DEBUG] Skipping sample visualization and continuing...")
        
        # Build the model
        print("Building model...")
        try:
            model, base_model = build_model()
            print("[DEBUG] Model building completed!")
        except Exception as e:
            print(f"[DEBUG] Error building model: {e}")
            raise e
        
        # Summary of the model
        print("[DEBUG] Displaying model summary...")
        try:
            model.summary()
            print("[DEBUG] Model summary completed!")
        except Exception as e:
            print(f"[DEBUG] Error displaying model summary: {e}")
            print("[DEBUG] Continuing without summary...")
        
        # Create callbacks for training
        mlflow_callback = MLflowCallback()
        
        # Early stopping callback for initial training
        early_stop_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        
        # Train the model with frozen base
        print("\nInitial training with frozen base model...")
        print(f"[DEBUG] Starting initial training phase (max {EPOCHS_INITIAL} epochs with early stopping patience={EARLY_STOP_PATIENCE})...")
        mlflow.log_metric("training_phase", 1)  # Initial training phase
        
        try:
            history = model.fit(
                train_ds,
                epochs=EPOCHS_INITIAL,
                validation_data=val_ds,
                callbacks=[mlflow_callback, early_stop_callback]
            )
            print(f"[DEBUG] Initial training completed! Trained for {len(history.history['loss'])} epochs")
            
        except Exception as e:
            print(f"[DEBUG] Error during initial training: {e}")
            raise e
        
        # Plot training history
        plot_history(history, 'initial_training_history.png')
        
        # Track best validation metrics from initial training
        actual_epochs_initial = len(history.history['loss'])
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        mlflow.log_metric("initial_epochs_trained", actual_epochs_initial)
        mlflow.log_metric("initial_best_val_accuracy", best_val_acc)
        mlflow.log_metric("initial_best_val_loss", best_val_loss)
        print(f"[INFO] Initial training stopped at epoch {actual_epochs_initial}/{EPOCHS_INITIAL}")
        
        # Fine-tuning phase (optional)
        if USE_FINE_TUNING:
            print("\nFine-tuning the model...")
            model = unfreeze_model(model, base_model)
            
            # Log new phase
            mlflow.log_metric("training_phase", 2)  # Fine-tuning phase
            
            # Early stopping callback for fine-tuning (more aggressive)
            early_stop_fine_tuning = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOP_PATIENCE_FINE_TUNING,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            )
            
            print(f"[DEBUG] Starting fine-tuning phase (max {EPOCHS_FINE_TUNING} epochs with early stopping patience={EARLY_STOP_PATIENCE_FINE_TUNING})...")
            
            # Fine-tune the model
            fine_tune_history = model.fit(
                train_ds,
                epochs=EPOCHS_FINE_TUNING,
                validation_data=val_ds,
                callbacks=[mlflow_callback, early_stop_fine_tuning]
            )
            
            print(f"[DEBUG] Fine-tuning completed! Trained for {len(fine_tune_history.history['loss'])} epochs")
            
            # Plot fine-tuning history
            plot_history(fine_tune_history, 'fine_tuning_history.png')
            
            # Track best validation metrics from fine-tuning
            actual_epochs_fine_tuning = len(fine_tune_history.history['loss'])
            best_val_acc_ft = max(fine_tune_history.history['val_accuracy'])
            best_val_loss_ft = min(fine_tune_history.history['val_loss'])
            mlflow.log_metric("fine_tuning_epochs_trained", actual_epochs_fine_tuning)
            mlflow.log_metric("fine_tuning_best_val_accuracy", best_val_acc_ft)
            mlflow.log_metric("fine_tuning_best_val_loss", best_val_loss_ft)
            print(f"[INFO] Fine-tuning stopped at epoch {actual_epochs_fine_tuning}/{EPOCHS_FINE_TUNING}")
            
            # Log improvement metrics
            mlflow.log_metric("accuracy_improvement", best_val_acc_ft - best_val_acc)
            mlflow.log_metric("loss_improvement", best_val_loss - best_val_loss_ft)
        else:
            print("\n[INFO] Fine-tuning is DISABLED - Skipping fine-tuning phase")
            # Use initial training metrics as fine-tuning metrics
            fine_tune_history = None
            actual_epochs_fine_tuning = 0
            best_val_acc_ft = best_val_acc
            best_val_loss_ft = best_val_loss
            mlflow.log_metric("fine_tuning_epochs_trained", 0)
            mlflow.log_metric("fine_tuning_skipped", 1)
        
        # Evaluate model with confusion matrix and detailed metrics
        print("\n" + "="*60)
        print("EVALUATING MODEL PERFORMANCE ON TEST SET")
        print("="*60)
        cm, classification_report_dict, class_accuracies, auc_score = evaluate_model_with_confusion_matrix(model, test_ds, class_names)
        
        # Also evaluate on validation set for comparison
        print("\n" + "="*60)
        print("EVALUATING MODEL PERFORMANCE ON VALIDATION SET (for comparison)")
        print("="*60)
        cm_val, classification_report_dict_val, class_accuracies_val = evaluate_model_with_confusion_matrix_validation(model, val_ds, class_names)
        
        # Capture end time after fine tuning
        end_time = datetime.now()
        
        # Count model parameters
        model_params = model.count_params()
        
        # Save the model
        save_model(model, "efficientnet_b0_244_perfect_balanced")
        
        # ============================================
        # SAVE TRAINING RESULTS TO JSON
        # ============================================
        print("\n" + "="*60)
        print("SAVING TRAINING RESULTS TO JSON")
        print("="*60)
        
        # Extract final metrics from training histories
        train_acc, valid_acc, train_fine_tuning_acc, valid_finetuning_acc = extract_final_metrics(history, fine_tune_history)
        
        # Extract dataset name from path
        dataset_name = os.path.basename(DATASET_PATH)
        
        # Extract comprehensive metrics for JSON logging
        test_metrics, per_class_metrics = extract_comprehensive_metrics(
            classification_report_dict, class_accuracies, auc_score, class_names
        )
        
        # Extract basic test metrics for backward compatibility
        precision = classification_report_dict['macro avg']['precision']
        recall = classification_report_dict['macro avg']['recall']
        f1_score = classification_report_dict['macro avg']['f1-score']
        
        # Save training results to JSON with comprehensive metrics
        save_training_results(
            dataset_name=dataset_name,
            model_name="EfficientNet-B0",
            batch_size=BATCH_SIZE,
            epochs=actual_epochs_initial,
            fine_tuning_epochs=actual_epochs_fine_tuning,
            train_acc=train_acc,
            valid_acc=valid_acc,
            train_fine_tuning_acc=train_fine_tuning_acc,
            valid_finetuning_acc=valid_finetuning_acc,
            output_dir=os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "models_results"),
            model_params=model_params,
            training_start_time=start_time,
            training_end_time=end_time,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc=auc_score,
            test_metrics=test_metrics,
            per_class_metrics=per_class_metrics
        )
        
        # Print summary of saved metrics
        print("\n✓ Saved comprehensive metrics including:")
        print(f"  - Epochs trained: Initial={actual_epochs_initial}, Fine-tuning={actual_epochs_fine_tuning}")
        print(f"  - Overall test metrics: Accuracy, Precision, Recall, F1-Score, AUC")
        print(f"  - Per-class metrics for {len(class_names)} classes: Accuracy, Precision, Recall, F1-Score, Support")
        
        # Calculate total time
        total_time = datetime.now() - start_time
        hours, remainder = divmod(total_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        print(f"Total training time: {training_time_str}")
        
        # Log training time
        mlflow.log_metric("training_time_seconds", total_time.total_seconds())
        mlflow.log_param("training_time", training_time_str)
        
        # Find test images to predict (from test folder)
        test_path = os.path.join(DATASET_PATH, "teste")
        test_images = []
        for class_name in class_names:
            class_path = os.path.join(test_path, class_name)
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
            # Criar um codificador JSON personalizado para lidar com tipos NumPy/TensorFlow
            class NumpyJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
            
            results_path = os.path.join(SCRIPT_DIR, "prediction_results_perfect_balanced.json")
            with open(results_path, "w") as f:
                json.dump(prediction_results, f, indent=2, cls=NumpyJSONEncoder)
            
            mlflow.log_artifact(results_path)
    
    return model

# Generate confusion matrix and classification report
def evaluate_model_with_confusion_matrix(model, val_ds, class_names):
    """
    Generate confusion matrix and detailed classification metrics
    """
    print("\nGenerating confusion matrix and classification report...")
    
    # Get predictions for the entire validation set
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)
        y_pred_proba.extend(predictions)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate AUC (multi-class: one-vs-rest)
    auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy per class (correctly)
    # Per-class accuracy = (TP + TN) / Total
    # TP = diagonal element, TN = sum of all elements except the row and column of the class
    class_accuracies = []
    total_samples = len(y_true)
    for i in range(len(class_names)):
        tp = cm[i, i]  # True Positives
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]  # True Negatives
        class_accuracy = (tp + tn) / total_samples
        class_accuracies.append(class_accuracy)
    class_accuracies = np.array(class_accuracies)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(12, 10))
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Plot confusion matrix with percentages
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Confusion Matrix - Absolute Values (EfficientNet-B0)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_efficientnet_b0_perfect_balanced.png')
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
    plt.title('Confusion Matrix - Normalized (EfficientNet-B0)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_norm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_efficientnet_b0_perfect_balanced.png')
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
    
    metrics_path = os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_efficientnet_b0_perfect_balanced.png')
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
    mlflow.log_metric("auc_score", auc_score)
    
    # Log per-class metrics
    for i, class_name in enumerate(class_names):
        safe_class_name = sanitize_mlflow_metric_name(class_name)
        mlflow.log_metric(f"precision_{safe_class_name}", report[class_name]['precision'])
        mlflow.log_metric(f"recall_{safe_class_name}", report[class_name]['recall'])
        mlflow.log_metric(f"f1_score_{safe_class_name}", report[class_name]['f1-score'])
        mlflow.log_metric(f"support_{safe_class_name}", report[class_name]['support'])
        mlflow.log_metric(f"accuracy_{safe_class_name}", class_accuracies[i])
    
    # Save detailed report as JSON
    report_path = os.path.join(SCRIPT_DIR, "classification_report_efficientnet_b0_perfect_balanced.json")
    
    # Create a comprehensive report
    detailed_report = {
        "model_name": "EfficientNet-B0",
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
    print(f"Macro Average Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Average Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Average F1-Score: {report['weighted avg']['f1-score']:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
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
    
    print("\n" + "="*60)
    
    return cm, report, class_accuracies, auc_score

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
    
    # Calculate accuracy per class (correctly)
    # Per-class accuracy = (TP + TN) / Total
    class_accuracies = []
    total_samples = len(y_true)
    for i in range(len(class_names)):
        tp = cm[i, i]  # True Positives
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]  # True Negatives
        class_accuracy = (tp + tn) / total_samples
        class_accuracies.append(class_accuracy)
    class_accuracies = np.array(class_accuracies)
    
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

# Execute main function when script is run
if __name__ == "__main__":
    model = main()


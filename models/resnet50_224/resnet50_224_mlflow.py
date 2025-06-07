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
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNING = 10
DATASET_PATH = "datasets/dataset_wagner_balanceado_2"
EXPERIMENT_NAME = "ResNet50-SkinCondition"

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
    
    # Apply ResNet-specific preprocessing
    train_ds = train_ds.map(
        lambda x, y: (keras.applications.resnet.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    val_ds = val_ds.map(
        lambda x, y: (keras.applications.resnet.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
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
            # Convert preprocessed image back to displayable format
            display_img = (images[i] + 1) / 2  # This is an approximation for visualization
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(display_img)
            plt.title(class_names[labels[i]])
            plt.axis("off")
    
    sample_img_path = os.path.join(SCRIPT_DIR, 'sample_images.png')
    plt.savefig(sample_img_path)
    print(f"Sample images saved to '{sample_img_path}'")
    
    # Log artifact to MLflow
    mlflow.log_artifact(sample_img_path)

# Build the model with ResNet50
def build_model():
    # Load the ResNet50 model with pre-trained ImageNet weights
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="max"
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    # The base model contains batchnorm layers that could be updated
    # during training even if trainable = False. To disable this,
    # we would need to set training = False
    x = base_model(inputs, training=False)
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
    # For ResNet50, we'll unfreeze the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
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
    # Apply ResNet-specific preprocessing
    img_array = keras.applications.resnet.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")
    
    # Show the original image with predicted class
    plt.figure(figsize=(6, 6))
    plt.imshow(keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE))
    plt.title(f"Predicted: {class_names[predicted_class]} ({confidence:.2f})")
    plt.axis("off")
    
    pred_img_path = os.path.join(SCRIPT_DIR, f'prediction_{os.path.basename(image_path)}.png')
    plt.savefig(pred_img_path)
    print(f"Prediction visualization saved to '{pred_img_path}'")
    
    # Log prediction as artifact
    mlflow.log_artifact(pred_img_path)
    
    # Log metrics for the prediction - sanitize class names for MLflow metric names
    prediction_metrics = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    
    for class_name, prob in prediction_metrics.items():
        # Use sanitized class name for metric name
        safe_class_name = sanitize_mlflow_metric_name(class_name)
        mlflow.log_metric(f"prediction_prob_{safe_class_name}", prob)
    
    mlflow.log_metric("prediction_confidence", confidence)
    
    return class_names[predicted_class], confidence, prediction_metrics

# Generate confusion matrix and classification report
def evaluate_model_with_confusion_matrix(model, val_ds, class_names):
    """
    Generate confusion matrix and detailed classification metrics
    """
    print("\nGenerating confusion matrix and classification report...")
    
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
    
    # Create confusion matrix visualization
    plt.figure(figsize=(12, 10))
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Plot confusion matrix with percentages
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Confusion Matrix - Absolute Values (ResNet50)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_resnet50.png')
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
    plt.title('Confusion Matrix - Normalized (ResNet50)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_norm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_resnet50.png')
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
    plt.title('Classification Metrics by Class (ResNet50)', fontsize=16, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    metrics_path = os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_resnet50.png')
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
    report_path = os.path.join(SCRIPT_DIR, "classification_report_resnet50.json")
    
    # Create a comprehensive report
    detailed_report = {
        "model_name": "ResNet50",
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
    print("CONFUSION MATRIX ANALYSIS - ResNet50")
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
            plt.title('Top Class Confusions - ResNet50', fontsize=16, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Invert y-axis to show highest confusions at top
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            confusion_plot_path = os.path.join(SCRIPT_DIR, 'class_confusions_plot_resnet50.png')
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
    
    comprehensive_plot_path = os.path.join(SCRIPT_DIR, 'comprehensive_error_analysis_resnet50.png')
    plt.savefig(comprehensive_plot_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive error analysis plot saved to '{comprehensive_plot_path}'")
    mlflow.log_artifact(comprehensive_plot_path)
    plt.close()
    
    print("\n" + "="*60)
    
    return cm, report, class_accuracies

# Save the model
def save_model(model, filename="resnet50_224"):
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    print(f"Model saved as {model_path}")
    
    # Log model to MLflow
    mlflow.keras.log_model(model, "model")
    mlflow.log_artifact(model_path)

# Main execution
def main():
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"ResNet50_{time.strftime('%Y%m%d-%H%M%S')}"):
        start_time = time.time()
        
        # Log parameters
        params = {
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            "batch_size": BATCH_SIZE,
            "epochs_initial": EPOCHS_INITIAL,
            "epochs_fine_tuning": EPOCHS_FINE_TUNING,
            "num_classes": NUM_CLASSES,
            "classes": str(class_names),
            "dataset_path": DATASET_PATH,
            "model_architecture": "ResNet50"
        }
        mlflow.log_params(params)
        
        # Prepare dataset
        print("Preparing dataset...")
        train_ds, val_ds = prepare_dataset()
        
        # Log dataset size
        train_size = sum(1 for _ in train_ds)
        val_size = sum(1 for _ in val_ds)
        mlflow.log_metric("train_batches", train_size)
        mlflow.log_metric("val_batches", val_size)
        
        # Display some sample images
        print("Visualizing sample images...")
        visualize_samples(train_ds)
        
        # Build the model
        print("Building model...")
        model, base_model = build_model()
        
        # Summary of the model
        model.summary()
        
        # Create callbacks
        mlflow_callback = MLflowCallback()
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        checkpoint = keras.callbacks.ModelCheckpoint(
            'resnet50_224_checkpoint.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        callbacks = [mlflow_callback, early_stopping, checkpoint]
        
        # Train the model with frozen base
        print("\nInitial training with frozen base model...")
        mlflow.log_metric("training_phase", 1)  # Initial training phase
        
        history = model.fit(
            train_ds,
            epochs=EPOCHS_INITIAL,
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        # Plot training history
        plot_history(history, 'initial_training_history.png')
        
        # Track best validation metrics from initial training
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        mlflow.log_metric("initial_best_val_accuracy", best_val_acc)
        mlflow.log_metric("initial_best_val_loss", best_val_loss)
        
        # Unfreeze and fine-tune
        print("\nFine-tuning the model...")
        model = unfreeze_model(model, base_model)
        
        # Log new phase
        mlflow.log_metric("training_phase", 2)  # Fine-tuning phase
        
        # Fine-tune the model
        fine_tune_history = model.fit(
            train_ds,
            epochs=EPOCHS_FINE_TUNING,
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        # Plot fine-tuning history
        plot_history(fine_tune_history, 'fine_tuning_history.png')
        
        # Track best validation metrics from fine-tuning
        best_val_acc_ft = max(fine_tune_history.history['val_accuracy'])
        best_val_loss_ft = min(fine_tune_history.history['val_loss'])
        mlflow.log_metric("fine_tuning_best_val_accuracy", best_val_acc_ft)
        mlflow.log_metric("fine_tuning_best_val_loss", best_val_loss_ft)
        
        # Log improvement metrics
        mlflow.log_metric("accuracy_improvement", best_val_acc_ft - best_val_acc)
        mlflow.log_metric("loss_improvement", best_val_loss - best_val_loss_ft)
        
        # Evaluate model with confusion matrix and detailed metrics
        print("\n" + "="*60)
        print("EVALUATING MODEL PERFORMANCE - ResNet50")
        print("="*60)
        cm, classification_report_dict, class_accuracies = evaluate_model_with_confusion_matrix(model, val_ds, class_names)
        
        # Save the model
        save_model(model, "resnet50_224")
        
        # Calculate total time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        print(f"Total training time: {training_time_str}")
        
        # Log training time
        mlflow.log_metric("training_time_seconds", total_time)
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
                    "confidence": float(confidence)
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
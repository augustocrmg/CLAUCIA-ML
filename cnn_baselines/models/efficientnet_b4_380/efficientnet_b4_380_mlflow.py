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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_logger import save_training_results, extract_final_metrics, convert_model_to_tflite

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("GPU Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('GPU memory growth enabled')
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled')
    except RuntimeError as e:
        print(e)

IMAGE_SIZE = (380, 380)
BATCH_SIZE = 32
EPOCHS_INITIAL = 100
EPOCHS_FINE_TUNING = 20
DATASET_PATH = os.environ.get("CLAUCIA_CNN_DATASET", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))), "datasets", "perfect_balanced_dataset_wagner_v4_1000"))
EXPERIMENT_NAME = "EfficientNetB4-SkinCondition-PerfectBalanced"
USE_FINE_TUNING = False
EARLY_STOP_PATIENCE = 5
EARLY_STOP_PATIENCE_FINE_TUNING = 5

train_path = os.path.join(DATASET_PATH, "treino")
class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
NUM_CLASSES = len(class_names)
print(f"Classes: {class_names}")
print(f"Number of classes: {NUM_CLASSES}")

def sanitize_mlflow_metric_name(name):
    return name.replace("(", "_").replace(")", "_").replace(",", "_").replace("&", "_").replace(";", "_")

def extract_comprehensive_metrics(classification_report_dict, class_accuracies, auc_score, class_names):
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

def prepare_dataset():
    print("Loading pre-split dataset...")
    train_path = os.path.join(DATASET_PATH, "treino")
    val_path = os.path.join(DATASET_PATH, "validacao")
    test_path = os.path.join(DATASET_PATH, "teste")
    
    for split_name, split_path in [("Train", train_path), ("Validation", val_path), ("Test", test_path)]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_name} directory not found: {split_path}")
        print(f"{split_name} path: {split_path}")
    
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
    print(f"\nTotal images: Train: {train_total}, Validation: {val_total}, Test: {test_total}, Total: {train_total + val_total + test_total}")
    
    def create_dataset_from_directory(directory_path, batch_size=BATCH_SIZE):
        print(f"\n[DEBUG] Creating dataset from {directory_path}...")
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
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        random.shuffle(file_list)
        file_paths = [file_path for file_path, _ in file_list]
        labels = [class_to_idx[class_name] for _, class_name in file_list]
        
        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((path_ds, label_ds))
        
        def load_and_preprocess_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32)
            return image, label
        
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        dataset = dataset.shuffle(buffer_size=min(1000, len(file_list)), seed=42)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        if len(file_list) < 1000:
            dataset = dataset.cache()
        print(f"[DEBUG] Dataset created successfully with batch size {batch_size}")
        return dataset
    
    print("\nCreating TensorFlow datasets...")
    train_ds = create_dataset_from_directory(train_path)
    val_ds = create_dataset_from_directory(val_path)
    test_ds = create_dataset_from_directory(test_path)
    print("\n[DEBUG] All datasets created successfully!")
    
    print("[DEBUG] Testing dataset iteration...")
    try:
        for batch in train_ds.take(1):
            images, labels = batch
            print(f"[DEBUG] Successfully loaded first batch: {len(images)} images, Image shape: {images[0].shape}")
            break
    except Exception as e:
        print(f"[DEBUG] Error: {e}")
        raise e
    
    return train_ds, val_ds, test_ds

def visualize_samples(dataset):
    try:
        plt.figure(figsize=(12, 12))
        sample_batch = None
        for batch in dataset.take(1):
            sample_batch = batch
            break
        if sample_batch is None:
            return
        images, labels = sample_batch
        num_samples = min(9, len(images))
        for i in range(num_samples):
            ax = plt.subplot(3, 3, i + 1)
            img = images[i].numpy()
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title(class_names[labels[i]])
            plt.axis("off")
        sample_img_path = os.path.join(SCRIPT_DIR, 'sample_images_perfect_balanced.png')
        plt.savefig(sample_img_path, dpi=150, bbox_inches='tight')
        plt.close()
        try:
            mlflow.log_artifact(sample_img_path)
        except:
            pass
    except Exception as e:
        print(f"[DEBUG] Error in visualize_samples: {e}")
        plt.close('all')

def build_model():
    try:
        base_model = keras.applications.EfficientNetB4(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3), pooling="max")
        base_model.trainable = False
        inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype='float32')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model, base_model
    except Exception as e:
        base_model = keras.applications.EfficientNetB4(include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3), pooling="max")
        base_model.trainable = False
        inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype='float32')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model, base_model

class MLflowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)

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
    mlflow.log_artifact(plot_path)

def unfreeze_model(model, base_model):
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def predict_image(model, image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]} ({confidence:.2f})")
    plt.axis("off")
    pred_img_path = os.path.join(SCRIPT_DIR, f'prediction_{os.path.basename(image_path)}.png')
    plt.savefig(pred_img_path)
    mlflow.log_artifact(pred_img_path)
    prediction_metrics = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    for class_name, prob in prediction_metrics.items():
        mlflow.log_metric(f"prediction_prob_{sanitize_mlflow_metric_name(class_name)}", prob)
    mlflow.log_metric("prediction_confidence", confidence)
    return class_names[predicted_class], confidence, prediction_metrics

def save_model(model, filename="efficientnet_b4_380_perfect_balanced"):
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    tflite_path = os.path.join(SCRIPT_DIR, f"{filename}.tflite")
    convert_model_to_tflite(model, tflite_path)
    mlflow.log_artifact(model_path)

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    start_time = datetime.now()
    
    with mlflow.start_run(run_name=f"EfficientNetB4_PerfectBalanced_{start_time.strftime('%Y%m%d-%H%M%S')}"):
        params = {
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}", "batch_size": BATCH_SIZE,
            "epochs_initial_max": EPOCHS_INITIAL, "epochs_fine_tuning_max": EPOCHS_FINE_TUNING if USE_FINE_TUNING else 0,
            "early_stop_patience": EARLY_STOP_PATIENCE, "num_classes": NUM_CLASSES, "classes": str(class_names),
            "dataset_path": DATASET_PATH, "model_architecture": "EfficientNetB4",
            "dataset_type": "pre_split_perfect_balanced", "split_strategy": "pre_split_folders",
            "early_stopping_enabled": "True", "use_fine_tuning": str(USE_FINE_TUNING)
        }
        mlflow.log_params(params)
        
        train_ds, val_ds, test_ds = prepare_dataset()
        try:
            visualize_samples(train_ds)
        except:
            pass
        
        model, base_model = build_model()
        try:
            model.summary()
        except:
            pass
        
        mlflow_callback = MLflowCallback()
        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1, mode='min')
        
        mlflow.log_metric("training_phase", 1)
        history = model.fit(train_ds, epochs=EPOCHS_INITIAL, validation_data=val_ds, callbacks=[mlflow_callback, early_stop_callback])
        plot_history(history, 'initial_training_history.png')
        
        actual_epochs_initial = len(history.history['loss'])
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        mlflow.log_metric("initial_epochs_trained", actual_epochs_initial)
        mlflow.log_metric("initial_best_val_accuracy", best_val_acc)
        mlflow.log_metric("initial_best_val_loss", best_val_loss)
        
        if USE_FINE_TUNING:
            model = unfreeze_model(model, base_model)
            mlflow.log_metric("training_phase", 2)
            early_stop_fine_tuning = keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE_FINE_TUNING, restore_best_weights=True, verbose=1, mode='min')
            fine_tune_history = model.fit(train_ds, epochs=EPOCHS_FINE_TUNING, validation_data=val_ds, callbacks=[mlflow_callback, early_stop_fine_tuning])
            plot_history(fine_tune_history, 'fine_tuning_history.png')
            actual_epochs_fine_tuning = len(fine_tune_history.history['loss'])
            mlflow.log_metric("fine_tuning_epochs_trained", actual_epochs_fine_tuning)
        else:
            fine_tune_history = None
            actual_epochs_fine_tuning = 0
            mlflow.log_metric("fine_tuning_epochs_trained", 0)
            mlflow.log_metric("fine_tuning_skipped", 1)
        
        cm, classification_report_dict, class_accuracies, auc_score = evaluate_model_with_confusion_matrix(model, test_ds, class_names)
        cm_val, classification_report_dict_val, class_accuracies_val = evaluate_model_with_confusion_matrix_validation(model, val_ds, class_names)
        
        end_time = datetime.now()
        model_params = model.count_params()
        save_model(model, "efficientnet_b4_380_perfect_balanced")
        
        train_acc, valid_acc, train_fine_tuning_acc, valid_finetuning_acc = extract_final_metrics(history, fine_tune_history)
        dataset_name = os.path.basename(DATASET_PATH)
        test_metrics, per_class_metrics = extract_comprehensive_metrics(classification_report_dict, class_accuracies, auc_score, class_names)
        precision = classification_report_dict['macro avg']['precision']
        recall = classification_report_dict['macro avg']['recall']
        f1_score = classification_report_dict['macro avg']['f1-score']
        
        save_training_results(
            dataset_name=dataset_name, model_name="EfficientNet-B4", batch_size=BATCH_SIZE,
            epochs=actual_epochs_initial, fine_tuning_epochs=actual_epochs_fine_tuning,
            train_acc=train_acc, valid_acc=valid_acc, train_fine_tuning_acc=train_fine_tuning_acc,
            valid_finetuning_acc=valid_finetuning_acc,
            output_dir=os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "models_results"),
            model_params=model_params, training_start_time=start_time, training_end_time=end_time,
            precision=precision, recall=recall, f1_score=f1_score, auc=auc_score,
            test_metrics=test_metrics, per_class_metrics=per_class_metrics
        )
        
        total_time = datetime.now() - start_time
        hours, remainder = divmod(total_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        mlflow.log_metric("training_time_seconds", total_time.total_seconds())
        mlflow.log_param("training_time", training_time_str)
        
        test_path = os.path.join(DATASET_PATH, "teste")
        test_images = []
        for class_name in class_names:
            class_path = os.path.join(test_path, class_name)
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                if files:
                    test_images.append(os.path.join(class_path, files[0]))
        
        if test_images:
            prediction_results = {}
            for img_path in test_images[:3]:
                class_name, confidence, class_probs = predict_image(model, img_path)
                img_basename = os.path.basename(img_path)
                true_class = os.path.basename(os.path.dirname(img_path))
                is_correct = class_name == true_class
                prediction_results[img_basename] = {"predicted_class": class_name, "true_class": true_class, "correct": is_correct, "confidence": float(confidence)}
                mlflow.log_metric(f"prediction_correct_{sanitize_mlflow_metric_name(img_basename)}", 1 if is_correct else 0)
            
            class NumpyJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super().default(obj)
            
            results_path = os.path.join(SCRIPT_DIR, "prediction_results_perfect_balanced.json")
            with open(results_path, "w") as f:
                json.dump(prediction_results, f, indent=2, cls=NumpyJSONEncoder)
            mlflow.log_artifact(results_path)
    
    return model

def evaluate_model_with_confusion_matrix(model, val_ds, class_names):
    y_true, y_pred, y_pred_proba = [], [], []
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_pred_proba.extend(predictions)
    
    y_true, y_pred, y_pred_proba = np.array(y_true), np.array(y_pred), np.array(y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    class_accuracies = []
    total_samples = len(y_true)
    for i in range(len(class_names)):
        tp = cm[i, i]
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        class_accuracies.append((tp + tn) / total_samples)
    class_accuracies = np.array(class_accuracies)
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    plt.figure(figsize=(12, 10))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5)
    plt.title('Confusion Matrix - Absolute Values (EfficientNet-B4)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_efficientnet_b4_perfect_balanced.png')
    plt.savefig(cm_path, dpi=300)
    mlflow.log_artifact(cm_path)
    plt.close()
    
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    sns.heatmap(cm_norm_df, annot=True, fmt='.2%', cmap='Blues', square=True, linewidths=0.5)
    plt.title('Confusion Matrix - Normalized (EfficientNet-B4)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_norm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_efficientnet_b4_perfect_balanced.png')
    plt.savefig(cm_norm_path, dpi=300)
    mlflow.log_artifact(cm_norm_path)
    plt.close()
    
    plt.figure(figsize=(14, 8))
    precision_values = [report[cn]['precision'] for cn in class_names]
    recall_values = [report[cn]['recall'] for cn in class_names]
    f1_values = [report[cn]['f1-score'] for cn in class_names]
    x = np.arange(len(class_names))
    width = 0.25
    plt.bar(x - width, precision_values, width, label='Precision', alpha=0.8)
    plt.bar(x, recall_values, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_values, width, label='F1-Score', alpha=0.8)
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Classification Metrics by Class', fontsize=16, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    metrics_path = os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_efficientnet_b4_perfect_balanced.png')
    plt.savefig(metrics_path, dpi=300)
    mlflow.log_artifact(metrics_path)
    plt.close()
    
    overall_accuracy = np.sum(y_true == y_pred) / len(y_true)
    mlflow.log_metric("confusion_matrix_accuracy", overall_accuracy)
    mlflow.log_metric("macro_avg_precision", report['macro avg']['precision'])
    mlflow.log_metric("macro_avg_recall", report['macro avg']['recall'])
    mlflow.log_metric("macro_avg_f1", report['macro avg']['f1-score'])
    mlflow.log_metric("weighted_avg_precision", report['weighted avg']['precision'])
    mlflow.log_metric("weighted_avg_recall", report['weighted avg']['recall'])
    mlflow.log_metric("weighted_avg_f1", report['weighted avg']['f1-score'])
    mlflow.log_metric("auc_score", auc_score)
    
    for i, cn in enumerate(class_names):
        safe_cn = sanitize_mlflow_metric_name(cn)
        mlflow.log_metric(f"precision_{safe_cn}", report[cn]['precision'])
        mlflow.log_metric(f"recall_{safe_cn}", report[cn]['recall'])
        mlflow.log_metric(f"f1_score_{safe_cn}", report[cn]['f1-score'])
        mlflow.log_metric(f"support_{safe_cn}", report[cn]['support'])
        mlflow.log_metric(f"accuracy_{safe_cn}", class_accuracies[i])
    
    report_path = os.path.join(SCRIPT_DIR, "classification_report_efficientnet_b4_perfect_balanced.json")
    detailed_report = {"model_name": "EfficientNet-B4", "classification_report": report, "class_accuracies": {class_names[i]: float(class_accuracies[i]) for i in range(len(class_names))}, "confusion_matrix": cm.tolist(), "normalized_confusion_matrix": cm_normalized.tolist()}
    with open(report_path, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    mlflow.log_artifact(report_path)
    
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}, AUC: {auc_score:.4f}")
    return cm, report, class_accuracies, auc_score

def evaluate_model_with_confusion_matrix_validation(model, val_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    class_accuracies = []
    total_samples = len(y_true)
    for i in range(len(class_names)):
        tp = cm[i, i]
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        class_accuracies.append((tp + tn) / total_samples)
    class_accuracies = np.array(class_accuracies)
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    overall_accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    mlflow.log_metric("val_confusion_matrix_accuracy", overall_accuracy)
    mlflow.log_metric("val_macro_avg_precision", report['macro avg']['precision'])
    mlflow.log_metric("val_macro_avg_recall", report['macro avg']['recall'])
    mlflow.log_metric("val_macro_avg_f1", report['macro avg']['f1-score'])
    mlflow.log_metric("val_weighted_avg_precision", report['weighted avg']['precision'])
    mlflow.log_metric("val_weighted_avg_recall", report['weighted avg']['recall'])
    mlflow.log_metric("val_weighted_avg_f1", report['weighted avg']['f1-score'])
    
    for i, cn in enumerate(class_names):
        safe_cn = sanitize_mlflow_metric_name(cn)
        mlflow.log_metric(f"val_precision_{safe_cn}", report[cn]['precision'])
        mlflow.log_metric(f"val_recall_{safe_cn}", report[cn]['recall'])
        mlflow.log_metric(f"val_f1_score_{safe_cn}", report[cn]['f1-score'])
        mlflow.log_metric(f"val_support_{safe_cn}", report[cn]['support'])
        mlflow.log_metric(f"val_accuracy_{safe_cn}", class_accuracies[i])
    
    print(f"\nValidation Set Overall Accuracy: {overall_accuracy:.4f}")
    return cm, report, class_accuracies

if __name__ == "__main__":
    model = main()

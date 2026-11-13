import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import mlflow
import mlflow.keras
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import pandas as pd
import glob
import random
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_logger import save_training_results, extract_final_metrics, convert_model_to_tflite
from runtime_config import configure_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure GPU memory growth, mixed precision and Blackwell sm_120 fallbacks.
# See models/runtime_config.py for the full rationale and env-var escape hatches
# (FORCE_CPU=1, DISABLE_MIXED_PRECISION=1).
configure_runtime()

IMAGE_SIZE = (456, 456)
BATCH_SIZE = 32
EPOCHS_INITIAL = 100
EPOCHS_FINE_TUNING = 20
DATASET_PATH = os.environ.get("CLAUCIA_CNN_DATASET", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))), "datasets", "perfect_balanced_dataset_wagner_v4_1000"))
EXPERIMENT_NAME = "EfficientNetB5-SkinCondition-PerfectBalanced"
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
    test_metrics = {"accuracy": float(classification_report_dict['accuracy']), "precision_macro": float(classification_report_dict['macro avg']['precision']), "recall_macro": float(classification_report_dict['macro avg']['recall']), "f1_score_macro": float(classification_report_dict['macro avg']['f1-score']), "precision_weighted": float(classification_report_dict['weighted avg']['precision']), "recall_weighted": float(classification_report_dict['weighted avg']['recall']), "f1_score_weighted": float(classification_report_dict['weighted avg']['f1-score']), "auc": float(auc_score)}
    per_class_metrics = {}
    for i, cn in enumerate(class_names):
        per_class_metrics[cn] = {"accuracy": float(class_accuracies[i]), "precision": float(classification_report_dict[cn]['precision']), "recall": float(classification_report_dict[cn]['recall']), "f1_score": float(classification_report_dict[cn]['f1-score']), "support": int(classification_report_dict[cn]['support'])}
    return test_metrics, per_class_metrics

def prepare_dataset():
    print("Loading pre-split dataset...")
    train_path = os.path.join(DATASET_PATH, "treino")
    val_path = os.path.join(DATASET_PATH, "validacao")
    test_path = os.path.join(DATASET_PATH, "teste")
    for split_name, split_path in [("Train", train_path), ("Validation", val_path), ("Test", test_path)]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_name} directory not found: {split_path}")

    def create_dataset_from_directory(directory_path, batch_size=BATCH_SIZE):
        file_list = []
        for cn in class_names:
            class_path = os.path.join(directory_path, cn)
            if os.path.isdir(class_path):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']:
                    for f in glob.glob(os.path.join(class_path, ext)) + glob.glob(os.path.join(class_path, ext.upper())):
                        file_list.append((f, cn))
        class_to_idx = {cn: idx for idx, cn in enumerate(class_names)}
        random.shuffle(file_list)
        file_paths = [fp for fp, _ in file_list]
        labels = [class_to_idx[cn] for _, cn in file_list]
        # Pin every input-pipeline op to the CPU. tf.data input pipelines should
        # never need GPU kernels, and pinning here avoids an `Equal` kernel
        # being JIT-compiled from PTX inside `dataset.shuffle(seed=...)`, which
        # crashes on GPUs newer than the TF binary supports (e.g. Blackwell
        # sm_120 -> CUDA_ERROR_INVALID_PTX / CUDA_ERROR_INVALID_HANDLE).
        with tf.device('/CPU:0'):
            path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
            label_ds = tf.data.Dataset.from_tensor_slices(labels)
            dataset = tf.data.Dataset.zip((path_ds, label_ds))
            def load_img(path, label):
                with tf.device('/CPU:0'):
                    image = tf.io.read_file(path)
                    image = tf.image.decode_image(image, channels=3, expand_animations=False)
                    image = tf.image.resize(image, IMAGE_SIZE)
                    image = tf.cast(image, tf.float32)
                return image, label
            dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            dataset = dataset.shuffle(buffer_size=min(1000, len(file_list)), seed=42).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
            if len(file_list) < 1000:
                dataset = dataset.cache()
        return dataset

    with tf.device('/CPU:0'):
        train_ds = create_dataset_from_directory(train_path)
        val_ds = create_dataset_from_directory(val_path)
        test_ds = create_dataset_from_directory(test_path)
    return train_ds, val_ds, test_ds

def visualize_samples(dataset):
    try:
        plt.figure(figsize=(12, 12))
        for batch in dataset.take(1):
            images, labels = batch
            for i in range(min(9, len(images))):
                plt.subplot(3, 3, i + 1)
                img = images[i].numpy()
                if img.max() > 1.0: img = img / 255.0
                plt.imshow(np.clip(img, 0, 1))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.savefig(os.path.join(SCRIPT_DIR, 'sample_images_perfect_balanced.png'), dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(os.path.join(SCRIPT_DIR, 'sample_images_perfect_balanced.png'))
    except:
        plt.close('all')

def build_model():
    try:
        base_model = keras.applications.EfficientNetB5(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3), pooling="max")
    except:
        base_model = keras.applications.EfficientNetB5(include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3), pooling="max")
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
        for k, v in (logs or {}).items():
            mlflow.log_metric(k, v, step=epoch)

def plot_history(history, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.legend()
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.legend()
    plt.savefig(os.path.join(SCRIPT_DIR, filename))
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, filename))

def unfreeze_model(model, base_model):
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def predict_image(model, image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = np.expand_dims(keras.preprocessing.image.img_to_array(img), axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]} ({confidence:.2f})")
    plt.axis("off")
    pred_path = os.path.join(SCRIPT_DIR, f'prediction_{os.path.basename(image_path)}.png')
    plt.savefig(pred_path)
    mlflow.log_artifact(pred_path)
    for i, cn in enumerate(class_names):
        mlflow.log_metric(f"prediction_prob_{sanitize_mlflow_metric_name(cn)}", float(predictions[0][i]))
    mlflow.log_metric("prediction_confidence", confidence)
    return class_names[predicted_class], confidence, {cn: float(predictions[0][i]) for i, cn in enumerate(class_names)}

def save_model(model, filename="efficientnet_b5_456_perfect_balanced"):
    model_path = os.path.join(SCRIPT_DIR, f"{filename}.h5")
    model.save(model_path)
    convert_model_to_tflite(model, os.path.join(SCRIPT_DIR, f"{filename}.tflite"))
    mlflow.log_artifact(model_path)

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    start_time = datetime.now()
    with mlflow.start_run(run_name=f"EfficientNetB5_PerfectBalanced_{start_time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params({"image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}", "batch_size": BATCH_SIZE, "epochs_initial_max": EPOCHS_INITIAL, "model_architecture": "EfficientNetB5", "dataset_type": "pre_split_perfect_balanced", "split_strategy": "pre_split_folders", "use_fine_tuning": str(USE_FINE_TUNING)})
        train_ds, val_ds, test_ds = prepare_dataset()
        try: visualize_samples(train_ds)
        except: pass
        model, base_model = build_model()
        mlflow_callback = MLflowCallback()
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1, mode='min')
        mlflow.log_metric("training_phase", 1)
        history = model.fit(train_ds, epochs=EPOCHS_INITIAL, validation_data=val_ds, callbacks=[mlflow_callback, early_stop])
        plot_history(history, 'initial_training_history.png')
        actual_epochs_initial = len(history.history['loss'])
        mlflow.log_metric("initial_epochs_trained", actual_epochs_initial)
        mlflow.log_metric("initial_best_val_accuracy", max(history.history['val_accuracy']))
        if USE_FINE_TUNING:
            model = unfreeze_model(model, base_model)
            mlflow.log_metric("training_phase", 2)
            fine_tune_history = model.fit(train_ds, epochs=EPOCHS_FINE_TUNING, validation_data=val_ds, callbacks=[mlflow_callback, keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE_FINE_TUNING, restore_best_weights=True, verbose=1, mode='min')])
            plot_history(fine_tune_history, 'fine_tuning_history.png')
            actual_epochs_fine_tuning = len(fine_tune_history.history['loss'])
        else:
            fine_tune_history = None
            actual_epochs_fine_tuning = 0
            mlflow.log_metric("fine_tuning_skipped", 1)
        cm, report, class_accuracies, auc_score = evaluate_model_with_confusion_matrix(model, test_ds, class_names)
        evaluate_model_with_confusion_matrix_validation(model, val_ds, class_names)
        end_time = datetime.now()
        save_model(model, "efficientnet_b5_456_perfect_balanced")
        train_acc, valid_acc, train_ft, valid_ft = extract_final_metrics(history, fine_tune_history)
        test_metrics, per_class_metrics = extract_comprehensive_metrics(report, class_accuracies, auc_score, class_names)
        save_training_results(dataset_name=os.path.basename(DATASET_PATH), model_name="EfficientNet-B5", batch_size=BATCH_SIZE, epochs=actual_epochs_initial, fine_tuning_epochs=actual_epochs_fine_tuning, train_acc=train_acc, valid_acc=valid_acc, train_fine_tuning_acc=train_ft, valid_finetuning_acc=valid_ft, output_dir=os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "models_results"), model_params=model.count_params(), training_start_time=start_time, training_end_time=end_time, precision=report['macro avg']['precision'], recall=report['macro avg']['recall'], f1_score=report['macro avg']['f1-score'], auc=auc_score, test_metrics=test_metrics, per_class_metrics=per_class_metrics)
        total_time = datetime.now() - start_time
        mlflow.log_metric("training_time_seconds", total_time.total_seconds())
        test_path = os.path.join(DATASET_PATH, "teste")
        test_images = [os.path.join(test_path, cn, os.listdir(os.path.join(test_path, cn))[0]) for cn in class_names if os.path.isdir(os.path.join(test_path, cn)) and os.listdir(os.path.join(test_path, cn))]
        if test_images:
            prediction_results = {}
            for img_path in test_images[:3]:
                cn, conf, _ = predict_image(model, img_path)
                true_class = os.path.basename(os.path.dirname(img_path))
                prediction_results[os.path.basename(img_path)] = {"predicted_class": cn, "true_class": true_class, "correct": cn == true_class, "confidence": float(conf)}
            with open(os.path.join(SCRIPT_DIR, "prediction_results_perfect_balanced.json"), "w") as f:
                json.dump(prediction_results, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o.tolist() if isinstance(o, np.ndarray) else o)
            mlflow.log_artifact(os.path.join(SCRIPT_DIR, "prediction_results_perfect_balanced.json"))
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
    class_accuracies = np.array([(cm[i, i] + np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]) / len(y_true) for i in range(len(class_names))])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True, fmt='d', cmap='Blues', square=True)
    plt.title('Confusion Matrix - EfficientNet-B5', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'confusion_matrix_efficientnet_b5_perfect_balanced.png'), dpi=300)
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, 'confusion_matrix_efficientnet_b5_perfect_balanced.png'))
    plt.close()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(cm_normalized, index=class_names, columns=class_names), annot=True, fmt='.2%', cmap='Blues', square=True)
    plt.title('Confusion Matrix - Normalized (EfficientNet-B5)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_efficientnet_b5_perfect_balanced.png'), dpi=300)
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_efficientnet_b5_perfect_balanced.png'))
    plt.close()
    plt.figure(figsize=(14, 8))
    x = np.arange(len(class_names))
    plt.bar(x - 0.25, [report[cn]['precision'] for cn in class_names], 0.25, label='Precision', alpha=0.8)
    plt.bar(x, [report[cn]['recall'] for cn in class_names], 0.25, label='Recall', alpha=0.8)
    plt.bar(x + 0.25, [report[cn]['f1-score'] for cn in class_names], 0.25, label='F1-Score', alpha=0.8)
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Classification Metrics by Class', fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_efficientnet_b5_perfect_balanced.png'), dpi=300)
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_efficientnet_b5_perfect_balanced.png'))
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
    with open(os.path.join(SCRIPT_DIR, "classification_report_efficientnet_b5_perfect_balanced.json"), 'w') as f:
        json.dump({"model_name": "EfficientNet-B5", "classification_report": report, "class_accuracies": {cn: float(class_accuracies[i]) for i, cn in enumerate(class_names)}, "confusion_matrix": cm.tolist(), "normalized_confusion_matrix": cm_normalized.tolist()}, f, indent=2)
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, "classification_report_efficientnet_b5_perfect_balanced.json"))
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
    class_accuracies = np.array([(cm[i, i] + np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]) / len(y_true) for i in range(len(class_names))])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    overall_accuracy = np.sum(y_true == y_pred) / len(y_true)
    mlflow.log_metric("val_confusion_matrix_accuracy", overall_accuracy)
    mlflow.log_metric("val_macro_avg_precision", report['macro avg']['precision'])
    mlflow.log_metric("val_macro_avg_recall", report['macro avg']['recall'])
    mlflow.log_metric("val_macro_avg_f1", report['macro avg']['f1-score'])
    for i, cn in enumerate(class_names):
        safe_cn = sanitize_mlflow_metric_name(cn)
        mlflow.log_metric(f"val_precision_{safe_cn}", report[cn]['precision'])
        mlflow.log_metric(f"val_recall_{safe_cn}", report[cn]['recall'])
        mlflow.log_metric(f"val_f1_score_{safe_cn}", report[cn]['f1-score'])
        mlflow.log_metric(f"val_accuracy_{safe_cn}", class_accuracies[i])
    print(f"\nValidation Set Overall Accuracy: {overall_accuracy:.4f}")
    return cm, report, class_accuracies

if __name__ == "__main__":
    model = main()

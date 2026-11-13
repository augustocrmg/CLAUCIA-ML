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

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 100
EPOCHS_FINE_TUNING = 20
DATASET_PATH = os.environ.get("CLAUCIA_CNN_DATASET", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))), "datasets", "perfect_balanced_dataset_wagner_v4_1000"))
EXPERIMENT_NAME = "ResNet50-SkinCondition-PerfectBalanced"
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

def extract_comprehensive_metrics(report, class_accuracies, auc_score, class_names):
    test_metrics = {"accuracy": float(report['accuracy']), "precision_macro": float(report['macro avg']['precision']), "recall_macro": float(report['macro avg']['recall']), "f1_score_macro": float(report['macro avg']['f1-score']), "precision_weighted": float(report['weighted avg']['precision']), "recall_weighted": float(report['weighted avg']['recall']), "f1_score_weighted": float(report['weighted avg']['f1-score']), "auc": float(auc_score)}
    per_class_metrics = {cn: {"accuracy": float(class_accuracies[i]), "precision": float(report[cn]['precision']), "recall": float(report[cn]['recall']), "f1_score": float(report[cn]['f1-score']), "support": int(report[cn]['support'])} for i, cn in enumerate(class_names)}
    return test_metrics, per_class_metrics

def prepare_dataset():
    train_path, val_path, test_path = os.path.join(DATASET_PATH, "treino"), os.path.join(DATASET_PATH, "validacao"), os.path.join(DATASET_PATH, "teste")
    for name, path in [("Train", train_path), ("Validation", val_path), ("Test", test_path)]:
        if not os.path.exists(path): raise FileNotFoundError(f"{name} directory not found: {path}")
    
    def create_ds(dir_path, batch_size=BATCH_SIZE):
        file_list = [(f, cn) for cn in class_names if os.path.isdir(os.path.join(dir_path, cn)) for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff'] for f in glob.glob(os.path.join(dir_path, cn, ext)) + glob.glob(os.path.join(dir_path, cn, ext.upper()))]
        class_to_idx = {cn: i for i, cn in enumerate(class_names)}
        random.shuffle(file_list)
        paths, labels = [f for f, _ in file_list], [class_to_idx[cn] for _, cn in file_list]
        ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(paths), tf.data.Dataset.from_tensor_slices(labels)))
        def load_img(p, l):
            img = tf.io.read_file(p)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            return tf.cast(tf.image.resize(img, IMAGE_SIZE), tf.float32), l
        ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).shuffle(min(1000, len(file_list)), seed=42).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds.cache() if len(file_list) < 1000 else ds
    return create_ds(train_path), create_ds(val_path), create_ds(test_path)

def build_model():
    try: base_model = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3), pooling="max")
    except: base_model = keras.applications.ResNet50(include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3), pooling="max")
    base_model.trainable = False
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype='float32')(layers.Dropout(0.2)(x))
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base_model

class MLflowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items(): mlflow.log_metric(k, v, step=epoch)

def unfreeze_model(model, base_model):
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization): layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def save_model(model, filename="resnet50_224_perfect_balanced"):
    model.save(os.path.join(SCRIPT_DIR, f"{filename}.h5"))
    convert_model_to_tflite(model, os.path.join(SCRIPT_DIR, f"{filename}.tflite"))
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, f"{filename}.h5"))

def evaluate_model_with_confusion_matrix(model, ds, class_names):
    y_true, y_pred, y_proba = [], [], []
    for imgs, lbls in ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(lbls.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_proba.extend(preds)
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    cm = confusion_matrix(y_true, y_pred)
    class_acc = np.array([(cm[i,i] + np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]) / len(y_true) for i in range(len(class_names))])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True, fmt='d', cmap='Blues', square=True)
    plt.title('Confusion Matrix - ResNet50', fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'confusion_matrix_resnet50_perfect_balanced.png'), dpi=300)
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, 'confusion_matrix_resnet50_perfect_balanced.png')); plt.close()
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(cm_norm, index=class_names, columns=class_names), annot=True, fmt='.2%', cmap='Blues', square=True)
    plt.title('Confusion Matrix - Normalized (ResNet50)', fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_resnet50_perfect_balanced.png'), dpi=300)
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, 'confusion_matrix_normalized_resnet50_perfect_balanced.png')); plt.close()
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(class_names))
    plt.bar(x - 0.25, [report[cn]['precision'] for cn in class_names], 0.25, label='Precision', alpha=0.8)
    plt.bar(x, [report[cn]['recall'] for cn in class_names], 0.25, label='Recall', alpha=0.8)
    plt.bar(x + 0.25, [report[cn]['f1-score'] for cn in class_names], 0.25, label='F1-Score', alpha=0.8)
    plt.xlabel('Classes'); plt.ylabel('Score'); plt.title('Classification Metrics by Class', fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right'); plt.legend(); plt.ylim(0, 1.1); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_resnet50_perfect_balanced.png'), dpi=300)
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, 'classification_metrics_by_class_resnet50_perfect_balanced.png')); plt.close()
    
    overall_acc = np.sum(y_true == y_pred) / len(y_true)
    mlflow.log_metric("confusion_matrix_accuracy", overall_acc)
    mlflow.log_metric("macro_avg_precision", report['macro avg']['precision'])
    mlflow.log_metric("macro_avg_recall", report['macro avg']['recall'])
    mlflow.log_metric("macro_avg_f1", report['macro avg']['f1-score'])
    mlflow.log_metric("auc_score", auc)
    for i, cn in enumerate(class_names):
        s = sanitize_mlflow_metric_name(cn)
        mlflow.log_metric(f"precision_{s}", report[cn]['precision'])
        mlflow.log_metric(f"recall_{s}", report[cn]['recall'])
        mlflow.log_metric(f"f1_score_{s}", report[cn]['f1-score'])
        mlflow.log_metric(f"accuracy_{s}", class_acc[i])
    with open(os.path.join(SCRIPT_DIR, "classification_report_resnet50_perfect_balanced.json"), 'w') as f:
        json.dump({"model_name": "ResNet50", "classification_report": report, "class_accuracies": {cn: float(class_acc[i]) for i, cn in enumerate(class_names)}, "confusion_matrix": cm.tolist()}, f, indent=2)
    mlflow.log_artifact(os.path.join(SCRIPT_DIR, "classification_report_resnet50_perfect_balanced.json"))
    return cm, report, class_acc, auc

def evaluate_model_with_confusion_matrix_validation(model, ds, class_names):
    y_true, y_pred = [], []
    for imgs, lbls in ds:
        y_true.extend(lbls.numpy())
        y_pred.extend(np.argmax(model.predict(imgs, verbose=0), axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    class_acc = np.array([(cm[i,i] + np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]) / len(y_true) for i in range(len(class_names))])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    overall_acc = np.sum(y_true == y_pred) / len(y_true)
    mlflow.log_metric("val_confusion_matrix_accuracy", overall_acc)
    mlflow.log_metric("val_macro_avg_f1", report['macro avg']['f1-score'])
    for i, cn in enumerate(class_names):
        s = sanitize_mlflow_metric_name(cn)
        mlflow.log_metric(f"val_precision_{s}", report[cn]['precision'])
        mlflow.log_metric(f"val_recall_{s}", report[cn]['recall'])
        mlflow.log_metric(f"val_accuracy_{s}", class_acc[i])
    return cm, report, class_acc

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    start_time = datetime.now()
    with mlflow.start_run(run_name=f"ResNet50_PerfectBalanced_{start_time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params({"image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}", "batch_size": BATCH_SIZE, "model_architecture": "ResNet50", "dataset_type": "pre_split_perfect_balanced", "split_strategy": "pre_split_folders", "use_fine_tuning": str(USE_FINE_TUNING)})
        train_ds, val_ds, test_ds = prepare_dataset()
        model, base_model = build_model()
        mlflow.log_metric("training_phase", 1)
        history = model.fit(train_ds, epochs=EPOCHS_INITIAL, validation_data=val_ds, callbacks=[MLflowCallback(), keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1, mode='min')])
        actual_epochs_initial = len(history.history['loss'])
        mlflow.log_metric("initial_epochs_trained", actual_epochs_initial)
        if USE_FINE_TUNING:
            model = unfreeze_model(model, base_model)
            fine_tune_history = model.fit(train_ds, epochs=EPOCHS_FINE_TUNING, validation_data=val_ds, callbacks=[MLflowCallback(), keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE_FINE_TUNING, restore_best_weights=True)])
            actual_epochs_fine_tuning = len(fine_tune_history.history['loss'])
        else:
            fine_tune_history = None
            actual_epochs_fine_tuning = 0
            mlflow.log_metric("fine_tuning_skipped", 1)
        cm, report, class_acc, auc = evaluate_model_with_confusion_matrix(model, test_ds, class_names)
        evaluate_model_with_confusion_matrix_validation(model, val_ds, class_names)
        save_model(model, "resnet50_224_perfect_balanced")
        train_acc, valid_acc, train_ft, valid_ft = extract_final_metrics(history, fine_tune_history)
        test_metrics, per_class_metrics = extract_comprehensive_metrics(report, class_acc, auc, class_names)
        save_training_results(dataset_name=os.path.basename(DATASET_PATH), model_name="ResNet50", batch_size=BATCH_SIZE, epochs=actual_epochs_initial, fine_tuning_epochs=actual_epochs_fine_tuning, train_acc=train_acc, valid_acc=valid_acc, train_fine_tuning_acc=train_ft, valid_finetuning_acc=valid_ft, output_dir=os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "models_results"), model_params=model.count_params(), training_start_time=start_time, training_end_time=datetime.now(), precision=report['macro avg']['precision'], recall=report['macro avg']['recall'], f1_score=report['macro avg']['f1-score'], auc=auc, test_metrics=test_metrics, per_class_metrics=per_class_metrics)
        mlflow.log_metric("training_time_seconds", (datetime.now() - start_time).total_seconds())
    return model

if __name__ == "__main__":
    model = main()

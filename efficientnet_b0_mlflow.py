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
EXPERIMENT_NAME = "EfficientNetB0-SkinCondition"

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
    
    sample_img_path = 'sample_images.png'
    plt.savefig(sample_img_path)
    print(f"Sample images saved to '{sample_img_path}'")
    
    # Log artifact to MLflow
    mlflow.log_artifact(sample_img_path)

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
    
    plt.savefig(filename)
    print(f"Training history saved to '{filename}'")
    
    # Log the plot as an artifact
    mlflow.log_artifact(filename)

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
    
    pred_img_path = f'prediction_{os.path.basename(image_path)}.png'
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

# Save the model
def save_model(model, filename="skin_condition_model"):
    model_path = f"{filename}.h5"
    model.save(model_path)
    print(f"Model saved as {model_path}")
    
    # Log the model to MLflow
    mlflow.keras.log_model(model, "model")
    mlflow.log_artifact(model_path)

# Main execution
def main():
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"EfficientNetB0_{time.strftime('%Y%m%d-%H%M%S')}"):
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
            "model_architecture": "EfficientNetB0"
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
        
        # Create MLflow callback for logging during training
        mlflow_callback = MLflowCallback()
        
        # Train the model with frozen base
        print("\nInitial training with frozen base model...")
        mlflow.log_metric("training_phase", 1)  # Initial training phase
        
        history = model.fit(
            train_ds,
            epochs=EPOCHS_INITIAL,
            validation_data=val_ds,
            callbacks=[mlflow_callback]
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
            callbacks=[mlflow_callback]
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
        
        # Save the model
        save_model(model, "skin_condition_efficientnet_model")
        
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
            
            results_path = "prediction_results.json"
            with open(results_path, "w") as f:
                json.dump(prediction_results, f, indent=2, cls=NumpyJSONEncoder)
            
            mlflow.log_artifact(results_path)
    
    return model

# Execute main function when script is run
if __name__ == "__main__":
    model = main() 
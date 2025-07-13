import json
import os
from datetime import datetime
import tensorflow as tf


def save_training_results(
    dataset_name,
    model_name,
    batch_size,
    epochs,
    fine_tuning_epochs,
    train_acc,
    valid_acc,
    train_fine_tuning_acc,
    valid_finetuning_acc,
    output_dir="../models_results",
    model_params=None,
    training_start_time=None,
    training_end_time=None
):
    """
    Save training results to JSON file.
    
    Args:
        dataset_name: Name of the dataset used
        model_name: Name of the model architecture
        batch_size: Batch size used for training
        epochs: Number of initial training epochs
        fine_tuning_epochs: Number of fine-tuning epochs
        train_acc: Final training accuracy from initial training
        valid_acc: Final validation accuracy from initial training
        train_fine_tuning_acc: Final training accuracy from fine-tuning
        valid_finetuning_acc: Final validation accuracy from fine-tuning
        output_dir: Directory to save the JSON file
        model_params: Number of parameters in the model
        training_start_time: Datetime when training started
        training_end_time: Datetime when training ended
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dictionary
    results = {
        "dataset": dataset_name,
        "modelo": model_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "fine_tuning_epochs": fine_tuning_epochs,
        "train_acc": float(train_acc),
        "valid_acc": float(valid_acc),
        "train_fine_tuning_acc": float(train_fine_tuning_acc),
        "valid_finetuning_acc": float(valid_finetuning_acc),
        "data_rodada": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add new fields if provided
    if model_params is not None:
        results["model_params"] = model_params
    if training_start_time is not None:
        results["training_start_time"] = training_start_time.strftime("%Y-%m-%d %H:%M:%S")
    if training_end_time is not None:
        results["training_end_time"] = training_end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate filename with pattern {model_name}_{date}.json
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{model_name}_{date_str}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Training results saved to: {filepath}")
    
    return filepath


def convert_model_to_tflite(model, output_path):
    """
    Convert a Keras model to TensorFlow Lite format.
    
    Args:
        model: Trained Keras model
        output_path: Path where to save the .tflite file
        
    Returns:
        str: Path to the saved .tflite file
    """
    try:
        # Convert the model to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ TensorFlow Lite model saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"✗ Error converting model to TensorFlow Lite: {str(e)}")
        return None


def extract_final_metrics(history_initial, history_fine_tune):
    """
    Extract final metrics from training histories.
    
    Args:
        history_initial: Keras History object from initial training
        history_fine_tune: Keras History object from fine-tuning
        
    Returns:
        tuple: (train_acc, valid_acc, train_fine_tuning_acc, valid_finetuning_acc)
    """
    
    # Extract final metrics from initial training
    train_acc = history_initial.history['accuracy'][-1]
    valid_acc = history_initial.history['val_accuracy'][-1]
    
    # Extract final metrics from fine-tuning - handle cases where fine-tuning didn't complete
    if (history_fine_tune and 
        hasattr(history_fine_tune, 'history') and 
        'accuracy' in history_fine_tune.history and 
        len(history_fine_tune.history['accuracy']) > 0):
        train_fine_tuning_acc = history_fine_tune.history['accuracy'][-1]
        valid_finetuning_acc = history_fine_tune.history['val_accuracy'][-1]
    else:
        # If fine-tuning didn't complete or was skipped, use initial training values
        print("Warning: Fine-tuning history not available or incomplete. Using initial training metrics.")
        train_fine_tuning_acc = train_acc
        valid_finetuning_acc = valid_acc
    
    return train_acc, valid_acc, train_fine_tuning_acc, valid_finetuning_acc 
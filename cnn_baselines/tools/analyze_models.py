import os
import pandas as pd
from pathlib import Path

def get_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def analyze_models(base_dirs):
    """Analyze model files in the specified directories"""
    # Dictionary to store results with model as key
    model_data = {}
    
    for base_dir in base_dirs:
        base_dir_path = Path(base_dir)
        base_dir_name = base_dir_path.name
        
        # Iterate through model subdirectories
        for model_dir in [d for d in base_dir_path.iterdir() if d.is_dir() and not d.name.startswith('__')]:
            model_name = model_dir.name
            key = (base_dir_name, model_name)
            
            if key not in model_data:
                model_data[key] = {
                    'source_folder': base_dir_name,
                    'model_name': model_name,
                    'size_h5': None,
                    'size_tflite': None
                }
            
            # Find all .h5 and .tflite files
            h5_files = list(model_dir.glob('*.h5'))
            tflite_files = list(model_dir.glob('*.tflite'))
            
            # Get size of .h5 file if it exists
            if h5_files:
                model_data[key]['size_h5'] = round(get_size_mb(h5_files[0]), 2)
                model_data[key]['h5_filename'] = h5_files[0].name
            
            # Get size of .tflite file if it exists
            if tflite_files:
                model_data[key]['size_tflite'] = round(get_size_mb(tflite_files[0]), 2)
                model_data[key]['tflite_filename'] = tflite_files[0].name
    
    # Convert dictionary to list for DataFrame
    results = list(model_data.values())
    return results

def main():
    # Define base directories to analyze
    # Diretórios com os modelos treinados (passe outros como argumentos, se necessário)
    import sys
    base_dirs = sys.argv[1:] or [
        "model_results_colab/models_colab_dataset_wagner_v4_balanceado_1000",
        "model_results_colab/models_dataset_wagner_v4_balanceado",
    ]
    
    # Analyze models
    results = analyze_models(base_dirs)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    columns = ['source_folder', 'model_name', 'h5_filename', 'size_h5', 'tflite_filename', 'size_tflite']
    df = df[columns]
    
    # Save results to CSV
    output_file = "model_analysis_report.csv"
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"Analysis complete. Found {len(results)} model entries.")
    print(f"Results saved to {output_file}")
    
    # Display the first few rows
    if results:
        print("\nSample of results:")
        print(df.head())

if __name__ == "__main__":
    main() 
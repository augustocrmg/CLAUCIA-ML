#!/usr/bin/env python3
"""
Script to check the status of all model scripts and their outputs
This utility helps verify which models are ready to run, which have been trained,
and what outputs already exist.
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path

def check_model_directory(model_path):
    """Check the contents and status of a model directory"""
    model_name = os.path.basename(model_path)
    
    # Find training script
    training_script = None
    for file in os.listdir(model_path):
        if file.endswith("_mlflow.py"):
            training_script = file
            break
    
    # Check for output files
    outputs = {
        'model_file': any(f.endswith('.h5') for f in os.listdir(model_path)),
        'confusion_matrix': any('confusion_matrix' in f for f in os.listdir(model_path)),
        'training_history': any('history' in f for f in os.listdir(model_path)),
        'sample_images': any('sample_images' in f for f in os.listdir(model_path)),
        'classification_report': any('classification_report' in f for f in os.listdir(model_path)),
        'prediction_results': any('prediction_results' in f for f in os.listdir(model_path))
    }
    
    # Count total files
    total_files = len(os.listdir(model_path))
    
    # Estimate if model has been trained (has multiple output files)
    trained_outputs = sum(outputs.values())
    likely_trained = trained_outputs >= 3  # Has at least 3 types of outputs
    
    # Get file sizes for model files
    model_files = [f for f in os.listdir(model_path) if f.endswith('.h5')]
    model_file_sizes = {}
    for mf in model_files:
        size = os.path.getsize(os.path.join(model_path, mf))
        model_file_sizes[mf] = size / (1024 * 1024)  # Size in MB
    
    return {
        'model_name': model_name,
        'path': model_path,
        'training_script': training_script,
        'script_exists': training_script is not None,
        'total_files': total_files,
        'outputs': outputs,
        'trained_outputs_count': trained_outputs,
        'likely_trained': likely_trained,
        'model_file_sizes_mb': model_file_sizes
    }

def print_model_status_table(models_status):
    """Print a formatted table showing model status"""
    print("\n" + "="*120)
    print("MODEL STATUS OVERVIEW")
    print("="*120)
    
    # Header
    print(f"{'Model Name':<25} {'Script':<8} {'Files':<6} {'Trained':<8} {'Model Size':<12} {'Outputs':<40}")
    print("-" * 120)
    
    # Sort models by name
    models_status.sort(key=lambda x: x['model_name'])
    
    for model in models_status:
        script_status = "✅ YES" if model['script_exists'] else "❌ NO"
        trained_status = "✅ YES" if model['likely_trained'] else "❌ NO"
        
        # Model file size info
        if model['model_file_sizes_mb']:
            sizes = [f"{name}: {size:.1f}MB" for name, size in model['model_file_sizes_mb'].items()]
            size_info = ", ".join(sizes)[:12]
        else:
            size_info = "None"
        
        # Output summary
        output_symbols = []
        output_symbols.append("📊" if model['outputs']['confusion_matrix'] else "❌")
        output_symbols.append("📈" if model['outputs']['training_history'] else "❌") 
        output_symbols.append("🤖" if model['outputs']['model_file'] else "❌")
        output_symbols.append("🖼️" if model['outputs']['sample_images'] else "❌")
        output_symbols.append("📋" if model['outputs']['classification_report'] else "❌")
        output_symbols.append("🎯" if model['outputs']['prediction_results'] else "❌")
        
        outputs_str = "".join(output_symbols)
        
        print(f"{model['model_name']:<25} {script_status:<8} {model['total_files']:<6} {trained_status:<8} {size_info:<12} {outputs_str:<40}")

def print_detailed_analysis(models_status):
    """Print detailed analysis of model status"""
    total_models = len(models_status)
    models_with_scripts = len([m for m in models_status if m['script_exists']])
    likely_trained = len([m for m in models_status if m['likely_trained']])
    ready_to_run = len([m for m in models_status if m['script_exists'] and not m['likely_trained']])
    
    print(f"\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    print(f"📊 Total models found: {total_models}")
    print(f"📝 Models with training scripts: {models_with_scripts}")
    print(f"✅ Models likely already trained: {likely_trained}")
    print(f"🚀 Models ready to run: {ready_to_run}")
    print(f"❌ Models missing scripts: {total_models - models_with_scripts}")
    
    if ready_to_run > 0:
        print(f"\n🚀 MODELS READY TO RUN:")
        for model in models_status:
            if model['script_exists'] and not model['likely_trained']:
                print(f"  - {model['model_name']}")
    
    if likely_trained > 0:
        print(f"\n✅ MODELS ALREADY TRAINED:")
        for model in models_status:
            if model['likely_trained']:
                sizes = list(model['model_file_sizes_mb'].values())
                avg_size = sum(sizes) / len(sizes) if sizes else 0
                print(f"  - {model['model_name']} (avg model size: {avg_size:.1f}MB)")
    
    missing_scripts = [m for m in models_status if not m['script_exists']]
    if missing_scripts:
        print(f"\n❌ MODELS MISSING TRAINING SCRIPTS:")
        for model in missing_scripts:
            print(f"  - {model['model_name']}")

def print_output_legend():
    """Print legend for output symbols"""
    print(f"\n" + "="*60)
    print("OUTPUT LEGEND")
    print("="*60)
    print("📊 Confusion Matrix")
    print("📈 Training History")
    print("🤖 Model File (.h5)")
    print("🖼️ Sample Images")
    print("📋 Classification Report")
    print("🎯 Prediction Results")
    print("\n✅ = Present, ❌ = Missing")

def save_status_report(models_status):
    """Save detailed status report to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"models_status_report_{timestamp}.json"
    
    # Create summary statistics
    total_models = len(models_status)
    models_with_scripts = len([m for m in models_status if m['script_exists']])
    likely_trained = len([m for m in models_status if m['likely_trained']])
    ready_to_run = len([m for m in models_status if m['script_exists'] and not m['likely_trained']])
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_models': total_models,
            'models_with_scripts': models_with_scripts,
            'likely_trained': likely_trained,
            'ready_to_run': ready_to_run,
            'missing_scripts': total_models - models_with_scripts
        },
        'models_ready_to_run': [
            m['model_name'] for m in models_status 
            if m['script_exists'] and not m['likely_trained']
        ],
        'models_already_trained': [
            m['model_name'] for m in models_status 
            if m['likely_trained']
        ],
        'models_missing_scripts': [
            m['model_name'] for m in models_status 
            if not m['script_exists']
        ],
        'detailed_status': models_status
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    return report_file

def check_execution_logs():
    """Check for existing execution logs"""
    logs_dir = "execution_logs"
    if not os.path.exists(logs_dir):
        print(f"\n📁 No execution logs directory found ({logs_dir})")
        return
    
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    summary_files = glob.glob(os.path.join(logs_dir, "*.json"))
    
    print(f"\n" + "="*60)
    print("EXECUTION LOGS FOUND")
    print("="*60)
    print(f"📝 Log files: {len(log_files)}")
    print(f"📊 Summary files: {len(summary_files)}")
    
    if log_files:
        print("\nRecent log files:")
        # Sort by modification time, newest first
        log_files.sort(key=os.path.getmtime, reverse=True)
        for log_file in log_files[:5]:  # Show last 5
            mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            size = os.path.getsize(log_file) / 1024  # Size in KB
            print(f"  - {os.path.basename(log_file)} ({mtime.strftime('%Y-%m-%d %H:%M')} - {size:.1f}KB)")
    
    if summary_files:
        print("\nRecent summary files:")
        summary_files.sort(key=os.path.getmtime, reverse=True)
        for summary_file in summary_files[:3]:  # Show last 3
            mtime = datetime.fromtimestamp(os.path.getmtime(summary_file))
            print(f"  - {os.path.basename(summary_file)} ({mtime.strftime('%Y-%m-%d %H:%M')})")

def main():
    """Main function to check all model statuses"""
    print("🔍 Checking status of all models in the models directory...")
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found!")
        return
    
    models_status = []
    
    # Check each subdirectory in models
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            status = check_model_directory(item_path)
            models_status.append(status)
    
    if not models_status:
        print("❌ No model directories found!")
        return
    
    # Print results
    print_model_status_table(models_status)
    print_output_legend()
    print_detailed_analysis(models_status)
    
    # Check for execution logs
    check_execution_logs()
    
    # Save report
    report_file = save_status_report(models_status)
    
    print(f"\n" + "="*60)
    print("STATUS CHECK COMPLETE")
    print("="*60)
    print(f"✅ Checked {len(models_status)} model directories")
    print(f"📄 Detailed report: {report_file}")
    
    # Recommendations
    ready_models = [m for m in models_status if m['script_exists'] and not m['likely_trained']]
    if ready_models:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  - Run training for {len(ready_models)} ready models")
        print(f"  - Use: python run_selected_models.py")
        print(f"  - Or: python run_all_models.py")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Script to run selected model training scripts
This script allows you to choose which specific models to train
instead of running all models sequentially.
"""

import os
import sys
import subprocess
import time
import json
import logging
from datetime import datetime

# Setup logging
def setup_logging():
    """Setup comprehensive logging for the execution"""
    log_dir = "execution_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"selected_models_execution_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def find_model_scripts():
    """Find all model training scripts in the models directory"""
    models_dir = "models"
    model_scripts = []
    
    if not os.path.exists(models_dir):
        logging.error(f"Models directory '{models_dir}' not found!")
        return []
    
    # List all subdirectories in models
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        
        if os.path.isdir(item_path):
            # Look for Python scripts in each model directory
            for file in os.listdir(item_path):
                if file.endswith("_mlflow.py"):
                    script_path = os.path.join(item_path, file)
                    model_scripts.append({
                        'model_name': item,
                        'script_path': script_path,
                        'script_name': file
                    })
                    break  # Take the first *_mlflow.py file found
    
    # Sort by model name for consistent display
    model_scripts.sort(key=lambda x: x['model_name'])
    
    return model_scripts

def select_models_interactively(model_scripts):
    """Allow user to interactively select which models to run"""
    print("\n📋 Available models:")
    print("=" * 50)
    
    for i, script in enumerate(model_scripts, 1):
        print(f"{i:2d}. {script['model_name']}")
    
    print("\n🎯 Model selection options:")
    print("  - Enter specific numbers (e.g., 1,3,5)")
    print("  - Enter ranges (e.g., 1-5)")
    print("  - Enter 'all' to select all models")
    print("  - Enter 'efficient' to select only EfficientNet models")
    print("  - Enter 'small' to select smaller models (B0, B1, B2)")
    print("  - Enter 'large' to select larger models (B5, B6, B7)")
    
    while True:
        try:
            selection = input(f"\nEnter your selection: ").strip()
            
            if selection.lower() == 'all':
                return model_scripts
            
            elif selection.lower() == 'efficient':
                return [s for s in model_scripts if 'efficientnet' in s['model_name'].lower()]
            
            elif selection.lower() == 'small':
                return [s for s in model_scripts if any(x in s['model_name'].lower() for x in ['b0', 'b1', 'b2', 'mobilenet'])]
            
            elif selection.lower() == 'large':
                return [s for s in model_scripts if any(x in s['model_name'].lower() for x in ['b5', 'b6', 'b7'])]
            
            else:
                # Parse numbers and ranges
                selected_indices = set()
                
                for part in selection.split(','):
                    part = part.strip()
                    
                    if '-' in part:
                        # Handle range
                        start, end = map(int, part.split('-'))
                        selected_indices.update(range(start, end + 1))
                    else:
                        # Handle single number
                        selected_indices.add(int(part))
                
                # Validate indices
                valid_indices = set(range(1, len(model_scripts) + 1))
                invalid_indices = selected_indices - valid_indices
                
                if invalid_indices:
                    print(f"❌ Invalid indices: {sorted(invalid_indices)}")
                    print(f"   Valid range: 1-{len(model_scripts)}")
                    continue
                
                # Return selected models
                selected_models = [model_scripts[i-1] for i in sorted(selected_indices)]
                
                print(f"\n✅ Selected {len(selected_models)} models:")
                for model in selected_models:
                    print(f"  - {model['model_name']}")
                
                return selected_models
                
        except ValueError:
            print("❌ Invalid input format. Please try again.")
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")

def run_model_script(model_info):
    """Run a single model training script"""
    model_name = model_info['model_name']
    script_path = model_info['script_path']
    
    logging.info(f"="*80)
    logging.info(f"STARTING MODEL: {model_name}")
    logging.info(f"Script: {script_path}")
    logging.info(f"="*80)
    
    start_time = time.time()
    
    try:
        # Change to the model's directory to ensure relative paths work correctly
        model_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours timeout
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log results
        if result.returncode == 0:
            logging.info(f"✅ SUCCESS: {model_name} completed successfully")
            logging.info(f"Execution time: {execution_time/60:.2f} minutes")
            status = "SUCCESS"
            error_msg = None
        else:
            logging.error(f"❌ FAILED: {model_name} failed with return code {result.returncode}")
            logging.error(f"STDERR: {result.stderr}")
            status = "FAILED"
            error_msg = result.stderr
        
        # Always log some output for debugging
        if result.stdout:
            # Log last 10 lines of stdout
            stdout_lines = result.stdout.strip().split('\n')
            logging.info(f"Last output lines from {model_name}:")
            for line in stdout_lines[-10:]:
                logging.info(f"  {line}")
        
        return {
            'model_name': model_name,
            'status': status,
            'execution_time': execution_time,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'error_message': error_msg,
            'stdout_lines': len(result.stdout.split('\n')) if result.stdout else 0,
            'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0
        }
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"⏰ TIMEOUT: {model_name} exceeded 2 hour limit")
        
        return {
            'model_name': model_name,
            'status': 'TIMEOUT',
            'execution_time': execution_time,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'error_message': 'Script execution timed out after 2 hours',
            'stdout_lines': 0,
            'stderr_lines': 0
        }
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"💥 ERROR: {model_name} encountered an exception: {str(e)}")
        
        return {
            'model_name': model_name,
            'status': 'ERROR',
            'execution_time': execution_time,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'error_message': str(e),
            'stdout_lines': 0,
            'stderr_lines': 0
        }

def save_execution_summary(results, log_file, selected_models):
    """Save execution summary to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"execution_logs/selected_execution_summary_{timestamp}.json"
    
    # Calculate overall statistics
    total_models = len(results)
    successful_models = len([r for r in results if r['status'] == 'SUCCESS'])
    failed_models = len([r for r in results if r['status'] == 'FAILED'])
    timeout_models = len([r for r in results if r['status'] == 'TIMEOUT'])
    error_models = len([r for r in results if r['status'] == 'ERROR'])
    
    total_time = sum(r['execution_time'] for r in results)
    
    summary = {
        'execution_info': {
            'timestamp': datetime.now().isoformat(),
            'log_file': log_file,
            'execution_type': 'selected_models',
            'selected_models': [m['model_name'] for m in selected_models],
            'total_execution_time_minutes': total_time / 60,
            'total_execution_time_hours': total_time / 3600
        },
        'overall_stats': {
            'total_models': total_models,
            'successful_models': successful_models,
            'failed_models': failed_models,
            'timeout_models': timeout_models,
            'error_models': error_models,
            'success_rate': (successful_models / total_models * 100) if total_models > 0 else 0
        },
        'model_results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Execution summary saved to: {summary_file}")
    return summary_file

def print_final_summary(results):
    """Print a comprehensive final summary"""
    total_models = len(results)
    successful_models = len([r for r in results if r['status'] == 'SUCCESS'])
    failed_models = len([r for r in results if r['status'] == 'FAILED'])
    timeout_models = len([r for r in results if r['status'] == 'TIMEOUT'])
    error_models = len([r for r in results if r['status'] == 'ERROR'])
    
    total_time = sum(r['execution_time'] for r in results)
    
    logging.info("\n" + "="*80)
    logging.info("FINAL EXECUTION SUMMARY")
    logging.info("="*80)
    logging.info(f"Total models executed: {total_models}")
    logging.info(f"✅ Successful: {successful_models}")
    logging.info(f"❌ Failed: {failed_models}")
    logging.info(f"⏰ Timeout: {timeout_models}")  
    logging.info(f"💥 Error: {error_models}")
    logging.info(f"Success rate: {(successful_models/total_models*100):.1f}%")
    logging.info(f"Total execution time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    
    if successful_models > 0:
        logging.info("\n✅ SUCCESSFUL MODELS:")
        for result in results:
            if result['status'] == 'SUCCESS':
                logging.info(f"  - {result['model_name']}: {result['execution_time']/60:.1f} minutes")
    
    if failed_models > 0 or timeout_models > 0 or error_models > 0:
        logging.info("\n❌ FAILED/PROBLEM MODELS:")
        for result in results:
            if result['status'] != 'SUCCESS':
                logging.info(f"  - {result['model_name']}: {result['status']} ({result['execution_time']/60:.1f} minutes)")
                if result['error_message']:
                    # Show first line of error message
                    error_preview = result['error_message'].split('\n')[0][:100]
                    logging.info(f"    Error: {error_preview}...")

def main():
    """Main execution function"""
    print("🎯 Starting selective execution of model training scripts")
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting selective model training execution")
    logging.info(f"Log file: {log_file}")
    
    # Find all model scripts
    model_scripts = find_model_scripts()
    
    if not model_scripts:
        logging.error("No model scripts found! Please check the models directory structure.")
        return
    
    logging.info(f"Found {len(model_scripts)} available model scripts")
    
    # Let user select models
    selected_models = select_models_interactively(model_scripts)
    
    if not selected_models:
        print("No models selected. Exiting.")
        return
    
    # Final confirmation
    estimated_time = len(selected_models) * 1.5  # Rough estimate of 1.5 hours per model
    print(f"\n⏱️  Estimated total time: {estimated_time:.1f} hours")
    
    response = input(f"Do you want to proceed with training {len(selected_models)} selected models? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Execution cancelled by user.")
        return
    
    # Execute selected models
    results = []
    overall_start_time = time.time()
    
    for i, model_info in enumerate(selected_models, 1):
        logging.info(f"\n📊 Progress: {i}/{len(selected_models)} models")
        logging.info(f"Remaining models: {len(selected_models) - i}")
        logging.info(f"Current model: {model_info['model_name']}")
        
        result = run_model_script(model_info)
        results.append(result)
        
        # Log intermediate progress
        completed_successful = len([r for r in results if r['status'] == 'SUCCESS'])
        logging.info(f"Progress update: {completed_successful}/{i} models successful so far")
        
        # Estimate remaining time
        if i > 0:
            avg_time_per_model = sum(r['execution_time'] for r in results) / i
            estimated_remaining = avg_time_per_model * (len(selected_models) - i) / 3600
            logging.info(f"Estimated remaining time: {estimated_remaining:.1f} hours")
    
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    logging.info(f"\n🏁 All selected models completed! Total time: {overall_time/3600:.2f} hours")
    
    # Save summary and print final results
    summary_file = save_execution_summary(results, log_file, selected_models)
    print_final_summary(results)
    
    logging.info(f"\nDetailed logs available in: {log_file}")
    logging.info(f"Execution summary available in: {summary_file}")
    
    print(f"\n🎉 Selected models execution completed!")
    print(f"📋 Check the summary file: {summary_file}")
    print(f"📝 Check the detailed logs: {log_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user (Ctrl+C)")
        logging.error("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
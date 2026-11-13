#!/usr/bin/env python3
"""
Script to run all model training scripts sequentially
This script will execute each model training script in the models directory
and provide comprehensive logging and progress tracking.

Usage:
    python run_all_models.py              # Normal mode with live output
    python run_all_models.py --quiet      # Quiet mode with minimal output
    python run_all_models.py --verbose    # Extra verbose mode
"""

import os
import sys
import subprocess
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Global verbosity settings
QUIET_MODE = False
VERBOSE_MODE = False

# Setup logging
def setup_logging():
    """Setup comprehensive logging for the execution"""
    log_dir = "execution_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model_execution_{timestamp}.log")
    
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
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
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
    
    # Sort by model name for consistent execution order
    model_scripts.sort(key=lambda x: x['model_name'])
    
    return model_scripts

def run_model_script(model_info):
    """Run a single model training script with real-time output display"""
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
        
        # Create the process with real-time output
        process = subprocess.Popen(
            [sys.executable, script_name],
            cwd=model_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Store output for later analysis
        all_output = []
        
        # Print a header for this model's output
        if not QUIET_MODE:
            print(f"\n🔄 LIVE OUTPUT FROM {model_name.upper()}:")
            print("─" * 80)
        
        # Read and display output in real-time
        try:
            while True:
                # Use timeout to check if process is still running
                try:
                    line = process.stdout.readline()
                    if not line:
                        break
                    
                    # Remove trailing newline to avoid double spacing
                    line = line.rstrip('\n')
                    
                    # Store for logging
                    all_output.append(line)
                    
                    # Display based on verbosity mode
                    if not QUIET_MODE:
                        # Display with timestamp and model prefix
                        timestamp = time.strftime("%H:%M:%S")
                        if VERBOSE_MODE:
                            print(f"[{timestamp}] {model_name}: {line}")
                        else:
                            # In normal mode, show output but less cluttered
                            if any(keyword in line.lower() for keyword in ['epoch', 'loss', 'accuracy', 'val_', 'error', 'warning', 'completed', 'starting']):
                                print(f"[{timestamp}] {model_name}: {line}")
                    
                    # Always log to file
                    logging.info(f"[{model_name}] {line}")
                    
                except Exception as e:
                    logging.warning(f"Error reading output from {model_name}: {e}")
                    break
            
            # Wait for process to complete with timeout
            try:
                return_code = process.wait(timeout=7200)  # 2 hours timeout
            except subprocess.TimeoutExpired:
                logging.error(f"⏰ TIMEOUT: {model_name} exceeded 2 hour limit")
                process.kill()
                process.wait()
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                print(f"\n⏰ {model_name} TIMED OUT after {execution_time/60:.1f} minutes")
                if not QUIET_MODE:
                    print("─" * 80)
                
                return {
                    'model_name': model_name,
                    'status': 'TIMEOUT',
                    'execution_time': execution_time,
                    'start_time': datetime.fromtimestamp(start_time).isoformat(),
                    'end_time': datetime.fromtimestamp(end_time).isoformat(),
                    'error_message': 'Script execution timed out after 2 hours',
                    'stdout_lines': len(all_output),
                    'stderr_lines': 0
                }
            
        except Exception as e:
            logging.error(f"💥 ERROR reading output from {model_name}: {str(e)}")
            process.kill()
            process.wait()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"\n💥 {model_name} ENCOUNTERED ERROR: {str(e)}")
            if not QUIET_MODE:
                print("─" * 80)
            
            return {
                'model_name': model_name,
                'status': 'ERROR',
                'execution_time': execution_time,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'error_message': str(e),
                'stdout_lines': len(all_output),
                'stderr_lines': 0
            }
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log results
        if return_code == 0:
            logging.info(f"✅ SUCCESS: {model_name} completed successfully")
            logging.info(f"Execution time: {execution_time/60:.2f} minutes")
            status = "SUCCESS"
            error_msg = None
            
            print(f"\n✅ {model_name} COMPLETED SUCCESSFULLY in {execution_time/60:.1f} minutes")
            if not QUIET_MODE:
                print("─" * 80)
        else:
            logging.error(f"❌ FAILED: {model_name} failed with return code {return_code}")
            status = "FAILED"
            error_msg = f"Process exited with return code {return_code}"
            
            print(f"\n❌ {model_name} FAILED with return code {return_code}")
            if not QUIET_MODE:
                print("─" * 80)
        
        # Log output summary
        logging.info(f"Total output lines from {model_name}: {len(all_output)}")
        
        return {
            'model_name': model_name,
            'status': status,
            'execution_time': execution_time,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'error_message': error_msg,
            'stdout_lines': len(all_output),
            'stderr_lines': 0
        }
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.error(f"💥 ERROR: {model_name} encountered an exception: {str(e)}")
        
        print(f"\n💥 {model_name} ENCOUNTERED EXCEPTION: {str(e)}")
        if not QUIET_MODE:
            print("─" * 80)
        
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

def save_execution_summary(results, log_file):
    """Save execution summary to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"execution_logs/execution_summary_{timestamp}.json"
    
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Execute all model training scripts sequentially')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Quiet mode - minimal output, only show progress and results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose mode - show extra debugging information')
    args = parser.parse_args()
    
    # Set global verbosity level
    global QUIET_MODE, VERBOSE_MODE
    QUIET_MODE = args.quiet
    VERBOSE_MODE = args.verbose
    
    if QUIET_MODE:
        print("🔇 Running in QUIET mode - minimal output")
    elif VERBOSE_MODE:
        print("🔊 Running in VERBOSE mode - extra debugging info")
    else:
        print("🔄 Running in NORMAL mode - live output display")
    
    print("🚀 Starting sequential execution of all model training scripts")
    print("This may take several hours to complete...")
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting model training execution batch")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Mode: {'QUIET' if QUIET_MODE else 'VERBOSE' if VERBOSE_MODE else 'NORMAL'}")
    
    # Find all model scripts
    model_scripts = find_model_scripts()
    
    if not model_scripts:
        logging.error("No model scripts found! Please check the models directory structure.")
        return
    
    logging.info(f"Found {len(model_scripts)} model scripts to execute:")
    for script in model_scripts:
        logging.info(f"  - {script['model_name']}: {script['script_path']}")
    
    # Ask for confirmation
    print(f"\nFound {len(model_scripts)} models to train:")
    for script in model_scripts:
        print(f"  - {script['model_name']}")
    
    response = input(f"\nDo you want to proceed with training all {len(model_scripts)} models? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Execution cancelled by user.")
        return
    
    # Execute all models
    results = []
    overall_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"🚀 STARTING BATCH EXECUTION OF {len(model_scripts)} MODELS")
    print(f"{'='*80}")
    
    for i, model_info in enumerate(model_scripts, 1):
        # Calculate estimated time remaining
        if i > 1:
            elapsed_time = time.time() - overall_start_time
            avg_time_per_model = elapsed_time / (i - 1)
            estimated_remaining = avg_time_per_model * (len(model_scripts) - i + 1)
            
            print(f"\n{'='*80}")
            print(f"📊 PROGRESS: {i}/{len(model_scripts)} models")
            print(f"⏱️  Elapsed: {elapsed_time/3600:.1f}h | Est. remaining: {estimated_remaining/3600:.1f}h")
            print(f"📈 Success rate so far: {len([r for r in results if r['status'] == 'SUCCESS'])}/{i-1}")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"📊 PROGRESS: {i}/{len(model_scripts)} models - STARTING FIRST MODEL")
            print(f"{'='*80}")
        
        logging.info(f"\n📊 Progress: {i}/{len(model_scripts)} models")
        logging.info(f"Remaining models: {len(model_scripts) - i}")
        
        result = run_model_script(model_info)
        results.append(result)
        
        # Show immediate result summary
        completed_successful = len([r for r in results if r['status'] == 'SUCCESS'])
        completed_failed = len([r for r in results if r['status'] != 'SUCCESS'])
        
        print(f"\n📋 BATCH PROGRESS UPDATE:")
        print(f"   Completed: {i}/{len(model_scripts)} models")
        print(f"   ✅ Successful: {completed_successful}")
        print(f"   ❌ Failed/Error: {completed_failed}")
        print(f"   📊 Success rate: {(completed_successful/i*100):.1f}%")
        
        logging.info(f"Progress update: {completed_successful}/{i} models successful so far")
        
        # Add a pause to let user see the summary
        if i < len(model_scripts):
            print(f"\n⏭️  Proceeding to next model in 3 seconds...")
            time.sleep(3)
    
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    logging.info(f"\n🏁 All models completed! Total time: {overall_time/3600:.2f} hours")
    
    # Save summary and print final results
    summary_file = save_execution_summary(results, log_file)
    print_final_summary(results)
    
    logging.info(f"\nDetailed logs available in: {log_file}")
    logging.info(f"Execution summary available in: {summary_file}")
    
    print(f"\n🎉 Batch execution completed!")
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
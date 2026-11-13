#!/usr/bin/env python3
"""
EfficientNet-B7 Training Script - Optimized for Limited GPU Memory
Specifically optimized for NVIDIA RTX 2070 SUPER (8GB) and similar GPUs.

This script implements several optimizations:
1. Reduced batch size and image resolution
2. Mixed precision training (FP16)
3. XLA compilation
4. Memory growth enabled
5. Optional gradient accumulation

Usage:
    python run_optimized.py
    
For even more memory-constrained environments, set:
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
"""

import os
import sys
import subprocess

def set_memory_optimizations():
    """Set environment variables for memory optimization"""
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    print("Memory optimization environment variables set:")
    print(f"  TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')}")
    print(f"  XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
    print(f"  TF_GPU_ALLOCATOR: {os.environ.get('TF_GPU_ALLOCATOR')}")

def main():
    print("="*60)
    print("EfficientNet-B7 Training - Memory Optimized")
    print("="*60)
    
    # Set optimization flags
    set_memory_optimizations()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, 'efficientnet_b7_600_mlflow.py')
    
    # Check if main script exists
    if not os.path.exists(main_script):
        print(f"ERROR: Main script not found at {main_script}")
        return 1
    
    print(f"\nStarting training with optimizations...")
    print(f"Script: {main_script}")
    print("Configuration:")
    print("  - Image size: 380x380 pixels")
    print("  - Batch size: 8")
    print("  - Mixed precision: Enabled")
    print("  - XLA compilation: Enabled")
    print("  - Memory growth: Enabled")
    
    try:
        # Run the main script
        result = subprocess.run([sys.executable, main_script], 
                              cwd=script_dir,
                              env=os.environ.copy())
        return result.returncode
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running training script: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
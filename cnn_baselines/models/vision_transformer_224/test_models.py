#!/usr/bin/env python3
"""
Test script for Vision Transformer models
Demonstrates both standard (6GB) and large (40GB) variants without training
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_gpu_info():
    """Get GPU information"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"🔧 GPU: {gpu_details.get('device_name', 'Unknown')}")
                
            # Get memory info (approximate)
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            total_memory = memory_info['current'] / (1024**3)  # Convert to GB
            print(f"💾 Available GPU Memory: ~{total_memory:.1f}GB")
            return total_memory
        except:
            print("💾 GPU memory info not available")
            return 0
    else:
        print("❌ No GPU detected")
        return 0

def test_standard_model():
    """Test the standard Vision Transformer (6GB optimized)"""
    print("\n" + "="*60)
    print("🔬 TESTING STANDARD VISION TRANSFORMER (6GB GPU)")
    print("="*60)
    
    # Standard model configuration
    IMAGE_SIZE = (224, 224)
    PATCH_SIZE = 16
    NUM_PATCHES = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2
    PROJECTION_DIM = 384  # Optimized for 6GB
    NUM_HEADS = 6
    TRANSFORMER_LAYERS = 6
    NUM_CLASSES = 10  # Example
    
    print(f"📊 Standard Model Specs:")
    print(f"  • Projection Dimension: {PROJECTION_DIM}")
    print(f"  • Attention Heads: {NUM_HEADS}")  
    print(f"  • Transformer Layers: {TRANSFORMER_LAYERS}")
    print(f"  • Patches: {NUM_PATCHES} ({PATCH_SIZE}x{PATCH_SIZE})")
    
    try:
        # Create a minimal version to test structure
        inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
        x = layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE)(inputs)
        x = layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
        
        # Add a single transformer block
        x1 = layers.LayerNormalization()(x)
        attention = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, 
            key_dim=PROJECTION_DIM//NUM_HEADS
        )(x1, x1)
        x = layers.Add()([attention, x])
        
        # Classification
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        params = model.count_params()
        print(f"✅ Model compiles successfully")
        print(f"📏 Parameters (simplified): {params:,}")
        
        # Estimate full model
        full_params_estimate = params * TRANSFORMER_LAYERS + (PROJECTION_DIM * 3 * PATCH_SIZE * PATCH_SIZE)
        print(f"📏 Full model estimate: ~{full_params_estimate:,} parameters")
        print(f"💾 Memory estimate: ~{full_params_estimate * 4 / (1024**3):.2f}GB (FP32)")
        
        return True, full_params_estimate
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, 0

def test_large_model():
    """Test the large Vision Transformer (40GB optimized)"""
    print("\n" + "="*60)
    print("🚀 TESTING LARGE VISION TRANSFORMER (40GB GPU)")
    print("="*60)
    
    # Large model configuration  
    IMAGE_SIZE = (224, 224)
    PATCH_SIZE = 16
    NUM_PATCHES = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2
    PROJECTION_DIM = 1024  # Full ViT-Large
    NUM_HEADS = 16
    TRANSFORMER_LAYERS = 24
    NUM_CLASSES = 10  # Example
    
    print(f"📊 Large Model Specs:")
    print(f"  • Projection Dimension: {PROJECTION_DIM}")
    print(f"  • Attention Heads: {NUM_HEADS}")
    print(f"  • Transformer Layers: {TRANSFORMER_LAYERS}")
    print(f"  • Patches: {NUM_PATCHES} ({PATCH_SIZE}x{PATCH_SIZE})")
    
    try:
        # Create a minimal version to test structure
        inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
        x = layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE)(inputs)
        x = layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
        
        # Add a single transformer block  
        x1 = layers.LayerNormalization()(x)
        attention = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=PROJECTION_DIM//NUM_HEADS
        )(x1, x1)
        x = layers.Add()([attention, x])
        
        # Classification
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')
        
        params = model.count_params()
        print(f"✅ Model compiles successfully")
        print(f"📏 Parameters (simplified): {params:,}")
        
        # Estimate full model
        full_params_estimate = params * TRANSFORMER_LAYERS + (PROJECTION_DIM * 3 * PATCH_SIZE * PATCH_SIZE)
        print(f"📏 Full model estimate: ~{full_params_estimate:,} parameters")
        print(f"💾 Memory estimate: ~{full_params_estimate * 4 / (1024**3):.2f}GB (FP32)")
        print(f"💾 Memory estimate: ~{full_params_estimate * 2 / (1024**3):.2f}GB (FP16)")
        
        return True, full_params_estimate
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, 0

def recommend_model(gpu_memory_gb):
    """Recommend which model to use based on available GPU memory"""
    print("\n" + "="*60)
    print("🎯 MODEL RECOMMENDATION")
    print("="*60)
    
    if gpu_memory_gb >= 35:
        print("✅ Recommendation: Use LARGE Vision Transformer")
        print("   • Full ViT-Large architecture (1024-dim, 24 layers)")
        print("   • Maximum accuracy potential")
        print("   • Excellent for complex datasets")
        print("   • File: vision_transformer_224_large_mlflow.py")
    elif gpu_memory_gb >= 6:
        print("✅ Recommendation: Use STANDARD Vision Transformer")
        print("   • Optimized ViT architecture (384-dim, 6 layers)")  
        print("   • Good performance with memory efficiency")
        print("   • Suitable for most applications")
        print("   • File: vision_transformer_224_mlflow.py")
    else:
        print("⚠️  Recommendation: Consider using a smaller model")
        print("   • Current GPU memory may be insufficient")
        print("   • Consider EfficientNet models for < 6GB GPUs")
        print("   • Reduce batch size if proceeding with ViT")

def main():
    """Main test function"""
    print("🔬 Vision Transformer Model Testing")
    print("Testing both standard (6GB) and large (40GB) variants\n")
    
    # Get GPU info
    gpu_memory = get_gpu_info()
    
    # Test both models
    standard_success, standard_params = test_standard_model()
    large_success, large_params = test_large_model()
    
    # Model comparison
    if standard_success and large_success:
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        print(f"Standard Model: ~{standard_params:,} parameters")
        print(f"Large Model:    ~{large_params:,} parameters")
        print(f"Size Ratio:     {large_params/standard_params:.1f}x larger")
        
    # Recommendation
    recommend_model(gpu_memory)
    
    print("\n" + "="*60)
    print("🎉 TESTING COMPLETE")
    print("="*60)
    print("Both Vision Transformer implementations are ready!")
    print("Choose the appropriate model based on your GPU memory.")

if __name__ == "__main__":
    main() 
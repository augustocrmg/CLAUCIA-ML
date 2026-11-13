# Vision Transformer 224x224 Large Model

Implementation of **Vision Transformer Large (ViT-Large)** for image classification with 224x224 input size, optimized for **40GB GPU memory**. This is the full-scale version of the Vision Transformer with maximum capacity and performance.

## 🚀 Architecture Specifications

### Model Parameters
- **Input Size**: 224x224x3
- **Patch Size**: 16x16 (resulting in 196 patches)
- **Projection Dimension**: 1024 (full ViT-Large dimension)
- **Number of Attention Heads**: 16 (full ViT-Large heads)
- **Transformer Layers**: 24 (full ViT-Large depth)
- **MLP Expansion**: 4x (4096 hidden units)
- **Classification Head**: [4096, 2048, 1024] units
- **Total Parameters**: ~300M+ (full ViT-Large scale)

### Training Configuration
- **Batch Size**: 64 (optimized for 40GB GPU)
- **Learning Rate**: 1e-4 with warmup and cosine decay
- **Weight Decay**: 0.1 (AdamW optimizer)
- **Dropout Rate**: 0.2
- **Attention Dropout**: 0.1
- **Stochastic Depth**: 0.1 (for regularization)
- **Epochs**: 30 initial training

## 🎯 Advanced Features

### 🔬 Regularization Techniques
✅ **Stochastic Depth**: Progressive layer dropout for better generalization  
✅ **Weight Decay**: L2 regularization with AdamW optimizer  
✅ **Attention Dropout**: Specialized dropout for multi-head attention  
✅ **Layer Dropout**: Multiple dropout layers throughout the network  

### 📈 Training Optimizations
✅ **Learning Rate Warmup**: 5 epochs warmup for stable training  
✅ **Cosine Annealing**: Smooth learning rate decay  
✅ **Early Stopping**: Prevents overfitting with patience  
✅ **Reduce LR on Plateau**: Adaptive learning rate reduction  
✅ **Mixed Precision**: FP16 training for memory efficiency  

### 🎨 Enhanced Data Augmentation
✅ **Geometric Augmentations**: Random flip, rotation (5°), zoom (30%)  
✅ **Color Augmentations**: Random brightness (20%) and contrast (20%)  
✅ **Advanced Normalization**: Layer-wise adaptive normalization  
✅ **Global Average Pooling**: Better feature aggregation  

## 💾 Memory & Performance

### Target Hardware
- **GPU Memory**: 40GB (RTX A6000, RTX 8000, V100, A100)
- **RAM**: 32GB+ recommended
- **VRAM Utilization**: ~35-38GB during training
- **Training Speed**: ~2-3x faster than 6GB version

### Memory Usage Breakdown
```
Model Parameters:     ~300M (~1.2GB)
Activations (FP16):   ~25-30GB
Gradients:           ~1.2GB
Optimizer States:    ~2.4GB
Data Buffers:        ~5-8GB
Total:              ~35-40GB
```

## 📊 Model Comparison

| Specification | Standard ViT | Large ViT (This Model) |
|---------------|--------------|------------------------|
| Parameters | ~103M | ~300M+ |
| Layers | 6 | 24 |
| Attention Heads | 6 | 16 |
| Hidden Dimension | 384 | 1024 |
| MLP Dimension | 768→384 | 4096→1024 |
| Batch Size | 16 | 64 |
| Target GPU | 6GB | 40GB |

## 🛠 Usage

### Basic Training
```bash
cd models/vision_transformer_224
python vision_transformer_224_large_mlflow.py
```

### System Requirements
```bash
# Check GPU memory
nvidia-smi

# Ensure you have at least 40GB GPU memory
# Recommended: RTX A6000, RTX 8000, V100 32GB, A100 40GB
```

## 📈 Expected Performance

### Training Characteristics
- **Convergence**: Faster convergence due to larger capacity
- **Accuracy**: Higher accuracy potential than smaller models
- **Stability**: More stable training with advanced regularization
- **Generalization**: Better feature representation and transfer learning

### Benchmark Expectations
- **Training Time**: 2-4 hours (depending on dataset size)
- **Validation Accuracy**: Expected 5-10% improvement over standard version
- **F1-Score**: Better per-class performance, especially on minority classes
- **Robustness**: More robust to data variations and edge cases

## 🔧 Advanced Configuration

### Custom Hyperparameters
```python
# Modify these in the script for experimentation
PROJECTION_DIM = 1024        # Model width
TRANSFORMER_LAYERS = 24      # Model depth
NUM_HEADS = 16              # Attention heads
BATCH_SIZE = 64             # Batch size
LEARNING_RATE = 1e-4        # Base learning rate
WEIGHT_DECAY = 0.1          # L2 regularization
STOCHASTIC_DEPTH_RATE = 0.1 # Stochastic depth
```

### Memory Optimization (if needed)
```python
# If you encounter OOM errors, try these settings:
BATCH_SIZE = 32              # Reduce batch size
PROJECTION_DIM = 768         # Reduce model width
TRANSFORMER_LAYERS = 18      # Reduce depth slightly
```

## 📋 Features Inherited from Base Model

✅ **MLflow Integration**: Complete experiment tracking  
✅ **Balanced Dataset Splits**: 70% train, 10% validation, 20% test  
✅ **Comprehensive Evaluation**: Confusion matrices, classification reports  
✅ **Multiple Output Formats**: H5 and TensorFlow Lite models  
✅ **Visualization**: Training history, sample predictions  
✅ **Detailed Logging**: Per-class metrics and predictions  
✅ **Integration**: Compatible with existing model_logger system  
✅ **JSON Results**: Training results saved for model comparison  

## 🎯 When to Use This Model

### ✅ Use Large Model When:
- You have 40GB+ GPU memory available
- Maximum accuracy is critical
- Working with complex/diverse datasets
- Need best possible feature representations
- Have sufficient training time and computational resources
- Planning to use for transfer learning or fine-tuning

### ❌ Use Standard Model When:
- Limited GPU memory (< 20GB)
- Quick prototyping or experimentation
- Resource-constrained environments
- Real-time inference requirements
- Simpler datasets with fewer classes

## 🔬 Technical Details

### Architecture Innovations
- **Full ViT-Large Scale**: Matches Google's ViT-Large/16 architecture
- **Optimized for Medical Images**: Specialized for skin condition classification
- **Advanced Regularization**: Multiple techniques to prevent overfitting
- **Efficient Training**: Optimized training loop with modern techniques

### Implementation Highlights
- **Custom Layers**: Patches, PatchEncoder, StochasticDepth
- **Mixed Precision**: Automatic float16/float32 handling
- **Memory Management**: Efficient GPU memory usage
- **Scalable Design**: Easy to modify for different datasets

## 📝 Outputs

### Generated Files
- `vision_transformer_224_large.h5` - Full Keras model
- `vision_transformer_224_large.tflite` - TensorFlow Lite model
- `training_history_large.png` - Training curves
- `confusion_matrix_vision_transformer_large.png` - Confusion matrix
- `classification_report_vision_transformer_large.json` - Detailed metrics
- `prediction_results_large.json` - Sample predictions

### MLflow Artifacts
- Model checkpoints and metadata
- Training visualizations
- Comprehensive metrics logging
- Experiment comparison data

## 🚨 Important Notes

1. **GPU Memory**: Requires minimum 35GB GPU memory for training
2. **Training Time**: Expect significantly longer training times
3. **Data Quality**: Large models are more sensitive to data quality
4. **Regularization**: Built-in regularization helps prevent overfitting
5. **Monitoring**: Watch for memory usage and adjust batch size if needed

---

**This large model represents the state-of-the-art Vision Transformer implementation optimized for high-end GPU hardware, providing maximum capacity and performance for demanding computer vision tasks.** 
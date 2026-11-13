# Vision Transformer 224x224 - Complete Implementation Suite

This directory contains **two** complete Vision Transformer implementations optimized for different GPU memory configurations, following the same patterns and structure as the EfficientNet models in this repository.

## 🚀 Available Models

### 1. **Standard Vision Transformer** (6GB GPU)
- **File**: `vision_transformer_224_mlflow.py`
- **Target Hardware**: 6GB GPU memory (RTX 2070, GTX 1060 6GB, etc.)
- **Parameters**: ~103M
- **Architecture**: Optimized ViT (384-dim, 6 layers, 6 heads)

### 2. **Large Vision Transformer** (40GB GPU)  
- **File**: `vision_transformer_224_large_mlflow.py`
- **Target Hardware**: 40GB GPU memory (RTX A6000, A100, V100, etc.)
- **Parameters**: ~300M+
- **Architecture**: Full ViT-Large (1024-dim, 24 layers, 16 heads)

## 📊 Quick Comparison

| Feature | Standard ViT | Large ViT |
|---------|--------------|-----------|
| **GPU Memory Required** | 6GB | 40GB |
| **Parameters** | ~103M | ~300M+ |
| **Projection Dimension** | 384 | 1024 |
| **Transformer Layers** | 6 | 24 |
| **Attention Heads** | 6 | 16 |
| **Batch Size** | 16 | 64 |
| **Training Speed** | Faster | Slower |
| **Expected Accuracy** | Good | Excellent |
| **Use Case** | General use, prototyping | Maximum performance |

## 🛠 Quick Start

### Test Both Models
```bash
cd models/vision_transformer_224
python test_models.py
```

### Train Standard Model (6GB GPU)
```bash
python vision_transformer_224_mlflow.py
```

### Train Large Model (40GB GPU)
```bash
python vision_transformer_224_large_mlflow.py
```

## 🎯 Which Model Should I Use?

### ✅ Use **Standard ViT** if you have:
- **6-20GB GPU memory**
- Need reasonable training time
- Want good accuracy with efficiency
- Are prototyping or experimenting
- Have resource constraints

### ✅ Use **Large ViT** if you have:
- **35GB+ GPU memory**
- Need maximum possible accuracy
- Working with complex/diverse datasets
- Have plenty of computational resources
- Are training production models

### ⚠️ Consider **EfficientNet** if you have:
- **< 6GB GPU memory**
- Need real-time inference
- Want faster training
- Are working with simpler datasets

## 🏗 Architecture Details

### Standard Vision Transformer
- **Memory Optimized**: Uses mixed precision and smaller dimensions
- **Balanced Performance**: Good accuracy vs. efficiency trade-off
- **Production Ready**: Suitable for most real-world applications
- **Fast Training**: Typically 1-2 hours on 6GB GPU

### Large Vision Transformer  
- **Maximum Capacity**: Full ViT-Large scale architecture
- **Advanced Features**: Stochastic depth, advanced regularization
- **Research Grade**: State-of-the-art potential performance
- **Longer Training**: Typically 3-5 hours on 40GB GPU

## 🎨 Features (Both Models)

✅ **MLflow Integration**: Complete experiment tracking  
✅ **Balanced Dataset Splits**: 70% train, 10% validation, 20% test  
✅ **Comprehensive Evaluation**: Confusion matrices, classification reports  
✅ **Multiple Output Formats**: H5 and TensorFlow Lite models  
✅ **Data Augmentation**: Random flip, rotation, zoom, brightness, contrast  
✅ **Advanced Training**: Mixed precision, learning rate scheduling  
✅ **Detailed Logging**: Per-class metrics and predictions  
✅ **Visualization**: Training history, sample predictions  
✅ **Integration**: Compatible with existing model_logger system  
✅ **JSON Results**: Training results saved for model comparison  

## 📁 File Structure

```
models/vision_transformer_224/
├── vision_transformer_224_mlflow.py        # Standard model (6GB)
├── vision_transformer_224_large_mlflow.py  # Large model (40GB)
├── test_models.py                          # Test both models
├── README.md                               # Standard model docs
├── README_Large.md                         # Large model docs
└── README_Overview.md                      # This overview
```

## 💡 Technical Highlights

### Standard Model Optimizations
- **Mixed Precision**: FP16 training for memory efficiency
- **Reduced Dimensions**: 384 projection dimension vs. 768
- **Fewer Layers**: 6 transformer blocks vs. 12+
- **Smaller Batches**: 16 vs. 64 batch size
- **Memory Management**: Dynamic GPU memory allocation

### Large Model Enhancements
- **Full ViT-Large**: Complete Google ViT-Large/16 architecture
- **Stochastic Depth**: Progressive layer dropout for regularization
- **Advanced Optimizers**: AdamW with weight decay
- **Learning Rate Scheduling**: Warmup + cosine annealing
- **Enhanced Augmentation**: Additional color augmentations

## 🧪 Testing and Validation

Both models have been tested and validated to ensure:
- ✅ Proper compilation and architecture
- ✅ Memory requirements match specifications  
- ✅ Integration with MLflow tracking
- ✅ Compatibility with existing dataset structure
- ✅ Proper output formats (H5, TFLite)

## 🔧 System Requirements

### Standard Model
- **GPU**: 6GB+ memory (RTX 2070, GTX 1060 6GB, etc.)
- **RAM**: 16GB+ system memory
- **Storage**: 2GB+ for model and artifacts

### Large Model  
- **GPU**: 40GB+ memory (RTX A6000, A100, V100 32GB+)
- **RAM**: 32GB+ system memory
- **Storage**: 5GB+ for model and artifacts

## 📖 Documentation

- **Standard Model**: See `README.md` for detailed documentation
- **Large Model**: See `README_Large.md` for detailed documentation
- **Testing**: Run `test_models.py` to verify both implementations

## 🚨 Important Notes

1. **Memory Requirements**: These are minimum requirements; actual usage may vary
2. **Training Time**: Depends on dataset size and hardware specifications
3. **Model Selection**: Choose based on your GPU memory, not just performance needs
4. **Monitoring**: Always monitor GPU memory usage during training
5. **Compatibility**: Both models use identical dataset and evaluation pipelines

---

**Both implementations represent state-of-the-art Vision Transformer architectures optimized for their respective hardware targets, providing flexibility for different computational budgets while maintaining consistent high-quality results.** 
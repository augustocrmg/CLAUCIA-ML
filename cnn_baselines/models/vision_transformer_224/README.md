# Vision Transformer 224x224

Implementation of Vision Transformer (ViT) for image classification with 224x224 input size, following the same patterns and structure as the EfficientNet models in this repository. **Optimized for 6GB GPU memory**.

## Architecture

- **Input Size**: 224x224x3
- **Patch Size**: 16x16 (resulting in 196 patches)
- **Projection Dimension**: 384 (optimized from 768)
- **Number of Heads**: 6 (optimized from 12)
- **Transformer Layers**: 6 (optimized from 12)
- **MLP Head Units**: [1024, 512] (optimized from [2048, 1024])
- **Batch Size**: 16 (optimized for memory efficiency)

## Memory Optimizations

🚀 **Mixed Precision Training**: Uses float16 for faster training and reduced memory usage
🚀 **Reduced Model Size**: Optimized parameters for 6GB GPU compatibility
🚀 **Efficient Batching**: Smaller batch sizes to prevent OOM errors
🚀 **GPU Memory Growth**: Dynamic GPU memory allocation

## Features

✅ **MLflow Integration**: Complete experiment tracking
✅ **Balanced Dataset Splits**: 70% train, 15% validation, 15% test
✅ **Comprehensive Evaluation**: Confusion matrices, classification reports
✅ **Multiple Output Formats**: H5 and TensorFlow Lite models
✅ **Data Augmentation**: Random flip, rotation, and zoom
✅ **Detailed Logging**: Per-class metrics and predictions
✅ **Visualization**: Training history, sample predictions
✅ **Memory Efficient**: Optimized for 6GB GPU memory

## Usage

```bash
cd models/vision_transformer_224
python vision_transformer_224_mlflow.py
```

## Model Parameters

The optimized Vision Transformer model contains approximately **103M parameters**, making it more memory-efficient while maintaining the powerful self-attention capabilities for capturing long-range dependencies in images.

## Results

- All training metrics are logged to MLflow
- Confusion matrices and classification reports are generated
- Model artifacts are saved in both H5 and TFLite formats
- Training results are saved to JSON for comparison with other models

## Technical Details

This implementation includes several optimizations for resource-constrained environments:

- **Mixed Precision**: Automatic mixed precision training for faster computation
- **Memory Management**: GPU memory growth enabled to prevent allocation issues
- **Balanced Architecture**: Smaller but still effective transformer blocks
- **Efficient Data Pipeline**: Optimized data loading and augmentation

## Notes

- Model is optimized for 6GB GPU memory (tested on RTX 2070 SUPER)
- Uses mixed precision training for memory efficiency
- Reduced from full ViT-Base architecture while maintaining core functionality
- Training from scratch without pre-training (may benefit from longer training) 
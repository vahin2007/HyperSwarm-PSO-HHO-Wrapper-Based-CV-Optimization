# HyperSwarm: PSO-HHO-Wrapper-Based-CV-Optimization
# CPU-Aware Hybrid Vision Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-Advanced_Trees-ffa500?style=for-the-badge) ![Optimization](https://img.shields.io/badge/Math-PSO--HHO-00599C?style=for-the-badge)

An enterprise-grade, hardware-adaptive computer vision pipeline engineered to achieve 98%+ accuracy across diverse imaging modalities. This system bypasses standard end-to-end CNN limitations by fusing Deep Transfer Learning, Meta-Heuristic Feature Selection, and Tree-Based Ensembles.

### 🔬 Supported Modalities
The pipeline is domain-agnostic and has been rigorously benchmarked against:
* **GastroEndoNet:** Gastrointestinal pathology and polyp classification.
* **IIIT-DMJ Smoke:** Environmental hazard and visual smoke detection.
* **Medical X-Ray:** High-resolution skeletal and pulmonary radiograph diagnostics.

The pipeline achieves exceptional accuracy through a combination of:
- Deep transfer learning (EfficientNetV2M)
- Hybrid metaheuristic feature selection (PSO-HHO)
- Ensemble learning (Random Forest + LightGBM)
- Test-Time Augmentation (TTA)

### ⚙️ Core Architecture & Pipeline
This is not a standard `.fit()` deep learning script. It is a multi-stage hybrid classifier:
1. **Hardware-Adaptive Compute:** Autonomously detects execution environments, applying 8GB VRAM limits and mixed-precision (FP16) on GPUs, while dynamically scaling down epochs and warmup stages for rapid CPU testing.
2. **Deep Feature Extraction:** Fine-tunes **EfficientNetV2M**, freezing early layers and extracting dense 512D embeddings from Layer `-4` to capture high-level semantic representations.
3. **Class Balancing:** Injects **BorderlineSMOTE** into the vector space to synthetically balance minority classes before ensemble training.
4. **Meta-Heuristic Optimization:** Deploys a custom **Fast PSO-HHO** (Particle Swarm + Harris Hawks Optimization) algorithm to autonomously reduce the 512D feature space to its most critical subset (154-435 features), dramatically reducing ensemble overfitting.
5. **Hybrid Ensemble & TTA:** Trains a parallel **LightGBM + Random Forest** soft-voting ensemble on the optimized feature vectors. Final inference utilizes Test-Time Augmentation (TTA) to fuse the CNN's softmax probabilities with the tree-ensemble's predictions.

---

### 🚀 Quick Start

**1. Install Dependencies**
```bash
pip install tensorflow>=2.10.0 scikit-learn imbalanced-learn lightgbm matplotlib tqdm joblib opencv-python
```

**2.Dataset Structure**
Ensure your target dataset follows a standard PyTorch/Keras directory structure:

```
dataset_name/
├── train/
│   ├── class_a/
│   └── class_b/
├── val/
└── test/
```
**3. Execution**
Open main.py, update the DATA_PATHS array to point to your local directories, and execute the pipeline. The system will auto-detect your hardware and scale the compute payload accordingly.

```
python main.py
```

📂 Output Artifacts
Upon completion of the training and optimization loops, the pipeline securely exports:

saved_models/best_finetuned_model.h5: The fine-tuned EfficientNetV2M feature extractor.

ultimate_optimized_model.joblib: The serialized LightGBM/RF ensemble, data scalers, PSO-HHO feature masks, and class configurations.

final_optimized_results.png: A high-resolution Confusion Matrix alongside the PSO-HHO convergence curve.


***

### Pre-Push Checklist

Before you run `git push`, you **must** ensure you have a `.gitignore` file in this repository so you don't accidentally push the massive datasets, the cached features, or the `.h5/.joblib` model weights. 

Create a `.gitignore` file in this folder and copy the contents of the given `.gitignore` file.

# ✨ Key Features

- **🚀 GPU/CPU Adaptive**: Automatically detects hardware and adjusts training accordingly
  - GPU: Full 12-epoch fine-tuning with 8GB memory limit
  - CPU: Fast 3-epoch fine-tuning for quick results
- **🎯 98%+ Accuracy Target**: Optimized hyperparameters for maximum performance
- **⚡ Smart Caching**: Features are cached to avoid redundant computation
- **🔄 Test-Time Augmentation**: Multiple predictions averaged for robustness
- **🎲 Class Balancing**: BorderlineSMOTE for handling imbalanced datasets
- **🧬 Feature Optimization**: PSO-HHO algorithm selects optimal feature subset
- **📊 Comprehensive Evaluation**: Detailed metrics and visualizations

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ultimate-classification-pipeline
```

2. **Create virtual environment** (recommended)
```bash
# Windows
python -m venv tf_gpu_env
tf_gpu_env\Scripts\activate

# Linux/Mac
python3 -m venv tf_gpu_env
source tf_gpu_env/bin/activate
```

3. **Install dependencies**
```bash
pip install tensorflow numpy scikit-learn imbalanced-learn lightgbm matplotlib tqdm joblib
```

4. **GPU Setup** (optional, for NVIDIA GPUs)
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow-gpu>=2.10.0

# Verify GPU detection
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## 📁 Dataset Structure

Organize your dataset in the following structure:

```
dataset_name/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── val/
│   ├── class1/
│   │   └── ...
│   └── class2/
│       └── ...
└── test/
    ├── class1/
    │   └── ...
    └── class2/
        └── ...
```

**Supported Formats**: JPG, JPEG, PNG, BMP

### Datasets Used in This Project

1. **GastroEndoNet**
   - Medical imaging dataset
   - Gastrointestinal pathology classification
   - Multiple disease categories

2. **IIITDMJ_Smoke**
   - Environmental monitoring dataset
   - Binary classification: Smoke vs No-Smoke
   - Real-world surveillance images

3. **X-Ray Dataset**
   - Medical diagnostic imaging
   - Various pathology detection
   - High-resolution radiographs

## 🚀 Usage

### Basic Training

1. **Update dataset paths** in the script:
```python
DATA_PATHS = [
    r'path/to/your/dataset/train',
    r'path/to/your/dataset/val',
    r'path/to/your/dataset/test'
]
```

2. **Run the pipeline**:
```bash
python ultimate_pipeline.py
```

### Configuration Options

#### Fine-Tuning Configuration
```python
FINE_TUNE_CONFIG = {
    'enabled': True,              # Enable fine-tuning
    'epochs': 12,                 # Training epochs (auto-adjusted for CPU)
    'unfreeze_layers': 100,       # Number of layers to unfreeze
    'initial_lr': 5e-5,           # Learning rate
    'use_augmentation': True,     # Data augmentation
    'early_stopping_patience': 5, # Early stopping patience
    'reduce_lr_patience': 3       # LR reduction patience
}
```

#### PSO-HHO Feature Selection
```python
PSO_CONFIG = {
    'enabled': True,              # Enable feature selection
    'population_size': 25,        # PSO population size
    'max_iterations': 35,         # Optimization iterations
    'min_features': 154,          # Minimum features (30% of 512)
    'max_features': 435,          # Maximum features (85% of 512)
    'threshold': 0.48,            # Selection threshold
    'n_jobs': -1                  # Parallel processing
}
```

#### Image and Batch Settings
```python
IMG_SIZE = (384, 384)   # Input image size
BATCH_SIZE = 16         # Batch size for training
FEATURE_DIM = 512       # Feature dimension
```

#### Test-Time Augmentation
```python
USE_TTA = True          # Enable TTA
TTA_STEPS = 5           # Number of augmentations
```

### Inference on New Images

```python
from ultimate_pipeline import predict_with_ultimate_model

# List of image paths
image_paths = [
    'path/to/image1.jpg',
    'path/to/image2.jpg',
    'path/to/image3.jpg'
]

# Make predictions
predictions = predict_with_ultimate_model(
    image_paths, 
    model_path='ultimate_optimized_model.joblib'
)

print("Predictions:", predictions)
```

## 🏗️ Architecture

### Pipeline Stages

1. **GPU/CPU Detection** 
   - Automatic hardware configuration
   - 8GB GPU memory limit
   - CPU-optimized training parameters

2. **Data Loading & Augmentation**
   - Efficient TensorFlow data pipelines
   - Real-time augmentation during training
   - Automatic train/val/test splitting

3. **Transfer Learning**
   - EfficientNetV2M backbone (pre-trained on ImageNet)
   - 2-stage fine-tuning:
     - Stage 1: Warmup (1-3 epochs)
     - Stage 2: Fine-tuning last 100 layers (3-12 epochs)

4. **Feature Extraction**
   - 512D features from layer -4
   - RobustScaler normalization
   - Feature caching for efficiency

5. **Class Balancing**
   - BorderlineSMOTE oversampling
   - Handles imbalanced datasets effectively

6. **Feature Selection**
   - PSO-HHO hybrid metaheuristic optimization
   - Selects 154-435 optimal features from 512
   - Parallel fitness evaluation

7. **Ensemble Training**
   - Random Forest (400 trees)
   - LightGBM (500 trees)
   - Soft voting classifier

8. **Hybrid Prediction**
   - Combines fine-tuned CNN + ensemble
   - Test-Time Augmentation (5 augmentations)
   - Adaptive weighting based on validation performance

## 📊 Performance

### Expected Accuracy Ranges

| Dataset | Training Time (GPU):12 Epochs| Training Time (CPU):3 Epochs | Expected Accuracy |
|---------|-------------------|-------------------|-------------------|
| GastroEndoNet | ~60-90 min | ~20-30 min | 96-98% |
| IIITDMJ_Smoke | ~45-75 min | ~15-25 min | 97-99% |
| X-Ray | ~60-90 min | ~20-30 min | 95-98% |

### Evaluation Metrics

The pipeline provides:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro-averaged F1 score
- **Confusion Matrix**: Detailed per-class performance
- **Classification Report**: Precision, Recall, F1 for each class
- **PSO-HHO Convergence**: Feature selection optimization curve

## 📈 Output Files

After training, the pipeline generates:

```
project_folder/
├── saved_models/
│   └── best_finetuned_model.h5          # Fine-tuned CNN model
├── feature_cache/
│   └── features_512d_<hash>.npz         # Cached features
├── ultimate_optimized_model.joblib       # Complete pipeline artifacts
└── final_optimized_results.png          # Confusion matrix + convergence plot
```

## 🔧 Troubleshooting

### Common Issues

**1. GPU Memory Error**
```
ValueError: Setting memory limit is required for GPU virtual devices
```
✅ **Fixed in latest version** - 8GB memory limit is now set automatically

**2. CPU Too Slow**
- Pipeline automatically reduces epochs from 12 to 3 on CPU
- Expected training time: 15-30 minutes
- Accuracy may be 1-2% lower than GPU version

**3. Out of Memory (OOM)**
```python
# Reduce batch size
BATCH_SIZE = 8  # or even 4

# Disable TTA during training
USE_TTA = False
```

**4. Imbalanced Dataset Issues**
```python
# Adjust SMOTE parameters
USE_ADVANCED_SMOTE = True  # Enable BorderlineSMOTE

# Or disable if dataset is balanced
USE_ADVANCED_SMOTE = False
```

**5. Low Accuracy**
- Ensure dataset is properly organized
- Check for data quality issues
- Increase fine-tuning epochs (GPU only)
- Enable TTA for better test performance

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free space

**Recommended (GPU):**
- GPU: NVIDIA GPU with 8GB+ VRAM (GTX 1070+, RTX 2060+)
- RAM: 16GB
- Storage: 20GB free space

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for additional datasets
- Integration of newer architectures (Vision Transformers)
- Advanced augmentation techniques
- Hyperparameter tuning automation
- Docker containerization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **EfficientNetV2**: Tan & Le (Google Research)
- **PSO-HHO**: Particle Swarm Optimization + Harris Hawks Optimization
- **Datasets**:
  - GastroEndoNet contributors
  - IIIT Delhi & IIIT-DM Jabalpur (Smoke dataset)
  - X-Ray dataset contributors

## 📧 Contact

For questions, issues, or collaborations:
- Create an issue on GitHub
- Email: [vahinsathu@example.com]

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{HyperSwarm,
  title={HyperSwarm: PSO-HHO-Wrapper-Based-CV-Optimization},
  author={Sathu Vahin Reddy},
  year={2024},
  url={https://github.com/yourusername/HyperSwarm: PSO-HHO-Wrapper-Based-CV-Optimization}
}
```

## 🗺️ Roadmap

- [ ] Add support for multi-label classification
- [ ] Implement k-fold cross-validation
- [ ] Add Grad-CAM visualization
- [ ] Docker deployment
- [ ] Web interface for inference
- [ ] Model quantization for edge deployment
- [ ] Support for video classification
- [ ] AutoML hyperparameter optimization

---

**Last Updated**: March 2025  
**Version**: 2.0 (GPU/CPU Adaptive)  
**Status**: Production Ready ✅

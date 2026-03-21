import os
import logging
import tensorflow as tf
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- CHOOSE YOUR DATASET ZIP AND UNZIPPED FOLDER NAME ---
# 1. Place your .zip file in the same directory as this script.
# 2. Set DATASET_ZIP_NAME to its name.
# 3. Set DATASET_ROOT_DIR to the name of the folder that appears after unzipping.
DATASET_ZIP_NAME = "IIITDMJ_Smoke.zip"
DATASET_ROOT_DIR = "IIITDMJ_Smoke"

# DATA_PATHS will be set automatically by the setup_dataset_and_get_paths function
DATA_PATHS = []

BASE_MODEL_NAME = 'EfficientNetV2M'
IMG_SIZE = (384, 384)

# Enhanced Fine-tuning configuration
FINE_TUNE_CONFIG = {
    'enabled': True,
    'epochs': 12,
    'unfreeze_layers': 100,
    'initial_lr': 5e-5,
    'use_augmentation': True,
    'early_stopping_patience': 4,
    'reduce_lr_patience': 3
}
BATCH_SIZE = 16

FEATURE_CACHE_DIR = './feature_cache'
MODEL_SAVE_DIR = './saved_models'
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# This must match the output of your feature extractor
FEATURE_DIM = 256 
PSO_CONFIG = {
    'enabled': True,
    'population_size': 30,
    'max_iterations': 50,
    'min_features': int(0.3 * FEATURE_DIM),  # 76
    'max_features': int(0.85 * FEATURE_DIM), # 217
    'threshold': 0.48,
    'use_cv': False,
    'n_jobs': -1,
    'feature_penalty': 0.05 # 5% penalty
}

SAVED_RF_PARAMS = {
    'n_estimators': 400,
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42
}

USE_ADVANCED_SMOTE = True
USE_TTA = True
TTA_STEPS = 5

np.random.seed(42)
tf.random.set_seed(42)

# Global flag for GPU availability
HAS_GPU = False

# ============================================================================
# GPU SETUP
# ============================================================================

def setup_gpu():
    """Configure GPU with memory limit or detect CPU mode."""
    global HAS_GPU, FINE_TUNE_CONFIG
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # 1. Set memory growth on PHYSICAL devices FIRST
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 2. THEN, set the logical device configuration
            # This combination can be unstable; wrap in try/except
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
                )
            except RuntimeError as e_limit:
                print(f"  Info: Could not set 8GB memory limit ({e_limit}). Using memory growth only.")

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f" GPU DETECTED: {gpus[0].name}")
            print(f"  Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
            
            HAS_GPU = True
            
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("  Mixed precision: Enabled (FP16)")
            except:
                print("  Mixed precision: Not available")
            
        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
            print("  Falling back to CPU mode...")
            HAS_GPU = False
    else:
        print(" NO GPU DETECTED - Running on CPU")
        HAS_GPU = False
    
    # Adjust training config based on hardware
    if not HAS_GPU:
        print("\n" + "=" * 70)
        print("CPU MODE ADJUSTMENTS")
        print("=" * 70)
        
        FINE_TUNE_CONFIG['epochs'] = 3
        FINE_TUNE_CONFIG['early_stopping_patience'] = 2
        FINE_TUNE_CONFIG['reduce_lr_patience'] = 1
        
    print("=" * 70 + "\n")
    
    np.random.seed(42)
    tf.random.set_seed(42)

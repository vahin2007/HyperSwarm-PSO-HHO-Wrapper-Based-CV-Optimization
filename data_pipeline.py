import os
import zipfile
import tensorflow as tf
from tqdm import tqdm
import config

# ============================================================================
# DATASET SETUP (WITH AUTO-UNZIP & PATH DETECTION)
# ============================================================================

def setup_dataset_and_get_paths(zip_name, target_dir):
    """
    Checks if target dir exists. If not, unzips file.
    Inspects dir and returns correct DATA_PATHS list (1, 2, or 3 paths),
    even if the data is nested one level deep.
    """
    print("=" * 70)
    print("DATASET SETUP & VALIDATION")
    print("=" * 70)

    # 1. Check if the unzipped folder already exists
    if not os.path.isdir(target_dir):
        print(f"Dataset folder '{target_dir}' not found.")
        zip_path = os.path.join(os.getcwd(), zip_name)
        
        # 2. Check if the zip file exists
        if not os.path.exists(zip_path):
            print(f"✗ ERROR: Zip file '{zip_path}' not found.")
            print("\nINSTRUCTIONS:")
            print(f"1. Please download the dataset ({zip_name})")
            print(f"2. Place it in this directory: {os.getcwd()}")
            print("3. Re-run the script.")
            print("=" * 70 + "\n")
            return None  # Failure

        # 3. Unzip the file
        print(f"✓ Found dataset zip: {zip_path}")
        print(f"  Unzipping to create '{target_dir}' folder...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file in tqdm(zip_ref.infolist(), desc="  Extracting", ncols=80, unit="file"):
                    zip_ref.extract(file, os.getcwd()) 
            print(f"✓ Successfully unzipped dataset.")
        
        except zipfile.BadZipFile:
            print(f"✗ ERROR: Failed to unzip '{zip_name}'. The file may be corrupt.")
            return None
        except Exception as e:
            print(f"✗ An unexpected error occurred during unzipping: {e}")
            return None
    
    # 4. --- NEW LOGIC: DETECT DATASET STRUCTURE ---
    print(f"✓ Dataset folder found: {target_dir}")
    print(f"Inspecting contents of '{target_dir}'...")
    
    current_path = target_dir
    # check for path structures inside the path
    train_path = os.path.join(current_path, 'train')
    val_path = os.path.join(current_path, 'val')
    test_path = os.path.join(current_path, 'test')
    
    # Scenario C (3-path): train, val, AND test folders exist
    if os.path.isdir(train_path) and os.path.isdir(val_path) and os.path.isdir(test_path):
        print(f"✓ Found 'train', 'val', 'test' subfolders.")
        print("  Setting mode to 3-path (Train/Val/Test).")
        print("=" * 70 + "\n")
        return [train_path, val_path, test_path]
    
    # Scenario B (2-path): train AND test folders exist
    elif os.path.isdir(train_path) and os.path.isdir(test_path):
        print(f"✓ Found 'train' and 'test' subfolders (but no 'val').")
        print("  Setting mode to 2-path (Train/Val, Test).")
        print("=" * 70 + "\n")
        return [train_path, test_path]
    
    # Scenario A (1-path): The current_path itself contains the class folders
    else:
        print(f"✓ No 'train/val/test' structure found.")
        print(f"  Assuming '{current_path}' contains the class folders.")
        print("  Setting mode to 1-path (will split 70/10/20).")
        print("=" * 70 + "\n")
        return [current_path] # Return the root folder

# ============================================================================
# DATA LOADING
# ============================================================================

def create_augmentation_layer():
    """Creates a sequential model for data augmentation."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.GaussianNoise(0.1),
    ])

def load_data_optimized(data_paths):
    """Loads and splits data based on the number of paths provided."""
    AUTOTUNE = tf.data.AUTOTUNE
    num_paths = len(data_paths)
    
    if num_paths == 3:
        return load_three_folders(data_paths[0], data_paths[1], data_paths[2], AUTOTUNE)
    elif num_paths == 2:
        return load_two_folders(data_paths[0], data_paths[1], AUTOTUNE)
    elif num_paths == 1:
        return load_single_folder(data_paths[0], AUTOTUNE)
    else:
        raise ValueError(f"Expected 1-3 paths, got {num_paths}")

def _rescale_and_prefetch(dataset, autotune, batch=True):
    """Helper to rescale, batch, and prefetch."""
    # Rescaling is handled by EfficientNetV2's preprocessing layer
    if batch:
        dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(buffer_size=autotune)

def load_three_folders(train_path, val_path, test_path, autotune):
    """Loads train, val, and test data from three separate directories."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='int', shuffle=True, seed=42
    )
    class_names = train_ds.class_names
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='int', shuffle=True, seed=42
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='int', shuffle=False, seed=42
    )
    
    if FINE_TUNE_CONFIG['use_augmentation']:
        augmentation = create_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=autotune
        )
    
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    
    return train_ds, val_ds, test_ds, class_names

def load_two_folders(train_val_path, test_path, autotune):
    """Loads train/val from one path (80/20 split) and test from another."""
    # Load unbatched to split by samples
    train_val_ds_unbatched = tf.keras.utils.image_dataset_from_directory(
        train_val_path, image_size=IMG_SIZE, batch_size=None,
        label_mode='int', shuffle=True, seed=42
    )
    class_names = train_val_ds_unbatched.class_names
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='int', shuffle=False, seed=42
    )

    dataset_size = tf.data.experimental.cardinality(train_val_ds_unbatched).numpy()
    train_size = int(0.8 * dataset_size)
    
    train_ds = train_val_ds_unbatched.take(train_size)
    val_ds = train_val_ds_unbatched.skip(train_size)
    
    if FINE_TUNE_CONFIG['use_augmentation']:
        augmentation = create_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=autotune
        )
    
    # Apply batching and prefetching
    train_ds = _rescale_and_prefetch(train_ds, autotune, batch=True)
    val_ds = _rescale_and_prefetch(val_ds, autotune, batch=True)
    test_ds = _rescale_and_prefetch(test_ds, autotune, batch=False)
    
    return train_ds, val_ds, test_ds, class_names

def load_single_folder(data_path, autotune):
    """Loads data from a single path and splits 70/10/20."""
    # Load unbatched to split by samples
    full_ds_unbatched = tf.keras.utils.image_dataset_from_directory(
        data_path, image_size=IMG_SIZE, batch_size=None,
        label_mode='int', shuffle=True, seed=42
    )
    class_names = full_ds_unbatched.class_names

    dataset_size = tf.data.experimental.cardinality(full_ds_unbatched).numpy()
    if dataset_size < 10:
        raise ValueError("Not enough images for splitting.")
        
    train_size = int(0.7 * dataset_size)
    val_size = int(0.10 * dataset_size)

    train_ds = full_ds_unbatched.take(train_size)
    remaining = full_ds_unbatched.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    
    if FINE_TUNE_CONFIG['use_augmentation']:
        augmentation = create_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=autotune
        )
    
    # Apply batching and prefetching
    train_ds = _rescale_and_prefetch(train_ds, autotune, batch=True)
    val_ds = _rescale_and_prefetch(val_ds, autotune, batch=True)
    test_ds = _rescale_and_prefetch(test_ds, autotune, batch=True)
    
    return train_ds, val_ds, test_ds, class_names
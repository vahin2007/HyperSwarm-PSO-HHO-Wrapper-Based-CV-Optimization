import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
import config

# ============================================================================
# MODEL CREATION & FINE-TUNING (CPU-AWARE)
# ============================================================================

def create_finetuned_model(num_classes, model_path=None):
    """Creates the fine-tunable EfficientNetV2M model."""
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        # Find base_model layer in the loaded model
        base_model = next((layer for layer in model.layers if "efficientnetv2" in layer.name), None)
        if base_model is None:
            raise ValueError("Could not find base model layer in loaded model.")
        return model, base_model
    
    base_model = tf.keras.applications.EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3),
        include_preprocessing=True # Use built-in preprocessing
    )
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dense(512, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # This is the layer we will extract features from (output shape 256)
    x = tf.keras.layers.Dense(256, activation='relu', name="feature_extractor_layer",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model

def fine_tune_model(model, base_model, train_ds, val_ds, num_classes):
    """Runs the 2-stage fine-tuning process."""
    
    # Stage 1: Warmup
    warmup_epochs = 1 if not HAS_GPU else 3
    print(f"Stage 1: Warmup training ({warmup_epochs} epoch{'s' if warmup_epochs > 1 else ''})...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=2, restore_best_weights=True
        )],
        verbose=1
    )
    
    # Stage 2: Fine-tune
    finetune_epochs = FINE_TUNE_CONFIG['epochs']
    print(f"\nStage 2: Fine-tuning ({finetune_epochs} epochs)...")
    
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = total_layers - FINE_TUNE_CONFIG['unfreeze_layers']
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_CONFIG['initial_lr']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=FINE_TUNE_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=FINE_TUNE_CONFIG['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, 'best_finetuned_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history1, history2

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_from_finetuned(model, dataset, desc=""):
    """Extracts the 256-dim features from the 'feature_extractor_layer'."""
    try:
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("feature_extractor_layer").output
        )
    except ValueError as e:
        print(f"Error creating feature extractor: {e}")
        print("Available layer names:", [layer.name for layer in model.layers])
        raise e

    features_list = []
    labels_list = []
    
    for batch_img, batch_lbl in tqdm(dataset, desc=f"  {desc}", ncols=80):
        batch_feat = feature_extractor(batch_img, training=False)
        features_list.append(batch_feat.numpy())
        labels_list.append(batch_lbl.numpy())
    
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    
    return features, labels

# ============================================================================
# ENSEMBLE
# ============================================================================

def create_powerful_ensemble(X_train, y_train, rf_params):
    """Creates a high-performance voting ensemble of RF and LGBM."""
    
    rf = RandomForestClassifier(**rf_params)
    
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lgbm', lgbm)],
        voting='soft',
        n_jobs=-1
    )
    
    print("Training ensemble...")
    start = time.time()
    ensemble.fit(X_train, y_train)
    elapsed = time.time() - start
    
    print(f"Ensemble trained in {elapsed:.1f}s\n")
    return ensemble
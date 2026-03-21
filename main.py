import os
import time
import hashlib
import warnings
import numpy as np
import tensorflow as tf
import joblib
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import BorderlineSMOTE

warnings.filterwarnings('ignore')

# --- Local Modules ---
import config
from data_pipeline import setup_dataset_and_get_paths, load_data_optimized
from models import create_finetuned_model, fine_tune_model, extract_features_from_finetuned, create_powerful_ensemble
from optimizer import FastPSOHHO
from evaluation import evaluate_comprehensive

def main_ultimate():
    overall_start = time.time()
    
    print("\n" + "=" * 70)
    print("OMNIVISION: 98%+ ACCURACY HYBRID PIPELINE")
    print("=" * 70)
    
    try:
        # 1. Hardware Setup (Modifies config globals dynamically)
        config.setup_gpu()
        
        # 2. Load Data
        train_ds, val_ds, test_ds, class_names = load_data_optimized(config.DATA_PATHS)
        num_classes = len(class_names)
        
        # 3. Create & Fine-tune Model
        finetuned_path = os.path.join(config.MODEL_SAVE_DIR, 'best_finetuned_model.h5')
        if config.FINE_TUNE_CONFIG['enabled']:
            model, base_model = create_finetuned_model(num_classes, finetuned_path if os.path.exists(finetuned_path) else None)
            if not os.path.exists(finetuned_path):
                model, hist1, hist2 = fine_tune_model(model, base_model, train_ds, val_ds, num_classes)
            else:
                print("Using existing fine-tuned model\n")
        
        # 4. Extract Features (512D)
        print("=" * 70 + "\nFEATURE EXTRACTION (512D)\n" + "=" * 70)
        cache_key = hashlib.md5(f"{str(config.DATA_PATHS[0])}_512d_optimized".encode()).hexdigest()[:16]
        cache_path = os.path.join(config.FEATURE_CACHE_DIR, f"features_512d_{cache_key}.npz")
        
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            train_feat, train_lab = data['train_features'], data['train_labels']
            val_feat, val_lab = data['val_features'], data['val_labels']
            test_feat, test_lab = data['test_features'], data['test_labels']
        else:
            train_feat, train_lab = extract_features_from_finetuned(model, train_ds, "Train")
            val_feat, val_lab = extract_features_from_finetuned(model, val_ds, "Val")
            test_feat, test_lab = extract_features_from_finetuned(model, test_ds, "Test")
            np.savez_compressed(cache_path, train_features=train_feat, train_labels=train_lab,
                                val_features=val_feat, val_labels=val_lab, test_features=test_feat, test_labels=test_lab)
        
        # 5. Scale Features
        scaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_feat)
        val_scaled = scaler.transform(val_feat)
        test_scaled = scaler.transform(test_feat)
        
        # 6. SMOTE Balancing
        if config.USE_ADVANCED_SMOTE:
            smote = BorderlineSMOTE(random_state=42, k_neighbors=5, m_neighbors=10)
            train_balanced, labels_balanced = smote.fit_resample(train_scaled, train_lab)
        else:
            train_balanced, labels_balanced = train_scaled, train_lab
            
        # 7. PSO-HHO Feature Selection
        print("=" * 70 + "\nPSO-HHO FEATURE SELECTION\n" + "=" * 70)
        if config.PSO_CONFIG['enabled']:
            pso = FastPSOHHO(X_train=train_balanced, y_train=labels_balanced, X_val=val_scaled, y_val=val_lab,
                             rf_params=config.SAVED_RF_PARAMS, pop_size=config.PSO_CONFIG['population_size'],
                             max_iter=config.PSO_CONFIG['max_iterations'], min_feat=config.PSO_CONFIG['min_features'],
                             max_feat=config.PSO_CONFIG['max_features'])
            mask, convergence = pso.optimize()
            X_train_sel, X_val_sel, X_test_sel = train_balanced[:, mask], val_scaled[:, mask], test_scaled[:, mask]
        else:
            X_train_sel, X_val_sel, X_test_sel, mask, convergence = train_balanced, val_scaled, test_scaled, None, []
            
        # 8. Train Ensemble & Evaluate
        ensemble_model = create_powerful_ensemble(X_train_sel, labels_balanced, config.SAVED_RF_PARAMS)
        best_acc, f1_macro, _ = evaluate_comprehensive(model, ensemble_model, test_ds, X_test_sel, test_lab, class_names, convergence)
        
        # 9. Save Artifacts
        artifacts = {
            'finetuned_model_path': finetuned_path, 'ensemble_model': ensemble_model, 'scaler': scaler,
            'mask': mask, 'class_names': class_names, 'accuracy': best_acc, 'f1_score': f1_macro,
            'use_tta': config.USE_TTA, 'tta_steps': config.TTA_STEPS if config.USE_TTA else 0
        }
        joblib.dump(artifacts, 'ultimate_optimized_model.joblib', compress=3)
        
        print(f"\nPipeline finished in {(time.time() - overall_start)/60:.1f} minutes.")
        return best_acc
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        return 0

if __name__ == "__main__":
    main_ultimate()
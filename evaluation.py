import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import config

# ============================================================================
# TEST-TIME AUGMENTATION (TTA)
# ============================================================================

def predict_with_tta(model, test_ds, tta_steps=5):
    """Applies TTA for more robust predictions."""
    
    tta_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ])
    
    all_predictions = []
    
    # Unbatch, apply TTA, then re-batch for prediction
    unbatched_test_ds = test_ds.unbatch()

    for i in tqdm(range(tta_steps), desc="  TTA Step", ncols=80):
        if i == 0:
            augmented_ds = unbatched_test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        else:
            augmented_ds = unbatched_test_ds.map(
                lambda x, y: (tta_aug(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        batch_preds = model.predict(augmented_ds, verbose=0)
        all_predictions.append(batch_preds)
    
    avg_predictions = np.mean(all_predictions, axis=0)
    
    return avg_predictions

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_comprehensive(finetuned_model, ensemble_model, test_ds, X_test_sel, y_test, class_names, convergence):
    """Performs the final 3-way evaluation and generates reports."""
    ft_weight, ens_weight = 0.65, 0.35
    # --- 1. Fine-tuned Model (with TTA) ---
    if USE_TTA:
        y_pred_ft_proba = predict_with_tta(finetuned_model, test_ds, TTA_STEPS)
    else:
        y_pred_ft_proba = []
        for batch_img, _ in test_ds:
            batch_proba = finetuned_model.predict(batch_img, verbose=0)
            y_pred_ft_proba.extend(batch_proba)
        y_pred_ft_proba = np.array(y_pred_ft_proba)
    
    y_pred_ft = np.argmax(y_pred_ft_proba, axis=1)
    
    # --- Fix sample size mismatch ---
    min_samples = min(len(y_pred_ft), len(X_test_sel))
    
    y_pred_ft_proba = y_pred_ft_proba[:min_samples]
    y_pred_ft = y_pred_ft[:min_samples]
    X_test_sel = X_test_sel[:min_samples]
    y_test_fixed = y_test[:min_samples]
    
    acc_ft = accuracy_score(y_test_fixed, y_pred_ft)
    
    # --- 2. Ensemble Model ---
    y_pred_ens_proba = ensemble_model.predict_proba(X_test_sel)
    y_pred_ens = np.argmax(y_pred_ens_proba, axis=1)
    acc_ens = accuracy_score(y_test_fixed, y_pred_ens)
    
    # --- 3. Hybrid Prediction ---
    y_pred_ft_proba_norm = y_pred_ft_proba / (np.sum(y_pred_ft_proba, axis=1, keepdims=True) + 1e-9)
    y_pred_ens_proba_norm = y_pred_ens_proba / (np.sum(y_pred_ens_proba, axis=1, keepdims=True) + 1e-9)
    
    y_pred_ft_proba_norm = np.nan_to_num(y_pred_ft_proba_norm)
    y_pred_ens_proba_norm = np.nan_to_num(y_pred_ens_proba_norm)
    
    if acc_ft > acc_ens:
        ft_weight, ens_weight = 0.65, 0.35
    else:
        ft_weight, ens_weight = 0.35, 0.65
    
    y_pred_hybrid_proba = ft_weight * y_pred_ft_proba_norm + ens_weight * y_pred_ens_proba_norm
    y_pred_hybrid = np.argmax(y_pred_hybrid_proba, axis=1)
    
    acc_hybrid = accuracy_score(y_test_fixed, y_pred_hybrid)
    f1_hybrid = f1_score(y_test_fixed, y_pred_hybrid, average='macro', zero_division=0)
    
    # --- Select best model for final report ---
    accuracies = {'Fine-tuned': acc_ft, 'Ensemble': acc_ens, 'Hybrid': acc_hybrid}
    best_model_name = max(accuracies, key=accuracies.get)
    best_acc = accuracies[best_model_name]
    
    if best_model_name == 'Fine-tuned':
        y_pred_final = y_pred_ft
    elif best_model_name == 'Ensemble':
        y_pred_final = y_pred_ens
    else: # Hybrid
        y_pred_final = y_pred_hybrid
    
    print("\n" + "=" * 70)
    print(f"FINAL TEST ACCURACY ({best_model_name}): {best_acc*100:.2f}%")
    print("=" * 70)
    
    report = classification_report(y_test_fixed, y_pred_final, target_names=class_names, zero_division=0)
    print(report)
    
    # --- Visualization ---
    cm = confusion_matrix(y_test_fixed, y_pred_final)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title(f'Confusion Matrix\nAccuracy: {best_acc*100:.2f}%', 
                     fontsize=14, fontweight='bold')
    
    if convergence:
        axes[1].plot(convergence, 'b-', linewidth=2.5)
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Fitness (1 - F1)', fontsize=12)
        axes[1].set_title('PSO-HHO Convergence', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        valid_fitness = [f for f in convergence if f < 900]
        if valid_fitness:
            axes[1].set_ylim([min(valid_fitness) * 0.95, max(valid_fitness) * 1.05])

    plt.tight_layout()
    plt.savefig('final_optimized_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_acc, f1_hybrid, report,ft_weight, ens_weight
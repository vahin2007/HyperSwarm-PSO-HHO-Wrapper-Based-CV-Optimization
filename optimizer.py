import math
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import config

# ============================================================================
# PSO-HHO (WITH FEATURE PENALTY)
# ============================================================================

class FastPSOHHO:
    """PSO-HHO with proper integer feature counts, using LightGBM."""
    
    def __init__(self, X_train, y_train, X_val, y_val, rf_params,
                 pop_size=25, max_iter=35, min_feat=154, max_feat=435):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.rf_params = rf_params
        self.n_features = X_train.shape[1]
        
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.min_features = max(1, min(int(min_feat), self.n_features))
        self.max_features = min(int(max_feat), self.n_features)
        
        print(f"PSO-HHO Config:")
        print(f"  Population: {self.pop_size}, Iterations: {self.max_iter}")
        print(f"  Feature range: [{self.min_features}, {self.max_features}] of {self.n_features}")
        
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.8
        self.c2 = 1.8
        
        self.best_fitness = float('inf')
        self.best_position = None
        self.convergence = []
    
    def evaluate_fitness_fast(self, position):
        """Fitness function using LightGBM + Feature Penalty."""
        selected = position > PSO_CONFIG['threshold']
        n_selected = np.sum(selected)
        
        if n_selected < self.min_features or n_selected > self.max_features or n_selected == 0:
            return 999.0
        
        X_train_sel = self.X_train[:, selected]
        X_val_sel = self.X_val[:, selected]
        
        try:
            lgbm = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                class_weight='balanced',
                n_jobs=PSO_CONFIG['n_jobs'],
                random_state=42,
                verbose=-1
            )
            
            lgbm.fit(X_train_sel, self.y_train)
            y_pred = lgbm.predict(X_val_sel)
            
            f1 = f1_score(self.y_val, y_pred, average='macro', zero_division=0)
            
            # --- FITNESS CALCULATION WITH PENALTY ---
            feature_ratio = n_selected / self.n_features
            
            # 1.0 - F1 (error) + 5% penalty for feature ratio
            fitness = (1.0 - f1) + (PSO_CONFIG['feature_penalty'] * feature_ratio)
            
            return fitness
            
        except Exception as e:
            return 999.0
    
    def levy_flight(self, dim):
        """Helper function for HHO exploitation."""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / np.abs(v)**(1 / beta)
        return 0.01 * step
    
    def optimize(self):
        """Run the parallel PSO-HHO optimization loop."""
        positions = np.random.rand(self.pop_size, self.n_features)
        velocities = np.random.rand(self.pop_size, self.n_features) * 0.1
        
        pbest_pos = positions.copy()
        pbest_fit = np.full(self.pop_size, float('inf'))
        
        pbar = tqdm(range(self.max_iter), desc="PSO-HHO Optimization", ncols=100)
        
        for t in pbar:
            w = self.w_max - (self.w_max - self.w_min) * (t / self.max_iter)
            
            # Run fitness evaluations in parallel on all CPU cores
            fitness_vals = Parallel(n_jobs=PSO_CONFIG['n_jobs'])(
                delayed(self.evaluate_fitness_fast)(positions[i]) 
                for i in range(self.pop_size)
            )
            
            # Update personal and global bests
            for i in range(self.pop_size):
                if fitness_vals[i] < pbest_fit[i]:
                    pbest_fit[i] = fitness_vals[i]
                    pbest_pos[i] = positions[i].copy()
                
                if fitness_vals[i] < self.best_fitness:
                    self.best_fitness = fitness_vals[i]
                    self.best_position = positions[i].copy()
            
            Y_mu = np.mean(positions, axis=0)
            
            # Update particle positions
            for i in range(self.pop_size):
                xi_0 = -1 + 2 * np.random.rand()
                xi = 2 * xi_0 * (1 - t / self.max_iter)
                
                if np.abs(xi) >= 1:
                    # Exploration (PSO)
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = (w * velocities[i] + 
                                   self.c1 * r1 * (pbest_pos[i] - positions[i]) +
                                   self.c2 * r2 * (self.best_position - positions[i]))
                    positions[i] = positions[i] + velocities[i]
                else:
                    # Exploitation (HHO)
                    J = 2 * (1 - np.random.rand())
                    rho = np.random.rand()
                    
                    if rho >= 0.5 and np.abs(xi) >= 0.5:
                        positions[i] = self.best_position - xi * np.abs(J * self.best_position - positions[i])
                    elif rho >= 0.5 and np.abs(xi) < 0.5:
                        positions[i] = self.best_position - xi * np.abs(J * self.best_position - Y_mu)
                    elif rho < 0.5 and np.abs(xi) >= 0.5:
                        LF = self.levy_flight(self.n_features)
                        positions[i] = self.best_position - xi * np.abs(J * self.best_position - positions[i]) + np.random.rand(self.n_features) * LF
                    else:
                        LF = self.levy_flight(self.n_features)
                        positions[i] = self.best_position - xi * np.abs(J * self.best_position - Y_mu) + np.random.rand(self.n_features) * LF
                
                positions[i] = np.clip(positions[i], 0, 1)
            
            self.convergence.append(self.best_fitness)
            
            # Update progress bar
            if self.best_position is not None:
                n_sel = np.sum(self.best_position > PSO_CONFIG['threshold'])
                best_f1_approx = (1 - (self.best_fitness - 0.05 * (n_sel / self.n_features))) * 100 
                pbar.set_postfix({'F1': f'{best_f1_approx:.2f}%', 'Feats': n_sel})
            else:
                pbar.set_postfix({'F1': 'N/A', 'Feats': 'N/A'})

        
        pbar.close()
        
        if self.best_position is None:
            mask = np.ones(self.n_features, dtype=bool)
        else:
            mask = self.best_position > PSO_CONFIG['threshold']
            
        n_sel = np.sum(mask)
        if self.best_fitness < 900:
             final_f1_approx = (1 - (self.best_fitness - 0.05 * (n_sel / self.n_features))) * 100
        else:
             final_f1_approx = 0.0

        print(f"\nPSO-HHO Complete:")
        print(f"  Best Val F1 (approx): {final_f1_approx:.2f}%")
        print(f"  Selected: {n_sel}/{self.n_features} features\n")
        
        return mask, self.convergence
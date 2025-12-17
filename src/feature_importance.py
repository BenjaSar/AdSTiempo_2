"""
Advanced Feature Importance Analysis using Permutation Method
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep learning libraries
import torch
import torch.nn as nn

# Import formatting utility
from utils.misc import print_box

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = None

# ============================================================================
# ADVANCED FEATURE IMPORTANCE ANALYSIS (OPTIONAL)
# ============================================================================

class FeatureImportance:
    """Analyze feature importance using permutation method"""
    
    @staticmethod
    def calculate_importance(model, test_loader, device, feature_names, 
                            n_repeats=5, max_batches=20, output_dir: str=None):
        """
        Calculate feature importance using permutation method
        
        Note: This is computationally expensive. Using subset of data.
        """
        global OUTPUT_DIR
        if output_dir is not None:
            OUTPUT_DIR = output_dir
        else:
            print("   ⚠️  OUTPUT_DIR not set. Plots will not be saved.")
            
        print_box("\nBONUS: FEATURE IMPORTANCE ANALYSIS")
        
        print(f"🔬 Calculating feature importance (this may take a while)...")
        print(f"   Using {n_repeats} permutations per feature")
        print(f"   Analyzing {max_batches} batches\n")
        
        model.eval()
        criterion = nn.MSELoss()
        
        # Calculate baseline loss
        baseline_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, batch_y, _ in test_loader:
                if batch_count >= max_batches:
                    break
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                baseline_loss += criterion(output, batch_y).item()
                batch_count += 1
        
        baseline_loss /= batch_count
        print(f"   Baseline loss: {baseline_loss:.6f}")
        
        # Calculate importance for each feature
        importance_scores = {}
        
        for feat_idx, feat_name in enumerate(feature_names):
            if feat_idx % 10 == 0:
                print(f"   Processing feature {feat_idx + 1}/{len(feature_names)}...")
            
            losses = []
            
            for _ in range(n_repeats):
                perm_loss = 0
                batch_count = 0
                
                with torch.no_grad():
                    for batch_x, batch_y, _ in test_loader:
                        if batch_count >= max_batches:
                            break
                        
                        batch_x = batch_x.clone()
                        
                        # Permute feature
                        perm_idx = torch.randperm(batch_x.size(0))
                        batch_x[:, :, feat_idx] = batch_x[perm_idx, :, feat_idx]
                        
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        output = model(batch_x)
                        perm_loss += criterion(output, batch_y).item()
                        batch_count += 1
                
                losses.append(perm_loss / batch_count)
            
            # Importance = increase in loss
            importance_scores[feat_name] = np.mean(losses) - baseline_loss
        
        print("   ✅ Feature importance calculated\n")
        
        # Sort and display
        sorted_importance = sorted(importance_scores.items(), 
                                  key=lambda x: abs(x[1]), reverse=True)
        
        print("📊 TOP 20 MOST IMPORTANT FEATURES")
        print("─" * 80)
        print(f"{'Rank':<6} {'Feature':<30} {'Importance Score':>20}")
        print("─" * 80)
        
        for rank, (feat, score) in enumerate(sorted_importance[:20], 1):
            print(f"{rank:<6} {feat:<30} {score:>20.6f}")
        
        print("─" * 80 + "\n")
        
        # Visualize
        FeatureImportance._plot_importance(sorted_importance[:20])
        
        return importance_scores
    
    @staticmethod
    def _plot_importance(top_features):
        """Plot feature importance"""
        global OUTPUT_DIR

        features, scores = zip(*top_features)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['#2E86AB' if s > 0 else '#C73E1D' for s in scores]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Importance Score (Increase in MSE)', fontsize=12)
        ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold', pad=20)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if OUTPUT_DIR is not None:
            plt.savefig(f'{OUTPUT_DIR}07_feature_importance.png', dpi=300, bbox_inches='tight')
            print("   ✅ Saved: 07_feature_importance.png\n")
        else:
            plt.show()
        plt.close()
"""
Compare Bitcoin Forecasting Results for Different Prediction Windows
Compares models trained with pred_len = 7, 10, 30, 45 days

Usage:
python compare_forecast_windows.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║          BITCOIN FORECAST COMPARISON - MULTIPLE PREDICTION WINDOWS            ║
║                     Comparing pred_len: 7, 10, 30, 45 days                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

# Import necessary classes from main script
from bitcoin_transformer import (
    BitcoinDataLoader, FeatureEngineer, TimeSeriesDataset, 
    TimeSeriesTransformer, TransformerTrainer, Evaluator
)


class ForecastComparator:
    """Compare forecasting results across different prediction windows"""
    
    def __init__(self, pred_lengths=[7, 30, 45]):
        self.pred_lengths = pred_lengths
        self.results = {}
        
    def load_results(self, results_dir='ventanas'):
        """Load pre-computed results from saved files"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                    LOADING PRE-COMPUTED RESULTS                               ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        for pred_len in self.pred_lengths:
            result_file = os.path.join(results_dir, f'results_{pred_len}days.npz')
            
            if not os.path.exists(result_file):
                print(f"⚠️  Results file not found: {result_file}")
                print(f"   Please run bitcoin_transformer.py with pred_len={pred_len} first")
                continue
            
            print(f"📂 Loading results for {pred_len}-day window...")
            
            # Load saved results
            data = np.load(result_file, allow_pickle=True)
            
            self.results[pred_len] = {
                'predictions': data['predictions'],
                'actuals': data['actuals'],
                'metrics': data['metrics'].item(),
                'test_dates': pd.to_datetime(data['test_dates']),
                'train_losses': data.get('train_losses', []),
                'val_losses': data.get('val_losses', [])
            }
            
            print(f"   ✅ Loaded {len(self.results[pred_len]['predictions'])} samples\n")
        
        if len(self.results) == 0:
            raise ValueError("No results loaded! Please train models first.")
        
        print(f"✅ Successfully loaded results for {len(self.results)} windows\n")
    
    def plot_comparison(self):
        """Create comprehensive comparison plots"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                        GENERATING COMPARISON PLOTS                            ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        self._plot_metrics_comparison()
        self._plot_predictions_comparison()
        self._plot_error_distribution_comparison()
        self._plot_training_comparison()
        
        print("✅ All comparison plots generated!\n")
    
    def _plot_metrics_comparison(self):
        """Compare key metrics across prediction windows"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Metrics Comparison Across Prediction Windows', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Prepare data
        pred_lens = sorted(self.results.keys())
        
        # 1. RMSE comparison
        ax = axes[0, 0]
        for pred_len in pred_lens:
            metrics = self.results[pred_len]['metrics']
            days = range(1, pred_len + 1)
            rmse_values = [metrics[f'Day_{i}']['RMSE'] for i in days]
            ax.plot(days, rmse_values, marker='o', label=f'{pred_len}-day', linewidth=2)
        ax.set_xlabel('Forecast Day', fontsize=11)
        ax.set_ylabel('RMSE ($)', fontsize=11)
        ax.set_title('RMSE vs Forecast Horizon', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. MAE comparison
        ax = axes[0, 1]
        for pred_len in pred_lens:
            metrics = self.results[pred_len]['metrics']
            days = range(1, pred_len + 1)
            mae_values = [metrics[f'Day_{i}']['MAE'] for i in days]
            ax.plot(days, mae_values, marker='o', label=f'{pred_len}-day', linewidth=2)
        ax.set_xlabel('Forecast Day', fontsize=11)
        ax.set_ylabel('MAE ($)', fontsize=11)
        ax.set_title('MAE vs Forecast Horizon', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. R² comparison
        ax = axes[0, 2]
        for pred_len in pred_lens:
            metrics = self.results[pred_len]['metrics']
            days = range(1, pred_len + 1)
            r2_values = [metrics[f'Day_{i}']['R2'] for i in days]
            ax.plot(days, r2_values, marker='o', label=f'{pred_len}-day', linewidth=2)
        ax.set_xlabel('Forecast Day', fontsize=11)
        ax.set_ylabel('R² Score', fontsize=11)
        ax.set_title('R² Score vs Forecast Horizon', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. MAPE comparison
        ax = axes[1, 0]
        for pred_len in pred_lens:
            metrics = self.results[pred_len]['metrics']
            days = range(1, pred_len + 1)
            mape_values = [metrics[f'Day_{i}']['MAPE'] for i in days]
            ax.plot(days, mape_values, marker='o', label=f'{pred_len}-day', linewidth=2)
        ax.set_xlabel('Forecast Day', fontsize=11)
        ax.set_ylabel('MAPE (%)', fontsize=11)
        ax.set_title('MAPE vs Forecast Horizon', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Directional Accuracy comparison
        ax = axes[1, 1]
        for pred_len in pred_lens:
            metrics = self.results[pred_len]['metrics']
            days = range(1, pred_len + 1)
            dir_acc_values = [metrics[f'Day_{i}']['Direction_Accuracy'] for i in days]
            ax.plot(days, dir_acc_values, marker='o', label=f'{pred_len}-day', linewidth=2)
        ax.set_xlabel('Forecast Day', fontsize=11)
        ax.set_ylabel('Directional Accuracy (%)', fontsize=11)
        ax.set_title('Directional Accuracy vs Forecast Horizon', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        
        # 6. Average metrics bar chart
        ax = axes[1, 2]
        avg_r2 = [np.mean([m['R2'] for m in self.results[pl]['metrics'].values()]) 
                  for pl in pred_lens]
        x = np.arange(len(pred_lens))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pred_lens)))
        bars = ax.bar(x, avg_r2, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Window', fontsize=11)
        ax.set_ylabel('Average R² Score', fontsize=11)
        ax.set_title('Average R² Score by Window', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{pl}-day' for pl in pred_lens])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_r2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('comparison_metrics.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: comparison_metrics.png")
        plt.close()
    
    def _plot_predictions_comparison(self):
        """Compare actual predictions showing real data vs predicted for each window"""
        pred_lens = sorted(self.results.keys())
        n_windows = len(pred_lens)
        
        # Create a figure with one subplot per prediction window
        fig, axes = plt.subplots(n_windows, 1, figsize=(20, 6 * n_windows))
        if n_windows == 1:
            axes = [axes]
        
        fig.suptitle('Bitcoin Price Forecasting - Real Data vs Predictions by Forecast Window', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for idx, pred_len in enumerate(pred_lens):
            ax = axes[idx]
            result = self.results[pred_len]
            
            # Plot actual prices (real data)
            ax.plot(result['test_dates'], result['actuals'][:, 0], 
                   label='Real Data', color='#2E86AB', linewidth=2.5, alpha=0.9)
            
            # Plot predicted prices
            ax.plot(result['test_dates'], result['predictions'][:, 0], 
                   label=f'Predicted ({pred_len}-day window)', 
                   color='#F18F01', linewidth=2, alpha=0.8, linestyle='--')
            
            # Calculate metrics for title
            r2 = r2_score(result['actuals'][:, 0], result['predictions'][:, 0])
            mae = mean_absolute_error(result['actuals'][:, 0], result['predictions'][:, 0])
            mape = np.mean(np.abs((result['actuals'][:, 0] - result['predictions'][:, 0]) / 
                                  result['actuals'][:, 0])) * 100
            
            ax.set_title(f'{pred_len}-Day Forecast Window | R²={r2:.3f} | MAE=${mae:,.0f} | MAPE={mape:.2f}%', 
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Bitcoin Price (USD)', fontsize=11)
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add shaded region for prediction error
            errors = result['predictions'][:, 0] - result['actuals'][:, 0]
            ax.fill_between(result['test_dates'], 
                           result['actuals'][:, 0], 
                           result['predictions'][:, 0],
                           alpha=0.2, color='red' if np.mean(errors) > 0 else 'green')
        
        plt.tight_layout()
        plt.savefig('comparison_predictions.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: comparison_predictions.png")
        plt.close()
        
        # Create an additional summary plot
        self._plot_predictions_summary()
    
    def _plot_predictions_summary(self):
        """Create a summary comparison on a single plot"""
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        pred_lens = sorted(self.results.keys())
        
        # Plot actual once
        result_first = self.results[pred_lens[0]]
        ax.plot(result_first['test_dates'], result_first['actuals'][:, 0], 
               label='Real Data', color='black', linewidth=3, alpha=0.9, zorder=10)
        
        # Plot predictions for each window
        colors = plt.cm.tab10(np.linspace(0, 0.8, len(pred_lens)))
        for idx, pred_len in enumerate(pred_lens):
            result = self.results[pred_len]
            r2 = r2_score(result['actuals'][:, 0], result['predictions'][:, 0])
            ax.plot(result['test_dates'], result['predictions'][:, 0], 
                   label=f'{pred_len}-day window (R²={r2:.3f})', 
                   color=colors[idx], linewidth=2, alpha=0.7, linestyle='--')
        
        ax.set_title('Bitcoin Price: Real Data vs Predictions (All Forecast Windows)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Bitcoin Price (USD)', fontsize=13)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig('comparison_predictions_summary.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: comparison_predictions_summary.png")
        plt.close()
    
    def _plot_error_distribution_comparison(self):
        """Compare error distributions across models"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Error Distribution Comparison', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        pred_lens = sorted(self.results.keys())
        
        # 1. Error histograms for Day 1
        ax = axes[0, 0]
        for pred_len in pred_lens:
            result = self.results[pred_len]
            errors = result['predictions'][:, 0] - result['actuals'][:, 0]
            ax.hist(errors, bins=30, alpha=0.5, label=f'{pred_len}-day', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error ($)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Day 1 Error Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Percentage error histograms for Day 1
        ax = axes[0, 1]
        for pred_len in pred_lens:
            result = self.results[pred_len]
            errors = result['predictions'][:, 0] - result['actuals'][:, 0]
            pct_errors = (errors / result['actuals'][:, 0]) * 100
            ax.hist(pct_errors, bins=30, alpha=0.5, label=f'{pred_len}-day', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Percentage Error (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Day 1 Percentage Error Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Box plot of errors by prediction window
        ax = axes[1, 0]
        error_data = []
        labels = []
        for pred_len in pred_lens:
            result = self.results[pred_len]
            errors = result['predictions'][:, 0] - result['actuals'][:, 0]
            error_data.append(errors)
            labels.append(f'{pred_len}d')
        
        bp = ax.boxplot(error_data, labels=labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0.2, 0.8, len(pred_lens)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Prediction Window', fontsize=11)
        ax.set_ylabel('Error ($)', fontsize=11)
        ax.set_title('Error Distribution by Window (Day 1)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Mean absolute error comparison
        ax = axes[1, 1]
        x = np.arange(len(pred_lens))
        mae_values = []
        for pred_len in pred_lens:
            result = self.results[pred_len]
            errors = result['predictions'][:, 0] - result['actuals'][:, 0]
            mae_values.append(np.mean(np.abs(errors)))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pred_lens)))
        bars = ax.bar(x, mae_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Window', fontsize=11)
        ax.set_ylabel('Mean Absolute Error ($)', fontsize=11)
        ax.set_title('MAE Comparison (Day 1)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{pl}-day' for pl in pred_lens])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, mae_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('comparison_errors.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: comparison_errors.png")
        plt.close()
    
    def _plot_training_comparison(self):
        """Compare training histories"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training History Comparison', 
                     fontsize=16, fontweight='bold')
        
        pred_lens = sorted(self.results.keys())
        
        # Training loss
        ax = axes[0]
        for pred_len in pred_lens:
            result = self.results[pred_len]
            epochs = range(1, len(result['train_losses']) + 1)
            ax.plot(epochs, result['train_losses'], label=f'{pred_len}-day', 
                   linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Validation loss
        ax = axes[1]
        for pred_len in pred_lens:
            result = self.results[pred_len]
            epochs = range(1, len(result['val_losses']) + 1)
            ax.plot(epochs, result['val_losses'], label=f'{pred_len}-day', 
                   linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss (MSE)', fontsize=12)
        ax.set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('comparison_training.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: comparison_training.png")
        plt.close()
    
    def print_summary_table(self):
        """Print summary comparison table"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                          SUMMARY COMPARISON TABLE                             ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        pred_lens = sorted(self.results.keys())
        
        print(f"{'Window':<12} {'Avg R²':>10} {'Avg MAE':>12} {'Avg MAPE':>10} "
              f"{'Avg Dir%':>10} {'Epochs':>8}")
        print("─" * 80)
        
        for pred_len in pred_lens:
            result = self.results[pred_len]
            metrics = result['metrics']
            
            avg_r2 = np.mean([m['R2'] for m in metrics.values()])
            avg_mae = np.mean([m['MAE'] for m in metrics.values()])
            avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
            avg_dir = np.mean([m['Direction_Accuracy'] for m in metrics.values()])
            n_epochs = len(result['train_losses'])
            
            print(f"{pred_len:>2}-day    │ "
                  f"{avg_r2:>9.4f} │ "
                  f"${avg_mae:>10,.2f} │ "
                  f"{avg_mape:>9.2f}% │ "
                  f"{avg_dir:>9.1f}% │ "
                  f"{n_epochs:>7}")
        
        print("─" * 80 + "\n")


def main():
    """Main execution - loads pre-trained results and creates comparison plots"""
    
    # Create comparator
    comparator = ForecastComparator(pred_lengths=[7, 30, 45])
    
    # Load pre-computed results from ventanas directory
    try:
        comparator.load_results(results_dir='ventanas')
    except ValueError as e:
        print(f"❌ {e}")
        print("\n💡 To generate results, run bitcoin_transformer.py with different pred_len values:")
        print("   Example: Modify CONFIG['pred_len'] to 7, 30, and 45 and run separately")
        return
    
    # Generate comparison plots
    comparator.plot_comparison()
    
    # Print summary
    comparator.print_summary_table()
    
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                        COMPARISON COMPLETE!                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
    
    print("📁 Generated comparison files:")
    print("   1. comparison_metrics.png - Detailed metrics comparison")
    print("   2. comparison_predictions.png - Individual window comparisons (Real vs Predicted)")
    print("   3. comparison_predictions_summary.png - All windows on one plot")
    print("   4. comparison_errors.png - Error distribution analysis")
    print("   5. comparison_training.png - Training history comparison")
    print()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

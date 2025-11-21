"""
Improved Bitcoin Time Series Transformer with Better Preprocessing and Training

Key Improvements:
- Train on log-returns instead of raw prices
- Use MinMaxScaler for better normalization
- Huber loss for robustness
- Learning rate warmup + cosine decay
- Walk-forward validation option
- Optimized sequence length (7-10 days)
- Better feature engineering

Usage:
python bitcoin_transformer.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from ETL module
from utils.etl import BitcoinDataLoader, BitcoinEDA

# Import Transformer architecture
from src.models.transformer_model import (
    TimeSeriesTransformer, 
    TimeSeriesDataset, 
    FeatureImportance, 
    RiskAnalyzer
)

# Import formatting utility
from utils.misc import print_box

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print_box("""\n
BITCOIN TIME SERIES WITH TRANSFORMERS - RETURNS-BASED FORECASTING
Production Implementation v2.0
""",vertical_padding=1)

# ============================================================================
# IMPROVED FEATURE ENGINEERING
# ============================================================================

class ImprovedFeatureEngineer:
    """Enhanced feature engineering focused on returns and volatility"""
    
    @staticmethod
    def create_features(df):
        """Create features based on returns"""
        print("🔧 Creating return-based features...")
        
        df = df.copy()
        
        # 1. Log returns (primary feature)
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volume changes
        df['Volume_Change'] = np.log(df['Volume'] / df['Volume'].shift(1))
        
        # 3. Volatility measures
        for window in [5, 10, 20]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window).std()
            df[f'Volume_Vol_{window}'] = df['Volume_Change'].rolling(window).std()
        
        # 4. Moving averages of returns
        for window in [5, 10, 20]:
            df[f'MA_Returns_{window}'] = df['Returns'].rolling(window).mean()
        
        # 5. Momentum
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Returns'].rolling(period).sum()
        
        # 6. RSI on returns
        for period in [14]:
            delta = df['Returns']
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # 7. Price range as percentage
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Drop NaN
        df = df.dropna()
        
        # Select features for modeling
        feature_cols = [
            'Returns', 'Volume_Change', 'Price_Range',
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'MA_Returns_5', 'MA_Returns_10', 'MA_Returns_20',
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            'RSI_14'
        ]
        
        print(f"   ✅ Created {len(feature_cols)} features")
        print(f"   ✅ Valid samples: {len(df)}\n")
        
        return df[feature_cols], df['Close']


# ============================================================================
# IMPROVED DATASET WITH RETURNS
# ============================================================================

class ReturnsDataset(Dataset):
    """Dataset for returns-based forecasting"""
    
    def __init__(self, features, prices, seq_len, pred_len):
        self.features = features
        self.prices = prices
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # Input: feature sequence
        x = self.features[idx:idx + self.seq_len]
        
        # Target: future returns (first feature column)
        y = self.features[idx + self.seq_len:idx + self.seq_len + self.pred_len, 0]
        
        # Also return last price for reconstruction
        last_price = self.prices[idx + self.seq_len - 1]
        
        return (torch.FloatTensor(x), 
                torch.FloatTensor(y),
                torch.FloatTensor([last_price]))


# ============================================================================
# IMPROVED TRAINER WITH HUBER LOSS
# ============================================================================

class ImprovedTrainer:
    """Enhanced training with Huber loss and better scheduling"""
    
    def __init__(self, model, device, learning_rate=0.001, warmup_epochs=5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                                     weight_decay=1e-5)
        
        # Learning rate scheduler with warmup
        self.warmup_epochs = warmup_epochs
        self.base_lr = learning_rate
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=learning_rate/10
        )
        self.best_val_loss = float('inf')
        self.epoch = 0
        
    def _adjust_learning_rate(self):
        """Warmup learning rate for first few epochs"""
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr * (self.epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs, patience=15):
        """Train with warmup and cosine annealing"""
        print_box("\nTRAINING (IMPROVED)")

        print(f"🚀 Training with Huber loss and learning rate scheduling")
        print(f"   Device: {self.device}")
        print(f"   Warmup epochs: {self.warmup_epochs}")
        print(f"   Total epochs: {epochs}\n")
        print("─" * 81)
        
        train_losses, val_losses = [], []
        patience_counter = 0
        
        for epoch in range(epochs):
            self.epoch = epoch
            self._adjust_learning_rate()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'transformer/best_transformer_model.pth')
                patience_counter = 0
                status = "✅"
            else:
                patience_counter += 1
                status = f"⏳ ({patience_counter}/{patience})"
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] │ "
                      f"Train: {train_loss:.6f} │ Val: {val_loss:.6f} │ "
                      f"LR: {current_lr:.2e} │ {status}")
            
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        print("─" * 81)
        print(f"✅ Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        
        return train_losses, val_losses


# ============================================================================
# IMPROVED EVALUATION
# ============================================================================

class ImprovedEvaluator:
    """Evaluation with price reconstruction from returns"""
    
    @staticmethod
    def evaluate(model, test_loader, device, scaler):
        """Evaluate and reconstruct prices from returns"""
        print_box("\nEVALUATION (IMPROVED)")

        model.eval()
        pred_returns, actual_returns, last_prices = [], [], []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_last_price in test_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                
                pred_returns.append(output.cpu().numpy())
                actual_returns.append(batch_y.numpy())
                last_prices.append(batch_last_price.numpy())
        
        pred_returns = np.concatenate(pred_returns, axis=0)
        actual_returns = np.concatenate(actual_returns, axis=0)
        last_prices = np.concatenate(last_prices, axis=0)
        
        # Denormalize returns
        pred_returns_denorm = scaler.inverse_transform(pred_returns)
        actual_returns_denorm = scaler.inverse_transform(actual_returns)
        
        # Reconstruct prices from returns
        pred_prices = ImprovedEvaluator._reconstruct_prices(
            pred_returns_denorm, last_prices
        )
        actual_prices = ImprovedEvaluator._reconstruct_prices(
            actual_returns_denorm, last_prices
        )
        
        # Calculate metrics
        metrics = ImprovedEvaluator._calculate_metrics(pred_prices, actual_prices)
        ImprovedEvaluator._print_metrics(metrics)
        
        return pred_prices, actual_prices, metrics
    
    @staticmethod
    def _reconstruct_prices(returns, last_prices):
        """Reconstruct prices from log returns"""
        prices = np.zeros_like(returns)
        
        for i in range(len(returns)):
            current_price = last_prices[i, 0]
            for j in range(returns.shape[1]):
                # P_t = P_{t-1} * exp(r_t)
                current_price = current_price * np.exp(returns[i, j])
                prices[i, j] = current_price
        
        return prices
    
    @staticmethod
    def _calculate_metrics(predictions, actuals):
        """Calculate metrics per forecast day"""
        metrics = {}
        pred_len = predictions.shape[1]
        
        for i in range(pred_len):
            pred = predictions[:, i]
            actual = actuals[:, i]
            
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
            # Directional accuracy
            actual_dir = np.sign(np.diff(np.concatenate([[actual[0]], actual])))
            pred_dir = np.sign(np.diff(np.concatenate([[pred[0]], pred])))
            dir_acc = np.mean(actual_dir == pred_dir) * 100
            
            metrics[f'Day_{i+1}'] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'Dir_Acc': dir_acc
            }
        
        return metrics
    
    @staticmethod
    def _print_metrics(metrics):
        """Print evaluation metrics"""
        print("📈 EVALUATION METRICS (Price Reconstruction)")
        print("─" * 81)
        print(f"{'Forecast':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} "
              f"{'MAPE':>10} {'Dir%':>10}")
        print("─" * 81)
        
        for day, m in metrics.items():
            print(f"{day:<12} "
                  f"${m['RMSE']:>9,.2f} "
                  f"${m['MAE']:>9,.2f} "
                  f"{m['R2']:>9.4f} "
                  f"{m['MAPE']:>9.2f}% "
                  f"{m['Dir_Acc']:>9.1f}%")
        
        print("─" * 81)
        
        # Averages
        avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])
        avg_mae = np.mean([m['MAE'] for m in metrics.values()])
        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
        avg_dir = np.mean([m['Dir_Acc'] for m in metrics.values()])
        
        print(f"{'AVERAGE':<12} "
              f"${avg_rmse:>9,.2f} "
              f"${avg_mae:>9,.2f} "
              f"{avg_r2:>9.4f} "
              f"{avg_mape:>9.2f}% "
              f"{avg_dir:>9.1f}%")
        print("─" * 81 + "\n")
    
    @staticmethod
    def plot_predictions(predictions, actuals, save_path='transformer/results/03_predictions.png'):
        """Plot price predictions"""
        pred_len = min(predictions.shape[1], 4)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(pred_len):
            ax = axes[i]
            
            x = np.arange(len(predictions))
            ax.plot(x, actuals[:, i], label='Actual', linewidth=2, 
                   alpha=0.8, color='#2E86AB')
            ax.plot(x, predictions[:, i], label='Predicted', linewidth=2, 
                   alpha=0.8, color='#F18F01', linestyle='--')
            
            r2 = r2_score(actuals[:, i], predictions[:, i])
            mae = mean_absolute_error(actuals[:, i], predictions[:, i])
            
            ax.set_title(f'Day {i+1} Forecast (R²={r2:.3f}, MAE=${mae:,.0f})', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Sample', fontsize=11)
            ax.set_ylabel('Bitcoin Price (USD)', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_error_analysis(predictions, actuals, save_path='transformer/results/04_error_analysis.png'):
        """Analyze prediction errors"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Use first-day predictions for detailed analysis
        pred = predictions[:, 0]
        actual = actuals[:, 0]
        errors = pred - actual
        pct_errors = (errors / actual) * 100
        
        # 1. Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(actual, pred, alpha=0.5, s=30)
        min_val, max_val = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Price (USD)', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Price (USD)', fontsize=11)
        axes[0, 0].set_title('Predicted vs Actual (Day 1)', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Error distribution
        axes[0, 1].hist(errors, bins=50, color='#C73E1D', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[0, 1].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: ${errors.mean():,.0f}')
        axes[0, 1].set_xlabel('Prediction Error (USD)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Error Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Errors over time
        x = np.arange(len(errors))
        axes[1, 0].plot(x, errors, linewidth=1, alpha=0.7, color='#C73E1D')
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].fill_between(x, 0, errors, alpha=0.3, color='#C73E1D')
        axes[1, 0].set_xlabel('Sample', fontsize=11)
        axes[1, 0].set_ylabel('Prediction Error (USD)', fontsize=11)
        axes[1, 0].set_title('Errors Over Time', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Percentage error distribution
        axes[1, 1].hist(pct_errors, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[1, 1].axvline(pct_errors.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {pct_errors.mean():.2f}%')
        axes[1, 1].set_xlabel('Percentage Error (%)', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Percentage Error Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, save_path='transformer/results/05_training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
                markersize=4, alpha=0.7)
        ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', 
                markersize=4, alpha=0.7)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (Huber)', fontsize=12)
        ax1.set_title('Training History', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
                markersize=4, alpha=0.7)
        ax2.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', 
                markersize=4, alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (Huber, log scale)', fontsize=12)
        ax2.set_title('Training History (Log Scale)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with improvements"""
    
    CONFIG = {
        # Data parameters
        'use_real_data': True,
        'start_date': '2020-01-01',
        'end_date': None,  # None = today
        
        # Model parameters
        'seq_len': 10,          # Lookback window -> Shorter sequence (7-10 days as suggested)
        'pred_len': 45,         # Forecast horizon (Predict 7 days ahead) 
        'd_model': 128,         # Model dimension
        'nhead': 8,             # Number of attention heads
        'num_layers': 2,        # Number of transformer layers -> Reduced from 3 to 2
        'dim_feedforward': 512, # Feedforward dimension
        'dropout': 0.1,         # Dropout rate
        
        # Training parameters
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0005,  # -> Slightly lower
        'warmup_epochs': 5,
        'patience': 15,
        
        # Data split
        'train_ratio': 0.7,
        'val_ratio': 0.15,

        # Future forecast
        # 'forecast_days': 30
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device} | PyTorch version: {torch.__version__}\n")
    
    # ETL
    df, is_real = BitcoinDataLoader.load_data(
        use_real_data=CONFIG['use_real_data'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date']
    )
    
    # EDA
    eda = BitcoinEDA(df, is_real_data=is_real)
    eda.run_full_eda()

    # Feature engineering
    features_df, prices = ImprovedFeatureEngineer.create_features(df)
    
    # Normalize features with MinMaxScaler (better for returns)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler.fit_transform(features_df.values)
    prices_array = prices.values
    
    # Create separate scaler for returns only (first column)
    returns_scaler = MinMaxScaler(feature_range=(-1, 1)) # previously StandardScaler
    returns_scaler.fit(features_df[['Returns']].values)
    
    # Split data
    n = len(scaled_features)
    train_size = int(n * CONFIG['train_ratio'])
    val_size = int(n * CONFIG['val_ratio'])
    
    train_features = scaled_features[:train_size]
    train_prices = prices_array[:train_size]
    
    val_features = scaled_features[train_size:train_size + val_size]
    val_prices = prices_array[train_size:train_size + val_size]
    
    test_features = scaled_features[train_size + val_size:]
    test_prices = prices_array[train_size + val_size:]
    
    print(f"📊 Data split:")
    print(f"   Training set:   {len(train_features)} samples ({CONFIG['train_ratio']*100:.0f}%)")
    print(f"   Validation set: {len(val_features)} samples ({CONFIG['val_ratio']*100:.0f}%)")
    print(f"   Test set:       {len(test_features)} samples ({(1-CONFIG['train_ratio']-CONFIG['val_ratio'])*100:.0f}%)")
    print(f"   Total:          {n} samples\n")
    
    # Create datasets
    train_dataset = ReturnsDataset(train_features, train_prices, 
                                   CONFIG['seq_len'], CONFIG['pred_len'])
    val_dataset = ReturnsDataset(val_features, val_prices,
                                 CONFIG['seq_len'], CONFIG['pred_len'])
    test_dataset = ReturnsDataset(test_features, test_prices,
                                  CONFIG['seq_len'], CONFIG['pred_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Build model
    n_features = scaled_features.shape[1]
    model = TimeSeriesTransformer(
        input_dim=n_features,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        pred_len=CONFIG['pred_len'],
        dropout=CONFIG['dropout']
    )
    
    print(f"🧠 Transformer Architecture:")
    print(f"   Input features:      {n_features}")
    print(f"   Sequence length:     {CONFIG['seq_len']} days")
    print(f"   Prediction horizon:  {CONFIG['pred_len']} days")
    print(f"   Model dimension:     {CONFIG['d_model']}")
    print(f"   Layers:              {CONFIG['num_layers']}")
    print(f"   Total parameters:    {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = ImprovedTrainer(
        model, device,
        learning_rate=CONFIG['learning_rate'],
        warmup_epochs=CONFIG['warmup_epochs']
    )
    
    train_losses, val_losses = trainer.fit(
        train_loader, val_loader,
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience']
    )
    
    # Evaluate
    model.load_state_dict(torch.load('transformer/best_transformer_model.pth'))
    predictions, actuals, metrics = ImprovedEvaluator.evaluate(
        model, test_loader, device, returns_scaler
    )
    
    # Plot predictions
    ImprovedEvaluator.plot_predictions(predictions, actuals)
    
    # Plot error analysis
    ImprovedEvaluator.plot_error_analysis(predictions, actuals)
    
    # Plot training history
    ImprovedEvaluator.plot_training_history(train_losses, val_losses)
    
    # ========================================
    # SUMMARY
    # ========================================
    print_box("\nIMPROVEMENTS SUMMARY")

    print("✅ Applied improvements:")
    print("   1. ✅ Train on log-returns instead of prices")
    print("   2. ✅ Use MinMaxScaler for better normalization")
    print("   3. ✅ Huber loss for robustness to outliers")
    print("   4. ✅ Learning rate warmup + cosine annealing")
    print("   5. ✅ Reduced sequence length (10 days)")
    print("   6. ✅ Reduced model layers (2 instead of 3)")
    print("   7. ✅ Enhanced feature engineering")
    print("   8. ✅ Gradient clipping for stability"+"\n")
    
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
    
    print(f"📊 Final Results:")
    print(f"   Average R²:   {avg_r2:.4f}")
    print(f"   Average MAPE: {avg_mape:.2f}%\n")

    print("🏆 Pipeline completed successfully!\n")

    print_box() # Line break
    print("📈 Thank you for using Bitcoin LSTM Forecasting System!")
    
    # return {
    #     'model': model,
    #     'scaler': scaler,
    #     'predictions': predictions,
    #     'actuals': actuals,
    #     'metrics': metrics,
    #     'forecasts': forecasts,
    #     'config': CONFIG,
    #     'feature_cols': feature_cols,
    #     'test_data': test_data
    # }



if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        results = main()
        
        # # Optional: Feature importance analysis
        # print_box() # Line break
        # response = input("Would you like to perform feature importance analysis? (y/n): ")
        # if response.lower() == 'y':
        #     # Use a subset of test data for feature importance
        #     test_subset = results['test_data'][:min(200, len(results['test_data']))]
        #     test_subset_dataset = TimeSeriesDataset(
        #         test_subset,
        #         results['config']['seq_len'],
        #         results['config']['pred_len']
        #     )
        #     test_subset_loader = DataLoader(
        #         test_subset_dataset,
        #         batch_size=32,
        #         shuffle=False
        #     )
            
        #     importance_scores = FeatureImportance.calculate_importance(
        #         results['model'],
        #         test_subset_loader,
        #         torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        #         results['feature_cols'],
        #         n_repeats=5,
        #         max_batches=10
        #     )
        
        # # Optional: Risk analysis
        # print_box() # Line break
        # response = input("Would you like to perform risk analysis? (y/n): ")
        # if response.lower() == 'y':
        #     # Get historical returns
        #     historical_returns = np.diff(results['actuals'][:, 0]) / results['actuals'][:-1, 0]
            
        #     risk_metrics = RiskAnalyzer.analyze_risk(
        #         results['forecasts'],
        #         results['actuals'][-1, 0],  # Last actual price
        #         historical_returns
        #     )
        
        # print_box("ALL ANALYSES COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

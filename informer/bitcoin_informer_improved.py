"""
Improved Informer for Bitcoin Forecasting with Better Preprocessing and Training

Key Improvements (same as Transformer improved):
- Train on log-returns instead of raw prices
- Use MinMaxScaler for better normalization
- Huber loss for robustness
- Learning rate warmup + cosine decay
- Optimized sequence length (10 days)
- Better feature engineering

Usage:
python informer/bitcoin_informer_improved.py
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

# Import from original informer
import sys
#sys.path.append('..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer.bitcoin_transformer import BitcoinDataLoader

# Import Informer architecture from bitcoin_informer
from informer.bitcoin_informer import Informer

plt.style.use('seaborn-v0_8-darkgrid')
torch.manual_seed(42)
np.random.seed(42)

print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║        IMPROVED INFORMER - RETURNS-BASED FORECASTING                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# IMPROVED FEATURE ENGINEERING (same as transformer improved)
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
# IMPROVED DATASET (same as transformer improved)
# ============================================================================

class InformerReturnsDataset(Dataset):
    """Dataset for Informer with returns-based forecasting"""
    
    def __init__(self, features, prices, seq_len, label_len, pred_len):
        self.features = features
        self.prices = prices
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # Input encoder: feature sequence
        x_enc = self.features[idx:idx + self.seq_len]
        
        # Input decoder: starts with label_len from end of encoder, then zeros
        x_dec_start = self.features[idx + self.seq_len - self.label_len:idx + self.seq_len]
        x_dec_zeros = np.zeros((self.pred_len, self.features.shape[1]))
        x_dec = np.vstack([x_dec_start, x_dec_zeros])
        
        # Target: future returns (first feature column)
        y = self.features[idx + self.seq_len:idx + self.seq_len + self.pred_len, 0]
        
        # Last price for reconstruction
        last_price = self.prices[idx + self.seq_len - 1]
        
        return (torch.FloatTensor(x_enc), 
                torch.FloatTensor(x_dec),
                torch.FloatTensor(y),
                torch.FloatTensor([last_price]))


# ============================================================================
# IMPROVED TRAINER (same as transformer improved)
# ============================================================================

class ImprovedInformerTrainer:
    """Enhanced training with Huber loss and better scheduling"""
    
    def __init__(self, model, device, learning_rate=0.001, warmup_epochs=5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                                     weight_decay=1e-5)
        
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
        
        for x_enc, x_dec, target, _ in train_loader:
            x_enc = x_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x_enc, x_dec)
            output = output.squeeze(-1)  # Remove last dimension: (batch, pred_len, 1) -> (batch, pred_len)
            loss = self.criterion(output, target)
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
            for x_enc, x_dec, target, _ in val_loader:
                x_enc = x_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                target = target.to(self.device)
                output = self.model(x_enc, x_dec)
                output = output.squeeze(-1)  # Remove last dimension: (batch, pred_len, 1) -> (batch, pred_len)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs, patience=15):
        """Train with warmup and cosine annealing"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                          TRAINING (IMPROVED INFORMER)                         ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        print(f"🚀 Training Informer with Huber loss and learning rate scheduling")
        print(f"   Device: {self.device}")
        print(f"   Warmup epochs: {self.warmup_epochs}")
        print(f"   Total epochs: {epochs}\n")
        print("─" * 80)
        
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
                torch.save(self.model.state_dict(), 'informer/best_improved_informer.pth')
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
        
        print("─" * 80)
        print(f"✅ Best validation loss: {self.best_val_loss:.6f}\n")
        
        return train_losses, val_losses


# ============================================================================
# IMPROVED EVALUATION (same as transformer improved)
# ============================================================================

class ImprovedInformerEvaluator:
    """Evaluation with price reconstruction from returns"""
    
    @staticmethod
    def evaluate(model, test_loader, device, scaler):
        """Evaluate and reconstruct prices from returns"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                         EVALUATION (IMPROVED INFORMER)                        ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        model.eval()
        pred_returns, actual_returns, last_prices = [], [], []
        
        with torch.no_grad():
            for x_enc, x_dec, target, last_price in test_loader:
                x_enc = x_enc.to(device)
                x_dec = x_dec.to(device)
                output = model(x_enc, x_dec)
                output = output.squeeze(-1)  # Remove last dimension: (batch, pred_len, 1) -> (batch, pred_len)
                
                pred_returns.append(output.cpu().numpy())
                actual_returns.append(target.numpy())
                last_prices.append(last_price.numpy())
        
        pred_returns = np.concatenate(pred_returns, axis=0)
        actual_returns = np.concatenate(actual_returns, axis=0)
        last_prices = np.concatenate(last_prices, axis=0)
        
        # Denormalize returns
        pred_returns_denorm = scaler.inverse_transform(pred_returns)
        actual_returns_denorm = scaler.inverse_transform(actual_returns)
        
        # Reconstruct prices from returns
        pred_prices = ImprovedInformerEvaluator._reconstruct_prices(
            pred_returns_denorm, last_prices
        )
        actual_prices = ImprovedInformerEvaluator._reconstruct_prices(
            actual_returns_denorm, last_prices
        )
        
        # Calculate metrics
        metrics = ImprovedInformerEvaluator._calculate_metrics(pred_prices, actual_prices)
        ImprovedInformerEvaluator._print_metrics(metrics)
        
        return pred_prices, actual_prices, metrics
    
    @staticmethod
    def _reconstruct_prices(returns, last_prices):
        """Reconstruct prices from log returns"""
        prices = np.zeros_like(returns)
        
        for i in range(len(returns)):
            current_price = last_prices[i, 0]
            for j in range(returns.shape[1]):
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
        print("─" * 90)
        print(f"{'Forecast':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} "
              f"{'MAPE':>10} {'Dir%':>10}")
        print("─" * 90)
        
        for day, m in metrics.items():
            print(f"{day:<12} "
                  f"${m['RMSE']:>9,.2f} "
                  f"${m['MAE']:>9,.2f} "
                  f"{m['R2']:>9.4f} "
                  f"{m['MAPE']:>9.2f}% "
                  f"{m['Dir_Acc']:>9.1f}%")
        
        print("─" * 90)
        
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
        print("─" * 90 + "\n")
    
    @staticmethod
    def plot_predictions(predictions, actuals, save_path='informer/informer/improved_predictions.png'):
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
        print(f"   ✅ Saved: {save_path}\n")
        plt.close()
    
    @staticmethod
    def plot_error_analysis(predictions, actuals, save_path='informer/informer/improved_error_analysis.png'):
        """Analyze prediction errors"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        pred = predictions[:, 0]
        actual = actuals[:, 0]
        errors = pred - actual
        pct_errors = (errors / actual) * 100
        
        # 1. Scatter plot
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
        print(f"   ✅ Saved: {save_path}\n")
        plt.close()
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, save_path='informer/informer/improved_training_history.png'):
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
        ax1.set_title('Training History - Informer', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
                markersize=4, alpha=0.7)
        ax2.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', 
                markersize=4, alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (Huber, log scale)', fontsize=12)
        ax2.set_title('Training History (Log Scale) - Informer', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}\n")
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with improvements"""
    
    CONFIG = {
        'use_real_data': True,
        'start_date': '2020-01-01',
        'end_date': None,
        
        'seq_len': 10,          # Shorter sequence
        'label_len': 5,         # Informer-specific
        'pred_len': 7,          # Predict 7 days ahead
        'd_model': 128,         # Slightly reduced
        'n_heads': 8,
        'e_layers': 2,          # Encoder layers
        'd_layers': 1,          # Decoder layers
        'd_ff': 512,
        'factor': 5,            # ProbSparse factor
        'dropout': 0.1,
        
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0005,
        'warmup_epochs': 5,
        'patience': 15,
        
        'train_ratio': 0.7,
        'val_ratio': 0.15,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device}\n")
    
    # Load data
    df, is_real = BitcoinDataLoader.load_data(
        use_real_data=CONFIG['use_real_data'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date']
    )
    
    print(f"✅ Loaded {len(df)} days of {'real' if is_real else 'synthetic'} data\n")
    
    # Feature engineering
    features_df, prices = ImprovedFeatureEngineer.create_features(df)
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler.fit_transform(features_df.values)
    prices_array = prices.values
    
    # Create separate scaler for returns only
    returns_scaler = MinMaxScaler(feature_range=(-1, 1))
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
    print(f"   Training:   {len(train_features)} samples")
    print(f"   Validation: {len(val_features)} samples")
    print(f"   Test:       {len(test_features)} samples\n")
    
    # Create datasets
    train_dataset = InformerReturnsDataset(
        train_features, train_prices,
        CONFIG['seq_len'], CONFIG['label_len'], CONFIG['pred_len']
    )
    val_dataset = InformerReturnsDataset(
        val_features, val_prices,
        CONFIG['seq_len'], CONFIG['label_len'], CONFIG['pred_len']
    )
    test_dataset = InformerReturnsDataset(
        test_features, test_prices,
        CONFIG['seq_len'], CONFIG['label_len'], CONFIG['pred_len']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Build model
    n_features = scaled_features.shape[1]
    model = Informer(
        enc_in=n_features, dec_in=n_features, c_out=1,  # Output 1 feature (returns)
        seq_len=CONFIG['seq_len'],
        label_len=CONFIG['label_len'],
        pred_len=CONFIG['pred_len'],
        factor=CONFIG['factor'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        e_layers=CONFIG['e_layers'],
        d_layers=CONFIG['d_layers'],
        d_ff=CONFIG['d_ff'],
        dropout=CONFIG['dropout']
    )
    
    print(f"🧠 Informer Architecture:")
    print(f"   Input features:      {n_features}")
    print(f"   Sequence length:     {CONFIG['seq_len']} days")
    print(f"   Label length:        {CONFIG['label_len']} days")
    print(f"   Prediction horizon:  {CONFIG['pred_len']} days")
    print(f"   Model dimension:     {CONFIG['d_model']}")
    print(f"   Encoder layers:      {CONFIG['e_layers']}")
    print(f"   Decoder layers:      {CONFIG['d_layers']}")
    print(f"   Total parameters:    {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    trainer = ImprovedInformerTrainer(
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
    os.makedirs('informer', exist_ok=True)
    model.load_state_dict(torch.load('informer/best_improved_informer.pth'))
    predictions, actuals, metrics = ImprovedInformerEvaluator.evaluate(
        model, test_loader, device, returns_scaler
    )
    
    # Plot predictions
    ImprovedInformerEvaluator.plot_predictions(predictions, actuals)
    
    # Plot error analysis
    ImprovedInformerEvaluator.plot_error_analysis(predictions, actuals)
    
    # Plot training history
    ImprovedInformerEvaluator.plot_training_history(train_losses, val_losses)
    
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                        IMPROVEMENTS SUMMARY                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
    
    print("✅ Applied improvements (same as Transformer):")
    print("   1. ✅ Train on log-returns instead of prices")
    print("   2. ✅ Use MinMaxScaler for better normalization")
    print("   3. ✅ Huber loss for robustness to outliers")
    print("   4. ✅ Learning rate warmup + cosine annealing")
    print("   5. ✅ Reduced sequence length (10 days)")
    print("   6. ✅ Reduced model layers (2 encoder, 1 decoder)")
    print("   7. ✅ Enhanced feature engineering")
    print("   8. ✅ Gradient clipping for stability")
    print("   9. ✅ ProbSparse attention for efficiency")
    print()
    
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
    
    print(f"📊 Final Results:")
    print(f"   Average R²:   {avg_r2:.4f}")
    print(f"   Average MAPE: {avg_mape:.2f}%")
    print()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

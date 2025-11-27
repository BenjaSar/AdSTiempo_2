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
python informer/bitcoin_informer.py
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
from src.etl import BitcoinDataLoader, ImprovedFeatureEngineer

# Import Informer architecture
from src.models.informer_model import (
    Informer,
    InformerReturnsDataset,
    ImprovedInformerTrainer,
    ImprovedInformerEvaluator
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
INFORMER: EFFICIENT LONG-SEQUENCE FORECASTING - RETURNS-BASED FORECASTING
Production Implementation v2.0
""",vertical_padding=1)

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
        'seq_len': 10,          # Lookback window -> Shorter for returns
        'label_len': 5,         # Label length (Informer-specific) -> Reduced
        'pred_len': 7,          # Forecast horizon (Predict 7 days ahead) 
        'd_model': 128,         # Model dimension -> Slightly reduced
        'n_heads': 8,           # Number of attention heads
        'e_layers': 2,          # Encoder layers
        'd_layers': 1,          # Decoder layers
        'd_ff': 512,            # Feedforward dimension -> Reduced for efficiency
        'factor': 5,            # ProbSparse factor
        'dropout': 0.1,         # Dropout rate

        # Training parameters
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0005, # -> Slightly lower
        'warmup_epochs': 5,
        'patience': 15,
        
        # Data split
        'train_ratio': 0.7,
        'val_ratio': 0.15,

        # Future forecast
        'forecast_days': 30
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device} | PyTorch version: {torch.__version__}\n")
    
    # ETL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df, is_real = BitcoinDataLoader.load_data(
        use_real_data=CONFIG['use_real_data'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date']
    )
    
    # Feature engineering
    features_df, prices = ImprovedFeatureEngineer.create_features(df)
    
    print(f"📊 Selected {len(features_df)} features for modeling\n")
    
    # NORMALIZE & SPLIT ~~~~~~~~~~~~~~~~~~~~~~~~~~
    print_box("DATA PREPARATION & SPLITTING")
    
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
    print(f"   Training set:   {len(train_features)} samples ({CONFIG['train_ratio']*100:.0f}%)")
    print(f"   Validation set: {len(val_features)} samples ({CONFIG['val_ratio']*100:.0f}%)")
    print(f"   Test set:       {len(test_features)} samples ({(1-CONFIG['train_ratio']-CONFIG['val_ratio'])*100:.0f}%)")
    print(f"   Total:          {n} samples\n")
    
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
    
    # Dataloaders
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
    
    # BUILD MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    model.load_state_dict(torch.load('informer/best_informer_model.pth'))
    predictions, actuals, metrics = ImprovedInformerEvaluator.evaluate(
        model, test_loader, device, returns_scaler
    )
    
    # RESULTS ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot results
    ImprovedInformerEvaluator.plot_predictions(predictions, actuals)
    ImprovedInformerEvaluator.plot_error_analysis(predictions, actuals)
    ImprovedInformerEvaluator.plot_training_history(train_losses, val_losses)
    
    # Process summary
    print_box("\nIMPROVEMENTS SUMMARY")

    print("✅ Applied improvements (same as Transformer):")
    print("   1. ✅ Train on log-returns instead of prices")
    print("   2. ✅ Use MinMaxScaler for better normalization")
    print("   3. ✅ Huber loss for robustness to outliers")
    print("   4. ✅ Learning rate warmup + cosine annealing")
    print("   5. ✅ Reduced sequence length (10 days)")
    print("   6. ✅ Reduced model layers (2 encoder, 1 decoder)")
    print("   7. ✅ Enhanced feature engineering")
    print("   8. ✅ Gradient clipping for stability")
    print("   9. ✅ ProbSparse attention for efficiency"+"\n")
    
    # Metrics
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
    
    print(f"📊 Final Results:")
    print(f"   Average R²:   {avg_r2:.4f}")
    print(f"   Average MAPE: {avg_mape:.2f}%\n")

    print("🏆 Pipeline completed successfully!\n")

    print_box() # Line break
    print("📈 Thank you for using Bitcoin LSTM Forecasting System!")    

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Set matplotlib backend
    import matplotlib 
    matplotlib.use('Agg') # Non-interactive backend
    
    # Run main pipeline
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
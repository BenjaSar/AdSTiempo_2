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
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Deep learning libraries
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from ETL module
from src.etl import (
    BitcoinDataLoader,
    ImprovedFeatureEngineer,
    ReturnsDataset,
    BitcoinEDA
)

# Import Transformer architecture
from src.models.transformer_model import (
    Transformer,
    ImprovedTrainer,
    ImprovedEvaluator,
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
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with improvements"""
    
    # CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    
    # EDA (Optional)
    eda = BitcoinEDA(df, is_real_data=is_real)
    eda.run_full_eda()

    # Feature engineering 
    features_df, prices = ImprovedFeatureEngineer.create_features(df)
    
    print(f"📊 Selected {len(features_df)} features for modeling\n")
    
    # NORMALIZE & SPLIT ~~~~~~~~~~~~~~~~~~~~~~~~~~
    print_box("DATA PREPARATION & SPLITTING")
    
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
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Build model
    n_features = scaled_features.shape[1]
    model = Transformer(
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
    
    # BUILD MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    
    # RESULTS ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot results
    ImprovedEvaluator.plot_predictions(predictions, actuals)
    ImprovedEvaluator.plot_error_analysis(predictions, actuals)
    ImprovedEvaluator.plot_training_history(train_losses, val_losses)
    
    # Process summary
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
    
    # Metrics
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
    
    print(f"📊 Final Results:")
    print(f"   Average R²:   {avg_r2:.4f}")
    print(f"   Average MAPE: {avg_mape:.2f}%\n")

    print("🏆 Pipeline completed successfully!\n")

    print_box() # Line break
    print("📈 Thank you for using Bitcoin Forecasting System!")
    
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

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Set matplotlib backend
    import matplotlib 
    matplotlib.use('Agg') # Non-interactive backend
    
    # Run main pipeline
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

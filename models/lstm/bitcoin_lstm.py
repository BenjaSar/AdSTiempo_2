"""
LSTM: Traditional Time-Series architechture for Forecasting (Benchmark)
Production-Ready Implementation with Bitcoin Price Prediction

Usage:
python lstm/bitcoin_lstm.py
"""

import numpy as np
import pandas as pd
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

# Import Informer architecture
from src.models.lstm_model import (
    LSTMModel, 
    ModelTrainer,
    Evaluator,
    FutureForecaster,
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
print()
print_box("""\n
BITCOIN TIME SERIES FORECASTING WITH LSTM
Production Implementation v1.0
""",vertical_padding=1)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CONFIG = {
        # Data parameters
        'use_real_data': True,
        'start_date': '2020-01-01',
        'end_date': None,  # None = today
        
        # Model parameters
        'seq_len': 60,          # Lookback window
        'pred_len': 45,         # Forecast horizon
        'hidden_dim': 128,      # LSTM hidden dimension (renamed from d_model)
        'num_layers': 3,        # Number of LSTM layers
        'dropout': 0.1,         # Dropout rate
        # TRANSFORMER SPECIFIC PARAMS 
        # 'd_model': 128,         # Model dimension
        # 'nhead': 8,             # Number of attention heads
        # 'dim_feedforward': 512, # Feedforward dimension
        
        # Training parameters
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
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
    df_features, prices = ImprovedFeatureEngineer.create_features(df)
    
    # Select features (exclude categorical time features)
    exclude_cols = ['DayOfWeek', 'Month', 'Quarter', 'DayOfMonth', 
                   'DayOfYear', 'WeekOfYear']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    print(f"📊 Selected {len(feature_cols)} features for modeling\n")
    
    # NORMALIZE & SPLIT ~~~~~~~~~~~~~~~~~~~~~~~~~~
    print_box("DATA PREPARATION & SPLITTING")
    
    # Normalize features with MinMaxScaler (better for returns)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler.fit_transform(df_features.values)
    prices_array = prices.values
    
    # Create separate scaler for returns only (first column)
    returns_scaler = MinMaxScaler(feature_range=(-1, 1)) # previously StandardScaler
    returns_scaler.fit(df_features[['Returns']].values)
    
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
    model = LSTMModel(
        input_dim=n_features,
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        pred_len=CONFIG['pred_len'],
        dropout=CONFIG['dropout']
    )
    
    print(f"🧠 Model architecture:")
    print(f"   Input dimension:     {len(feature_cols)}")
    print(f"   Hidden dimension:    {CONFIG['hidden_dim']}")
    print(f"   LSTM layers:         {CONFIG['num_layers']}")
    print(f"   Output dimension:    {CONFIG['pred_len']}")
    print(f"   Total parameters:    {sum(p.numel() for p in model.parameters()):,}\n")
    
    # BUILD MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train
    trainer = ModelTrainer(
        model, device, 
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    train_losses, val_losses = trainer.fit(
        train_loader, val_loader, 
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience']
    )
    
    # Evaluate
    model.load_state_dict(torch.load('lstm/best_lstm_model.pth'))
    
    close_idx = feature_cols.index('Returns')
    predictions, actuals, metrics = Evaluator.evaluate(
        model, test_loader, device, scaler, close_idx
    )

    # FORECASTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    last_sequence = test_features[-CONFIG['seq_len']:]
    
    forecasts = FutureForecaster.forecast_recursive(
        model, last_sequence, scaler, close_idx, 
        n_days=CONFIG['forecast_days'], device=device
    )
    
    # Prepare visualization data
    historical_prices = prices.values[-200:]
    historical_dates = df_features.index[-200:]
    
    last_date = df_features.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=CONFIG['forecast_days'],
        freq='D'
    )
    
    FutureForecaster.plot_forecast(
        historical_prices, historical_dates,
        forecasts, forecast_dates
    )
    
    # RESULTS ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get test dates
    test_start_idx = train_size + val_size + CONFIG['seq_len']
    test_dates = df_features.index[test_start_idx:test_start_idx + len(predictions)]
    
    # Plot results
    Evaluator.plot_predictions(predictions, actuals, test_dates)
    Evaluator.plot_error_analysis(predictions, actuals, test_dates)
    
    # Process summary
    print_box("\nEXECUTION SUMMARY")

    print("📁 Generated files:")
    print("   1. 01_comprehensive_eda.png - Comprehensive exploratory data analysis")
    print("   2. 02_advanced_analysis.png - Advanced statistical analysis")
    print("   3. 03_training_history.png - Model training progress")
    print("   4. 04_predictions.png - Test set predictions")
    print("   5. 05_error_analysis.png - Prediction error analysis")
    print("   6. 06_future_forecast.png - Future price forecast")
    print("   7. best_lstm_model.pth - Saved model weights")
    
    print("\n📊 Final Results:")
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    avg_mae = np.mean([m['MAE'] for m in metrics.values()])
    avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
    avg_dir = np.mean([m['Direction_Accuracy'] for m in metrics.values()])
    
    print(f"   Average R²:                  {avg_r2:.4f}")
    print(f"   Average MAE:                ${avg_mae:,.2f}")
    print(f"   Average MAPE:               {avg_mape:.2f}%")
    print(f"   Average Direction Accuracy: {avg_dir:.1f}%")
    print(f"   Current Bitcoin Price:      ${prices.iloc[-1]:,.2f}")
    print(f"   30-day Forecast:            ${forecasts[-1]:,.2f}")
    print(f"   Expected 30-day Return:     {((forecasts[-1] / prices.iloc[-1] - 1) * 100):.2f}%\n")

    print("🏆 Pipeline completed successfully!\n")

    print_box() # Line break
    print(f"📈 Thank you for using Bitcoin Forecasting System!")
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'forecasts': forecasts,
        'config': CONFIG,
        'feature_cols': feature_cols,
        'test_data': test_features
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Set matplotlib backend
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
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
        
        # print_box("\nALL ANALYSES COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

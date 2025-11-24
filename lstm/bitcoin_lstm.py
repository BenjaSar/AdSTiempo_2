"""
LSTM: Traditional Time-Series architechture for Forecasting (Benchmark)
Production-Ready Implementation with Bitcoin Price Prediction

Usage:
python lstm/bitcoin_lstm.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader #  Dataset, 
from sklearn.preprocessing import StandardScaler #, MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from ETL module
from src.etl import BitcoinDataLoader, BitcoinEDA

# Import Informer architecture
from src.models.lstm_model import (
    LSTMModel,
    TimeSeriesDataset, 
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
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Advanced feature engineering for time series"""
    
    @staticmethod
    def create_features(df, verbose=True):
        """Create comprehensive feature set"""
        if verbose:
            print_box("\nFEATURE ENGINEERING")
            print("🔧 Creating features...")
        
        df = df.copy()
        original_cols = len(df.columns)
        
        # 1. Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
        df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        
        # 2. Moving averages
        for window in [5, 7, 14, 21, 30, 60]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
            
        # 3. Exponential moving averages
        for span in [12, 26]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
            
        # 4. Volatility indicators
        for window in [5, 10, 20, 30]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
            df[f'ATR_{window}'] = (df['High'] - df['Low']).rolling(window=window).mean()
        
        # 5. Momentum indicators
        for period in [5, 10, 14, 21]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
                                   df['Close'].shift(period)) * 100
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        # 6. RSI
        for period in [14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # 7. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 8. Bollinger Bands
        for window in [20]:
            df[f'BB_Middle_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'BB_Std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'BB_Upper_{window}'] = df[f'BB_Middle_{window}'] + 2 * df[f'BB_Std_{window}']
            df[f'BB_Lower_{window}'] = df[f'BB_Middle_{window}'] - 2 * df[f'BB_Std_{window}']
            df[f'BB_Width_{window}'] = (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}']) / df[f'BB_Middle_{window}']
            df[f'BB_Position_{window}'] = (df['Close'] - df[f'BB_Lower_{window}']) / (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}'])
        
        # 9. Volume indicators
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # 10. Lagged features
        for lag in [1, 2, 3, 5, 7]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # 11. Time-based features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfMonth'] = df.index.day
        df['DayOfYear'] = df.index.dayofyear
        df['WeekOfYear'] = df.index.isocalendar().week
        
        # Cyclical encoding
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Drop NaN values
        df = df.dropna()
        
        new_cols = len(df.columns)
        
        if verbose:
            print(f"   ✅ Created {new_cols - original_cols} new features")
            print(f"   ✅ Valid samples: {len(df)}\n")
        
        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # ========================================
    # CONFIGURATION
    # ========================================
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
        'epochs': 2, #100
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
    df_features = FeatureEngineer.create_features(df, verbose=True)
    
    # Select features (exclude categorical time features)
    exclude_cols = ['DayOfWeek', 'Month', 'Quarter', 'DayOfMonth', 
                   'DayOfYear', 'WeekOfYear']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    df_model = df_features[feature_cols].copy()
    
    print(f"📊 Selected {len(feature_cols)} features for modeling\n")
    
    # ========================================
    # NORMALIZE & SPLIT
    # ========================================
    print_box("\nDATA PREPARATION & SPLITTING")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_model.values)
    
    # Split data
    n = len(scaled_data)
    train_size = int(n * CONFIG['train_ratio'])
    val_size = int(n * CONFIG['val_ratio'])
    
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size + val_size]
    test_data = scaled_data[train_size + val_size:]
    
    print(f"📊 Data split:")
    print(f"   Training set:   {len(train_data)} samples ({CONFIG['train_ratio']*100:.0f}%)")
    print(f"   Validation set: {len(val_data)} samples ({CONFIG['val_ratio']*100:.0f}%)")
    print(f"   Test set:       {len(test_data)} samples ({(1-CONFIG['train_ratio']-CONFIG['val_ratio'])*100:.0f}%)")
    print(f"   Total:          {n} samples\n")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, CONFIG['seq_len'], CONFIG['pred_len'])
    val_dataset = TimeSeriesDataset(val_data, CONFIG['seq_len'], CONFIG['pred_len'])
    test_dataset = TimeSeriesDataset(test_data, CONFIG['seq_len'], CONFIG['pred_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # ========================================
    # BUILD MODEL
    # ========================================
    model = LSTMModel(
        input_dim=len(feature_cols),
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
    
    # ========================================
    # TRAIN MODEL
    # ========================================
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
    
    # ========================================
    # LOAD BEST MODEL & EVALUATE
    # ========================================
    model.load_state_dict(torch.load('lstm/best_lstm_model.pth'))
    
    close_idx = feature_cols.index('Close')
    predictions, actuals, metrics = Evaluator.evaluate(
        model, test_loader, device, scaler, close_idx
    )
    
    # Get test dates
    test_start_idx = train_size + val_size + CONFIG['seq_len']
    test_dates = df_features.index[test_start_idx:test_start_idx + len(predictions)]
    
    # Plot predictions
    Evaluator.plot_predictions(predictions, actuals, test_dates)
    Evaluator.plot_error_analysis(predictions, actuals, test_dates)
    
    # ========================================
    # FUTURE FORECASTING
    # ========================================
    last_sequence = test_data[-CONFIG['seq_len']:]
    
    forecasts = FutureForecaster.forecast_recursive(
        model, last_sequence, scaler, close_idx, 
        n_days=CONFIG['forecast_days'], device=device
    )
    
    # Prepare visualization data
    historical_prices = df_model['Close'].values[-200:]
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
    
    # ========================================
    # SUMMARY
    # ========================================
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
    
    print(f"   Average R² Score:           {avg_r2:.4f}")
    print(f"   Average MAE:                ${avg_mae:,.2f}")
    print(f"   Average MAPE:               {avg_mape:.2f}%")
    print(f"   Average Direction Accuracy: {avg_dir:.1f}%")
    print(f"   Current Bitcoin Price:      ${df['Close'].iloc[-1]:,.2f}")
    print(f"   30-day Forecast:            ${forecasts[-1]:,.2f}")
    print(f"   Expected 30-day Return:     {((forecasts[-1] / df['Close'].iloc[-1] - 1) * 100):.2f}%\n")

    print("🏆 Pipeline completed successfully!\n")

    print_box() # Line break
    print(f"📈 Thank you for using Bitcoin LSTM Forecasting System!")
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'forecasts': forecasts,
        'config': CONFIG,
        'feature_cols': feature_cols,
        'test_data': test_data
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

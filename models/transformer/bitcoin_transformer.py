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
import sys
import warnings
warnings.filterwarnings('ignore')
 
# Deep learning libraries
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Logging configuration
import datetime
from utils.logging import get_logger

# Configure Logger before main execution
LOG_FILENAME = f"logs/run_transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = get_logger(name='BTC_Transformer', log_file=LOG_FILENAME)

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
    FutureForecaster
)

# Import Feature importance & Risk analysis modules
from src.feature_importance import FeatureImportance 
from src.risk_analyzer import RiskAnalyzer

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
        'patience': 30,          # Early stopping patience -> Increased
        
        # Data split
        'train_ratio': 0.7,
        'val_ratio': 0.15,

        # Future forecast
        'forecast_days': 30
    }

    # 📝 LOGGING: Configuration parameters
    logger.info("⚙️  STARTING EXPERIMENT WITH CONFIGURATION:")
    for key, value in CONFIG.items():
        logger.info(f"   - {key}: {value}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device} | PyTorch version: {torch.__version__}\n")
    logger.info(f"⚙️  Device: {device} | PyTorch version: {torch.__version__}")
    
    # ETL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info("--- 1. DATA LOADING & ETL ---")
    logger.info(f"Loading data from {CONFIG['start_date']} to {CONFIG['end_date'] if CONFIG['end_date'] else 'today'}...")
    df, is_real = BitcoinDataLoader.load_data(
        use_real_data=CONFIG['use_real_data'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date']
    )
    
    # EDA (Optional)
    eda = BitcoinEDA(df, is_real_data=is_real, output_dir='models/transformer/results/')
    eda.run_full_eda()
    logger.info("EDA completed and plots saved.")

    # Feature engineering 
    features_df, prices = ImprovedFeatureEngineer.create_features(df)
    
    print(f"📊 Selected {len(features_df)} features for modeling\n")
    logger.info(f"📊 Feature Engineering: Selected {len(features_df.columns)} features for modeling.")
    logger.info(f"Features list: {list(features_df.columns)}")

    # NORMALIZE & SPLIT ~~~~~~~~~~~~~~~~~~~~~~~~~~
    print_box("DATA PREPARATION & SPLITTING")
    logger.info("--- 2. DATA PREPARATION & SPLITTING ---")
    
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
    
    # Console output for data split
    print(f"📊 Data split:")
    print(f"   Training set:   {len(train_features)} samples ({CONFIG['train_ratio']*100:.0f}%)")
    print(f"   Validation set: {len(val_features)} samples ({CONFIG['val_ratio']*100:.0f}%)")
    print(f"   Test set:       {len(test_features)} samples ({(1-CONFIG['train_ratio']-CONFIG['val_ratio'])*100:.0f}%)")
    print(f"   Total:          {n} samples\n")

    # 📝 LOGGING: Data split details
    logger.info("📊 Data split details:")
    logger.info(f"   Total samples:    {n}")
    logger.info(f"   Training set:     {len(train_features)} samples ({CONFIG['train_ratio']*100:.0f}%)")
    logger.info(f"   Validation set:   {len(val_features)} samples ({CONFIG['val_ratio']*100:.0f}%)")
    logger.info(f"   Test set:         {len(test_features)} samples ({(1-CONFIG['train_ratio']-CONFIG['val_ratio'])*100:.0f}%)")

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
    logger.info(f"Dataloaders created with batch size {CONFIG['batch_size']}.")
    
    # Build model
    logger.info("--- 3. MODEL INITIALIZATION & TRAINING ---")
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

    # Console output for model architecture
    print(f"🧠 Transformer Architecture:")
    print(f"   Input features:      {n_features}")
    print(f"   Sequence length:     {CONFIG['seq_len']} days")
    print(f"   Prediction horizon:  {CONFIG['pred_len']} days")
    print(f"   Model dimension:     {CONFIG['d_model']}")
    print(f"   Layers:              {CONFIG['num_layers']}")
    print(f"   Total parameters:    {sum(p.numel() for p in model.parameters()):,}")
    
    # 📝 LOGGING: Model Architecture
    logger.info("🧠 Transformer Architecture:")
    logger.info(f"   Input features:      {n_features}")
    logger.info(f"   Sequence length:     {CONFIG['seq_len']} days")
    logger.info(f"   Prediction horizon:  {CONFIG['pred_len']} days")
    logger.info(f"   Model dimension:     {CONFIG['d_model']}")
    logger.info(f"   Layers:              {CONFIG['num_layers']}")
    logger.info(f"   Total parameters:    {sum(p.numel() for p in model.parameters()):,}")
    
    # BUILD MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train
    trainer = ImprovedTrainer(
        model, device,
        learning_rate=CONFIG['learning_rate'],
        warmup_epochs=CONFIG['warmup_epochs']
    )
    
    logger.info(f"Starting training for {CONFIG['epochs']} epochs with LR {CONFIG['learning_rate']} and patience {CONFIG['patience']}...")
    train_losses, val_losses = trainer.fit(
        train_loader, val_loader,
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience']
    )
    logger.info("Training finished. Best model state loaded.")
    
    # Evaluate
    logger.info("--- 4. MODEL EVALUATION ---")
    model.load_state_dict(torch.load('models/transformer/best_transformer_model.pth'))
    predictions, actuals, metrics = ImprovedEvaluator.evaluate(
        model, test_loader, device, returns_scaler
    )
    
    # RESULTS ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info("Generating plots for prediction comparison, error analysis, and training history...")
    # Plot results
    ImprovedEvaluator.plot_predictions(predictions, actuals)
    ImprovedEvaluator.plot_error_analysis(predictions, actuals)
    ImprovedEvaluator.plot_training_history(train_losses, val_losses)
    
    # Process summary
    print_box("\nIMPROVEMENTS SUMMARY")

    # Console output for improvements
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
    
    # Console output for final results
    print(f"📊 Final Results:")
    print(f"   Average R²:   {avg_r2:.4f}")
    print(f"   Average MAPE: {avg_mape:.2f}%\n")
    print("🏆 Pipeline completed successfully!\n")
    
    # 📝 LOGGING: Final Metrics
    logger.info("--- 5. FINAL RESULTS ---")
    logger.info(f"📊 Average R² (Test Set):   {avg_r2:.4f}")
    logger.info(f"📊 Average MAPE (Test Set): {avg_mape:.2f}%")
    logger.info("🏆 Pipeline completed.")

    print_box() # Line break
    print("📈 Thank you for using Bitcoin Forecasting System!")

    # ---------------------------------------------------------
    # 1. GENERATE FUTURE FORECASTS (Add this block)
    # ---------------------------------------------------------
    
    # Get the last sequence from data to start the recursive prediction
    # Shape needs to be (seq_len, n_features)
    last_sequence_data = test_features[-CONFIG['seq_len']:]
    
    # Generate recursive forecasts
    forecasts = FutureForecaster.forecast_recursive(
        model=model,
        last_sequence=last_sequence_data,
        scaler=scaler, # Use the full scaler for inverse transform
        close_idx=0,
        n_days=CONFIG['forecast_days'],
        device=device
    )

    results = {
        'model': model,
        'scaler': scaler,
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'forecasts': forecasts,
        'config': CONFIG,
        'feature_cols': features_df.columns.tolist()
    }

    # Optional: Feature importance analysis
    print_box() # Line break
    logger.info("\n--- 6. POST-ANALYSIS OPTIONS ---")
    response = input("Would you like to perform feature importance analysis? (y/n): ")
    if response.lower() == 'y':
        logger.info("Starting Feature Importance Analysis...")
        # Use a subset of the test set for importance calculation
        test_subset_dataset = ReturnsDataset(
            test_features[:min(200, len(test_prices))],
            test_prices[:min(200, len(test_prices))],
            results['config']['seq_len'],
            results['config']['pred_len']
        )

        # Create DataLoader for the subset
        test_subset_loader = DataLoader(
            test_subset_dataset,
            batch_size=32,
            shuffle=False
        )

        # Calculate feature importance
        importance_scores = FeatureImportance.calculate_importance(
            results['model'],
            test_subset_loader,
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            results['feature_cols'],
            n_repeats=5,
            max_batches=10,
            output_dir='models/transformer/results/'
        )
        
        # Console output for importance scores
        print("RESULTS:")
        [print(f"   - {key}: {value:.3f}") for key, value in importance_scores.items()]
        logger.info(f"Feature Importance Scores: {importance_scores}")
    else:
        logger.info("Feature importance analysis skipped.")
    
    # Optional: Risk analysis
    print_box() # Line break
    response = input("Would you like to perform risk analysis? (y/n): ")
    if response.lower() == 'y':
        logger.info("Starting Risk Analysis...")
        # Get historical returns
        historical_returns = np.diff(results['actuals'][:, 0]) / results['actuals'][:-1, 0]
        
        # Calculate risk metrics
        risk_metrics = RiskAnalyzer.analyze_risk(
            results['forecasts'],
            results['actuals'][-1, 0],  # Last actual price
            historical_returns
        )

        # Console output for risk metrics
        print("RESULTS:")
        [print(f"   - {key}: {value:.3f}") for key, value in risk_metrics.items()]
        logger.info(f"Risk Metrics: {risk_metrics}")
    else:
        logger.info("Risk analysis skipped.")


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
    
        print_box("ALL ANALYSES COMPLETED SUCCESSFULLY!")
        logger.info("✅ ALL ANALYSES COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        logger.warning("⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        logger.error(f"❌ Critical Error occurred: {str(e)}", exc_info=True)
        # import traceback
        # traceback.print_exc()

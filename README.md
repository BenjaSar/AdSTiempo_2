# Bitcoin Price Forecasting with Transformer & Informer

Advanced time series forecasting project comparing Transformer and Informer architectures for Bitcoin price prediction, with returns-based training improvements.

## 📋 Project Overview

This project implements and compares two state-of-the-art deep learning architectures for Bitcoin price forecasting:
- **Transformer**: Standard attention-based architecture
- **Informer**: Efficient transformer with ProbSparse attention (O(L log L) complexity)

Each architecture is implemented in two versions:
1. **Original**: Price-based training with standard techniques
2. **Improved**: Returns-based training with advanced optimization

## 🎯 Key Features

- ✅ Real Bitcoin data fetching via Yahoo Finance (yfinance)
- ✅ Comprehensive Exploratory Data Analysis (EDA)
- ✅ Returns-based and price-based training approaches
- ✅ Advanced feature engineering (volatility, momentum, RSI)
- ✅ Multiple forecast horizons (7, 30, 45 days)
- ✅ Detailed performance metrics (RMSE, MAE, R², MAPE, Directional Accuracy)
- ✅ Extensive visualizations and comparison plots

## 🚀 Installation

### Option 1: Using pip (Recommended for this project)

```bash
# Navigate to project directory
cd F:\IA\AdST2_v2

# Create virtual environment
python -m venv adst2

# Activate environment (Windows)
adst2\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate adst2
```

## 📦 Requirements

- Python 3.11
- PyTorch 2.1.0
- NumPy 1.26.0
- Pandas 2.1.0
- Matplotlib 3.8.0
- Seaborn 0.13.0
- Scikit-learn 1.3.0
- SciPy 1.11.0
- yfinance 0.2.31

## 📂 Project Structure

```
AdST2_v2/
├── bitcoin_transformer.py              # Original Transformer implementation
├── bitcoin_transformer_improved.py     # Improved Transformer (returns-based)
├── compare_forecast_windows.py         # Compare different forecast windows
├── informer/
│   ├── bitcoin_informer.py            # Original Informer implementation
│   └── bitcoin_informer_improved.py   # Improved Informer (returns-based)
├── ventanas/                          # Results storage for different windows
├── environment.yml                    # Conda environment file
├── requirements.txt                   # Pip requirements file
└── README.md                          # This file
```

## 🎮 Usage

### 1. Train Original Transformer

```bash
python bitcoin_transformer.py
```

**Outputs:**
- `best_bitcoin_model.pth` - Model weights
- `01_comprehensive_eda.png` - EDA visualizations
- `02_advanced_analysis.png` - Advanced analysis
- `03_predictions.png` - Prediction plots
- `04_error_analysis.png` - Error analysis
- `05_training_history.png` - Training curves

### 2. Train Improved Transformer (Recommended)

```bash
python bitcoin_transformer_improved.py
```

**Outputs:**
- `best_improved_model.pth` - Model weights
- `improved_predictions.png` - Prediction plots
- `improved_error_analysis.png` - Error analysis
- `improved_training_history.png` - Training curves

### 3. Train Original Informer

```bash
python informer/bitcoin_informer.py
```

**Outputs:**
- `informer/best_informer_model.pth` - Model weights
- `informer/01_comprehensive_eda.png` - EDA visualizations
- `informer/02_advanced_analysis.png` - Advanced analysis
- `informer/03_predictions.png` - Prediction plots

### 4. Train Improved Informer (Recommended)

```bash
python informer/bitcoin_informer_improved.py
```

**Outputs:**
- `informer/best_improved_informer.pth` - Model weights
- `informer/improved_predictions.png` - Prediction plots
- `informer/improved_error_analysis.png` - Error analysis
- `informer/improved_training_history.png` - Training curves

### 5. Compare Forecast Windows

First, train models with different prediction lengths (modify `CONFIG['pred_len']`):
- 7 days
- 30 days
- 45 days

Then run the comparison:

```bash
python compare_forecast_windows.py
```

**Outputs:**
- `comparison_metrics.png` - Metrics comparison
- `comparison_predictions.png` - Individual window comparisons
- `comparison_predictions_summary.png` - All windows overlay
- `comparison_errors.png` - Error distributions
- `comparison_training.png` - Training history

## 🔬 Model Architectures

### Transformer
- **Architecture**: Standard multi-head self-attention
- **Complexity**: O(L²) where L is sequence length
- **Best for**: Moderate sequence lengths (7-60 days)

### Informer
- **Architecture**: ProbSparse self-attention with distilling
- **Complexity**: O(L log L) 
- **Best for**: Long sequences (efficient for 30-90 days)
- **Key innovations**:
  - ProbSparse attention mechanism
  - Self-attention distilling
  - Generative decoder

## 📊 Improvements Applied

### Original Version
- Raw price prediction
- StandardScaler normalization
- MSE loss
- Fixed learning rate
- Sequence length: 60 days

### Improved Version
- ✅ **Returns-based training**: Train on log-returns, reconstruct prices
- ✅ **MinMaxScaler**: Better normalization for returns distribution
- ✅ **Huber loss**: More robust to outliers
- ✅ **Learning rate scheduling**: Warmup (5 epochs) + cosine annealing
- ✅ **Optimized architecture**: Shorter sequences (10 days), reduced layers
- ✅ **Enhanced features**: 13 features including volatility, momentum, RSI
- ✅ **Gradient clipping**: Training stability
- ✅ **Directional accuracy**: Additional evaluation metric

## 📈 Performance Metrics

Both models report:
- **RMSE**: Root Mean Squared Error (USD)
- **MAE**: Mean Absolute Error (USD)
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error (%)
- **Directional Accuracy**: Trend prediction accuracy (%)

**Target Performance (Improved Versions):**
- R² > 0.5
- MAPE < 5%
- Directional Accuracy > 50%

## 🎨 Visualizations

### EDA Plots
- Price trends with volume
- Returns distribution and Q-Q plot
- Rolling volatility
- Feature correlations
- Moving averages
- Seasonal patterns (day/month)

### Prediction Plots
- Predicted vs Actual prices
- Multi-day forecasts
- Error distributions
- Training history (loss curves)

### Comparison Plots
- Metrics across forecast horizons
- Real data vs predictions by window
- Error analysis by window
- Training convergence comparison

## 🔧 Configuration

Key parameters in both scripts:

```python
CONFIG = {
    # Data
    'use_real_data': True,
    'start_date': '2020-01-01',
    
    # Model
    'seq_len': 10,      # Input sequence length
    'pred_len': 7,      # Prediction horizon
    'd_model': 128,     # Model dimension
    'nhead': 8,         # Attention heads
    'num_layers': 2,    # Transformer layers
    
    # Training
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0005,
    'patience': 15,
}
```

## 📝 Notes

- **GPU Support**: Automatically uses CUDA if available
- **Data Source**: Real-time Bitcoin data from Yahoo Finance
- **Fallback**: Synthetic data generation if yfinance unavailable
- **Checkpoint**: Best model saved based on validation loss
- **Early Stopping**: Prevents overfitting with patience parameter

## 🐛 Troubleshooting

### yfinance Not Installing
```bash
pip install yfinance --upgrade
```

### CUDA Out of Memory
Reduce `batch_size` in CONFIG:
```python
'batch_size': 16,  # Or even 8
```

### Import Errors
Ensure environment is activated:
```bash
adst2\Scripts\activate  # Windows
```

## 📚 References

1. **Transformer**: Vaswani et al., "Attention is All You Need" (2017)
2. **Informer**: Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (2021)
3. **Returns-based Training**: Standard practice in financial forecasting

## 👨‍💻 Author

Applied Data Science & Transformers 2 - Project v2

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Yahoo Finance for Bitcoin data
- PyTorch team for the framework
- Original Informer paper authors

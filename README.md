# Bitcoin Price Forecasting with Transformer & Informer

Advanced time series forecasting project comparing Transformer, Informer and LSTM architectures for Bitcoin price prediction, with returns-based training improvements.

## 📋 Project Overview

This project implements and compares two state-of-the-art deep learning architectures for Bitcoin price forecasting:
- **Transformer**: Standard attention-based architecture,
- **Informer**: Efficient transformer with ProbSparse attention (O(L log L) complexity),
- **LSTM**: Traditional Time Series architecture.

Read in detail in [this section](#-model-architectures)

## 🎯 Key Features

- ✅ Real Bitcoin data fetching via Yahoo Finance (yfinance)
- ✅ Comprehensive Exploratory Data Analysis (EDA)
- ✅ Returns-based and price-based training approaches
- ✅ Advanced feature engineering (volatility, momentum, RSI)
- ✅ Multiple forecast horizons (7, 30, 45 days)
- ✅ Detailed performance metrics (RMSE, MAE, R², MAPE, Directional Accuracy)
- ✅ Extensive visualizations and comparison plots

## 📂 Project Structure

```
AdST2/
├── forecast_windows/
│   ├── results/                            # Results storage 
│   └── compare_forecast_windows.py         # Comparison between forecasts
├── informer/
│   ├── results/                            # Results storage 
│   ├── best_informer_model.pth             # Model weights
│   └── bitcoin_informer.py                 # Informer implementation (Improvement)
├── lstm/
│   ├── results/                            # Results storage 
│   ├── best_lstm_model.pth                 # Model weights
│   └── bitcoin_lstm.py                     # LSTM implementation (Benchmark)
├── tests/
│   ├── check_cuda.py                       # Verifies GPU availability
│   ├── fix_api.py                          # If Yahoo Finance API is not working, this fixs it.
│   └── test_api.py                         # Checks Yahoo Finance API is working
├── models/transformer/
│   ├── results/                            # Results storage 
│   ├── best_transformer_model.pth          # Model weights
│   └── bitcoin_transformer.py              # Transformer implementation (Original)
├── .gitignore
├── CRITERIOS_EVALUACION.md
├── environment.yml
├── README.md
├── EDA.ipynb                               # Exploratory Data Analysis ??
└── requirements.txt
```

## 🚀 Installation

### Option 1: Using pip (Recommended for this project)

```bash
# Navigate to project directory
cd path/to/your/project  # Replace with your actual project directory

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

### Finally: Run unit tests
Verify everything is working properly
```bash
python -m pytest
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

<<<<<<< Updated upstream
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

=======
>>>>>>> Stashed changes
## 🎮 Usage

### 1. Train Model

You can choose between different models:
1. transformer
1. informer
1. lstm

Replace *{model}* with the one you've choosed.

```bash
<<<<<<< Updated upstream
python bitcoin_transformer.py
```

**Outputs:**
- `best_bitcoin_model.pth` - Model weights
=======
python {model}/bitcoin_{model}.py
```

**Outputs:**
- `best_{model}_model.pth` - Model weights
>>>>>>> Stashed changes
- `01_comprehensive_eda.png` - EDA visualizations
- `02_advanced_analysis.png` - Advanced analysis
- `03_predictions.png` - Prediction plots
- `04_error_analysis.png` - Error analysis
- `05_training_history.png` - Training curves

<<<<<<< Updated upstream
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
=======
### 2. Compare Forecast Windows
>>>>>>> Stashed changes

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

### LSTM
- **Architecture**: Long Short-Term Memory with stacked layers
- **Complexity**: O(L) sequential processing
- **Best for**: Capturing temporal dependencies and long-term patterns
- **Key features**:
  - Gated memory cells
  - Bidirectional processing option
  - Superior to vanilla RNN for long sequences

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

During the development phase, we've explored two different implementations. We successfully improved model accuracy by applying the following changes.

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

For clarity, we decided to remove these development versions from the main branch. The original implementation can be analyzed side-by-side on the 'Legacy' branch.

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

### yfinance not working
```bash
python tests/fix_api.py
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

### Run unit tests
```bash
python -m pytest -v
```

## 📚 References
1.  **LSTM**: [Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)

2.  **Transformer**: [Vaswani et al., "Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)

3.  **Informer**: [Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (2021)](https://arxiv.org/abs/2012.07436)

4.  **Returns-based Training**: [Standard practice in financial forecasting (Methodology)](https://ssrn.com/abstract=3971306)

## 👨‍💻 Author
Applied Data Science & Transformers 2 - Project v2

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for Bitcoin data
- PyTorch team for the framework
- Original Informer paper authors

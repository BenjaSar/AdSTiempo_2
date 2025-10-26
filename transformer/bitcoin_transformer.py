"""
Production-Ready Bitcoin Time Series Transformer
Complete implementation with real data download and analysis

Requirements:
pip install torch numpy pandas matplotlib seaborn scikit-learn yfinance

Usage:
python bitcoin_transformer_production.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import yfinance for real data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not available. Install with: pip install yfinance")
    print("   Using synthetic data instead...\n")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           BITCOIN TIME SERIES FORECASTING WITH TRANSFORMERS                   ║
║                     Production Implementation v1.0                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# DATA LOADER
# ============================================================================

class BitcoinDataLoader:
    """Load Bitcoin data from Yahoo Finance or generate synthetic data"""
    
    @staticmethod
    def load_real_data(start_date='2020-01-01', end_date=None):
        """Load real Bitcoin data from Yahoo Finance"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            print(f"📥 Downloading Bitcoin data from {start_date} to {end_date or 'today'}...")
            btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
            
            if len(btc) == 0:
                print("⚠️  No data returned from Yahoo Finance")
                return None
            
            # Rename columns to standard format
            df = btc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            print(f"✅ Downloaded {len(df)} days of Bitcoin data")
            print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"   Latest price: ${df['Close'].iloc[-1]:,.2f}\n")
            
            return df
            
        except Exception as e:
            print(f"⚠️  Error downloading data: {e}")
            return None
    
    @staticmethod
    def generate_synthetic_data(start_date='2020-01-01', end_date='2024-10-01'):
        """Generate realistic synthetic Bitcoin data"""
        print("🔧 Generating synthetic Bitcoin data...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Realistic Bitcoin price simulation
        np.random.seed(42)
        
        # Components
        trend = np.linspace(10000, 65000, n)  # Long-term uptrend
        seasonal = 5000 * np.sin(2 * np.pi * np.arange(n) / 365)  # Annual cycle
        cycles = 3000 * np.sin(2 * np.pi * np.arange(n) / 90)  # Quarterly cycle
        noise = np.random.normal(0, 1500, n).cumsum() * 0.15  # Random walk
        volatility = np.random.normal(0, 800, n)  # Daily volatility
        
        # Combine components
        close_price = trend + seasonal + cycles + noise + volatility
        close_price = np.maximum(close_price, 5000)  # Floor price
        
        # Generate OHLC
        df = pd.DataFrame({
            'Open': close_price * (1 + np.random.uniform(-0.015, 0.015, n)),
            'High': close_price * (1 + np.random.uniform(0.005, 0.025, n)),
            'Low': close_price * (1 + np.random.uniform(-0.025, -0.005, n)),
            'Close': close_price,
            'Volume': np.random.uniform(2e9, 5e10, n)
        }, index=dates)
        
        # Ensure High >= Close >= Low
        df['High'] = df[['High', 'Close', 'Open']].max(axis=1)
        df['Low'] = df[['Low', 'Close', 'Open']].min(axis=1)
        
        print(f"✅ Generated {len(df)} days of synthetic data")
        print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}\n")
        
        return df
    
    @classmethod
    def load_data(cls, use_real_data=True, start_date='2020-01-01', end_date=None):
        """Load data (real if available, otherwise synthetic)"""
        if use_real_data and YFINANCE_AVAILABLE:
            df = cls.load_real_data(start_date, end_date)
            if df is not None:
                return df, True
        
        # Fallback to synthetic data
        return cls.generate_synthetic_data(start_date, end_date or '2024-10-01'), False


# ============================================================================
# EDA MODULE
# ============================================================================

class BitcoinEDA:
    """Comprehensive Exploratory Data Analysis"""
    
    def __init__(self, df, is_real_data=False):
        self.df = df.copy()
        self.is_real_data = is_real_data
        
    def run_full_eda(self):
        """Execute complete EDA pipeline"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                      STEP 1: EXPLORATORY DATA ANALYSIS                        ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        self._basic_stats()
        self._analyze_returns()
        self._plot_comprehensive_eda()
        
    def _basic_stats(self):
        """Display basic statistics"""
        print("📊 BASIC STATISTICS")
        print("─" * 80)
        print(f"Dataset Shape: {self.df.shape[0]} days × {self.df.shape[1]} features")
        print(f"Date Range: {self.df.index.min().date()} to {self.df.index.max().date()}")
        print(f"Trading Days: {len(self.df)}")
        print(f"Data Type: {'Real (Yahoo Finance)' if self.is_real_data else 'Synthetic'}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        
        print(f"\nPrice Statistics:")
        print(f"  Current Price: ${self.df['Close'].iloc[-1]:,.2f}")
        print(f"  Highest Price: ${self.df['High'].max():,.2f}")
        print(f"  Lowest Price: ${self.df['Low'].min():,.2f}")
        print(f"  Average Price: ${self.df['Close'].mean():,.2f}")
        print(f"  Std Deviation: ${self.df['Close'].std():,.2f}")
        
    def _analyze_returns(self):
        """Analyze return characteristics"""
        returns = self.df['Close'].pct_change().dropna()
        
        print(f"\n📈 RETURNS ANALYSIS")
        print("─" * 80)
        print(f"Mean Daily Return: {returns.mean()*100:.3f}%")
        print(f"Daily Volatility: {returns.std()*100:.3f}%")
        print(f"Annualized Return: {returns.mean()*365*100:.2f}%")
        print(f"Annualized Volatility: {returns.std()*np.sqrt(365)*100:.2f}%")
        print(f"Sharpe Ratio (Rf=0): {returns.mean()/returns.std()*np.sqrt(365):.3f}")
        print(f"Max Daily Gain: {returns.max()*100:.2f}%")
        print(f"Max Daily Loss: {returns.min()*100:.2f}%")
        
    def _plot_comprehensive_eda(self):
        """Create comprehensive EDA visualizations"""
        print(f"\n📊 Creating visualizations...")
        
        # Figure 1: Price Analysis
        fig1 = plt.figure(figsize=(18, 12))
        gs1 = fig1.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1.1 Price trend with volume
        ax1 = fig1.add_subplot(gs1[0, :])
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', 
                color='#2E86AB', linewidth=2)
        ax1.fill_between(self.df.index, self.df['Low'], self.df['High'], 
                         alpha=0.2, color='#A23B72')
        ax1.set_title('Bitcoin Price Trend with High-Low Range', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Volume on secondary axis
        ax1_vol = ax1.twinx()
        ax1_vol.bar(self.df.index, self.df['Volume'], alpha=0.3, color='green', 
                   label='Volume')
        ax1_vol.set_ylabel('Volume', fontsize=12)
        ax1_vol.legend(loc='upper right', fontsize=10)
        
        # 1.2 Returns distribution
        ax2 = fig1.add_subplot(gs1[1, 0])
        returns = self.df['Close'].pct_change().dropna()
        ax2.hist(returns, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {returns.mean()*100:.3f}%')
        ax2.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 1.3 QQ plot
        ax3 = fig1.add_subplot(gs1[1, 1])
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 1.4 Rolling volatility
        ax4 = fig1.add_subplot(gs1[2, 0])
        volatility_30 = returns.rolling(window=30).std() * np.sqrt(365) * 100
        ax4.plot(self.df.index[1:], volatility_30, color='#C73E1D', linewidth=2)
        ax4.set_title('30-Day Rolling Volatility (Annualized)', 
                     fontsize=14, fontweight='bold')
        ax4.set_ylabel('Volatility (%)')
        ax4.set_xlabel('Date')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(self.df.index[1:], volatility_30, alpha=0.3, color='#C73E1D')
        
        # 1.5 Correlation heatmap
        ax5 = fig1.add_subplot(gs1[2, 1])
        corr = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
                   ax=ax5, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
        ax5.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.savefig('01_comprehensive_eda.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 01_comprehensive_eda.png")
        plt.close()
        
        # Figure 2: Advanced Analysis
        self._plot_advanced_analysis()
        
    def _plot_advanced_analysis(self):
        """Advanced analysis plots"""
        fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 2.1 Moving averages
        axes[0, 0].plot(self.df.index, self.df['Close'], label='Close', 
                       linewidth=1.5, alpha=0.7)
        axes[0, 0].plot(self.df.index, self.df['Close'].rolling(7).mean(), 
                       label='MA-7', linewidth=2)
        axes[0, 0].plot(self.df.index, self.df['Close'].rolling(30).mean(), 
                       label='MA-30', linewidth=2)
        axes[0, 0].plot(self.df.index, self.df['Close'].rolling(90).mean(), 
                       label='MA-90', linewidth=2)
        axes[0, 0].set_title('Moving Averages', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2.2 Seasonality (Day of Week)
        df_temp = self.df.copy()
        df_temp['DayOfWeek'] = df_temp.index.dayofweek
        df_temp['Returns'] = df_temp['Close'].pct_change()
        day_returns = df_temp.groupby('DayOfWeek')['Returns'].mean() * 100
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        colors = ['green' if x > 0 else 'red' for x in day_returns]
        axes[0, 1].bar(range(7), day_returns, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(days)
        axes[0, 1].set_title('Average Returns by Day of Week', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Average Return (%)')
        axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 2.3 Monthly seasonality
        df_temp['Month'] = df_temp.index.month
        month_returns = df_temp.groupby('Month')['Returns'].mean() * 100
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        colors = ['green' if x > 0 else 'red' for x in month_returns]
        axes[1, 0].bar(range(1, 13), month_returns, color=colors, alpha=0.7, 
                      edgecolor='black')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(months, rotation=45)
        axes[1, 0].set_title('Average Returns by Month', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Average Return (%)')
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 2.4 Cumulative returns
        cumulative_returns = (1 + df_temp['Returns'].fillna(0)).cumprod()
        axes[1, 1].plot(self.df.index, cumulative_returns, linewidth=2, color='#2E86AB')
        axes[1, 1].fill_between(self.df.index, 1, cumulative_returns, alpha=0.3)
        axes[1, 1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].axhline(1, color='black', linestyle='--', linewidth=1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_advanced_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 02_advanced_analysis.png")
        plt.close()
        
        print()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Advanced feature engineering for time series"""
    
    @staticmethod
    def create_features(df, verbose=True):
        """Create comprehensive feature set"""
        if verbose:
            print("╔═══════════════════════════════════════════════════════════════════════════════╗")
            print("║                     STEP 2: FEATURE ENGINEERING                               ║")
            print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
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
            print(f"   ✅ Total features: {new_cols}")
            print(f"   ✅ Valid samples after cleaning: {len(df)}\n")
        
        return df


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""
    
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 0]  # Close price
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting"""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, 
                 dim_feedforward, pred_len, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, dim_feedforward // 2)
        self.fc3 = nn.Linear(dim_feedforward // 2, pred_len)
        
        self.relu = nn.ReLU()
        
    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        
        # Project to d_model dimensions
        src = self.input_projection(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        # Use last time step
        output = output[:, -1, :]
        
        # Output layers
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.relu(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)
        
        return output


# ============================================================================
# TRAINER
# ============================================================================

class TransformerTrainer:
    """Training pipeline"""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                                     weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs, patience=10):
        """Train the model"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                          STEP 3: MODEL TRAINING                               ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        train_losses = []
        val_losses = []
        patience_counter = 0
        
        print(f"🚀 Training started with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Early stopping patience: {patience}\n")
        print("─" * 80)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_bitcoin_model.pth')
                patience_counter = 0
                status = "✅ (saved)"
            else:
                patience_counter += 1
                status = f"⏳ (patience: {patience_counter}/{patience})"
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] │ "
                      f"Train Loss: {train_loss:.6f} │ "
                      f"Val Loss: {val_loss:.6f} │ {status}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
        
        print("─" * 80)
        print(f"✅ Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.6f}\n")
        
        self._plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def _plot_training_history(self, train_losses, val_losses):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
                markersize=4, alpha=0.7)
        ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', 
                markersize=4, alpha=0.7)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training History', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
                markersize=4, alpha=0.7)
        ax2.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', 
                markersize=4, alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (MSE, log scale)', fontsize=12)
        ax2.set_title('Training History (Log Scale)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('03_training_history.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 03_training_history.png\n")
        plt.close()


# ============================================================================
# EVALUATION & FORECASTING
# ============================================================================

class Evaluator:
    """Model evaluation and prediction"""
    
    @staticmethod
    def evaluate(model, test_loader, device, scaler, close_idx=0):
        """Evaluate model on test set"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                         STEP 4: MODEL EVALUATION                              ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        model.eval()
        predictions = []
        actuals = []
        
        print("📊 Generating predictions on test set...")
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                predictions.append(output.cpu().numpy())
                actuals.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Inverse transform
        pred_len = predictions.shape[1]
        n_features = scaler.n_features_in_
        
        pred_rescaled = np.zeros_like(predictions)
        actual_rescaled = np.zeros_like(actuals)
        
        for i in range(pred_len):
            # Create dummy array with all features
            pred_full = np.zeros((predictions.shape[0], n_features))
            actual_full = np.zeros((actuals.shape[0], n_features))
            
            pred_full[:, close_idx] = predictions[:, i]
            actual_full[:, close_idx] = actuals[:, i]
            
            pred_rescaled[:, i] = scaler.inverse_transform(pred_full)[:, close_idx]
            actual_rescaled[:, i] = scaler.inverse_transform(actual_full)[:, close_idx]
        
        print("   ✅ Predictions generated\n")
        
        # Calculate metrics
        metrics = Evaluator._calculate_metrics(pred_rescaled, actual_rescaled)
        Evaluator._print_metrics(metrics)
        
        return pred_rescaled, actual_rescaled, metrics
    
    @staticmethod
    def _calculate_metrics(predictions, actuals):
        """Calculate evaluation metrics"""
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
            actual_direction = np.sign(np.diff(np.concatenate([[actual[0]], actual])))
            pred_direction = np.sign(np.diff(np.concatenate([[pred[0]], pred])))
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            metrics[f'Day_{i+1}'] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'Direction_Accuracy': direction_accuracy
            }
        
        return metrics
    
    @staticmethod
    def _print_metrics(metrics):
        """Print evaluation metrics"""
        print("📈 EVALUATION METRICS")
        print("─" * 80)
        
        # Header
        print(f"{'Forecast':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} "
              f"{'MAPE':>10} {'Dir_Acc':>10}")
        print("─" * 80)
        
        # Print each day
        for day, m in metrics.items():
            print(f"{day:<12} "
                  f"${m['RMSE']:>9,.2f} "
                  f"${m['MAE']:>9,.2f} "
                  f"{m['R2']:>9.4f} "
                  f"{m['MAPE']:>9.2f}% "
                  f"{m['Direction_Accuracy']:>9.1f}%")
        
        print("─" * 80)
        
        # Average metrics
        avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])
        avg_mae = np.mean([m['MAE'] for m in metrics.values()])
        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
        avg_dir = np.mean([m['Direction_Accuracy'] for m in metrics.values()])
        
        print(f"{'AVERAGE':<12} "
              f"${avg_rmse:>9,.2f} "
              f"${avg_mae:>9,.2f} "
              f"{avg_r2:>9.4f} "
              f"{avg_mape:>9.2f}% "
              f"{avg_dir:>9.1f}%")
        print("─" * 80 + "\n")
    
    @staticmethod
    def plot_predictions(predictions, actuals, dates, save_path='04_predictions.png'):
        """Visualize predictions"""
        pred_len = predictions.shape[1]
        n_plots = min(pred_len, 4)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(n_plots):
            ax = axes[i]
            
            # Plot actual and predicted
            ax.plot(dates, actuals[:, i], label='Actual', linewidth=2, 
                   alpha=0.8, color='#2E86AB')
            ax.plot(dates, predictions[:, i], label='Predicted', linewidth=2, 
                   alpha=0.8, color='#F18F01', linestyle='--')
            
            # Calculate metrics for title
            r2 = r2_score(actuals[:, i], predictions[:, i])
            mae = mean_absolute_error(actuals[:, i], predictions[:, i])
            
            ax.set_title(f'Day {i+1} Forecast (R²={r2:.3f}, MAE=${mae:,.0f})', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
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
    def plot_error_analysis(predictions, actuals, dates, save_path='05_error_analysis.png'):
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
        axes[1, 0].plot(dates, errors, linewidth=1, alpha=0.7, color='#C73E1D')
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].fill_between(dates, 0, errors, alpha=0.3, color='#C73E1D')
        axes[1, 0].set_xlabel('Date', fontsize=11)
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


# ============================================================================
# FUTURE FORECASTING
# ============================================================================

class FutureForecaster:
    """Generate future forecasts"""
    
    @staticmethod
    def forecast_recursive(model, last_sequence, scaler, close_idx, 
                          n_days=30, device='cpu'):
        """
        Recursive forecasting: predict one step, use it for next prediction
        """
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                        STEP 5: FUTURE FORECASTING                             ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        print(f"🔮 Generating {n_days}-day forecast...")
        
        model.eval()
        forecasts = []
        current_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for day in range(n_days):
                # Predict next step
                pred = model(current_seq)
                next_val_scaled = pred[0, 0].item()
                
                # Inverse transform to get actual price
                dummy = np.zeros((1, scaler.n_features_in_))
                dummy[0, close_idx] = next_val_scaled
                next_val_actual = scaler.inverse_transform(dummy)[0, close_idx]
                forecasts.append(next_val_actual)
                
                # Update sequence
                new_features = np.zeros((1, scaler.n_features_in_))
                new_features[0, close_idx] = next_val_scaled
                
                current_seq = torch.cat([
                    current_seq[:, 1:, :],
                    torch.FloatTensor(new_features).unsqueeze(0).to(device)
                ], dim=1)
        
        forecasts = np.array(forecasts)
        
        print(f"   ✅ Forecast complete\n")
        print("📊 FORECAST SUMMARY")
        print("─" * 80)
        print(f"Next day price:    ${forecasts[0]:,.2f}")
        print(f"7-day price:       ${forecasts[6]:,.2f}")
        print(f"14-day price:      ${forecasts[13]:,.2f}")
        print(f"30-day price:      ${forecasts[-1]:,.2f}")
        print(f"Expected return:   {((forecasts[-1] / forecasts[0] - 1) * 100):.2f}%")
        print("─" * 80 + "\n")
        
        return forecasts
    
    @staticmethod
    def plot_forecast(historical_data, historical_dates, forecasts, 
                     forecast_dates, save_path='06_future_forecast.png'):
        """Visualize future forecast"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
        
        # Full view
        ax1.plot(historical_dates, historical_data, label='Historical Data', 
                linewidth=2, color='#2E86AB', alpha=0.8)
        ax1.plot(forecast_dates, forecasts, label='Forecast', 
                linewidth=2.5, color='#F18F01', linestyle='--', marker='o', 
                markersize=4, alpha=0.9)
        
        # Confidence interval (±2σ based on recent volatility)
        recent_returns = np.diff(historical_data[-60:]) / historical_data[-60:-1]
        std = np.std(recent_returns) * historical_data[-1]
        expanding_std = std * np.sqrt(np.arange(1, len(forecasts) + 1))
        
        ax1.fill_between(forecast_dates, 
                        forecasts - 2*expanding_std, 
                        forecasts + 2*expanding_std, 
                        alpha=0.2, color='#F18F01', 
                        label='95% Confidence Interval')
        
        ax1.axvline(historical_dates[-1], color='green', linestyle=':', 
                   linewidth=2, label='Forecast Start', alpha=0.7)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Bitcoin Price (USD)', fontsize=12)
        ax1.set_title('Bitcoin Price Forecast - Full View', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'${x:,.0f}'))
        
        # Zoomed view (last 90 days + forecast)
        zoom_days = 90
        ax2.plot(historical_dates[-zoom_days:], historical_data[-zoom_days:], 
                label='Historical Data', linewidth=2, color='#2E86AB', alpha=0.8)
        ax2.plot(forecast_dates, forecasts, label='Forecast', 
                linewidth=2.5, color='#F18F01', linestyle='--', marker='o', 
                markersize=4, alpha=0.9)
        ax2.fill_between(forecast_dates, 
                        forecasts - 2*expanding_std, 
                        forecasts + 2*expanding_std, 
                        alpha=0.2, color='#F18F01')
        ax2.axvline(historical_dates[-1], color='green', linestyle=':', 
                   linewidth=2, alpha=0.7)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Bitcoin Price (USD)', fontsize=12)
        ax2.set_title('Bitcoin Price Forecast - Recent 90 Days + Forecast', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}\n")
        plt.close()


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
        'd_model': 128,         # Model dimension
        'nhead': 8,             # Number of attention heads
        'num_layers': 3,        # Number of transformer layers
        'dim_feedforward': 512, # Feedforward dimension
        'dropout': 0.1,         # Dropout rate
        
        # Training parameters
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 15,
        
        # Data split
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        # test_ratio = 1 - train_ratio - val_ratio = 0.15
        
        # Future forecast
        'forecast_days': 30
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device}")
    print(f"⚙️  PyTorch version: {torch.__version__}\n")
    
    # ========================================
    # LOAD DATA
    # ========================================
    df, is_real = BitcoinDataLoader.load_data(
        use_real_data=CONFIG['use_real_data'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date']
    )
    
    # ========================================
    # EDA
    # ========================================
    eda = BitcoinEDA(df, is_real_data=is_real)
    eda.run_full_eda()
    
    # ========================================
    # FEATURE ENGINEERING
    # ========================================
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
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                       DATA PREPARATION & SPLITTING                            ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
    
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
    model = TimeSeriesTransformer(
        input_dim=len(feature_cols),
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        pred_len=CONFIG['pred_len'],
        dropout=CONFIG['dropout']
    )
    
    print(f"🧠 Model architecture:")
    print(f"   Input dimension:     {len(feature_cols)}")
    print(f"   Model dimension:     {CONFIG['d_model']}")
    print(f"   Attention heads:     {CONFIG['nhead']}")
    print(f"   Transformer layers:  {CONFIG['num_layers']}")
    print(f"   Feedforward dim:     {CONFIG['dim_feedforward']}")
    print(f"   Output dimension:    {CONFIG['pred_len']}")
    print(f"   Total parameters:    {sum(p.numel() for p in model.parameters()):,}\n")
    
    # ========================================
    # TRAIN MODEL
    # ========================================
    trainer = TransformerTrainer(
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
    model.load_state_dict(torch.load('best_bitcoin_model.pth'))
    
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
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                           EXECUTION SUMMARY                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
    
    print("✅ Pipeline completed successfully!\n")
    
    print("📁 Generated files:")
    print("   1. 01_comprehensive_eda.png - Comprehensive exploratory data analysis")
    print("   2. 02_advanced_analysis.png - Advanced statistical analysis")
    print("   3. 03_training_history.png - Model training progress")
    print("   4. 04_predictions.png - Test set predictions")
    print("   5. 05_error_analysis.png - Prediction error analysis")
    print("   6. 06_future_forecast.png - Future price forecast")
    print("   7. best_bitcoin_model.pth - Saved model weights")
    
    print("\n📊 Key Results:")
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
    print(f"   Expected 30-day Return:     {((forecasts[-1] / df['Close'].iloc[-1] - 1) * 100):.2f}%")
    
    print("\n" + "=" * 80)
    print("🎉 Thank you for using Bitcoin Transformer Forecasting System!")
    print("=" * 80 + "\n")
    
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
# ADVANCED FEATURE IMPORTANCE ANALYSIS (OPTIONAL)
# ============================================================================

class FeatureImportance:
    """Analyze feature importance using permutation method"""
    
    @staticmethod
    def calculate_importance(model, test_loader, device, feature_names, 
                            n_repeats=5, max_batches=20):
        """
        Calculate feature importance using permutation method
        
        Note: This is computationally expensive. Using subset of data.
        """
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                    BONUS: FEATURE IMPORTANCE ANALYSIS                         ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        print(f"🔬 Calculating feature importance (this may take a while)...")
        print(f"   Using {n_repeats} permutations per feature")
        print(f"   Analyzing {max_batches} batches\n")
        
        model.eval()
        criterion = nn.MSELoss()
        
        # Calculate baseline loss
        baseline_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if batch_count >= max_batches:
                    break
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                baseline_loss += criterion(output, batch_y).item()
                batch_count += 1
        
        baseline_loss /= batch_count
        print(f"   Baseline loss: {baseline_loss:.6f}")
        
        # Calculate importance for each feature
        importance_scores = {}
        
        for feat_idx, feat_name in enumerate(feature_names):
            if feat_idx % 10 == 0:
                print(f"   Processing feature {feat_idx + 1}/{len(feature_names)}...")
            
            losses = []
            
            for _ in range(n_repeats):
                perm_loss = 0
                batch_count = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        if batch_count >= max_batches:
                            break
                        
                        batch_x = batch_x.clone()
                        
                        # Permute feature
                        perm_idx = torch.randperm(batch_x.size(0))
                        batch_x[:, :, feat_idx] = batch_x[perm_idx, :, feat_idx]
                        
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        output = model(batch_x)
                        perm_loss += criterion(output, batch_y).item()
                        batch_count += 1
                
                losses.append(perm_loss / batch_count)
            
            # Importance = increase in loss
            importance_scores[feat_name] = np.mean(losses) - baseline_loss
        
        print("   ✅ Feature importance calculated\n")
        
        # Sort and display
        sorted_importance = sorted(importance_scores.items(), 
                                  key=lambda x: abs(x[1]), reverse=True)
        
        print("📊 TOP 20 MOST IMPORTANT FEATURES")
        print("─" * 80)
        print(f"{'Rank':<6} {'Feature':<30} {'Importance Score':>20}")
        print("─" * 80)
        
        for rank, (feat, score) in enumerate(sorted_importance[:20], 1):
            print(f"{rank:<6} {feat:<30} {score:>20.6f}")
        
        print("─" * 80 + "\n")
        
        # Visualize
        FeatureImportance._plot_importance(sorted_importance[:20])
        
        return importance_scores
    
    @staticmethod
    def _plot_importance(top_features):
        """Plot feature importance"""
        features, scores = zip(*top_features)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['#2E86AB' if s > 0 else '#C73E1D' for s in scores]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Importance Score (Increase in MSE)', fontsize=12)
        ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold', pad=20)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('07_feature_importance.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 07_feature_importance.png\n")
        plt.close()


# ============================================================================
# RISK ANALYSIS
# ============================================================================

class RiskAnalyzer:
    """Analyze investment risk based on forecasts"""
    
    @staticmethod
    def analyze_risk(forecasts, current_price, historical_returns):
        """Calculate risk metrics"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                        BONUS: RISK ANALYSIS                                   ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        # Calculate metrics
        expected_return = (forecasts[-1] / current_price - 1) * 100
        forecast_returns = np.diff(forecasts) / forecasts[:-1]
        forecast_volatility = np.std(forecast_returns) * np.sqrt(365) * 100
        
        historical_vol = np.std(historical_returns) * np.sqrt(365) * 100
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(historical_returns, 5) * current_price
        
        # Maximum drawdown in forecast
        cumulative = np.cumprod(1 + forecast_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe = expected_return / forecast_volatility if forecast_volatility > 0 else 0
        
        print("📊 RISK METRICS")
        print("─" * 80)
        print(f"Expected 30-Day Return:        {expected_return:>10.2f}%")
        print(f"Forecast Volatility (Annual):  {forecast_volatility:>10.2f}%")
        print(f"Historical Volatility:         {historical_vol:>10.2f}%")
        print(f"Value at Risk (95%, 1-day):    ${var_95:>10,.2f}")
        print(f"Maximum Forecast Drawdown:     {max_drawdown:>10.2f}%")
        print(f"Sharpe Ratio (Rf=0):           {sharpe:>10.2f}")
        print("─" * 80)
        
        # Risk assessment
        print("\n🎯 RISK ASSESSMENT")
        print("─" * 80)
        
        if forecast_volatility > historical_vol * 1.2:
            print("⚠️  WARNING: Forecast shows higher volatility than historical")
        elif forecast_volatility < historical_vol * 0.8:
            print("✅ Forecast shows lower volatility than historical")
        else:
            print("ℹ️  Forecast volatility is consistent with historical patterns")
        
        if abs(max_drawdown) > 15:
            print(f"⚠️  WARNING: Significant drawdown expected ({max_drawdown:.1f}%)")
        else:
            print(f"✅ Moderate drawdown risk ({max_drawdown:.1f}%)")
        
        if sharpe > 1.0:
            print(f"✅ Favorable risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe > 0:
            print(f"ℹ️  Moderate risk-adjusted returns (Sharpe: {sharpe:.2f})")
        else:
            print(f"⚠️  Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
        
        print("─" * 80 + "\n")
        
        return {
            'expected_return': expected_return,
            'forecast_volatility': forecast_volatility,
            'historical_volatility': historical_vol,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe
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
        
        # Optional: Feature importance analysis
        print("\n" + "=" * 80)
        response = input("Would you like to perform feature importance analysis? (y/n): ")
        if response.lower() == 'y':
            # Use a subset of test data for feature importance
            test_subset = results['test_data'][:min(200, len(results['test_data']))]
            test_subset_dataset = TimeSeriesDataset(
                test_subset,
                results['config']['seq_len'],
                results['config']['pred_len']
            )
            test_subset_loader = DataLoader(
                test_subset_dataset,
                batch_size=32,
                shuffle=False
            )
            
            importance_scores = FeatureImportance.calculate_importance(
                results['model'],
                test_subset_loader,
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                results['feature_cols'],
                n_repeats=5,
                max_batches=10
            )
        
        # Optional: Risk analysis
        print("\n" + "=" * 80)
        response = input("Would you like to perform risk analysis? (y/n): ")
        if response.lower() == 'y':
            # Get historical returns
            historical_returns = np.diff(results['actuals'][:, 0]) / results['actuals'][:-1, 0]
            
            risk_metrics = RiskAnalyzer.analyze_risk(
                results['forecasts'],
                results['actuals'][-1, 0],  # Last actual price
                historical_returns
            )
        
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "ALL ANALYSES COMPLETED SUCCESSFULLY!" + " " * 21 + "║")
        print("╚" + "═" * 78 + "╝\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

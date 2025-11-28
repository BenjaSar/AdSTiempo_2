"""
ETL Module for Bitcoin Data: Loading and Synthetic Data Generation

This module provides functionality to load real Bitcoin data from Yahoo Finance
and generate realistic synthetic Bitcoin data for analysis and modeling.
"""

import functools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep learning libraries
import torch
from torch.utils.data import Dataset

OUTPUT_DIR = 'docs/results/' # Default directory to save EDA outputs

# Logging utility
from utils.loggin import get_logger
logger = get_logger(__name__)

# Try to import yfinance for real data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    logger.debug("yfinance successfully imported for real Bitcoin data loading.")
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available. Install with: pip install yfinance")
    print("   Using synthetic data instead...\n")
    logger.warning("yfinance not available. Install with: pip install yfinance")
    logger.info("Falling back to synthetic data for Bitcoin dataset.")

# Import formatting utility
from utils.misc import print_box

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
            btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=True)
            
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
        """
        Generate realistic synthetic Bitcoin data
        This function simulates Bitcoin price movements using a combination of trend,
        seasonal patterns, cycles, and random noise.
        """
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

    def print_status(func):
        """Decorador que imprime el estado de la carga de datos."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            df, is_real = func(*args, **kwargs) # Ejecuta el método original (load_data)
            
            if df is not None:
                status_message = f"✅ Loaded {len(df)} days of {'real' if is_real else 'synthetic'} data.\n"
                print(status_message)
    
            return df, is_real
        return wrapper
    
    @classmethod
    @print_status
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

    global OUTPUT_DIR

    def __init__(self, df: pd.DataFrame, output_dir: str=None, is_real_data: bool=False):
        self.df = df.copy()
        self.is_real_data = is_real_data
        if output_dir is not None:
            # Sanitize and normalize output directory
            if not isinstance(output_dir, str) or not output_dir.strip():
                raise ValueError("output_dir must be a non-empty string")
            sanitized_out = os.path.abspath(os.path.normpath(os.path.expanduser(output_dir.strip())))
            forbidden = ['/etc', '/bin', '/usr', '/var', '/boot', '/dev', '/proc', '/run', '/sys']
            for p in forbidden:
                if sanitized_out == p or sanitized_out.startswith(p + os.path.sep):
                    raise ValueError("output_dir points to a forbidden system directory")
            self.output_dir = sanitized_out
        else:
            self.output_dir = OUTPUT_DIR
        
    def run_full_eda(self):
        """Execute complete EDA pipeline"""
        print_box("\nEXPLORATORY DATA ANALYSIS (EDA)")
        
        os.makedirs(self.output_dir, exist_ok=True) # Create directory if not exists

        self._basic_stats()
        self._analyze_returns()
        self._plot_comprehensive_eda()
        
    def _basic_stats(self):
        """Display basic statistics"""
        print("📊 BASIC STATISTICS")
        print("─" * 81)
        print(f"Dataset Shape: {self.df.shape[0]} days × {self.df.shape[1]} features")
        print(f"Date Range: {self.df.index.min().date()} to {self.df.index.max().date()}")
        print(f"Trading Days: {len(self.df)}")
        print(f"Data Type: {'Real (Yahoo Finance)' if self.is_real_data else 'Synthetic'}\n")
        print(f"Statistical Summary:")
        print(self.df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
        
        # Check for missing values
        if not self.df.isnull().sum().sum():
            print("\nNo missing values detected.")
        else:
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
        print("─" * 81)
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
        
        plt.savefig(os.path.join(self.output_dir,'01_comprehensive_eda.png'), dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {self.output_dir}01_comprehensive_eda.png")
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
        plt.savefig(os.path.join(self.output_dir,'02_advanced_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {self.output_dir}02_advanced_analysis.png")
        plt.close()
        
        print()

# # ============================================================================
# # FEATURE ENGINEERING
# # ============================================================================

# class FeatureEngineer:
#     """Advanced feature engineering for time series"""
    
#     @staticmethod
#     def create_features(df, verbose=True):
#         """Create comprehensive feature set"""
#         if verbose:
#             print("╔═══════════════════════════════════════════════════════════════════════════════╗")
#             print("║                     STEP 2: FEATURE ENGINEERING                               ║")
#             print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
#             print("🔧 Creating features...")
        
#         df = df.copy()
#         original_cols = len(df.columns)
        
#         # 1. Price-based features
#         df['Returns'] = df['Close'].pct_change()
#         df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
#         df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
#         df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
#         df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
#         df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        
#         # 2. Moving averages
#         for window in [5, 7, 14, 21, 30, 60]:
#             df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
#             df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
            
#         # 3. Exponential moving averages
#         for span in [12, 26]:
#             df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
            
#         # 4. Volatility indicators
#         for window in [5, 10, 20, 30]:
#             df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
#             df[f'ATR_{window}'] = (df['High'] - df['Low']).rolling(window=window).mean()
        
#         # 5. Momentum indicators
#         for period in [5, 10, 14, 21]:
#             df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
#                                    df['Close'].shift(period)) * 100
#             df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
#         # 6. RSI
#         for period in [14, 21]:
#             delta = df['Close'].diff()
#             gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#             loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#             rs = gain / loss
#             df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
#         # 7. MACD
#         exp1 = df['Close'].ewm(span=12, adjust=False).mean()
#         exp2 = df['Close'].ewm(span=26, adjust=False).mean()
#         df['MACD'] = exp1 - exp2
#         df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
#         df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
#         # 8. Bollinger Bands
#         for window in [20]:
#             df[f'BB_Middle_{window}'] = df['Close'].rolling(window=window).mean()
#             df[f'BB_Std_{window}'] = df['Close'].rolling(window=window).std()
#             df[f'BB_Upper_{window}'] = df[f'BB_Middle_{window}'] + 2 * df[f'BB_Std_{window}']
#             df[f'BB_Lower_{window}'] = df[f'BB_Middle_{window}'] - 2 * df[f'BB_Std_{window}']
#             df[f'BB_Width_{window}'] = (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}']) / df[f'BB_Middle_{window}']
#             df[f'BB_Position_{window}'] = (df['Close'] - df[f'BB_Lower_{window}']) / (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}'])
        
#         # 9. Volume indicators
#         df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
#         df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
#         df['Volume_Change'] = df['Volume'].pct_change()
        
#         # 10. Lagged features
#         for lag in [1, 2, 3, 5, 7]:
#             df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
#             df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
#         # 11. Time-based features
#         df['DayOfWeek'] = df.index.dayofweek
#         df['Month'] = df.index.month
#         df['Quarter'] = df.index.quarter
#         df['DayOfMonth'] = df.index.day
#         df['DayOfYear'] = df.index.dayofyear
#         df['WeekOfYear'] = df.index.isocalendar().week
        
#         # Cyclical encoding
#         df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
#         df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
#         df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
#         df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
#         df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
#         df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
#         # Drop NaN values
#         df = df.dropna()
        
#         new_cols = len(df.columns)
        
#         if verbose:
#             print(f"   ✅ Created {new_cols - original_cols} new features")
#             print(f"   ✅ Total features: {new_cols}")
#             print(f"   ✅ Valid samples after cleaning: {len(df)}\n")
        
#         return df


# ============================================================================
# IMPROVED FEATURE ENGINEERING
# ============================================================================

class ImprovedFeatureEngineer:
    """Enhanced feature engineering focused on returns and volatility"""
    
    @staticmethod
    def create_features(df):
        """Create features based on returns"""
        feature_logger = get_logger(__name__)
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
        feature_logger.info(f"✅ Created {len(feature_cols)} features")
        
        return df[feature_cols], df['Close']

# # ============================================================================
# # PYTORCH DATASET
# # ============================================================================

# class TimeSeriesDataset(Dataset):
#     """PyTorch Dataset for time series"""
    
#     def __init__(self, data, seq_len, pred_len):
#         self.data = data
#         self.seq_len = seq_len
#         self.pred_len = pred_len
        
#     def __len__(self):
#         return len(self.data) - self.seq_len - self.pred_len + 1
    
#     def __getitem__(self, idx):
#         x = self.data[idx:idx + self.seq_len]
#         y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 0]  # Close price
#         return torch.FloatTensor(x), torch.FloatTensor(y)
    
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
# IMPROVED INFORMER DATASET (same as transformer improved)
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
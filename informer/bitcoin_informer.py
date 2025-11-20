"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
Production-Ready Implementation with Bitcoin Price Prediction

Based on: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
(Zhou et al., 2021)

Key Features:
- ProbSparse Self-Attention (O(L log L) complexity)
- Self-attention distilling for efficiency
- Generative style decoder
- Complete EDA and visualization suite

Requirements:
pip install torch numpy pandas matplotlib seaborn scikit-learn yfinance scipy

Usage:
python bitcoin_informer.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

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
║           INFORMER: EFFICIENT LONG-SEQUENCE FORECASTING                       ║
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
        
        # Create output directory if it doesn't exist
        os.makedirs('informer', exist_ok=True)
        
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
        
        plt.savefig('informer/01_comprehensive_eda.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: informer/01_comprehensive_eda.png")
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
        plt.savefig('informer/02_advanced_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: informer/02_advanced_analysis.png")
        plt.close()
        
        print()


# ============================================================================
# INFORMER ARCHITECTURE COMPONENTS
# ============================================================================

class TriangularCausalMask:
    """Triangular causal mask for decoder"""
    
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    
    @property
    def mask(self):
        return self._mask


class ProbMask:
    """Probability mask for ProbSparse attention"""
    
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                            torch.arange(H)[None, :, None],
                            index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    """
    ProbSparse Self-Attention
    Key innovation: Only compute attention for top-u queries
    Reduces complexity from O(L^2) to O(L log L)
    """
    
    def __init__(self, mask_flag=True, factor=5, scale=None, 
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert(L_Q == L_V)
            context = V.cumsum(dim=-2)
        
        return context
    
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        attn = torch.softmax(scores, dim=-1)
        
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], 
                  torch.arange(H)[None, :, None],
                  index, :] = attn
            return context_in, attns
        else:
            return context_in, None
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    """Attention layer wrapper"""
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        
        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    """Conv layer for self-attention distilling"""
    
    def __init__(self, c_in):
        super().__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """Informer encoder layer with distilling"""
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """Informer encoder with distilling"""
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        
    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns


class DecoderLayer(nn.Module):
    """Informer decoder layer"""
    
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)


class Decoder(nn.Module):
    """Informer decoder"""
    
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class DataEmbedding(nn.Module):
    """Data embedding with positional encoding"""
    
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """Positional embedding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * 
                    -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Informer(nn.Module):
    """Informer: Efficient Long Sequence Forecasting"""
    
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=512, dropout=0.0, activation='gelu',
                 output_attention=False, distil=True):
        super().__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention
        
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, 
                                    output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, 
                                    output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, 
                                    output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class InformerDataset(Dataset):
    """Dataset for Informer"""
    
    def __init__(self, data, seq_len, label_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


# ============================================================================
# TRAINER
# ============================================================================

class InformerTrainer:
    """Training pipeline for Informer"""
    
    def __init__(self, model, device, learning_rate=0.0001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, label_len, pred_len):
        self.model.train()
        total_loss = 0
        
        for seq_x, seq_y in train_loader:
            seq_x, seq_y = seq_x.to(self.device), seq_y.to(self.device)
            
            dec_inp = torch.zeros_like(seq_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([seq_y[:, :label_len, :], dec_inp], dim=1).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(seq_x, dec_inp)
            
            loss = self.criterion(outputs, seq_y[:, -pred_len:, :])
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, label_len, pred_len):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for seq_x, seq_y in val_loader:
                seq_x, seq_y = seq_x.to(self.device), seq_y.to(self.device)
                
                dec_inp = torch.zeros_like(seq_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([seq_y[:, :label_len, :], dec_inp], dim=1).to(self.device)
                
                outputs = self.model(seq_x, dec_inp)
                loss = self.criterion(outputs, seq_y[:, -pred_len:, :])
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs, label_len, pred_len, patience=10):
        """Train the model"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                          STEP 2: MODEL TRAINING                               ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        train_losses = []
        val_losses = []
        patience_counter = 0
        
        print(f"🚀 Training started")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}\n")
        print("─" * 80)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, label_len, pred_len)
            val_loss = self.validate(val_loader, label_len, pred_len)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'informer/best_informer_model.pth')
                patience_counter = 0
                status = "✅"
            else:
                patience_counter += 1
                status = f"⏳ ({patience_counter}/{patience})"
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] │ "
                      f"Train: {train_loss:.6f} │ Val: {val_loss:.6f} {status}")
            
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        print("─" * 80)
        print(f"✅ Training completed! Best val loss: {self.best_val_loss:.6f}\n")
        
        return train_losses, val_losses


# ============================================================================
# EVALUATION
# ============================================================================

class InformerEvaluator:
    """Model evaluation"""
    
    @staticmethod
    def evaluate(model, test_loader, device, scaler, config):
        """Evaluate model"""
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                         STEP 3: MODEL EVALUATION                              ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")
        
        model.eval()
        predictions = []
        actuals = []
        
        print("📊 Generating predictions...")
        
        with torch.no_grad():
            for seq_x, seq_y in test_loader:
                seq_x, seq_y = seq_x.to(device), seq_y.to(device)
                
                dec_inp = torch.zeros_like(seq_y[:, -config['pred_len']:, :]).float()
                dec_inp = torch.cat([seq_y[:, :config['label_len'], :], dec_inp], dim=1).to(device)
                
                output = model(seq_x, dec_inp)
                predictions.append(output.cpu().numpy())
                actuals.append(seq_y[:, -config['pred_len']:, :].cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Inverse transform
        pred_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        actual_rescaled = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
        
        print("   ✅ Predictions generated\n")
        
        # Calculate metrics
        metrics = InformerEvaluator._calculate_metrics(pred_rescaled, actual_rescaled, config['pred_len'])
        InformerEvaluator._print_metrics(metrics)
        
        return pred_rescaled, actual_rescaled, metrics
    
    @staticmethod
    def _calculate_metrics(predictions, actuals, pred_len):
        """Calculate metrics"""
        metrics = {}
        
        for i in range(pred_len):
            pred = predictions[:, i, 0]
            actual = actuals[:, i, 0]
            
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
            metrics[f'Day_{i+1}'] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            }
        
        return metrics
    
    @staticmethod
    def _print_metrics(metrics):
        """Print metrics"""
        print("📈 EVALUATION METRICS")
        print("─" * 80)
        print(f"{'Forecast':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE':>10}")
        print("─" * 80)
        
        for day, m in metrics.items():
            print(f"{day:<12} ${m['RMSE']:>9,.2f} ${m['MAE']:>9,.2f} "
                  f"{m['R2']:>9.4f} {m['MAPE']:>9.2f}%")
        
        print("─" * 80)
        
        avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])
        avg_mae = np.mean([m['MAE'] for m in metrics.values()])
        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
        
        print(f"{'AVERAGE':<12} ${avg_rmse:>9,.2f} ${avg_mae:>9,.2f} "
              f"{avg_r2:>9.4f} {avg_mape:>9.2f}%")
        print("─" * 80 + "\n")
    
    @staticmethod
    def plot_predictions(predictions, actuals, save_path='informer/03_predictions.png'):
        """Plot predictions"""
        pred_len = predictions.shape[1]
        n_plots = min(pred_len, 4)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(n_plots):
            ax = axes[i]
            
            x = np.arange(len(predictions))
            ax.plot(x, actuals[:, i, 0], label='Actual', linewidth=2, alpha=0.8, color='#2E86AB')
            ax.plot(x, predictions[:, i, 0], label='Predicted', linewidth=2, alpha=0.8, 
                   color='#F18F01', linestyle='--')
            
            r2 = r2_score(actuals[:, i, 0], predictions[:, i, 0])
            mae = mean_absolute_error(actuals[:, i, 0], predictions[:, i, 0])
            
            ax.set_title(f'Day {i+1} Forecast (R²={r2:.3f}, MAE=${mae:,.0f})', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Sample', fontsize=11)
            ax.set_ylabel('Bitcoin Price (USD)', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}\n")
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    CONFIG = {
        'use_real_data': True,
        'start_date': '2020-01-01',
        'end_date': None,
        
        'seq_len': 60,
        'label_len': 48,
        'pred_len': 7,
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,
        'factor': 5,
        'dropout': 0.05,
        
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.0001,
        'patience': 10,
        
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
    
    # EDA
    eda = BitcoinEDA(df, is_real_data=is_real)
    eda.run_full_eda()
    
    # Prepare data
    data = df[['Close']].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Split
    n = len(scaled_data)
    train_size = int(n * CONFIG['train_ratio'])
    val_size = int(n * CONFIG['val_ratio'])
    
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size + val_size]
    test_data = scaled_data[train_size + val_size:]
    
    # Datasets
    train_dataset = InformerDataset(train_data, CONFIG['seq_len'], 
                                   CONFIG['label_len'], CONFIG['pred_len'])
    val_dataset = InformerDataset(val_data, CONFIG['seq_len'],
                                 CONFIG['label_len'], CONFIG['pred_len'])
    test_dataset = InformerDataset(test_data, CONFIG['seq_len'],
                                  CONFIG['label_len'], CONFIG['pred_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Model
    model = Informer(
        enc_in=1, dec_in=1, c_out=1,
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
    
    print(f"🧠 Informer: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Train
    trainer = InformerTrainer(model, device, CONFIG['learning_rate'])
    train_losses, val_losses = trainer.fit(
        train_loader, val_loader, CONFIG['epochs'],
        CONFIG['label_len'], CONFIG['pred_len'], CONFIG['patience']
    )
    
    # Evaluate
    model.load_state_dict(torch.load('informer/best_informer_model.pth'))
    predictions, actuals, metrics = InformerEvaluator.evaluate(
        model, test_loader, device, scaler, CONFIG
    )
    
    # Plot
    InformerEvaluator.plot_predictions(predictions, actuals)
    
    print("\n✅ Informer training and evaluation complete!\n")


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

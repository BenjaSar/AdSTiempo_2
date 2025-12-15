"""
Risk analysis module for investment forecasts
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep learning libraries
import torch

# Import formatting utility
from utils.misc import print_box

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# RISK ANALYSIS
# ============================================================================

class RiskAnalyzer:
    """Analyze investment risk based on forecasts"""
    
    @staticmethod
    def analyze_risk(forecasts, current_price, historical_returns):
        """Calculate risk metrics"""
        print_box("\nBONUS: RISK ANALYSIS")

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

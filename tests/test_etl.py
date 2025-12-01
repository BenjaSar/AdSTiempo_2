import pytest
import numpy as np
import torch
from src.etl import ReturnsDataset

def test_synthetic_data_generation(synthetic_data):
    """Test if synthetic data has correct structure."""
    assert not synthetic_data.empty
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    assert all(col in synthetic_data.columns for col in expected_cols)
    assert len(synthetic_data) > 0

def test_feature_engineering(processed_features):
    """Test feature creation logic."""
    features, prices = processed_features
    
    # Check for NaNs
    assert not features.isnull().values.any(), "Features contain NaNs"
    assert not prices.isnull().values.any(), "Prices contain NaNs"
    
    # Check dimensions
    assert len(features) == len(prices)
    assert 'Returns' in features.columns
    assert 'RSI_14' in features.columns

def test_dataset_shapes(processed_features):
    """Test PyTorch Dataset output shapes."""
    features, prices = processed_features
    seq_len = 10
    pred_len = 5
    
    # Mock scaling
    features_np = features.values
    prices_np = prices.values
    
    dataset = ReturnsDataset(features_np, prices_np, seq_len, pred_len)
    
    # Get one item
    x, y, last_price = dataset[0]
    
    # x shape: (seq_len, n_features)
    assert x.shape == (seq_len, features_np.shape[1])
    # y shape: (pred_len,) -> target is returns
    assert y.shape == (pred_len,)
    # last_price shape: (1,)
    assert last_price.shape == (1,)
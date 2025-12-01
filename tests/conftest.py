import pytest
import pandas as pd
import numpy as np
import torch
import sys
import os

# Add the project root to sys.path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.etl import BitcoinDataLoader, ImprovedFeatureEngineer

@pytest.fixture(scope="session")
def synthetic_data():
    """Generates synthetic dataframe once for the whole test session."""
    df = BitcoinDataLoader.generate_synthetic_data(
        start_date='2023-01-01', 
        end_date='2023-04-01' # Short period for speed
    )
    return df

@pytest.fixture(scope="session")
def processed_features(synthetic_data):
    """Returns processed features and target prices."""
    features, prices = ImprovedFeatureEngineer.create_features(synthetic_data)
    return features, prices

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
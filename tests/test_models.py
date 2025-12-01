import pytest
import torch

# Import Transformer architecture
from src.models.transformer_model import (
    Transformer,
    ImprovedTrainer,
    ImprovedEvaluator,
    FeatureImportance, 
    RiskAnalyzer
)

# Import Informer architecture
from src.models.informer_model import (
    Informer,
    ImprovedInformerTrainer,
    ImprovedInformerEvaluator
)

# Import LSTM architecture
from src.models.lstm_model import (
    LSTMModel, 
    ModelTrainer,
    Evaluator,
    FutureForecaster,
    FeatureImportance, 
    RiskAnalyzer
)

# Import ETL components
from src.etl import (
    BitcoinDataLoader,
    ImprovedFeatureEngineer,
    ReturnsDataset,
    BitcoinEDA
)
@pytest.mark.parametrize("model_class", [Transformer, LSTMModel])
def test_standard_models_forward_pass(model_class):
    """
    Test Transformer and LSTM forward passes by conditionally passing
    only the arguments accepted by each model class.
    """
    batch_size = 4
    seq_len = 10
    input_dim = 12
    pred_len = 7
    
    # Create dummy input: (batch, seq, features)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    # 1. Define arguments common to both models
    kwargs = {
        'input_dim': input_dim,
        'num_layers': 1,
        'pred_len': pred_len,
        'dropout': 0.1 # Added dropout as it's common and optional
    }
    
    # 2. Add model-specific arguments
    if model_class == LSTMModel:
        # LSTMModel uses 'hidden_dim'
        kwargs['hidden_dim'] = 32 
    
    elif model_class == Transformer:
        # Transformer uses 'd_model', 'nhead', 'dim_feedforward'
        kwargs['d_model'] = 32
        kwargs['nhead'] = 4
        kwargs['dim_feedforward'] = 64
        # Note: We rely on the Transformer class definition for argument names.
    
    # Initialize model using dictionary unpacking (**)
    model = model_class(**kwargs)
    
    # Forward pass
    output = model(dummy_input)
    
    # Check shape: (batch, pred_len)
    assert output.shape == (batch_size, pred_len)
    
def test_informer_forward_pass():
    """Test Informer forward pass (requires encoder/decoder inputs)."""
    batch_size = 4
    seq_len = 10
    label_len = 5
    pred_len = 7
    n_features = 12
    
    model = Informer(
        enc_in=n_features, dec_in=n_features, c_out=1,
        seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        d_model=32, n_heads=2, e_layers=1, d_layers=1, d_ff=64
    )
    
    # Informer needs x_enc and x_dec
    x_enc = torch.randn(batch_size, seq_len, n_features)
    x_dec = torch.randn(batch_size, label_len + pred_len, n_features)
    
    output = model(x_enc, x_dec)
    
    # Output shape: (batch, pred_len, c_out)
    assert output.shape == (batch_size, pred_len, 1)
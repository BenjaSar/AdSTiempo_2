import pytest
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from src.etl import ReturnsDataset, InformerReturnsDataset
from src.models.transformer_model import Transformer, ImprovedTrainer
from src.models.informer_model import Informer, ImprovedInformerTrainer

def test_transformer_training_pipeline(processed_features, device):
    """
    E2E Smoke Test: Runs one epoch of Transformer training.
    """
    features, prices = processed_features
    
    # configuration for speed
    SEQ_LEN = 10
    PRED_LEN = 5
    BATCH_SIZE = 4
    
    # Prepare Data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features.values)
    prices_array = prices.values
    
    dataset = ReturnsDataset(scaled_features, prices_array, SEQ_LEN, PRED_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # Prepare Model
    model = Transformer(
        input_dim=scaled_features.shape[1],
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        pred_len=PRED_LEN
    )
    
    # Train for 1 epoch
    trainer = ImprovedTrainer(model, device, learning_rate=0.01, warmup_epochs=0)
    
    # We catch exceptions to fail the test if training crashes
    try:
        train_losses, _ = trainer.fit(loader, loader, epochs=1, patience=1)
        assert len(train_losses) == 1
        assert train_losses[0] > 0
    except Exception as e:
        pytest.fail(f"Transformer training pipeline crashed: {e}")

def test_informer_training_pipeline(processed_features, device):
    """
    E2E Smoke Test: Runs one epoch of Informer training.
    """
    features, prices = processed_features
    
    # configuration
    SEQ_LEN = 10
    LABEL_LEN = 5
    PRED_LEN = 5
    BATCH_SIZE = 4
    
    # Prepare Data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features.values)
    prices_array = prices.values
    
    dataset = InformerReturnsDataset(
        scaled_features, prices_array, SEQ_LEN, LABEL_LEN, PRED_LEN
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # Prepare Model
    model = Informer(
        enc_in=scaled_features.shape[1],
        dec_in=scaled_features.shape[1],
        c_out=1,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        d_model=16, n_heads=2, e_layers=1, d_layers=1
    )
    
    # Train
    trainer = ImprovedInformerTrainer(model, device, learning_rate=0.01, warmup_epochs=0)
    
    try:
        train_losses, _ = trainer.fit(loader, loader, epochs=1, patience=1)
        assert len(train_losses) == 1
    except Exception as e:
        pytest.fail(f"Informer training pipeline crashed: {e}")
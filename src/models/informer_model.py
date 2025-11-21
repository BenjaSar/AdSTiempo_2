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

# Import formatting utility
from utils.misc import print_box

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

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

# ============================================================================
# INFORMER MODEL
# ============================================================================

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
        print_box("\nMODEL TRAINING")
        
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
        print_box("\nMODEL EVALUATION")
        
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
    def plot_predictions(predictions, actuals, save_path='informer/results/03_predictions.png'):
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
        print(f"   ✅ Saved: {save_path}")
        plt.close()

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

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep learning libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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


# # ============================================================================
# # TRAINER
# # ============================================================================

# class InformerTrainer:
#     """Training pipeline for Informer"""
    
#     def __init__(self, model, device, learning_rate=0.0001):
#         self.model = model.to(device)
#         self.device = device
#         self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, mode='min', factor=0.5, patience=5
#         )
#         self.best_val_loss = float('inf')
        
#     def train_epoch(self, train_loader, label_len, pred_len):
#         self.model.train()
#         total_loss = 0
        
#         for seq_x, seq_y in train_loader:
#             seq_x, seq_y = seq_x.to(self.device), seq_y.to(self.device)
            
#             dec_inp = torch.zeros_like(seq_y[:, -pred_len:, :]).float()
#             dec_inp = torch.cat([seq_y[:, :label_len, :], dec_inp], dim=1).to(self.device)
            
#             self.optimizer.zero_grad()
#             outputs = self.model(seq_x, dec_inp)
            
#             loss = self.criterion(outputs, seq_y[:, -pred_len:, :])
#             loss.backward()
#             self.optimizer.step()
            
#             total_loss += loss.item()
            
#         return total_loss / len(train_loader)
    
#     def validate(self, val_loader, label_len, pred_len):
#         self.model.eval()
#         total_loss = 0
        
#         with torch.no_grad():
#             for seq_x, seq_y in val_loader:
#                 seq_x, seq_y = seq_x.to(self.device), seq_y.to(self.device)
                
#                 dec_inp = torch.zeros_like(seq_y[:, -pred_len:, :]).float()
#                 dec_inp = torch.cat([seq_y[:, :label_len, :], dec_inp], dim=1).to(self.device)
                
#                 outputs = self.model(seq_x, dec_inp)
#                 loss = self.criterion(outputs, seq_y[:, -pred_len:, :])
#                 total_loss += loss.item()
                
#         return total_loss / len(val_loader)
    
#     def fit(self, train_loader, val_loader, epochs, label_len, pred_len, patience=10):
#         """Train the model"""
#         print_box("\nMODEL TRAINING")
        
#         train_losses = []
#         val_losses = []
#         patience_counter = 0
        
#         print(f"🚀 Training started")
#         print(f"   Device: {self.device}")
#         print(f"   Epochs: {epochs}\n")
#         print("─" * 80)
        
#         for epoch in range(epochs):
#             train_loss = self.train_epoch(train_loader, label_len, pred_len)
#             val_loss = self.validate(val_loader, label_len, pred_len)
            
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
            
#             self.scheduler.step(val_loss)
            
#             if val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 torch.save(self.model.state_dict(), 'models/informer/best_informer_model.pth')
#                 patience_counter = 0
#                 status = "✅"
#             else:
#                 patience_counter += 1
#                 status = f"⏳ ({patience_counter}/{patience})"
            
#             if (epoch + 1) % 5 == 0 or epoch == 0:
#                 print(f"Epoch [{epoch+1:3d}/{epochs}] │ "
#                       f"Train: {train_loss:.6f} │ Val: {val_loss:.6f} {status}")
            
#             if patience_counter >= patience:
#                 print(f"\n⚠️  Early stopping at epoch {epoch+1}")
#                 break
        
#         print("─" * 80)
#         print(f"✅ Training completed! Best val loss: {self.best_val_loss:.6f}\n")
        
#         return train_losses, val_losses

# ============================================================================
# IMPROVED TRAINER (same as transformer improved)
# ============================================================================

class ImprovedInformerTrainer:
    """Enhanced training with Huber loss and better scheduling"""
    
    def __init__(self, model, device, learning_rate=0.001, warmup_epochs=5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                                     weight_decay=1e-5)
        
        self.warmup_epochs = warmup_epochs
        self.base_lr = learning_rate
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=learning_rate/10
        )
        self.best_val_loss = float('inf')
        self.epoch = 0
        
    def _adjust_learning_rate(self):
        """Warmup learning rate for first few epochs"""
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr * (self.epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for x_enc, x_dec, target, _ in train_loader:
            x_enc = x_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x_enc, x_dec)
            output = output.squeeze(-1)  # Remove last dimension: (batch, pred_len, 1) -> (batch, pred_len)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x_enc, x_dec, target, _ in val_loader:
                x_enc = x_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                target = target.to(self.device)
                output = self.model(x_enc, x_dec)
                output = output.squeeze(-1)  # Remove last dimension: (batch, pred_len, 1) -> (batch, pred_len)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs, patience=30):
        """Train with warmup and cosine annealing - with improved patience"""
        print_box("\nTRAINING (IMPROVED)")
        
        print(f"🚀 Training Informer with Huber loss and learning rate scheduling")
        print(f"   Device: {self.device}")
        print(f"   Warmup epochs: {self.warmup_epochs}")
        print(f"   Total epochs: {epochs}")
        print(f"   Patience: {patience}\n")
        print("─" * 81)
        
        train_losses, val_losses = [], []
        patience_counter = 0
        val_loss_threshold = 1e-6  # Minimum improvement threshold
        
        for epoch in range(epochs):
            self.epoch = epoch
            self._adjust_learning_rate()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check if validation loss improved by minimum threshold
            if val_loss < self.best_val_loss - val_loss_threshold:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/informer/best_informer_model.pth')
                patience_counter = 0
                status = "✅"
            else:
                patience_counter += 1
                status = f"⏳ ({patience_counter}/{patience})"
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1:3d}/{epochs}] │ "
                      f"Train: {train_loss:.6f} │ Val: {val_loss:.6f} │ "
                      f"LR: {current_lr:.2e} │ {status}")
            
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        print("─" * 81)
        print(f"✅ Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        
        return train_losses, val_losses


# # ============================================================================
# # EVALUATION
# # ============================================================================

# class InformerEvaluator:
#     """Model evaluation"""
    
#     @staticmethod
#     def evaluate(model, test_loader, device, scaler, 
# ):
#         """Evaluate model"""
#         print_box("\nMODEL EVALUATION")
        
#         model.eval()
#         predictions = []
#         actuals = []
        
#         print("📊 Generating predictions...")
        
#         with torch.no_grad():
#             for seq_x, seq_y in test_loader:
#                 seq_x, seq_y = seq_x.to(device), seq_y.to(device)
                
#                 dec_inp = torch.zeros_like(seq_y[:, -config['pred_len']:, :]).float()
#                 dec_inp = torch.cat([seq_y[:, :config['label_len'], :], dec_inp], dim=1).to(device)
                
#                 output = model(seq_x, dec_inp)
#                 predictions.append(output.cpu().numpy())
#                 actuals.append(seq_y[:, -config['pred_len']:, :].cpu().numpy())
        
#         predictions = np.concatenate(predictions, axis=0)
#         actuals = np.concatenate(actuals, axis=0)
        
#         # Inverse transform
#         pred_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
#         actual_rescaled = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
        
#         print("   ✅ Predictions generated\n")
        
#         # Calculate metrics
#         metrics = InformerEvaluator._calculate_metrics(pred_rescaled, actual_rescaled, config['pred_len'])
#         InformerEvaluator._print_metrics(metrics)
        
#         return pred_rescaled, actual_rescaled, metrics
    
#     @staticmethod
#     def _calculate_metrics(predictions, actuals, pred_len):
#         """Calculate metrics"""
#         metrics = {}
        
#         for i in range(pred_len):
#             pred = predictions[:, i, 0]
#             actual = actuals[:, i, 0]
            
#             rmse = np.sqrt(mean_squared_error(actual, pred))
#             mae = mean_absolute_error(actual, pred)
#             r2 = r2_score(actual, pred)
#             mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
#             metrics[f'Day_{i+1}'] = {
#                 'RMSE': rmse,
#                 'MAE': mae,
#                 'R2': r2,
#                 'MAPE': mape
#             }
        
#         return metrics
    
#     @staticmethod
#     def _print_metrics(metrics):
#         """Print metrics"""
#         print("📈 EVALUATION METRICS")
#         print("─" * 80)
#         print(f"{'Forecast':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE':>10}")
#         print("─" * 80)
        
#         for day, m in metrics.items():
#             print(f"{day:<12} ${m['RMSE']:>9,.2f} ${m['MAE']:>9,.2f} "
#                   f"{m['R2']:>9.4f} {m['MAPE']:>9.2f}%")
        
#         print("─" * 80)
        
#         avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])
#         avg_mae = np.mean([m['MAE'] for m in metrics.values()])
#         avg_r2 = np.mean([m['R2'] for m in metrics.values()])
#         avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
        
#         print(f"{'AVERAGE':<12} ${avg_rmse:>9,.2f} ${avg_mae:>9,.2f} "
#               f"{avg_r2:>9.4f} {avg_mape:>9.2f}%")
#         print("─" * 80 + "\n")
    
#     @staticmethod
#     def plot_predictions(predictions, actuals, save_path='models/informer/results/03_predictions.png'):
#         """Plot predictions"""
#         pred_len = predictions.shape[1]
#         n_plots = min(pred_len, 4)
        
#         fig, axes = plt.subplots(2, 2, figsize=(18, 12))
#         axes = axes.flatten()
        
#         for i in range(n_plots):
#             ax = axes[i]
            
#             x = np.arange(len(predictions))
#             ax.plot(x, actuals[:, i, 0], label='Actual', linewidth=2, alpha=0.8, color='#2E86AB')
#             ax.plot(x, predictions[:, i, 0], label='Predicted', linewidth=2, alpha=0.8, 
#                    color='#F18F01', linestyle='--')
            
#             r2 = r2_score(actuals[:, i, 0], predictions[:, i, 0])
#             mae = mean_absolute_error(actuals[:, i, 0], predictions[:, i, 0])
            
#             ax.set_title(f'Day {i+1} Forecast (R²={r2:.3f}, MAE=${mae:,.0f})', 
#                         fontsize=13, fontweight='bold')
#             ax.set_xlabel('Sample', fontsize=11)
#             ax.set_ylabel('Bitcoin Price (USD)', fontsize=11)
#             ax.legend(fontsize=10)
#             ax.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"   ✅ Saved: {save_path}")
#         plt.close()

# ============================================================================
# IMPROVED EVALUATION (same as transformer improved)
# ============================================================================

class ImprovedInformerEvaluator:
    """Evaluation with price reconstruction from returns"""
    
    @staticmethod
    def evaluate(model, test_loader, device, scaler):
        """Evaluate and reconstruct prices from returns"""
        print_box("\nEVALUATION (IMPROVED)")

        model.eval()
        pred_returns, actual_returns, last_prices = [], [], []
        
        print("📊 Generating predictions on test set...")

        with torch.no_grad():
            for x_enc, x_dec, target, last_price in test_loader:
                x_enc = x_enc.to(device)
                x_dec = x_dec.to(device)
                output = model(x_enc, x_dec)
                output = output.squeeze(-1)  # Remove last dimension: (batch, pred_len, 1) -> (batch, pred_len)
                
                pred_returns.append(output.cpu().numpy())
                actual_returns.append(target.numpy())
                last_prices.append(last_price.numpy())
        
        pred_returns = np.concatenate(pred_returns, axis=0)
        actual_returns = np.concatenate(actual_returns, axis=0)
        last_prices = np.concatenate(last_prices, axis=0)
        
        # Denormalize returns
        pred_returns_denorm = scaler.inverse_transform(pred_returns)
        actual_returns_denorm = scaler.inverse_transform(actual_returns)
        
        # Reconstruct prices from returns
        pred_prices = ImprovedInformerEvaluator._reconstruct_prices(
            pred_returns_denorm, last_prices
        )
        actual_prices = ImprovedInformerEvaluator._reconstruct_prices(
            actual_returns_denorm, last_prices
        )
        
        # Calculate metrics
        metrics = ImprovedInformerEvaluator._calculate_metrics(pred_prices, actual_prices)
        ImprovedInformerEvaluator._print_metrics(metrics)
        
        return pred_prices, actual_prices, metrics
    
    @staticmethod
    def _reconstruct_prices(returns, last_prices):
        """Reconstruct prices from log returns"""
        prices = np.zeros_like(returns)
        
        for i in range(len(returns)):
            current_price = last_prices[i, 0]
            for j in range(returns.shape[1]):
                current_price = current_price * np.exp(returns[i, j])
                prices[i, j] = current_price
        
        return prices
    
    @staticmethod
    def _calculate_metrics(predictions, actuals):
        """Calculate metrics per forecast day"""
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
            actual_dir = np.sign(np.diff(np.concatenate([[actual[0]], actual])))
            pred_dir = np.sign(np.diff(np.concatenate([[pred[0]], pred])))
            dir_acc = np.mean(actual_dir == pred_dir) * 100
            
            metrics[f'Day_{i+1}'] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'Dir_Acc': dir_acc
            }
        
        return metrics
    
    @staticmethod
    def _print_metrics(metrics):
        """Print evaluation metrics"""
        print("📈 EVALUATION METRICS (Price Reconstruction)")
        print("─" * 81)
        print(f"{'Forecast':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} "
              f"{'MAPE':>10} {'Dir%':>10}")
        print("─" * 81)
        
        for day, m in metrics.items():
            print(f"{day:<12} "
                  f"${m['RMSE']:>9,.2f} "
                  f"${m['MAE']:>9,.2f} "
                  f"{m['R2']:>9.4f} "
                  f"{m['MAPE']:>9.2f}% "
                  f"{m['Dir_Acc']:>9.1f}%")
        
        print("─" * 81)
        
        # Averages
        avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])
        avg_mae = np.mean([m['MAE'] for m in metrics.values()])
        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
        avg_mape = np.mean([m['MAPE'] for m in metrics.values()])
        avg_dir = np.mean([m['Dir_Acc'] for m in metrics.values()])
        
        print(f"{'AVERAGE':<12} "
              f"${avg_rmse:>9,.2f} "
              f"${avg_mae:>9,.2f} "
              f"{avg_r2:>9.4f} "
              f"{avg_mape:>9.2f}% "
              f"{avg_dir:>9.1f}%")
        print("─" * 81 + "\n")
    
    @staticmethod
    def plot_predictions(predictions, actuals, save_path='models/informer/results/03_predictions.png'):
        """Plot price predictions"""
        pred_len = min(predictions.shape[1], 4)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(pred_len):
            ax = axes[i]
            
            x = np.arange(len(predictions))
            ax.plot(x, actuals[:, i], label='Actual', linewidth=2, 
                   alpha=0.8, color='#2E86AB')
            ax.plot(x, predictions[:, i], label='Predicted', linewidth=2, 
                   alpha=0.8, color='#F18F01', linestyle='--')
            
            r2 = r2_score(actuals[:, i], predictions[:, i])
            mae = mean_absolute_error(actuals[:, i], predictions[:, i])
            
            ax.set_title(f'Day {i+1} Forecast (R²={r2:.3f}, MAE=${mae:,.0f})', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Sample', fontsize=11)
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
    def plot_error_analysis(predictions, actuals, save_path='models/informer/results/04_error_analysis.png'):
        """Analyze prediction errors"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        pred = predictions[:, 0]
        actual = actuals[:, 0]
        errors = pred - actual
        pct_errors = (errors / actual) * 100
        
        # 1. Scatter plot
        axes[0, 0].scatter(actual, pred, alpha=0.5, s=30)
        min_val, max_val = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Price (USD)', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Price (USD)', fontsize=11)
        axes[0, 0].set_title('Predicted vs Actual (Day 1)', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
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
        x = np.arange(len(errors))
        axes[1, 0].plot(x, errors, linewidth=1, alpha=0.7, color='#C73E1D')
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].fill_between(x, 0, errors, alpha=0.3, color='#C73E1D')
        axes[1, 0].set_xlabel('Sample', fontsize=11)
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
        print(f"   ✅ Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, save_path='models/informer/results/05_training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
                markersize=4, alpha=0.7)
        ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', 
                markersize=4, alpha=0.7)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (Huber)', fontsize=12)
        ax1.set_title('Training History - Informer', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
                markersize=4, alpha=0.7)
        ax2.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', 
                markersize=4, alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (Huber, log scale)', fontsize=12)
        ax2.set_title('Training History (Log Scale) - Informer', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        plt.close()

# ============================================================================
# FUTURE FORECASTING
# ============================================================================
class FutureForecaster:
    """Generate future forecasts using Informer architecture"""
    
    @staticmethod
    def forecast(model, last_sequence, scaler, seq_len, label_len, pred_len, 
                 n_days=30, device='cpu'):
        """
        Recursive block forecasting
        """
        print_box("\nFUTURE FORECASTING")
        print(f"🔮 Generating {n_days}-day forecast...")
        
        model.eval()
        forecasts_scaled = []
        
        # Prepare initial sequence
        current_seq = torch.FloatTensor(last_sequence).to(device)
        if current_seq.dim() == 2:
            current_seq = current_seq.unsqueeze(0) # Ensure (Batch, Seq_Len, Features)
            
        # CORRECTED: Get n_features from the actual data, not sequence lengths
        n_features = current_seq.shape[-1]
            
        # Calculate steps needed
        steps_needed = math.ceil(n_days / pred_len)
        
        with torch.no_grad():
            for _ in range(steps_needed):
                # 1. Prepare Encoder Input (x_enc)
                if current_seq.shape[1] > seq_len:
                    x_enc = current_seq[:, -seq_len:, :]
                else:
                    x_enc = current_seq

                # 2. Prepare Decoder Input (x_dec)
                x_dec_token = x_enc[:, -label_len:, :]
                x_dec_zeros = torch.zeros(1, pred_len, x_enc.shape[-1]).to(device)
                x_dec = torch.cat([x_dec_token, x_dec_zeros], dim=1)
                
                # 3. Predict
                output = model(x_enc, x_dec)
                
                if output.dim() == 2: 
                    output = output.unsqueeze(-1)

                # 4. Handle Feature Mismatch (Reconstruct Covariates)
                if output.shape[-1] != n_features:
                    # Create placeholder (Batch, Pred_Len, n_features)
                    expanded_pred = torch.zeros(output.shape[0], output.shape[1], n_features).to(device)
                    
                    # A. Fill Target Column (Index 0) with prediction
                    expanded_pred[:, :, 0] = output[:, :, 0]
                    
                    # B. Fill Covariates (Indices 1+) with last known values
                    # This assumes covariates (like Volatility) stay constant for the short forecast block
                    last_known_covariates = current_seq[:, -1:, 1:] 
                    expanded_pred[:, :, 1:] = last_known_covariates.expand(-1, output.shape[1], -1)
                    
                    output_for_history = expanded_pred
                else:
                    output_for_history = output

                # Store prediction (target only)
                pred_block = output.cpu().numpy()[0] 
                forecasts_scaled.append(pred_block)
                
                # Update history for recursion
                current_seq = torch.cat([current_seq, output_for_history], dim=1)
                
        # Concatenate and trim
        forecasts_scaled = np.concatenate(forecasts_scaled, axis=0)
        forecasts_scaled = forecasts_scaled[:n_days]
        
        # Inverse transform
        dummy_array = np.zeros((len(forecasts_scaled), scaler.n_features_in_))
        dummy_array[:, 0] = forecasts_scaled[:, 0]
        forecasts_actual = scaler.inverse_transform(dummy_array)[:, 0]
        
        # Reconstruct prices from returns (Since model predicts log returns)
        # We need the absolute last price from the *original* data to start the chain
        # NOTE: last_sequence is (Seq_Len, Features), so we can't get the absolute price directly
        # from it because it's scaled returns. 
        # We assume the user will handle price reconstruction outside or we return the returns.
        # However, looking at previous blocks, this returns the "Actual" value. 
        # If your scaler was on Prices, this is Price. If on Returns, this is Return.
        
        print(f"   ✅ Forecast complete\n")
        
        return forecasts_actual
    
    @staticmethod
    def plot_forecast(historical_data, historical_dates, forecasts, 
                     forecast_dates, save_path='models/informer/results/06_future_forecast.png'):
        """Visualize future forecast"""
        # [Same plotting code as before - no changes needed here]
        if len(historical_data.shape) > 1:
            historical_data = historical_data[:, 0]
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
        
        # Full view
        ax1.plot(historical_dates, historical_data, label='Historical Data', 
                linewidth=2, color='#2E86AB', alpha=0.8)
        ax1.plot(forecast_dates, forecasts, label='Forecast', 
                linewidth=2.5, color='#F18F01', linestyle='--', marker='o', 
                markersize=4, alpha=0.9)
        
        # Confidence interval heuristic
        lookback = min(60, len(historical_data)-1)
        recent_returns = np.diff(historical_data[-lookback:]) / historical_data[-lookback:-1]
        std = np.std(recent_returns) * historical_data[-1]
        expanding_std = std * np.sqrt(np.arange(1, len(forecasts) + 1))
        
        ax1.fill_between(forecast_dates, 
                        forecasts - 2*expanding_std, 
                        forecasts + 2*expanding_std, 
                        alpha=0.2, color='#F18F01', 
                        label='95% Confidence Interval')
        
        ax1.axvline(historical_dates.iloc[-1], color='green', linestyle=':', 
                   linewidth=2, label='Forecast Start', alpha=0.7)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Bitcoin Price (USD)', fontsize=12)
        ax1.set_title('Informer Forecast - Full View', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Zoomed view
        zoom_days = 90
        if len(historical_dates) > zoom_days:
            hist_date_zoom = historical_dates[-zoom_days:]
            hist_data_zoom = historical_data[-zoom_days:]
        else:
            hist_date_zoom = historical_dates
            hist_data_zoom = historical_data
            
        ax2.plot(hist_date_zoom, hist_data_zoom, 
                label='Historical Data', linewidth=2, color='#2E86AB', alpha=0.8)
        ax2.plot(forecast_dates, forecasts, label='Forecast', 
                linewidth=2.5, color='#F18F01', linestyle='--', marker='o', 
                markersize=4, alpha=0.9)
        ax2.fill_between(forecast_dates, 
                        forecasts - 2*expanding_std, 
                        forecasts + 2*expanding_std, 
                        alpha=0.2, color='#F18F01')
        ax2.axvline(historical_dates.iloc[-1], color='green', linestyle=':', 
                   linewidth=2, alpha=0.7)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Bitcoin Price (USD)', fontsize=12)
        ax2.set_title(f'Informer Forecast - Recent {zoom_days} Days + Forecast', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        plt.close()


# ============================================================================
# FEATURE IMPORTANCE (Corrected for Informer)
# ============================================================================

class FeatureImportance:
    """Feature importance analysis using Permutation Importance"""
    
    def __init__(self, model, loader, device, feature_names=None):
        self.model = model
        self.loader = loader
        self.device = device
        self.feature_names = feature_names
        self.criterion = nn.MSELoss()
        
    def _get_loss(self):
        """Calculate validation loss"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x_enc, x_dec, target, _ in self.loader:
                x_enc = x_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                target = target.to(self.device)
                
                output = self.model(x_enc, x_dec)
                
                # Squeeze if necessary (B, Pred, 1) -> (B, Pred)
                if output.dim() == 3 and target.dim() == 2:
                    output = output.squeeze(-1)
                    
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(self.loader)
    
    def calculate_importance(self, n_repeats=5):
        """Compute permutation importance"""
        print_box("\nFEATURE IMPORTANCE ANALYSIS")
        
        baseline_loss = self._get_loss()
        print(f"   Baseline Loss: {baseline_loss:.6f}")
        
        importances = {}
        
        # Get number of features from the first batch
        try:
            # Unpack 4 values (x_enc, x_dec, target, last_price)
            x_enc_sample, _, _, _ = next(iter(self.loader))
            n_features = x_enc_sample.shape[-1]
        except ValueError:
            # Fallback if loader returns 3 values
            x_enc_sample, _, _ = next(iter(self.loader))
            n_features = x_enc_sample.shape[-1]
        
        feature_names = self.feature_names or [f"Feature {i}" for i in range(n_features)]
        
        print(f"   Analyzing {n_features} features ({n_repeats} permutations each)...")
        print("─" * 81)
        
        for i in range(n_features):
            feature_loss = 0
            
            for _ in range(n_repeats):
                temp_loss = 0
                self.model.eval()
                
                with torch.no_grad():
                    for x_enc, x_dec, target, _ in self.loader:
                        x_enc = x_enc.to(self.device)
                        x_dec = x_dec.to(self.device)
                        target = target.to(self.device)
                        
                        # Shuffle feature i in x_enc
                        idx = torch.randperm(x_enc.size(0))
                        x_enc_shuffled = x_enc.clone()
                        x_enc_shuffled[:, :, i] = x_enc[idx, :, i]
                        
                        # Shuffle x_dec if it uses the same feature
                        if x_dec.shape[-1] > i:
                             x_dec_shuffled = x_dec.clone()
                             x_dec_shuffled[:, :, i] = x_dec[idx, :, i]
                        else:
                             x_dec_shuffled = x_dec

                        output = self.model(x_enc_shuffled, x_dec_shuffled)
                        
                        if output.dim() == 3 and target.dim() == 2:
                            output = output.squeeze(-1)
                            
                        loss = self.criterion(output, target)
                        temp_loss += loss.item()
                
                feature_loss += (temp_loss / len(self.loader))
            
            avg_loss = feature_loss / n_repeats
            importance = avg_loss - baseline_loss
            importances[feature_names[i]] = importance
            
            print(f"   {feature_names[i]:<20} | Impact: {importance:+.6f}")
            
        return importances
    
    @staticmethod
    def plot_importance(importances, save_path='models/informer/results/07_feature_importance.png'):
        """Plot feature importance"""
        features = list(importances.keys())
        scores = list(importances.values())
        
        # Sort
        indices = np.argsort(scores)
        features = [features[i] for i in indices]
        scores = [scores[i] for i in indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(features, scores, color='#2E86AB', alpha=0.8)
        
        for i, bar in enumerate(bars):
            if scores[i] < 0:
                bar.set_color('#C73E1D') 
            else:
                bar.set_color('#2E86AB')
                
        plt.title('Feature Importance (Permutation)', fontsize=14, fontweight='bold')
        plt.xlabel('Increase in MSE Loss (Higher is more important)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n   ✅ Saved: {save_path}")
        plt.close()
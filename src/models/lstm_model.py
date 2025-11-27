import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import formatting utility
from utils.misc import print_box

# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, pred_len, dropout=0.1):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of features in the hidden state
            num_layers (int): Number of stacked LSTM layers
            pred_len (int): Forecast horizon (output dimension)
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        # LSTM layer
        # batch_first=True causes input/output tensors to be (batch_size, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout between LSTM layers
        )
        
        # Layer normalization (often helpful)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Fully connected output layers to map from hidden_dim to pred_len
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, pred_len)
        
    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        
        # Initialize hidden and cell states (defaults to zeros if not provided)
        h_0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).to(src.device)
        c_0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).to(src.device)
        
        # Pass through LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        # c_n shape: (num_layers, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(src)
        
        # We use the output from the last time step
        # This contains the summarized information of the entire sequence
        last_time_step_out = lstm_out[:, -1, :]
        
        # Apply layer norm
        last_time_step_out = self.layer_norm(last_time_step_out)
        
        # Pass through output layers
        output = self.relu(self.fc1(last_time_step_out))
        output = self.dropout(output)
        output = self.fc2(output)
        
        # Final output shape: (batch_size, pred_len)
        return output


# ============================================================================
# TRAINER
# ============================================================================

class ModelTrainer:
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

        for batch_x, batch_y, _ in train_loader:
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
            for batch_x, batch_y, _ in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs, patience=10):
        """Train the model"""
        print_box("\nMODEL TRAINING")

        train_losses = []
        val_losses = []
        patience_counter = 0
        
        print(f"🚀 Training started with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Early stopping patience: {patience}\n")
        print("─" * 81)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'lstm/best_lstm_model.pth')
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
        
        print("─" * 81)
        print(f"✅ Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        
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
        plt.savefig('lstm/results/03_training_history.png', dpi=300, bbox_inches='tight')
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
        print_box("\nMODEL EVALUATION")
        
        model.eval()
        predictions = []
        actuals = []
        
        print("📊 Generating predictions on test set...")
        
        with torch.no_grad():
            for batch_x, batch_y, _ in test_loader:
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
        print("─" * 81)
        
        # Header
        print(f"{'Forecast':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} "
              f"{'MAPE':>10} {'Dir_Acc':>10}")
        print("─" * 81)
        
        # Print each day
        for day, m in metrics.items():
            print(f"{day:<12} "
                  f"${m['RMSE']:>9,.2f} "
                  f"${m['MAE']:>9,.2f} "
                  f"{m['R2']:>9.4f} "
                  f"{m['MAPE']:>9.2f}% "
                  f"{m['Direction_Accuracy']:>9.1f}%")
        
        print("─" * 81)
        
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
        print("─" * 81 + "\n")
    
    @staticmethod
    def plot_predictions(predictions, actuals, dates, save_path='lstm/results/04_predictions.png'):
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
    def plot_error_analysis(predictions, actuals, dates, save_path='lstm/results/05_error_analysis.png'):
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
        print(f"   ✅ Saved: {save_path}")
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
        print_box("\nFUTURE FORECASTING")
        
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
        print("─" * 81)
        print(f"Next day price:    ${forecasts[0]:,.2f}")
        print(f"7-day price:       ${forecasts[6]:,.2f}")
        print(f"14-day price:      ${forecasts[13]:,.2f}")
        print(f"30-day price:      ${forecasts[-1]:,.2f}")
        print(f"Expected return:   {((forecasts[-1] / forecasts[0] - 1) * 100):.2f}%")
        print("─" * 81 + "\n")
        
        return forecasts
    
    @staticmethod
    def plot_forecast(historical_data, historical_dates, forecasts, 
                     forecast_dates, save_path='lstm/results/06_future_forecast.png'):
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
        print(f"   ✅ Saved: {save_path}")
        plt.close()

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
            for batch_x, batch_y, _ in test_loader:
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
                    for batch_x, batch_y, _ in test_loader:
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
        print("─" * 81)
        print(f"{'Rank':<6} {'Feature':<30} {'Importance Score':>20}")
        print("─" * 81)
        
        for rank, (feat, score) in enumerate(sorted_importance[:20], 1):
            print(f"{rank:<6} {feat:<30} {score:>20.6f}")
        
        print("─" * 81 + "\n")
        
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
        plt.savefig('lstm/results/07_feature_importance.png', dpi=300, bbox_inches='tight')
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
        print("─" * 81)
        print(f"Expected 30-Day Return:        {expected_return:>10.2f}%")
        print(f"Forecast Volatility (Annual):  {forecast_volatility:>10.2f}%")
        print(f"Historical Volatility:         {historical_vol:>10.2f}%")
        print(f"Value at Risk (95%, 1-day):    ${var_95:>10,.2f}")
        print(f"Maximum Forecast Drawdown:     {max_drawdown:>10.2f}%")
        print(f"Sharpe Ratio (Rf=0):           {sharpe:>10.2f}")
        print("─" * 81)
        
        # Risk assessment
        print("\n🎯 RISK ASSESSMENT")
        print("─" * 81)
        
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
        
        print("─" * 81 + "\n")
        
        return {
            'expected_return': expected_return,
            'forecast_volatility': forecast_volatility,
            'historical_volatility': historical_vol,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe
        }

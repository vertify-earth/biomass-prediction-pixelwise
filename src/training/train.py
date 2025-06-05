import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import logging
from datetime import datetime
from tqdm import tqdm
import joblib

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiomassModelTrainer:
    """Trainer for biomass prediction models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_r2_values = []
        self.val_r2_values = []
        self.best_epoch = 0
        
        # Set random seeds for reproducibility
        self._set_seed(config.random_seed)
    
    def _set_seed(self, seed):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train(self, model, train_loader, val_loader):
        """Train the model using early stopping"""
        model = model.to(self.device)
        criterion = nn.MSELoss()
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        start_time = time.time()
        
        logger.info(f"Starting training for up to {self.config.max_epochs} epochs...")
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_outputs = []
            train_targets = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
            
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                train_outputs.extend(outputs.detach().cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_outputs.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            self.val_losses.append(val_loss)
            
            # Calculate R² scores
            train_r2 = r2_score(train_targets, train_outputs)
            val_r2 = r2_score(val_targets, val_outputs)
            
            self.train_r2_values.append(train_r2)
            self.val_r2_values.append(val_r2)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                self.best_epoch = epoch
            else:
                patience_counter += 1
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}"
            )
            
            # Check early stopping condition
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best model from epoch {self.best_epoch+1}")
        
        return model
    
    def evaluate(self, model, test_loader, original_scale=False, epsilon=1.0):
        """Evaluate the model on test data"""
        model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        # Convert arrays
        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)
        
        # Calculate metrics on log scale
        log_metrics = {
            'r2': r2_score(all_targets, all_outputs),
            'rmse': np.sqrt(mean_squared_error(all_targets, all_outputs)),
            'mae': mean_absolute_error(all_targets, all_outputs)
        }
        
        # If we used log transform, also calculate metrics on original scale
        if original_scale:
            orig_outputs = np.exp(all_outputs) - epsilon
            orig_targets = np.exp(all_targets) - epsilon
            
            # Ensure non-negative values
            orig_outputs = np.maximum(orig_outputs, 0)
            orig_targets = np.maximum(orig_targets, 0)
            
            orig_metrics = {
                'r2': r2_score(orig_targets, orig_outputs),
                'rmse': np.sqrt(mean_squared_error(orig_targets, orig_outputs)),
                'mae': mean_absolute_error(orig_targets, orig_outputs)
            }
            
            results = {
                'log_scale': {
                    'predictions': all_outputs,
                    'targets': all_targets,
                    'metrics': log_metrics
                },
                'original_scale': {
                    'predictions': orig_outputs,
                    'targets': orig_targets,
                    'metrics': orig_metrics
                }
            }
        else:
            results = {
                'metrics': log_metrics,
                'predictions': all_outputs,
                'targets': all_targets
            }
        
        # Log evaluation results
        logger.info(f"Evaluation results:")
        if original_scale:
            logger.info(f"  Log scale - R²: {log_metrics['r2']:.4f}, RMSE: {log_metrics['rmse']:.4f}, MAE: {log_metrics['mae']:.4f}")
            logger.info(f"  Original scale - R²: {orig_metrics['r2']:.4f}, RMSE: {orig_metrics['rmse']:.4f}, MAE: {orig_metrics['mae']:.4f}")
        else:
            logger.info(f"  R²: {log_metrics['r2']:.4f}, RMSE: {log_metrics['rmse']:.4f}, MAE: {log_metrics['mae']:.4f}")
        
        return results
    
    def save_model(self, model, data_dict, results, save_dir=None):
        """Save model and related data"""
        if save_dir is None:
            save_dir = os.path.join(self.config.results_dir, 'models', 
                                   f"{self.config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Save model weights
        model_path = os.path.join(save_dir, 'model.pt')
        torch.save(model.state_dict(), model_path)
        
        # 2. Save model architecture
        arch_path = os.path.join(save_dir, 'architecture.txt')
        with open(arch_path, 'w') as f:
            f.write(str(model))
        
        # 3. Save feature scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        joblib.dump(data_dict['scaler'], scaler_path)
        
        # 4. Save feature names
        features_path = os.path.join(save_dir, 'feature_names.txt')
        with open(features_path, 'w') as f:
            for name in data_dict['feature_names']:
                f.write(f"{name}\n")
        
        # 5. Save training history
        history = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_r2': self.train_r2_values,
            'val_r2': self.val_r2_values
        })
        history_path = os.path.join(save_dir, 'training_history.csv')
        history.to_csv(history_path, index=False)
        
        # 6. Save configuration
        config_path = os.path.join(save_dir, 'config.txt')
        with open(config_path, 'w') as f:
            for key, value in vars(self.config).items():
                f.write(f"{key}: {value}\n")
        
        # 7. Save evaluation metrics
        metrics_path = os.path.join(save_dir, 'evaluation_metrics.csv')
        
        if 'original_scale' in results:
            metrics_df = pd.DataFrame([{
                'Log_R2': results['log_scale']['metrics']['r2'],
                'Log_RMSE': results['log_scale']['metrics']['rmse'],
                'Log_MAE': results['log_scale']['metrics']['mae'],
                'Original_R2': results['original_scale']['metrics']['r2'],
                'Original_RMSE': results['original_scale']['metrics']['rmse'],
                'Original_MAE': results['original_scale']['metrics']['mae']
            }])
        else:
            metrics_df = pd.DataFrame([{
                'R2': results['metrics']['r2'],
                'RMSE': results['metrics']['rmse'],
                'MAE': results['metrics']['mae']
            }])
        
        metrics_df.to_csv(metrics_path, index=False)
        
        # 8. Save complete model package
        package = {
            'config': self.config,
            'feature_names': data_dict['feature_names'],
            'n_features': data_dict['n_features'],
            'scaler': data_dict['scaler'],
            'use_log_transform': self.config.use_log_transform,
            'epsilon': self.config.epsilon
        }
        
        package_path = os.path.join(save_dir, 'model_package.pkl')
        joblib.dump(package, package_path)
        
        logger.info(f"Model and related data saved to {save_dir}")
        
        return save_dir
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        plt.figure(figsize=(12, 10))
        
        # Plot loss curves
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.axvline(x=self.best_epoch, color='r', linestyle='--', label='Best Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot R² curves
        plt.subplot(2, 1, 2)
        plt.plot(self.train_r2_values, label='Training R²')
        plt.plot(self.val_r2_values, label='Validation R²')
        plt.axvline(x=self.best_epoch, color='r', linestyle='--', label='Best Model')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.title('Training and Validation R²')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_predictions(self, results, title=None, save_path=None):
        """Plot predicted vs actual biomass values"""
        if 'original_scale' in results:
            # Use original scale if available
            predictions = results['original_scale']['predictions']
            targets = results['original_scale']['targets']
            metrics = results['original_scale']['metrics']
            scale_note = "(Original Scale)"
        else:
            # Otherwise use whatever scale we have
            predictions = results['predictions']
            targets = results['targets']
            metrics = results['metrics']
            scale_note = "(Log Scale)" if self.config.use_log_transform else ""
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(targets, predictions, alpha=0.5, s=10)
        
        # Add perfect prediction line
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Add labels and title
        plt.xlabel('Actual Biomass (Mg/ha)')
        plt.ylabel('Predicted Biomass (Mg/ha)')
        
        if title:
            plt.title(f'{title} {scale_note}\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}')
        else:
            plt.title(f'Predicted vs Actual Biomass {scale_note}\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}')
        
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Predictions plot saved to {save_path}")
        
        return plt.gcf()
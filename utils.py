import os
import rasterio
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
import random
import joblib
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_memory():
    """Clean up memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_metrics(y_true, y_pred):
    """Calculate common regression metrics"""
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }
    return metrics

def create_biomass_map(model, image_path, scaler, use_log_transform=True, epsilon=1.0,
                      output_path=None, chunk_size=1000, device=None):
    """
    Create biomass prediction map from a satellite image
    
    Args:
        model: PyTorch model
        image_path: Path to satellite image
        scaler: Feature scaler
        use_log_transform: Whether to convert from log scale
        epsilon: Epsilon value for log transform
        output_path: Path to save output GeoTIFF
        chunk_size: Chunk size for processing
        device: PyTorch device
        
    Returns:
        Numpy array with biomass predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    logger.info(f"Creating biomass map from {image_path}")
    
    try:
        with rasterio.open(image_path) as src:
            # Read image metadata
            meta = src.meta.copy()
            height, width = src.height, src.width
            
            # Prepare output array
            predictions = np.zeros((height, width), dtype=np.float32)
            
            # Process image in chunks
            for y_start in range(0, height, chunk_size):
                y_end = min(y_start + chunk_size, height)
                
                for x_start in range(0, width, chunk_size):
                    x_end = min(x_start + chunk_size, width)
                    
                    # Read window
                    window = ((y_start, y_end), (x_start, x_end))
                    data = src.read(window=window)
                    
                    # Skip empty chunks
                    if np.all(data == 0) or not np.any(np.isfinite(data)):
                        continue
                    
                    # Reshape to pixels x bands
                    chunk_h, chunk_w = y_end - y_start, x_end - x_start
                    data_reshaped = data.reshape(data.shape[0], -1).T
                    
                    # Create mask for valid pixels
                    valid_mask = np.all(np.isfinite(data_reshaped), axis=1)
                    
                    if not np.any(valid_mask):
                        continue
                    
                    # Extract valid pixels
                    valid_data = data_reshaped[valid_mask]
                    
                    # Scale features
                    valid_data_scaled = scaler.transform(valid_data)
                    
                    # Make predictions
                    with torch.no_grad():
                        tensor = torch.tensor(valid_data_scaled, dtype=torch.float32).to(device)
                        preds = model(tensor).cpu().numpy()
                    
                    # Convert from log scale if needed
                    if use_log_transform:
                        preds = np.exp(preds) - epsilon
                        preds = np.maximum(preds, 0)  # Ensure non-negative
                    
                    # Create output chunk
                    output_chunk = np.zeros(chunk_h * chunk_w)
                    output_chunk[valid_mask] = preds
                    output_chunk = output_chunk.reshape(chunk_h, chunk_w)
                    
                    # Add to output
                    predictions[y_start:y_end, x_start:x_end] = output_chunk
            
            # Save output if specified
            if output_path:
                # Update metadata for single band output
                meta.update(
                    dtype='float32',
                    count=1,
                    nodata=0
                )
                
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(predictions, 1)
                    
                logger.info(f"Saved biomass map to {output_path}")
                
            return predictions
            
    except Exception as e:
        logger.error(f"Error creating biomass map: {e}")
        return None

def visualize_predictions(predictions, actual=None, title=None, 
                         cmap='viridis', figsize=(10, 8), save_path=None):
    """
    Visualize biomass predictions
    
    Args:
        predictions: Predicted values
        actual: Actual values (optional)
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    if actual is not None:
        plt.scatter(actual, predictions, alpha=0.5, s=10)
        
        # Add perfect prediction line
        min_val = min(np.min(actual), np.min(predictions))
        max_val = max(np.max(actual), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate metrics
        metrics = calculate_metrics(actual, predictions)
        
        plt.xlabel('Actual Biomass (Mg/ha)')
        plt.ylabel('Predicted Biomass (Mg/ha)')
        
        if title:
            plt.title(f"{title}\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
        else:
            plt.title(f"Prediction Performance\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
    else:
        plt.imshow(predictions, cmap=cmap)
        plt.colorbar(label='Biomass (Mg/ha)')
        
        if title:
            plt.title(title)
        else:
            plt.title("Predicted Biomass (Mg/ha)")
    
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()

def load_model_package(package_path, model_path=None, device=None):
    """
    Load model package and weights
    
    Args:
        package_path: Path to model package
        model_path: Path to model weights (optional)
        device: PyTorch device (optional)
        
    Returns:
        tuple: (model, package)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load package
        package = joblib.load(package_path)
        
        # Determine model path if not provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(package_path), 'model.pt')
        
        # Load model
        from model import StableResNet
        model = StableResNet(n_features=package['n_features'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        
        return model, package
    
    except Exception as e:
        logger.error(f"Error loading model package: {e}")
        return None, None
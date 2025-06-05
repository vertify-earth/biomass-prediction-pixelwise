import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging
from typing import Dict, List, Tuple, Optional
from feature_engineering import extract_features_for_training

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_process_tile(satellite_path, biomass_path, config):
    """Load and process a single satellite/biomass tile pair"""
    logger.info(f"Loading tile: {os.path.basename(satellite_path)}")
    
    try:
        # Load satellite data
        with rasterio.open(satellite_path) as src:
            satellite_data = src.read()
            transform = src.transform
            crs = src.crs
            metadata = {
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'transform': transform,
                'crs': crs
            }
        
        # Load biomass data
        with rasterio.open(biomass_path) as src:
            biomass_data = src.read(1)
        
        logger.info(f"  Satellite shape: {satellite_data.shape}")
        logger.info(f"  Biomass shape: {biomass_data.shape}")
        
        # Create validity mask
        # Valid pixels must:
        # 1. Have finite values in all satellite bands
        # 2. Have finite biomass values
        # 3. Have positive biomass values
        # 4. Have reasonable biomass values (< 1000 Mg/ha)
        sat_finite_mask = np.all(np.isfinite(satellite_data), axis=0)
        bio_finite_mask = np.isfinite(biomass_data)
        bio_positive_mask = biomass_data > 0
        bio_reasonable_mask = biomass_data < 1000  # Arbitrary upper limit, adjust as needed
        
        valid_mask = (
            sat_finite_mask & 
            bio_finite_mask & 
            bio_positive_mask & 
            bio_reasonable_mask
        )
        
        valid_percent = np.mean(valid_mask) * 100
        logger.info(f"  Valid pixels: {valid_percent:.1f}%")
        
        # Extract features for model training
        features, targets, feature_names = extract_features_for_training(
            satellite_data, biomass_data, valid_mask, config
        )
        
        return features, targets, feature_names, metadata
        
    except Exception as e:
        logger.error(f"Error processing tile {satellite_path}: {e}")
        return None, None, None, None

def load_biomass_dataset(config):
    """Load and combine data from all tiles"""
    logger.info(f"Loading biomass data from {len(config.raster_pairs)} tiles...")
    
    all_features = []
    all_targets = []
    feature_names = None
    metadata = None
    
    for i, (sat_path, bio_path) in enumerate(config.raster_pairs):
        logger.info(f"Processing tile {i+1}/{len(config.raster_pairs)}")
        
        features, targets, names, tile_metadata = load_and_process_tile(sat_path, bio_path, config)
        
        if features is not None:
            all_features.append(features)
            all_targets.append(targets)
            
            if feature_names is None:
                feature_names = names
                metadata = tile_metadata  # Store first tile's metadata
            
            logger.info(f"  Added {len(targets):,} samples")
        
        # Clean up memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine all data
    if all_features:
        X = np.vstack(all_features)
        y = np.hstack(all_targets)
        
        logger.info(f"Final dataset: {X.shape[0]:,} samples with {X.shape[1]} features")
        
        return X, y, feature_names, metadata
    else:
        raise ValueError("No valid data could be loaded from any tile")

def prepare_training_data(config):
    """Prepare data for model training"""
    logger.info("Preparing training data...")
    
    # Load data from all tiles
    X, y, feature_names, metadata = load_biomass_dataset(config)
    
    # Create train/val/test splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.val_size/(1-config.test_size), 
        random_state=config.random_seed
    )
    
    logger.info(f"Train: {X_train.shape[0]:,} samples")
    logger.info(f"Validation: {X_val.shape[0]:,} samples")
    logger.info(f"Test: {X_test.shape[0]:,} samples")
    
    # Initialize feature scaler based on config
    if config.scale_method == "robust":
        scaler = RobustScaler()
    elif config.scale_method == "standard":
        scaler = StandardScaler()
    elif config.scale_method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {config.scale_method}")
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Store original targets if using log transform
    if config.use_log_transform:
        y_train_orig = np.exp(y_train) - config.epsilon
        y_val_orig = np.exp(y_val) - config.epsilon
        y_test_orig = np.exp(y_test) - config.epsilon
        
        original_targets = {
            'train': y_train_orig,
            'val': y_val_orig,
            'test': y_test_orig
        }
    else:
        original_targets = None
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Return all necessary data
    data_dict = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_names': feature_names,
        'scaler': scaler,
        'original_targets': original_targets,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'metadata': metadata,
        'n_features': X.shape[1]
    }
    
    return data_dict

class BiomassImageDataset(Dataset):
    """Dataset for pixel-wise prediction on full satellite images"""
    
    def __init__(self, image_path, scaler, chunk_size=1024):
        self.image_path = image_path
        self.scaler = scaler
        self.chunk_size = chunk_size
        
        with rasterio.open(image_path) as src:
            self.height = src.height
            self.width = src.width
            self.count = src.count
            self.transform = src.transform
            self.crs = src.crs
            self.dtype = src.dtypes[0]
        
        # Calculate number of chunks
        self.n_chunks_h = int(np.ceil(self.height / self.chunk_size))
        self.n_chunks_w = int(np.ceil(self.width / self.chunk_size))
        self.total_chunks = self.n_chunks_h * self.n_chunks_w
    
    def __len__(self):
        return self.total_chunks
    
    def __getitem__(self, idx):
        # Calculate chunk coordinates
        chunk_row = idx // self.n_chunks_w
        chunk_col = idx % self.n_chunks_w
        
        # Calculate pixel coordinates
        row_start = chunk_row * self.chunk_size
        col_start = chunk_col * self.chunk_size
        row_end = min(row_start + self.chunk_size, self.height)
        col_end = min(col_start + self.chunk_size, self.width)
        
        # Load data for this chunk
        with rasterio.open(self.image_path) as src:
            data = src.read(window=((row_start, row_end), (col_start, col_end)))
        
        # Reshape to pixels x bands
        pixels = data.reshape(self.count, -1).T
        
        # Scale features
        pixels_scaled = self.scaler.transform(pixels)
        
        # Create mask for valid pixels (no NaN or Inf values)
        valid_mask = np.all(np.isfinite(pixels), axis=1)
        
        # Create chunk metadata
        metadata = {
            'row_start': row_start,
            'row_end': row_end,
            'col_start': col_start,
            'col_end': col_end,
            'shape': (row_end - row_start, col_end - col_start),
            'valid_mask': valid_mask
        }
        
        return torch.tensor(pixels_scaled, dtype=torch.float32), metadata
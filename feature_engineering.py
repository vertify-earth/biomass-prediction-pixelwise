import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
import warnings
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_divide(a, b, fill_value=0.0):
    """Safe division that handles zeros in the denominator"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    # Handle NaN/Inf in inputs
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=1e-10, posinf=1e10, neginf=-1e10)
    
    if a.ndim == 0 and b.ndim > 0:
        a = np.full_like(b, a)
    elif b.ndim == 0 and a.ndim > 0:
        b = np.full_like(a, b)
    elif a.ndim == 0 and b.ndim == 0:
        if abs(b) < 1e-10:
            return fill_value
        else:
            return float(a / b)
    
    mask = np.abs(b) < 1e-10
    result = np.full_like(a, fill_value, dtype=np.float32)
    if np.any(~mask):
        result[~mask] = a[~mask] / b[~mask]
    
    result = np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return result

def calculate_spectral_indices(satellite_data):
    """Calculate spectral indices from satellite bands"""
    logger.info("Calculating spectral indices...")
    
    indices = {}
    n_bands = satellite_data.shape[0]
    
    # Enhanced band mapping with error checking
    def safe_get_band(idx):
        return satellite_data[idx] if idx < n_bands else None
    
    # Sentinel-2 bands (assuming standard band order)
    # B2(blue), B3(green), B4(red), B8(nir), B11(swir1), B12(swir2)
    try:
        blue = safe_get_band(1)  # Adjust indices based on your data
        green = safe_get_band(2)
        red = safe_get_band(3)
        nir = safe_get_band(7)
        swir1 = safe_get_band(9)
        swir2 = safe_get_band(10)
        
        if all(b is not None for b in [red, nir]):
            # NDVI (Normalized Difference Vegetation Index)
            indices['NDVI'] = safe_divide(nir - red, nir + red)
            
            if blue is not None and green is not None:
                # EVI (Enhanced Vegetation Index)
                indices['EVI'] = 2.5 * safe_divide(nir - red, nir + 6*red - 7.5*blue + 1)
                
                # SAVI (Soil Adjusted Vegetation Index)
                indices['SAVI'] = 1.5 * safe_divide(nir - red, nir + red + 0.5)
                
                # MSAVI2 (Modified Soil Adjusted Vegetation Index)
                indices['MSAVI2'] = 0.5 * (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red)))
                
                # NDWI (Normalized Difference Water Index)
                indices['NDWI'] = safe_divide(green - nir, green + nir)
        
        if swir1 is not None and nir is not None:
            # NDMI (Normalized Difference Moisture Index)
            indices['NDMI'] = safe_divide(nir - swir1, nir + swir1)
        
        if swir2 is not None and nir is not None:
            # NBR (Normalized Burn Ratio)
            indices['NBR'] = safe_divide(nir - swir2, nir + swir2)
            
    except Exception as e:
        logger.warning(f"Error calculating spectral indices: {e}")
    
    # Clean up None values and NaNs
    indices = {k: np.nan_to_num(v, nan=0.0) for k, v in indices.items() if v is not None}
    
    logger.info(f"Calculated {len(indices)} spectral indices")
    return indices

def extract_texture_features(satellite_data, config):
    """Extract texture features from satellite data"""
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.filters import sobel
    
    if not config.use_texture_features:
        return {}
    
    logger.info("Extracting texture features...")
    texture_features = {}
    height, width = satellite_data.shape[1], satellite_data.shape[2]
    
    # Select representative bands for texture analysis 
    # (e.g., NIR bands which are good for vegetation structure)
    key_bands = [7]  # NIR band
    
    for band_idx in key_bands:
        if band_idx >= satellite_data.shape[0]:
            continue
            
        try:
            band = satellite_data[band_idx].copy()
            
            # Normalize to 0-255 for texture analysis
            band_min, band_max = np.nanpercentile(band[~np.isnan(band)], [1, 99])
            band_norm = np.clip((band - band_min) / (band_max - band_min + 1e-8), 0, 1)
            band_norm = (band_norm * 255).astype(np.uint8)
            
            # Replace NaN with median
            band_norm = np.nan_to_num(band_norm, nan=np.nanmedian(band_norm))
            
            # Edge detection using Sobel
            sobel_response = sobel(band_norm.astype(float))
            texture_features[f'Sobel_B{band_idx}'] = sobel_response
            
            # Local Binary Pattern
            try:
                lbp = local_binary_pattern(band_norm, 8, 1, method='uniform')
                texture_features[f'LBP_B{band_idx}'] = lbp
            except Exception as e:
                logger.warning(f"Error calculating LBP for band {band_idx}: {e}")
                
            # GLCM properties - simplified approach for efficiency
            # Calculate GLCM on a representative patch rather than the entire image
            sample_size = min(128, band_norm.shape[0], band_norm.shape[1])
            center_y, center_x = band_norm.shape[0]//2, band_norm.shape[1]//2
            offset = sample_size // 2
            patch = band_norm[center_y-offset:center_y+offset, center_x-offset:center_x+offset]
            
            if patch.size > 0:
                try:
                    glcm = graycomatrix(patch, [1], [0], levels=256, symmetric=True, normed=True)
                    # Extract properties as scalar values
                    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
                        value = float(graycoprops(glcm, prop)[0, 0])
                        # Create 2D arrays with the scalar value
                        texture_features[f'GLCM_{prop}_B{band_idx}'] = np.full((height, width), value, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Error calculating GLCM for band {band_idx}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error processing band {band_idx} for texture: {e}")
    
    logger.info(f"Extracted {len(texture_features)} texture features")
    return texture_features

def calculate_spatial_features(satellite_data, indices, config):
    """Calculate spatial context features like gradients"""
    if not config.use_spatial_features:
        return {}
    
    logger.info("Calculating spatial features...")
    spatial_features = {}
    height, width = satellite_data.shape[1], satellite_data.shape[2]
    
    # Key bands for spatial analysis
    key_bands = [7]  # NIR band
    
    for band_idx in key_bands:
        if band_idx < satellite_data.shape[0]:
            try:
                band = satellite_data[band_idx].copy()
                band = np.nan_to_num(band, nan=0.0)
                
                # Calculate gradients
                grad_y, grad_x = np.gradient(band)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                spatial_features[f'Gradient_B{band_idx}'] = grad_magnitude
                
            except Exception as e:
                logger.warning(f"Error calculating spatial features for band {band_idx}: {e}")
    
    # Gradient features for NDVI if available
    if 'NDVI' in indices:
        try:
            ndvi_clean = np.nan_to_num(indices['NDVI'], nan=0.0)
            
            # Calculate gradients
            grad_y, grad_x = np.gradient(ndvi_clean)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            spatial_features['NDVI_gradient'] = grad_magnitude
            
        except Exception as e:
            logger.warning(f"Error calculating gradient for NDVI: {e}")
    
    logger.info(f"Calculated {len(spatial_features)} spatial features")
    return spatial_features

def calculate_pca_features(satellite_data, config):
    """Calculate PCA features from satellite bands"""
    if not config.use_pca_features:
        return {}
    
    logger.info(f"Calculating {config.pca_components} PCA features...")
    
    # Reshape for PCA (pixels x bands)
    height, width = satellite_data.shape[1], satellite_data.shape[2]
    bands_reshaped = satellite_data.reshape(satellite_data.shape[0], -1).T
    
    # Handle NaN values
    valid_mask = ~np.any(np.isnan(bands_reshaped), axis=1)
    bands_clean = bands_reshaped[valid_mask]
    
    if len(bands_clean) == 0:
        logger.warning("No valid data for PCA")
        return {}
    
    try:
        # Standardize and apply PCA
        scaler = StandardScaler()
        bands_scaled = scaler.fit_transform(bands_clean)
        
        pca = PCA(n_components=min(config.pca_components, bands_scaled.shape[1]))
        pca_features = pca.fit_transform(bands_scaled)
        
        # Create full PCA array
        pca_full = np.zeros((height * width, pca_features.shape[1]))
        pca_full[valid_mask] = pca_features
        pca_full = pca_full.reshape(height, width, pca_features.shape[1])
        
        # Convert to dictionary format
        pca_dict = {}
        for i in range(pca_full.shape[2]):
            pca_dict[f'PCA_{i+1:02d}'] = pca_full[:, :, i]
        
        explained_variance = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance: {explained_variance:.3f}")
        
        return pca_dict
        
    except Exception as e:
        logger.warning(f"Error calculating PCA: {e}")
        return {}

def extract_features_for_training(satellite_data, biomass_data, valid_mask, config):
    """Extract features from valid pixels for model training"""
    logger.info("Extracting features from valid pixels...")
    
    # Get valid pixel coordinates
    valid_y, valid_x = np.where(valid_mask)
    n_valid = len(valid_y)
    
    # Sample for testing if needed
    if hasattr(config, 'max_samples_per_tile') and config.max_samples_per_tile and n_valid > config.max_samples_per_tile:
        np.random.seed(config.random_seed)
        indices = np.random.choice(n_valid, config.max_samples_per_tile, replace=False)
        valid_y = valid_y[indices]
        valid_x = valid_x[indices]
        n_valid = len(valid_y)
        logger.info(f"Sampled {n_valid} pixels for training")
    
    # Extract all features
    all_features = {}
    
    # 1. Original bands
    for i in range(satellite_data.shape[0]):
        band_data = satellite_data[i].copy()
        band_data = np.nan_to_num(band_data, nan=0.0)
        all_features[f'Band_{i+1:02d}'] = band_data
    
    # 2. Spectral indices
    if config.use_advanced_indices:
        indices = calculate_spectral_indices(satellite_data)
        all_features.update(indices)
    else:
        indices = {}
    
    # 3. Texture features
    if config.use_texture_features:
        texture_features = extract_texture_features(satellite_data, config)
        all_features.update(texture_features)
    
    # 4. Spatial features
    if config.use_spatial_features:
        spatial_features = calculate_spatial_features(satellite_data, indices, config)
        all_features.update(spatial_features)
    
    # 5. PCA features
    if config.use_pca_features:
        pca_features = calculate_pca_features(satellite_data, config)
        all_features.update(pca_features)
    
    # Create feature matrix
    feature_names = list(all_features.keys())
    feature_matrix = np.zeros((n_valid, len(feature_names)), dtype=np.float32)
    
    for i, feature_name in enumerate(feature_names):
        feature_data = all_features[feature_name]
        feature_values = feature_data[valid_y, valid_x]
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        feature_matrix[:, i] = feature_values
    
    # Extract targets
    biomass_targets = biomass_data[valid_y, valid_x].astype(np.float32)
    
    # Apply log transform if specified
    if config.use_log_transform:
        biomass_targets = np.log(biomass_targets + config.epsilon)
    
    logger.info(f"Extracted features: {feature_matrix.shape}, Targets: {biomass_targets.shape}")
    
    return feature_matrix, biomass_targets, feature_names
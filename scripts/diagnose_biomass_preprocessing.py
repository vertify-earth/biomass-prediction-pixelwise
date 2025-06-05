#!/usr/bin/env python3
"""
Diagnostic Tool for Biomass Prediction Preprocessing (Fixed Version)
Analyzes the preprocessing steps to identify potential issues causing uniform predictions

Author: najahpokkiri
Date: 2025-05-17
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap for anomaly detection
anomaly_cmap = LinearSegmentedColormap.from_list('anomaly', 
                                                ['darkblue', 'blue', 'lightblue', 
                                                'white', 'lightcoral', 'red', 'darkred'])

# Suppress warnings
warnings.filterwarnings("ignore")

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
        print(f"Warning: Error calculating spectral indices: {e}")
    
    # Clean up None values and NaNs
    indices = {k: np.nan_to_num(v, nan=0.0) for k, v in indices.items() if v is not None}
    return indices

def extract_texture_features(satellite_data, config):
    """Extract texture features from satellite data"""
    try:
        from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
        from skimage.filters import sobel
        skimage_available = True
    except ImportError:
        print("Warning: scikit-image not available. Texture features will be disabled.")
        skimage_available = False
    
    if not getattr(config, 'use_texture_features', True) or not skimage_available:
        return {}
    
    texture_features = {}
    height, width = satellite_data.shape[1], satellite_data.shape[2]
    
    # Select representative bands for texture analysis (e.g., NIR bands)
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
                print(f"Warning: Error calculating LBP for band {band_idx}: {e}")
                
        except Exception as e:
            print(f"Warning: Error processing band {band_idx} for texture: {e}")
    
    return texture_features

def calculate_spatial_features(satellite_data, indices, config):
    """Calculate spatial context features like gradients"""
    if not getattr(config, 'use_spatial_features', True):
        return {}
    
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
                print(f"Warning: Error calculating spatial features for band {band_idx}: {e}")
    
    # Gradient features for NDVI if available
    if 'NDVI' in indices:
        try:
            ndvi_clean = np.nan_to_num(indices['NDVI'], nan=0.0)
            
            # Calculate gradients
            grad_y, grad_x = np.gradient(ndvi_clean)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            spatial_features['NDVI_gradient'] = grad_magnitude
            
        except Exception as e:
            print(f"Warning: Error calculating gradient for NDVI: {e}")
    
    return spatial_features

def calculate_pca_features(satellite_data, config):
    """Calculate PCA features from satellite bands"""
    if not getattr(config, 'use_pca_features', True):
        return {}
    
    # Reshape for PCA (pixels x bands)
    height, width = satellite_data.shape[1], satellite_data.shape[2]
    bands_reshaped = satellite_data.reshape(satellite_data.shape[0], -1).T
    
    # Handle NaN values
    valid_mask = ~np.any(np.isnan(bands_reshaped), axis=1)
    bands_clean = bands_reshaped[valid_mask]
    
    if len(bands_clean) == 0:
        print("Warning: No valid data for PCA")
        return {}
    
    try:
        # Standardize and apply PCA
        scaler = StandardScaler()
        bands_scaled = scaler.fit_transform(bands_clean)
        
        pca = PCA(n_components=min(getattr(config, 'pca_components', 10), bands_scaled.shape[1]))
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
        print(f"PCA explained variance: {explained_variance:.3f}")
        
        return pca_dict
        
    except Exception as e:
        print(f"Warning: Error calculating PCA: {e}")
        return {}

def analyze_chunking_effects(image_path, chunk_size, overlap, output_dir):
    """Analyze the effects of chunking on an image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the image to get dimensions
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        print(f"Image size: {width}x{height}, {src.count} bands")
        
        # Calculate chunk boundaries
        chunks = []
        for y in range(0, height, chunk_size - overlap):
            for x in range(0, width, chunk_size - overlap):
                # Calculate actual chunk size (may be smaller at edges)
                w = min(chunk_size, width - x)
                h = min(chunk_size, height - y)
                chunks.append((x, y, w, h))
        
        # Create visualization of chunk boundaries
        grid_image = np.zeros((height, width), dtype=np.uint8)
        for x, y, w, h in chunks:
            # Mark chunk boundaries
            boundary_width = 2
            
            # Top and bottom boundaries
            if y > 0:  # Only draw top boundary if not at the top edge of the image
                grid_image[y:y+boundary_width, x:x+w] = 255
            if y+h < height:  # Only draw bottom boundary if not at the bottom edge
                grid_image[y+h-boundary_width:y+h, x:x+w] = 255
                
            # Left and right boundaries
            if x > 0:  # Only draw left boundary if not at the left edge
                grid_image[y:y+h, x:x+boundary_width] = 255
            if x+w < width:  # Only draw right boundary if not at the right edge
                grid_image[y:y+h, x+w-boundary_width:x+w] = 255

        # Save the grid image without trying to visualize it
        plt.figure(figsize=(12, 10))
        plt.imshow(grid_image, cmap='gray')
        plt.title(f"Chunk Grid (size={chunk_size}, overlap={overlap})")
        plt.savefig(os.path.join(output_dir, "chunk_grid.png"))
        plt.close()
        
        # Now try to create a simpler visualization with a synthetic image
        downsample = max(1, min(width, height) // 1000)
        grid_small = grid_image[::downsample, ::downsample]
        
        plt.figure(figsize=(12, 10))
        plt.imshow(grid_small, cmap='gray')
        plt.title(f"Downsampled Chunk Grid (size={chunk_size}, overlap={overlap})")
        plt.savefig(os.path.join(output_dir, "chunk_grid_downsampled.png"))
        plt.close()
        
        # Create text-based visualization of chunks
        with open(os.path.join(output_dir, "chunk_analysis.txt"), "w") as f:
            f.write(f"Image dimensions: {width}x{height}\n")
            f.write(f"Chunk size: {chunk_size}\n")
            f.write(f"Overlap: {overlap}\n")
            f.write(f"Total chunks: {len(chunks)}\n\n")
            
            # Find bottom-left problem area (estimated based on your image)
            problem_area = {'x': 0, 'y': int(height*0.6), 'width': int(width*0.3), 'height': int(height*0.4)}
            f.write(f"Problem area (estimated): x={problem_area['x']}, y={problem_area['y']}, " 
                   f"width={problem_area['width']}, height={problem_area['height']}\n\n")
            
            # Find chunks containing the problem area
            problem_chunks = []
            for i, (x, y, w, h) in enumerate(chunks):
                # Check if chunk overlaps with problem area
                if (x < problem_area['x'] + problem_area['width'] and 
                    x + w > problem_area['x'] and 
                    y < problem_area['y'] + problem_area['height'] and 
                    y + h > problem_area['y']):
                    problem_chunks.append((i, x, y, w, h))
            
            f.write(f"Chunks in problem area: {len(problem_chunks)}\n\n")
            
            for i, (chunk_idx, x, y, w, h) in enumerate(problem_chunks):
                f.write(f"Chunk {chunk_idx}: x={x}, y={y}, width={w}, height={h}\n")

def extract_all_features(satellite_data, config, band_subset=None):
    """Extract all features from satellite data for analysis"""
    print(f"Extracting features from data with shape {satellite_data.shape}")
    
    # Use subset of bands if specified
    if band_subset is not None:
        bands_to_use = [i for i in band_subset if i < satellite_data.shape[0]]
        satellite_data = satellite_data[bands_to_use]
        print(f"Using band subset: {bands_to_use}")
    
    # Dictionary to hold all features
    all_features = {}
    
    # Original bands
    for i in range(satellite_data.shape[0]):
        all_features[f'Band_{i+1:02d}'] = satellite_data[i]
    
    # Add spectral indices
    indices = calculate_spectral_indices(satellite_data)
    all_features.update(indices)
    
    # Add texture features if enabled
    texture_features = extract_texture_features(satellite_data, config)
    all_features.update(texture_features)
    
    # Add spatial features if enabled
    spatial_features = calculate_spatial_features(satellite_data, indices, config)
    all_features.update(spatial_features)
    
    # Add PCA features if enabled
    pca_features = calculate_pca_features(satellite_data, config)
    all_features.update(pca_features)
    
    return all_features

def calculate_feature_statistics(features):
    """Calculate statistics for each feature"""
    stats = {}
    for name, data in features.items():
        valid_mask = ~np.isnan(data)
        if np.any(valid_mask):
            valid_data = data[valid_mask]
            stats[name] = {
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'nan_pct': float(100 * np.sum(~valid_mask) / data.size),
                'zeros_pct': float(100 * np.sum(valid_data == 0) / np.sum(valid_mask)),
                'uniform_pct': calculate_uniformity(data)
            }
        else:
            stats[name] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'nan_pct': 100.0,
                'zeros_pct': 0.0,
                'uniform_pct': 0.0
            }
    return stats

def calculate_uniformity(data, window_size=5, threshold=0.01):
    """Calculate percentage of area with uniform values"""
    if data.size == 0:
        return 0.0
    
    try:
        # Calculate local variance
        data_clean = np.nan_to_num(data, nan=0.0)
        local_mean = uniform_filter(data_clean, size=window_size)
        local_squaremean = uniform_filter(data_clean**2, size=window_size)
        local_variance = local_squaremean - local_mean**2
        
        # Calculate global variance
        global_variance = np.var(data_clean)
        
        # Identify areas with very low variance
        if global_variance > 0:
            low_variance_mask = local_variance < threshold * global_variance
            return float(100 * np.sum(low_variance_mask) / low_variance_mask.size)
        else:
            return 100.0  # If global variance is 0, everything is uniform
    except Exception as e:
        print(f"Error calculating uniformity: {e}")
        return 0.0

def visualize_features(region_data, features, output_dir, region_name):
    """Visualize key features for a region"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define key features to visualize
    key_features = [
        # Original bands
        'Band_01', 'Band_07', 'Band_10',
        
        # Vegetation indices
        'NDVI', 'EVI', 'SAVI',
        
        # Water and moisture indices
        'NDWI', 'NDMI', 'NBR',
        
        # Texture features
        'Sobel_B7', 
        
        # Spatial features
        'Gradient_B7', 'NDVI_gradient',
        
        # PCA features
        'PCA_01', 'PCA_02', 'PCA_03'
    ]
    
    # Only visualize features that exist
    features_to_plot = [f for f in key_features if f in features]
    
    for feature_name in features_to_plot:
        try:
            feature_data = features[feature_name]
            
            plt.figure(figsize=(10, 8))
            
            # Create a valid mask
            valid_mask = ~np.isnan(feature_data)
            
            # Create masked array
            masked_data = np.ma.masked_where(~valid_mask, feature_data)
            
            # Get valid min/max values for better visualization
            if np.any(valid_mask):
                vmin = np.nanpercentile(feature_data, 1)
                vmax = np.nanpercentile(feature_data, 99)
            else:
                vmin, vmax = 0, 1
            
            # Plot the feature
            plt.imshow(masked_data, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(label=feature_name)
            plt.title(f"{feature_name} - {region_name}")
            
            # Add statistics
            if np.any(valid_mask):
                stats_text = (
                    f"Mean: {np.nanmean(feature_data):.2f}, "
                    f"Std: {np.nanstd(feature_data):.2f}\n"
                    f"Min: {np.nanmin(feature_data):.2f}, "
                    f"Max: {np.nanmax(feature_data):.2f}\n"
                    f"NaN: {100 * np.sum(~valid_mask) / feature_data.size:.1f}%"
                )
                plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{region_name}_{feature_name}.png"))
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing {feature_name}: {e}")

def get_feature_anomalies(feature_data, window_size=5, threshold=3.0):
    """Detect anomalous regions in a feature"""
    # Replace NaNs with 0
    clean_data = np.nan_to_num(feature_data, nan=0.0)
    
    # Calculate local statistics
    local_mean = uniform_filter(clean_data, size=window_size)
    local_squaremean = uniform_filter(clean_data**2, size=window_size)
    local_variance = local_squaremean - local_mean**2
    
    # Calculate global statistics
    global_mean = np.mean(clean_data)
    global_std = np.std(clean_data)
    
    if global_std == 0:
        return np.zeros_like(clean_data)
    
    # Calculate z-scores
    z_scores = (local_mean - global_mean) / global_std
    
    # Calculate variance ratio
    global_var = np.var(clean_data)
    if global_var > 0:
        variance_ratio = local_variance / global_var
    else:
        variance_ratio = np.zeros_like(local_variance)
    
    # Combine anomaly indicators
    anomaly_score = np.abs(z_scores) * (1 + np.exp(-variance_ratio))
    
    return anomaly_score

def analyze_feature_anomalies(features, output_dir):
    """Analyze features for anomalous regions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select key features for analysis
    key_features = [
        # Vegetation indices
        'NDVI', 'EVI', 
        
        # Water/moisture indices
        'NDWI', 'NDMI', 
        
        # Texture/edge features
        'Sobel_B7', 'NDVI_gradient',
        
        # PCA features
        'PCA_01', 'PCA_02'
    ]
    
    # Only analyze features that exist
    features_to_analyze = [f for f in key_features if f in features]
    
    # Create combined anomaly map
    combined_anomaly = None
    
    for feature_name in features_to_analyze:
        try:
            feature_data = features[feature_name]
            
            # Calculate anomaly scores
            anomaly_scores = get_feature_anomalies(feature_data)
            
            # Update combined anomaly map
            if combined_anomaly is None:
                combined_anomaly = anomaly_scores
            else:
                combined_anomaly = np.maximum(combined_anomaly, anomaly_scores)
            
            # Visualize anomaly map
            plt.figure(figsize=(10, 8))
            plt.imshow(anomaly_scores, cmap=anomaly_cmap, vmin=0, vmax=5)
            plt.colorbar(label="Anomaly Score")
            plt.title(f"Anomaly Map - {feature_name}")
            plt.savefig(os.path.join(output_dir, f"anomaly_{feature_name}.png"))
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing anomalies for {feature_name}: {e}")
    
    # Save combined anomaly map
    if combined_anomaly is not None:
        plt.figure(figsize=(12, 10))
        plt.imshow(combined_anomaly, cmap=anomaly_cmap, vmin=0, vmax=5)
        plt.colorbar(label="Combined Anomaly Score")
        plt.title("Combined Anomaly Map")
        plt.savefig(os.path.join(output_dir, "combined_anomaly_map.png"))
        plt.close()

def analyze_image_and_preprocessing(config, output_dir="biomass_preprocessing_diagnosis"):
    """Main function to analyze image preprocessing"""
    # Setup
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Started analysis at {datetime.now()}")
    print(f"Output directory: {output_dir}")
    print(f"Input image: {config.input_image}")
    
    # Check chunking effects first
    print("\nAnalyzing potential chunking effects...")
    chunking_dir = os.path.join(output_dir, "chunking_analysis")
    analyze_chunking_effects(
        config.input_image, 
        getattr(config, 'chunk_size', 1024), 
        getattr(config, 'overlap', 32),
        chunking_dir
    )
    
    # Now analyze the image and preprocessing
    print("\nAnalyzing image and preprocessing...")
    try:
        with rasterio.open(config.input_image) as src:
            # Record basic information
            width, height = src.width, src.height
            n_bands = src.count
            print(f"Image dimensions: {width}x{height}, {n_bands} bands")
            
            # Define regions to analyze - adjust based on problem area
            bottom_left_y = int(height * 0.6)  # Estimated position of problem area
            regions = {
                "problem_area": Window(0, bottom_left_y, int(width*0.3), int(height*0.4)),  # Bottom left
                "control_area": Window(int(width*0.6), 0, int(width*0.4), int(height*0.4))  # Upper right
            }
            
            # Check if image is too large for full analysis
            if width * height > 10000000:  # If image has more than 10 million pixels
                print("Image is very large. Using downsampling for full image analysis.")
                downsample = max(2, int(np.sqrt(width * height / 1000000)))
                
                # Create a simple summary of image stats without loading full data
                stats_file = os.path.join(output_dir, "image_stats.txt")
                with open(stats_file, "w") as f:
                    f.write(f"Image dimensions: {width}x{height}, {n_bands} bands\n")
                    f.write(f"Pixel count: {width*height}\n")
                    f.write(f"Data type: {src.dtypes[0]}\n")
                    f.write(f"CRS: {src.crs}\n")
                    f.write(f"Transform: {src.transform}\n")
                    
                    # Add rough band statistics using small sample
                    sample_size = 1000
                    random_rows = np.random.randint(0, height, sample_size)
                    random_cols = np.random.randint(0, width, sample_size)
                    
                    f.write("\nBand statistics from random sample:\n")
                    for band_idx in range(min(5, n_bands)):  # First 5 bands
                        try:
                            # Read samples
                            samples = [src.read(band_idx+1, window=Window(col, row, 1, 1))[0, 0] 
                                      for row, col in zip(random_rows, random_cols)]
                            samples = np.array([s for s in samples if not np.isnan(s)])
                            
                            if len(samples) > 0:
                                f.write(f"Band {band_idx+1}: Mean={np.mean(samples):.2f}, "
                                       f"Std={np.std(samples):.2f}, "
                                       f"Range=[{np.min(samples):.2f}, {np.max(samples):.2f}]\n")
                        except Exception as e:
                            f.write(f"Band {band_idx+1}: Error reading statistics - {e}\n")
            
            # Analyze each region
            for region_name, window in regions.items():
                print(f"\nAnalyzing region: {region_name}")
                
                region_dir = os.path.join(output_dir, f"region_{region_name}")
                os.makedirs(region_dir, exist_ok=True)
                
                try:
                    # Read region data with error handling
                    try:
                        region_data = src.read(window=window)
                        print(f"Region data shape: {region_data.shape}")
                    except Exception as e:
                        print(f"Error reading region {region_name}: {e}")
                        continue
                    
                    if region_data.size == 0 or region_data.ndim < 3:
                        print(f"Invalid region data for {region_name}, skipping")
                        continue
                    
                    # Extract and analyze features
                    features = extract_all_features(region_data, config)
                    
                    # Calculate statistics
                    feature_stats = calculate_feature_statistics(features)
                    
                    # Save statistics
                    with open(os.path.join(region_dir, "feature_statistics.txt"), "w") as f:
                        f.write(f"Region: {region_name}\n")
                        f.write(f"Data shape: {region_data.shape}\n\n")
                        
                        for feature_name, stats in feature_stats.items():
                            f.write(f"{feature_name}:\n")
                            for stat_name, value in stats.items():
                                f.write(f"  {stat_name}: {value}\n")
                            f.write("\n")
                    
                    # Visualize key features
                    visualize_features(region_data, features, region_dir, region_name)
                    
                    # Analyze anomalies
                    analyze_feature_anomalies(features, os.path.join(region_dir, "anomalies"))
                    
                except Exception as e:
                    import traceback
                    print(f"Error processing region {region_name}: {e}")
                    print(traceback.format_exc())
                    
    except Exception as e:
        import traceback
        print(f"Error opening image: {e}")
        print(traceback.format_exc())
    
    # Generate summary report
    print("\nGenerating summary report...")
    with open(os.path.join(output_dir, "analysis_summary.txt"), "w") as f:
        elapsed = time.time() - start_time
        f.write(f"Analysis completed at {datetime.now()}\n")
        f.write(f"Total analysis time: {elapsed:.1f} seconds\n\n")
        
        f.write("Key findings to look for:\n")
        f.write("1. Check for chunking effects in the chunking_analysis directory\n")
        f.write("2. Compare feature statistics between problem_area and control_area\n")
        f.write("3. Look for anomalies in the anomalies subdirectories\n")
        f.write("4. Check for high uniformity percentages in the feature statistics\n")
        f.write("5. Review the visualizations for obvious patterns at chunk boundaries\n\n")
        
        f.write("Potential fixes if chunking issues are found:\n")
        f.write("1. Increase overlap between chunks\n")
        f.write("2. Adjust the weighting function for merging chunks\n")
        f.write("3. Apply smoothing at chunk boundaries\n")
        f.write("4. Use different sampling or normalization strategies\n")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Total time: {time.time() - start_time:.1f} seconds")

# Simple configuration class
class Config:
    def __init__(self, 
                 input_image, 
                 chunk_size=1024, 
                 overlap=32, 
                 use_advanced_indices=True,
                 use_texture_features=True,
                 use_spatial_features=True,
                 use_pca_features=True,
                 pca_components=10):
        self.input_image = input_image
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_advanced_indices = use_advanced_indices
        self.use_texture_features = use_texture_features
        self.use_spatial_features = use_spatial_features
        self.use_pca_features = use_pca_features
        self.pca_components = pca_components

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose preprocessing issues in biomass prediction")
    parser.add_argument("input_image", help="Path to input satellite image")
    parser.add_argument("--output-dir", default="biomass_preprocessing_diagnosis", 
                       help="Output directory for diagnostic results")
    parser.add_argument("--chunk-size", type=int, default=1024, 
                       help="Chunk size used in processing")
    parser.add_argument("--overlap", type=int, default=32, 
                       help="Overlap size used in chunk processing")
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        input_image=args.input_image,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    # Run analysis
    analyze_image_and_preprocessing(config, args.output_dir)
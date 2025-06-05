#!/usr/bin/env python3
"""
Biomass Prediction Inference Script

This script loads a trained biomass prediction model and runs inference on a satellite image.
It can process single images or entire directories of satellite imagery.

Usage:
  python predict_biomass.py --input /path/to/satellite.tif --output /path/to/output.tif
  python predict_biomass.py --input_dir /path/to/satellite/images --output_dir /path/to/outputs

Author: vertify.earth
Created for: GIZ Forest Forward initiative
Date: May 2025
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import time
from datetime import datetime
from pathlib import Path
import matplotlib.colors as colors
import io

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the model module is in path
sys.path.append(os.path.dirname(__file__))
from src.models.model import StableResNet

class BiomassPredictionInference:
    """Class for running biomass prediction inference"""
    
    def __init__(self, model_dir=None):
        """
        Initialize the inference engine
        
        Args:
            model_dir: Directory containing model files (model.pt and model_package.pkl)
        """
        self.model = None
        self.package = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default model directory if not provided
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the model
        self.load_model(model_dir)
    
    def load_model(self, model_dir):
        """
        Load the model and preprocessing pipeline
        
        Args:
            model_dir: Directory containing model files
        """
        try:
            # Load model package
            package_path = os.path.join(model_dir, 'model_package.pkl')
            self.package = joblib.load(package_path)
            
            # Initialize model
            self.model = StableResNet(n_features=self.package['n_features'])
            model_path = os.path.join(model_dir, 'model.pt')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            logger.info(f"Number of features: {self.package['n_features']}")
            logger.info(f"Log transform: {self.package['use_log_transform']}")
            logger.info(f"Using device: {self.device}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_biomass(self, image_path, output_path=None, visualization_path=None, 
                        visualization_type="heatmap", chunk_size=1000, return_image=False,
                        return_stats=False):
        """
        Predict biomass from a satellite image
        
        Args:
            image_path: Path to satellite image (GeoTIFF)
            output_path: Path to save output biomass prediction (GeoTIFF)
            visualization_path: Path to save visualization (PNG)
            visualization_type: Type of visualization ('heatmap' or 'rgb_overlay')
            chunk_size: Size of chunks for processing large images
            return_image: Whether to return a PIL Image of the visualization
            return_stats: Whether to return statistics about the prediction
            
        Returns:
            Tuple of (PIL Image, statistics) if requested, otherwise None
        """
        try:
            start_time = time.time()
            logger.info(f"Processing image: {image_path}")
            
            # Open the image file
            with rasterio.open(image_path) as src:
                image = src.read()
                height, width = image.shape[1], image.shape[2]
                transform = src.transform
                crs = src.crs
                
                # Check if number of bands matches expected features
                if image.shape[0] < self.package['n_features']:
                    msg = f"Error: Image has {image.shape[0]} bands, but model expects at least {self.package['n_features']} features."
                    logger.error(msg)
                    return None
                
                logger.info(f"Image dimensions: {height}x{width} pixels, {image.shape[0]} bands")
                
                # Process in chunks to avoid memory issues
                predictions = np.zeros((height, width), dtype=np.float32)
                
                # Create mask for valid pixels (not NaN or Inf)
                valid_mask = np.all(np.isfinite(image), axis=0)
                
                # Process image in chunks
                total_chunks = ((height + chunk_size - 1) // chunk_size) * ((width + chunk_size - 1) // chunk_size)
                chunk_counter = 0
                
                for y_start in range(0, height, chunk_size):
                    y_end = min(y_start + chunk_size, height)
                    
                    for x_start in range(0, width, chunk_size):
                        x_end = min(x_start + chunk_size, width)
                        
                        chunk_counter += 1
                        if chunk_counter % 10 == 0 or chunk_counter == total_chunks:
                            logger.info(f"Processing chunk {chunk_counter}/{total_chunks}")
                        
                        # Get chunk mask
                        chunk_mask = valid_mask[y_start:y_end, x_start:x_end]
                        if not np.any(chunk_mask):
                            continue
                        
                        # Extract valid pixels
                        valid_y, valid_x = np.where(chunk_mask)
                        
                        # Extract features for valid pixels
                        pixel_features = []
                        for i, j in zip(valid_y, valid_x):
                            # Extract bands
                            pixel_values = image[:, y_start+i, x_start+j]
                            pixel_features.append(pixel_values)
                        
                        # Convert to array and scale features
                        pixel_features = np.array(pixel_features)
                        pixel_features_scaled = self.package['scaler'].transform(pixel_features)
                        
                        # Make predictions in batches to handle large chunks
                        batch_size = 10000
                        batch_predictions = []
                        
                        for batch_start in range(0, len(pixel_features_scaled), batch_size):
                            batch_end = min(batch_start + batch_size, len(pixel_features_scaled))
                            batch = pixel_features_scaled[batch_start:batch_end]
                            
                            with torch.no_grad():
                                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                                pred_batch = self.model(batch_tensor).cpu().numpy()
                                batch_predictions.append(pred_batch)
                        
                        # Combine batch predictions
                        batch_predictions = np.concatenate(batch_predictions)
                        
                        # Convert from log scale if needed
                        if self.package['use_log_transform']:
                            batch_predictions = np.exp(batch_predictions) - self.package['epsilon']
                            batch_predictions = np.maximum(batch_predictions, 0)  # Ensure non-negative
                        
                        # Insert predictions back into the image
                        for idx, (i, j) in enumerate(zip(valid_y, valid_x)):
                            predictions[y_start+i, x_start+j] = batch_predictions[idx]
                
                # Save output if specified
                if output_path:
                    # Create output directory if it doesn't exist
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    
                    # Create output GeoTIFF
                    output_meta = src.meta.copy()
                    output_meta.update({
                        'count': 1,
                        'dtype': 'float32'
                    })
                    
                    with rasterio.open(output_path, 'w', **output_meta) as dst:
                        dst.write(predictions[np.newaxis, :, :])
                    
                    logger.info(f"Saved biomass prediction to {output_path}")
                
                # Create visualization if requested
                if visualization_path or return_image:
                    plt.figure(figsize=(12, 8))
                    
                    if visualization_type == "heatmap":
                        # Create heatmap
                        plt.imshow(predictions, cmap='viridis')
                        plt.colorbar(label='Biomass (Mg/ha)')
                        plt.title('Predicted Above-Ground Biomass')
                        
                    elif visualization_type == "rgb_overlay":
                        # Create RGB + overlay
                        if image.shape[0] >= 3:
                            # Use first 3 bands as RGB
                            rgb = image[[0, 1, 2]].transpose(1, 2, 0)
                            rgb = np.clip((rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) - np.percentile(rgb, 2)), 0, 1)
                            
                            plt.imshow(rgb)
                            
                            # Create mask for overlay (where we have predictions)
                            mask = ~np.isclose(predictions, 0)
                            overlay = np.zeros((height, width, 4))
                            
                            # Create colormap for biomass
                            norm = colors.Normalize(vmin=np.percentile(predictions[mask], 5), 
                                                  vmax=np.percentile(predictions[mask], 95))
                            cmap = plt.cm.viridis
                            
                            # Apply colormap
                            overlay[..., :3] = cmap(norm(predictions))[..., :3]
                            overlay[..., 3] = np.where(mask, 0.7, 0)  # Set alpha channel
                            
                            plt.imshow(overlay)
                            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                                        label='Biomass (Mg/ha)')
                            plt.title('Biomass Prediction Overlay')
                        else:
                            plt.imshow(predictions, cmap='viridis')
                            plt.colorbar(label='Biomass (Mg/ha)')
                            plt.title('Predicted Above-Ground Biomass')
                    
                    # Save visualization if specified
                    if visualization_path:
                        # Create output directory if it doesn't exist
                        os.makedirs(os.path.dirname(os.path.abspath(visualization_path)), exist_ok=True)
                        plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
                        logger.info(f"Saved visualization to {visualization_path}")
                    
                    # Return PIL Image if requested
                    pil_img = None
                    if return_image:
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        pil_img = Image.open(buf)
                    
                    plt.close()
                
                # Calculate statistics if requested
                stats = None
                if return_stats:
                    valid_predictions = predictions[valid_mask]
                    stats = {
                        'Mean Biomass': f"{np.mean(valid_predictions):.2f} Mg/ha",
                        'Median Biomass': f"{np.median(valid_predictions):.2f} Mg/ha",
                        'Min Biomass': f"{np.min(valid_predictions):.2f} Mg/ha",
                        'Max Biomass': f"{np.max(valid_predictions):.2f} Mg/ha",
                        'Total Biomass': f"{np.sum(valid_predictions) * (transform[0] * transform[0]) / 10000:.2f} Mg",
                        'Area': f"{np.sum(valid_mask) * (transform[0] * transform[0]) / 10000:.2f} hectares"
                    }
                
                # Log performance
                elapsed_time = time.time() - start_time
                logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
                
                # Return results if requested
                if return_image or return_stats:
                    return pil_img, stats
                
                return True
                
        except Exception as e:
            import traceback
            logger.error(f"Error predicting biomass: {e}\n{traceback.format_exc()}")
            return None
    
    def process_directory(self, input_dir, output_dir, visualization_dir=None, 
                         visualization_type="heatmap", file_pattern="*.tif"):
        """
        Process all satellite images in a directory
        
        Args:
            input_dir: Directory containing satellite images
            output_dir: Directory to save output predictions
            visualization_dir: Directory to save visualizations
            visualization_type: Type of visualization ('heatmap' or 'rgb_overlay')
            file_pattern: Pattern to match input files
            
        Returns:
            Number of successfully processed files
        """
        try:
            # Create output directories
            os.makedirs(output_dir, exist_ok=True)
            if visualization_dir:
                os.makedirs(visualization_dir, exist_ok=True)
            
            # Find all matching files
            from glob import glob
            input_files = glob(os.path.join(input_dir, file_pattern))
            
            if not input_files:
                logger.warning(f"No files matching pattern '{file_pattern}' found in {input_dir}")
                return 0
            
            logger.info(f"Found {len(input_files)} files to process")
            
            # Process each file
            success_count = 0
            for i, input_file in enumerate(input_files):
                try:
                    filename = os.path.basename(input_file)
                    logger.info(f"Processing file {i+1}/{len(input_files)}: {filename}")
                    
                    # Create output paths
                    output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_biomass.tif")
                    
                    visualization_file = None
                    if visualization_dir:
                        visualization_file = os.path.join(visualization_dir, f"{os.path.splitext(filename)[0]}_biomass.png")
                    
                    # Process the file
                    result = self.predict_biomass(
                        input_file, 
                        output_file, 
                        visualization_file,
                        visualization_type
                    )
                    
                    if result:
                        success_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {input_file}: {e}")
            
            logger.info(f"Successfully processed {success_count}/{len(input_files)} files")
            return success_count
            
        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            return 0

def main():
    """Main entry point for biomass prediction inference"""
    parser = argparse.ArgumentParser(description='Biomass Prediction Inference')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Path to satellite image (GeoTIFF)')
    input_group.add_argument('--input_dir', type=str, help='Directory containing satellite images')
    
    # Output options
    parser.add_argument('--output', type=str, help='Path to save output biomass prediction (GeoTIFF)')
    parser.add_argument('--output_dir', type=str, help='Directory to save output predictions')
    
    # Visualization options
    parser.add_argument('--visualization', type=str, help='Path to save visualization (PNG)')
    parser.add_argument('--visualization_dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--visualization_type', type=str, default='heatmap', 
                        choices=['heatmap', 'rgb_overlay'], help='Type of visualization')
    
    # Model options
    parser.add_argument('--model_dir', type=str, help='Directory containing model files')
    
    # Processing options
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of chunks for processing large images')
    parser.add_argument('--file_pattern', type=str, default='*.tif', help='Pattern to match input files')
    
    args = parser.parse_args()
    
    # Check argument consistency
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    
    # Initialize inference engine
    inference = BiomassPredictionInference(args.model_dir)
    
    if args.input:
        # Process a single file
        logger.info(f"Processing file: {args.input}")
        result = inference.predict_biomass(
            args.input, 
            args.output, 
            args.visualization,
            args.visualization_type,
            args.chunk_size
        )
        
        if result:
            logger.info("Processing completed successfully")
            return 0
        else:
            logger.error("Processing failed")
            return 1
    
    elif args.input_dir:
        # Process a directory of files
        logger.info(f"Processing directory: {args.input_dir}")
        success_count = inference.process_directory(
            args.input_dir, 
            args.output_dir, 
            args.visualization_dir,
            args.visualization_type,
            args.file_pattern
        )
        
        if success_count > 0:
            logger.info(f"Processing completed successfully for {success_count} files")
            return 0
        else:
            logger.error("Processing failed for all files")
            return 1

if __name__ == "__main__":
    sys.exit(main())
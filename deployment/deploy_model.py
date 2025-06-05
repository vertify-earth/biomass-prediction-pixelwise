#!/usr/bin/env python3
"""
Script to deploy a trained biomass model to HuggingFace
"""
import os
import sys
import argparse
import logging
import tempfile
from datetime import datetime
from huggingface_hub import HfApi, create_repo

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_repo_exists(repo_id, token=None, repo_type="space"):
    """Ensure the repository exists, create it if necessary"""
    try:
        # Initialize API
        api = HfApi(token=token)
        
        # Check if repository exists using the api.repo_exists method
        try:
            exists = api.repo_exists(repo_id=repo_id, repo_type=repo_type)
        except:
            # Fallback for older versions of huggingface_hub
            try:
                api.get_repo_details(repo_id=repo_id, repo_type=repo_type)
                exists = True
            except:
                exists = False
        
        if not exists:
            logger.info(f"Repository {repo_id} does not exist. Creating it...")
            # Create repository
            create_repo(
                repo_id=repo_id,
                token=token,
                repo_type=repo_type,
                space_sdk="gradio",
                private=False
            )
            logger.info(f"Repository {repo_id} created successfully.")
        else:
            logger.info(f"Repository {repo_id} already exists.")
        
        return True
    except Exception as e:
        logger.error(f"Error creating repository: {e}")
        return False

def prepare_huggingface_repo(model_dir, hf_repo_path, readme=None):
    """
    Prepare a directory for Hugging Face deployment
    
    Args:
        model_dir: Directory containing model files
        hf_repo_path: Path where to create the HF repo
        readme: Path to README.md file (optional)
    """
    # Create repo directory
    os.makedirs(hf_repo_path, exist_ok=True)
    
    # Copy model files
    required_files = ['model.pt', 'model_package.pkl']
    for file in required_files:
        src_path = os.path.join(model_dir, file)
        if os.path.exists(src_path):
            import shutil
            shutil.copy(src_path, os.path.join(hf_repo_path, file))
            logger.info(f"Copied {file} to {hf_repo_path}")
        else:
            logger.error(f"Required file not found: {src_path}")
            return False
    
    # Create app.py with the Gradio interface
    app_py_content = '''
import os
import sys
import torch
import numpy as np
import gradio as gr
import joblib
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import matplotlib.colors as colors
from PIL import Image
import io

class StableResNet(torch.nn.Module):
    """Numerically stable ResNet for biomass regression"""
    def __init__(self, n_features, dropout=0.2):
        super().__init__()
        
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(n_features, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.layer1 = self._make_simple_resblock(256, 256)
        self.layer2 = self._make_simple_resblock(256, 128)
        self.layer3 = self._make_simple_resblock(128, 64)
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _make_simple_resblock(self, in_dim, out_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.BatchNorm1d(out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.BatchNorm1d(out_dim),
            torch.nn.ReLU()
        ) if in_dim == out_dim else torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.BatchNorm1d(out_dim),
            torch.nn.ReLU(),
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        identity = x
        out = self.layer1(x)
        x = out + identity
        
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.regressor(x)
        return x.squeeze()

class BiomassPredictorApp:
    """Gradio app for biomass prediction"""
    
    def __init__(self):
        self.model = None
        self.package = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model package
        self.load_model()
    
    def load_model(self):
        """Load the model and preprocessing pipeline"""
        try:
            # Load model package
            self.package = joblib.load('model_package.pkl')
            
            # Initialize model
            self.model = StableResNet(n_features=self.package['n_features'])
            self.model.load_state_dict(torch.load('model.pt', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully")
            print(f"Number of features: {self.package['n_features']}")
            print(f"Log transform: {self.package['use_log_transform']}")
            print(f"Using device: {self.device}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_biomass(self, image_file, display_type="heatmap"):
        """Predict biomass from a satellite image"""
        try:
            # Open the image file
            with rasterio.open(image_file.name) as src:
                image = src.read()
                height, width = image.shape[1], image.shape[2]
                transform = src.transform
                crs = src.crs
                
                # Check if number of bands matches expected features
                if image.shape[0] < self.package['n_features']:
                    return None, f"Error: Image has {image.shape[0]} bands, but model expects at least {self.package['n_features']} features."
                
                print(f"Processing image: {height}x{width} pixels, {image.shape[0]} bands")
                
                # Process in chunks to avoid memory issues
                chunk_size = 1000
                predictions = np.zeros((height, width), dtype=np.float32)
                
                # Create mask for valid pixels (not NaN or Inf)
                valid_mask = np.all(np.isfinite(image), axis=0)
                
                # Process image in chunks
                for y_start in range(0, height, chunk_size):
                    y_end = min(y_start + chunk_size, height)
                    
                    for x_start in range(0, width, chunk_size):
                        x_end = min(x_start + chunk_size, width)
                        
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
                        
                        # Make predictions
                        with torch.no_grad():
                            batch_tensor = torch.tensor(pixel_features_scaled, dtype=torch.float32).to(self.device)
                            batch_predictions = self.model(batch_tensor).cpu().numpy()
                        
                        # Convert from log scale if needed
                        if self.package['use_log_transform']:
                            batch_predictions = np.exp(batch_predictions) - self.package['epsilon']
                            batch_predictions = np.maximum(batch_predictions, 0)  # Ensure non-negative
                        
                        # Insert predictions back into the image
                        for idx, (i, j) in enumerate(zip(valid_y, valid_x)):
                            predictions[y_start+i, x_start+j] = batch_predictions[idx]
                
                # Create visualization
                plt.figure(figsize=(12, 8))
                
                if display_type == "heatmap":
                    # Create heatmap
                    plt.imshow(predictions, cmap='viridis')
                    plt.colorbar(label='Biomass (Mg/ha)')
                    plt.title('Predicted Above-Ground Biomass')
                    
                elif display_type == "rgb_overlay":
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
                
                # Save figure to bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                # Create summary statistics
                valid_predictions = predictions[valid_mask]
                stats = {
                    'Mean Biomass': f"{np.mean(valid_predictions):.2f} Mg/ha",
                    'Median Biomass': f"{np.median(valid_predictions):.2f} Mg/ha",
                    'Min Biomass': f"{np.min(valid_predictions):.2f} Mg/ha",
                    'Max Biomass': f"{np.max(valid_predictions):.2f} Mg/ha",
                    'Total Biomass': f"{np.sum(valid_predictions) * (transform[0] * transform[0]) / 10000:.2f} Mg",
                    'Area': f"{np.sum(valid_mask) * (transform[0] * transform[0]) / 10000:.2f} hectares"
                }
                
                # Format statistics as markdown
                stats_md = "### Biomass Statistics\\n\\n"
                stats_md += "| Metric | Value |\\n|--------|-------|\\n"
                for k, v in stats.items():
                    stats_md += f"| {k} | {v} |\\n"
                
                # Close the plot
                plt.close()
                
                # Return visualization and statistics
                return Image.open(buf), stats_md
                
        except Exception as e:
            import traceback
            return None, f"Error predicting biomass: {str(e)}\\n\\n{traceback.format_exc()}"

    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="Biomass Prediction Model") as interface:
            gr.Markdown("# Above-Ground Biomass Prediction")
            gr.Markdown("""
            Upload a multi-band satellite image to predict above-ground biomass (AGB) across the landscape.
            
            **Requirements:**
            - Image must be a GeoTIFF with spectral bands
            - For best results, image should contain similar bands to those used in training
            """)
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.File(
                        label="Upload Satellite Image (GeoTIFF)",
                        file_types=[".tif", ".tiff"]
                    )
                    
                    display_type = gr.Radio(
                        choices=["heatmap", "rgb_overlay"],
                        value="heatmap",
                        label="Display Type"
                    )
                    
                    submit_btn = gr.Button("Generate Biomass Prediction")
                
                with gr.Column():
                    output_image = gr.Image(
                        label="Biomass Prediction Map",
                        type="pil"
                    )
                    
                    output_stats = gr.Markdown(
                        label="Statistics"
                    )
            
            with gr.Accordion("About", open=False):
                gr.Markdown("""
                ## About This Model
                
                This biomass prediction model uses the StableResNet architecture to predict above-ground biomass from satellite imagery.
                
                ### Model Details
                
                - Architecture: StableResNet
                - Input: Multi-spectral satellite imagery
                - Output: Above-ground biomass (Mg/ha)
                - Creator: vertify.earth
                - Partner: GIZ Forest Forward initiative
                - Date: 2025-05-16
                
                ### How It Works
                
                1. The model extracts features from each pixel in the satellite image
                2. These features are processed through the StableResNet model
                3. The model outputs a biomass prediction for each pixel
                4. Results are visualized as a heatmap or RGB overlay
                """)
            
            submit_btn.click(
                fn=self.predict_biomass,
                inputs=[input_image, display_type],
                outputs=[output_image, output_stats]
            )
        
        return interface

def launch_app():
    """Launch the Gradio app"""
    app = BiomassPredictorApp()
    interface = app.create_interface()
    interface.launch()

if __name__ == "__main__":
    launch_app()
    '''
    
    # Write app.py to the repo path
    with open(os.path.join(hf_repo_path, 'app.py'), 'w') as f:
        f.write(app_py_content)
    logger.info("Created app.py")
    
    # Create requirements.txt
    with open(os.path.join(hf_repo_path, 'requirements.txt'), 'w') as f:
        f.write('\n'.join([
            'torch>=1.10.0',
            'rasterio>=1.2.0',
            'numpy>=1.20.0',
            'scikit-learn>=1.0.0',
            'matplotlib>=3.5.0',
            'gradio>=3.0.0',
            'joblib>=1.1.0',
            'pillow>=8.0.0'
        ]))
    logger.info("Created requirements.txt")
    
    # Create or copy README.md
    if readme and os.path.exists(readme):
        import shutil
        shutil.copy(readme, os.path.join(hf_repo_path, 'README.md'))
        logger.info(f"Copied README.md from {readme}")
    else:
        # Create basic README
        with open(os.path.join(hf_repo_path, 'README.md'), 'w') as f:
            f.write("""# Biomass Prediction Model

This repository contains a trained model for predicting above-ground biomass (AGB) from satellite imagery.

## Model Description

The model uses a StableResNet architecture to predict biomass values from multi-spectral satellite imagery on a per-pixel basis.

## Usage

1. Upload a multi-band satellite image (GeoTIFF)
2. Select the display type (heatmap or RGB overlay)
3. Click "Generate Biomass Prediction"
4. View the resulting biomass map and statistics

## Requirements

- The image must be in GeoTIFF format
- The image should contain multiple spectral bands similar to those used in training

## Model Details

- Architecture: StableResNet
- Input: Multi-spectral satellite data
- Output: Above-ground biomass in Mg/ha (megagrams per hectare)
- Creator: vertify.earth
- Partner: GIZ Forest Forward initiative
- Date: 2025-05-16
""")
        logger.info("Created basic README.md")
    
    return True

def deploy_to_huggingface(hf_repo_path, hf_repo_id, token=None):
    """
    Deploy the model to HuggingFace Hub
    
    Args:
        hf_repo_path: Local path to the repo
        hf_repo_id: HuggingFace repo ID (username/repo-name)
        token: HuggingFace token (optional)
    """
    try:
        from huggingface_hub import HfApi
        
        # Initialize API
        api = HfApi(token=token)
        
        # Upload files to the existing repository
        logger.info(f"Uploading files to repository: {hf_repo_id}")
        api.upload_folder(
            folder_path=hf_repo_path,
            repo_id=hf_repo_id,
            repo_type="space",
            token=token
        )
        
        logger.info(f"Model deployed to HuggingFace Hub: {hf_repo_id}")
        logger.info(f"View at: https://huggingface.co/spaces/{hf_repo_id}")
        return True
    
    except ImportError:
        logger.error("huggingface_hub package not installed. Please install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Error deploying to HuggingFace: {e}")
        return False

def main():
    """Deploy a previously trained model to HuggingFace"""
    parser = argparse.ArgumentParser(description='Deploy trained biomass model to HuggingFace')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Directory containing the trained model (with model.pt and model_package.pkl)')
    parser.add_argument('--hf_repo', type=str, required=True, 
                        help='HuggingFace repository name (e.g., vertify/biomass-prediction)')
    parser.add_argument('--hf_token', type=str,
                        help='HuggingFace token (needed for private repos or creating new repos)')
    parser.add_argument('--readme', type=str, 
                        help='Path to custom README.md file for the HuggingFace repo')
    
    args = parser.parse_args()
    
    # Check if the model directory exists and contains required files
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        return 1
    
    required_files = ['model.pt', 'model_package.pkl']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.model_dir, f))]
    if missing_files:
        logger.error(f"Missing required files in model directory: {', '.join(missing_files)}")
        return 1
    
    try:
        print("\n" + "="*70)
        print("🚀 DEPLOYING MODEL TO HUGGINGFACE")
        print("="*70)
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model directory: {args.model_dir}")
        print(f"HuggingFace repo: {args.hf_repo}")
        print("="*70 + "\n")
        
        # STEP 1: Explicitly ensure the repository exists
        logger.info("Ensuring repository exists...")
        repo_exists = ensure_repo_exists(args.hf_repo, args.hf_token, repo_type="space")
        
        if not repo_exists:
            logger.error("Failed to create or verify repository existence.")
            print("\n" + "="*70)
            print("❌ REPOSITORY CREATION FAILED")
            print("="*70)
            return 1
        
        # STEP 2: Create temporary directory for HF repo
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Prepare the repository
            success = prepare_huggingface_repo(args.model_dir, tmp_dir, args.readme)
            
            if success:
                # STEP 3: Deploy to HuggingFace
                deploy_success = deploy_to_huggingface(tmp_dir, args.hf_repo, args.hf_token)
                
                if deploy_success:
                    print("\n" + "="*70)
                    print("✅ MODEL DEPLOYED SUCCESSFULLY")
                    print(f"View at: https://huggingface.co/spaces/{args.hf_repo}")
                    print("="*70)
                    return 0
                else:
                    print("\n" + "="*70)
                    print("❌ DEPLOYMENT FAILED")
                    print("="*70)
                    return 1
            else:
                print("\n" + "="*70)
                print("❌ FAILED TO PREPARE REPOSITORY")
                print("="*70)
                return 1
    
    except ImportError:
        logger.error("huggingface_hub package not installed. Please install with: pip install huggingface_hub")
        return 1
    except Exception as e:
        logger.error(f"Error deploying model: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
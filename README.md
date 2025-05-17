# Biomass Prediction Training Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About

This repository contains the complete training pipeline for the Biomass Prediction Model developed by vertify.earth for the GIZ Forest Forward initiative. The model uses multi-spectral satellite imagery to predict above-ground biomass (AGB) in forest ecosystems, providing valuable data for carbon monitoring and sustainable forest management.

## Repository Structure

```
biomass-prediction/
│
├── app.py                    # Gradio interface for model deployment
├── config.py                 # Configuration parameters
├── data_processing.py        # Data preprocessing utilities
├── feature_engineering.py    # Spectral indices and feature extraction
├── main.py                   # Main script to run the pipeline
├── model.py                  # Neural network architecture (StableResNet)
├── pipeline.py               # End-to-end training pipeline
├── requirements.txt          # Required Python packages
├── train.py                  # Model training and evaluation
└── utils.py                  # Helper functions
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/vertify/biomass-prediction-training.git
cd biomass-prediction-training
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

### Input Data Format

The pipeline expects paired satellite-biomass data in the following format:

1. **Satellite Data**: Multi-band GeoTIFF files with the following bands:
   - Sentinel-1: VV and VH polarizations
   - Sentinel-2: Bands 2 (Blue), 3 (Green), 4 (Red), 8 (NIR), 11 (SWIR1), 12 (SWIR2)
   - Landsat-8: Optional additional bands
   - PALSAR: Optional L-band SAR data
   - Digital Elevation Model: Height, slope, aspect

2. **Biomass Ground Truth**: Single-band GeoTIFF files containing above-ground biomass values in Mg/ha

### Data Preparation

Before running the training pipeline, prepare your data as follows:

1. Ensure all rasters are co-registered and have the same resolution
2. Reproject data to a common coordinate reference system (CRS)
3. Configure the data paths in `config.py`

Example of setting up data paths:
```python
raster_pairs = [
    ("/path/to/satellite_image1.tif", "/path/to/biomass_groundtruth1.tif"),
    ("/path/to/satellite_image2.tif", "/path/to/biomass_groundtruth2.tif"),
    # Add more pairs as needed
]
```

## StableResNet Architecture

The core model is a custom ResNet-inspired architecture designed specifically for stable biomass regression:

```python
class StableResNet(nn.Module):
    def __init__(self, n_features, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.layer1 = self._make_simple_resblock(256, 256)
        self.layer2 = self._make_simple_resblock(256, 128)
        self.layer3 = self._make_simple_resblock(128, 64)
        
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
```

Key features of this architecture:
- Layer normalization for improved training stability
- Residual connections to facilitate gradient flow
- Progressive dimensionality reduction
- Dropout for regularization

## Feature Engineering

The pipeline includes comprehensive feature engineering capabilities:

1. **Spectral Indices**: NDVI, EVI, SAVI, MSAVI2, NDWI, NDMI, NBR, etc.
2. **Texture Features**: Local Binary Patterns (LBP), Gray-Level Co-occurrence Matrix (GLCM), Sobel edge detection
3. **Spatial Context**: Gradient magnitude and direction
4. **PCA Features**: Dimensionality reduction of spectral bands

Example of spectral indices calculation:
```python
def calculate_spectral_indices(satellite_data):
    # Calculate NDVI
    indices['NDVI'] = safe_divide(nir - red, nir + red)
    
    # Calculate EVI
    indices['EVI'] = 2.5 * safe_divide(nir - red, nir + 6*red - 7.5*blue + 1)
    
    # Calculate SAVI
    indices['SAVI'] = 1.5 * safe_divide(nir - red, nir + red + 0.5)
    
    # Other indices...
```

## Running the Pipeline

### Quick Test

To run a quick test on a small subset of data:

```bash
python main.py --mode test
```

This will:
- Use only the first data pair
- Run for only 10 epochs
- Disable advanced feature engineering
- Limit to 5000 samples per tile

### Full Training

For complete model training with all features:

```bash
python main.py --mode full --data_dir /path/to/data --results_dir /path/to/results
```

### Configuration Options

Modify `config.py` to customize the training process:

```python
# Feature engineering options
use_log_transform = True        # Log-transform biomass values
use_advanced_indices = True     # Calculate spectral indices
use_texture_features = True     # Extract texture features
use_spatial_features = True     # Calculate spatial gradients
use_pca_features = True         # Apply PCA to input bands

# Model hyperparameters
dropout_rate = 0.2              # Dropout probability
batch_size = 256                # Training batch size
learning_rate = 0.001           # Initial learning rate
weight_decay = 1e-5             # L2 regularization
max_epochs = 300                # Maximum training epochs
patience = 30                   # Early stopping patience
```

## Model Evaluation

The training pipeline automatically evaluates the model using:

1. **R²**: Coefficient of determination
2. **RMSE**: Root Mean Square Error
3. **MAE**: Mean Absolute Error

Metrics are calculated on both log-transformed and original scales. The pipeline generates visualizations including:

- Training and validation loss curves
- Training and validation R² curves
- Predicted vs. actual biomass scatter plots

## Model Deployment

After training, deploy your model to HuggingFace:

```bash
python main.py --mode full --deploy --hf_repo vertify/biomass-prediction --hf_token YOUR_HF_TOKEN
```

This will:
1. Package the model with all necessary components
2. Create a Gradio app for interactive usage
3. Upload everything to HuggingFace Spaces

## Example Workflow

Here's an end-to-end example of training and deploying a biomass prediction model:

```bash
# 1. Clone the repository
git clone https://github.com/vertify/biomass-prediction-training.git
cd biomass-prediction-training

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit config.py to set data paths
# Edit as needed...

# 4. Run a quick test
python main.py --mode test

# 5. Run full training
python main.py --mode full

# 6. Deploy to HuggingFace
python main.py --mode full --deploy --hf_repo vertify/biomass-prediction --hf_token YOUR_HF_TOKEN
```

## Performance Optimization

For large datasets or limited hardware:

- **Memory Efficiency**: Processing is done in chunks to handle large rasters
- **GPU Acceleration**: Automatically uses CUDA when available
- **Early Stopping**: Prevents overfitting and reduces training time
- **Sampling**: Option to sample a subset of pixels for quick experimentation

## Extending the Pipeline

### Adding New Satellite Sensors

To add support for a new satellite sensor:

1. Update the feature extraction in `feature_engineering.py`
2. Add band mappings for the new sensor
3. Modify spectral index calculations if needed

### Custom Model Architectures

To implement a different model architecture:

1. Create a new class in `model.py`
2. Update the `initialize_model` function
3. Modify `config.py` to use the new model type

## Troubleshooting

### Common Issues

- **Out of Memory**: Reduce `batch_size` or `chunk_size`
- **NaN Values**: Check for missing data in input rasters
- **Poor Performance**: Try different feature combinations or adjust hyperparameters
- **CUDA Errors**: Verify GPU compatibility and CUDA installation

### Logging

The pipeline includes comprehensive logging to help diagnose issues:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

Logs are saved to `{results_dir}/logs/pipeline_{mode}_{timestamp}.log`

## Contributing

Contributions to improve the biomass prediction pipeline are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Developed by vertify.earth for the GIZ Forest Forward initiative
- Training data sources include field measurements from various research institutions
- Satellite imagery from ESA Copernicus Programme (Sentinel-1, Sentinel-2) and NASA/USGS (Landsat-8)

## Citation

```
@misc{vertify2025biomass,
  author = {vertify.earth},
  title = {Biomass Prediction Training Pipeline},
  year = {2025},
  publisher = {GitHub},
  note = {Developed for GIZ Forest Forward initiative},
  howpublished = {\url{https://github.com/vertify/biomass-prediction-training}}
}
```

## Contact

For questions, feedback, or collaboration opportunities, please reach out via:
- GitHub: [vertify](https://github.com/vertify)
- Email: info@vertify.earth
- Website: [vertify.earth](https://vertify.earth)
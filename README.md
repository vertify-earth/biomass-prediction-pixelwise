# Deep Learning for Above Ground Biomass Estimation:  Pixel-wise Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a comprehensive machine learning pipeline for estimating above-ground biomass (AGB) using satellite remote sensing data with pixel-level prediction. It implements advanced feature engineering and a stable ResNet architecture optimized for biomass regression at individual pixel resolution.

## What Is This Model For

This model is trained to predict above-ground biomass (AGB) in tropical and subtropical forests using multi-source satellite imagery at the pixel level. Specifically:

- **Prediction Unit**: Estimates biomass at individual pixel level (typically 10-40m resolution depending on input data)
- **Output**: Biomass density in Mg/ha (megagrams per hectare)
- **Input Data**: Processes multi-sensor data including Sentinel-1, Sentinel-2, Landsat-8, PALSAR, and DEM
- **Application Scope**: Best suited for tropical and subtropical forest ecosystems in South/Southeast Asia
- **Biomass Range**: Validated for forests with biomass between ~40-460 Mg/ha

## Overview

The pipeline is designed to handle multi-source satellite imagery and corresponding biomass data with pixel-level precision. Key aspects include:

- **End-to-End Workflow**: From raw data ingestion to model training, evaluation, and deployment
- **Advanced Feature Engineering**: Comprehensive spectral indices, texture features, spatial gradients, and PCA components
- **Stable Neural Architecture**: Utilizes a custom StableResNet with residual connections and layer normalization for robust biomass regression
- **Multi-Site Data Processing**: Capable of processing and integrating data from multiple geographically distinct study sites
- **Flexible Deployment**: Includes HuggingFace deployment capabilities with Gradio interface
- **Memory Efficient Processing**: Chunk-based processing for handling large satellite images

## Performance Metrics

### Model Architecture

The final prediction model uses **StableResNet**, a custom ResNet-inspired architecture with residual connections and layer normalization. This architecture provides:
- **Robust pixel-level predictions** - optimized for individual pixel biomass estimation
- **Stable training convergence** - layer normalization prevents training instabilities
- **Generalizable across forest types** - performance validated across diverse tropical ecosystems

### Overall Model Performance

| Metric | Value |
|--------|-------|
| **R²** | **0.87** |
| **RMSE** | **28.7 Mg/ha** |
| **MAE** | **19.5 Mg/ha** |

### Performance Assessment

With an **R² of 0.87** and **RMSE of 28.7 Mg/ha**, the model demonstrates **strong performance** for pixel-level biomass estimation across diverse tropical forest conditions. The model achieves:

- **High correlation** between predicted and actual biomass values
- **Reasonable prediction accuracy** with mean absolute error of 19.5 Mg/ha
<!-- - **Robust generalization** across different forest types and biomass ranges -->
- **Pixel-level precision** enabling detailed biomass mapping

The performance metrics indicate the model is well-suited for operational biomass mapping applications in tropical and subtropical forest ecosystems.

## Model Architecture

The core model implements **StableResNet**, a custom ResNet-inspired architecture designed specifically for stable biomass regression:

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
- **Layer Normalization**: Improves training stability and convergence
- **Residual Connections**: Facilitates gradient flow and prevents vanishing gradients
- **Progressive Dimensionality Reduction**: Efficiently processes high-dimensional feature vectors
- **Dropout Regularization**: Prevents overfitting on complex feature combinations

## Feature Engineering

The pipeline includes comprehensive feature engineering capabilities:

### 1. Spectral Indices
- **Vegetation Indices**: NDVI, EVI, SAVI, MSAVI2
- **Water Indices**: NDWI, NDMI
- **Burn Indices**: NBR, NDVI-based ratios
- **Enhanced Ratios**: Custom band ratios optimized for biomass estimation

### 2. Texture Features
- **Local Binary Patterns (LBP)**: Captures local texture variations
- **Gray-Level Co-occurrence Matrix (GLCM)**: Statistical texture measures
- **Sobel Edge Detection**: Gradient-based edge information

### 3. Spatial Context Features
- **Gradient Magnitude**: Spatial intensity variations
- **Gradient Direction**: Directional spatial patterns
- **Local Standard Deviation**: Neighborhood variability measures

### 4. Dimensionality Reduction
- **PCA Features**: Principal component analysis of spectral bands
- **Configurable Components**: Adjustable number of PCA components (default: 25)

## Training Data

The model is trained on data from multiple forest sites in India and Thailand, covering diverse biomass conditions:

| Site       | Location                    | Area (km²) | Biomass Range (Mg/ha) | Mean ± Std Dev (Mg/ha) | Forest / Terrain Type                  |
| ---------- | --------------------------- | ---------- | --------------------- | ---------------------- | -------------------------------------- |
| Yellapur   | Karnataka, India            | 312        | 47 to 322             | 215 ± 53               | Tropical semi-evergreen forest         |
| Betul      | Madhya Pradesh, India       | 105        | 7 to 128              | 93 ± 27                | Dry deciduous forest                   |
| Achanakmar | Chhattisgarh, India         | 117        | 74 to 229             | 169 ± 28               | Moist deciduous forest, hilly terrain  |
| Khaoyai    | Nakhon Ratchasima, Thailand | 47         | 179 to 436            | 275 ± 47               | Tropical evergreen forest, mountainous |

The AGB ground data is sourced from the study: Rodda, S.R., Fararoda, R., Gopalakrishnan, R. et al. LiDAR-based reference aboveground biomass maps for tropical forests of South Asia and Central America.

### Satellite Data Used
The model integrates data from multiple satellite sensors:
- **Sentinel-1**: C-band SAR (VV, VH polarizations)
- **Sentinel-2**: Multispectral 10-20m bands (B2, B3, B4, B8, B11, B12)
- **Landsat-8**: Optical bands
- **PALSAR**: L-band SAR
- **Digital Elevation Model**: Topographic information (height, slope, aspect)

## Repository Structure

```
.
├── configs/                # Configuration files
│   └── config.py          # Main configuration dataclass
├── data/                   # Data directory
│   ├── satellite/          # Satellite imagery files
│   ├── biomass/           # Ground truth biomass maps
│   └── README.md          # Data structure documentation
├── src/                   # Source code
│   ├── data/              # Data processing and feature engineering
│   │   ├── data_processing.py
│   │   └── feature_engineering.py
│   ├── models/            # Model architecture
│   │   └── model.py       # StableResNet implementation
│   ├── training/          # Training pipeline
│   │   ├── train.py       # Model trainer
│   │   └── pipeline.py    # End-to-end pipeline
│   ├── inference/         # Inference utilities
│   │   └── predict_biomass.py
│   └── utils/             # Utility functions
│       └── utils.py
├── scripts/               # Execution scripts
│   └── main.py           # Main pipeline script
├── deployment/           # Deployment utilities
│   └── deploy_model.py   # HuggingFace deployment
├── docs/                 # Documentation
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
├── setup.py             # Package setup
└── README.md            # This file
```

## Installation

### Requirements

- Python 3.8+ 
- PyTorch 1.10+
- CUDA-compatible GPU (recommended for training)
- Sufficient RAM (>=16GB recommended) and disk space

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vertify-earth/biomass-prediction-pixelwise.git
   cd biomass-prediction-pixelwise
   ```

2. **Create and activate environment:**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate biomass-prediction
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

## Configuration

The pipeline behavior is controlled by the `BiomassPipelineConfig` dataclass in `configs/config.py`.

### Key Configuration Parameters

```python
@dataclass
class BiomassPipelineConfig:
    # Core settings
    mode: str = "full"  # Options: 'test', 'full'
    random_seed: int = 42
    
    # Data paths - Update with your data paths
    raster_pairs: List[tuple] = [
        ("data/satellite/site1_satellite.tif", "data/biomass/site1_biomass.tif"),
        ("data/satellite/site2_satellite.tif", "data/biomass/site2_biomass.tif"),
        # Add more pairs as needed
    ]
    
    # Feature engineering settings
    use_log_transform: bool = True
    use_advanced_indices: bool = True
    use_texture_features: bool = True
    use_spatial_features: bool = True
    use_pca_features: bool = True
    pca_components: int = 25
    
    # Model settings
    model_type: str = "StableResNet"
    dropout_rate: float = 0.2
    batch_size: int = 256
    learning_rate: float = 0.001
    max_epochs: int = 300
    patience: int = 30
```

## Data Preparation

### Input Data Format

1. **Satellite Data**: Multi-band GeoTIFF files with the following bands:
   - Sentinel-1: VV and VH polarizations
   - Sentinel-2: Bands 2 (Blue), 3 (Green), 4 (Red), 8 (NIR), 11 (SWIR1), 12 (SWIR2)
   - Landsat-8: Optional additional bands
   - PALSAR: Optional L-band SAR data
   - Digital Elevation Model: Height, slope, aspect

2. **Biomass Ground Truth**: Single-band GeoTIFF files containing above-ground biomass values in Mg/ha

### Data Setup

1. Place satellite imagery in `data/satellite/`
2. Place biomass ground truth in `data/biomass/`
3. Update `raster_pairs` in `configs/config.py` with your data paths
4. Ensure all rasters are co-registered and have the same resolution

## Usage

### Quick Test

Run a quick test on a subset of data:

```bash
python scripts/main.py --mode test
```

This will:
- Use only the first data pair
- Run for only 10 epochs
- Disable advanced feature engineering
- Limit to 5000 samples per tile

### Full Training

For complete model training with all features:

```bash
python scripts/main.py --mode full --data_dir /path/to/data --results_dir /path/to/results
```

### Custom Configuration

Modify the configuration for your specific needs:

```python
# In config.py
config = BiomassPipelineConfig(
    mode="full",
    use_texture_features=False,  # Disable texture features for faster training
    pca_components=15,           # Reduce PCA components
    max_epochs=100,             # Fewer epochs
    batch_size=512              # Larger batch size
)
```

## Model Deployment

### HuggingFace Deployment

Deploy your trained model to HuggingFace Spaces:

```bash
python scripts/main.py --mode full --deploy --hf_repo your-username/biomass-prediction --hf_token YOUR_HF_TOKEN
```

This will:
1. Package the model with all necessary components
2. Create a Gradio app for interactive usage
3. Upload everything to HuggingFace Spaces

### Local Inference

Use the trained model for local predictions:

```python
from src.inference.predict_biomass import BiomassPredictionInference

# Initialize inference engine
predictor = BiomassPredictionInference(model_dir="results/models/")

# Predict biomass for a satellite image
predictor.predict_biomass(
    image_path="path/to/satellite_image.tif",
    output_path="path/to/biomass_prediction.tif",
    visualization_path="path/to/visualization.png"
)
```

## Features

- **Configurable Feature Engineering**: Spectral indices, texture features, spatial features, and PCA
- **Robust Data Processing**: Handles missing values, outliers, and large datasets efficiently
- **Advanced Model Architecture**: StableResNet with residual connections and normalization
- **Flexible Training**: Configurable loss functions, learning rate scheduling, and early stopping
- **Memory Efficient**: Chunk-based processing for large satellite images
- **Comprehensive Evaluation**: Multiple metrics and visualization outputs
- **Easy Deployment**: One-command deployment to HuggingFace Spaces
- **Extensible Design**: Easy to add new sensors, features, or model architectures

## Performance Optimization

For large datasets or limited hardware:

- **Memory Efficiency**: Processing is done in chunks to handle large rasters
- **GPU Acceleration**: Automatically uses CUDA when available
- **Early Stopping**: Prevents overfitting and reduces training time
- **Configurable Sampling**: Option to sample a subset of pixels for experimentation
- **Robust Scaling**: Multiple scaling options (standard, robust, minmax)

## Model Loading and Inference

### Loading the Trained Model

```python
import torch
import joblib
from src.models.model import StableResNet

# Load model package
package = joblib.load('results/models/model_package.pkl')

# Initialize and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StableResNet(n_features=package['n_features'])
model.load_state_dict(torch.load('results/models/model.pt', map_location=device))
model.eval()

# Make predictions on new data
with torch.no_grad():
    predictions = model(new_features)
    # Convert from log scale to original biomass (Mg/ha) if log transform was used
    if package['use_log_transform']:
        biomass_predictions = torch.exp(predictions) - package['epsilon']
```

## Extending the Pipeline

### Adding New Satellite Sensors

1. Update band mappings in `src/data/feature_engineering.py`
2. Add new spectral indices if needed
3. Update configuration to include new bands

### Custom Model Architectures

1. Create new model class in `src/models/model.py`
2. Update `initialize_model` function
3. Modify configuration to use new model type

### Custom Feature Engineering

1. Add new feature extraction functions in `src/data/feature_engineering.py`
2. Update configuration options
3. Ensure features are properly scaled and named

<!-- ## Testing

Run unit tests to verify functionality:

```bash
pytest tests/
``` -->

## Troubleshooting

### Common Issues

- **Out of Memory**: Reduce `batch_size` or enable chunked processing
- **NaN Values**: Check for missing data in input rasters and adjust preprocessing
- **Poor Performance**: Try different feature combinations or adjust hyperparameters
- **CUDA Errors**: Verify GPU compatibility and CUDA installation

### Logging

Comprehensive logging helps diagnose issues:
- Logs are saved to `{results_dir}/logs/pipeline_{mode}_{timestamp}.log`
- Use different log levels for debugging: `INFO`, `DEBUG`, `WARNING`, `ERROR`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{vertify2025biomass_pixelwise,
  author = {vertify.earth},
  title = {Biomass Prediction Pixelwise Training Pipeline},
  year = {2025},
  publisher = {GitHub},
  note = {Developed for GIZ Forest Forward initiative},
  howpublished = {\url{https://github.com/vertify-earth/biomass-prediction-pixelwise}}
}
```

## Contact

For questions, feedback, or collaboration opportunities, please reach out via:
- GitHub: [vertify-earth](https://github.com/vertify-earth)
- Email: info@vertify.earth
- Website: [vertify.earth](https://vertify.earth)

## Acknowledgements

- Developed by vertify.earth for the GIZ Forest Forward initiative
- Training data sources include field measurements from various research institutions
- Satellite imagery from ESA Copernicus Programme (Sentinel-1, Sentinel-2) and NASA/USGS (Landsat-8)
- LiDAR-based biomass reference data from Rodda, S.R., Fararoda, R., Gopalakrishnan, R. et al.
import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from datetime import datetime

@dataclass
class BiomassPipelineConfig:
    """Configuration for Biomass Prediction Pipeline"""
    # Core settings
    mode: str = "full"  # Options: 'test', 'full'
    random_seed: int = 42
    project_name: str = "biomass-prediction"
    created_by: str = "vertify.earth"
    created_date: str = "2025-05-16"
    
    # Data paths - These should be updated with your own data paths
    raster_pairs: List[tuple] = field(default_factory=lambda: [
        ("data/satellite/s1_s2_l8_palsar_ch_dem_yellapur_2020.tif",
         "data/biomass/agbd_yellapur_reprojected_1.tif"),
        ("data/satellite/s1_s2_l8_palsar_ch_dem_betul_2020.tif",
         "data/biomass/agbd_betul_reprojected_1.tif"),
        ("data/satellite/s1_s2_l8_palsar_ch_goa_achankumar_2020_clipped.tif",
         "data/biomass/02_Achanakmar_AGB40_band1_onImgGrid.tif"),
        ("data/satellite/s1_s2_l8_palsar_ch_dem_Khaoyai_2020_clipped.tif",
         "data/biomass/05_Khaoyai_AGB40_band1_onImgGrid.tif")
    ])
    data_dir: str = "data/"
    results_dir: str = field(default_factory=lambda: f"biomass_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Feature engineering settings
    use_log_transform: bool = True
    epsilon: float = 1.0
    use_advanced_indices: bool = True
    use_texture_features: bool = True
    use_spatial_features: bool = True
    use_pca_features: bool = True
    pca_components: int = 25
    scale_method: str = "robust"  # Options: 'standard', 'minmax', 'robust'
    outlier_removal: bool = True
    outlier_threshold: float = 3.0
    
    # Model settings
    model_type: str = "StableResNet"
    dropout_rate: float = 0.2
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 300
    patience: int = 30
    test_size: float = 0.15
    val_size: float = 0.15
    
    # Deployment settings
    huggingface_repo: str = "vertify/biomass-prediction"
    quantize_model: bool = False
    
    def __post_init__(self):
        """Adjust settings based on mode"""
        if self.mode == "test":
            # Quick testing settings
            self.raster_pairs = self.raster_pairs[:1]  # Use only first tile
            self.max_epochs = 10
            self.batch_size = 32
            self.use_texture_features = False
            self.use_spatial_features = False
            self.use_pca_features = False
            self.max_samples_per_tile = 5000
            self.pca_components = 10
        else:
            # Full mode settings
            self.max_samples_per_tile = None
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "visualizations"), exist_ok=True)
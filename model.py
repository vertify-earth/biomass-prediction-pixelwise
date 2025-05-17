import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StableResNet(nn.Module):
    """Numerically stable ResNet for biomass regression"""
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
        
        self._init_weights()
    
    def _make_simple_resblock(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        ) if in_dim == out_dim else nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        identity = x
        out = self.layer1(x)
        x = out + identity
        
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.regressor(x)
        return x.squeeze()

def initialize_model(config, n_features):
    """Initialize the appropriate model based on config"""
    if config.model_type == "StableResNet":
        model = StableResNet(n_features=n_features, dropout=config.dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
        
    return model